//! TRELLIS.2 sparse primitive unit tests:
//!
//! 1. Coordinate hashmap build/lookup;
//! 2. Submanifold 3x3x3 sparse conv vs a dense (naive loop) reference on a
//!    4^3 grid with holes, plus hand-picked identity/shift kernels and the
//!    flex_gemm `[out, kd, kh, kw, in]` loader transpose;
//! 3. C2S child ordering + feature selection (hand case), C2S/S2C round trip,
//!    and the `repeat_interleave` skin-connection helper;
//! 4. Exact sparse full attention (self + cross, chunked queries) vs a naive
//!    per-head softmax reference.
//!
//! All inputs are deterministic patterns (no RNG dependency); everything runs
//! on CPU in F32.

use std::collections::HashMap;

use candle_core::{Device, Tensor};
use lux3d_core::runtime::{
    CoordMap, SparseConv3d, channel_to_spatial, kernel_offset_delta, repeat_channels,
    sparse_cross_attention, sparse_self_attention, spatial_to_channel,
};

fn pattern_f32(index: usize) -> f32 {
    ((index % 23) as f32 - 11.0) * 0.1
}

fn tensor_from(values: Vec<f32>, shape: (usize, usize)) -> Tensor {
    Tensor::from_vec(values, vec![shape.0, shape.1], &Device::Cpu).expect("tensor")
}

/// Dense reference submanifold conv: `out[i,o] = bias[o] + sum_offsets
/// sum_c W[o,kd,kh,kw,c] * dense[coord_i + delta, c]`, zero where the
/// neighbor is not an active voxel.
fn dense_conv_reference(
    coords: &[[i32; 3]],
    feats: &[Vec<f32>],
    weight: &[Vec<Vec<f32>>], // [o][k = kd*9+kh*3+kw][c]
    bias: &[f32],
    channels: usize,
    out_channels: usize,
) -> Vec<Vec<f32>> {
    let lookup: HashMap<[i32; 3], usize> =
        coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    let mut out = vec![vec![0.0_f32; out_channels]; coords.len()];
    for (i, coord) in coords.iter().enumerate() {
        for o in 0..out_channels {
            let mut acc = bias[o];
            for k in 0..27 {
                let delta = kernel_offset_delta(k);
                let neighbor = [
                    coord[0] + delta[0],
                    coord[1] + delta[1],
                    coord[2] + delta[2],
                ];
                if let Some(&src) = lookup.get(&neighbor) {
                    for c in 0..channels {
                        acc += weight[o][k][c] * feats[src][c];
                    }
                }
            }
            out[i][o] = acc;
        }
    }
    out
}

#[test]
fn coord_map_builds_and_looks_up() {
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 0, 3], [7, 2, 5]];
    let map = CoordMap::new(&coords);
    assert_eq!(map.len(), 3);
    assert_eq!(map.get([1, 0, 3]), Some(1));
    assert_eq!(map.get([7, 2, 5]), Some(2));
    assert_eq!(map.get([9, 9, 9]), None);
    assert!(!map.is_empty());
    assert_eq!(CoordMap::new(&[]).len(), 0);
}

#[test]
fn kernel_offset_deltas_cover_the_3x3x3_kernel() {
    assert_eq!(kernel_offset_delta(0), [-1, -1, -1]);
    assert_eq!(kernel_offset_delta(13), [0, 0, 0]);
    assert_eq!(kernel_offset_delta(26), [1, 1, 1]);
    // kw fastest: k = kd*9 + kh*3 + kw; k=5 -> (kd,kh,kw)=(0,1,2).
    assert_eq!(kernel_offset_delta(5), [-1, 0, 1]);
    let deltas: std::collections::HashSet<[i32; 3]> = (0..27).map(kernel_offset_delta).collect();
    assert_eq!(deltas.len(), 27, "27 distinct offsets");
    for k in 0..27 {
        let d = kernel_offset_delta(k);
        let recomputed =
            ((d[0] + 1) as usize) * 9 + ((d[1] + 1) as usize) * 3 + (d[2] + 1) as usize;
        assert_eq!(recomputed, k, "offset {k} round trips");
    }
}

/// Sparse conv == dense reference on a 4^3 grid with holes.
#[test]
fn sparse_conv_matches_dense_reference() {
    const RES: i32 = 4;
    const CHANNELS: usize = 3;
    const OUT_CHANNELS: usize = 5;
    // Grid with holes: skip every 5th row-major cell.
    let mut coords = Vec::new();
    for x in 0..RES {
        for y in 0..RES {
            for z in 0..RES {
                let cell = (x * RES * RES + y * RES + z) as usize;
                if cell % 5 != 0 {
                    coords.push([x, y, z]);
                }
            }
        }
    }
    assert_eq!(coords.len(), 64 - 13);

    // Deterministic features/weights.
    let feats: Vec<Vec<f32>> = coords
        .iter()
        .enumerate()
        .map(|(i, _)| (0..CHANNELS).map(|c| pattern_f32(i * 13 + c * 7)).collect())
        .collect();
    // weight[o][k][c] with k = kd*9 + kh*3 + kw (flex_gemm semantics).
    let weight: Vec<Vec<Vec<f32>>> = (0..OUT_CHANNELS)
        .map(|o| {
            (0..27)
                .map(|k| {
                    (0..CHANNELS)
                        .map(|c| pattern_f32(o * 31 + k * 5 + c * 3) * 0.5)
                        .collect()
                })
                .collect()
        })
        .collect();
    let bias: Vec<f32> = (0..OUT_CHANNELS)
        .map(|o| ((o * 3) % 7) as f32 * 0.02 - 0.06)
        .collect();

    // Build the checkpoint-layout weight [Cout, 3, 3, 3, Cin] from
    // weight[o][k][c]: flat k flattens (kd, kh, kw) with kw fastest.
    let mut checkpoint_weight = vec![0.0_f32; OUT_CHANNELS * 27 * CHANNELS];
    for o in 0..OUT_CHANNELS {
        for k in 0..27 {
            let kd = k / 9;
            let kh = (k % 9) / 3;
            let kw = k % 3;
            for c in 0..CHANNELS {
                let flat = ((o * 27 + kd * 9 + kh * 3 + kw) * CHANNELS + c) as usize;
                checkpoint_weight[flat] = weight[o][k][c];
            }
        }
    }
    let vb = candle_nn::VarBuilder::from_tensors(
        [
            (
                "weight".to_string(),
                Tensor::from_vec(
                    checkpoint_weight,
                    (OUT_CHANNELS, 3, 3, 3, CHANNELS),
                    &Device::Cpu,
                )
                .expect("w"),
            ),
            (
                "bias".to_string(),
                Tensor::from_vec(bias.clone(), (OUT_CHANNELS,), &Device::Cpu).expect("b"),
            ),
        ]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let conv = SparseConv3d::load(vb, CHANNELS, OUT_CHANNELS, true).expect("conv load");
    let feats_tensor = tensor_from(
        feats.iter().flatten().copied().collect(),
        (coords.len(), CHANNELS),
    );
    let out = conv.forward(&coords, &feats_tensor).expect("conv forward");
    assert_eq!(out.dims(), &[coords.len(), OUT_CHANNELS]);
    let out_values = out.to_vec2::<f32>().expect("out");

    let expected = dense_conv_reference(&coords, &feats, &weight, &bias, CHANNELS, OUT_CHANNELS);
    let mut max_diff = 0.0_f32;
    for (i, row) in expected.iter().enumerate() {
        for (o, &value) in row.iter().enumerate() {
            max_diff = max_diff.max((out_values[i][o] - value).abs());
        }
    }
    println!("sparse conv vs dense reference max_abs_diff: {max_diff:.3e}");
    assert!(max_diff < 1e-4, "conv parity max_diff {max_diff:.3e}");
}

/// Hand-picked kernels: identity (center only) and a +x shift kernel.
#[test]
fn sparse_conv_hand_picked_kernels() {
    const CHANNELS: usize = 2;
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 0, 0], [1, 1, 0], [3, 0, 0]];
    let feats = tensor_from(
        (0..coords.len() * CHANNELS).map(pattern_f32).collect(),
        (coords.len(), CHANNELS),
    );
    let value_at = |row: usize, c: usize| pattern_f32(row * CHANNELS + c);

    // Identity kernel: only the center offset (k=13) has W[o,13,c] = I.
    let mut w = vec![0.0_f32; CHANNELS * 27 * CHANNELS];
    for c in 0..CHANNELS {
        let flat = ((c * 27 + 13) * CHANNELS + c) as usize;
        w[flat] = 1.0;
    }
    let vb = candle_nn::VarBuilder::from_tensors(
        [(
            "weight".to_string(),
            Tensor::from_vec(w, (CHANNELS, 3, 3, 3, CHANNELS), &Device::Cpu).expect("w"),
        )]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let conv = SparseConv3d::load(vb, CHANNELS, CHANNELS, false).expect("load");
    let out = conv.forward(&coords, &feats).expect("forward");
    let out = out.to_vec2::<f32>().expect("out");
    for i in 0..coords.len() {
        for c in 0..CHANNELS {
            assert!(
                (out[i][c] - value_at(i, c)).abs() < 1e-6,
                "identity kernel row {i} ch {c}"
            );
        }
    }

    // Shift kernel: k=22 (delta +x: kd=2, kh=1, kw=1) with W[o,22,c] = I,
    // zero elsewhere -> out[i] = feats[neighbor at +x] or 0 when missing.
    let mut w = vec![0.0_f32; CHANNELS * 27 * CHANNELS];
    for c in 0..CHANNELS {
        let flat = ((c * 27 + 22) * CHANNELS + c) as usize;
        w[flat] = 1.0;
    }
    let vb = candle_nn::VarBuilder::from_tensors(
        [(
            "weight".to_string(),
            Tensor::from_vec(w, (CHANNELS, 3, 3, 3, CHANNELS), &Device::Cpu).expect("w"),
        )]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let conv = SparseConv3d::load(vb, CHANNELS, CHANNELS, false).expect("load");
    let out = conv.forward(&coords, &feats).expect("forward");
    let out = out.to_vec2::<f32>().expect("out");
    let lookup: HashMap<[i32; 3], usize> =
        coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    for (i, coord) in coords.iter().enumerate() {
        let neighbor = lookup.get(&[coord[0] + 1, coord[1], coord[2]]);
        for c in 0..CHANNELS {
            let expected = neighbor.map(|&src| value_at(src, c)).unwrap_or(0.0);
            assert!(
                (out[i][c] - expected).abs() < 1e-6,
                "shift kernel row {i} ch {c}"
            );
        }
    }
}

/// The loader must read `[Cout, kd, kh, kw, Cin]` (flex_gemm) and produce
/// `[27, Cout, Cin]` with k = kd*9 + kh*3 + kw.
#[test]
fn sparse_conv_loader_transposes_flex_gemm_layout() {
    const CIN: usize = 2;
    const COUT: usize = 3;
    // Distinct values so any index permutation error shows up.
    let values: Vec<f32> = (0..COUT * 27 * CIN)
        .map(|i| (i as f32 + 1.0) / 100.0)
        .collect();
    // checkpoint[o, kd, kh, kw, c] = values[(o*27 + kd*9 + kh*3 + kw)*CIN + c]
    let vb = candle_nn::VarBuilder::from_tensors(
        [(
            "weight".to_string(),
            Tensor::from_vec(values.clone(), (COUT, 3, 3, 3, CIN), &Device::Cpu).expect("w"),
        )]
        .into_iter()
        .collect(),
        candle_core::DType::F32,
        &Device::Cpu,
    );
    let conv = SparseConv3d::load(vb, CIN, COUT, false).expect("load");
    let w27 = conv.weight27().to_vec3::<f32>().expect("w27");
    assert_eq!(w27.len(), 27);
    for k in 0..27 {
        let kd = k / 9;
        let kh = (k % 9) / 3;
        let kw = k % 3;
        for o in 0..COUT {
            for c in 0..CIN {
                let expected = values[(o * 27 + kd * 9 + kh * 3 + kw) * CIN + c];
                assert_eq!(w27[k][o][c], expected, "k={k} o={o} c={c}");
            }
        }
    }
}

/// C2S child ordering and `[child, channel]` feature selection (hand case).
#[test]
fn c2s_order_and_feature_selection() {
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 0, 0], [0, 1, 1]];
    const CHILD_CHANNELS: usize = 2;
    // conv1-style feats [3, 8*2]: row = parent*8 + subidx, value pattern.
    let feats: Vec<f32> = (0..3 * 16).map(|i| (i as f32) / 10.0).collect();
    // leaves: p0 {0, 3, 5}, p1 {1}, p2 none (logits > 0).
    let subdiv = tensor_from(
        vec![
            1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, // p0: 0,3,5
            -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, // p1: 1
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, // p2: none
        ],
        (3, 8),
    );
    let out =
        channel_to_spatial(&coords, &tensor_from(feats.clone(), (3, 16)), &subdiv).expect("c2s");
    // Children grouped by parent in parent order, ascending subidx
    // (subidx = dx + 2*dy + 4*dz).
    assert_eq!(out.coords, vec![[0, 0, 0], [1, 1, 0], [1, 0, 1], [3, 0, 0]]);
    let selected = out
        .feats
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("feats");
    let src = |parent: usize, subidx: usize| (parent * 8 + subidx) * CHILD_CHANNELS;
    let expected_rows = [src(0, 0), src(0, 3), src(0, 5), src(1, 1)];
    for (row, start) in expected_rows.iter().enumerate() {
        for ch in 0..CHILD_CHANNELS {
            assert_eq!(selected[row * CHILD_CHANNELS + ch], feats[start + ch]);
        }
    }
    // sub_binary echoes the binarized logits.
    let sub = out
        .sub_binary
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("sub");
    assert_eq!(sub[0], 1.0);
    assert_eq!(sub[1], 0.0);
    assert_eq!(sub[9], 1.0);
    assert_eq!(sub.len(), 24);
}

/// Full round trip: all 8 children per parent — S2C -> C2S(all leaves)
/// restores the original row-major coords and features.
#[test]
fn c2s_s2c_round_trip() {
    // Parents: row-major subset of the 2^3 parent grid.
    let parent_coords: Vec<[i32; 3]> = vec![[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 1, 1]];
    const C: usize = 4;
    // Children: every child of every parent, row-major over the 4^3 grid
    // (grouped by parent then ascending subidx == row-major child order).
    let mut coords = Vec::new();
    let mut feats = Vec::new();
    for parent in &parent_coords {
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    coords.push([parent[0] * 2 + dx, parent[1] * 2 + dy, parent[2] * 2 + dz]);
                    for c in 0..C {
                        feats.push(((coords.len() * 5 + c * 3) % 13) as f32 * 0.1 - 0.6);
                    }
                }
            }
        }
    }
    let feats_tensor = tensor_from(feats.clone(), (coords.len(), C));

    let s2c = spatial_to_channel(&coords, &feats_tensor).expect("s2c");
    assert_eq!(s2c.coords, parent_coords, "parents row-major sorted");
    let s2c_values = s2c.feats.to_vec2::<f32>().expect("s2c feats");
    assert_eq!(s2c_values.len(), parent_coords.len());
    // Each parent row holds its children at slot subidx = dx+2dy+4dz,
    // layout [parent, child, channel].
    let lookup: HashMap<[i32; 3], usize> =
        coords.iter().enumerate().map(|(i, c)| (*c, i)).collect();
    for p in 0..parent_coords.len() {
        let parent = parent_coords[p];
        for dz in 0..2usize {
            for dy in 0..2usize {
                for dx in 0..2usize {
                    let subidx = dx + 2 * dy + 4 * dz;
                    let child = lookup[&[
                        parent[0] * 2 + dx as i32,
                        parent[1] * 2 + dy as i32,
                        parent[2] * 2 + dz as i32,
                    ]];
                    for c in 0..C {
                        assert_eq!(
                            s2c_values[p][subidx * C + c],
                            feats[child * C + c],
                            "parent {p} subidx {subidx} ch {c}"
                        );
                    }
                }
            }
        }
    }

    // C2S with every child a leaf restores the input exactly.
    let ones = Tensor::from_vec(
        vec![1.0_f32; 8 * parent_coords.len()],
        (parent_coords.len(), 8),
        &Device::Cpu,
    )
    .expect("ones");
    let c2s = channel_to_spatial(&s2c.coords, &s2c.feats, &ones).expect("c2s");
    assert_eq!(c2s.coords, coords, "round-trip coords identical");
    let c2s_values = c2s
        .feats
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("c2s feats");
    assert_eq!(c2s_values, feats, "round-trip feats identical");
}

/// S2C zero-fills children that are not present (the reference zero-init).
#[test]
fn s2c_zero_fills_missing_children() {
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 1, 1]];
    const C: usize = 2;
    let feats = tensor_from(vec![1.0, 2.0, 3.0, 4.0], (2, C));
    let out = spatial_to_channel(&coords, &feats).expect("s2c");
    // Both children share the parent [0,0,0]: subidx 0 and 7.
    assert_eq!(out.coords, vec![[0, 0, 0]]);
    let values = out
        .feats
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("feats");
    assert_eq!(values.len(), 8 * C);
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 2.0);
    assert_eq!(values[7 * C], 3.0);
    assert_eq!(values[7 * C + 1], 4.0);
    for slot in 1..7 {
        for c in 0..C {
            assert_eq!(values[slot * C + c], 0.0, "slot {slot} ch {c} zero-filled");
        }
    }
}

/// The skin-connection helper reproduces `repeat_interleave(dim=1)`.
#[test]
fn repeat_channels_matches_repeat_interleave() {
    let feats = tensor_from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3));
    let out = repeat_channels(&feats, 2).expect("repeat");
    assert_eq!(out.dims(), &[2, 6]);
    let values = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("values");
    assert_eq!(
        values,
        vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0]
    );
    // repeats = 1 is a no-op clone.
    let same = repeat_channels(&feats, 1).expect("repeat 1");
    assert_eq!(
        same.flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("v"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

/// Naive per-head softmax attention reference (plain loops, no candle).
fn naive_attention(
    q: &[Vec<f32>], // [H][T*hd]
    k: &[Vec<f32>],
    v: &[Vec<f32>],
    heads: usize,
    tokens: usize,
    kv_tokens: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; heads * tokens * head_dim];
    for h in 0..heads {
        for t in 0..tokens {
            let mut scores = vec![0.0_f32; kv_tokens];
            let mut max_score = f32::NEG_INFINITY;
            for t2 in 0..kv_tokens {
                let mut dot = 0.0_f32;
                for d in 0..head_dim {
                    dot += q[h][t * head_dim + d] * k[h][t2 * head_dim + d];
                }
                scores[t2] = dot * scale;
                max_score = max_score.max(scores[t2]);
            }
            let mut sum = 0.0_f32;
            for s in scores.iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            for s in scores.iter_mut() {
                *s /= sum;
            }
            for t2 in 0..kv_tokens {
                for d in 0..head_dim {
                    out[h * tokens * head_dim + t * head_dim + d] +=
                        scores[t2] * v[h][t2 * head_dim + d];
                }
            }
        }
    }
    out
}

fn pattern_tensor_3d(h: usize, t: usize, hd: usize, seed: usize) -> Tensor {
    let values: Vec<f32> = (0..h * t * hd)
        .map(|i| pattern_f32(i * 7 + seed * 3))
        .collect();
    Tensor::from_vec(values, (h, t, hd), &Device::Cpu).expect("tensor")
}

#[test]
fn sparse_self_attention_matches_naive() -> candle_core::Result<()> {
    const H: usize = 2;
    const T: usize = 17;
    const HD: usize = 8;
    let q = pattern_tensor_3d(H, T, HD, 1);
    let k = pattern_tensor_3d(H, T, HD, 2);
    let v = pattern_tensor_3d(H, T, HD, 3);
    // Fused qkv layout [T, 3, H, hd]: stack over axis 1 -> [H, 3, T, hd],
    // then permute to [T, 3, H, hd].
    let qkv = Tensor::stack(&[&q, &k, &v], 1)?; // [H, 3, T, hd]
    let qkv = qkv.permute((2, 1, 0, 3))?.contiguous()?; // [T, 3, H, hd]
    for chunk in [5usize, T] {
        let out = sparse_self_attention(&qkv, chunk).expect("attn");
        assert_eq!(out.dims(), &[T, H, HD]);
        let out = out
            .permute((1, 0, 2))?
            .contiguous()?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("out"); // [H, T, hd]
        let qv = q.flatten_all().and_then(|t| t.to_vec1::<f32>()).expect("q");
        let kv = k.flatten_all().and_then(|t| t.to_vec1::<f32>()).expect("k");
        let vv = v.flatten_all().and_then(|t| t.to_vec1::<f32>()).expect("v");
        let expected = naive_attention(
            &[qv[0..T * HD].to_vec(), qv[T * HD..2 * T * HD].to_vec()],
            &[kv[0..T * HD].to_vec(), kv[T * HD..2 * T * HD].to_vec()],
            &[vv[0..T * HD].to_vec(), vv[T * HD..2 * T * HD].to_vec()],
            H,
            T,
            T,
            HD,
        );
        let max_diff = expected
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("sparse self attention chunk {chunk} max_abs_diff: {max_diff:.3e}");
        assert!(
            max_diff < 1e-5,
            "self attention chunk {chunk} diff {max_diff:.3e}"
        );
    }
    Ok(())
}

#[test]
fn sparse_cross_attention_matches_naive() -> candle_core::Result<()> {
    const H: usize = 2;
    const T: usize = 9;
    const TKV: usize = 7;
    const HD: usize = 8;
    let q_h = pattern_tensor_3d(H, T, HD, 4); // [H, T, hd] (naive reference layout)
    let q = q_h.permute((1, 0, 2))?.contiguous()?; // [T, H, hd] helper layout
    let k = pattern_tensor_3d(H, TKV, HD, 5);
    let v = pattern_tensor_3d(H, TKV, HD, 6);
    // kv layout [Tkv, 2, H, hd].
    let kv = Tensor::stack(&[&k, &v], 1)?; // [H, 2, Tkv, hd]
    let kv = kv.permute((2, 1, 0, 3))?.contiguous()?; // [Tkv, 2, H, hd]
    for chunk in [3usize, T] {
        let out = sparse_cross_attention(&q, &kv, chunk).expect("attn");
        assert_eq!(out.dims(), &[T, H, HD]);
        let out = out
            .permute((1, 0, 2))?
            .contiguous()?
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("out");
        let qv = q_h
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("q");
        let kvv = k.flatten_all().and_then(|t| t.to_vec1::<f32>()).expect("k");
        let vv = v.flatten_all().and_then(|t| t.to_vec1::<f32>()).expect("v");
        let expected = naive_attention(
            &[qv[0..T * HD].to_vec(), qv[T * HD..2 * T * HD].to_vec()],
            &[
                kvv[0..TKV * HD].to_vec(),
                kvv[TKV * HD..2 * TKV * HD].to_vec(),
            ],
            &[
                vv[0..TKV * HD].to_vec(),
                vv[TKV * HD..2 * TKV * HD].to_vec(),
            ],
            H,
            T,
            TKV,
            HD,
        );
        let max_diff = expected
            .iter()
            .zip(out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        println!("sparse cross attention chunk {chunk} max_abs_diff: {max_diff:.3e}");
        assert!(
            max_diff < 1e-5,
            "cross attention chunk {chunk} diff {max_diff:.3e}"
        );
    }
    Ok(())
}
