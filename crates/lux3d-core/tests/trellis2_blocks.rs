//! Unit tests for the TRELLIS.2 transformer machinery
//! (`runtime/trellis2_blocks.rs`): 3D RoPE math, q/k RMS norm, adaLN
//! modulation chunking, timestep embedding and tiny block forwards.
//!
//! All expected values are hand-computed from the reference formulas
//! (`trellis2/modules/attention/rope.py`, `attention/modules.py`,
//! `models/sparse_structure_flow.py`, `modules/transformer/modulated.py`).

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    Trellis2AdaLn, Trellis2DenseBlock, Trellis2QkRmsNorm, Trellis2Rope3d, Trellis2TimestepEmbedder,
    chunk_modulation, cubic_token_coords, timestep_embedding,
};

fn const_tensor(shape: &[usize], value: f32) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(vec![value; len], shape.to_vec(), &Device::Cpu).expect("tensor")
}

fn pattern_tensor(shape: &[usize]) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn tensor_map(entries: Vec<(&str, Tensor)>) -> candle_nn::VarBuilder<'static> {
    let map: HashMap<String, Tensor> = entries
        .into_iter()
        .map(|(name, tensor)| (name.to_string(), tensor))
        .collect();
    candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Hand-computed RoPE tables: head_dim 12 -> freq_dim 2 per axis,
/// freqs = [1.0, 10000^-0.5]; coord (1, 0, 0) rotates the x-frequency pairs
/// by angle 1 / 0.01, leaves y/z and the pad pair at angle 0.
#[test]
fn rope_tables_match_reference_formula() {
    let rope = Trellis2Rope3d::new(12);
    let coords: Vec<i64> = vec![1, 0, 0, 0, 1, 0];
    let (cos, sin) = rope.cos_sin(&coords, &Device::Cpu).expect("cos_sin");
    assert_eq!(cos.dims(), &[2, 6]);
    let cos_values = cos.to_vec2::<f32>().expect("cos");
    let sin_values = sin.to_vec2::<f32>().expect("sin");
    let freq2 = 1.0 / 10000_f32.powf(0.5);
    for (token, coord) in coords.chunks_exact(3).enumerate() {
        // Axis-major angle layout [x f0, x f1, y f0, y f1, z f0, z f1].
        let angles = [
            coord[0] as f32 * 1.0,
            coord[0] as f32 * freq2,
            coord[1] as f32 * 1.0,
            coord[1] as f32 * freq2,
            coord[2] as f32 * 1.0,
            coord[2] as f32 * freq2,
        ];
        for (j, &angle) in angles.iter().enumerate() {
            assert!(
                (cos_values[token][j] - angle.cos()).abs() < 1e-6,
                "cos[{token}][{j}]"
            );
            assert!(
                (sin_values[token][j] - angle.sin()).abs() < 1e-6,
                "sin[{token}][{j}]"
            );
        }
    }
}

/// The complex rotation applied per interleaved pair:
/// `out = (re + i*im) * (cos + i*sin) = (re*cos - im*sin) + i*(re*sin +
/// im*cos)`; the zero-angle pad pair is the identity.
#[test]
fn rope_apply_matches_complex_product_and_preserves_norm() {
    let rope = Trellis2Rope3d::new(12);
    let coords: Vec<i64> = vec![1, 0, 0];
    let (cos, sin) = rope.cos_sin(&coords, &Device::Cpu).expect("cos_sin");
    // 6 complex pairs.
    let x: Vec<f32> = vec![1.0, 0.0, 2.0, 3.0, 5.0, 0.0, 7.0, 1.0, 0.0, 2.0, 4.0, 4.0];
    let xs = Tensor::from_vec(x.clone(), (1, 1, 1, 12), &Device::Cpu).expect("x");
    let out = rope.apply(&xs, &cos, &sin).expect("apply");
    assert_eq!(out.dims(), &[1, 1, 1, 12]);
    let out = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out");

    let freq2 = 1.0 / 10000_f32.powf(0.5);
    let phases = [(1.0_f32, 1.0_f32), (freq2, freq2), (0.0, 0.0)];
    for pair in 0..6usize {
        let (angle_cos, angle_sin) = if pair < 2 {
            (phases[pair].0.cos(), phases[pair].1.sin())
        } else {
            (1.0, 0.0) // z coordinates zero + pad pair: angle 0.
        };
        let (re, im) = (x[2 * pair], x[2 * pair + 1]);
        let want_re = re * angle_cos - im * angle_sin;
        let want_im = re * angle_sin + im * angle_cos;
        assert!((out[2 * pair] - want_re).abs() < 1e-5, "pair {pair} re");
        assert!((out[2 * pair + 1] - want_im).abs() < 1e-5, "pair {pair} im");
    }
    // Rotation is orthogonal: energy per token is preserved.
    let energy_in: f32 = x.chunks(2).map(|p| p[0] * p[0] + p[1] * p[1]).sum();
    let energy_out: f32 = out.chunks(2).map(|p| p[0] * p[0] + p[1] * p[1]).sum();
    assert!((energy_in - energy_out).abs() < 1e-4);
}

/// Dense-grid token order: `index = x * res^2 + y * res + z` (meshgrid ij,
/// row-major flatten), matching `SparseStructureFlowModel` rope_phases.
#[test]
fn cubic_token_coords_are_row_major_ij_meshgrid() {
    let coords = cubic_token_coords(3);
    assert_eq!(coords.len(), 27 * 3);
    let index = 1 * 9 + 2 * 3 + 0; // x=1, y=2, z=0
    assert_eq!(&coords[index * 3..index * 3 + 3], &[1, 2, 0]);
    let last = 26;
    assert_eq!(&coords[last * 3..last * 3 + 3], &[2, 2, 2]);
}

/// `F.normalize(x.float(), dim=-1) * gamma * sqrt(head_dim)` per head.
#[test]
fn qk_rms_norm_matches_reference() {
    let vb = tensor_map(vec![("gamma", const_tensor(&[1, 4], 1.0))]);
    let norm = Trellis2QkRmsNorm::load(vb, 1, 4).expect("norm");
    let xs =
        Tensor::from_vec(vec![3.0_f32, 4.0, 0.0, 0.0], (1, 1, 1, 4), &Device::Cpu).expect("xs");
    let out = norm.forward(&xs).expect("forward");
    let out = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out");
    // unit-normalized * gamma * sqrt(4)
    let expected = [0.6 * 2.0, 0.8 * 2.0, 0.0, 0.0];
    for (a, b) in out.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
    // Non-uniform gamma scales per head-dim element.
    let vb = tensor_map(vec![("gamma", const_tensor(&[1, 4], 2.0))]);
    let norm = Trellis2QkRmsNorm::load(vb, 1, 4).expect("norm");
    let out = norm.forward(&xs).expect("forward2");
    let out = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out2");
    for (a, b) in out.iter().zip([2.4, 3.2, 0.0, 0.0].iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

/// `timestep_embedding`: freqs = exp(-ln(10000) * arange(half)/half),
/// `[cos(t*freqs), sin(t*freqs)]`.
#[test]
fn timestep_embedding_matches_reference() {
    let emb = timestep_embedding(500.0, 4, &Device::Cpu).expect("emb");
    let values = emb.to_vec1::<f32>().expect("values");
    let freq1 = (-10000.0_f64.ln() / 2.0).exp() as f32; // exp(-ln(10000)*1/2)
    let expected = [
        500.0f32.cos(),
        (500.0 * freq1).cos(),
        500.0f32.sin(),
        (500.0 * freq1).sin(),
    ];
    for (a, b) in values.iter().zip(expected.iter()) {
        assert!((a - b).abs() < 1e-5, "{a} vs {b}");
    }
}

/// `TimestepEmbedder.forward` output shape `[1, hidden]`.
#[test]
fn timestep_embedder_forward_shape() {
    let hidden = 8;
    let vb = tensor_map(vec![
        ("mlp.0.weight", pattern_tensor(&[hidden, 256])),
        ("mlp.0.bias", const_tensor(&[hidden], 0.1)),
        ("mlp.2.weight", pattern_tensor(&[hidden, hidden])),
        ("mlp.2.bias", const_tensor(&[hidden], -0.05)),
    ]);
    let embedder = Trellis2TimestepEmbedder::load(vb, hidden).expect("embedder");
    let out = embedder.forward(937.5).expect("forward");
    assert_eq!(out.dims(), &[1, hidden]);
}

/// Shared adaLN `SiLU -> Linear(hidden, 6*hidden)` output shape.
#[test]
fn ada_ln_forward_shape() {
    let hidden = 8;
    let vb = tensor_map(vec![
        ("1.weight", pattern_tensor(&[6 * hidden, hidden])),
        ("1.bias", const_tensor(&[6 * hidden], 0.2)),
    ]);
    let ada_ln = Trellis2AdaLn::load(vb, hidden).expect("ada_ln");
    let t_emb = const_tensor(&[1, hidden], 1.0);
    let out = ada_ln.forward(&t_emb).expect("forward");
    assert_eq!(out.dims(), &[1, 6 * hidden]);
}

/// adaLN chunk order: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
/// gate_mlp) from `chunk(6, dim=1)`, broadcast to `[B, 1, C]`.
#[test]
fn ada_ln_modulation_chunks_in_reference_order() {
    let values: Vec<f32> = (1..=12).map(|v| v as f32).collect();
    let mod_vec = Tensor::from_vec(values, (1, 12), &Device::Cpu).expect("mod");
    let chunks = chunk_modulation(&mod_vec).expect("chunks");
    let expected: [[f32; 2]; 6] = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
    ];
    for (chunk, want) in chunks.iter().zip(expected.iter()) {
        assert_eq!(chunk.dims(), &[1, 1, 2]);
        let values = chunk
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("v");
        assert_eq!(values.as_slice(), want.as_slice());
    }
}

/// Tiny `Trellis2DenseBlock` forward: shape preserved, finite output, and
/// the block output changes when the adaLN vector changes (modulation is
/// actually consumed).
#[test]
fn tiny_dense_block_forward_shape() {
    let channels = 12;
    let heads = 2; // head_dim 6 -> freq_dim 1
    let cond_channels = 8;
    let mlp_hidden = 24;
    let mut entries = Vec::new();
    entries.push(("norm2.weight", const_tensor(&[channels], 1.0)));
    entries.push(("norm2.bias", const_tensor(&[channels], 0.0)));
    entries.push(("modulation", pattern_tensor(&[6 * channels])));
    entries.push((
        "self_attn.to_qkv.weight",
        pattern_tensor(&[3 * channels, channels]),
    ));
    entries.push(("self_attn.to_qkv.bias", const_tensor(&[3 * channels], 0.0)));
    entries.push((
        "self_attn.to_out.weight",
        pattern_tensor(&[channels, channels]),
    ));
    entries.push(("self_attn.to_out.bias", const_tensor(&[channels], 0.0)));
    entries.push((
        "self_attn.q_rms_norm.gamma",
        const_tensor(&[heads, channels / heads], 1.0),
    ));
    entries.push((
        "self_attn.k_rms_norm.gamma",
        const_tensor(&[heads, channels / heads], 1.0),
    ));
    entries.push((
        "cross_attn.to_q.weight",
        pattern_tensor(&[channels, channels]),
    ));
    entries.push(("cross_attn.to_q.bias", const_tensor(&[channels], 0.0)));
    entries.push((
        "cross_attn.to_kv.weight",
        pattern_tensor(&[2 * channels, cond_channels]),
    ));
    entries.push(("cross_attn.to_kv.bias", const_tensor(&[2 * channels], 0.0)));
    entries.push((
        "cross_attn.to_out.weight",
        pattern_tensor(&[channels, channels]),
    ));
    entries.push(("cross_attn.to_out.bias", const_tensor(&[channels], 0.0)));
    entries.push((
        "cross_attn.q_rms_norm.gamma",
        const_tensor(&[heads, channels / heads], 1.0),
    ));
    entries.push((
        "cross_attn.k_rms_norm.gamma",
        const_tensor(&[heads, channels / heads], 1.0),
    ));
    entries.push(("mlp.mlp.0.weight", pattern_tensor(&[mlp_hidden, channels])));
    entries.push(("mlp.mlp.0.bias", const_tensor(&[mlp_hidden], 0.0)));
    entries.push(("mlp.mlp.2.weight", pattern_tensor(&[channels, mlp_hidden])));
    entries.push(("mlp.mlp.2.bias", const_tensor(&[channels], 0.0)));
    let vb = tensor_map(entries);

    let block =
        Trellis2DenseBlock::load(vb, channels, cond_channels, heads, mlp_hidden, true, true)
            .expect("block");

    let tokens = 5;
    let xs = pattern_tensor(&[1, tokens, channels]);
    let t_emb = pattern_tensor(&[1, 6 * channels]).abs().expect("abs");
    let context = pattern_tensor(&[1, 4, cond_channels]);
    let rope = Trellis2Rope3d::new(channels / heads);
    let coords: Vec<i64> = (0..tokens as i64)
        .flat_map(|i| vec![i, 2 * i, 3 * i])
        .collect();
    let (cos, sin) = rope.cos_sin(&coords, &Device::Cpu).expect("cos_sin");
    let out = block
        .forward(&xs, &t_emb, &context, &cos, &sin)
        .expect("forward");
    assert_eq!(out.dims(), &[1, tokens, channels]);
    let values = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out");
    assert!(values.iter().all(|v| v.is_finite()));

    // Modulating differently must change the output (adaLN is wired in).
    let t_emb_alt = const_tensor(&[1, 6 * channels], 3.0);
    let out_alt = block
        .forward(&xs, &t_emb_alt, &context, &cos, &sin)
        .expect("forward alt");
    let values_alt = out_alt
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out alt");
    let max_diff = values
        .iter()
        .zip(values_alt.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff > 1e-3, "adaLN modulation had no effect");
}
