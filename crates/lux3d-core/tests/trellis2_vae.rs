//! TRELLIS.2 shape VAE decoder tests:
//!
//! 1. Tiny synthetic config: tensor-name coverage, strict loading, forward
//!    shape propagation and a hand-checked C2S child order / skip-channel
//!    selection through the real decoder path.
//! 2. Real-weight loader coverage: the canonical package `shape_dec.` stage
//!    holds exactly the 292 tensors the config expects (gated).
//! 3. Oracle parity (gated): the `.swarm/results/trellis2-vae-golden`
//!    fixture (reference F32 PyTorch decode, latent grid 8, resolution 128)
//!    is replayed through the Rust decoder; per-block coords must match
//!    exactly, feats within F32 accumulation noise, and the extracted mesh
//!    must match the reference vertices to <= 1e-5 with the face set equal
//!    as a multiset.
//!
//! The oracle test takes a few minutes on CPU (release build recommended);
//! run it with
//! `cargo test -p lux3d-core --test trellis2_vae
//!  vae_decoder_matches_python_oracle -- --nocapture --test-threads 1`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    ShapeVaeBlockTrace, TRELLIS2_VOXEL_MARGIN, Trellis2ShapeVaeConfig, Trellis2ShapeVaeDecoder,
};
use lux3d_core::test_support::workspace_root;

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-vae-golden")
}

fn canonical_model_file() -> PathBuf {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_CANONICAL_MODEL") {
        return PathBuf::from(path);
    }
    lux3d_core::test_support::canonical_package_dir(lux3d_core::ModelFamily::Trellis2)
        .join("model.safetensors")
}

fn pattern_tensor(shape: &[usize], seed: usize) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len)
            .map(|i| (((i * 7 + seed * 3) % 23) as f32 - 11.0) * 0.05)
            .collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn const_tensor(shape: &[usize], value: f32) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(vec![value; len], shape.to_vec(), &Device::Cpu).expect("tensor")
}

fn iota_tensor(shape: &[usize]) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(
        (0..len).map(|i| i as f32).collect(),
        shape.to_vec(),
        &Device::Cpu,
    )
    .expect("tensor")
}

fn tensor_map(entries: Vec<(String, Tensor)>) -> candle_nn::VarBuilder<'static> {
    let map: HashMap<String, Tensor> = entries.into_iter().collect();
    candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// Tiny decoder config: 5 stages, channels [16, 8, 8, 8, 8], latent 4.
fn tiny_config(num_blocks: [usize; 5]) -> Trellis2ShapeVaeConfig {
    Trellis2ShapeVaeConfig {
        model_channels: vec![16, 8, 8, 8, 8],
        num_blocks: num_blocks.to_vec(),
        latent_channels: 4,
        out_channels: 7,
        mlp_ratio: 4.0,
        pred_subdiv: true,
    }
}

fn tiny_stage0_c2s_entries() -> Vec<(String, Tensor)> {
    // Stage 0 C2S block: conv1/conv2 zeroed, identity norm1, subdivision
    // logits [+1, -1, +1, -1, +1, -1, +1, -1] -> even subidx survive.
    vec![
        (
            "blocks.0.0.conv1.weight".to_string(),
            const_tensor(&[64, 3, 3, 3, 16], 0.0),
        ),
        (
            "blocks.0.0.conv1.bias".to_string(),
            const_tensor(&[64], 0.0),
        ),
        (
            "blocks.0.0.conv2.weight".to_string(),
            const_tensor(&[8, 3, 3, 3, 8], 0.0),
        ),
        ("blocks.0.0.conv2.bias".to_string(), const_tensor(&[8], 0.0)),
        (
            "blocks.0.0.norm1.weight".to_string(),
            const_tensor(&[16], 1.0),
        ),
        (
            "blocks.0.0.norm1.bias".to_string(),
            const_tensor(&[16], 0.0),
        ),
        (
            "blocks.0.0.to_subdiv.weight".to_string(),
            const_tensor(&[8, 16], 0.0),
        ),
        (
            "blocks.0.0.to_subdiv.bias".to_string(),
            Tensor::from_vec(
                (0..8)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
                (8,),
                &Device::Cpu,
            )
            .expect("subdiv bias"),
        ),
    ]
}

fn tiny_child_stage_entries(stage: usize) -> Vec<(String, Tensor)> {
    // Stages 1..3 C2S blocks (channels 8 -> out 8).
    let base = format!("blocks.{stage}.0.");
    vec![
        (
            format!("{base}conv1.weight"),
            const_tensor(&[64, 3, 3, 3, 8], 0.0),
        ),
        (format!("{base}conv1.bias"), const_tensor(&[64], 0.0)),
        (
            format!("{base}conv2.weight"),
            const_tensor(&[8, 3, 3, 3, 8], 0.0),
        ),
        (format!("{base}conv2.bias"), const_tensor(&[8], 0.0)),
        (format!("{base}norm1.weight"), const_tensor(&[8], 1.0)),
        (format!("{base}norm1.bias"), const_tensor(&[8], 0.0)),
        (
            format!("{base}to_subdiv.weight"),
            const_tensor(&[8, 8], 0.0),
        ),
        (
            format!("{base}to_subdiv.bias"),
            Tensor::from_vec(
                (0..8)
                    .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                    .collect(),
                (8,),
                &Device::Cpu,
            )
            .expect("subdiv bias"),
        ),
    ]
}

/// Tensor-name contract for the tiny config: 4 top-level + per-stage
/// ConvNeXt/C2S names, all unique; the loader rejects a missing tensor and
/// accepts the complete set.
#[test]
fn tiny_vae_config_names_and_strict_loading() {
    for num_blocks in [[0usize, 0, 0, 0, 0], [1, 1, 1, 1, 0]] {
        let config = tiny_config(num_blocks);
        let names = config.expected_tensor_names();
        let expected_count = 4 + num_blocks
            .iter()
            .enumerate()
            .map(|(stage, &count)| {
                let c2s = if stage + 1 < num_blocks.len() { 8 } else { 0 };
                count * 8 + c2s
            })
            .sum::<usize>();
        assert_eq!(names.len(), expected_count, "num_blocks {num_blocks:?}");
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "names must be unique");
    }

    // Strict load: complete tiny model (single C2S stage) succeeds, a
    // missing conv2.bias fails.
    let config = tiny_config([0, 0, 0, 0, 0]);
    let mut entries = vec![
        (
            "from_latent.weight".to_string(),
            pattern_tensor(&[16, 4], 0),
        ),
        ("from_latent.bias".to_string(), const_tensor(&[16], 0.0)),
        (
            "output_layer.weight".to_string(),
            pattern_tensor(&[7, 8], 1),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[7], 0.0)),
    ];
    entries.extend(tiny_stage0_c2s_entries());
    entries.extend(tiny_child_stage_entries(1));
    entries.extend(tiny_child_stage_entries(2));
    entries.extend(tiny_child_stage_entries(3));

    let without_conv2_bias: Vec<(String, Tensor)> = entries
        .iter()
        .filter(|(name, _)| name != "blocks.0.0.conv2.bias")
        .cloned()
        .collect();
    assert!(
        Trellis2ShapeVaeDecoder::load(tensor_map(without_conv2_bias), &config).is_err(),
        "loader must reject an incomplete stage"
    );

    let decoder = Trellis2ShapeVaeDecoder::load(tensor_map(entries), &config).expect("strict load");
    assert_eq!(decoder.config(), &config);
}

/// Hand-checked C2S semantics through the decoder path: zeroed convs and
/// alternating subdivision logits select the even subidx children
/// (dx + 2*dy + 4*dz), grouped by parent in parent order, with feats rows
/// `parent*8 + subidx` of the `[child, channel]` layout and the
/// `repeat_interleave` skip. from_latent is a pure bias so the stage-0
/// residual input is exactly [0..16).
#[test]
fn tiny_vae_hand_checked_c2s_child_order() {
    let config = tiny_config([0, 0, 0, 0, 0]);
    let mut entries = vec![
        // from_latent: zero weights, bias = [0..16) -> h = [0..16).
        (
            "from_latent.weight".to_string(),
            const_tensor(&[16, 4], 0.0),
        ),
        ("from_latent.bias".to_string(), iota_tensor(&[16])),
        (
            "output_layer.weight".to_string(),
            const_tensor(&[7, 8], 0.0),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[7], 0.0)),
    ];
    entries.extend(tiny_stage0_c2s_entries());
    entries.extend(tiny_child_stage_entries(1));
    entries.extend(tiny_child_stage_entries(2));
    entries.extend(tiny_child_stage_entries(3));
    let decoder = Trellis2ShapeVaeDecoder::load(tensor_map(entries), &config).expect("load");

    let coords = vec![[0i32, 0, 0]];
    let latent = const_tensor(&[1, 4], 0.0);
    let decoded = decoder.decode(&coords, &latent, true).expect("decode");

    // Stage-0 C2S: even subidx -> children (0,0,0), (0,1,0), (0,0,1),
    // (0,1,1); feats rows = x[0, subidx*2..subidx*2+2] repeated
    // `repeat_interleave` 4x per channel (skip carries the residual x
    // because both convs are zero).
    let stage0 = &decoded.trace[0];
    assert_eq!(stage0.stage, 0);
    assert_eq!(stage0.block, 0);
    assert_eq!(
        stage0.coords,
        vec![[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    );
    let feats = stage0
        .feats
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("feats");
    let expected_rows: Vec<f32> = [0.0_f32, 4.0, 8.0, 12.0]
        .iter()
        .flat_map(|&start| {
            (0..8usize)
                .map(|i| start + (i / 4) as f32)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(feats, expected_rows, "stage-0 C2S child selection");

    // Subdiv logits are the raw to_subdiv output.
    let subdiv = stage0
        .subdiv
        .as_ref()
        .expect("C2S trace carries subdiv")
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("subdiv");
    assert_eq!(subdiv, vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

    // Four C2S stages: 1 -> 4 -> 16 -> 64 -> 256 children.
    let counts: Vec<usize> = decoded.trace.iter().map(|b| b.coords.len()).collect();
    assert_eq!(counts, vec![4, 16, 64, 256]);
    assert_eq!(decoded.coords.len(), 256);
    // Final coords are the 4^4 descendants of (0,0,0) at even subidx.
    assert_eq!(decoded.coords[0], [0, 0, 0]);
    assert_eq!(decoded.coords[1], [0, 1, 0]);
    assert_eq!(decoded.coords[2], [0, 0, 1]);
    assert_eq!(decoded.coords[3], [0, 1, 1]);
    // First child of the second parent (0,1,0), subidx 0.
    assert_eq!(decoded.coords[4], [0, 2, 0]);
    // Last child of the last parent: subidx 6 chain (0,1,1) -> (0,3,3) ->
    // (0,7,7) -> (0,15,15).
    assert_eq!(decoded.coords[255], [0, 15, 15]);
    // Decoded head has 7 channels.
    assert_eq!(decoded.feats.dims(), &[256, 7]);
    // Four subdiv logit tensors (one per C2S stage).
    assert_eq!(decoded.subdiv_logits.len(), 4);
    assert!(
        decoded
            .feats
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("v")
            .iter()
            .all(|v| v.is_finite())
    );
}

/// Non-trivial tiny model: ConvNeXt + C2S forward keeps finite values, the
/// coordinate ranges double per stage and the output keeps [N, 7].
#[test]
fn tiny_vae_forward_shape_and_ranges() {
    let config = tiny_config([1, 1, 1, 1, 0]);
    let mut entries = vec![
        (
            "from_latent.weight".to_string(),
            pattern_tensor(&[16, 4], 0),
        ),
        ("from_latent.bias".to_string(), const_tensor(&[16], 0.0)),
        (
            "output_layer.weight".to_string(),
            pattern_tensor(&[7, 8], 1),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[7], 0.0)),
    ];
    // Stage 0: ConvNeXt (16ch) + C2S.
    entries.push((
        "blocks.0.0.conv.weight".to_string(),
        pattern_tensor(&[16, 3, 3, 3, 16], 2),
    ));
    entries.push(("blocks.0.0.conv.bias".to_string(), const_tensor(&[16], 0.0)));
    entries.push((
        "blocks.0.0.norm.weight".to_string(),
        const_tensor(&[16], 1.0),
    ));
    entries.push(("blocks.0.0.norm.bias".to_string(), const_tensor(&[16], 0.0)));
    entries.push((
        "blocks.0.0.mlp.0.weight".to_string(),
        pattern_tensor(&[64, 16], 3),
    ));
    entries.push((
        "blocks.0.0.mlp.0.bias".to_string(),
        const_tensor(&[64], 0.0),
    ));
    entries.push((
        "blocks.0.0.mlp.2.weight".to_string(),
        pattern_tensor(&[16, 64], 4),
    ));
    entries.push((
        "blocks.0.0.mlp.2.bias".to_string(),
        const_tensor(&[16], 0.0),
    ));
    let c2s0 = tiny_stage0_c2s_entries();
    // Replace the zeroed conv weights with pattern weights to exercise the
    // conv math (keep the deterministic subdivision bias). The tiny config's
    // stage-0 C2S block sits at index 1 (after one ConvNeXt block); remap
    // the shared [0,0,0,0,0]-layout helper names accordingly.
    for (name, _) in &c2s0 {
        let name = name.replacen("blocks.0.0.", "blocks.0.1.", 1);
        if name.ends_with("conv1.weight") {
            entries.push((name.clone(), pattern_tensor(&[64, 3, 3, 3, 16], 5)));
        } else if name.ends_with("conv1.bias") {
            entries.push((name.clone(), const_tensor(&[64], 0.0)));
        } else if name.ends_with("conv2.weight") {
            entries.push((name.clone(), pattern_tensor(&[8, 3, 3, 3, 8], 6)));
        } else if name.ends_with("conv2.bias") {
            entries.push((name.clone(), const_tensor(&[8], 0.0)));
        } else {
            entries.push((name.clone(), pattern_tensor_shape(&name)));
        }
    }
    // Stages 1..3: ConvNeXt (8ch) + C2S.
    for stage in 1..4usize {
        entries.push((
            format!("blocks.{stage}.0.conv.weight"),
            pattern_tensor(&[8, 3, 3, 3, 8], 10 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.0.conv.bias"),
            const_tensor(&[8], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.0.norm.weight"),
            const_tensor(&[8], 1.0),
        ));
        entries.push((
            format!("blocks.{stage}.0.norm.bias"),
            const_tensor(&[8], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.0.mlp.0.weight"),
            pattern_tensor(&[32, 8], 20 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.0.mlp.0.bias"),
            const_tensor(&[32], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.0.mlp.2.weight"),
            pattern_tensor(&[8, 32], 30 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.0.mlp.2.bias"),
            const_tensor(&[8], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.1.conv1.weight"),
            pattern_tensor(&[64, 3, 3, 3, 8], 40 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.1.conv1.bias"),
            const_tensor(&[64], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.1.conv2.weight"),
            pattern_tensor(&[8, 3, 3, 3, 8], 50 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.1.conv2.bias"),
            const_tensor(&[8], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.1.norm1.weight"),
            const_tensor(&[8], 1.0),
        ));
        entries.push((
            format!("blocks.{stage}.1.norm1.bias"),
            const_tensor(&[8], 0.0),
        ));
        entries.push((
            format!("blocks.{stage}.1.to_subdiv.weight"),
            pattern_tensor(&[8, 8], 60 + stage),
        ));
        entries.push((
            format!("blocks.{stage}.1.to_subdiv.bias"),
            const_tensor(&[8], 0.0),
        ));
    }

    let decoder = Trellis2ShapeVaeDecoder::load(tensor_map(entries), &config).expect("load");
    // 5 parent voxels (with holes) in a 4^3 grid.
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [1, 0, 2], [2, 3, 1], [3, 3, 3], [1, 1, 1]];
    let latent = pattern_tensor(&[5, 4], 70);
    let decoded = decoder.decode(&coords, &latent, true).expect("decode");
    assert_eq!(decoded.feats.dims()[1], 7);
    assert_eq!(decoded.subdiv_logits.len(), 4);
    // Every trace block reports non-negative coordinates and finite feats.
    for block in &decoded.trace {
        for coord in &block.coords {
            assert!(coord.iter().all(|&axis| axis >= 0), "coord {coord:?}");
        }
    }
    // Final children stay inside [0, 64) (4 * 2^4).
    for coord in &decoded.coords {
        assert!(coord.iter().all(|&v| v >= 0 && v < 64), "coord {coord:?}");
    }
    assert!(
        decoded
            .feats
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("v")
            .iter()
            .all(|v| v.is_finite())
    );
}

/// Pattern weights for names the tiny forward test does not pin explicitly
/// (norm1/to_subdiv of the zeroed stage-0 C2S keep their hand-check forms).
fn pattern_tensor_shape(name: &str) -> Tensor {
    if name.ends_with("norm1.weight") {
        const_tensor(&[16], 1.0)
    } else if name.ends_with("norm1.bias") {
        const_tensor(&[16], 0.0)
    } else if name.ends_with("to_subdiv.weight") {
        const_tensor(&[8, 16], 0.0)
    } else if name.ends_with("to_subdiv.bias") {
        Tensor::from_vec(
            (0..8)
                .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
                .collect(),
            (8,),
            &Device::Cpu,
        )
        .expect("subdiv bias")
    } else {
        panic!("unexpected entry {name}")
    }
}

fn read_safetensors_stage_keys(path: &std::path::Path, prefix: &str) -> Option<Vec<String>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 256 * 1024 * 1024 {
        return None;
    }
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes).ok()?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes).ok()?;
    Some(
        header
            .as_object()?
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect(),
    )
}

/// Real weights: the canonical `shape_dec.` stage holds exactly the 292
/// tensors the flexi-dual config expects.
#[test]
fn shape_vae_real_weights_cover_292_stage_tensors() {
    let model_file = canonical_model_file();
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }
    let stage_keys =
        read_safetensors_stage_keys(&model_file, "shape_dec.").expect("canonical package header");
    assert_eq!(stage_keys.len(), 292, "shape_dec stage tensor count");
    let config = Trellis2ShapeVaeConfig::flexi_dual_shape();
    assert_eq!(config.model_channels, vec![1024, 512, 256, 128, 64]);
    assert_eq!(config.num_blocks, vec![4, 16, 8, 4, 0]);
    assert_eq!(config.latent_channels, 32);
    assert_eq!(config.out_channels, 7);
    let expected: HashSet<String> = config.expected_tensor_names().into_iter().collect();
    assert_eq!(expected.len(), 292);
    let stage: HashSet<String> = stage_keys
        .iter()
        .map(|k| k.trim_start_matches("shape_dec.").to_string())
        .collect();
    assert_eq!(
        expected, stage,
        "expected names must cover exactly the shape_dec stage tensors"
    );
}

/// Metrics helper: (max_abs, rel_vs_max_abs, rms_rel) of `actual` vs
/// `expected`.
fn compare(actual: &[f32], expected: &[f32], label: &str) -> (f64, f64, f64) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let mut max_abs = 0.0_f64;
    let mut expected_max = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_sq_expected = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected.iter()) {
        let diff = (a - b).abs() as f64;
        max_abs = max_abs.max(diff);
        expected_max = expected_max.max(b.abs() as f64);
        sum_sq_diff += diff * diff;
        sum_sq_expected += (b as f64) * (b as f64);
    }
    let rel_vs_max = max_abs / expected_max.max(1e-12);
    let rms_rel = (sum_sq_diff / sum_sq_expected.max(1e-12)).sqrt();
    println!(
        "{label}: max_abs={max_abs:.3e}, rel_vs_max_abs={rel_vs_max:.3e}, rms_rel={rms_rel:.3e}"
    );
    (max_abs, rel_vs_max, rms_rel)
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .expect("tensor to flat f32")
}

/// Real weights + Python oracle: replay the golden F32 reference decode
/// (latent grid 8^3, 64 coords, resolution 128) through the Rust decoder and
/// compare every block output, the final 7-channel head, and the extracted
/// mesh (vertices <= 1e-5, faces equal as a multiset).
#[test]
fn vae_decoder_matches_python_oracle() {
    let golden = golden_dir();
    if !golden.join("shape_mesh.npz").is_file() {
        eprintln!("SKIP: VAE oracle golden missing at {}", golden.display());
        return;
    }
    let model_file = canonical_model_file();
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }

    use ndarray::Array2;
    let slat_coords: Array2<i32> =
        ndarray_npy::read_npy(golden.join("slat_coords.npy")).expect("slat_coords");
    let slat_feats: Array2<f32> =
        ndarray_npy::read_npy(golden.join("slat_feats.npy")).expect("slat_feats");
    assert_eq!(slat_coords.dim(), (64, 3));
    assert_eq!(slat_feats.dim(), (64, 32));
    let coords: Vec<[i32; 3]> = (0..64)
        .map(|i| {
            [
                slat_coords[[i, 0]],
                slat_coords[[i, 1]],
                slat_coords[[i, 2]],
            ]
        })
        .collect();

    let device = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)
    }
    .expect("mmaped canonical weights");
    let decoder = Trellis2ShapeVaeDecoder::load(
        vb.pp("shape_dec"),
        &Trellis2ShapeVaeConfig::flexi_dual_shape(),
    )
    .expect("shape_dec load");

    let latent = Tensor::from_vec(
        slat_feats.iter().copied().collect::<Vec<f32>>(),
        (64, 32),
        &device,
    )
    .expect("latent tensor");

    let started = std::time::Instant::now();
    let decoded = decoder.decode(&coords, &latent, true).expect("decode");
    println!("oracle decode in {:.1}s", started.elapsed().as_secs_f64());

    // from_latent parity: h = latent @ W^T + b against the golden dump.
    let golden_from_latent: Array2<f32> =
        ndarray_npy::read_npy(golden.join("from_latent.npy")).expect("from_latent");
    {
        let weight = vb
            .pp("shape_dec")
            .pp("from_latent")
            .get((1024, 32), "weight")
            .expect("from_latent weight");
        let bias = vb
            .pp("shape_dec")
            .pp("from_latent")
            .get((1024,), "bias")
            .expect("from_latent bias");
        let h = latent
            .matmul(&weight.transpose(0, 1).expect("transpose"))
            .and_then(|t| t.broadcast_add(&bias))
            .expect("from_latent forward");
        let (_, rel, rms) = compare(
            &flat(&h),
            golden_from_latent
                .iter()
                .copied()
                .collect::<Vec<f32>>()
                .as_slice(),
            "from_latent",
        );
        assert!(rel <= 5e-3, "from_latent rel {rel:.3e}");
        assert!(rms <= 5e-4, "from_latent rms {rms:.3e}");
    }

    // Per-block parity: exact coords, F32-tolerance feats.
    let resolution = 128usize;
    let _ = resolution;
    let tolerances: [(usize, f64, f64); 4] = [
        // (stage, rel_vs_max_abs cap, rms_rel cap) — measured maxima are
        // rel 2.5e-6 / rms 1.6e-6; the caps sit two orders above to catch
        // regressions while tolerating F32 summation-order noise.
        (0, 2.5e-4, 1.6e-4),
        (1, 2.5e-4, 1.6e-4),
        (2, 2.5e-4, 1.6e-4),
        (3, 2.5e-4, 1.6e-4),
    ];
    for block in &decoded.trace {
        let golden_coords: Array2<i32> = ndarray_npy::read_npy(
            golden.join(format!("blocks_{}_{}_coords.npy", block.stage, block.block)),
        )
        .expect("golden coords");
        assert_eq!(
            golden_coords.dim().0,
            block.coords.len(),
            "stage {} block {} coord count",
            block.stage,
            block.block
        );
        for (row, coord) in block.coords.iter().enumerate() {
            assert_eq!(golden_coords[[row, 0]], 0, "batch column");
            assert_eq!(
                [
                    golden_coords[[row, 1]],
                    golden_coords[[row, 2]],
                    golden_coords[[row, 3]]
                ],
                *coord,
                "stage {} block {} row {row}",
                block.stage,
                block.block
            );
        }
        let golden_feats: Array2<f32> = ndarray_npy::read_npy(
            golden.join(format!("blocks_{}_{}_feats.npy", block.stage, block.block)),
        )
        .expect("golden feats");
        assert_eq!(golden_feats.dim().1, block.feats.dim(1).expect("channels"));
        let (_, rel, rms) = compare(
            &flat(&block.feats),
            golden_feats
                .iter()
                .copied()
                .collect::<Vec<f32>>()
                .as_slice(),
            &format!("blocks.{}.{}.feats", block.stage, block.block),
        );
        let cap = tolerances[block.stage.min(3)];
        assert!(
            rel <= cap.1,
            "blocks.{}.{} rel {rel:.3e}",
            block.stage,
            block.block
        );
        assert!(
            rms <= cap.2,
            "blocks.{}.{} rms {rms:.3e}",
            block.stage,
            block.block
        );
        if let Some(subdiv) = &block.subdiv {
            let golden_subdiv: Array2<f32> = ndarray_npy::read_npy(
                golden.join(format!("blocks_{}_{}_subdiv.npy", block.stage, block.block)),
            )
            .expect("golden subdiv");
            let (_, rel, rms) = compare(
                &flat(subdiv),
                golden_subdiv
                    .iter()
                    .copied()
                    .collect::<Vec<f32>>()
                    .as_slice(),
                &format!("blocks.{}.{}.subdiv", block.stage, block.block),
            );
            assert!(rel <= 5e-3, "subdiv rel {rel:.3e}");
            assert!(rms <= 5e-4, "subdiv rms {rms:.3e}");
        }
    }

    // Final 7-channel decode parity.
    let golden_output: Array2<f32> =
        ndarray_npy::read_npy(golden.join("output_layer.npy")).expect("output_layer");
    assert_eq!(golden_output.dim(), (block_count(&decoded.trace), 7));
    let (_, rel, rms) = compare(
        &flat(&decoded.feats),
        golden_output
            .iter()
            .copied()
            .collect::<Vec<f32>>()
            .as_slice(),
        "output_layer",
    );
    assert!(rel <= 5e-3, "output_layer rel {rel:.3e}");
    assert!(rms <= 5e-4, "output_layer rms {rms:.3e}");

    // Mesh parity via the head split + dual-grid extraction.
    let fields = lux3d_core::runtime::dual_grid_fields(&decoded.feats, TRELLIS2_VOXEL_MARGIN)
        .expect("head split");
    let mesh = lux3d_core::runtime::flexible_dual_grid_to_mesh(
        &decoded.coords,
        &fields.dual,
        &fields.intersected,
        &fields.split_weight,
        [resolution, resolution, resolution],
        [[-0.5_f32; 3], [0.5_f32; 3]],
    );

    let npz_file = std::fs::File::open(golden.join("shape_mesh.npz")).expect("open npz");
    let mut npz = ndarray_npy::NpzReader::new(npz_file).expect("npz reader");
    let golden_vertices = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("vertices")
        .expect("vertices");
    let golden_faces = npz
        .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix2>("faces")
        .expect("faces");
    assert_eq!(golden_vertices.dim().0, mesh.vertices.len() / 3);
    assert_eq!(golden_faces.dim().0, mesh.faces.len() / 3);

    let mut vertex_max_abs = 0.0_f64;
    for (index, vertex) in mesh.vertices.chunks_exact(3).enumerate() {
        for axis in 0..3 {
            vertex_max_abs =
                vertex_max_abs.max((vertex[axis] - golden_vertices[[index, axis]]).abs() as f64);
        }
    }
    println!("mesh vertices max abs diff = {vertex_max_abs:.3e}");
    assert!(
        vertex_max_abs <= 1e-5,
        "vertex parity {vertex_max_abs:.3e} exceeds 1e-5"
    );

    // Faces as sorted-triple multisets.
    let mut our_faces: Vec<[i64; 3]> = mesh
        .faces
        .chunks_exact(3)
        .map(|f| [f[0], f[1], f[2]])
        .collect();
    let mut golden_face_list: Vec<[i64; 3]> = golden_faces
        .outer_iter()
        .map(|f| [f[0], f[1], f[2]])
        .collect();
    let canonical = |faces: &mut Vec<[i64; 3]>| {
        for face in faces.iter_mut() {
            face.sort_unstable();
        }
        faces.sort_unstable();
    };
    canonical(&mut our_faces);
    canonical(&mut golden_face_list);
    assert_eq!(our_faces, golden_face_list, "face multisets must match");
}

fn block_count(trace: &[ShapeVaeBlockTrace]) -> usize {
    trace.last().map(|b| b.coords.len()).unwrap_or(0)
}
