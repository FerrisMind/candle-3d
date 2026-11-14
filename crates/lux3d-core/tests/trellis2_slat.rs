//! TRELLIS.2 shape SLAT flow DiT tests:
//!
//! 1. Static config coverage: 640 tensor names, matching the canonical
//!    package stage;
//! 2. Tiny-config synthetic forward (shape preservation, coord validation);
//! 3. Real-weight loader coverage + golden comparison against the Python
//!    SLAT oracle (`.swarm/results/trellis2-slat-golden/slat_golden.npz`):
//!    golden cond + coords + injected noise feats -> 4-step Euler trace ->
//!    `shape_slat_norm` (+ denorm). The oracle ran the flow DiT in bf16 on
//!    CUDA; the Rust path is F32, so per-step tolerances cover the
//!    reference's own precision noise (measured values are printed).
//!
//! Tests skip (with a message) when the golden npz or the canonical weights
//! package is absent. The real-weight test loads ~2.6 GiB (bf16) -> 5.2 GiB
//! F32 and runs 4 sampler steps on CPU (~15 min in the dev profile); run it
//! with `cargo test -p lux3d-core --test trellis2_slat -- --nocapture
//! --test-threads 1`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    Trellis2SamplerParams, Trellis2SlatConfig, Trellis2SlatDit, sample_sparse,
};
use lux3d_core::test_support::workspace_root;

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-slat-golden")
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

fn tensor_map(entries: Vec<(&str, Tensor)>) -> candle_nn::VarBuilder<'static> {
    let map: HashMap<String, Tensor> = entries
        .into_iter()
        .map(|(name, tensor)| (name.to_string(), tensor))
        .collect();
    candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu)
}

/// The real checkpoint config must map to exactly 640 unique tensor names,
/// matching the merged package stage (verified against the safetensors
/// header in `slat_dit_real_weights_load_and_golden_sample`).
#[test]
fn slat_config_expected_tensor_names_cover_the_checkpoint() {
    for (config, resolution) in [
        (Trellis2SlatConfig::shape_512(), 32usize),
        (Trellis2SlatConfig::shape_1024(), 64usize),
    ] {
        assert_eq!(config.resolution, resolution);
        assert_eq!(config.in_channels, 32);
        assert_eq!(config.out_channels, 32);
        assert_eq!(config.model_channels, 1536);
        assert_eq!(config.cond_channels, 1024);
        assert_eq!(config.num_blocks, 30);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.head_dim(), 128);
        assert_eq!(config.mlp_hidden(), 8192);
        let names = config.expected_tensor_names();
        assert_eq!(names.len(), 640, "shape_dit stage tensor count");
        let unique: HashSet<&String> = names.iter().collect();
        assert_eq!(unique.len(), 640, "expected names must be unique");
        assert!(names.contains(&"input_layer.weight".to_string()));
        assert!(names.contains(&"blocks.0.norm2.weight".to_string()));
        assert!(names.contains(&"blocks.29.self_attn.q_rms_norm.gamma".to_string()));
        assert!(names.contains(&"blocks.29.cross_attn.k_rms_norm.gamma".to_string()));
        assert!(names.contains(&"out_layer.bias".to_string()));
    }
}

/// Tiny-config SLAT DiT (synthetic weights): forward preserves `[N, C]`,
/// RoPE tables follow the token coords, invalid inputs are rejected.
#[test]
fn tiny_slat_dit_forward_shape() {
    let channels = 16;
    let heads = 2; // head_dim 8, rope freq_dim 1
    let head_dim = channels / heads;
    let cond_channels = 8;
    let mlp_hidden = 24;
    let in_channels = 4;
    let resolution = 8;
    let config = Trellis2SlatConfig {
        resolution,
        in_channels,
        out_channels: 4,
        model_channels: channels,
        cond_channels,
        num_blocks: 1,
        num_heads: heads,
        mlp_ratio: mlp_hidden as f64 / channels as f64,
        share_mod: true,
        qk_rms_norm: true,
        qk_rms_norm_cross: true,
    };
    let entries: Vec<(&str, Tensor)> = vec![
        (
            "t_embedder.mlp.0.weight",
            pattern_tensor(&[channels, 256], 0),
        ),
        ("t_embedder.mlp.0.bias", const_tensor(&[channels], 0.1)),
        (
            "t_embedder.mlp.2.weight",
            pattern_tensor(&[channels, channels], 1),
        ),
        ("t_embedder.mlp.2.bias", const_tensor(&[channels], -0.05)),
        (
            "adaLN_modulation.1.weight",
            pattern_tensor(&[6 * channels, channels], 2),
        ),
        (
            "adaLN_modulation.1.bias",
            const_tensor(&[6 * channels], 0.2),
        ),
        (
            "input_layer.weight",
            pattern_tensor(&[channels, in_channels], 3),
        ),
        ("input_layer.bias", const_tensor(&[channels], 0.0)),
        (
            "out_layer.weight",
            pattern_tensor(&[in_channels, channels], 4),
        ),
        ("out_layer.bias", const_tensor(&[in_channels], 0.0)),
        ("blocks.0.norm2.weight", const_tensor(&[channels], 1.0)),
        ("blocks.0.norm2.bias", const_tensor(&[channels], 0.0)),
        ("blocks.0.modulation", pattern_tensor(&[6 * channels], 5)),
        (
            "blocks.0.self_attn.to_qkv.weight",
            pattern_tensor(&[3 * channels, channels], 6),
        ),
        (
            "blocks.0.self_attn.to_qkv.bias",
            const_tensor(&[3 * channels], 0.0),
        ),
        (
            "blocks.0.self_attn.to_out.weight",
            pattern_tensor(&[channels, channels], 7),
        ),
        (
            "blocks.0.self_attn.to_out.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.self_attn.q_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.self_attn.k_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.cross_attn.to_q.weight",
            pattern_tensor(&[channels, channels], 8),
        ),
        (
            "blocks.0.cross_attn.to_q.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.to_kv.weight",
            pattern_tensor(&[2 * channels, cond_channels], 9),
        ),
        (
            "blocks.0.cross_attn.to_kv.bias",
            const_tensor(&[2 * channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.to_out.weight",
            pattern_tensor(&[channels, channels], 10),
        ),
        (
            "blocks.0.cross_attn.to_out.bias",
            const_tensor(&[channels], 0.0),
        ),
        (
            "blocks.0.cross_attn.q_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.cross_attn.k_rms_norm.gamma",
            const_tensor(&[heads, head_dim], 1.0),
        ),
        (
            "blocks.0.mlp.mlp.0.weight",
            pattern_tensor(&[mlp_hidden, channels], 11),
        ),
        ("blocks.0.mlp.mlp.0.bias", const_tensor(&[mlp_hidden], 0.0)),
        (
            "blocks.0.mlp.mlp.2.weight",
            pattern_tensor(&[channels, mlp_hidden], 12),
        ),
        ("blocks.0.mlp.mlp.2.bias", const_tensor(&[channels], 0.0)),
    ];
    let vb = tensor_map(entries.clone());
    // A prefix without the tensors must fail the loader (strict coverage).
    assert!(Trellis2SlatDit::load(vb.pp("missing"), &config).is_err());
    let vb = tensor_map(entries);
    let dit = Trellis2SlatDit::load(vb, &config).expect("dit load");

    // Deterministic sparse coordinate set on the 8^3 grid (with holes).
    let coords: Vec<[i32; 3]> = (0..resolution)
        .flat_map(|x| {
            (0..resolution).flat_map(move |y| {
                (0..resolution)
                    .filter(move |&z| (x * 64 + y * 8 + z) % 5 != 0)
                    .map(move |z| [x as i32, y as i32, z as i32])
            })
        })
        .collect();
    let tokens = coords.len();
    assert!(tokens > 300);
    let feats = pattern_tensor(&[tokens, in_channels], 13);
    let cond = pattern_tensor(&[1, 5, cond_channels], 14);
    let out = dit.forward(&feats, &coords, 500.0, &cond).expect("forward");
    assert_eq!(out.dims(), &[tokens, 4]);
    let values = out
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("out");
    assert!(values.iter().all(|v| v.is_finite()));

    // RoPE tables follow the token coords: [N, head_dim/2].
    let (cos, sin) = dit
        .rope_tables_for(&coords, &Device::Cpu)
        .expect("rope tables");
    assert_eq!(cos.dims(), &[tokens, head_dim / 2]);
    assert_eq!(sin.dims(), &[tokens, head_dim / 2]);
    // Row 0 (coord (0, 0, 1)) must differ from a (0, 0, 0)-token table.
    let coords0: Vec<[i32; 3]> = vec![[0, 0, 0]];
    let (cos0, _) = dit.rope_tables_for(&coords0, &Device::Cpu).expect("rope");
    let cos_row = cos
        .narrow(0, 0, 1)
        .expect("row")
        .squeeze(0)
        .expect("squeeze");
    let cos0 = cos0.squeeze(0).expect("squeeze");
    let diff = cos_row
        .sub(&cos0)
        .expect("sub")
        .abs()
        .expect("abs")
        .sum_all()
        .expect("sum")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(diff > 0.0, "RoPE phases must depend on the coordinates");

    // Invalid inputs must be rejected.
    let bad_coords: Vec<[i32; 3]> = vec![[0, 0, resolution as i32]];
    assert!(
        dit.forward(
            &pattern_tensor(&[1, in_channels], 15),
            &bad_coords,
            1.0,
            &cond
        )
        .is_err()
    );
    let neg_coords: Vec<[i32; 3]> = vec![[-1, 0, 0]];
    assert!(
        dit.forward(
            &pattern_tensor(&[1, in_channels], 16),
            &neg_coords,
            1.0,
            &cond
        )
        .is_err()
    );
    assert!(
        dit.forward(
            &pattern_tensor(&[tokens + 1, in_channels], 17),
            &coords,
            1.0,
            &cond
        )
        .is_err()
    );
}

fn read_safetensors_stage_keys(path: &Path, prefix: &str) -> Option<Vec<String>> {
    let file = std::fs::read(path).ok()?;
    if file.len() < 8 {
        return None;
    }
    let header_len = u64::from_le_bytes([
        file[0], file[1], file[2], file[3], file[4], file[5], file[6], file[7],
    ]) as usize;
    if file.len() < 8 + header_len {
        return None;
    }
    let header: serde_json::Value = serde_json::from_slice(&file[8..8 + header_len]).ok()?;
    Some(
        header
            .as_object()?
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect(),
    )
}

/// Real weights + golden oracle: strict loader coverage over the canonical
/// package stage, then the 4-step sampler from the golden cond/coords/noise
/// compared per-step against the bf16 CUDA reference.
#[test]
fn slat_dit_real_weights_load_and_golden_sample() {
    let golden = golden_dir();
    let npz_path = golden.join("slat_golden.npz");
    let model_file = canonical_model_file();
    if !npz_path.is_file() {
        eprintln!("SKIP: golden npz missing at {}", npz_path.display());
        return;
    }
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }

    // ---- Load the golden arrays (npz = zip of .npy) ----
    let file = std::fs::File::open(&npz_path).expect("open npz");
    let mut npz = ndarray_npy::NpzReader::new(file).expect("npz reader");
    let cond_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("cond")
        .expect("cond");
    let coords_array = npz
        .by_name::<ndarray::OwnedRepr<i32>, ndarray::Ix2>("coords_32")
        .expect("coords_32");
    let noise_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("shape_noise")
        .expect("shape_noise");
    let x_t_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("shape_x_t_steps")
        .expect("shape_x_t_steps");
    let x0_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>("shape_pred_x0_steps")
        .expect("shape_pred_x0_steps");
    let norm_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("shape_slat_norm")
        .expect("shape_slat_norm");
    let denorm_array = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("shape_slat_denorm")
        .expect("shape_slat_denorm");
    assert_eq!(cond_array.shape(), &[1, 1029, 1024]);
    assert_eq!(coords_array.shape(), &[3591, 4]);
    assert_eq!(noise_array.shape(), &[3591, 32]);
    assert_eq!(x_t_array.shape(), &[4, 3591, 32]);
    let tokens = 3591usize;

    let coords: Vec<[i32; 3]> = (0..tokens)
        .map(|i| {
            let b = coords_array[[i, 0]];
            assert_eq!(b, 0, "single batch element");
            [
                coords_array[[i, 1]],
                coords_array[[i, 2]],
                coords_array[[i, 3]],
            ]
        })
        .collect();

    // ---- Stage tensor coverage from the canonical package header ----
    let stage_keys = read_safetensors_stage_keys(&model_file, "shape_dit_512.")
        .expect("canonical package header");
    assert_eq!(
        stage_keys.len(),
        640,
        "shape_dit_512 stage tensor count in the canonical package"
    );
    let config = Trellis2SlatConfig::shape_512();
    let expected: HashSet<String> = config.expected_tensor_names().into_iter().collect();
    let stage: HashSet<String> = stage_keys
        .iter()
        .map(|k| k.trim_start_matches("shape_dit_512.").to_string())
        .collect();
    assert_eq!(
        expected, stage,
        "expected names must cover exactly the stage tensors"
    );

    // ---- Strict real-weight load ----
    let device = Device::Cpu;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[model_file.clone()], DType::F32, &device)
    }
    .expect("mmaped canonical weights");
    let load_started = std::time::Instant::now();
    let dit = Trellis2SlatDit::load(vb.pp("shape_dit_512"), &config).expect("shape_dit_512 load");
    println!(
        "shape_dit_512 loaded in {:.1}s",
        load_started.elapsed().as_secs_f64()
    );

    let cond = Tensor::from_vec(
        cond_array.iter().copied().collect(),
        (1, 1029, 1024),
        &device,
    )
    .expect("cond tensor");
    let noise = Tensor::from_vec(noise_array.iter().copied().collect(), (tokens, 32), &device)
        .expect("noise");

    // Golden settings: steps 4, guidance_strength 1.0 (single cond pass),
    // guidance_rescale 0.0, interval [0.6, 1.0], rescale_t 3.0, sigma_min 1e-5.
    let params = Trellis2SamplerParams {
        steps: 4,
        sigma_min: 1e-5,
        guidance_strength: 1.0,
        guidance_rescale: 0.0,
        guidance_interval: [0.6, 1.0],
        rescale_t: 3.0,
    };
    let forward_started = std::time::Instant::now();
    let model = |x: &Tensor, t_model: f32, c: &Tensor, coords: &[[i32; 3]]| {
        dit.forward(x, coords, t_model, c)
    };
    let (samples, trace) =
        sample_sparse(&model, &coords, &noise, &cond, &cond, &params).expect("sample");
    println!(
        "4-step sparse sampler forward in {:.1}s",
        forward_started.elapsed().as_secs_f64()
    );
    assert_eq!(samples.dims(), &[tokens, 32]);

    // Schedule: golden t_pairs ((1.0, 0.9), (0.9, 0.75), (0.75, 0.5), (0.5, 0)).
    let golden_pairs = [(1.0, 0.9), (0.9, 0.75), (0.75, 0.5), (0.5, 0.0)];
    assert_eq!(trace.t_pairs.len(), 4);
    for (step, (t, t_prev)) in golden_pairs.iter().enumerate() {
        let (got_t, got_prev) = trace.t_pairs[step];
        assert!(
            (got_t - t).abs() < 1e-9 && (got_prev - t_prev).abs() < 1e-9,
            "t pair {step}: got ({got_t}, {got_prev}), want ({t}, {t_prev})"
        );
    }

    let compare = |actual: &[f32], expected: &[f32], label: &str| {
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
        (rel_vs_max, rms_rel)
    };

    // Tolerances reflect the bf16 CUDA reference vs F32 compute (same
    // situation as the SS stage golden test: the reference velocity carries
    // ~3% relative bf16 noise; the Euler update averages it over dt and the
    // 30-block trajectory amplifies accumulated drift, so the last-step and
    // final caps are much looser than the early per-step caps).
    let x_t_cap = [2e-2_f64, 2e-2, 2e-2, 1.5e-1];
    let x0_cap = [8e-2_f64, 8e-2, 8e-2, 3e-1];
    for step in 0..4 {
        let actual = trace.x_t_steps[step]
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("x_t");
        let golden: Vec<f32> = x_t_array
            .index_axis(ndarray::Axis(0), step)
            .iter()
            .copied()
            .collect();
        let (rel, rms) = compare(&actual, &golden, &format!("x_t step {step}"));
        assert!(rel <= x_t_cap[step], "x_t step {step} rel {rel:.3e}");
        assert!(rms <= x_t_cap[step], "x_t step {step} rms_rel {rms:.3e}");

        let actual_x0 = trace.pred_x0_steps[step]
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("x0");
        let golden_x0: Vec<f32> = x0_array
            .index_axis(ndarray::Axis(0), step)
            .iter()
            .copied()
            .collect();
        let (rel, rms) = compare(&actual_x0, &golden_x0, &format!("pred_x0 step {step}"));
        assert!(rel <= x0_cap[step], "pred_x0 step {step} rel {rel:.3e}");
        assert!(rms <= x0_cap[step], "pred_x0 step {step} rms_rel {rms:.3e}");
    }

    // Final normalized SLAT: the last x_t must equal the saved output.
    let actual_norm = samples
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("slat_norm");
    let golden_norm: Vec<f32> = norm_array.iter().copied().collect();
    let (norm_rel, norm_rms) = compare(&actual_norm, &golden_norm, "shape_slat_norm");
    assert!(norm_rel <= 1.5e-1, "slat_norm rel {norm_rel:.3e}");
    assert!(norm_rms <= 6e-2, "slat_norm rms_rel {norm_rms:.3e}");

    // Denorm check: slat_norm * std + mean against the saved denorm (the
    // normalization vectors come from the manifest, pipeline.json source).
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("manifest.json")).expect("manifest"))
            .expect("manifest json");
    let mean: Vec<f32> = manifest["shape_normalization"]["mean"]
        .as_array()
        .expect("mean")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    let std: Vec<f32> = manifest["shape_normalization"]["std"]
        .as_array()
        .expect("std")
        .iter()
        .map(|v| v.as_f64().expect("f64") as f32)
        .collect();
    assert_eq!(mean.len(), 32);
    let mut denorm = vec![0.0_f32; actual_norm.len()];
    for row in 0..tokens {
        for c in 0..32 {
            denorm[row * 32 + c] = actual_norm[row * 32 + c] * std[c] + mean[c];
        }
    }
    let golden_denorm: Vec<f32> = denorm_array.iter().copied().collect();
    let (denorm_rel, denorm_rms) = compare(&denorm, &golden_denorm, "shape_slat_denorm");
    assert!(denorm_rel <= 2e-1, "slat_denorm rel {denorm_rel:.3e}");
    assert!(denorm_rms <= 1e-1, "slat_denorm rms_rel {denorm_rms:.3e}");
}
