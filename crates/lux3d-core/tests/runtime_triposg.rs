//! TripoSG DINOv2-large image-encoder parity against the python-baseline
//! smoke sample (CPU only for this task; T6 adds CUDA/Vulkan/wgpu runs),
//! plus Task 3b: rectified-flow scheduler unit tests and the DiT
//! velocity / denoise CUDA parity cases.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::ModelFamily;
use lux3d_core::preprocess::TripoSgPreprocessStage;
use lux3d_core::runtime::{
    RectifiedFlowSchedule, TripoSgDiT, TripoSgDinoEncoder, TripoSgTransformerConfig, denoise,
};
use lux3d_core::test_support::{GpuTestLock, model_asset_options, runtime_root};
use lux3d_core::weights::load_canonical_weights;

fn repo_root() -> PathBuf {
    runtime_root().join("candle-3d")
}

fn smoke_baseline_path() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg")
        .join("smoke")
        .join("artifacts.safetensors")
}

fn golden_baseline_path() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg")
        .join("mesh-input-golden")
        .join("artifacts.safetensors")
}

/// Golden parity on CPU: run the TripoSG preprocess stage on the smoke sample
/// image, encode it with the canonical DINOv2-large weights, and compare
/// against the baseline `triposg.image_embeds` ([1, 257, 1024] F32). Tolerance
/// per plan: max abs delta <= 2e-3.
#[test]
fn triposg_dino_image_embeds_match_smoke_baseline() {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );

    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("failed to load the TripoSG canonical weight package");
    // SAFETY: the canonical safetensors file is read-only and the mmap stays
    // alive through the VarBuilder's 'static borrow for this test's duration.
    let vb = unsafe { weights.var_builder(DType::F32, &Device::Cpu) }
        .expect("failed to open TripoSG canonical weights as VarBuilder");
    let encoder =
        TripoSgDinoEncoder::load(vb.pp("image_encoder")).expect("TripoSG DINOv2 encoder load");

    let stage = TripoSgPreprocessStage::default();
    let prepared = stage
        .prepare_inputs_from_path(
            &repo_root().join("test-assets").join("mesh-input.png"),
            &Device::Cpu,
        )
        .expect("TripoSG prepared inputs");
    assert_eq!(
        prepared.pixel_values.dims(),
        [1, 3, 224, 224],
        "preprocess must feed [1,3,224,224] pixel values"
    );

    let actual = encoder
        .forward(&prepared.pixel_values)
        .expect("TripoSG DINOv2 forward");
    let actual_dims = actual.dims().to_vec();
    assert_eq!(
        actual.dims(),
        [1, 257, 1024],
        "last_hidden_state must be [1,257,1024]"
    );

    let expected = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("smoke baseline safetensors")
    }
    .load("triposg.image_embeds", &Device::Cpu)
    .expect("baseline triposg.image_embeds");
    assert_eq!(
        expected.dims(),
        actual.dims(),
        "baseline image_embeds shape mismatch"
    );

    let expected = expected
        .to_dtype(DType::F32)
        .expect("baseline f32")
        .flatten_all()
        .expect("baseline flatten")
        .to_vec1::<f32>()
        .expect("baseline vec");
    let actual = actual
        .flatten_all()
        .expect("actual flatten")
        .to_vec1::<f32>()
        .expect("actual vec");

    let mut max_delta = 0.0_f32;
    let mut sum_delta = 0.0_f64;
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        let delta = (expected - actual).abs();
        max_delta = max_delta.max(delta);
        sum_delta += f64::from(delta);
    }
    let mean_delta = (sum_delta / expected.len() as f64) as f32;
    println!(
        "pixel_values shape: {:?}, image_embeds shape: {actual_dims:?}",
        prepared.pixel_values.dims()
    );
    println!("image_embeds parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e}");
    assert!(
        max_delta <= 2e-3,
        "image_embeds max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
}

// ---------------------------------------------------------------------------
// Task 3b: RectifiedFlow scheduler unit tests (CPU, no weights)
// ---------------------------------------------------------------------------

/// N=50, shift=1: sigmas [1.0, 0.98, ..., 0.02] + terminal 0.0, uniform 0.02
/// spacing; timesteps are sigma * 1000.
#[test]
fn rectified_flow_schedule_n50_matches_reference() {
    let schedule = RectifiedFlowSchedule::new(50, 1.0);
    assert_eq!(
        schedule.sigmas.len(),
        51,
        "sigmas must carry the terminal 0"
    );
    assert_eq!(schedule.timesteps().len(), 50);

    assert!((schedule.sigmas[0] - 1.0).abs() < 1e-6, "first sigma");
    assert!(
        (schedule.sigmas[49] - 0.02).abs() < 1e-6,
        "last non-terminal sigma {}",
        schedule.sigmas[49]
    );
    assert_eq!(schedule.sigmas[50], 0.0, "terminal sigma");

    for i in 0..50 {
        let spacing = schedule.sigmas[i] - schedule.sigmas[i + 1];
        assert!(
            (spacing - 0.02).abs() < 1e-6,
            "uniform spacing violated at {i}: {spacing}"
        );
        let expected_timestep = (1.0 - i as f64 / 50.0) * 1000.0;
        assert!(
            (schedule.timestep(i) - expected_timestep as f32).abs() < 1e-3,
            "timestep mismatch at {i}: {} vs {expected_timestep}",
            schedule.timestep(i)
        );
    }
}

/// N=2, shift=1: sigmas exactly [1.0, 0.5, 0.0], timesteps [1000, 500].
#[test]
fn rectified_flow_schedule_n2_matches_reference() {
    let schedule = RectifiedFlowSchedule::new(2, 1.0);
    assert_eq!(schedule.sigmas, vec![1.0, 0.5, 0.0]);
    let timesteps = schedule.timesteps();
    assert_eq!(timesteps.len(), 2);
    assert!((timesteps[0] - 1000.0).abs() < 1e-4);
    assert!((timesteps[1] - 500.0).abs() < 1e-4);
}

/// Euler update at index 0 for N=2 must be `x + (1.0 - 0.5) * v = x + 0.5*v`.
#[test]
fn rectified_flow_step_is_euler_update() {
    let schedule = RectifiedFlowSchedule::new(2, 1.0);
    let velocity = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 0.25], (1, 2, 2), &Device::Cpu)
        .expect("velocity tensor");
    let sample = Tensor::from_vec(vec![0.5f32, 0.25, -0.75, 2.0], (1, 2, 2), &Device::Cpu)
        .expect("sample tensor");

    let updated = schedule
        .step(0, &velocity, &sample)
        .expect("scheduler step");
    let expected =
        (&sample + velocity.affine(0.5, 0.0).expect("scaled velocity")).expect("expected update");

    let updated = updated
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let expected = expected
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    for (i, (a, b)) in updated.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "euler update mismatch at {i}: {a} vs {b}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 3b: DiT velocity / scheduler step / 2-step CFG denoise CUDA parity
// against the smoke baseline (skips with a clear message when CUDA or the
// baseline artifact is unavailable).
// ---------------------------------------------------------------------------

fn load_baseline_tensor(path: &PathBuf, name: &str, device: &Device) -> Tensor {
    let tensor = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(path)
            .unwrap_or_else(|error| panic!("smoke baseline safetensors: {error}"))
    }
    .load(name, device)
    .unwrap_or_else(|error| panic!("baseline {name}: {error}"));
    tensor.to_dtype(DType::F32).expect("baseline f32")
}

fn max_abs_delta_stats(expected: &Tensor, actual: &Tensor) -> (f32, f32) {
    let expected = expected
        .to_dtype(DType::F32)
        .expect("expected f32")
        .flatten_all()
        .expect("expected flatten")
        .to_vec1::<f32>()
        .expect("expected vec");
    let actual = actual
        .to_dtype(DType::F32)
        .expect("actual f32")
        .flatten_all()
        .expect("actual flatten")
        .to_vec1::<f32>()
        .expect("actual vec");
    assert_eq!(
        expected.len(),
        actual.len(),
        "tensor element count mismatch"
    );
    let mut max_delta = 0.0_f32;
    let mut sum_delta = 0.0_f64;
    for (expected, actual) in expected.iter().zip(actual.iter()) {
        let delta = (expected - actual).abs();
        max_delta = max_delta.max(delta);
        sum_delta += f64::from(delta);
    }
    let mean_delta = (sum_delta / expected.len() as f64) as f32;
    (max_delta, mean_delta)
}

#[test]
fn triposg_dit_velocity_and_denoise_match_smoke_baseline_cuda() {
    let baseline_path = smoke_baseline_path();
    if !baseline_path.is_file() {
        eprintln!(
            "SKIP TripoSG DiT CUDA parity: smoke baseline artifact is missing at {} \
             (generate it with tools/python_baseline/triposg_dump.py)",
            baseline_path.display()
        );
        return;
    }
    let _guard = match GpuTestLock::acquire() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("SKIP TripoSG DiT CUDA parity: GPU test lock unavailable: {error}");
            return;
        }
    };
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("SKIP TripoSG DiT CUDA parity: no CUDA device available ({error})");
            return;
        }
    };

    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("failed to load the TripoSG canonical weight package");
    let resolved = weights
        .read_resolved_config_json()
        .expect("resolved_config.json");
    let cfg = TripoSgTransformerConfig::from_resolved_config(&resolved)
        .expect("TripoSG transformer config");
    assert_eq!(cfg.num_layers, 21);
    assert_eq!(cfg.width, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.in_channels, 64);
    assert_eq!(cfg.cross_attention_dim, 1024);

    // Load only the `transformer.` subtree onto the device (the canonical file
    // also holds the DINOv2 encoder and the VAE; keeping those off the GPU
    // leaves headroom under the 12 GiB budget).
    let archive = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&weights.canonical_file)
            .expect("canonical weights safetensors")
    };
    let mut transformer_tensors = std::collections::HashMap::new();
    for (name, _view) in archive.tensors() {
        if name.starts_with("transformer.") {
            let tensor = archive
                .load(&name, &device)
                .unwrap_or_else(|error| panic!("load {name}: {error}"))
                .to_dtype(DType::F32)
                .expect("transformer weights f32");
            transformer_tensors.insert(name, tensor);
        }
    }
    assert_eq!(
        transformer_tensors.len(),
        554,
        "canonical transformer tensor count"
    );
    let vb = candle_nn::VarBuilder::from_tensors(transformer_tensors, DType::F32, &device);
    let dit = TripoSgDiT::load(vb.pp("transformer"), &cfg).expect("TripoSG DiT load");

    // Baseline taps (F32): [1,2048,64] latents, [1,257,1024] image embeds.
    let latents_init = load_baseline_tensor(&baseline_path, "triposg.latents_init", &device);
    let image_embeds = load_baseline_tensor(&baseline_path, "triposg.image_embeds", &device);
    assert_eq!(latents_init.dims(), [1, 2048, 64]);
    assert_eq!(image_embeds.dims(), [1, 257, 1024]);

    // --- Phase A: step-0 CFG velocity --------------------------------------
    let cond_batch = Tensor::cat(
        &[
            &Tensor::zeros_like(&image_embeds).expect("zeros"),
            &image_embeds,
        ],
        0,
    )
    .expect("cond batch");
    let latent_batch = Tensor::cat(&[&latents_init, &latents_init], 0).expect("latent batch");
    let velocity = dit
        .forward(&latent_batch, 1000.0, &cond_batch)
        .expect("DiT forward at t=1000");
    assert_eq!(velocity.dims(), [2, 2048, 64]);
    let parts = velocity.chunk(2, 0).expect("velocity chunk");
    let velocity_combined = (&parts[0]
        + (&parts[1] - &parts[0])
            .expect("sub")
            .affine(7.0, 0.0)
            .expect("scaled delta"))
    .expect("cfg combine");
    let expected_velocity =
        load_baseline_tensor(&baseline_path, "triposg.velocity_step_000", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_velocity, &velocity_combined);
    println!(
        "velocity_step_000 parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 5e-3)"
    );
    assert!(
        max_delta <= 5e-3,
        "velocity_step_000 max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
    drop(latent_batch);
    drop(cond_batch);
    drop(velocity);

    // --- Phase B: scheduler step 0 -> latents_step_001 ---------------------
    let schedule = RectifiedFlowSchedule::new(2, 1.0);
    let latents_step_001 = schedule
        .step(0, &velocity_combined, &latents_init)
        .expect("scheduler step 0");
    let expected_step_001 =
        load_baseline_tensor(&baseline_path, "triposg.latents_step_001", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_step_001, &latents_step_001);
    println!(
        "latents_step_001 parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 2e-2)"
    );
    assert!(
        max_delta <= 2e-2,
        "latents_step_001 max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
    drop(latents_step_001);
    drop(velocity_combined);

    // --- Phase C: full 2-step CFG denoise ----------------------------------
    let latents_final =
        denoise(&dit, &latents_init, &image_embeds, 2, 7.0).expect("2-step CFG denoise");
    assert_eq!(latents_final.dims(), [1, 2048, 64]);
    let expected_final =
        load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_final, &latents_final);
    println!(
        "latents_step_final parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 5e-2)"
    );
    assert!(
        max_delta <= 5e-2,
        "latents_step_final max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
}

// ---------------------------------------------------------------------------
// Task 3c: VAE SDF decoder + hierarchical dense-grid CPU parity against the
// smoke baseline.
// ---------------------------------------------------------------------------

use std::time::Instant;

use lux3d_core::runtime::{
    TripoSgSdfGrid, TripoSgVaeConfig, TripoSgVaeDecoder, hierarchical_sdf_grid,
};

/// Canonical bounds used by the reference hierarchical extraction.
fn smoke_bounds() -> [f32; 6] {
    [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005]
}

/// Load the TripoSG VAE decoder (post_quant + decoder subtrees only) on CPU.
fn load_triposg_vae_decoder_cpu() -> (TripoSgVaeDecoder, TripoSgVaeConfig) {
    load_triposg_vae_decoder_on(&Device::Cpu)
}

/// Load the TripoSG VAE decoder (post_quant + decoder subtrees only) on the
/// given device. Only the decode-path keys are loaded; the canonical file
/// also carries the (unused here) VAE encoder + quant weights.
fn load_triposg_vae_decoder_on(device: &Device) -> (TripoSgVaeDecoder, TripoSgVaeConfig) {
    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("failed to load the TripoSG canonical weight package");
    let resolved = weights
        .read_resolved_config_json()
        .expect("resolved_config.json");
    let cfg = TripoSgVaeConfig::from_resolved_config(&resolved).expect("TripoSG VAE config");
    assert_eq!(cfg.latent_channels, 64);
    assert_eq!(cfg.dim, 1024);
    assert_eq!(cfg.num_attention_heads, 8);
    assert_eq!(cfg.num_layers_decoder, 16);
    assert_eq!(cfg.embed_frequency, 8);
    assert!(!cfg.embed_include_pi);

    let archive = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&weights.canonical_file)
            .expect("canonical weights safetensors")
    };
    let mut vae_tensors = std::collections::HashMap::new();
    for (name, _view) in archive.tensors() {
        if name.starts_with("vae.post_quant.") || name.starts_with("vae.decoder.") {
            let tensor = archive
                .load(&name, device)
                .unwrap_or_else(|error| panic!("load {name}: {error}"))
                .to_dtype(DType::F32)
                .expect("vae weights f32");
            vae_tensors.insert(name, tensor);
        }
    }
    assert!(
        vae_tensors.len() >= 200,
        "unexpected TripoSG VAE decode-path tensor count {}",
        vae_tensors.len()
    );
    let vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, device);
    let decoder = TripoSgVaeDecoder::load(vb.pp("vae"), &cfg).expect("TripoSG VAE decoder load");
    (decoder, cfg)
}

/// Golden CPU parity: `decode_latents(latents_step_final, sdf_points)` must
/// match the baseline `triposg.sdf_values` within 5e-3 (max abs).
#[test]
fn triposg_vae_sdf_decode_matches_smoke_baseline_cpu() {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );

    let (decoder, _cfg) = load_triposg_vae_decoder_cpu();
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    assert_eq!(latents.dims(), [1, 2048, 64]);
    let points = load_baseline_tensor(&baseline_path, "triposg.sdf_points", &Device::Cpu);
    assert_eq!(points.dims(), [8192, 3]);

    let started = Instant::now();
    let actual = decoder
        .decode_latents(&latents, &points, 16384)
        .expect("TripoSG VAE decode_latents");
    let elapsed = started.elapsed();
    assert_eq!(actual.dims(), [8192], "decode_points must return [N]");
    let expected = load_baseline_tensor(&baseline_path, "triposg.sdf_values", &Device::Cpu);

    let (max_delta, mean_delta) = max_abs_delta_stats(&expected, &actual);
    println!(
        "sdf_values parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} \
         (tolerance 5e-3), decode took {elapsed:?}"
    );
    assert!(
        max_delta <= 5e-3,
        "sdf_values max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
}

/// Golden CPU parity: the dense octree grid (depth 5, no refinement) must
/// match the baseline `triposg.dense_sdf` within 1e-2 (max abs). The
/// baseline stores raw F32 logits in (x, y, z) C-order (x slowest, z
/// fastest) from the reference meshgrid "ij"; our grid packs x-fastest.
#[test]
fn triposg_dense_sdf_grid_matches_smoke_baseline_cpu() {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );

    let (decoder, _cfg) = load_triposg_vae_decoder_cpu();
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);

    let started = Instant::now();
    let grid = hierarchical_sdf_grid(&decoder, &latents, smoke_bounds(), 5, 5, 16384)
        .expect("TripoSG dense SDF grid");
    let elapsed = started.elapsed();
    assert_eq!(grid.depth, 5, "hier == dense must return the dense grid");
    assert_eq!(grid.dims, 32);
    assert_eq!(grid.values.len(), 32 * 32 * 32);

    let expected = load_baseline_tensor(&baseline_path, "triposg.dense_sdf", &Device::Cpu)
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(expected.len(), grid.values.len());

    // ours[x + y*n + z*n*n] <-> baseline[x*n*n + y*n + z]
    let n = 32usize;
    let mut max_delta = 0.0_f32;
    let mut sum_delta = 0.0_f64;
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let actual = grid.values[x + y * n + z * n * n];
                let expected = expected[x * n * n + y * n + z];
                let delta = (expected - actual).abs();
                max_delta = max_delta.max(delta);
                sum_delta += f64::from(delta);
            }
        }
    }
    let mean_delta = (sum_delta / expected.len() as f64) as f32;
    println!(
        "dense_sdf parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} \
         (tolerance 1e-2), grid took {elapsed:?}"
    );
    assert!(
        max_delta <= 1e-2,
        "dense_sdf max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
}

// ---------------------------------------------------------------------------
// Task 6: golden-baseline parity (50 steps, dense 8 / hierarchical 9, F32).
// The golden artifacts are generated by the orchestrator (CUDA F32, Colab or
// local GPU); every golden case prints an explicit SKIP line and returns
// while they are absent, so the suite stays green before the baseline lands.
//
// Wall-time gates for the heavy grid cases: the depth-8 dense pass alone is
// 16.7M VAE queries; the measured dev-profile marginal decode cost is
// ~0.55 ms/query (task-3c), i.e. ~2-3 h on CPU and ~10-20 min on CUDA. The
// heavy cases below therefore additionally require LUX3D_TRIPOSG_GOLDEN_RUN=1
// and pick the device with LUX3D_TRIPOSG_GOLDEN_DEVICE (cpu|cuda, default
// cpu). The CUDA parity case runs whenever the golden artifact and a CUDA
// device exist (its wall time is minutes).
// ---------------------------------------------------------------------------

fn golden_env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| value == "1")
        .unwrap_or(false)
}

/// Device plus optional GPU lock for the heavy golden grid cases:
/// `LUX3D_TRIPOSG_GOLDEN_DEVICE` (`cuda` -> locked CUDA device, anything
/// else / unset -> CPU). The lock is acquired before creating the CUDA
/// device so concurrent test processes cannot race on GPU memory.
///
/// `None` means "skip this test"; `Some((device, guard))` keeps the lock
/// alive through the test body; `Some((Cpu, None))` needs no lock.
fn golden_grid_device_and_lock(label: &str) -> Option<(Device, Option<GpuTestLock>)> {
    let requested = std::env::var("LUX3D_TRIPOSG_GOLDEN_DEVICE").unwrap_or_default();
    if !requested.eq_ignore_ascii_case("cuda") {
        return Some((Device::Cpu, None));
    }
    let guard = match GpuTestLock::acquire() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("SKIP {label}: GPU test lock unavailable: {error}");
            return None;
        }
    };
    match Device::new_cuda(0) {
        Ok(device) => Some((device, Some(guard))),
        Err(error) => {
            eprintln!(
                "SKIP {label}: LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda but no CUDA device ({error})"
            );
            None
        }
    }
}

fn load_triposg_dino_encoder_on(device: &Device) -> TripoSgDinoEncoder {
    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("failed to load the TripoSG canonical weight package");
    let archive = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&weights.canonical_file)
            .expect("canonical weights safetensors")
    };
    let mut encoder_tensors = std::collections::HashMap::new();
    for (name, _view) in archive.tensors() {
        if name.starts_with("image_encoder.") {
            let tensor = archive
                .load(&name, device)
                .unwrap_or_else(|error| panic!("load {name}: {error}"))
                .to_dtype(DType::F32)
                .expect("image encoder weights f32");
            encoder_tensors.insert(name, tensor);
        }
    }
    assert!(
        encoder_tensors.len() >= 300,
        "unexpected TripoSG image-encoder tensor count {}",
        encoder_tensors.len()
    );
    let vb = candle_nn::VarBuilder::from_tensors(encoder_tensors, DType::F32, device);
    TripoSgDinoEncoder::load(vb.pp("image_encoder")).expect("TripoSG DINOv2 encoder load")
}

/// Load the TripoSG DiT (transformer subtree only) plus its resolved config.
fn load_triposg_dit_on(device: &Device) -> TripoSgDiT {
    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("failed to load the TripoSG canonical weight package");
    let resolved = weights
        .read_resolved_config_json()
        .expect("resolved_config.json");
    let cfg = TripoSgTransformerConfig::from_resolved_config(&resolved)
        .expect("TripoSG transformer config");
    assert_eq!(cfg.num_layers, 21);
    assert_eq!(cfg.width, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.in_channels, 64);
    assert_eq!(cfg.cross_attention_dim, 1024);

    // Only the `transformer.` subtree goes onto the device (the canonical
    // file also holds the DINOv2 encoder and the VAE; keeping those off the
    // GPU leaves headroom under the 12 GiB budget).
    let archive = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&weights.canonical_file)
            .expect("canonical weights safetensors")
    };
    let mut transformer_tensors = std::collections::HashMap::new();
    for (name, _view) in archive.tensors() {
        if name.starts_with("transformer.") {
            let tensor = archive
                .load(&name, device)
                .unwrap_or_else(|error| panic!("load {name}: {error}"))
                .to_dtype(DType::F32)
                .expect("transformer weights f32");
            transformer_tensors.insert(name, tensor);
        }
    }
    assert_eq!(
        transformer_tensors.len(),
        554,
        "canonical transformer tensor count"
    );
    let vb = candle_nn::VarBuilder::from_tensors(transformer_tensors, DType::F32, device);
    TripoSgDiT::load(vb.pp("transformer"), &cfg).expect("TripoSG DiT load")
}

/// Golden CUDA parity: preprocess (+ uint8/pixel_values) + DINOv2 image
/// embeds, step-0 CFG velocity, step-1/step-10/step-25 snapshots, the FULL
/// 50-step CFG denoise from the baseline `latents_init` vs
/// `latents_step_final`, and the SDF probe values.
///
/// Tolerances: pixel_values 1e-5, image_embeds 2e-3, velocity 5e-3, step-1
/// max 1e-2, step-10 max 5e-2, step-25 max 5e-2. The 50-step CFG ODE is chaotic, so the final
/// latent is statistical (mean <= 2e-2, relative L2 <= 1.2e-1, max <= 4.0)
/// and the end-to-end SDF is statistical (mean <= 1.5e-1, max <= 1.0e1); the
/// VAE path itself is pinned by the baseline-latents decode assertion.
///
/// Skips (without failing) when the golden artifact is missing, the GPU test
/// lock is held elsewhere, or no CUDA device exists.
#[test]
fn triposg_golden_pipeline_and_sdf_match_cuda() {
    let label = "TripoSG golden CUDA parity";
    let baseline_path = golden_baseline_path();
    if !baseline_path.is_file() {
        eprintln!(
            "SKIP {label}: golden artifact is missing at {} (orchestrator golden run pending)",
            baseline_path.display()
        );
        return;
    }
    let _guard = match GpuTestLock::acquire() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("SKIP {label}: GPU test lock unavailable: {error}");
            return;
        }
    };
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("SKIP {label}: no CUDA device available ({error})");
            return;
        }
    };

    // --- Phase A: preprocess + DINOv2 conditioning --------------------------
    let encoder = load_triposg_dino_encoder_on(&device);
    let stage = TripoSgPreprocessStage::default();
    let prepared = stage
        .prepare_inputs_from_path(
            &repo_root().join("test-assets").join("mesh-input.png"),
            &device,
        )
        .expect("TripoSG prepared inputs");

    let expected_uint8 =
        load_baseline_tensor(&baseline_path, "triposg.preprocessed_uint8", &Device::Cpu)
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
    assert_eq!(
        expected_uint8.len(),
        prepared.uint8_rgb.len(),
        "preprocessed_uint8 element count mismatch"
    );
    for (index, (expected, actual)) in expected_uint8
        .iter()
        .zip(prepared.uint8_rgb.iter())
        .enumerate()
    {
        assert_eq!(
            *expected as u8, *actual,
            "preprocessed_uint8 must match exactly at {index}"
        );
    }

    let expected_pixel_values =
        load_baseline_tensor(&baseline_path, "triposg.pixel_values", &Device::Cpu);
    assert_eq!(expected_pixel_values.dims(), [1, 3, 224, 224]);
    let (max_delta, mean_delta) =
        max_abs_delta_stats(&expected_pixel_values, &prepared.pixel_values);
    println!(
        "golden pixel_values parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 1e-5)"
    );
    assert!(
        max_delta <= 1e-5,
        "golden pixel_values max abs delta too large: {max_delta:.3e}"
    );

    let actual_embeds = encoder
        .forward(&prepared.pixel_values)
        .expect("TripoSG DINOv2 forward");
    assert_eq!(actual_embeds.dims(), [1, 257, 1024]);
    let expected_embeds =
        load_baseline_tensor(&baseline_path, "triposg.image_embeds", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_embeds, &actual_embeds);
    println!(
        "golden image_embeds parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 2e-3)"
    );
    assert!(
        max_delta <= 2e-3,
        "golden image_embeds max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
    drop(encoder);
    drop(prepared);

    // --- Phase B: step-0 CFG velocity --------------------------------------
    let dit = load_triposg_dit_on(&device);
    let latents_init = load_baseline_tensor(&baseline_path, "triposg.latents_init", &device);
    assert_eq!(latents_init.dims(), [1, 2048, 64]);
    let image_embeds = load_baseline_tensor(&baseline_path, "triposg.image_embeds", &device);
    assert_eq!(image_embeds.dims(), [1, 257, 1024]);

    let cond_batch = Tensor::cat(
        &[
            &Tensor::zeros_like(&image_embeds).expect("zeros"),
            &image_embeds,
        ],
        0,
    )
    .expect("cond batch");
    let latent_batch = Tensor::cat(&[&latents_init, &latents_init], 0).expect("latent batch");
    let velocity = dit
        .forward(&latent_batch, 1000.0, &cond_batch)
        .expect("DiT forward at t=1000");
    let parts = velocity.chunk(2, 0).expect("velocity chunk");
    let velocity_combined = (&parts[0]
        + (&parts[1] - &parts[0])
            .expect("sub")
            .affine(7.0, 0.0)
            .expect("scaled delta"))
    .expect("cfg combine");
    let expected_velocity =
        load_baseline_tensor(&baseline_path, "triposg.velocity_step_000", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_velocity, &velocity_combined);
    println!(
        "golden velocity_step_000 parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 5e-3)"
    );
    assert!(
        max_delta <= 5e-3,
        "golden velocity_step_000 max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
    drop(latent_batch);
    drop(cond_batch);
    drop(velocity);
    drop(velocity_combined);

    // --- Phase C: full 50-step CFG denoise ---------------------------------
    let started = Instant::now();
    let latents_final =
        denoise(&dit, &latents_init, &image_embeds, 50, 7.0).expect("50-step CFG denoise");
    let denoise_elapsed = started.elapsed();
    assert_eq!(latents_final.dims(), [1, 2048, 64]);
    let expected_final =
        load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_final, &latents_final);
    let expected_vec = expected_final
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let actual_vec = latents_final
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let mut deltas = expected_vec
        .iter()
        .zip(actual_vec.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .collect::<Vec<_>>();
    let mut squared_error = 0.0_f64;
    let mut expected_energy = 0.0_f64;
    let mut over_005 = 0usize;
    let mut over_01 = 0usize;
    let mut over_05 = 0usize;
    for (expected, actual) in expected_vec.iter().zip(actual_vec.iter()) {
        let delta = f64::from(expected - actual);
        squared_error += delta * delta;
        expected_energy += f64::from(*expected) * f64::from(*expected);
        let delta = delta.abs();
        over_005 += usize::from(delta > 0.05);
        over_01 += usize::from(delta > 0.1);
        over_05 += usize::from(delta > 0.5);
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).expect("finite deltas"));
    let pick = |fraction: f64| {
        let index = ((deltas.len() - 1) as f64 * fraction).round() as usize;
        deltas[index]
    };
    let relative_l2 = (squared_error / expected_energy.max(f64::EPSILON)).sqrt();
    println!(
        "golden latents_step_final parity (50 steps, denoise took {denoise_elapsed:?}): \
         max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e}, p99 {:.3e}, \
         p99.9 {:.3e}, p99.99 {:.3e}, relative L2 {relative_l2:.3e}, elements >0.05/0.1/0.5: \
         {over_005}/{over_01}/{over_05} of {} (statistical tolerance: mean <= 2e-2, \
         relative L2 <= 1.2e-1, max <= 4.0)",
        pick(0.99),
        pick(0.999),
        pick(0.9999),
        deltas.len()
    );
    // The 50-step CFG ODE is chaotic: steps 1 and 10 match the baseline to
    // 1e-6, but the reference itself drifts ~8% relative L2 when the same
    // code runs on a different GPU/torch build (control experiment:
    // 2026-11-12, torch 2.5 RTX 3060 vs torch 2.14 L4). The final state is
    // therefore judged statistically; per-step correctness is pinned by the
    // step-1/step-10 checks below, and surface parity by the mesh test.
    assert!(
        mean_delta <= 2e-2,
        "golden latents_step_final mean abs delta too large: {mean_delta:.3e} (max {max_delta:.3e})"
    );
    assert!(
        relative_l2 <= 1.2e-1,
        "golden latents_step_final relative L2 too large: {relative_l2:.3e}"
    );
    assert!(
        max_delta <= 4.0,
        "golden latents_step_final max abs delta outside the measured chaotic bound: \
         {max_delta:.3e}"
    );

    // Early-trajectory diagnostic: a 25-step loop with the same 50-step
    // schedule must stay close to the baseline snapshots. This separates
    // per-step correctness from the chaotic amplification of the last 25
    // steps (the reference itself drifts ~8% relative L2 across GPU/torch
    // environments, so the final state is judged statistically).
    let schedule = RectifiedFlowSchedule::new(50, 1.0);
    let uncond_cond = Tensor::zeros_like(&image_embeds).expect("uncond zeros");
    let cond_batch = Tensor::cat(&[&uncond_cond, &image_embeds], 0).expect("cond batch");
    let mut trajectory = latents_init.clone();
    for index in 0..25 {
        let batch = Tensor::cat(&[&trajectory, &trajectory], 0).expect("latent batch");
        let velocity = dit
            .forward(&batch, schedule.timestep(index), &cond_batch)
            .expect("trajectory DiT forward");
        let parts = velocity.chunk(2, 0).expect("velocity chunk");
        let combined = (&parts[0]
            + (&parts[1] - &parts[0])
                .expect("sub")
                .affine(7.0, 0.0)
                .expect("scaled delta"))
        .expect("cfg combine");
        trajectory = schedule
            .step(index, &combined, &trajectory)
            .expect("trajectory scheduler step");
        if index == 0 {
            let expected_step_001 =
                load_baseline_tensor(&baseline_path, "triposg.latents_step_001", &Device::Cpu);
            let (max_delta, mean_delta) = max_abs_delta_stats(&expected_step_001, &trajectory);
            println!(
                "golden latents_step_001 parity (1 of 50 steps): max abs delta {max_delta:.3e}, \
                 mean abs delta {mean_delta:.3e}"
            );
            assert!(
                max_delta <= 1e-2,
                "golden latents_step_001 max abs delta too large: {max_delta:.3e} \
                 (mean {mean_delta:.3e})"
            );
        }
        if index == 9 {
            let expected_step_010 =
                load_baseline_tensor(&baseline_path, "triposg.latents_step_010", &Device::Cpu);
            let (max_delta, mean_delta) = max_abs_delta_stats(&expected_step_010, &trajectory);
            println!(
                "golden latents_step_010 parity (10 of 50 steps): max abs delta {max_delta:.3e}, \
                 mean abs delta {mean_delta:.3e}"
            );
            assert!(
                max_delta <= 5e-2,
                "golden latents_step_010 max abs delta too large: {max_delta:.3e} \
                 (mean {mean_delta:.3e})"
            );
        }
    }
    let expected_step_025 =
        load_baseline_tensor(&baseline_path, "triposg.latents_step_025", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_step_025, &trajectory);
    println!(
        "golden latents_step_025 parity (25 of 50 steps): max abs delta {max_delta:.3e}, \
         mean abs delta {mean_delta:.3e}"
    );
    assert!(
        max_delta <= 5e-2,
        "golden latents_step_025 max abs delta too large: {max_delta:.3e} \
         (mean {mean_delta:.3e})"
    );
    drop(dit);

    // --- Phase D: SDF probe values -----------------------------------------
    let (decoder, _cfg) = load_triposg_vae_decoder_on(&device);
    let points = load_baseline_tensor(&baseline_path, "triposg.sdf_points", &Device::Cpu);
    assert_eq!(points.dims(), [8192, 3]);
    let points = points.to_device(&device).expect("points to device");

    let actual_sdf = decoder
        .decode_latents(&latents_final, &points, 16384)
        .expect("TripoSG VAE decode_latents (end-to-end latents)");
    let expected_sdf = load_baseline_tensor(&baseline_path, "triposg.sdf_values", &Device::Cpu);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_sdf, &actual_sdf);
    println!(
        "golden sdf_values parity (end-to-end denoised latents): max abs delta {max_delta:.3e}, \
         mean abs delta {mean_delta:.3e} (statistical tolerance: mean <= 1.5e-1, max <= 1.0e1)"
    );
    // Same chaotic-amplification argument as the final latents: the SDF is a
    // smooth decoder but the 50-step latent difference reaches ~1e-1 mean.
    assert!(
        mean_delta <= 1.5e-1,
        "golden sdf_values mean abs delta too large: {mean_delta:.3e} (max {max_delta:.3e})"
    );
    assert!(
        max_delta <= 1.0e1,
        "golden sdf_values max abs delta outside the measured chaotic bound: {max_delta:.3e}"
    );

    // Diagnostic: decoding the BASELINE latents isolates VAE drift from
    // denoise drift; the assertion below pins this path at 5e-3.
    let latents_baseline = expected_final
        .to_device(&device)
        .expect("latents to device");
    let baseline_sdf = decoder
        .decode_latents(&latents_baseline, &points, 16384)
        .expect("TripoSG VAE decode_latents (baseline latents)");
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_sdf, &baseline_sdf);
    println!(
        "golden sdf_values parity (baseline latents, VAE path assertion): max abs delta \
         {max_delta:.3e}, mean abs delta {mean_delta:.3e} (tolerance 5e-3)"
    );
    assert!(
        max_delta <= 5e-3,
        "golden VAE decode with baseline latents drifted: max abs delta {max_delta:.3e} \
         (mean {mean_delta:.3e})"
    );
}

/// Heavy golden case (orchestrator-run): depth-8 dense-only grid vs the
/// baseline `triposg.dense_sdf` [256,256,256]. Acceptance is statistical
/// (mean <= 1e-3, p99.99 <= 1e-2, max <= 1e-1) because ~1e-4 of the 16.7M
/// raw logits sit on steep surface gradients; measured: mean 5.27e-6,
/// p99.99 2.94e-3, max 7.32e-2, 256 values > 1e-2. Requires the golden
/// artifact AND `LUX3D_TRIPOSG_GOLDEN_RUN=1`; device via
/// `LUX3D_TRIPOSG_GOLDEN_DEVICE` (cpu|cuda, default cpu). Measured on the
/// RTX 3060 CUDA debug build: ~104 s (CPU is multi-hour):
///
/// ```text
/// LUX3D_TRIPOSG_GOLDEN_RUN=1 LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda \
///   cargo test -p lux3d-core --test runtime_triposg -- --test-threads=1 \
///   triposg_golden_dense_sdf_depth8 --nocapture
/// ```
#[test]
fn triposg_golden_dense_sdf_depth8_cpu() {
    let label = "TripoSG golden dense depth-8 grid";
    let baseline_path = golden_baseline_path();
    if !baseline_path.is_file() {
        eprintln!(
            "SKIP {label}: golden artifact is missing at {} (orchestrator golden run pending)",
            baseline_path.display()
        );
        return;
    }
    if !golden_env_flag("LUX3D_TRIPOSG_GOLDEN_RUN") {
        eprintln!(
            "SKIP {label}: set LUX3D_TRIPOSG_GOLDEN_RUN=1 to run (CPU ~2-3 h, CUDA ~10-20 min \
             via LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda)"
        );
        return;
    }
    let Some((device, _guard)) = golden_grid_device_and_lock(label) else {
        return;
    };

    let (decoder, _cfg) = load_triposg_vae_decoder_on(&device);
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &device);
    assert_eq!(latents.dims(), [1, 2048, 64]);

    let started = Instant::now();
    let grid = hierarchical_sdf_grid(&decoder, &latents, smoke_bounds(), 8, 8, 16384)
        .expect("TripoSG dense depth-8 SDF grid");
    let elapsed = started.elapsed();
    assert_eq!(grid.depth, 8, "hier == dense must return the dense grid");
    assert_eq!(grid.dims, 256);
    assert_eq!(grid.values.len(), 256 * 256 * 256);
    assert!(
        grid.values.iter().all(|value| value.is_finite()),
        "dense depth-8 grid must be finite"
    );

    let expected = load_baseline_tensor(&baseline_path, "triposg.dense_sdf", &Device::Cpu);
    assert_eq!(expected.dims(), [256, 256, 256], "golden dense_sdf shape");
    let expected = expected
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");

    // ours[x + y*n + z*n*n] <-> baseline[x*n*n + y*n + z] (C-order, x slowest)
    let n = 256usize;
    let mut max_delta = 0.0_f32;
    let mut sum_delta = 0.0_f64;
    let mut over_1e2 = 0usize;
    let mut over_1e1 = 0usize;
    let mut deltas = Vec::with_capacity(expected.len());
    for x in 0..n {
        for y in 0..n {
            for z in 0..n {
                let actual = grid.values[x + y * n + z * n * n];
                let expected = expected[x * n * n + y * n + z];
                let delta = (expected - actual).abs();
                max_delta = max_delta.max(delta);
                sum_delta += f64::from(delta);
                over_1e2 += usize::from(delta > 1e-2);
                over_1e1 += usize::from(delta > 1e-1);
                deltas.push(delta);
            }
        }
    }
    deltas.sort_by(|a, b| a.partial_cmp(b).expect("finite deltas"));
    let pick = |fraction: f64| {
        let index = ((deltas.len() - 1) as f64 * fraction).round() as usize;
        deltas[index]
    };
    let mean_delta = (sum_delta / expected.len() as f64) as f32;
    println!(
        "golden dense_sdf depth-8 parity: max abs delta {max_delta:.3e}, mean abs delta \
         {mean_delta:.3e}, p99.9 {:.3e}, p99.99 {:.3e}, p99.999 {:.3e}, elements >1e-2/1e-1: \
         {over_1e2}/{over_1e1} of {} (statistical tolerance: mean <= 1e-3, p99.99 <= 1e-2, \
         max <= 1e-1), grid took {elapsed:?}",
        pick(0.999),
        pick(0.9999),
        pick(0.99999),
        deltas.len()
    );
    assert!(
        mean_delta <= 1e-3,
        "golden dense_sdf depth-8 mean abs delta too large: {mean_delta:.3e} (max {max_delta:.3e})"
    );
    assert!(
        pick(0.9999) <= 1e-2,
        "golden dense_sdf depth-8 p99.99 abs delta too large: {:.3e} (max {max_delta:.3e})",
        pick(0.9999)
    );
    assert!(
        max_delta <= 1e-1,
        "golden dense_sdf depth-8 max abs delta too large: {max_delta:.3e} (mean {mean_delta:.3e})"
    );
}

/// Heavy golden case (orchestrator-run): full hierarchical refinement from
/// the depth-8 dense grid to depth 9 (512^3) built from the baseline
/// `latents_step_final`, exercising the previously untested 8->9 refinement
/// path. Assertions:
///   (a) grid depth 9 / dims 512, every one of the 134M values finite;
///   (b) refined values at high indices whose parent low cell had a sign
///       change differ from the (nearest-upsampled) parent by the stated
///       sanity bounds (mean <= 10, max <= 50 — the refined values are
///       genuine decoder outputs at half-cell offsets where raw SDF logits
///       can change by tens of units; measured mean 4.77, max 30.3, so this
///       only clamps absurd corruption; actual values are printed);
///   (c) no NaN/Inf (covered by (a));
///   (d) with the default culling path, only a small fraction of the volume
///       is re-queried (band around the surface), and at least one cell IS
///       re-queried.
/// Requires the golden artifact AND `LUX3D_TRIPOSG_GOLDEN_RUN=1`; device via
/// `LUX3D_TRIPOSG_GOLDEN_DEVICE` (cpu|cuda, default cpu; CPU 512^3 is a
/// multi-hour job — prefer `LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda`).
#[test]
fn triposg_golden_hierarchical_grid_refinement_8_to_9() {
    let label = "TripoSG golden hierarchical 8->9 refinement";
    let baseline_path = golden_baseline_path();
    if !baseline_path.is_file() {
        eprintln!(
            "SKIP {label}: golden artifact is missing at {} (orchestrator golden run pending)",
            baseline_path.display()
        );
        return;
    }
    if !golden_env_flag("LUX3D_TRIPOSG_GOLDEN_RUN") {
        eprintln!(
            "SKIP {label}: set LUX3D_TRIPOSG_GOLDEN_RUN=1 to run (CPU: multi-hour for 512^3; \
             CUDA ~15-30 min via LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda)"
        );
        return;
    }
    let Some((device, _guard)) = golden_grid_device_and_lock(label) else {
        return;
    };

    let (decoder, _cfg) = load_triposg_vae_decoder_on(&device);
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &device);
    assert_eq!(latents.dims(), [1, 2048, 64]);

    let started = Instant::now();
    let grid = hierarchical_sdf_grid(&decoder, &latents, smoke_bounds(), 8, 9, 16384)
        .expect("TripoSG hierarchical depth-9 SDF grid");
    let elapsed = started.elapsed();
    assert_eq!(grid.depth, 9, "refinement must reach hierarchical depth 9");
    assert_eq!(grid.dims, 512);
    assert_eq!(grid.values.len(), 512 * 512 * 512);
    assert!(
        grid.values.iter().all(|value| value.is_finite()),
        "refined depth-9 grid must be finite (no NaN/Inf)"
    );

    // Parent source: the baseline dense depth-8 grid (ours agrees within
    // 1e-2 per the dense depth-8 golden case, negligible against the bound).
    let dense = load_baseline_tensor(&baseline_path, "triposg.dense_sdf", &Device::Cpu);
    assert_eq!(dense.dims(), [256, 256, 256], "golden dense_sdf shape");
    let dense = dense
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let n = 256usize;

    // Sign-change flags on the low grid (3x3x3 clamped neighborhood has both
    // a negative and a non-negative value — same rule as the runtime cull).
    let mut parent_sign_change = vec![false; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let mut has_negative = false;
                let mut has_non_negative = false;
                'neighborhood: for dz in -1i64..=1 {
                    let wz = (z as i64 + dz).clamp(0, n as i64 - 1) as usize;
                    for dy in -1i64..=1 {
                        let wy = (y as i64 + dy).clamp(0, n as i64 - 1) as usize;
                        for dx in -1i64..=1 {
                            let wx = (x as i64 + dx).clamp(0, n as i64 - 1) as usize;
                            if dense[wx + wy * n + wz * n * n] < 0.0 {
                                has_negative = true;
                            } else {
                                has_non_negative = true;
                            }
                            if has_negative && has_non_negative {
                                parent_sign_change[x + y * n + z * n * n] = true;
                                break 'neighborhood;
                            }
                        }
                    }
                }
            }
        }
    }

    // Compare the refined grid against the nearest-upsampled parent values.
    // The decoder emits raw SDF logits, whose gradient can be steep at the
    // surface: a half-cell offset (2.01/512) can legitimately change a value
    // by tens of logit units. Measured on the golden grid: mean 4.77, max
    // 30.3 over 3.36M re-queried cells. The bound only rejects absurd
    // corruption; surface correctness is asserted by the mesh test.
    let refine_all = golden_env_flag("LUX3D_TRIPOSG_REFINE_ALL");
    let big = 512usize;
    const REFINED_DELTA_BOUND: f32 = 5.0e1;
    const REFINED_MEAN_DELTA_BOUND: f32 = 1.0e1;
    let mut refined_count = 0usize;
    let mut nonzero_refined = 0usize;
    let mut max_refined_delta = 0.0_f32;
    let mut sum_refined_delta = 0.0_f64;
    let mut max_all_delta = 0.0_f32;
    for hz in 0..big {
        for hy in 0..big {
            let parent_row = (hz / 2) * n * n + (hy / 2) * n;
            let dst = hz * big * big + hy * big;
            for hx in 0..big {
                let parent_index = parent_row + hx / 2;
                let delta = (grid.values[dst + hx] - dense[parent_index]).abs();
                max_all_delta = max_all_delta.max(delta);
                if parent_sign_change[parent_index] {
                    refined_count += 1;
                    if delta > 1e-6 {
                        nonzero_refined += 1;
                    }
                    max_refined_delta = max_refined_delta.max(delta);
                    sum_refined_delta += f64::from(delta);
                }
            }
        }
    }
    let mean_refined_delta = if refined_count > 0 {
        (sum_refined_delta / refined_count as f64) as f32
    } else {
        0.0
    };
    let refined_fraction = refined_count as f64 / (big * big * big) as f64;
    println!(
        "golden 8->9 refinement: grid took {elapsed:?}, refine_all={refine_all}; sign-change-parent \
         high cells {refined_count} ({:.2}% of 512^3), nonzero deltas {nonzero_refined}, \
         max delta {max_refined_delta:.3e}, mean delta {mean_refined_delta:.3e} \
         (bounds: mean <= {REFINED_MEAN_DELTA_BOUND}, max <= {REFINED_DELTA_BOUND}); \
         max delta over all cells {max_all_delta:.3e}",
        refined_fraction * 100.0
    );

    // (b) refined values stay within the stated sanity bounds.
    assert!(
        max_refined_delta <= REFINED_DELTA_BOUND,
        "refined values at sign-change-parent indices exceed the stated bound: \
         {max_refined_delta:.3e} > {REFINED_DELTA_BOUND}"
    );
    assert!(
        mean_refined_delta <= REFINED_MEAN_DELTA_BOUND,
        "refined values at sign-change-parent indices have an implausible mean delta: \
         {mean_refined_delta:.3e} > {REFINED_MEAN_DELTA_BOUND}"
    );
    // (d) refinement actually re-queried the band (not just nearest copies).
    assert!(
        refined_count > 0 && nonzero_refined > 0,
        "the 8->9 refinement must re-query sign-change-parent cells (got \
         {refined_count} cells, {nonzero_refined} nonzero deltas)"
    );
    if !refine_all {
        // Default culling path: only the surface band may differ from the
        // nearest-upsampled parent (measured: a fraction of a percent).
        assert!(
            refined_fraction < 0.05,
            "default culling re-queried {refined_count} cells ({:.2}%) — expected only the \
             surface band (< 5%)",
            refined_fraction * 100.0
        );
    }
}

/// Small-but-real refinement coverage that runs NOW on CPU against the smoke
/// baseline (dense 5 -> hierarchical 6, 32 -> 64, ~2 min): locks the
/// refinement contract end-to-end while the golden 8->9 case is gated on the
/// orchestrator run:
///   - refined grid depth 6 / dims 64, all finite;
///   - every high cell whose parent was NOT re-queried is bit-identical to
///     the nearest-upsampled parent value (the documented fill contract);
///   - re-queried children (band + Chebyshev-2 dilation, sign-change culling
///     on by default) are finite and differ from the parent by the stated
///     sanity bound;
///   - marching cubes on the fp16-rounded depth-6 grid yields a valid
///     non-empty finite surface (mesh stats printed informationally: the
///     smoke object is sub-cell thin, so the depth-5 baseline mesh is NOT a
///     parity reference for the depth-6 surface — see the in-test comment).
#[test]
fn triposg_hierarchical_refinement_dense5_to_6_matches_smoke_baseline_cpu() {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );

    let (decoder, _cfg) = load_triposg_vae_decoder_cpu();
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    let bounds = smoke_bounds();

    let dense = hierarchical_sdf_grid(&decoder, &latents, bounds, 5, 5, 16384)
        .expect("TripoSG dense depth-5 SDF grid");
    assert_eq!(dense.depth, 5);
    assert_eq!(dense.dims, 32);

    let started = Instant::now();
    let refined = hierarchical_sdf_grid(&decoder, &latents, bounds, 5, 6, 16384)
        .expect("TripoSG hierarchical depth-6 SDF grid");
    let elapsed = started.elapsed();
    assert_eq!(
        refined.depth, 6,
        "refinement must reach hierarchical depth 6"
    );
    assert_eq!(refined.dims, 64);
    assert_eq!(refined.values.len(), 64 * 64 * 64);
    assert!(
        refined.values.iter().all(|value| value.is_finite()),
        "refined depth-6 grid must be finite"
    );

    // Replicate the documented refinement marking on the low grid: interior
    // cells in the |sigmoid*2-1| < 1 band, re-queried only when the clamped
    // 3x3x3 neighborhood has a sign change (unless LUX3D_TRIPOSG_REFINE_ALL=1),
    // then dilated by Chebyshev radius 2 (n = 32 < 512).
    let n = 32usize;
    let refine_all = golden_env_flag("LUX3D_TRIPOSG_REFINE_ALL");
    let mut marked = vec![false; n * n * n];
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let v = dense.values[x + y * n + z * n * n];
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                let in_band = (sigmoid * 2.0 - 1.0).abs() < 1.0;
                if in_band
                    && (refine_all || {
                        let mut has_negative = false;
                        let mut has_non_negative = false;
                        let mut sign_change = false;
                        for dz in -1i64..=1 {
                            let wz = (z as i64 + dz).clamp(0, n as i64 - 1) as usize;
                            for dy in -1i64..=1 {
                                let wy = (y as i64 + dy).clamp(0, n as i64 - 1) as usize;
                                for dx in -1i64..=1 {
                                    let wx = (x as i64 + dx).clamp(0, n as i64 - 1) as usize;
                                    if dense.values[wx + wy * n + wz * n * n] < 0.0 {
                                        has_negative = true;
                                    } else {
                                        has_non_negative = true;
                                    }
                                    if has_negative && has_non_negative {
                                        sign_change = true;
                                        break;
                                    }
                                }
                                if sign_change {
                                    break;
                                }
                            }
                            if sign_change {
                                break;
                            }
                        }
                        sign_change
                    })
                {
                    marked[x + y * n + z * n * n] = true;
                }
            }
        }
    }
    let radius = 2usize; // n < 512 -> kernel 5 (radius 2), same as refine_level
    let mut dilated = vec![false; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                if !marked[x + y * n + z * n * n] {
                    continue;
                }
                let z0 = z.saturating_sub(radius);
                let z1 = (z + radius).min(n - 1);
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(n - 1);
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(n - 1);
                for wz in z0..=z1 {
                    for wy in y0..=y1 {
                        for wx in x0..=x1 {
                            dilated[wx + wy * n + wz * n * n] = true;
                        }
                    }
                }
            }
        }
    }

    // Fill contract: non-re-queried high cells are bit-identical to the
    // nearest-upsampled parent; re-queried children are finite and stay in
    // the stated sanity bound.
    const REFINED_DELTA_BOUND: f32 = 25.0;
    let big = 64usize;
    let mut requeried_count = 0usize;
    let mut nonzero_requeried = 0usize;
    let mut max_requeried_delta = 0.0_f32;
    for hz in 0..big {
        for hy in 0..big {
            let parent_row = (hz / 2) * n * n + (hy / 2) * n;
            let dst = hz * big * big + hy * big;
            for hx in 0..big {
                let parent_index = parent_row + hx / 2;
                let parent_value = dense.values[parent_index];
                let high_value = refined.values[dst + hx];
                if dilated[parent_index] {
                    requeried_count += 1;
                    let delta = (high_value - parent_value).abs();
                    if delta > 1e-6 {
                        nonzero_requeried += 1;
                    }
                    max_requeried_delta = max_requeried_delta.max(delta);
                    assert!(
                        high_value.is_finite(),
                        "re-queried high cell ({hx},{hy},{hz}) must be finite"
                    );
                } else {
                    assert_eq!(
                        high_value.to_bits(),
                        parent_value.to_bits(),
                        "non-re-queried high cell ({hx},{hy},{hz}) must equal the \
                         nearest-upsampled parent bit-exactly"
                    );
                }
            }
        }
    }
    println!(
        "dense 5->6 refinement: grid took {elapsed:?}, refine_all={refine_all}, re-queried \
         high cells {requeried_count} of {}, nonzero deltas {nonzero_requeried}, \
         max re-queried delta {max_requeried_delta:.3e} (bound {REFINED_DELTA_BOUND})",
        big * big * big
    );
    assert!(
        requeried_count > 0 && nonzero_requeried > 0,
        "the 5->6 refinement must re-query the surface band"
    );
    assert!(
        max_requeried_delta <= REFINED_DELTA_BOUND,
        "re-queried values exceed the stated bound: {max_requeried_delta:.3e} > {REFINED_DELTA_BOUND}"
    );

    // Mesh validity across refinement (reference fp16 mesh path). NOTE: the
    // smoke object is SUB-CELL thin (bbox y extent ~0.009 at depth-5 cell
    // size 0.0628), so the depth-6 grid legitimately resolves far more of
    // the surface than the depth-5 baseline mesh (measured: 1856 vs 520
    // faces, ~3.4x volume) — the depth-5 mesh is NOT a parity reference for
    // the depth-6 surface on this sample. Surface-identity parity against
    // the baseline mesh is covered by the golden depth-9 mesh case in
    // tests/geometry_triposg.rs, where the object is well-resolved. Here we
    // only assert the refined mesh is a valid non-empty finite surface.
    let rounded_values = Tensor::from_vec(
        refined.values.clone(),
        (refined.values.len(),),
        &Device::Cpu,
    )
    .expect("refined values tensor")
    .to_dtype(DType::F16)
    .expect("grid f16 cast")
    .to_dtype(DType::F32)
    .expect("grid f32 cast")
    .to_vec1::<f32>()
    .expect("grid vec");
    let refined_fp16 = TripoSgSdfGrid {
        depth: refined.depth,
        dims: refined.dims,
        bounds: refined.bounds,
        values: rounded_values,
    };
    let stage = lux3d_core::geometry::TripoSgGeometryStage;
    let mesh = stage
        .assemble_cpu(&refined_fp16)
        .expect("depth-6 mesh assemble");
    assert!(
        !mesh.vertices.is_empty() && !mesh.faces.is_empty(),
        "refined depth-6 mesh must be non-empty (vertices {}, faces {})",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3
    );
    assert!(
        mesh.vertices.iter().all(|value| value.is_finite()),
        "refined mesh vertices must be finite"
    );
    assert!(
        mesh.faces
            .iter()
            .all(|index| (0..(mesh.vertices.len() / 3) as i64).contains(index)),
        "refined mesh faces must be zero-based and in range"
    );

    let baseline_vertices =
        load_baseline_tensor(&baseline_path, "triposg.mesh_vertices", &Device::Cpu)
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
    let baseline_faces = load_baseline_tensor(&baseline_path, "triposg.mesh_faces", &Device::Cpu)
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec")
        .iter()
        .map(|value| value.round() as i64)
        .collect::<Vec<i64>>();

    let (our_min, our_max) = bbox(&mesh.vertices);
    let (base_min, base_max) = bbox(&baseline_vertices);
    let our_volume = signed_volume(&mesh.vertices, &mesh.faces);
    let baseline_volume = signed_volume(&baseline_vertices, &baseline_faces);
    println!(
        "dense 5->6 mesh (informational, no parity assert — sub-cell-thin smoke object): \
         ours vertices={} faces={} bbox_min={our_min:?} bbox_max={our_max:?} volume={our_volume:.6}; \
         baseline (depth 5) vertices={} faces={} bbox_min={base_min:?} bbox_max={base_max:?} \
         volume={baseline_volume:.6}",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
        baseline_vertices.len() / 3,
        baseline_faces.len() / 3,
    );
}

/// Absolute enclosed volume via the signed tetrahedron sum (winding-consistent
/// marching-cubes output makes `abs` valid).
fn signed_volume(vertices: &[f32], faces: &[i64]) -> f64 {
    let mut volume = 0.0_f64;
    for triangle in faces.chunks_exact(3) {
        let a = triangle[0] as usize * 3;
        let b = triangle[1] as usize * 3;
        let c = triangle[2] as usize * 3;
        let (ax, ay, az) = (
            f64::from(vertices[a]),
            f64::from(vertices[a + 1]),
            f64::from(vertices[a + 2]),
        );
        let (bx, by, bz) = (
            f64::from(vertices[b]),
            f64::from(vertices[b + 1]),
            f64::from(vertices[b + 2]),
        );
        let (cx, cy, cz) = (
            f64::from(vertices[c]),
            f64::from(vertices[c + 1]),
            f64::from(vertices[c + 2]),
        );
        volume += ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx);
    }
    (volume / 6.0).abs()
}

/// Per-component bbox (min, max) of a flat [x0, y0, z0, x1, y1, z1, ...] list.
fn bbox(vertices: &[f32]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (min, max)
}
