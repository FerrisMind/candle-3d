//! TRELLIS.2 pipeline orchestration tests (T5 geometry slice):
//!
//! 1. Non-gated: config defaults mirror `pipeline.json`,
//!    `from_resolved_config` overrides, RNG determinism/distribution, run
//!    options, raw-dir stage-file validation + `pipeline.json` parsing.
//! 2. Gated (`models/TRELLIS.2-4B` raw dir, or `LUX3D_TRELLIS2_RAW_DIR`):
//!    the pipeline golden shape decode replayed at resolution 128 — mesh
//!    parity against the Python oracle `shape_mesh.npz` (vertices <= 1e-5,
//!    faces equal as a multiset), bbox/NaN checks, export smoke.
//! 3. Fully gated (`LUX3D_TRELLIS2_RUN_E2E=1`): the real 512 image-to-mesh
//!    run on raw weights. Run it in release (CPU cost):
//!    `LUX3D_TRELLIS2_RUN_E2E=1 cargo test --release -p lux3d-core --test
//!    pipeline_trellis2 pipeline_full_512_run_e2e -- --nocapture
//!    --test-threads 1`. `LUX3D_TRELLIS2_E2E_STEPS` overrides the sampler
//!    step count for a shorter smoke run (default 12 = reference);
//!    `LUX3D_TRELLIS2_E2E_DEVICE` selects cpu|cuda|vulkan|wgpu (default cpu).

use std::path::PathBuf;

use candle_core::{Device, Tensor};
use lux3d_core::contracts::Trellis2PipelineType;
use lux3d_core::runtime::{
    TRELLIS2_SHAPE_SLAT_MEAN, TRELLIS2_SHAPE_SLAT_STD, TRELLIS2_VOXEL_MARGIN, Trellis2Pipeline,
    Trellis2PipelineConfig, Trellis2Rng, Trellis2RunOptions, Trellis2SamplerParams,
};
use lux3d_core::test_support::workspace_root;

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-vae-golden")
}

fn raw_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_RAW_DIR") {
        let path = PathBuf::from(path);
        return path.is_dir().then_some(path);
    }
    let path = workspace_root().join("models").join("TRELLIS.2-4B");
    path.is_dir().then_some(path)
}

/// Device for the gated e2e run: `LUX3D_TRELLIS2_E2E_DEVICE` =
/// `cpu|cuda|vulkan|wgpu` (default `cpu`). The matching cargo feature must be
/// enabled; a backend that cannot be constructed is a hard error.
fn e2e_device() -> Device {
    let requested = std::env::var("LUX3D_TRELLIS2_E2E_DEVICE")
        .unwrap_or_else(|_| "cpu".to_string())
        .to_lowercase();
    match requested.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)
            .expect("LUX3D_TRELLIS2_E2E_DEVICE=cuda requires a CUDA device"),
        "vulkan" => Device::new_vulkan(0)
            .expect("LUX3D_TRELLIS2_E2E_DEVICE=vulkan requires a Vulkan device"),
        "wgpu" => {
            Device::new_wgpu(0).expect("LUX3D_TRELLIS2_E2E_DEVICE=wgpu requires a wgpu device")
        }
        other => panic!("LUX3D_TRELLIS2_E2E_DEVICE must be cpu|cuda|vulkan|wgpu, got {other}"),
    }
}

fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lux3d-trellis2-pipeline-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// The 9 raw stage files `load_from_raw_dir` requires (T6 added the 1024
/// shape DiT for the cascade; T7 added the texture DiT/VAE stages).
const RAW_STAGE_FILES: [(&str, &str); 9] = [
    ("image_cond", "external/dinov3/model.safetensors"),
    ("ss_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    (
        "ss_dec",
        "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    (
        "shape_dit_512",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "shape_dit_1024",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    ),
    (
        "shape_dec",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    ),
    (
        "tex_dit_512",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "tex_dit_1024",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
    (
        "tex_dec",
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
    ),
];

#[test]
fn pipeline_config_defaults_match_pipeline_json() {
    let config = Trellis2PipelineConfig::default();
    assert_eq!(config.shape_slat_mean, TRELLIS2_SHAPE_SLAT_MEAN.to_vec());
    assert_eq!(config.shape_slat_std, TRELLIS2_SHAPE_SLAT_STD.to_vec());
    assert_eq!(config.shape_slat_mean.len(), 32);
    assert_eq!(config.shape_slat_std.len(), 32);
    // Spot-check exact pipeline.json values.
    assert_eq!(config.shape_slat_mean[0], 0.781296);
    assert_eq!(config.shape_slat_mean[20], 3.790713);
    assert_eq!(config.shape_slat_mean[31], 1.578665);
    assert_eq!(config.shape_slat_std[0], 5.972266);
    assert_eq!(config.shape_slat_std[26], 6.110351);
    assert_eq!(config.shape_slat_std[31], 5.405691);

    assert_eq!(config.ss_sampler, Trellis2SamplerParams::default());
    assert_eq!(
        config.ss_sampler,
        Trellis2SamplerParams {
            steps: 12,
            sigma_min: 1e-5,
            guidance_strength: 7.5,
            guidance_rescale: 0.7,
            guidance_interval: [0.6, 1.0],
            rescale_t: 5.0,
        }
    );
    assert_eq!(
        config.shape_sampler,
        Trellis2SamplerParams {
            steps: 12,
            sigma_min: 1e-5,
            guidance_strength: 7.5,
            guidance_rescale: 0.5,
            guidance_interval: [0.6, 1.0],
            rescale_t: 3.0,
        }
    );
    assert_eq!(config.ss_resolution, 32);
    assert_eq!(config.shape_resolution, 512);
    assert_eq!(config.voxel_margin, TRELLIS2_VOXEL_MARGIN);
    assert_eq!(config.voxel_margin, 0.5);
}

#[test]
fn from_resolved_config_applies_overrides() {
    let mean: Vec<f64> = (0..32).map(|i| i as f64 * 0.25).collect();
    // Exact-in-binary steps so the f32 casts compare exactly.
    let std: Vec<f64> = (0..32).map(|i| 1.0 + i as f64 * 0.125).collect();
    let value = serde_json::json!({
        "pipeline_args": {
            "shape_slat_normalization": { "mean": mean, "std": std },
            "sparse_structure_sampler": {
                "args": { "sigma_min": 0.002 },
                "params": {
                    "steps": 9,
                    "guidance_strength": 4.0,
                    "guidance_rescale": 0.3,
                    "guidance_interval": [0.5, 0.95],
                    "rescale_t": 4.0
                }
            }
        }
    });
    let config = Trellis2PipelineConfig::from_resolved_config(&value);
    assert_eq!(
        config.shape_slat_mean,
        (0..32).map(|i| i as f32 * 0.25).collect::<Vec<_>>()
    );
    assert_eq!(
        config.shape_slat_std,
        (0..32).map(|i| 1.0 + i as f32 * 0.125).collect::<Vec<_>>()
    );
    assert_eq!(config.ss_sampler.steps, 9);
    assert_eq!(config.ss_sampler.sigma_min, 0.002);
    assert_eq!(config.ss_sampler.guidance_strength, 4.0);
    assert_eq!(config.ss_sampler.guidance_rescale, 0.3);
    assert_eq!(config.ss_sampler.guidance_interval, [0.5, 0.95]);
    assert_eq!(config.ss_sampler.rescale_t, 4.0);
    // Sections not present keep the defaults.
    assert_eq!(
        config.shape_sampler,
        Trellis2PipelineConfig::default().shape_sampler
    );
    assert_eq!(config.shape_resolution, 512);

    // Raw pipeline.json nesting (no `pipeline_args` wrapper) also applies.
    let raw = serde_json::json!({
        "shape_slat_normalization": { "mean": mean }
    });
    let config = Trellis2PipelineConfig::from_resolved_config(&raw);
    assert_eq!(
        config.shape_slat_mean,
        (0..32).map(|i| i as f32 * 0.25).collect::<Vec<_>>()
    );
    assert_eq!(config.shape_slat_std, TRELLIS2_SHAPE_SLAT_STD.to_vec());

    // Wrong-length vectors are ignored (defaults stay).
    let bad_mean = vec![0.0_f64; 8];
    let bad = serde_json::json!({
        "shape_slat_normalization": { "mean": bad_mean }
    });
    let config = Trellis2PipelineConfig::from_resolved_config(&bad);
    assert_eq!(config.shape_slat_mean, TRELLIS2_SHAPE_SLAT_MEAN.to_vec());
}

#[test]
fn rng_deterministic_and_distributed() {
    let a = Trellis2Rng::new(42).randn(512);
    let b = Trellis2Rng::new(42).randn(512);
    assert_eq!(a, b, "same seed -> identical stream");
    let c = Trellis2Rng::new(43).randn(512);
    assert_ne!(a, c, "different seeds -> different stream");

    let samples = Trellis2Rng::new(7).randn(20000);
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&v| (v as f64 - mean) * (v as f64 - mean))
        .sum::<f64>()
        / n;
    assert!(mean.abs() < 0.05, "mean {mean:.4}");
    assert!((0.95..1.05).contains(&var.sqrt()), "std {:.4}", var.sqrt());
}

#[test]
fn run_options_defaults_match_reference_signature() {
    let options = Trellis2RunOptions::default();
    assert_eq!(options.seed, 42);
    assert_eq!(options.pipeline_type, Trellis2PipelineType::Cascade1024);
    assert_eq!(options.ss_steps, 12);
    assert_eq!(options.shape_steps, 12);
    assert_eq!(options.tex_steps, 12);
    assert_eq!(options.max_num_tokens, 49152);
    assert!(options.preprocess);
}

#[test]
fn raw_dir_rejects_missing_stage_files() {
    let dir = temp_dir("empty");
    let result = Trellis2Pipeline::load_from_raw_dir(&dir, &Device::Cpu);
    assert!(result.is_err(), "missing stage files must fail");
    // With all 6 files present (empty placeholders) the load succeeds and
    // parses pipeline.json overrides.
    for (_, relative) in RAW_STAGE_FILES {
        let file = dir.join(relative);
        std::fs::create_dir_all(file.parent().expect("parent")).expect("dirs");
        std::fs::write(&file, b"").expect("placeholder");
    }
    let ones = vec![1.0_f64; 32];
    let twos = vec![2.0_f64; 32];
    std::fs::write(
        dir.join("pipeline.json"),
        serde_json::json!({
            "pipeline_args": {
                "shape_slat_normalization": {
                    "mean": ones,
                    "std": twos
                }
            }
        })
        .to_string(),
    )
    .expect("pipeline.json");
    let pipeline =
        Trellis2Pipeline::load_from_raw_dir(&dir, &Device::Cpu).expect("load with placeholders");
    assert_eq!(pipeline.config.shape_slat_mean, vec![1.0; 32]);
    assert_eq!(pipeline.config.shape_slat_std, vec![2.0; 32]);

    // The single-resolution `1024` pipeline variant is rejected before any
    // weight loading (it lands with the texture stage T7; the cascade is
    // implemented since T6). The SS flow latent is [1, 8, 16^3].
    let cond = Tensor::zeros((1, 4), candle_core::DType::F32, &Device::Cpu).expect("cond");
    let ss_init = Tensor::zeros(
        (1usize, 8, 16, 16, 16),
        candle_core::DType::F32,
        &Device::Cpu,
    )
    .expect("ss init");
    let options = Trellis2RunOptions {
        pipeline_type: Trellis2PipelineType::R1024,
        ..Trellis2RunOptions::default()
    };
    let error = pipeline
        .run_with_latents(&cond, &ss_init, None, None, &Device::Cpu, &options)
        .expect_err("single-resolution 1024 rejected until T7");
    let message = error.to_string();
    assert!(message.contains("not implemented"), "message: {message}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Gated golden replay through the pipeline surface: raw-dir stage loading +
/// `decode_shape_slat` at the oracle resolution + mesh parity against the
/// Python oracle. Skips without the raw dir or the golden fixture.
#[test]
fn pipeline_golden_shape_decode_at_oracle_resolution() {
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let golden = golden_dir();
    if !golden.join("shape_mesh.npz").is_file() {
        eprintln!("SKIP: VAE oracle golden missing at {}", golden.display());
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
    let latent = Tensor::from_vec(
        slat_feats.iter().copied().collect::<Vec<f32>>(),
        (64, 32),
        &Device::Cpu,
    )
    .expect("latent tensor");

    let pipeline =
        Trellis2Pipeline::load_from_raw_dir(&raw, &Device::Cpu).expect("raw-dir pipeline");
    let started = std::time::Instant::now();
    let stages = pipeline
        .decode_shape_slat(&coords, &latent, 128, &Device::Cpu)
        .expect("golden decode through the pipeline");
    println!(
        "pipeline golden decode in {:.1}s",
        started.elapsed().as_secs_f64()
    );
    let mesh = stages.mesh.expect("decode_shape_slat produces a mesh");
    mesh.validate().expect("mesh buffers valid");
    assert_eq!(mesh.resolution, 128);
    assert_eq!(mesh.vertex_count(), 7666);
    assert_eq!(mesh.face_count(), 12186);
    assert_eq!(mesh.voxel_count(), 7666);
    assert!(!mesh.has_non_finite_vertex());

    // Golden mesh parity.
    let npz_file = std::fs::File::open(golden.join("shape_mesh.npz")).expect("open npz");
    let mut npz = ndarray_npy::NpzReader::new(npz_file).expect("npz reader");
    let golden_vertices = npz
        .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix2>("vertices")
        .expect("vertices");
    let golden_faces = npz
        .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix2>("faces")
        .expect("faces");
    assert_eq!(golden_vertices.dim(), (7666, 3));
    assert_eq!(golden_faces.dim(), (12186, 3));

    let mut max_abs = 0.0_f32;
    for (index, vertex) in mesh.vertices.chunks_exact(3).enumerate() {
        for axis in 0..3 {
            max_abs = max_abs.max((vertex[axis] - golden_vertices[[index, axis]]).abs());
        }
    }
    println!("pipeline golden mesh vertex max abs diff: {max_abs:.3e}");
    assert!(max_abs <= 1e-5, "vertex max abs diff {max_abs:.3e}");

    let mut rust_faces: Vec<[i64; 3]> = mesh
        .faces
        .chunks_exact(3)
        .map(|f| {
            let mut f = [f[0], f[1], f[2]];
            f.sort_unstable();
            f
        })
        .collect();
    let mut golden_face_list: Vec<[i64; 3]> = golden_faces
        .rows()
        .into_iter()
        .map(|row| {
            let mut f = [row[0], row[1], row[2]];
            f.sort_unstable();
            f
        })
        .collect();
    rust_faces.sort_unstable();
    golden_face_list.sort_unstable();
    assert_eq!(rust_faces, golden_face_list, "faces equal as a multiset");

    // Bbox / NaN checks (acceptance gate).
    let (min, max) = mesh.bbox().expect("bbox");
    for axis in 0..3 {
        assert!(min[axis] >= -0.5 && min[axis] <= 0.5, "min {min:?}");
        assert!(max[axis] >= -0.5 && max[axis] <= 0.5, "max {max:?}");
    }

    // Export smoke through the pipeline surface.
    let obj_path = std::env::temp_dir().join(format!(
        "lux3d-trellis2-pipeline-golden-{}.obj",
        std::process::id()
    ));
    let format = pipeline.export_mesh(&mesh, &obj_path).expect("export obj");
    assert_eq!(format, lux3d_core::export::MeshFormat::Obj);
    let bytes = std::fs::read(&obj_path).expect("obj read");
    assert!(bytes.len() > 100_000, "obj payload {}", bytes.len());
    let _ = std::fs::remove_file(&obj_path);

    // Contract conversion of the golden mesh.
    let contract = mesh.to_contract().expect("contract");
    assert_eq!(contract.vertices.len(), 7666 * 3);
    assert_eq!(contract.faces.len(), 12186 * 3);
}

/// Fully gated real-weight 512 image-to-mesh run. See the module docs for
/// the exact command; skips unless `LUX3D_TRELLIS2_RUN_E2E=1`.
#[test]
fn pipeline_full_512_run_e2e() {
    if std::env::var("LUX3D_TRELLIS2_RUN_E2E").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set LUX3D_TRELLIS2_RUN_E2E=1 to run the real 512 pipeline");
        return;
    }
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };

    // Synthetic RGBA input: opaque rounded square on transparent background.
    let image_path = temp_dir("e2e").join("input.png");
    {
        let mut image = image::RgbaImage::new(96, 96);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let inside = (16..80).contains(&x) && (16..80).contains(&y);
            *pixel = if inside {
                image::Rgba([180, 120, 60, 255])
            } else {
                image::Rgba([0, 0, 0, 0])
            };
        }
        image.save(&image_path).expect("save input png");
    }

    let steps: usize = std::env::var("LUX3D_TRELLIS2_E2E_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let device = e2e_device();
    println!("512 e2e device: {:?}", device.location());
    let pipeline = Trellis2Pipeline::load_from_raw_dir(&raw, &device).expect("raw-dir pipeline");
    let options = Trellis2RunOptions {
        pipeline_type: Trellis2PipelineType::R512,
        ss_steps: steps,
        shape_steps: steps,
        ..Trellis2RunOptions::default()
    };
    let started = std::time::Instant::now();
    let mesh = pipeline
        .run_from_path(&image_path, &device, &options)
        .expect("512 e2e run");
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "512 e2e run: {steps} steps, {} vertices, {} faces, {elapsed:.1}s",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
    );

    // Acceptance: non-empty mesh, bbox in [-0.5, 0.5], no NaN.
    assert!(mesh.vertices.len() >= 3, "non-empty mesh");
    assert!(!mesh.faces.is_empty(), "faces > 0");
    for vertex in mesh.vertices.chunks_exact(3) {
        for value in vertex {
            assert!(value.is_finite(), "vertex NaN/Inf");
        }
    }
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in mesh.vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    println!("bbox min {min:?} max {max:?}");
    // Backend telemetry: CPU-fallback counters for the wgpu/vulkan paths
    // (0 expected for a fully on-device run).
    #[cfg(feature = "wgpu")]
    println!(
        "[candle-obs] wgpu cpu_fallback={} host_compute={}",
        candle_core::wgpu_cpu_fallback_count(),
        candle_core::wgpu_host_compute_count()
    );
    #[cfg(feature = "vulkan")]
    println!(
        "[candle-obs] vulkan cpu_fallback={} host_compute={}",
        candle_core::vulkan_cpu_fallback_count(),
        candle_core::vulkan_host_compute_count()
    );
    // The dual-grid vertex `(coord + dual) * voxel_size - 0.5` can exceed
    // the box by up to voxel_size/2 when a boundary voxel's dual saturates
    // (sigmoid underflow -> dual = -0.5 at coord 0); the reference does not
    // clamp either. Accept the geometric limit 0.5 + 1/(2*resolution).
    let bound = 0.5 + 1.0 / (2.0 * 512.0) + 1e-6;
    for axis in 0..3 {
        assert!(
            min[axis] >= -bound && max[axis] <= bound,
            "bbox within [-0.5 - voxel/2, 0.5 + voxel/2]: min {min:?} max {max:?}"
        );
    }
    assert_eq!(mesh.resolution, 512);

    // Export the produced mesh next to the test temp dir.
    let voxel_count = mesh.voxel_coords.len() / 3;
    let attr_channels = if voxel_count == 0 || mesh.voxel_attrs.is_empty() {
        7
    } else {
        mesh.voxel_attrs.len() / voxel_count
    };
    let buffers = lux3d_core::geometry::Trellis2MeshBuffers {
        resolution: mesh.resolution,
        vertices: mesh.vertices.clone(),
        faces: mesh.faces.clone(),
        voxel_coords: mesh.voxel_coords.clone(),
        voxel_attrs: mesh.voxel_attrs.clone(),
        voxel_attr_channels: attr_channels,
        vertex_colors: None,
    };
    let out = temp_dir("e2e").join("mesh.obj");
    pipeline
        .export_mesh(&buffers, &out)
        .expect("export e2e mesh");
    println!("e2e mesh exported to {}", out.display());
}
