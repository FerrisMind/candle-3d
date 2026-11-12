//! TripoSG pipeline assembly tests (Task 4, CPU-only; T6 adds CUDA/Vulkan/wgpu
//! end-to-end runs).
//!
//! The canonical package is 7.9 GB and its manifest checksum validation takes
//! ~15 s per load, so this binary loads the pipeline exactly once (the smoke
//! test) and everything else stays off the weights:
//! - the seeded-RNG test runs through the public `TripoSgPipeline::
//!   sample_latents` helper (no bundle is built);
//! - the options-validation test runs `TripoSgRunOptions::validate` directly;
//! - the smoke test runs preprocess + DINOv2 + mesh extraction (baseline
//!   latents, octree depth 5/5) + OBJ export. A full CPU denoise is NOT run
//!   here (50 CPU DiT steps would take far too long; parity denoise lives in
//!   tests/runtime_triposg.rs and T6).

use std::{
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use candle_core::{DType, Device, Tensor};
use lux3d_core::{
    ModelFamily,
    error::Lux3dError,
    runtime::{TripoSgPipeline, TripoSgRunOptions},
    test_support::{GpuTestLock, model_asset_options, runtime_root},
};

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

fn load_baseline_tensor(path: &PathBuf, name: &str, device: &Device) -> Tensor {
    let tensor = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(path)
            .unwrap_or_else(|error| panic!("smoke baseline safetensors: {error}"))
    }
    .load(name, device)
    .unwrap_or_else(|error| panic!("baseline {name}: {error}"));
    tensor.to_dtype(DType::F32).expect("baseline f32")
}

// ---------------------------------------------------------------------------
// 1. Seeded RNG determinism (no weights touched)
// ---------------------------------------------------------------------------

/// The latent initializer must be a pure function of the seed: same seed ->
/// identical tensors, seed + 1 -> different tensors, all values finite and
/// the shape exactly `[1, num_tokens, channels]`.
#[test]
fn rng_is_deterministic_and_seeded() {
    let device = Device::Cpu;

    let first =
        TripoSgPipeline::sample_latents(42, 2048, 64, &device).expect("seeded latents (seed 42)");
    assert_eq!(first.dims(), [1, 2048, 64], "latent shape must be exact");

    let first_values = first
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(first_values.len(), 2048 * 64);
    assert!(
        first_values.iter().all(|value| value.is_finite()),
        "seeded latents must be finite"
    );

    let repeat = TripoSgPipeline::sample_latents(42, 2048, 64, &device)
        .expect("seeded latents (seed 42, again)")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(
        first_values, repeat,
        "same seed must reproduce identical latents"
    );

    let shifted = TripoSgPipeline::sample_latents(43, 2048, 64, &device)
        .expect("seeded latents (seed 43)")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert_ne!(
        first_values, shifted,
        "different seeds must produce different latents"
    );

    // Rough distribution sanity: standard-normal Box-Muller output over 131072
    // samples must stay well inside +-5 sigma and have a small |mean|.
    let max_abs = first_values.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    let mean = first_values.iter().sum::<f32>() / first_values.len() as f32;
    println!(
        "seeded latents: n={}, max |v| {max_abs:.3}, mean {mean:.3e}",
        first_values.len()
    );
    assert!(
        max_abs < 5.5,
        "implausible normal sample: max |v| {max_abs}"
    );
    assert!(mean.abs() < 0.05, "implausible normal mean: {mean:.3e}");
}

// ---------------------------------------------------------------------------
// 2. Pipeline smoke: preprocess -> embeds -> baseline latents -> mesh -> OBJ
// ---------------------------------------------------------------------------

/// Load once, run the CPU smoke chain from the baseline latents (a full CPU
/// denoise is out of scope here; T6 covers end-to-end on accelerators).
#[test]
fn pipeline_mesh_and_export_smoke() {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );

    let load_started = Instant::now();
    let pipeline = TripoSgPipeline::load(model_asset_options(ModelFamily::TripoSg))
        .expect("TripoSG pipeline load");
    println!(
        "pipeline load took {:?} (canonical package hashing dominates)",
        load_started.elapsed()
    );

    // Preprocess + DINOv2 conditioning.
    let prepared = pipeline
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

    let embed_started = Instant::now();
    let image_embeds = pipeline
        .image_embeds(&prepared.pixel_values)
        .expect("TripoSG image embeds");
    let embed_elapsed = embed_started.elapsed();
    assert_eq!(image_embeds.dims(), [1, 257, 1024]);
    let embed_values = image_embeds
        .flatten_all()
        .expect("embed flatten")
        .to_vec1::<f32>()
        .expect("embed vec");
    assert!(
        embed_values.iter().all(|value| value.is_finite()),
        "image embeds must be finite"
    );
    println!(
        "image embeds took {embed_elapsed:?}, max |v| {:.3}",
        embed_values.iter().fold(0.0_f32, |m, v| m.max(v.abs()))
    );

    // Baseline latents stand in for the denoise output (CPU denoise skipped).
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    assert_eq!(latents.dims(), [1, 2048, 64]);

    // NOTE: the brief suggested octree depth 4, but the smoke object decodes
    // into a thin closed shell (y extent ~0.009 around y=-0.4397): at depth 4
    // (node spacing 0.1256) NO grid node falls inside the shell, so the SDF
    // has no sign change and marching cubes yields an empty mesh. Depth 5
    // (spacing 0.0628, node layer y=-0.4398 inside the shell — the same depth
    // the baseline was generated at) is the cheapest setting that produces a
    // non-empty surface, so the smoke runs 5/5 with the brief's chunk size.
    let options = TripoSgRunOptions {
        dense_octree_depth: 5,
        hierarchical_octree_depth: 5,
        // The smoke exercises the non-flash depth-5 path; the reference
        // default is flash depth 9 and is covered by the dedicated flash tests.
        use_flash_decoder: false,
        query_chunk_size: 4096,
        ..Default::default()
    };

    let mesh_started = Instant::now();
    let mesh = pipeline
        .extract_mesh(&latents, &options)
        .expect("mesh extraction");
    let mesh_elapsed = mesh_started.elapsed();

    let vertex_dims = mesh.vertices.dims().to_vec();
    let face_dims = mesh.faces.dims().to_vec();
    assert_eq!(vertex_dims.len(), 2, "vertices must be rank 2");
    assert_eq!(vertex_dims[1], 3, "vertices must be [N, 3]");
    assert_eq!(face_dims.len(), 2, "faces must be rank 2");
    assert_eq!(face_dims[1], 3, "faces must be [F, 3]");
    let vertex_count = vertex_dims[0];
    let face_count = face_dims[0];
    assert!(vertex_count > 0 && face_count > 0, "mesh must be non-empty");
    assert_eq!(mesh.vertices.dtype(), DType::F32, "vertices must be F32");
    assert_eq!(mesh.faces.dtype(), DType::I64, "faces must be I64");
    assert!(
        matches!(mesh.vertices.device(), Device::Cpu),
        "buffers must live on the latents device"
    );

    let vertex_values = mesh
        .vertices
        .flatten_all()
        .expect("vertex flatten")
        .to_vec1::<f32>()
        .expect("vertex vec");
    assert!(
        vertex_values.iter().all(|value| value.is_finite()),
        "mesh vertices must be finite"
    );
    let face_values = mesh
        .faces
        .flatten_all()
        .expect("face flatten")
        .to_vec1::<i64>()
        .expect("face vec");
    assert!(
        face_values
            .iter()
            .all(|index| (0..vertex_count as i64).contains(index)),
        "face indices must be zero-based and in range"
    );

    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    for vertex in vertex_values.chunks_exact(3) {
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(vertex[axis]);
            bbox_max[axis] = bbox_max[axis].max(vertex[axis]);
        }
    }
    println!(
        "mesh extraction took {mesh_elapsed:?}: vertices {vertex_count}, faces {face_count}, \
         bbox_min {bbox_min:?}, bbox_max {bbox_max:?}"
    );

    // Exercise the reference-default flash branch end-to-end at the cheapest
    // reference depth whose 64^3 dense grid still intersects this smoke
    // object's thin shell (depth 5 misses it; depth 6 yields 26,624 vertices
    // in the reference dump).
    let flash_options = TripoSgRunOptions {
        use_flash_decoder: true,
        flash_octree_depth: 6,
        query_chunk_size: 4096,
        ..Default::default()
    };
    let flash_started = Instant::now();
    let flash_mesh = pipeline
        .extract_mesh(&latents, &flash_options)
        .expect("flash mesh extraction");
    let flash_vertex_count = flash_mesh.vertices.dims()[0];
    let flash_face_count = flash_mesh.faces.dims()[0];
    assert!(
        flash_vertex_count > 0 && flash_face_count > 0,
        "flash mesh must be non-empty"
    );
    assert_eq!(flash_mesh.vertices.dtype(), DType::F32);
    assert_eq!(flash_mesh.faces.dtype(), DType::I64);
    let flash_vertices = flash_mesh
        .vertices
        .flatten_all()
        .expect("flash vertex flatten")
        .to_vec1::<f32>()
        .expect("flash vertex vec");
    assert!(
        flash_vertices.iter().all(|value| value.is_finite()),
        "flash mesh vertices must be finite"
    );
    assert_eq!(
        flash_mesh.vertices.dims()[0],
        flash_vertices.len() / 3,
        "flash mesh vertex buffer shape"
    );
    println!(
        "flash mesh extraction took {:?}: vertices {flash_vertex_count}, faces {flash_face_count}",
        flash_started.elapsed()
    );

    // Export and verify the OBJ text.
    let export_started = Instant::now();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let out_dir = std::env::temp_dir().join(format!(
        "lux3d-triposg-task4-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).expect("temp export dir");
    let obj_path = out_dir.join("mesh.obj");
    pipeline.export_obj(&mesh, &obj_path).expect("obj export");
    let export_elapsed = export_started.elapsed();

    let body = std::fs::read(&obj_path).expect("obj read");
    assert!(!body.is_empty(), "OBJ file must be non-empty");
    let body = String::from_utf8(body).expect("obj must be utf-8");
    assert!(body.contains("v "), "OBJ must contain vertex lines");
    assert!(body.contains("f "), "OBJ must contain face lines");
    let v_lines = body.lines().filter(|line| line.starts_with("v ")).count();
    let f_lines = body.lines().filter(|line| line.starts_with("f ")).count();
    assert_eq!(v_lines, vertex_count, "vertex line count mismatch");
    assert_eq!(f_lines, face_count, "face line count mismatch");
    println!(
        "obj export took {export_elapsed:?}: {} bytes at {}",
        body.len(),
        obj_path.display()
    );

    let _ = std::fs::remove_dir_all(&out_dir);
}

// ---------------------------------------------------------------------------
// 3. Run-option validation (no weights touched)
// ---------------------------------------------------------------------------

/// Zero steps / zero tokens / octree depth 11 / non-finite guidance must all
/// be rejected with messages naming the offending field.
#[test]
fn options_are_validated() {
    let cases = [
        (
            TripoSgRunOptions {
                steps: 0,
                ..Default::default()
            },
            "steps",
        ),
        (
            TripoSgRunOptions {
                num_tokens: 0,
                ..Default::default()
            },
            "num_tokens",
        ),
        (
            TripoSgRunOptions {
                dense_octree_depth: 11,
                ..Default::default()
            },
            "dense_octree_depth",
        ),
        (
            TripoSgRunOptions {
                hierarchical_octree_depth: 11,
                ..Default::default()
            },
            "hierarchical_octree_depth",
        ),
        (
            TripoSgRunOptions {
                flash_octree_depth: 0,
                ..Default::default()
            },
            "flash_octree_depth",
        ),
        (
            TripoSgRunOptions {
                flash_octree_depth: 11,
                ..Default::default()
            },
            "flash_octree_depth",
        ),
        (
            TripoSgRunOptions {
                guidance_scale: f32::NAN,
                ..Default::default()
            },
            "guidance_scale",
        ),
    ];

    for (options, field) in cases {
        let error = options
            .validate()
            .expect_err(&format!("{field} must be rejected"));
        match error {
            Lux3dError::CanonicalWeightsValidation { message } => {
                assert!(
                    message.contains(field),
                    "validation message `{message}` must mention `{field}`"
                );
                println!("rejected {field}: {message}");
            }
            other => panic!("unexpected error kind for {field}: {other:?}"),
        }
    }

    // Defaults themselves are valid.
    TripoSgRunOptions::default()
        .validate()
        .expect("defaults valid");
}

// ---------------------------------------------------------------------------
// 4. Golden pipeline end-to-end (Task 6, CUDA, orchestrator-run)
// ---------------------------------------------------------------------------

/// Full end-to-end golden run: `TripoSgPipeline::infer_from_path` with the
/// golden defaults (50 steps, guidance 7.0, seed 42, 2048 tokens, dense 8 /
/// hierarchical 9) on CUDA, then mesh extraction.
///
/// Gating (never silently skipped):
/// - `LUX3D_TRIPOSG_GOLDEN_PIPELINE=1` must be set (the 50-step run costs
///   minutes on CUDA and is pointless before the baseline exists);
/// - the golden artifact must exist;
/// - a CUDA device must be available (50 CPU DiT steps at the measured
///   ~183 s/step dev-profile cost would take > 2 h).
///
/// Recommended orchestrator command:
/// ```text
/// LUX3D_TRIPOSG_GOLDEN_PIPELINE=1 cargo test -p lux3d-core --test pipeline_triposg \
///   -- --test-threads=1 triposg_golden_pipeline_from_path --nocapture
/// ```
///
/// Parity modes:
/// - default: public seeded path (`infer_from_path`). The image embeds ARE
///   comparable (<= 2e-3); the latents are NOT numerically comparable to the
///   baseline because the public seeded sampler is intentionally not
///   cross-framework identical (plan scope decision 3) — the delta is printed
///   informationally only, and the mesh must be a sane non-empty surface.
/// - `LUX3D_TRIPOSG_GOLDEN_INJECT_LATENTS=1`: parity mode — the baseline
///   `latents_init` is injected through `infer_with_latents`. The 50-step CFG
///   ODE is chaotic, so the final latents use statistical bounds
///   (mean <= 2e-2, max <= 4.0); step-1/step-10 tight parity lives in the
///   runtime golden test.
#[test]
fn triposg_golden_pipeline_from_path_cuda() {
    let label = "TripoSG golden pipeline (infer_from_path, CUDA)";
    let baseline_path = golden_baseline_path();
    let golden_pipeline_requested = std::env::var("LUX3D_TRIPOSG_GOLDEN_PIPELINE")
        .map(|value| value == "1")
        .unwrap_or(false);
    if !golden_pipeline_requested {
        eprintln!(
            "SKIP {label}: set LUX3D_TRIPOSG_GOLDEN_PIPELINE=1 to run the 50-step end-to-end \
             golden pipeline (CUDA, minutes)"
        );
        return;
    }
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

    let inject_latents = std::env::var("LUX3D_TRIPOSG_GOLDEN_INJECT_LATENTS")
        .map(|value| value == "1")
        .unwrap_or(false);

    let load_started = Instant::now();
    let pipeline = TripoSgPipeline::load(model_asset_options(ModelFamily::TripoSg))
        .expect("TripoSG pipeline load");
    println!("pipeline load took {:?}", load_started.elapsed());

    // Golden defaults must be exactly the reference run options.
    let options = TripoSgRunOptions::default();
    assert_eq!(
        options,
        TripoSgRunOptions {
            steps: 50,
            guidance_scale: 7.0,
            seed: 42,
            num_tokens: 2048,
            dense_octree_depth: 8,
            hierarchical_octree_depth: 9,
            use_flash_decoder: true,
            flash_octree_depth: 9,
            query_chunk_size: 16384,
        },
        "golden defaults must stay 50/7.0/42/2048/8/9/flash-9/16384"
    );

    let expected_final =
        load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    let expected_final_device = expected_final
        .to_device(&device)
        .expect("latents to device");

    let (latents, latents_parity_asserted) = if inject_latents {
        // Parity mode: preprocess + DINOv2, then denoise the BASELINE init
        // latents (RNG excluded from parity per plan scope decision 3).
        let prepared = pipeline
            .prepare_inputs_from_path(
                &repo_root().join("test-assets").join("mesh-input.png"),
                &device,
            )
            .expect("TripoSG prepared inputs");
        let image_embeds = pipeline
            .image_embeds(&prepared.pixel_values)
            .expect("TripoSG image embeds");
        assert_eq!(image_embeds.dims(), [1, 257, 1024]);
        let expected_embeds =
            load_baseline_tensor(&baseline_path, "triposg.image_embeds", &Device::Cpu);
        let (max_delta, mean_delta) = max_abs_delta_stats(&expected_embeds, &image_embeds);
        println!(
            "golden image_embeds parity: max abs delta {max_delta:.3e}, mean abs delta {mean_delta:.3e} \
             (tolerance 2e-3)"
        );
        assert!(
            max_delta <= 2e-3,
            "golden image_embeds max abs delta too large: {max_delta:.3e}"
        );

        let latents_init = load_baseline_tensor(&baseline_path, "triposg.latents_init", &device);
        assert_eq!(latents_init.dims(), [1, 2048, 64]);
        let started = Instant::now();
        let latents = pipeline
            .infer_with_latents(&image_embeds, &latents_init, &options)
            .expect("TripoSG infer_with_latents (baseline latents_init)");
        println!("50-step denoise took {:?}", started.elapsed());
        (latents, true)
    } else {
        // Public seeded path: full infer_from_path.
        let started = Instant::now();
        let inference = pipeline
            .infer_from_path(
                &repo_root().join("test-assets").join("mesh-input.png"),
                &device,
                &options,
            )
            .expect("TripoSG infer_from_path (golden defaults)");
        println!("infer_from_path took {:?}", started.elapsed());
        assert_eq!(inference.latents_init.dims(), [1, 2048, 64]);
        (inference.latents, false)
    };

    assert_eq!(latents.dims(), [1, 2048, 64]);
    let (max_delta, mean_delta) = max_abs_delta_stats(&expected_final_device, &latents);
    if latents_parity_asserted {
        println!(
            "golden latents_step_final parity (injected baseline latents_init): max abs delta \
             {max_delta:.3e}, mean abs delta {mean_delta:.3e} (statistical tolerance: \
             mean <= 2e-2, max <= 4.0)"
        );
        // The 50-step CFG ODE is chaotic (steps 1/10 match to 1e-6; the
        // reference itself drifts ~8% relative L2 across GPU/torch builds),
        // so the final latent is judged statistically; see the runtime golden
        // test for the per-step checks and the measured control experiment.
        assert!(
            mean_delta <= 2e-2,
            "golden latents_step_final mean abs delta too large: {mean_delta:.3e} \
             (max {max_delta:.3e})"
        );
        assert!(
            max_delta <= 4.0,
            "golden latents_step_final max abs delta outside the measured chaotic bound: \
             {max_delta:.3e}"
        );
    } else {
        // Informational only: the public seeded sampler is intentionally not
        // cross-framework identical (plan scope decision 3).
        println!(
            "golden latents_step_final delta (seeded public path, informational only — \
             cross-framework RNG parity is excluded by plan scope decision 3): max abs delta \
             {max_delta:.3e}, mean abs delta {mean_delta:.3e}"
        );
    }

    // Mesh extraction on the golden grid settings must yield a sane surface.
    let mesh_started = Instant::now();
    let mesh = pipeline
        .extract_mesh(&latents, &options)
        .expect("golden mesh extraction");
    println!("mesh extraction took {:?}", mesh_started.elapsed());

    let vertex_dims = mesh.vertices.dims().to_vec();
    let face_dims = mesh.faces.dims().to_vec();
    assert_eq!(vertex_dims.len(), 2, "vertices must be rank 2");
    assert_eq!(vertex_dims[1], 3, "vertices must be [N, 3]");
    assert_eq!(face_dims.len(), 2, "faces must be rank 2");
    assert_eq!(face_dims[1], 3, "faces must be [F, 3]");
    let vertex_count = vertex_dims[0];
    let face_count = face_dims[0];
    assert!(
        vertex_count > 0 && face_count > 0,
        "golden mesh must be non-empty (vertices {vertex_count}, faces {face_count})"
    );
    assert_eq!(mesh.vertices.dtype(), DType::F32, "vertices must be F32");
    assert_eq!(mesh.faces.dtype(), DType::I64, "faces must be I64");

    let vertex_values = mesh
        .vertices
        .to_device(&Device::Cpu)
        .expect("vertices to cpu")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert!(
        vertex_values.iter().all(|value| value.is_finite()),
        "golden mesh vertices must be finite"
    );
    let mut bbox_min = [f32::INFINITY; 3];
    let mut bbox_max = [f32::NEG_INFINITY; 3];
    for vertex in vertex_values.chunks_exact(3) {
        for axis in 0..3 {
            bbox_min[axis] = bbox_min[axis].min(vertex[axis]);
            bbox_max[axis] = bbox_max[axis].max(vertex[axis]);
        }
    }
    for axis in 0..3 {
        assert!(
            (-1.005..=1.005).contains(&bbox_min[axis])
                && (-1.005..=1.005).contains(&bbox_max[axis]),
            "golden mesh bbox must stay inside the canonical bounds (axis {axis})"
        );
    }
    println!(
        "golden mesh: vertices {vertex_count}, faces {face_count}, bbox_min {bbox_min:?}, \
         bbox_max {bbox_max:?}"
    );

    // Presence tap: the golden artifact must carry the mesh buffers.
    let baseline_vertices =
        load_baseline_tensor(&baseline_path, "triposg.mesh_vertices", &Device::Cpu);
    assert_eq!(baseline_vertices.dims().len(), 2);
    assert_eq!(
        baseline_vertices.dims()[1],
        3,
        "baseline mesh vertices [N, 3]"
    );
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
