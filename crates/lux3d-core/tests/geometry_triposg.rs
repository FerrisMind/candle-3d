//! TripoSG marching-cubes mesh parity against the python-baseline smoke
//! sample (CPU, dense depth 5 == hierarchical depth 5, no refinement).
//!
//! The reference decodes into an fp16 volume before
//! `measure.marching_cubes(..., level=0, method="lewiner")`, so this test
//! rounds the decoded F32 grid through F16 and back before assembly (the
//! reference mesh path). Topology (vertex/face counts) is NOT expected to
//! match skimage's lewiner tables; the assertions cover non-emptiness,
//! finiteness, per-component bbox (<= 2e-2) and absolute volume (<= 5%).

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use lux3d_core::ModelFamily;
use lux3d_core::geometry::TripoSgGeometryStage;
use lux3d_core::runtime::{
    TripoSgSdfGrid, TripoSgVaeConfig, TripoSgVaeDecoder, hierarchical_sdf_grid,
};
use lux3d_core::test_support::{GpuTestLock, model_asset_options, runtime_root};
use lux3d_core::weights::load_canonical_weights;

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

fn load_triposg_vae_decoder_cpu() -> (TripoSgVaeDecoder, TripoSgVaeConfig) {
    load_triposg_vae_decoder_on(&Device::Cpu)
}

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
    let vb = candle_nn::VarBuilder::from_tensors(vae_tensors, DType::F32, device);
    let decoder = TripoSgVaeDecoder::load(vb.pp("vae"), &cfg).expect("TripoSG VAE decoder load");
    (decoder, cfg)
}

/// Build the smoke dense grid (depth 5, dense == hierarchical -> no
/// refinement) with the canonical bounds.
fn build_smoke_grid() -> TripoSgSdfGrid {
    let baseline_path = smoke_baseline_path();
    assert!(
        baseline_path.is_file(),
        "TripoSG smoke baseline artifact is missing: {} (generate it with \
         tools/python_baseline/triposg_dump.py)",
        baseline_path.display()
    );
    let (decoder, _cfg) = load_triposg_vae_decoder_cpu();
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &Device::Cpu);
    let bounds = [-1.005_f32, -1.005, -1.005, 1.005, 1.005, 1.005];
    hierarchical_sdf_grid(&decoder, &latents, bounds, 5, 5, 16384).expect("TripoSG dense SDF grid")
}

/// Absolute enclosed volume via the signed tetrahedron sum (same formula for
/// both meshes; winding-consistent output of marching cubes makes `abs`
/// valid).
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

#[test]
fn triposg_mesh_matches_smoke_baseline_cpu() {
    let baseline_path = smoke_baseline_path();

    // Reference mesh path: the grid values go through fp16 before marching
    // cubes (the reference decodes into fp16 volumes; the baseline dense_sdf
    // tap is the pre-cast F32, but the MESH tap came from the fp16 grid).
    let grid = build_smoke_grid();
    let rounded_values = Tensor::from_vec(grid.values.clone(), (grid.values.len(),), &Device::Cpu)
        .expect("grid values tensor")
        .to_dtype(DType::F16)
        .expect("grid f16 cast")
        .to_dtype(DType::F32)
        .expect("grid f32 cast")
        .to_vec1::<f32>()
        .expect("grid vec");
    let grid_fp16 = TripoSgSdfGrid {
        depth: grid.depth,
        dims: grid.dims,
        bounds: grid.bounds,
        values: rounded_values,
    };

    let stage = TripoSgGeometryStage;
    let started = Instant::now();
    let mesh = stage
        .assemble_cpu(&grid_fp16)
        .expect("TripoSG mesh assemble");
    let elapsed = started.elapsed();

    assert_eq!(mesh.resolution, 32, "smoke mesh extracts at depth 5");
    assert_eq!(mesh.sdf_level, 0.0);
    assert!(
        !mesh.vertices.is_empty() && !mesh.faces.is_empty(),
        "assembled mesh must be non-empty (vertices {}, faces {})",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3
    );
    assert!(
        mesh.vertices.iter().all(|value| value.is_finite()),
        "mesh vertices must be finite"
    );
    assert!(
        mesh.faces
            .iter()
            .all(|index| (0..(mesh.vertices.len() / 3) as i64).contains(index)),
        "mesh faces must be zero-based and in range"
    );

    // Baseline mesh (faces stored as f32 -> round to i64; 0-based).
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
    let mut max_bbox_delta = 0.0_f32;
    for axis in 0..3 {
        max_bbox_delta = max_bbox_delta.max((our_min[axis] - base_min[axis]).abs());
        max_bbox_delta = max_bbox_delta.max((our_max[axis] - base_max[axis]).abs());
    }

    let our_volume = signed_volume(&mesh.vertices, &mesh.faces);
    let baseline_volume = signed_volume(&baseline_vertices, &baseline_faces);
    let volume_ratio = if baseline_volume > 0.0 {
        (our_volume - baseline_volume).abs() / baseline_volume
    } else {
        f64::INFINITY
    };

    println!(
        "mesh: ours vertices={} faces={} bbox_min={our_min:?} bbox_max={our_max:?} volume={our_volume:.6}; \
         baseline vertices={} faces={} bbox_min={base_min:?} bbox_max={base_max:?} volume={baseline_volume:.6}; \
         max bbox delta {max_bbox_delta:.3e} (tolerance 2e-2), relative volume delta {volume_ratio:.3e} \
         (tolerance 5e-2), assemble took {elapsed:?}",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
        baseline_vertices.len() / 3,
        baseline_faces.len() / 3,
    );

    assert!(
        max_bbox_delta <= 2e-2,
        "mesh bbox delta too large: {max_bbox_delta:.3e}"
    );
    assert!(
        volume_ratio <= 5e-2,
        "mesh volume delta too large: {volume_ratio:.3e}"
    );
}

// ---------------------------------------------------------------------------
// Task 6: golden mesh parity (dense depth 8 -> hierarchical depth 9 grid,
// reference fp16 mesh path). The golden artifacts are generated by the
// orchestrator (CUDA F32); this case prints an explicit SKIP while they are
// absent. Because the depth-9 grid build is a heavy job (CPU: multi-hour;
// CUDA ~15-30 min), it additionally requires LUX3D_TRIPOSG_GOLDEN_RUN=1 and
// picks the device with LUX3D_TRIPOSG_GOLDEN_DEVICE (cpu|cuda, default cpu):
//
// ```text
// LUX3D_TRIPOSG_GOLDEN_RUN=1 LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda \
//   cargo test -p lux3d-core --test geometry_triposg -- --test-threads=1 \
//   triposg_golden_mesh_matches_baseline --nocapture
// ```
//
// The DEFAULT culling path (LUX3D_TRIPOSG_REFINE_ALL unset) is under test. If
// the mesh deviates beyond the bounds below, re-run with
// LUX3D_TRIPOSG_REFINE_ALL=1 (reference-exact full-band queries) and classify
// whether the deviation is culling-related — do not change runtime code.
// ---------------------------------------------------------------------------

fn golden_env_flag(name: &str) -> bool {
    std::env::var(name)
        .map(|value| value == "1")
        .unwrap_or(false)
}

#[test]
fn triposg_golden_mesh_matches_baseline_cpu() {
    let label = "TripoSG golden mesh parity";
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
            "SKIP {label}: set LUX3D_TRIPOSG_GOLDEN_RUN=1 to run (CPU: multi-hour for the 512^3 \
             grid; CUDA ~15-30 min via LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda)"
        );
        return;
    }
    let requested = std::env::var("LUX3D_TRIPOSG_GOLDEN_DEVICE").unwrap_or_default();
    let (device, _guard) = if requested.eq_ignore_ascii_case("cuda") {
        let _guard = match GpuTestLock::acquire() {
            Ok(guard) => guard,
            Err(error) => {
                eprintln!("SKIP {label}: GPU test lock unavailable: {error}");
                return;
            }
        };
        match Device::new_cuda(0) {
            Ok(device) => (device, Some(_guard)),
            Err(error) => {
                eprintln!(
                    "SKIP {label}: LUX3D_TRIPOSG_GOLDEN_DEVICE=cuda but no CUDA device ({error})"
                );
                return;
            }
        }
    } else {
        (Device::Cpu, None)
    };

    let (decoder, _cfg) = load_triposg_vae_decoder_on(&device);
    let latents = load_baseline_tensor(&baseline_path, "triposg.latents_step_final", &device);
    assert_eq!(latents.dims(), [1, 2048, 64]);
    let bounds = [-1.005_f32, -1.005, -1.005, 1.005, 1.005, 1.005];

    let started = Instant::now();
    let grid = hierarchical_sdf_grid(&decoder, &latents, bounds, 8, 9, 16384)
        .expect("TripoSG hierarchical depth-9 SDF grid");
    let grid_elapsed = started.elapsed();
    assert_eq!(grid.depth, 9, "golden mesh extracts from the depth-9 grid");
    assert_eq!(grid.dims, 512);

    // Reference mesh path: the grid values go through fp16 before marching
    // cubes (the reference decodes into an fp16 volume).
    let rounded_values = Tensor::from_vec(grid.values.clone(), (grid.values.len(),), &Device::Cpu)
        .expect("grid values tensor")
        .to_dtype(DType::F16)
        .expect("grid f16 cast")
        .to_dtype(DType::F32)
        .expect("grid f32 cast")
        .to_vec1::<f32>()
        .expect("grid vec");
    let grid_fp16 = TripoSgSdfGrid {
        depth: grid.depth,
        dims: grid.dims,
        bounds: grid.bounds,
        values: rounded_values,
    };

    let stage = TripoSgGeometryStage;
    let assemble_started = Instant::now();
    let mesh = stage
        .assemble_cpu(&grid_fp16)
        .expect("TripoSG mesh assemble");
    let assemble_elapsed = assemble_started.elapsed();

    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    assert!(
        vertex_count > 0 && face_count > 0,
        "golden mesh must be non-empty (vertices {vertex_count}, faces {face_count})"
    );
    assert!(
        mesh.vertices.iter().all(|value| value.is_finite()),
        "golden mesh vertices must be finite"
    );
    assert!(
        mesh.faces
            .iter()
            .all(|index| (0..vertex_count as i64).contains(index)),
        "golden mesh faces must be zero-based and in range"
    );

    // Baseline mesh taps (faces stored as f32 -> round to i64; 0-based).
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
    let baseline_vertex_count = baseline_vertices.len() / 3;
    let baseline_face_count = baseline_faces.len() / 3;

    let (our_min, our_max) = bbox(&mesh.vertices);
    let (base_min, base_max) = bbox(&baseline_vertices);
    let mut max_bbox_delta = 0.0_f32;
    for axis in 0..3 {
        max_bbox_delta = max_bbox_delta.max((our_min[axis] - base_min[axis]).abs());
        max_bbox_delta = max_bbox_delta.max((our_max[axis] - base_max[axis]).abs());
    }

    let our_volume = signed_volume(&mesh.vertices, &mesh.faces);
    let baseline_volume = signed_volume(&baseline_vertices, &baseline_faces);
    let volume_ratio = if baseline_volume > 0.0 {
        (our_volume - baseline_volume).abs() / baseline_volume
    } else {
        f64::INFINITY
    };
    let face_count_ratio = if baseline_face_count > 0 {
        (face_count.abs_diff(baseline_face_count)) as f64 / baseline_face_count as f64
    } else {
        f64::INFINITY
    };

    let refine_all = golden_env_flag("LUX3D_TRIPOSG_REFINE_ALL");
    println!(
        "golden mesh: ours vertices={vertex_count} faces={face_count} bbox_min={our_min:?} \
         bbox_max={our_max:?} volume={our_volume:.6}; baseline vertices={baseline_vertex_count} \
         faces={baseline_face_count} bbox_min={base_min:?} bbox_max={base_max:?} \
         volume={baseline_volume:.6}; max bbox delta {max_bbox_delta:.3e} (tolerance 2e-2), \
         relative volume delta {volume_ratio:.3e} (tolerance 1e-2), face count ratio \
         {face_count_ratio:.3e} (tolerance 1e-2); refine_all={refine_all}; grid {grid_elapsed:?}, \
         assemble+compare {assemble_elapsed:?}"
    );

    assert!(
        max_bbox_delta <= 2e-2,
        "golden mesh bbox delta too large: {max_bbox_delta:.3e} — if this persists, re-run with \
         LUX3D_TRIPOSG_REFINE_ALL=1 (reference-exact full band) and classify before touching \
         runtime code"
    );
    assert!(
        volume_ratio <= 1e-2,
        "golden mesh volume delta too large: {volume_ratio:.3e} — if this persists, re-run with \
         LUX3D_TRIPOSG_REFINE_ALL=1 and classify"
    );
    assert!(
        face_count_ratio <= 1e-2,
        "golden mesh face count differs beyond 1%: ours {face_count} vs baseline \
         {baseline_face_count} — if this persists, re-run with LUX3D_TRIPOSG_REFINE_ALL=1 and \
         classify (mcubes vs skimage-lewiner topology noise)"
    );
}
