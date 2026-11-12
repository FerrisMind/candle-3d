//! TripoSG flash decoder / clean-room DiffDMC parity against the reference
//! Python baseline.
//!
//! The baseline (`tools/python_baseline/triposg_flash_dump.py`) runs the
//! unmodified reference `flash_extract_geometry` with Diso's CUDA `DiffDMC`
//! and captures the dense depth-6 grid passed to DMC plus the grid-space and
//! world-space mesh.

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::ModelFamily;
use lux3d_core::runtime::{
    TripoSgDmcMesh, TripoSgVaeConfig, TripoSgVaeDecoder, compact_dmc_mesh,
    flash_dual_marching_cubes, flash_resolutions, flash_sdf_grid,
};
use lux3d_core::test_support::{model_asset_options, runtime_root};
use lux3d_core::weights::load_canonical_weights;

fn baseline_path() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg_flash")
        .join("depth6")
        .join("artifacts.safetensors")
}

fn baseline_tensor(name: &str, device: &Device) -> Tensor {
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(baseline_path())
            .unwrap_or_else(|error| panic!("flash baseline safetensors: {error}"))
    }
    .load(name, device)
    .unwrap_or_else(|error| panic!("baseline {name}: {error}"))
    .to_dtype(DType::F32)
    .expect("baseline f32")
}

fn load_decoder() -> TripoSgVaeDecoder {
    load_decoder_on(&Device::Cpu)
}

fn load_decoder_on(device: &Device) -> TripoSgVaeDecoder {
    let weights = load_canonical_weights(
        ModelFamily::TripoSg,
        model_asset_options(ModelFamily::TripoSg),
    )
    .expect("TripoSG canonical weights");
    let resolved = weights
        .read_resolved_config_json()
        .expect("resolved_config.json");
    let cfg = TripoSgVaeConfig::from_resolved_config(&resolved).expect("TripoSG VAE config");
    let archive = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&weights.canonical_file)
            .expect("canonical safetensors")
    };
    let mut tensors = HashMap::new();
    for (name, _view) in archive.tensors() {
        if name.starts_with("vae.post_quant.") || name.starts_with("vae.decoder.") {
            let tensor = archive
                .load(&name, device)
                .unwrap_or_else(|error| panic!("load {name}: {error}"))
                .to_dtype(DType::F32)
                .expect("vae weights f32");
            tensors.insert(name, tensor);
        }
    }
    let vb = candle_nn::VarBuilder::from_tensors(tensors, DType::F32, device);
    TripoSgVaeDecoder::load(vb.pp("vae"), &cfg).expect("VAE decoder")
}

fn max_abs_delta(actual: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .filter(|(a, e)| a.is_finite() && e.is_finite())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f32, f32::max)
}

#[test]
fn flash_resolution_schedule_matches_reference() {
    // Reference `flash_extract_geometry` schedule, `min_resolution=63`,
    // `mini_grid_num=4`. Verified from the shipped Python implementation.
    assert_eq!(flash_resolutions(5), vec![31]);
    assert_eq!(flash_resolutions(6), vec![63]);
    assert_eq!(flash_resolutions(7), vec![63, 126]);
    assert_eq!(flash_resolutions(8), vec![63, 126, 252]);
    assert_eq!(flash_resolutions(9), vec![63, 126, 252, 504]);
    assert_eq!(flash_resolutions(10), vec![63, 126, 252, 504, 1008]);
}

#[test]
fn flash_sdf_grid_matches_reference() {
    let depth = 6u32;
    let dims = 64usize;
    let latents = baseline_tensor("flash.latents", &Device::Cpu);
    let expected = baseline_tensor("flash.dmc_input_sdf", &Device::Cpu)
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec")
        // The reference divides the decoder logits by the mutated
        // `octree_resolution` variable (32 after the schedule loop) and
        // negates them before DMC.
        .into_iter()
        .map(|value| -value * 32.0)
        .collect::<Vec<_>>();
    let decoder = load_decoder();
    let grid = flash_sdf_grid(
        &decoder,
        &latents,
        [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        depth,
        16384,
    )
    .expect("flash sdf grid");
    assert_eq!(grid.dims, dims);
    assert_eq!(grid.values.len(), dims * dims * dims);
    let delta = max_abs_delta(&grid.values, &expected);
    println!("flash grid max delta: {delta:.6e}");
    assert!(delta <= 5e-3, "flash grid delta {delta}");
}

#[test]
fn flash_dmc_synthetic_oblique() {
    let path = runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg_flash")
        .join("synthetic_oblique8.safetensors");
    let archive =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.expect("oblique");
    let sdf = archive
        .load("sdf_ref", &Device::Cpu)
        .expect("sdf")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    let expected = archive
        .load("verts", &Device::Cpu)
        .expect("verts")
        .to_vec2::<f32>()
        .expect("verts");
    let grid = lux3d_core::runtime::TripoSgFlashGrid {
        depth: 0,
        dims: 8,
        bounds: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        values: sdf.iter().map(|value| -*value).collect(),
    };
    let mesh = flash_dual_marching_cubes(&grid).expect("dmc");
    println!(
        "oblique rust {} verts {} faces; reference {} verts {} faces",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
        expected.len(),
        0
    );
    let mut rust = mesh
        .vertices
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    let mut reference = expected.clone();
    rust.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    reference.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    let delta = rust
        .iter()
        .zip(&reference)
        .map(|(a, b)| {
            (a[0] - b[0])
                .abs()
                .max((a[1] - b[1]).abs())
                .max((a[2] - b[2]).abs())
        })
        .fold(0.0_f32, f32::max);
    println!("oblique sorted vertex max delta: {delta:.6e}");
    for (a, b) in rust.iter().zip(&reference).take(8) {
        println!("  rust {a:?} ref {b:?}");
    }
}

fn depth_path(depth: u32) -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg_flash")
        .join(format!("depth{depth}"))
        .join("artifacts.safetensors")
}

#[test]
#[ignore = "needs the depth-7 reference dump; heavy CPU decoder run"]
fn flash_depth7_matches_reference() {
    check_flash_depth(7, 127, 10_138);
}

#[test]
#[ignore = "needs the depth-9 reference dump; 505^3 grid and full DMC, several minutes"]
fn flash_depth9_matches_reference() {
    check_flash_depth(9, 505, 279_090);
}

#[test]
#[ignore = "needs the depth-9 reference dump; 505^3 grid and full DMC on CUDA"]
fn flash_depth9_grid_matches_reference_cuda() {
    let device = match Device::new_cuda(0) {
        Ok(device) => device,
        Err(error) => panic!("CUDA device required for this test: {error}"),
    };
    check_flash_depth_on(9, 505, 279_090, &device);
}

fn check_flash_depth(depth: u32, dims: usize, expected_processed_vertices: usize) {
    check_flash_depth_on(depth, dims, expected_processed_vertices, &Device::Cpu);
}

fn check_flash_depth_on(
    depth: u32,
    dims: usize,
    expected_processed_vertices: usize,
    device: &Device,
) {
    let path = depth_path(depth);
    assert!(
        path.is_file(),
        "missing flash depth-{depth} baseline: {}",
        path.display()
    );
    let archive =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.expect("flash baseline");
    let latents = archive
        .load("flash.latents", &Device::Cpu)
        .expect("latents")
        .to_device(device)
        .expect("latents to test device");
    let expected = archive
        .load("flash.dmc_input_sdf", &Device::Cpu)
        .expect("sdf")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(expected.len(), dims * dims * dims);
    let decoder = load_decoder_on(device);
    let grid = flash_sdf_grid(
        &decoder,
        &latents,
        [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        depth,
        16384,
    )
    .expect("flash sdf grid");
    assert_eq!(grid.dims, dims);

    // Reference stores `-logits / 32`; compare raw logits back in the same
    // scale. Unqueried refinement cells are NaN on both sides.
    let mut finite = 0usize;
    let mut mask_mismatch = 0usize;
    let mut delta = 0.0f32;
    for (actual, expected) in grid.values.iter().zip(&expected) {
        let expected = -expected * 32.0;
        match (actual.is_finite(), expected.is_finite()) {
            (true, true) => {
                finite += 1;
                delta = delta.max((actual - expected).abs());
            }
            (false, false) => {}
            _ => mask_mismatch += 1,
        }
    }
    println!(
        "flash depth {depth}: {finite} finite cells, mask mismatches {mask_mismatch}, max delta {delta:.6e}"
    );
    assert_eq!(mask_mismatch, 0, "flash depth {depth} NaN mask mismatch");
    assert!(delta <= 5e-3, "flash depth {depth} grid delta {delta}");

    let mesh = flash_dual_marching_cubes(&grid).expect("flash dmc");
    check_dmc_against_reference(depth, &mesh, &archive, expected_processed_vertices);
}

/// Validate only the DMC stage against a reference dump by feeding the
/// reference SDF back through the Rust extractor. This avoids re-running the
/// expensive 505^3 VAE grid decode when only mesh/topology parity is in doubt.
#[test]
#[ignore = "needs the depth-9 reference dump; 505^3 DMC only"]
fn flash_depth9_dmc_matches_reference() {
    let depth = 9u32;
    let dims = 505usize;
    let path = depth_path(depth);
    assert!(
        path.is_file(),
        "missing flash depth-{depth} baseline: {}",
        path.display()
    );
    let archive =
        unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.expect("flash baseline");
    let reference_sdf = archive
        .load("flash.dmc_input_sdf", &Device::Cpu)
        .expect("sdf")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(reference_sdf.len(), dims * dims * dims);
    let grid = lux3d_core::runtime::TripoSgFlashGrid {
        depth,
        dims,
        bounds: [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        // `flash_dual_marching_cubes` divides raw logits by 2^min(depth,5);
        // the reference stores `-logits / 32`.
        values: reference_sdf.iter().map(|value| -*value * 32.0).collect(),
    };
    let mesh = flash_dual_marching_cubes(&grid).expect("flash dmc");
    check_dmc_against_reference(depth, &mesh, &archive, 279_090);
}

fn check_dmc_against_reference(
    depth: u32,
    mesh: &TripoSgDmcMesh,
    archive: &candle_core::safetensors::MmapedSafetensors,
    expected_processed_vertices: usize,
) {
    let ref_vertices = archive
        .load("flash.dmc_vertices_grid", &Device::Cpu)
        .expect("vertices")
        .to_vec2::<f32>()
        .expect("vertices");
    let ref_faces = archive
        .load("flash.dmc_faces_grid", &Device::Cpu)
        .expect("faces")
        .to_vec2::<f32>()
        .expect("faces");
    println!(
        "flash depth {depth}: rust {} verts {} faces, reference {} verts {} faces",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
        ref_vertices.len(),
        ref_faces.len()
    );
    assert!(
        (mesh.vertices.len() / 3).abs_diff(ref_vertices.len())
            <= (ref_vertices.len() / 1000).max(64),
        "flash depth {depth}: raw vertex count {} vs reference {}",
        mesh.vertices.len() / 3,
        ref_vertices.len()
    );

    // The reference pipeline runs `trimesh.Trimesh(..., process=True)`, which
    // merges vertices within 1e-8 and keeps only the face-referenced subset.
    // `expected_processed_vertices` is the concrete count produced by trimesh
    // 5.2.0 on the reference raw DMC output.
    let compact = compact_dmc_mesh(mesh);
    let mut referenced = std::collections::BTreeSet::new();
    for face in &ref_faces {
        for index in face {
            referenced.insert(*index as i64);
        }
    }
    println!(
        "flash depth {depth}: compacted {} verts ({} raw referenced), reference trimesh {}",
        compact.vertices.len() / 3,
        referenced.len(),
        expected_processed_vertices
    );
    assert!(
        referenced.len() >= expected_processed_vertices,
        "reference trimesh cannot add vertices"
    );
    assert!(
        (compact.vertices.len() / 3).abs_diff(expected_processed_vertices)
            <= (expected_processed_vertices / 10_000).max(16),
        "flash depth {depth}: compacted {} verts vs reference trimesh {}",
        compact.vertices.len() / 3,
        expected_processed_vertices
    );
    let compact_face_delta = compact.faces.len().abs_diff(mesh.faces.len());
    assert!(
        compact_face_delta <= (mesh.faces.len() / 10_000).max(8),
        "flash depth {depth}: compaction removed {compact_face_delta} faces"
    );
    // Ambiguous checkerboard cells are resolved with isomesh's separate rule
    // instead of Diso's CC BY-NC `problematicConfigs` table, which can add or
    // remove a handful of quads (depth 7: 4 of 20256 faces).
    let face_delta = (mesh.faces.len() / 3).abs_diff(ref_faces.len());
    let face_tolerance = (ref_faces.len() / 1000).max(64);
    assert!(
        face_delta <= face_tolerance,
        "flash depth {depth} face count delta {face_delta} > {face_tolerance}"
    );
    let scale = 2.01f32 / (1u32 << depth) as f32;
    let rust_vertices = mesh
        .vertices
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    let expected_vertices = ref_vertices
        .into_iter()
        .map(|row| {
            [
                row[0] * scale - 1.005,
                row[1] * scale - 1.005,
                row[2] * scale - 1.005,
            ]
        })
        .collect::<Vec<_>>();
    let key = |v: &[f32; 3]| {
        [
            (v[0] * 20_000.0).round() as i64,
            (v[1] * 20_000.0).round() as i64,
            (v[2] * 20_000.0).round() as i64,
        ]
    };
    let mut counts: HashMap<[i64; 3], i64> = HashMap::new();
    for v in &expected_vertices {
        *counts.entry(key(v)).or_default() += 1;
    }
    let mut matched = 0usize;
    for v in &rust_vertices {
        if let Some(slot) = counts.get_mut(&key(v)) {
            if *slot > 0 {
                *slot -= 1;
                matched += 1;
            }
        }
    }
    let fraction = matched as f64 / rust_vertices.len() as f64;
    println!(
        "flash depth {depth}: vertex multiset match {matched}/{} ({:.3}%)",
        rust_vertices.len(),
        100.0 * fraction
    );
    assert!(
        fraction >= 0.999,
        "flash depth {depth} vertex match {fraction}"
    );
}

#[test]
fn flash_dmc_synthetic_nan_boundary() {
    let path = runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg_flash")
        .join("synthetic_nan9.safetensors");
    let archive = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }.expect("nan");
    let sdf = archive
        .load("sdf_ref", &Device::Cpu)
        .expect("sdf")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    let expected = archive
        .load("verts", &Device::Cpu)
        .expect("verts")
        .to_vec2::<f32>()
        .expect("verts");
    let grid = lux3d_core::runtime::TripoSgFlashGrid {
        depth: 0,
        dims: 9,
        bounds: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        values: sdf.iter().map(|value| -*value).collect(),
    };
    let mesh = flash_dual_marching_cubes(&grid).expect("dmc");
    assert_eq!(mesh.vertices.len() / 3, expected.len());
    let mut rust = mesh
        .vertices
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    let mut reference = expected.clone();
    rust.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    reference.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    let mut delta = 0.0f32;
    for (row, reference) in rust.iter().zip(&reference) {
        for axis in 0..3 {
            delta = delta.max((row[axis] - reference[axis]).abs());
        }
    }
    println!("synthetic NaN boundary vertex max delta: {delta:.6e}");
    assert!(delta <= 1e-5, "NaN boundary delta {delta}");
}

#[test]
fn flash_compact_dmc_mesh_matches_trimesh_processing() {
    let mesh = TripoSgDmcMesh {
        vertices: vec![
            0.0,
            0.0,
            0.0, // 0
            1.0,
            0.0,
            0.0, // 1
            0.0,
            1.0,
            0.0, // 2
            0.0,
            0.0,
            0.0, // 3: exact duplicate of vertex 0
            9.0,
            9.0,
            9.0, // 4: unreferenced
            f32::NAN,
            0.0,
            0.0, // 5: non-finite vertex
            1.0e-9,
            -1.0e-9,
            0.0, // 6: distinct last-ulp vertex (trimesh tolerance is not used)
        ],
        faces: vec![
            0, 1, 2, // kept
            3, 1, 2, // kept, reuses merged vertex 0
            6, 1, 2, // kept, remains distinct
            0, 0, 1, // degenerate after merge
            0, 5, 2, // non-finite
        ],
    };
    let compact = compact_dmc_mesh(&mesh);
    assert_eq!(
        compact.vertices,
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0e-9, -1.0e-9, 0.0]
        ]
        .concat()
    );
    assert_eq!(compact.faces, vec![0, 1, 2, 0, 1, 2, 3, 1, 2]);
}

#[test]
fn flash_dmc_synthetic_plane() {
    let path = runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg_flash")
        .join("synthetic_plane4.safetensors");
    let sdf = unsafe { candle_core::safetensors::MmapedSafetensors::new(&path) }
        .expect("synthetic")
        .load("sdf_ref", &Device::Cpu)
        .expect("sdf")
        .flatten_all()
        .expect("flat")
        .to_vec1::<f32>()
        .expect("vec");
    let grid = lux3d_core::runtime::TripoSgFlashGrid {
        depth: 0,
        dims: 4,
        bounds: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        values: sdf.iter().map(|value| -*value).collect(),
    };
    let mesh = flash_dual_marching_cubes(&grid).expect("dmc");
    println!("synthetic rust verts {}", mesh.vertices.len() / 3);
    for row in mesh.vertices.chunks_exact(3).take(20) {
        println!("  {row:?}");
    }
    println!("synthetic rust faces {}", mesh.faces.len() / 3);
}

#[test]
fn flash_dmc_matches_reference() {
    let dims = 64usize;
    let sdf = baseline_tensor("flash.dmc_input_sdf", &Device::Cpu)
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    // Feed the exact reference SDF so this test isolates DMC topology and
    // vertex averaging from decoder numeric drift (covered separately).
    let grid = lux3d_core::runtime::TripoSgFlashGrid {
        depth: 6,
        dims,
        bounds: [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        values: sdf.iter().map(|value| -*value * 32.0).collect(),
    };
    let mesh = flash_dual_marching_cubes(&grid).expect("flash dmc");

    let ref_vertices = baseline_tensor("flash.dmc_vertices_grid", &Device::Cpu)
        .to_vec2::<f32>()
        .expect("vertices")
        .into_iter()
        .map(|row| {
            // Grid space -> world, same as reference world mapping.
            [
                row[0] / 64.0 * 2.01 - 1.005,
                row[1] / 64.0 * 2.01 - 1.005,
                row[2] / 64.0 * 2.01 - 1.005,
            ]
        })
        .collect::<Vec<_>>();
    let ref_faces = baseline_tensor("flash.dmc_faces_grid", &Device::Cpu)
        .to_vec2::<f32>()
        .expect("faces");
    println!(
        "flash dmc: rust {} verts {} faces, reference {} verts {} faces",
        mesh.vertices.len() / 3,
        mesh.faces.len() / 3,
        ref_vertices.len(),
        ref_faces.len()
    );
    assert_eq!(mesh.vertices.len() / 3, ref_vertices.len());
    assert_eq!(mesh.faces.len() / 3, ref_faces.len());

    let rust_vertices = mesh
        .vertices
        .chunks_exact(3)
        .map(|row| [row[0], row[1], row[2]])
        .collect::<Vec<_>>();
    let expected_vertices = ref_vertices.clone();

    let bbox = |verts: &[[f32; 3]]| {
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in verts {
            for axis in 0..3 {
                lo[axis] = lo[axis].min(v[axis]);
                hi[axis] = hi[axis].max(v[axis]);
            }
        }
        (lo, hi)
    };
    let (rust_lo, rust_hi) = bbox(&rust_vertices);
    let (ref_lo, ref_hi) = bbox(&expected_vertices);
    println!("flash dmc rust bbox: {rust_lo:?} .. {rust_hi:?}");
    println!("flash dmc ref  bbox: {ref_lo:?} .. {ref_hi:?}");
    let bbox_delta = rust_lo
        .iter()
        .chain(&rust_hi)
        .zip(ref_lo.iter().chain(&ref_hi))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(bbox_delta <= 1e-4, "dmc bbox delta {bbox_delta}");

    // The clean-room DMC resolves ambiguous checkerboard faces with the
    // isomesh "separate" rule instead of Diso's `problematicConfigs` table
    // (CC BY-NC data we deliberately do not copy). Interior vertices are
    // bit-identical; only a handful of ambiguous cells may connect their
    // MC cycles differently. Compare position multisets, not insertion order.
    let multiset_match = |actual: &[[f32; 3]], expected: &[[f32; 3]], scale: f32| {
        let key = |v: &[f32; 3]| {
            [
                (v[0] * scale).round() as i64,
                (v[1] * scale).round() as i64,
                (v[2] * scale).round() as i64,
            ]
        };
        let mut counts: HashMap<[i64; 3], i64> = HashMap::new();
        for v in expected {
            *counts.entry(key(v)).or_default() += 1;
        }
        let mut matched = 0usize;
        for v in actual {
            if let Some(slot) = counts.get_mut(&key(v)) {
                if *slot > 0 {
                    *slot -= 1;
                    matched += 1;
                }
            }
        }
        matched
    };
    let vertex_matched = multiset_match(&rust_vertices, &expected_vertices, 20_000.0);
    let vertex_fraction = vertex_matched as f64 / rust_vertices.len() as f64;
    println!(
        "flash dmc vertex multiset match: {vertex_matched}/{} ({:.3}%)",
        rust_vertices.len(),
        100.0 * vertex_fraction
    );
    assert!(
        vertex_fraction >= 0.999,
        "dmc vertex match {vertex_fraction}"
    );

    let rust_centroids = face_centroids(&mesh.vertices, &mesh.faces);
    let expected_centroids = face_centroids(
        &ref_vertices.iter().flatten().copied().collect::<Vec<_>>(),
        &ref_faces
            .iter()
            .flatten()
            .map(|value| *value as i64)
            .collect::<Vec<_>>(),
    );
    let centroid_matched = multiset_match(&rust_centroids, &expected_centroids, 1_000.0);
    let centroid_fraction = centroid_matched as f64 / rust_centroids.len() as f64;
    println!(
        "flash dmc triangle-centroid multiset match: {centroid_matched}/{} ({:.3}%)",
        rust_centroids.len(),
        100.0 * centroid_fraction
    );
    // Diso picks quad diagonals with CUDA f32 `F.normalize`; near-coplanar
    // quads can flip the diagonal on rounding alone. Both choices are valid
    // triangulations of the same quad, so accept the small residual after the
    // exact topology/count/bbox checks above.
    assert!(
        centroid_fraction >= 0.95,
        "dmc centroid match {centroid_fraction}"
    );
}

fn face_centroids(vertices: &[f32], faces: &[i64]) -> Vec<[f32; 3]> {
    faces
        .chunks_exact(3)
        .map(|face| {
            let mut centroid = [0.0f32; 3];
            for &index in face {
                let base = index as usize * 3;
                centroid[0] += vertices[base] / 3.0;
                centroid[1] += vertices[base + 1] / 3.0;
                centroid[2] += vertices[base + 2] / 3.0;
            }
            centroid
        })
        .collect()
}
