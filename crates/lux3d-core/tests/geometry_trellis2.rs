//! TRELLIS.2 geometry buffer + export smoke tests: `Trellis2MeshBuffers`
//! validation, contract conversion, and OBJ/PLY/GLB/STL writes through
//! `Trellis2ExportStage`.

use lux3d_core::contracts::{ExportStage, ModelFamily, Trellis2Mesh};
use lux3d_core::export::{MeshFormat, Trellis2ExportStage};
use lux3d_core::geometry::{Trellis2GeometryStage, Trellis2MeshBuffers};

/// Closed 2x2x2 cube extracted through the real dual-grid extractor: 8
/// dual vertices, 12 triangles, all vertices used.
fn cube_buffers() -> Trellis2MeshBuffers {
    let mut coords = Vec::new();
    for x in 0..2i32 {
        for y in 0..2 {
            for z in 0..2 {
                coords.push([x, y, z]);
            }
        }
    }
    let mesh = lux3d_core::runtime::flexible_dual_grid_to_mesh(
        &coords,
        &vec![[0.1_f32; 3]; coords.len()],
        &vec![[true, true, true]; coords.len()],
        &vec![1.0; coords.len()],
        [2, 2, 2],
        [[-0.5; 3], [0.5; 3]],
    );
    let voxel_coords = coords.iter().flat_map(|c| c.iter().copied()).collect();
    let voxel_attrs = vec![0.5_f32; coords.len() * 7];
    Trellis2GeometryStage
        .assemble(512, mesh.vertices, mesh.faces, voxel_coords, voxel_attrs, 7)
        .expect("assemble")
}

#[test]
fn buffers_counts_bbox_and_validation() {
    let buffers = cube_buffers();
    assert_eq!(buffers.resolution, 512);
    assert_eq!(buffers.vertex_count(), 8);
    assert_eq!(buffers.face_count(), 12);
    assert_eq!(buffers.voxel_count(), 8);
    let (min, max) = buffers.bbox().expect("bbox");
    for axis in 0..3 {
        assert!(min[axis] > -0.5 && min[axis] < 0.0);
        assert!(max[axis] > 0.0 && max[axis] < 0.5);
    }
    assert!(!buffers.has_non_finite_vertex());
    buffers.validate().expect("validate");

    // Empty mesh: no bbox, validation still fine.
    let empty = Trellis2MeshBuffers::default();
    assert!(empty.bbox().is_none());
    assert_eq!(empty.voxel_count(), 0);
    empty.validate().expect("empty validates");
}

#[test]
fn validation_rejects_broken_buffers() {
    let base = cube_buffers();

    let mut broken = base.clone();
    broken.vertices.push(0.0);
    assert!(broken.validate().is_err(), "vertex len not multiple of 3");

    let mut broken = base.clone();
    broken.faces.push(0);
    assert!(broken.validate().is_err(), "face len not multiple of 3");

    let mut broken = base.clone();
    broken.vertices[0] = f32::NAN;
    assert!(broken.validate().is_err(), "NaN vertex");
    assert!(broken.has_non_finite_vertex());

    let mut broken = base.clone();
    broken.faces[0] = 8; // vertex_count == 8
    assert!(broken.validate().is_err(), "face index out of range");
    let mut broken = base.clone();
    broken.faces[0] = -1;
    assert!(broken.validate().is_err(), "negative face index");

    let mut broken = base.clone();
    broken.voxel_attrs.push(0.0);
    assert!(broken.validate().is_err(), "attrs len not divisible");

    let mut broken = base.clone();
    broken.voxel_coords.push(0);
    assert!(
        broken.validate().is_err(),
        "voxel coords len not multiple of 3"
    );
}

#[test]
fn to_contract_roundtrip() {
    let buffers = cube_buffers();
    let contract: Trellis2Mesh = buffers.to_contract().expect("to_contract");
    assert_eq!(contract.family, ModelFamily::Trellis2);
    assert_eq!(contract.resolution, 512);
    assert_eq!(contract.vertices, buffers.vertices);
    assert_eq!(contract.faces, buffers.faces);
    assert_eq!(contract.voxel_coords, buffers.voxel_coords);
    assert_eq!(contract.voxel_attrs, buffers.voxel_attrs);
}

#[test]
fn export_smoke_obj_ply_glb_stl() {
    let stage = Trellis2ExportStage;
    let buffers = cube_buffers();
    let pid = std::process::id();
    let mut paths = Vec::new();

    for (extension, expected) in [
        ("obj", MeshFormat::Obj),
        ("ply", MeshFormat::Ply),
        ("glb", MeshFormat::Glb),
        ("stl", MeshFormat::Stl),
    ] {
        let path = std::env::temp_dir().join(format!("lux3d-trellis2-geom-{pid}.{extension}"));
        let format = stage.write_mesh(&buffers, &path).expect(extension);
        assert_eq!(format, expected);
        let bytes = std::fs::read(&path).expect("read back");
        assert!(!bytes.is_empty(), "{extension} output empty");
        paths.push((path, bytes));
    }

    let (obj_path, obj_bytes) = &paths[0];
    let obj_text = String::from_utf8(obj_bytes.clone()).expect("obj ascii");
    let v_count = obj_text.lines().filter(|l| l.starts_with("v ")).count();
    let f_count = obj_text.lines().filter(|l| l.starts_with("f ")).count();
    assert_eq!(v_count, 8, "OBJ vertex lines");
    assert_eq!(f_count, 12, "OBJ face lines");
    // 1-based face indices.
    assert!(obj_text.contains("f 1 "), "first face references vertex 1");

    let (_, ply_bytes) = &paths[1];
    let ply_text = String::from_utf8_lossy(ply_bytes);
    assert!(ply_text.starts_with("ply"));
    assert!(ply_text.contains("element face 12"));

    let (_, glb_bytes) = &paths[2];
    assert_eq!(&glb_bytes[0..4], b"glTF", "GLB magic");

    let _ = obj_path;
    for (path, _) in &paths {
        let _ = std::fs::remove_file(path);
    }
}

#[test]
fn export_plan_descriptor() {
    let contract = cube_buffers().to_contract().expect("contract");
    let plan = Trellis2ExportStage.export(&contract).expect("plan");
    assert_eq!(plan.primary_extension, "glb");
    assert!(plan.alternate_extensions.contains(&"obj".to_string()));
    assert!(plan.alternate_extensions.contains(&"ply".to_string()));
}
