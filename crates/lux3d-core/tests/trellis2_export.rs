//! TRELLIS.2 export stage tests (T7): the PBR voxel sidecar round-trip,
//! payload rejection, the JSON layout descriptor, and the per-format vertex
//! color behavior (OBJ positions-only; PLY/GLB carry the sampled colors).

use std::path::PathBuf;

use lux3d_core::contracts::{ExportStage, ModelFamily, Trellis2Mesh};
use lux3d_core::export::{MeshFormat, Trellis2ExportStage, pbrvox};
use lux3d_core::geometry::Trellis2MeshBuffers;

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "lux3d-trellis2-export-{name}-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");
    dir
}

/// A tiny textured mesh: two voxels with distinct 6-channel PBR attributes
/// on an 8^3 grid, 4 vertices, 2 triangles, sampled vertex colors.
fn textured_buffers() -> Trellis2MeshBuffers {
    let coords = vec![[0i32, 0, 0], [1, 0, 0]];
    let mut voxel_coords = Vec::new();
    for coord in &coords {
        voxel_coords.extend_from_slice(coord);
    }
    // Row 0: red plastic; row 1: dark metal.
    let voxel_attrs = vec![
        0.9_f32, 0.1, 0.1, 0.0, 0.6, 1.0, // base_color, metallic, roughness, alpha
        0.2, 0.2, 0.25, 0.8, 0.3, 1.0,
    ];
    Trellis2MeshBuffers {
        resolution: 8,
        vertices: vec![
            -0.5, -0.5, -0.5, // vertex 0
            0.0, -0.5, -0.5, // vertex 1
            0.0, 0.0, -0.5, // vertex 2
            -0.5, 0.0, -0.5, // vertex 3
        ],
        faces: vec![0, 1, 2, 0, 2, 3],
        voxel_coords,
        voxel_attrs,
        voxel_attr_channels: pbrvox::CHANNELS,
        // Four vertices: left edge blends the two voxels, right edge is the
        // second voxel.
        vertex_colors: Some(vec![
            0.5, 0.15, 0.1, // vertex 0: 0.5 (0.9+0.2), 0.5 (0.1+0.2), 0.5 (0.1+0.25)
            0.5, 0.15, 0.175, // vertex 1
            0.2, 0.2, 0.25, // vertex 2: inside voxel 1
            0.2, 0.2, 0.25, // vertex 3
        ]),
    }
}

/// The binary sidecar: header fields and every record round-trip exactly
/// (3 packed i32 coords + 6 f32 attributes, little-endian).
#[test]
fn pbrvox_bin_round_trip() {
    let stage = Trellis2ExportStage;
    let mesh = textured_buffers();
    let dir = temp_dir("roundtrip");
    let path = dir.join("mesh.pbrvox.bin");
    stage.write_pbrvox_bin(&mesh, &path).expect("write bin");

    let bytes = std::fs::read(&path).expect("read bin");
    assert_eq!(&bytes[..8], &pbrvox::MAGIC, "magic L3DPBR1\\0");
    let version = u32::from_le_bytes(bytes[8..12].try_into().expect("version"));
    assert_eq!(version, pbrvox::VERSION);
    let resolution = u32::from_le_bytes(bytes[12..16].try_into().expect("resolution"));
    assert_eq!(resolution, mesh.resolution);
    let count = u64::from_le_bytes(bytes[16..24].try_into().expect("count"));
    assert_eq!(count, mesh.voxel_count() as u64);
    assert_eq!(
        bytes.len(),
        24 + (count as usize) * (3 * 4 + 6 * 4),
        "record size 3 i32 + 6 f32"
    );

    let mut offset = 24usize;
    for voxel in 0..mesh.voxel_count() {
        for axis in 0..3 {
            let coord = i32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("coord"));
            assert_eq!(coord, mesh.voxel_coords[voxel * 3 + axis]);
            offset += 4;
        }
        for channel in 0..pbrvox::CHANNELS {
            let value = f32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("attr"));
            assert_eq!(
                value.to_bits(),
                mesh.voxel_attrs[voxel * pbrvox::CHANNELS + channel].to_bits(),
                "voxel {voxel} channel {channel} bit-exact"
            );
            offset += 4;
        }
    }
    assert_eq!(offset, bytes.len(), "no trailing bytes");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Bad payloads are rejected: wrong channel count, coord/attr count
/// mismatch, malformed vertex colors.
#[test]
fn pbr_sidecar_rejects_bad_payloads() {
    let stage = Trellis2ExportStage;
    let dir = temp_dir("reject");

    // 7-channel shape attrs: no PBR sidecar.
    let mut mesh = textured_buffers();
    mesh.voxel_attrs = vec![0.5_f32; 2 * 7];
    mesh.voxel_attr_channels = 7;
    assert!(stage.write_pbrvox_bin(&mesh, &dir.join("a.bin")).is_err());
    assert!(stage.write_pbr_json(&mesh, &dir.join("a.json")).is_err());

    // Coord count mismatch: one coord row for two attribute rows.
    let mut mesh = textured_buffers();
    mesh.voxel_coords = vec![0, 0, 0];
    assert!(stage.write_pbrvox_bin(&mesh, &dir.join("b.bin")).is_err());
    assert!(stage.write_pbr_json(&mesh, &dir.join("b.json")).is_err());

    // Malformed vertex colors: not three per vertex.
    let mut mesh = textured_buffers();
    mesh.vertex_colors = Some(vec![0.0_f32; 5]);
    assert!(mesh.validate().is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The JSON sidecar: parseable, describes format/version/resolution/count,
/// the PBR layout and one range per channel covering the payload values.
#[test]
fn pbr_json_content() {
    let stage = Trellis2ExportStage;
    let mesh = textured_buffers();
    let dir = temp_dir("json");
    let path = dir.join("mesh.pbr.json");
    stage.write_pbr_json(&mesh, &path).expect("write json");

    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("read json")).expect("parse json");
    assert_eq!(json["format"], "pbrvox");
    assert_eq!(json["version"], pbrvox::VERSION);
    assert_eq!(json["resolution"], mesh.resolution);
    assert_eq!(json["voxel_count"], mesh.voxel_count());
    assert_eq!(json["channels"], pbrvox::CHANNELS);
    assert_eq!(json["layout"]["base_color"], serde_json::json!([0, 3]));
    assert_eq!(json["layout"]["metallic"], 3);
    assert_eq!(json["layout"]["roughness"], 4);
    assert_eq!(json["layout"]["alpha"], 5);

    // Range channels: base_color r in [0.2, 0.9]; metallic in [0.0, 0.8].
    let ranges = json["ranges"].as_array().expect("ranges");
    assert_eq!(ranges.len(), pbrvox::CHANNELS);
    assert_eq!(ranges[0], serde_json::json!([0.2_f64, 0.9_f64]));
    assert_eq!(ranges[3], serde_json::json!([0.0_f64, 0.8_f64]));
    let _ = std::fs::remove_dir_all(&dir);
}

/// Per-format color behavior: OBJ stays positions-only, PLY carries uchar
/// RGB, GLB carries a float VEC3 COLOR_0 accessor; without colors no color
/// surface appears.
#[test]
fn vertex_colors_per_format() {
    let stage = Trellis2ExportStage;
    let colored = textured_buffers();
    let mut plain = colored.clone();
    plain.vertex_colors = None;
    let dir = temp_dir("formats");

    // OBJ: positions only, even with colors present.
    let obj = dir.join("mesh.obj");
    assert!(matches!(
        stage.write_mesh(&colored, &obj).expect("obj"),
        MeshFormat::Obj
    ));
    let obj_text = std::fs::read_to_string(&obj).expect("obj text");
    assert!(obj_text.contains("v -0.5"), "positions present");
    assert!(!obj_text.contains("0.9 0.1 0.1"), "no color payload in OBJ");

    // PLY: uchar red/green/blue properties with the sampled colors.
    let ply = dir.join("mesh.ply");
    assert!(matches!(
        stage.write_mesh(&colored, &ply).expect("ply"),
        MeshFormat::Ply
    ));
    let ply_bytes = std::fs::read(&ply).expect("ply bytes");
    let header_end = ply_bytes
        .windows(10)
        .position(|w| w == b"end_header")
        .expect("end_header");
    let header = String::from_utf8_lossy(&ply_bytes[..header_end]).to_string();
    assert!(header.contains("property uchar red"));
    assert!(header.contains("property uchar green"));
    assert!(header.contains("property uchar blue"));
    // Plain mesh: no color properties.
    let ply_plain = dir.join("plain.ply");
    stage.write_mesh(&plain, &ply_plain).expect("ply plain");
    let plain_header =
        String::from_utf8_lossy(&std::fs::read(&ply_plain).expect("plain bytes")[..header_end])
            .to_string();
    assert!(!plain_header.contains("property uchar red"));

    // GLB: COLOR_0 accessor on the textured mesh, absent on the plain one.
    let glb = dir.join("mesh.glb");
    assert!(matches!(
        stage.write_mesh(&colored, &glb).expect("glb"),
        MeshFormat::Glb
    ));
    let glb_bytes = std::fs::read(&glb).expect("glb bytes");
    let json_len = u32::from_le_bytes(glb_bytes[12..16].try_into().expect("len")) as usize;
    let glb_json: serde_json::Value =
        serde_json::from_slice(&glb_bytes[20..20 + json_len]).expect("glb json");
    let primitive = &glb_json["meshes"][0]["primitives"][0];
    assert_eq!(primitive["attributes"]["COLOR_0"], 1, "COLOR_0 accessor");
    let accessor = &glb_json["accessors"][1];
    assert_eq!(accessor["type"], "VEC3");
    assert_eq!(accessor["componentType"], 5126, "float component type");

    let glb_plain = dir.join("plain.glb");
    stage.write_mesh(&plain, &glb_plain).expect("glb plain");
    let plain_bytes = std::fs::read(&glb_plain).expect("plain glb");
    let plain_len = u32::from_le_bytes(plain_bytes[12..16].try_into().expect("len")) as usize;
    let plain_json: serde_json::Value =
        serde_json::from_slice(&plain_bytes[20..20 + plain_len]).expect("plain json");
    assert!(
        plain_json["meshes"][0]["primitives"][0]["attributes"]
            .get("COLOR_0")
            .is_none(),
        "no COLOR_0 without colors"
    );

    // write_obj keeps the positions-only contract explicitly.
    stage
        .write_obj(&colored, &dir.join("explicit.obj"))
        .expect("explicit obj");
    let explicit = std::fs::read_to_string(dir.join("explicit.obj")).expect("explicit");
    assert!(!explicit.contains("0.9 0.1 0.1"));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `write_mesh_with_pbr` writes the mesh plus the sidecar pair; the mesh
/// contract conversion ignores (but tolerates) the texture products.
#[test]
fn write_mesh_with_pbr_and_contract() {
    let stage = Trellis2ExportStage;
    let mesh = textured_buffers();
    let dir = temp_dir("withpbr");
    let path = dir.join("mesh.glb");
    let sidecar = stage.write_mesh_with_pbr(&mesh, &path).expect("with pbr");
    assert!(path.is_file());
    assert!(sidecar.is_file());
    assert!(sidecar.ends_with("mesh.pbrvox.bin"));
    let json_path = dir.join("mesh.pbr.json");
    assert!(json_path.is_file());

    // Geometry-only mesh (7 channels): the mesh writes fine, the sidecar
    // rejects the payload.
    let mut plain = mesh.clone();
    plain.voxel_attrs = vec![0.5_f32; 2 * 7];
    plain.voxel_attr_channels = 7;
    plain.vertex_colors = None;
    let plain_path = dir.join("plain.obj");
    assert!(matches!(
        stage.write_mesh(&plain, &plain_path).expect("plain mesh"),
        MeshFormat::Obj
    ));
    assert!(
        stage
            .write_mesh_with_pbr(&plain, &dir.join("plain2.obj"))
            .is_err()
    );

    // Contract conversion validates and keeps the voxel payload.
    let contract: Trellis2Mesh = mesh.to_contract().expect("contract");
    assert_eq!(contract.family, ModelFamily::Trellis2);
    assert_eq!(contract.resolution, 8);
    assert_eq!(contract.voxel_attrs.len(), 12);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The export plan documents the texture artifacts.
#[test]
fn export_plan_mentions_colors_and_sidecar() {
    let plan = Trellis2ExportStage
        .export(&Trellis2Mesh {
            family: ModelFamily::Trellis2,
            vertices: vec![0.0; 9],
            faces: vec![0, 1, 2],
            voxel_coords: vec![0, 0, 0],
            voxel_attrs: vec![0.0; 6],
            resolution: 8,
        })
        .expect("plan");
    assert_eq!(plan.primary_extension, "glb");
    let notes = plan.utility_notes.join("\n");
    assert!(notes.contains("vertex colors"));
    assert!(notes.contains("pbrvox.bin"));
}
