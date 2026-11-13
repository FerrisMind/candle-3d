//! Shared triangle-mesh export writers.
//!
//! Meshes are format-agnostic `vertices + faces` buffers; the output format is
//! selected from the destination extension. GLB is the primary interchange
//! format, while OBJ stays available for debugging and PLY/STL cover tools that
//! do not read glTF.
//!
//! GLB is written as a minimal glTF 2.0 binary container: one mesh, one
//! triangle primitive, `POSITION`, optional `COLOR_0`, and 32-bit indices.
//! Normals are omitted; glTF clients compute flat normals when they are absent.

use std::{fmt::Write as _, path::Path};

use serde_json::json;

use crate::{Result, error::Lux3dError};

/// Supported triangle-mesh artifact formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    /// Binary glTF 2.0 (`.glb`), the primary mesh artifact.
    Glb,
    /// ASCII Wavefront OBJ (`.obj`), kept for debugging.
    Obj,
    /// ASCII PLY with a triangle face element (`.ply`).
    Ply,
    /// Binary STL (`.stl`); vertex colors are not representable.
    Stl,
}

impl MeshFormat {
    /// Resolve a mesh format from a case-insensitive file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        Self::from_extension(extension).map_err(|_| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "unsupported mesh output extension `{extension}` for `{}`; use .glb, .obj, .ply or .stl",
                path.display()
            ),
        })
    }

    /// Resolve a mesh format from a bare extension such as `glb`.
    pub fn from_extension(extension: &str) -> Result<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "glb" => Ok(Self::Glb),
            "obj" => Ok(Self::Obj),
            "ply" => Ok(Self::Ply),
            "stl" => Ok(Self::Stl),
            other => Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "unsupported mesh output extension `{other}`; use .glb, .obj, .ply or .stl"
                ),
            }),
        }
    }

    /// Canonical lowercase extension.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Glb => "glb",
            Self::Obj => "obj",
            Self::Ply => "ply",
            Self::Stl => "stl",
        }
    }

    /// HTTP content type used by the server download route.
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Glb => "model/gltf-binary",
            Self::Obj => "model/obj",
            Self::Ply => "model/ply",
            Self::Stl => "model/stl",
        }
    }
}

/// Borrowed CPU mesh buffers.
#[derive(Debug, Clone, Copy)]
pub struct MeshView<'a> {
    /// Flat `[x, y, z, ...]` vertex positions.
    pub vertices: &'a [f32],
    /// Flat zero-based triangle indices.
    pub faces: &'a [i64],
    /// Optional flat `[r, g, b, ...]` vertex colors in `[0, 1]`.
    pub colors: Option<&'a [f32]>,
}

/// Legacy OBJ formatting knobs used by the TripoSG/TripoSR writers.
#[derive(Debug, Clone, Copy)]
pub struct ObjStyle {
    /// Exact header line, or `None` for a bare OBJ.
    pub header: Option<&'static str>,
    /// Fixed decimal places per coordinate, or `None` for `Display`.
    pub decimals: Option<usize>,
    /// Append `r g b` to each `v` line when colors are available.
    pub vertex_colors: bool,
}

impl ObjStyle {
    /// TripoSG style: trimesh header, 8 decimal places, positions only.
    pub const TRIPOSG: Self = Self {
        header: Some("# https://github.com/mikedh/trimesh"),
        decimals: Some(8),
        vertex_colors: false,
    };

    /// TripoSR style: trimesh header, default float formatting, vertex colors.
    pub const TRIPOSR: Self = Self {
        header: Some("# https://github.com/mikedh/trimesh"),
        decimals: None,
        vertex_colors: true,
    };
}

/// Write a triangle mesh, selecting the format from `path`'s extension.
pub fn write_mesh(mesh: &MeshView<'_>, path: &Path, obj_style: ObjStyle) -> Result<MeshFormat> {
    let format = MeshFormat::from_path(path)?;
    match format {
        MeshFormat::Glb => write_glb(mesh, path)?,
        MeshFormat::Obj => write_obj(mesh, path, obj_style)?,
        MeshFormat::Ply => write_ply(mesh, path)?,
        MeshFormat::Stl => write_stl(mesh, path)?,
    }
    Ok(format)
}

/// Write a binary glTF 2.0 (`.glb`) triangle mesh.
pub fn write_glb(mesh: &MeshView<'_>, path: &Path) -> Result<()> {
    validate(mesh)?;
    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    if vertex_count == 0 || face_count == 0 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "GLB export requires at least one vertex and one face".to_string(),
        });
    }
    if vertex_count > u32::MAX as usize {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!("GLB export supports at most {} vertices", u32::MAX),
        });
    }

    let (min, max) = position_bounds(mesh.vertices);
    let positions_offset = 0usize;
    let positions_len = vertex_count * 3 * std::mem::size_of::<f32>();
    let colors_len = mesh
        .colors
        .map_or(0, |_| vertex_count * 3 * std::mem::size_of::<f32>());
    let colors_offset = positions_offset + positions_len;
    let indices_offset = colors_offset + colors_len;
    let indices_len = face_count * 3 * std::mem::size_of::<u32>();

    let mut binary = Vec::with_capacity(indices_offset + indices_len);
    for value in mesh.vertices {
        binary.extend_from_slice(&value.to_le_bytes());
    }
    if let Some(colors) = mesh.colors {
        for value in colors {
            binary.extend_from_slice(&value.clamp(0.0, 1.0).to_le_bytes());
        }
    }
    for index in mesh.faces {
        let index = u32::try_from(*index).map_err(|_| invalid_face_index())?;
        binary.extend_from_slice(&index.to_le_bytes());
    }

    let mut buffer_views = vec![json!({
        "buffer": 0,
        "byteOffset": positions_offset,
        "byteLength": positions_len,
        "target": 34962,
    })];
    let mut accessors = vec![json!({
        "bufferView": 0,
        "componentType": 5126,
        "count": vertex_count,
        "type": "VEC3",
        "min": [min[0], min[1], min[2]],
        "max": [max[0], max[1], max[2]],
    })];
    let mut attributes = serde_json::Map::new();
    attributes.insert("POSITION".to_string(), json!(0));
    if mesh.colors.is_some() {
        buffer_views.push(json!({
            "buffer": 0,
            "byteOffset": colors_offset,
            "byteLength": colors_len,
            "target": 34962,
        }));
        accessors.push(json!({
            "bufferView": 1,
            "componentType": 5126,
            "count": vertex_count,
            "type": "VEC3",
        }));
        attributes.insert("COLOR_0".to_string(), json!(1));
    }
    let indices_accessor = accessors.len();
    buffer_views.push(json!({
        "buffer": 0,
        "byteOffset": indices_offset,
        "byteLength": indices_len,
        "target": 34963,
    }));
    accessors.push(json!({
        "bufferView": buffer_views.len() - 1,
        "componentType": 5125,
        "count": face_count * 3,
        "type": "SCALAR",
    }));

    let document = json!({
        "asset": {
            "version": "2.0",
            "generator": "lux3d",
        },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "name": "mesh",
            "primitives": [{
                "attributes": attributes,
                "indices": indices_accessor,
                "mode": 4,
            }],
        }],
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": binary.len() }],
    });
    let json =
        serde_json::to_vec(&document).map_err(|error| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to serialize GLB JSON: {error}"),
        })?;
    write_glb_container(path, &json, &binary)
}

/// Write an ASCII PLY triangle mesh (`vertex` + `face` elements).
pub fn write_ply(mesh: &MeshView<'_>, path: &Path) -> Result<()> {
    validate(mesh)?;
    let vertex_count = mesh.vertices.len() / 3;
    let face_count = mesh.faces.len() / 3;
    let mut body = String::new();
    writeln!(body, "ply").expect("writing to String cannot fail");
    writeln!(body, "format ascii 1.0").expect("writing to String cannot fail");
    writeln!(body, "element vertex {vertex_count}").expect("writing to String cannot fail");
    writeln!(body, "property float x").expect("writing to String cannot fail");
    writeln!(body, "property float y").expect("writing to String cannot fail");
    writeln!(body, "property float z").expect("writing to String cannot fail");
    if mesh.colors.is_some() {
        writeln!(body, "property uchar red").expect("writing to String cannot fail");
        writeln!(body, "property uchar green").expect("writing to String cannot fail");
        writeln!(body, "property uchar blue").expect("writing to String cannot fail");
    }
    writeln!(body, "element face {face_count}").expect("writing to String cannot fail");
    writeln!(body, "property list uchar int vertex_indices")
        .expect("writing to String cannot fail");
    writeln!(body, "end_header").expect("writing to String cannot fail");
    for index in 0..vertex_count {
        let offset = index * 3;
        if let Some(colors) = mesh.colors {
            writeln!(
                body,
                "{} {} {} {} {} {}",
                mesh.vertices[offset],
                mesh.vertices[offset + 1],
                mesh.vertices[offset + 2],
                color_u8(colors[offset]),
                color_u8(colors[offset + 1]),
                color_u8(colors[offset + 2]),
            )
            .expect("writing to String cannot fail");
        } else {
            writeln!(
                body,
                "{} {} {}",
                mesh.vertices[offset],
                mesh.vertices[offset + 1],
                mesh.vertices[offset + 2],
            )
            .expect("writing to String cannot fail");
        }
    }
    for face in mesh.faces.chunks_exact(3) {
        writeln!(body, "3 {} {} {}", face[0], face[1], face[2])
            .expect("writing to String cannot fail");
    }
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

/// Write a binary STL triangle mesh. Vertex colors are intentionally dropped
/// because STL has no portable per-vertex color channel.
pub fn write_stl(mesh: &MeshView<'_>, path: &Path) -> Result<()> {
    validate(mesh)?;
    let triangle_count = mesh.faces.len() / 3;
    if triangle_count > u32::MAX as usize {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!("STL export supports at most {} triangles", u32::MAX),
        });
    }
    let mut body = Vec::with_capacity(84 + triangle_count * 50);
    let mut header = [0u8; 80];
    let label = b"lux3d binary STL";
    header[..label.len()].copy_from_slice(label);
    body.extend_from_slice(&header);
    body.extend_from_slice(&(triangle_count as u32).to_le_bytes());
    for face in mesh.faces.chunks_exact(3) {
        let a = vertex(mesh.vertices, face[0])?;
        let b = vertex(mesh.vertices, face[1])?;
        let c = vertex(mesh.vertices, face[2])?;
        let normal = face_normal(a, b, c);
        for value in normal.into_iter().chain(a).chain(b).chain(c) {
            body.extend_from_slice(&value.to_le_bytes());
        }
        body.extend_from_slice(&0u16.to_le_bytes());
    }
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

pub fn write_obj(mesh: &MeshView<'_>, path: &Path, style: ObjStyle) -> Result<()> {
    validate(mesh)?;
    let mut body = String::new();
    if let Some(header) = style.header {
        writeln!(body, "{header}").expect("writing to String cannot fail");
    }
    for index in 0..(mesh.vertices.len() / 3) {
        let offset = index * 3;
        let (x, y, z) = (
            mesh.vertices[offset],
            mesh.vertices[offset + 1],
            mesh.vertices[offset + 2],
        );
        match (style.decimals, style.vertex_colors, mesh.colors) {
            (Some(decimals), false, _) => {
                writeln!(body, "v {x:.decimals$} {y:.decimals$} {z:.decimals$}")
                    .expect("writing to String cannot fail");
            }
            (Some(decimals), true, Some(colors)) => writeln!(
                body,
                "v {x:.decimals$} {y:.decimals$} {z:.decimals$} {:.decimals$} {:.decimals$} {:.decimals$}",
                colors[offset].clamp(0.0, 1.0),
                colors[offset + 1].clamp(0.0, 1.0),
                colors[offset + 2].clamp(0.0, 1.0),
            )
            .expect("writing to String cannot fail"),
            (None, false, _) => writeln!(body, "v {x} {y} {z}")
                .expect("writing to String cannot fail"),
            (None, true, Some(colors)) => writeln!(
                body,
                "v {x} {y} {z} {} {} {}",
                colors[offset].clamp(0.0, 1.0),
                colors[offset + 1].clamp(0.0, 1.0),
                colors[offset + 2].clamp(0.0, 1.0),
            )
            .expect("writing to String cannot fail"),
            // `vertex_colors` requested without a color buffer: positions only.
            (_, true, None) => writeln!(body, "v {x} {y} {z}")
                .expect("writing to String cannot fail"),
        }
    }
    for face in mesh.faces.chunks_exact(3) {
        writeln!(body, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1)
            .expect("writing to String cannot fail");
    }
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

fn write_glb_container(path: &Path, json: &[u8], binary: &[u8]) -> Result<()> {
    const MAGIC: u32 = 0x4654_6C67;
    const VERSION: u32 = 2;
    const JSON_CHUNK: u32 = 0x4E4F_534A;
    const BIN_CHUNK: u32 = 0x004E_4942;

    let json_padding = (4 - json.len() % 4) % 4;
    let bin_padding = (4 - binary.len() % 4) % 4;
    let json_len = json.len() + json_padding;
    let bin_len = binary.len() + bin_padding;
    let total_len = 12 + 8 + json_len + 8 + bin_len;
    let total_len_u32 =
        u32::try_from(total_len).map_err(|_| Lux3dError::CanonicalWeightsValidation {
            message: "GLB output exceeds the 4 GiB container limit".to_string(),
        })?;

    let mut body = Vec::with_capacity(total_len);
    body.extend_from_slice(&MAGIC.to_le_bytes());
    body.extend_from_slice(&VERSION.to_le_bytes());
    body.extend_from_slice(&total_len_u32.to_le_bytes());
    body.extend_from_slice(&(json_len as u32).to_le_bytes());
    body.extend_from_slice(&JSON_CHUNK.to_le_bytes());
    body.extend_from_slice(json);
    body.extend(std::iter::repeat_n(b' ', json_padding));
    body.extend_from_slice(&(bin_len as u32).to_le_bytes());
    body.extend_from_slice(&BIN_CHUNK.to_le_bytes());
    body.extend_from_slice(binary);
    body.extend(std::iter::repeat_n(0u8, bin_padding));
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

fn validate(mesh: &MeshView<'_>) -> Result<()> {
    if mesh.vertices.len() % 3 != 0 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "mesh vertex buffer length must be a multiple of 3".to_string(),
        });
    }
    if mesh.faces.len() % 3 != 0 {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "mesh face buffer length must be a multiple of 3".to_string(),
        });
    }
    if let Some(colors) = mesh.colors {
        if colors.len() != mesh.vertices.len() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "mesh color buffer must match the vertex buffer length".to_string(),
            });
        }
    }
    let vertex_count = mesh.vertices.len() / 3;
    for value in mesh.vertices {
        if !value.is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "mesh export rejected a non-finite vertex".to_string(),
            });
        }
    }
    for index in mesh.faces {
        if *index < 0 || (*index as usize) >= vertex_count {
            return Err(invalid_face_index());
        }
    }
    Ok(())
}

fn invalid_face_index() -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: "mesh export rejected a face index outside the vertex buffer".to_string(),
    }
}

fn position_bounds(vertices: &[f32]) -> ([f32; 3], [f32; 3]) {
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

fn vertex(vertices: &[f32], index: i64) -> Result<[f32; 3]> {
    let index = usize::try_from(index).map_err(|_| invalid_face_index())?;
    let offset = index.checked_mul(3).ok_or_else(invalid_face_index)?;
    let vertex = vertices
        .get(offset..offset + 3)
        .ok_or_else(invalid_face_index)?;
    Ok([vertex[0], vertex[1], vertex[2]])
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if length == 0.0 || !length.is_finite() {
        [0.0, 0.0, 0.0]
    } else {
        [normal[0] / length, normal[1] / length, normal[2] / length]
    }
}

fn color_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(extension: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-mesh-export-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    fn triangle() -> MeshView<'static> {
        MeshView {
            vertices: &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            faces: &[0, 1, 2],
            colors: None,
        }
    }

    #[test]
    fn glb_writer_emits_valid_container_and_accessors() {
        let path = temp_path("glb");
        write_mesh(&triangle(), &path, ObjStyle::TRIPOSG).expect("write glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);

        assert_eq!(&bytes[..4], b"glTF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            bytes.len()
        );

        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json_type = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert_eq!(json_type, 0x4E4F_534A);
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + json_len]).expect("glb json");
        assert_eq!(json["asset"]["version"], "2.0");
        assert_eq!(json["meshes"][0]["primitives"][0]["mode"], 4);
        assert_eq!(json["accessors"][0]["count"], 3);
        assert_eq!(json["accessors"][0]["min"], json!([0.0, 0.0, 0.0]));
        assert_eq!(json["accessors"][0]["max"], json!([1.0, 1.0, 0.0]));
        assert_eq!(json["accessors"][1]["componentType"], 5125);
        assert_eq!(json["accessors"][1]["count"], 3);

        let bin_offset = 20 + json_len + 8;
        let bin_len = u32::from_le_bytes(bytes[bin_offset - 8..bin_offset - 4].try_into().unwrap());
        assert_eq!(
            u32::from_le_bytes(bytes[bin_offset - 4..bin_offset].try_into().unwrap()),
            0x004E_4942
        );
        let bin = &bytes[bin_offset..bin_offset + bin_len as usize];
        assert_eq!(f32::from_le_bytes(bin[0..4].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_le_bytes(bin[12..16].try_into().unwrap()), 1.0);
        assert_eq!(u32::from_le_bytes(bin[36..40].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bin[44..48].try_into().unwrap()), 2);
    }

    #[test]
    fn glb_writer_preserves_vertex_colors() {
        let colors = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mesh = MeshView {
            colors: Some(&colors),
            ..triangle()
        };
        let path = temp_path("glb");
        write_glb(&mesh, &path).expect("write colored glb");
        let bytes = std::fs::read(&path).expect("read glb");
        let _ = std::fs::remove_file(&path);
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let json: serde_json::Value =
            serde_json::from_slice(&bytes[20..20 + json_len]).expect("glb json");
        assert_eq!(
            json["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"],
            1
        );
        assert_eq!(json["accessors"][1]["count"], 3);
    }

    #[test]
    fn glb_writer_rejects_out_of_range_faces() {
        let mesh = MeshView {
            faces: &[0, 1, 3],
            ..triangle()
        };
        let path = temp_path("glb");
        let error = write_glb(&mesh, &path).expect_err("invalid face must fail");
        assert!(error.to_string().contains("face index"));
        assert!(!path.exists());
    }

    #[test]
    fn ply_and_stl_writers_emit_expected_headers() {
        let ply = temp_path("ply");
        write_mesh(&triangle(), &ply, ObjStyle::TRIPOSG).expect("write ply");
        let body = std::fs::read_to_string(&ply).expect("read ply");
        let _ = std::fs::remove_file(&ply);
        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("element face 1"));
        assert!(body.contains("\n3 0 1 2\n"));

        let stl = temp_path("stl");
        write_mesh(&triangle(), &stl, ObjStyle::TRIPOSG).expect("write stl");
        let bytes = std::fs::read(&stl).expect("read stl");
        let _ = std::fs::remove_file(&stl);
        assert_eq!(bytes.len(), 84 + 50);
        assert_eq!(u32::from_le_bytes(bytes[80..84].try_into().unwrap()), 1);
        // Unit normal along +Z, then vertex a at (0, 0, 0).
        assert_eq!(f32::from_le_bytes(bytes[92..96].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[96..100].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_le_bytes(bytes[108..112].try_into().unwrap()), 1.0);
    }
}
