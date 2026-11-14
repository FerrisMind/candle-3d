//! TRELLIS.2 export stage: dual-grid mesh writer with optional vertex
//! colors and the PBR voxel sidecar.
//!
//! OBJ output stays positions-only (the shared OBJ writer has no color
//! surface; the reference exports textured OBJs via UV atlases, out of
//! scope). PLY/GLB carry the per-vertex base colors sampled by the texture
//! stage when present. The `<stem>.pbrvox.bin` sidecar serializes the
//! sparse PBR voxel grid (magic `L3DPBR1\0`, u32 version, u32 resolution,
//! u64 voxel count, then one little-endian record per active voxel: 3
//! packed i32 coords + 6 f32 attributes) and the optional
//! `<stem>.pbr.json` describes the layout and value ranges. The reference
//! coordinate frame is preserved: the OBJ/PLY/GLB carry the `[-0.5, 0.5]^3`
//! vertices as decoded, with no axis remap. OBJ formatting follows the
//! shared trimesh style (8 decimals, 1-based face indices).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, Trellis2Mesh},
    error::Lux3dError,
    geometry::Trellis2MeshBuffers,
};

use super::mesh::{MeshFormat, MeshView, ObjStyle, write_mesh, write_obj};

/// PBR voxel sidecar constants (`L3DPBR1\0` magic, format version 1).
pub mod pbrvox {
    /// File magic: `L3DPBR1\0` (8 bytes).
    pub const MAGIC: [u8; 8] = *b"L3DPBR1\0";
    /// Sidecar format version.
    pub const VERSION: u32 = 1;
    /// Per-record attribute channel count (base_color 3 + metallic +
    /// roughness + alpha).
    pub const CHANNELS: usize = 6;
}

/// Export-stage descriptor for the TRELLIS.2 family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trellis2ExportStage;

impl ExportStage<Trellis2Mesh> for Trellis2ExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &Trellis2Mesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "glb".to_string(),
            alternate_extensions: vec!["obj".to_string(), "ply".to_string()],
            utility_notes: vec![
                "GLB is the primary artifact; format follows the output extension".to_string(),
                "OBJ: v lines carry positions only with 8 decimals; f indices are 1-based"
                    .to_string(),
                "PLY/GLB: per-vertex colors sampled from the decoded PBR attributes".to_string(),
                "reference coordinate frame [-0.5, 0.5]^3, no axis remap".to_string(),
                "optional <stem>.pbrvox.bin sidecar + <stem>.pbr.json layout descriptor"
                    .to_string(),
            ],
        })
    }
}

/// Shared sidecar guard: the payload must carry the 6 PBR channels and one
/// coordinate row per attribute row (the header `count` is derived from the
/// attributes, so a mismatched coordinate buffer would corrupt the records).
fn validate_pbr_payload(mesh: &Trellis2MeshBuffers) -> Result<()> {
    mesh.validate()?;
    if mesh.voxel_attr_channels != pbrvox::CHANNELS {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!(
                "PBR sidecar needs {} attribute channels, mesh carries {}",
                pbrvox::CHANNELS,
                mesh.voxel_attr_channels
            ),
        });
    }
    if mesh.voxel_coords.len() / 3 != mesh.voxel_count() {
        return Err(Lux3dError::ModelAssetResolution {
            message: format!(
                "PBR sidecar coord count {} does not match the attribute row count {}",
                mesh.voxel_coords.len() / 3,
                mesh.voxel_count()
            ),
        });
    }
    Ok(())
}

impl Trellis2ExportStage {
    /// Write the mesh buffers in the format selected by `path`'s extension.
    /// PLY/GLB include the sampled vertex colors when the mesh carries
    /// them; OBJ stays positions-only.
    pub fn write_mesh(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<MeshFormat> {
        mesh.validate()?;
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: mesh.vertex_colors.as_deref(),
        };
        write_mesh(&view, path, ObjStyle::TRIPOSG)
    }

    /// Write the mesh buffers as UTF-8 ASCII OBJ (positions only).
    pub fn write_obj(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        mesh.validate()?;
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: None,
        };
        write_obj(&view, path, ObjStyle::TRIPOSG)
    }

    /// Write the mesh plus the PBR voxel sidecar pair:
    /// `<stem>.pbrvox.bin` (sparse grid records) and `<stem>.pbr.json`
    /// (layout descriptor). Returns the sidecar path.
    pub fn write_mesh_with_pbr(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<PathBuf> {
        self.write_mesh(mesh, path)?;
        let stem = path.parent().unwrap_or_else(|| Path::new(".")).join(
            path.file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new("mesh")),
        );
        self.write_pbr_vox(mesh, &stem)
    }

    /// Write the `<stem>.pbrvox.bin` + `<stem>.pbr.json` sidecar pair for
    /// the mesh's PBR voxel payload. Fails when the mesh does not carry the
    /// 6-channel PBR attributes (geometry-only runs export the mesh alone).
    pub fn write_pbr_vox(&self, mesh: &Trellis2MeshBuffers, stem: &Path) -> Result<PathBuf> {
        self.write_pbrvox_bin(mesh, &stem.with_extension("pbrvox.bin"))?;
        self.write_pbr_json(mesh, &stem.with_extension("pbr.json"))?;
        Ok(stem.with_extension("pbrvox.bin"))
    }

    /// The binary PBR voxel sidecar. Layout (little-endian throughout):
    /// magic `L3DPBR1\0`, u32 version 1, u32 resolution, u64 voxel count,
    /// then `count` records of 3 packed i32 coords + 6 f32 attributes.
    pub fn write_pbrvox_bin(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        validate_pbr_payload(mesh)?;
        let count = mesh.voxel_count();
        let file = File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&pbrvox::MAGIC).map_err(pbr_io(path))?;
        writer
            .write_all(&pbrvox::VERSION.to_le_bytes())
            .map_err(pbr_io(path))?;
        writer
            .write_all(&mesh.resolution.to_le_bytes())
            .map_err(pbr_io(path))?;
        writer
            .write_all(&(count as u64).to_le_bytes())
            .map_err(pbr_io(path))?;
        for voxel in 0..count {
            let coord = voxel * 3;
            for axis in 0..3 {
                writer
                    .write_all(&mesh.voxel_coords[coord + axis].to_le_bytes())
                    .map_err(pbr_io(path))?;
            }
            let attr = voxel * pbrvox::CHANNELS;
            for channel in 0..pbrvox::CHANNELS {
                writer
                    .write_all(&mesh.voxel_attrs[attr + channel].to_le_bytes())
                    .map_err(pbr_io(path))?;
            }
        }
        writer.flush().map_err(pbr_io(path))?;
        Ok(())
    }

    /// The JSON sidecar describing the PBR voxel layout and observed value
    /// ranges (min/max per channel over the active voxels, or `null` for an
    /// empty grid).
    pub fn write_pbr_json(&self, mesh: &Trellis2MeshBuffers, path: &Path) -> Result<()> {
        validate_pbr_payload(mesh)?;
        let count = mesh.voxel_count();
        let mut ranges: [Option<(f32, f32)>; pbrvox::CHANNELS] = Default::default();
        for voxel in 0..count {
            let attr = voxel * pbrvox::CHANNELS;
            for (channel, range) in ranges.iter_mut().enumerate() {
                let value = mesh.voxel_attrs[attr + channel];
                match range {
                    Some((min, max)) => {
                        *min = min.min(value);
                        *max = max.max(value);
                    }
                    None => *range = Some((value, value)),
                }
            }
        }
        // Debug formatting keeps the decimal point ("0.0"), so every range
        // entry parses as a JSON float.
        let range_json = |range: &Option<(f32, f32)>| match range {
            Some((min, max)) => format!("[{min:?}, {max:?}]"),
            None => "null".to_string(),
        };
        let json = format!(
            "{{\n  \"format\": \"pbrvox\",\n  \"version\": {},\n  \"resolution\": {},\n  \
             \"voxel_count\": {},\n  \"channels\": {},\n  \"layout\": {{\n    \
             \"base_color\": [0, 3],\n    \"metallic\": 3,\n    \"roughness\": 4,\n    \
             \"alpha\": 5\n  }},\n  \"ranges\": [\n    {},\n    {},\n    {},\n    {},\n    \
             {},\n    {}\n  ]\n}}\n",
            pbrvox::VERSION,
            mesh.resolution,
            count,
            pbrvox::CHANNELS,
            range_json(&ranges[0]),
            range_json(&ranges[1]),
            range_json(&ranges[2]),
            range_json(&ranges[3]),
            range_json(&ranges[4]),
            range_json(&ranges[5]),
        );
        std::fs::write(path, json).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })
    }
}

/// Closure factory for the sidecar IO error mapping.
fn pbr_io(path: &Path) -> impl Fn(std::io::Error) -> Lux3dError + '_ {
    move |source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    }
}
