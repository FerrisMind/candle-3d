//! TripoSG export stage: ASCII OBJ writer for the marching-cubes surface.
//!
//! Output matches the reference `mesh.export(...)` (trimesh) shape: a
//! `# trimesh` header line, one `v x y z` line per vertex (positions only —
//! the TripoSG SDF surface carries no vertex colors) and one `f a b c` line
//! per face with 1-based indices. Vertices are written with 8 decimal digits
//! like trimesh's default OBJ digits. Non-finite vertices are rejected before
//! any file is created.

use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, TripoSgMesh},
    error::Lux3dError,
    geometry::TripoSgMeshCpu,
};

/// Static export-stage descriptor for the TripoSG family: OBJ is the primary
/// (and only) artifact format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoSgExportStage;

impl ExportStage<TripoSgMesh> for TripoSgExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &TripoSgMesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "obj".to_string(),
            alternate_extensions: Vec::new(),
            utility_notes: vec![
                "ascii OBJ: v lines carry positions only (no vertex colors)".to_string(),
                "f indices are 1-based".to_string(),
                "no decimation; the marching-cubes surface is written as-is".to_string(),
            ],
        })
    }
}

impl TripoSgExportStage {
    /// Write the CPU mesh as UTF-8 ASCII OBJ (buffered writer).
    pub fn write_obj(&self, mesh: &TripoSgMeshCpu, path: &Path) -> Result<()> {
        if mesh.vertices.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "TripoSG OBJ export rejected a non-finite vertex",
            ));
        }
        let file = File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# https://github.com/mikedh/trimesh").map_err(|source| {
            Lux3dError::ModelAssetIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        for vertex in mesh.vertices.chunks_exact(3) {
            writeln!(
                writer,
                "v {:.8} {:.8} {:.8}",
                vertex[0], vertex[1], vertex[2]
            )
            .map_err(|source| Lux3dError::ModelAssetIo {
                path: path.to_path_buf(),
                source,
            })?;
        }
        for face in mesh.faces.chunks_exact(3) {
            writeln!(writer, "f {} {} {}", face[0] + 1, face[1] + 1, face[2] + 1).map_err(
                |source| Lux3dError::ModelAssetIo {
                    path: path.to_path_buf(),
                    source,
                },
            )?;
        }
        writer.flush().map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(())
    }
}
