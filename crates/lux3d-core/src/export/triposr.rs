use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, TripoMesh},
    error::Lux3dError,
    geometry::TripoMeshCpu,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoExportStage;

impl ExportStage<TripoMesh> for TripoExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &TripoMesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "obj".to_string(),
            alternate_extensions: Vec::new(),
            utility_notes: vec![
                "vertex colors by default".to_string(),
                "texture baking stays outside the core runtime contract".to_string(),
            ],
        })
    }
}

impl TripoExportStage {
    pub fn write_obj(&self, mesh: &TripoMeshCpu, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "# https://github.com/mikedh/trimesh").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        for index in 0..(mesh.vertices.len() / 3) {
            let offset = index * 3;
            writeln!(
                writer,
                "v {} {} {} {} {} {}",
                mesh.vertices[offset],
                mesh.vertices[offset + 1],
                mesh.vertices[offset + 2],
                mesh.vertex_colors[offset].clamp(0.0, 1.0),
                mesh.vertex_colors[offset + 1].clamp(0.0, 1.0),
                mesh.vertex_colors[offset + 2].clamp(0.0, 1.0)
            )
            .map_err(|source| Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            })?;
        }
        for index in 0..(mesh.faces.len() / 3) {
            let offset = index * 3;
            writeln!(
                writer,
                "f {} {} {}",
                mesh.faces[offset] + 1,
                mesh.faces[offset + 1] + 1,
                mesh.faces[offset + 2] + 1
            )
            .map_err(|source| Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            })?;
        }
        writer
            .flush()
            .map_err(|source| Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(())
    }
}
