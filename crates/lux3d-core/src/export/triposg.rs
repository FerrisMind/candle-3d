//! TripoSG export stage: format-agnostic triangle-mesh writer.
//!
//! GLB is the primary artifact; OBJ, PLY and STL are selected by file
//! extension. The TripoSG SDF surface carries no vertex colors, so every
//! format stores positions and triangle indices only.
//!
//! OBJ output matches the reference `mesh.export(...)` (trimesh) shape: a
//! `# trimesh` header line, one `v x y z` line per vertex with 8 decimal
//! digits, and one `f a b c` line per face with 1-based indices.

use std::path::Path;

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, TripoSgMesh},
    geometry::TripoSgMeshCpu,
};

use super::mesh::{MeshFormat, MeshView, ObjStyle, write_mesh, write_obj};

/// Export-stage descriptor for the TripoSG family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoSgExportStage;

impl ExportStage<TripoSgMesh> for TripoSgExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &TripoSgMesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "glb".to_string(),
            alternate_extensions: vec!["obj".to_string(), "ply".to_string(), "stl".to_string()],
            utility_notes: vec![
                "GLB is the primary artifact; format follows the output extension".to_string(),
                "OBJ: v lines carry positions only with 8 decimals; f indices are 1-based"
                    .to_string(),
                "no vertex colors; the marching-cubes surface is written as-is".to_string(),
            ],
        })
    }
}

impl TripoSgExportStage {
    /// Write the CPU mesh in the format selected by `path`'s extension.
    pub fn write_mesh(&self, mesh: &TripoSgMeshCpu, path: &Path) -> Result<MeshFormat> {
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: None,
        };
        write_mesh(&view, path, ObjStyle::TRIPOSG)
    }

    /// Write the CPU mesh as UTF-8 ASCII OBJ, preserving the historical output.
    pub fn write_obj(&self, mesh: &TripoSgMeshCpu, path: &Path) -> Result<()> {
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: None,
        };
        write_obj(&view, path, ObjStyle::TRIPOSG)
    }
}
