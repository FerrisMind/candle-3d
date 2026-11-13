//! TripoSR export stage: format-agnostic triangle-mesh writer.
//!
//! GLB is the primary artifact; OBJ, PLY and STL are selected by file
//! extension. TripoSR carries per-vertex colors, so GLB uses a `COLOR_0`
//! attribute and PLY uses RGB vertex properties. STL cannot represent vertex
//! colors and therefore drops them.

use std::path::Path;

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, TripoMesh},
    geometry::TripoMeshCpu,
};

use super::mesh::{MeshFormat, MeshView, ObjStyle, write_mesh, write_obj};

/// Export-stage descriptor for the TripoSR family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoExportStage;

impl ExportStage<TripoMesh> for TripoExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &TripoMesh) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "glb".to_string(),
            alternate_extensions: vec!["obj".to_string(), "ply".to_string(), "stl".to_string()],
            utility_notes: vec![
                "GLB is the primary artifact; format follows the output extension".to_string(),
                "vertex colors are emitted as COLOR_0 (GLB), RGB (PLY) and v-line colors (OBJ)"
                    .to_string(),
                "STL drops vertex colors".to_string(),
            ],
        })
    }
}

impl TripoExportStage {
    /// Write the CPU mesh in the format selected by `path`'s extension.
    pub fn write_mesh(&self, mesh: &TripoMeshCpu, path: &Path) -> Result<MeshFormat> {
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: Some(&mesh.vertex_colors),
        };
        write_mesh(&view, path, ObjStyle::TRIPOSR)
    }

    /// Write the CPU mesh as ASCII OBJ, preserving the historical output.
    pub fn write_obj(&self, mesh: &TripoMeshCpu, path: &Path) -> Result<()> {
        let view = MeshView {
            vertices: &mesh.vertices,
            faces: &mesh.faces,
            colors: Some(&mesh.vertex_colors),
        };
        write_obj(&view, path, ObjStyle::TRIPOSR)
    }
}
