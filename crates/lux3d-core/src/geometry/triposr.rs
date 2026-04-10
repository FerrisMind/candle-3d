use candle_core::{Device, Tensor};
use lin_alg::f32::Vec3;
use mcubes::{MarchingCubes, MeshSide};

use crate::{
    Result,
    contracts::{GeometryStage, ModelFamily, TripoMesh, TripoSceneCode},
    neural::TripoDensityGridCpu,
    runtime::TripoMeshBuffers,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TripoMeshCpu {
    pub resolution: u32,
    pub threshold: f32,
    pub vertices: Vec<f32>,
    pub faces: Vec<i64>,
    pub vertex_colors: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoGeometryStage;

impl GeometryStage<TripoSceneCode, TripoMesh> for TripoGeometryStage {
    fn assemble(&self, _scene: &TripoSceneCode) -> Result<TripoMesh> {
        Ok(TripoMesh {
            family: ModelFamily::TripoSr,
            surface_extractor: "marching_cubes",
            default_resolution: 256,
            vertex_color_mode: "vertex_colors",
            texture_baking_extension: false,
        })
    }
}

impl TripoGeometryStage {
    pub fn assemble_cpu(&self, grid: &TripoDensityGridCpu) -> Result<TripoMeshCpu> {
        let mc = MarchingCubes::new(
            (
                grid.resolution as usize,
                grid.resolution as usize,
                grid.resolution as usize,
            ),
            (1.0, 1.0, 1.0),
            (
                (grid.resolution - 1) as f32,
                (grid.resolution - 1) as f32,
                (grid.resolution - 1) as f32,
            ),
            Vec3::new(0.0, 0.0, 0.0),
            grid.density_values.clone(),
            grid.threshold,
        )
        .map_err(|_| {
            crate::error::Lux3dError::InvalidInput("failed to initialize marching cubes")
        })?;
        let mesh = mc.generate(MeshSide::OutsideOnly);

        let mut unique_vertices = Vec::new();
        let mut faces = Vec::with_capacity(mesh.indices.len());
        let mut index_map = std::collections::HashMap::new();
        for &idx in &mesh.indices {
            let vertex = &mesh.vertices[idx];
            let px = vertex.posit.x * 2.0 * grid.radius - grid.radius;
            let py = vertex.posit.y * 2.0 * grid.radius - grid.radius;
            let pz = vertex.posit.z * 2.0 * grid.radius - grid.radius;
            let key = [px.to_bits(), py.to_bits(), pz.to_bits()];
            let out_idx = if let Some(existing) = index_map.get(&key) {
                *existing
            } else {
                let next = (unique_vertices.len() / 3) as i64;
                unique_vertices.extend([px, py, pz]);
                index_map.insert(key, next);
                next
            };
            faces.push(out_idx);
        }

        Ok(TripoMeshCpu {
            resolution: grid.resolution,
            threshold: grid.threshold,
            vertices: unique_vertices,
            faces,
            vertex_colors: Vec::new(),
        })
    }

    pub fn attach_vertex_colors(
        &self,
        mut mesh: TripoMeshCpu,
        vertex_colors: Vec<f32>,
    ) -> TripoMeshCpu {
        mesh.vertex_colors = vertex_colors;
        mesh
    }

    pub fn materialize_buffers(
        &self,
        mesh: &TripoMeshCpu,
        device: &Device,
    ) -> Result<TripoMeshBuffers> {
        let vertex_tensor =
            Tensor::from_vec(mesh.vertices.clone(), (mesh.vertices.len() / 3, 3), device).map_err(
                |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to materialize TripoSR vertices: {source}"),
                },
            )?;
        let face_tensor = Tensor::from_vec(mesh.faces.clone(), (mesh.faces.len() / 3, 3), device)
            .map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSR faces: {source}"),
            }
        })?;
        let vertex_colors = Tensor::from_vec(
            mesh.vertex_colors.clone(),
            (mesh.vertex_colors.len() / 3, 3),
            device,
        )
        .map_err(
            |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSR vertex colors: {source}"),
            },
        )?;

        Ok(TripoMeshBuffers {
            vertices: vertex_tensor,
            faces: face_tensor,
            vertex_colors,
        })
    }

    pub fn cpu_from_buffers(&self, mesh: &TripoMeshBuffers) -> Result<TripoMeshCpu> {
        let vertices = mesh
            .vertices
            .to_device(&Device::Cpu)
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to move TripoSR vertices to cpu")
            })?
            .flatten_all()
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to flatten TripoSR vertices")
            })?
            .to_vec1::<f32>()
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to read TripoSR vertices")
            })?;
        let faces = mesh
            .faces
            .to_device(&Device::Cpu)
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to move TripoSR faces to cpu")
            })?
            .flatten_all()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to flatten TripoSR faces"))?
            .to_vec1::<i64>()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to read TripoSR faces"))?;
        let vertex_colors = mesh
            .vertex_colors
            .to_device(&Device::Cpu)
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to move TripoSR colors to cpu")
            })?
            .flatten_all()
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to flatten TripoSR colors")
            })?
            .to_vec1::<f32>()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to read TripoSR colors"))?;

        Ok(TripoMeshCpu {
            resolution: 0,
            threshold: 0.0,
            vertices,
            faces,
            vertex_colors,
        })
    }
}
