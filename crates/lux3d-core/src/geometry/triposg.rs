//! TripoSG marching-cubes geometry: SDF grid -> mesh (CPU).
//!
//! Reference semantics (`triposg/inference_utils.py`
//! `hierarchical_extract_geometry` final stage, skimage
//! `measure.marching_cubes(..., level=0, method="lewiner")`):
//!
//! - Marching cubes at iso level 0 on the (negated-logit) SDF grid.
//! - Vertex world mapping: `vertices / 2^depth * bbox_size + bbox_min`,
//!   i.e. index `i` maps to `bounds_min + i * (bounds_size / n)` — the mcubes
//!   scale is `bounds_size / n` (2.01 / n for the canonical +/-1.005 bounds),
//!   NOT `bounds_size / (n - 1)`.
//! - No axis reorder (unlike TripoSR's `v_pos[..., [2, 1, 0]]`) and no
//!   vertex welding beyond position-bit deduplication; faces are emitted
//!   zero-based here and become 1-based at OBJ export time.
//! - Topology is NOT expected to match skimage's lewiner output (different
//!   tables); parity is asserted on bbox / volume in the tests.

use lin_alg::f32::Vec3;
use mcubes::{MarchingCubes, MeshSide};

use crate::{
    Result,
    contracts::{GeometryStage, ModelFamily, TripoSgMesh, TripoSgSceneCode},
    error::Lux3dError,
    runtime::TripoSgSdfGrid,
};

/// CPU mesh extracted from a TripoSG SDF grid.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgMeshCpu {
    /// Grid side length the mesh was extracted at (`2^depth`).
    pub resolution: u32,
    /// Iso level used (0.0: positive logit = inside).
    pub sdf_level: f32,
    /// World bounds the grid spans (`[min_x, min_y, min_z, max_x, max_y, max_z]`).
    pub bounds: [f32; 6],
    /// Vertex positions, flat `[x0, y0, z0, x1, ...]`.
    pub vertices: Vec<f32>,
    /// Triangle indices, zero-based (export adds 1).
    pub faces: Vec<i64>,
}

/// Static geometry-stage descriptor for the TripoSG family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoSgGeometryStage;

impl GeometryStage<TripoSgSceneCode, TripoSgMesh> for TripoSgGeometryStage {
    fn assemble(&self, _scene: &TripoSgSceneCode) -> Result<TripoSgMesh> {
        Ok(TripoSgMesh {
            family: ModelFamily::TripoSg,
            surface_extractor: "flash_diffdmc",
            default_dense_octree_depth: 8,
            default_hierarchical_octree_depth: 9,
            default_use_flash_decoder: true,
            default_flash_octree_depth: 9,
            sdf_level: 0.0,
            bounds: [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
        })
    }
}

impl TripoSgGeometryStage {
    /// Extract the level-0 surface from a decoded SDF grid (CPU).
    ///
    /// The grid must already carry the values marching cubes should see. The
    /// reference decoder casts its logits to fp16 before `marching_cubes`;
    /// the production pipeline assembles the raw F32 grid, while the golden
    /// parity tests round through fp16 first to reproduce the reference mesh
    /// path exactly.
    pub fn assemble_cpu(&self, grid: &TripoSgSdfGrid) -> Result<TripoSgMeshCpu> {
        let n = grid.dims;
        if n == 0 || grid.values.len() != n * n * n {
            return Err(Lux3dError::InvalidInput(
                "TripoSG SDF grid length does not match dims^3",
            ));
        }
        if grid.values.iter().any(|value| !value.is_finite()) {
            return Err(Lux3dError::InvalidInput(
                "TripoSG SDF grid contains non-finite values",
            ));
        }

        let size = (
            grid.bounds[3] - grid.bounds[0],
            grid.bounds[4] - grid.bounds[1],
            grid.bounds[5] - grid.bounds[2],
        );
        // scale = size / sampling_interval = bounds_size / n (2.01 / n for the
        // canonical bounds), offset = bounds_min. Vertex index i on each axis
        // lands at bounds_min + i * bounds_size / n, matching the reference
        // `vertices / 2^depth * bbox_size + bbox_min` (NOT size / (n - 1)).
        let mc = MarchingCubes::new(
            (n, n, n),
            (size.0, size.1, size.2),
            (n as f32, n as f32, n as f32),
            Vec3::new(grid.bounds[0], grid.bounds[1], grid.bounds[2]),
            grid.values.clone(),
            grid_depth0_level(),
        )
        .map_err(|_| Lux3dError::InvalidInput("failed to initialize TripoSG marching cubes"))?;
        let mesh = mc.generate(MeshSide::OutsideOnly);

        // Deduplicate vertices by position bits (same pattern as TripoSR);
        // NO axis reorder — the grid axes already match the reference's
        // meshgrid "ij" (x, y, z).
        let mut unique_vertices = Vec::new();
        let mut faces = Vec::with_capacity(mesh.indices.len());
        let mut index_map = std::collections::HashMap::new();
        for &idx in &mesh.indices {
            let posit = mesh.vertices[idx].posit;
            let (px, py, pz) = (posit.x, posit.y, posit.z);
            if !(px.is_finite() && py.is_finite() && pz.is_finite()) {
                return Err(Lux3dError::InvalidInput(
                    "TripoSG marching cubes produced non-finite vertices",
                ));
            }
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

        Ok(TripoSgMeshCpu {
            resolution: n as u32,
            sdf_level: grid_depth0_level(),
            bounds: grid.bounds,
            vertices: unique_vertices,
            faces,
        })
    }
}

/// Iso level for the TripoSG surface: the reference runs marching cubes at
/// level 0 on the negated logits (positive = inside).
fn grid_depth0_level() -> f32 {
    0.0
}
