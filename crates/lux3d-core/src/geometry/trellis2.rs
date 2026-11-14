//! TRELLIS.2 geometry stage: portable dual-grid mesh buffers.
//!
//! The runtime pipeline decodes the shape SLAT and extracts the O-Voxel
//! dual-grid mesh (`runtime::trellis2_mesh`); this module holds the plain
//! CPU buffers plus voxel payload that the export stage and the
//! `Trellis2Mesh` contract consume. The reference coordinate frame is kept
//! unchanged: vertices live in the reference `[-0.5, 0.5]^3` box with no
//! axis remap.

use crate::{ModelFamily, Result, contracts::Trellis2Mesh, error::Lux3dError};

/// CPU mesh + voxel payload produced by the TRELLIS.2 shape decode.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Trellis2MeshBuffers {
    /// Voxel grid resolution the mesh was extracted at (512 for the 512
    /// pipeline).
    pub resolution: u32,
    /// Flat `[x, y, z, ...]` f32 vertex positions, one triple per active
    /// voxel (dual-grid vertices; may include unused voxels).
    pub vertices: Vec<f32>,
    /// Flat zero-based triangle indices.
    pub faces: Vec<i64>,
    /// Flat `[x, y, z, ...]` i32 active-voxel coordinates (row order matches
    /// `voxel_attrs` rows).
    pub voxel_coords: Vec<i32>,
    /// Flat `[N, attrs]` decoded per-voxel attributes (7 channels for the
    /// shape decode; 6 PBR channels when the texture stage ran).
    pub voxel_attrs: Vec<f32>,
    /// Attribute channel count (`voxel_attrs.len() / voxel_count`).
    pub voxel_attr_channels: usize,
    /// Optional per-vertex base colors sampled from the decoded PBR
    /// attributes (flat `[V, 3]` RGB rows in [0, 1]; `None` for
    /// geometry-only runs).
    pub vertex_colors: Option<Vec<f32>>,
}

impl Trellis2MeshBuffers {
    /// Number of dual-grid vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / 3
    }

    /// Number of triangles.
    pub fn face_count(&self) -> usize {
        self.faces.len() / 3
    }

    /// Number of active voxels carrying attributes.
    pub fn voxel_count(&self) -> usize {
        if self.voxel_attr_channels == 0 {
            0
        } else {
            self.voxel_attrs.len() / self.voxel_attr_channels
        }
    }

    /// Axis-aligned vertex bounds `(min, max)`, or `None` for an empty mesh.
    pub fn bbox(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        let mut any = false;
        for vertex in self.vertices.chunks_exact(3) {
            any = true;
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        if any { Some((min, max)) } else { None }
    }

    /// True when any vertex is non-finite.
    pub fn has_non_finite_vertex(&self) -> bool {
        self.vertices.iter().any(|v| !v.is_finite())
    }

    /// Structural validation: buffer length divisibility, finite vertices,
    /// face indices in range, voxel payload consistency.
    pub fn validate(&self) -> Result<()> {
        if self.vertices.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh vertex buffer length must be a multiple of 3",
            ));
        }
        if self.faces.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh face buffer length must be a multiple of 3",
            ));
        }
        if self.has_non_finite_vertex() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 mesh contains a non-finite vertex",
            ));
        }
        let vertex_count = self.vertex_count();
        for &index in &self.faces {
            if index < 0 || (vertex_count > 0 && index as usize >= vertex_count) {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 mesh face index outside the vertex buffer",
                ));
            }
        }
        if self.voxel_coords.len() % 3 != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 voxel coordinate buffer length must be a multiple of 3",
            ));
        }
        if self.voxel_attr_channels > 0 && self.voxel_attrs.len() % self.voxel_attr_channels != 0 {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 voxel attribute buffer length does not match the channel count",
            ));
        }
        if let Some(colors) = &self.vertex_colors {
            if colors.len() != self.vertex_count() * 3 {
                return Err(Lux3dError::InvalidInput(
                    "Trellis2 vertex color buffer must hold three channels per vertex",
                ));
            }
        }
        Ok(())
    }

    /// Convert into the serialized `Trellis2Mesh` contract struct.
    pub fn to_contract(&self) -> Result<Trellis2Mesh> {
        self.validate()?;
        Ok(Trellis2Mesh {
            family: ModelFamily::Trellis2,
            vertices: self.vertices.clone(),
            faces: self.faces.clone(),
            voxel_coords: self.voxel_coords.clone(),
            voxel_attrs: self.voxel_attrs.clone(),
            resolution: self.resolution,
        })
    }
}

/// Static geometry-stage descriptor for the TRELLIS.2 family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trellis2GeometryStage;

impl Trellis2GeometryStage {
    /// Assemble mesh buffers from the decode products. Faces and vertices
    /// come straight from the dual-grid extraction; the voxel payload keeps
    /// the active-voxel coordinates and decoded attributes (the texture
    /// stage swaps them for the 6 PBR channels and fills `vertex_colors`).
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        &self,
        resolution: u32,
        vertices: Vec<f32>,
        faces: Vec<i64>,
        voxel_coords: Vec<i32>,
        voxel_attrs: Vec<f32>,
        voxel_attr_channels: usize,
    ) -> Result<Trellis2MeshBuffers> {
        let buffers = Trellis2MeshBuffers {
            resolution,
            vertices,
            faces,
            voxel_coords,
            voxel_attrs,
            voxel_attr_channels,
            vertex_colors: None,
        };
        buffers.validate()?;
        Ok(buffers)
    }
}
