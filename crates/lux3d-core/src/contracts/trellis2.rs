use serde::{Deserialize, Serialize};

use crate::contracts::{ModelFamily, SpatialSize, TensorContract};

/// Pipeline variant selected by `pipeline_type` in the TRELLIS.2 reference
/// configs. `512` runs the single-resolution shape path, `1024` runs the
/// single-resolution path at 64^3 sparse structure, and `1024_cascade` runs
/// the 32^3 -> 1024 shape cascade (the reference default).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trellis2PipelineType {
    #[serde(rename = "512")]
    R512,
    #[serde(rename = "1024")]
    R1024,
    #[serde(rename = "1024_cascade")]
    Cascade1024,
}

impl Trellis2PipelineType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::R512 => "512",
            Self::R1024 => "1024",
            Self::Cascade1024 => "1024_cascade",
        }
    }
}

/// Raw image handed to the TRELLIS.2 preprocess stage.
///
/// `rgba` carries per-pixel premultiplied-capable source alpha for the
/// foreground-bbox/square-crop path; `rgb` is used when the source carries
/// no usable alpha and is first passed through RMBG-2.0/BiRefNet to produce
/// the alpha mask (a missing remover is an error, never an opaque fallback).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trellis2RawInput {
    pub original_size: SpatialSize,
    pub rgba: Option<Vec<[f32; 4]>>,
    pub rgb: Option<Vec<[f32; 3]>>,
}

/// Prepared RGB image fed to the DINOv3 ViT-L/16 image-conditioning encoder.
///
/// `uint8_rgb` is the resized square crop in [H, W, 3] row-major layout;
/// `point` is optional 2D point conditioning reserved by the frozen surface
/// (unused by the reference image-to-3D pipeline today).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trellis2PreparedImage {
    pub family: ModelFamily,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub uint8_rgb: Vec<u8>,
    pub point: Option<[f32; 2]>,
    /// Static description of the normalized pixel tensor consumed by DINOv3.
    pub pixel_values: TensorContract,
}

/// Static description of the mesh extracted from the decoded dual-grid
/// (O-Voxel) field.
///
/// `vertices` are flattened `[x, y, z]` f32 triples in the reference
/// `[-0.5, 0.5]^3` frame, `faces` are 0-based triangle indices,
/// `voxel_coords` are flattened active-voxel `[x, y, z]` i32 triples,
/// `voxel_attrs` are the per-voxel decoded attribute channels (7 for the
/// shape decode), and `resolution` is the voxel grid resolution the mesh was
/// extracted at.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trellis2Mesh {
    pub family: ModelFamily,
    pub vertices: Vec<f32>,
    pub faces: Vec<i64>,
    pub voxel_coords: Vec<i32>,
    pub voxel_attrs: Vec<f32>,
    pub resolution: u32,
}
