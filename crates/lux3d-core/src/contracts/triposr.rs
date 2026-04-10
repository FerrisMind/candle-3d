use serde::{Deserialize, Serialize};

use crate::contracts::{ModelFamily, NormalizationStats, SpatialSize, TensorContract};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoRawInput {
    pub original_size: SpatialSize,
    pub rgb: [f32; 3],
    pub rgba: Option<[f32; 4]>,
}

impl TripoRawInput {
    pub fn new(original_size: SpatialSize, rgb: [f32; 3], rgba: Option<[f32; 4]>) -> Self {
        Self {
            original_size,
            rgb,
            rgba,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoPreparedImage {
    pub family: ModelFamily,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub background_value: f32,
    pub composited_rgb: [f32; 3],
    pub normalized_rgb: [f32; 3],
    pub normalization: NormalizationStats,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoSceneCode {
    pub family: ModelFamily,
    pub image_tokenizer: &'static str,
    pub transformer_layers: usize,
    pub cross_attention_dim: usize,
    pub triplane: TensorContract,
    pub radius: f32,
    pub density_activation: &'static str,
    pub density_bias: f32,
    pub samples_per_ray: usize,
    pub marching_cubes_threshold: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoMesh {
    pub family: ModelFamily,
    pub surface_extractor: &'static str,
    pub default_resolution: u32,
    pub vertex_color_mode: &'static str,
    pub texture_baking_extension: bool,
}
