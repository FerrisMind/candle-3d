use serde::{Deserialize, Serialize};

use crate::contracts::{ModelFamily, NormalizationStats, SpatialSize, TensorContract};

/// Raw image handed to the TripoSG preprocess stage.
///
/// `rgb` is the background-composited color used when the source carries no
/// usable alpha channel; `rgba` is set when the caller supplies premultiplied
/// source alpha for the foreground-crop/pad path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoSgRawInput {
    pub original_size: SpatialSize,
    pub rgb: Option<[f32; 3]>,
    pub rgba: Option<[f32; 4]>,
}

/// Prepared 224x224 image fed to the DINOv2 image encoder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoSgPreparedImage {
    pub family: ModelFamily,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub background_value: f32,
    pub normalization: NormalizationStats,
    pub uint8_rgb: Vec<u8>,
    pub pixel_values: TensorContract,
}

/// Static description of the latent scene produced by the DiT rectified-flow
/// denoise loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoSgSceneCode {
    pub family: ModelFamily,
    pub image_encoder: &'static str,
    pub dit_layers: usize,
    pub dit_width: usize,
    pub cross_attention_dim: usize,
    pub latent_tokens: usize,
    pub latent_channels: usize,
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
    pub latents: TensorContract,
}

/// Static description of the mesh extracted from the decoded SDF field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripoSgMesh {
    pub family: ModelFamily,
    pub surface_extractor: &'static str,
    pub default_dense_octree_depth: u32,
    pub default_hierarchical_octree_depth: u32,
    /// Reference `use_flash_decoder` default (true).
    pub default_use_flash_decoder: bool,
    /// Reference `flash_octree_depth` default (9).
    pub default_flash_octree_depth: u32,
    pub sdf_level: f32,
    pub bounds: [f32; 6],
}
