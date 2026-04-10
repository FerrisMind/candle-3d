use serde::{Deserialize, Serialize};

use crate::contracts::{
    ModelFamily, Pi3InputSource, Pi3OptionalConditions, RgbRange, SpatialSize, TensorContract,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pi3xPreparedBatch {
    pub family: ModelFamily,
    pub source: Pi3InputSource,
    pub sampled_frames: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub resize_scale: [f32; 2],
    pub interval: usize,
    pub pixel_limit: u32,
    pub pixel_range: RgbRange,
    pub optional_conditions: Pi3OptionalConditions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3xSceneCode {
    pub family: ModelFamily,
    pub encoder_backbone: &'static str,
    pub positional_encoding: &'static str,
    pub decoder_depth: usize,
    pub pose_inject_blocks: usize,
    pub metric_decoder: &'static str,
    pub local_points: TensorContract,
    pub confidence_logits: TensorContract,
    pub camera_poses: TensorContract,
    pub rays: TensorContract,
    pub metric: TensorContract,
    pub optional_conditions: Pi3OptionalConditions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3xPointCloud {
    pub family: ModelFamily,
    pub points: TensorContract,
    pub confidence_logits: TensorContract,
    pub assembly_path: &'static str,
    pub supports_sim3_fusion_extension: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3xVoPointCloud {
    pub family: ModelFamily,
    pub points: TensorContract,
    pub confidence_logits: TensorContract,
    pub camera_poses: TensorContract,
    pub assembly_path: &'static str,
    pub supports_sim3_fusion_extension: bool,
}
