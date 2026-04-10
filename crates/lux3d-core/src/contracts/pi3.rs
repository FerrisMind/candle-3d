use serde::{Deserialize, Serialize};

use crate::contracts::{ModelFamily, Pi3InputSource, RgbRange, SpatialSize, TensorContract};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3OptionalConditions {
    pub depth: bool,
    pub intrinsics: bool,
    pub poses: bool,
    pub rays: bool,
}

impl Pi3OptionalConditions {
    pub const fn interface_ready() -> Self {
        Self {
            depth: true,
            intrinsics: true,
            poses: true,
            rays: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3RawInput {
    pub source: Pi3InputSource,
    pub frame_count: usize,
    pub original_size: SpatialSize,
    pub interval: usize,
    pub pixel_limit: u32,
}

impl Pi3RawInput {
    pub fn directory(frame_count: usize, width: u32, height: u32, interval: usize) -> Self {
        Self {
            source: Pi3InputSource::Directory,
            frame_count,
            original_size: SpatialSize::new(width, height),
            interval,
            pixel_limit: 255_000,
        }
    }

    pub fn video(frame_count: usize, width: u32, height: u32, interval: usize) -> Self {
        Self {
            source: Pi3InputSource::Video,
            frame_count,
            original_size: SpatialSize::new(width, height),
            interval,
            pixel_limit: 255_000,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pi3PreparedBatch {
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
pub struct Pi3SceneCode {
    pub family: ModelFamily,
    pub encoder_backbone: &'static str,
    pub positional_encoding: &'static str,
    pub decoder_depth: usize,
    pub local_points: TensorContract,
    pub confidence_logits: TensorContract,
    pub camera_poses: TensorContract,
    pub optional_conditions: Pi3OptionalConditions,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pi3PointCloud {
    pub family: ModelFamily,
    pub points: TensorContract,
    pub confidence_logits: TensorContract,
    pub assembly_path: &'static str,
    pub supports_sim3_fusion_extension: bool,
}
