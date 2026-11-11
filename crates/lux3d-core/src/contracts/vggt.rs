use serde::{Deserialize, Serialize};

use crate::contracts::{ModelFamily, Pi3InputSource, RgbRange, SpatialSize, TensorContract};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VggtRawInput {
    pub source: Pi3InputSource,
    pub frame_count: usize,
    pub original_size: SpatialSize,
    pub interval: usize,
}

impl VggtRawInput {
    pub fn directory(frame_count: usize, width: u32, height: u32, interval: usize) -> Self {
        Self {
            source: Pi3InputSource::Directory,
            frame_count,
            original_size: SpatialSize::new(width, height),
            interval,
        }
    }

    pub fn video(frame_count: usize, width: u32, height: u32, interval: usize) -> Self {
        Self {
            source: Pi3InputSource::Video,
            frame_count,
            original_size: SpatialSize::new(width, height),
            interval,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VggtPreparedBatch {
    pub family: ModelFamily,
    pub source: Pi3InputSource,
    pub sampled_frames: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub interval: usize,
    pub pixel_range: RgbRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VggtSceneCode {
    pub family: ModelFamily,
    pub backbone: &'static str,
    pub aggregator_depth: usize,
    pub camera_refine_iterations: usize,
    pub world_points: TensorContract,
    pub world_points_conf: TensorContract,
    pub depth: TensorContract,
    pub depth_conf: TensorContract,
    pub pose_enc: TensorContract,
    pub extrinsics: TensorContract,
    pub intrinsics: TensorContract,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VggtPointCloud {
    pub family: ModelFamily,
    pub points: TensorContract,
    pub confidence: TensorContract,
    pub assembly_path: &'static str,
    pub supports_depth_unprojection: bool,
}
