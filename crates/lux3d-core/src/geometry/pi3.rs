use candle_core::{DType, Device};

use crate::{
    Result,
    contracts::{GeometryStage, ModelFamily, Pi3PointCloud, Pi3SceneCode},
    runtime::Pi3InferenceOutput,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Pi3PointCloudCpu {
    pub points: Vec<f32>,
    pub colors: Vec<u8>,
    pub vertex_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3GeometryStage;

impl GeometryStage<Pi3SceneCode, Pi3PointCloud> for Pi3GeometryStage {
    fn assemble(&self, scene: &Pi3SceneCode) -> Result<Pi3PointCloud> {
        let point_dims = scene.local_points.dims.to_vec();

        Ok(Pi3PointCloud {
            family: ModelFamily::Pi3,
            points: crate::contracts::TensorContract::new("pi3.points", point_dims),
            confidence_logits: scene.confidence_logits.clone(),
            assembly_path: "world_points = camera_poses * homogenize(local_points)",
            supports_sim3_fusion_extension: false,
        })
    }
}

impl Pi3GeometryStage {
    pub fn assemble_cpu(&self, output: &Pi3InferenceOutput) -> Result<Pi3PointCloudCpu> {
        let mask = output
            .export_mask
            .to_dtype(DType::U8)
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to cast export mask to u8")
            })?
            .flatten_all()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to flatten export mask"))?
            .to_vec1::<u8>()
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to read export mask values")
            })?;
        let points = output
            .points
            .to_device(&Device::Cpu)
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to move points to cpu"))?
            .flatten_all()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to flatten points"))?
            .to_vec1::<f32>()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to read point values"))?;
        let rgb = output
            .rgb_frames
            .to_device(&Device::Cpu)
            .map_err(|_| {
                crate::error::Lux3dError::InvalidInput("failed to move rgb frames to cpu")
            })?
            .permute((0, 2, 3, 1))
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to permute rgb frames"))?
            .flatten_all()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to flatten rgb frames"))?
            .to_vec1::<f32>()
            .map_err(|_| crate::error::Lux3dError::InvalidInput("failed to read rgb values"))?;

        let vertex_count = mask.iter().filter(|&&value| value != 0).count();
        let mut filtered_points = Vec::with_capacity(vertex_count * 3);
        let mut filtered_colors = Vec::with_capacity(vertex_count * 3);
        for (idx, &keep) in mask.iter().enumerate() {
            if keep == 0 {
                continue;
            }
            let offset = idx * 3;
            filtered_points.extend_from_slice(&points[offset..offset + 3]);
            filtered_colors.push((rgb[offset].clamp(0.0, 1.0) * 255.0).round() as u8);
            filtered_colors.push((rgb[offset + 1].clamp(0.0, 1.0) * 255.0).round() as u8);
            filtered_colors.push((rgb[offset + 2].clamp(0.0, 1.0) * 255.0).round() as u8);
        }

        Ok(Pi3PointCloudCpu {
            points: filtered_points,
            colors: filtered_colors,
            vertex_count,
        })
    }
}
