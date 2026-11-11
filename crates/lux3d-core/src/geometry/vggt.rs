use candle_core::{DType, Device};

use crate::{
    Result,
    contracts::{GeometryStage, ModelFamily, VggtPointCloud, VggtSceneCode},
    error::Lux3dError,
    runtime::VggtInferenceOutput,
};

#[derive(Debug, Clone, PartialEq)]
pub struct VggtPointCloudCpu {
    pub points: Vec<f32>,
    pub colors: Vec<u8>,
    pub vertex_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VggtGeometryStage;

impl GeometryStage<VggtSceneCode, VggtPointCloud> for VggtGeometryStage {
    fn assemble(&self, scene: &VggtSceneCode) -> Result<VggtPointCloud> {
        let dims = &scene.world_points.dims;
        let (last, leading) = dims.split_last().ok_or_else(|| {
            Lux3dError::InvalidInput(
                "VGGT world_points contract must declare at least one dimension",
            )
        })?;
        if *last != 3 {
            return Err(Lux3dError::InvalidInput(
                "VGGT world_points contract must end with an xyz dimension of 3",
            ));
        }
        let vertex_count = leading.iter().product::<usize>();

        Ok(VggtPointCloud {
            family: ModelFamily::Vggt,
            points: crate::contracts::TensorContract::new("vggt.points", vec![vertex_count, 3]),
            confidence: scene.world_points_conf.clone(),
            assembly_path: "world_points = point_head(aggregated_tokens)",
            supports_depth_unprojection: true,
        })
    }
}

impl VggtGeometryStage {
    /// Filter `world_points` by the strict `world_points_conf > conf_threshold`
    /// mask and attach colors from the preprocessed RGB frames, matching the
    /// VGGT demo export (`world_points_conf > 1.0` by default).
    pub fn assemble_cpu(
        &self,
        output: &VggtInferenceOutput,
        conf_threshold: f32,
    ) -> Result<VggtPointCloudCpu> {
        // Strict `>`: a confidence exactly equal to the threshold is excluded.
        let mask = output
            .world_points_conf
            .gt(conf_threshold)
            .map_err(|_| Lux3dError::InvalidInput("failed to threshold VGGT world_points_conf"))?
            .to_dtype(DType::U8)
            .map_err(|_| Lux3dError::InvalidInput("failed to cast VGGT export mask to u8"))?
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT export mask to cpu"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT export mask"))?
            .to_vec1::<u8>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT export mask values"))?;
        let points = output
            .world_points
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT world points to cpu"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT world points"))?
            .to_vec1::<f32>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT world point values"))?;
        let rgb = output
            .rgb_frames
            .to_device(&Device::Cpu)
            .map_err(|_| Lux3dError::InvalidInput("failed to move VGGT rgb frames to cpu"))?
            .permute((0, 2, 3, 1))
            .map_err(|_| Lux3dError::InvalidInput("failed to permute VGGT rgb frames"))?
            .flatten_all()
            .map_err(|_| Lux3dError::InvalidInput("failed to flatten VGGT rgb frames"))?
            .to_vec1::<f32>()
            .map_err(|_| Lux3dError::InvalidInput("failed to read VGGT rgb values"))?;

        // `[1,S,H,W,3]` and `[S,H,W,3]` share the same row-major flat order.
        if points.len() != mask.len() * 3 || rgb.len() != mask.len() * 3 {
            return Err(Lux3dError::InvalidInput(
                "VGGT world points / rgb frames do not match export mask element count",
            ));
        }

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

        Ok(VggtPointCloudCpu {
            points: filtered_points,
            colors: filtered_colors,
            vertex_count,
        })
    }
}
