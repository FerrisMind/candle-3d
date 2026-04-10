use std::path::Path;

use candle_core::Device;

use crate::{
    Result,
    contracts::{
        ModelFamily, Pi3OptionalConditions, Pi3PreparedBatch, Pi3RawInput, PreprocessStage,
        RgbRange, SpatialSize,
    },
    error::Lux3dError,
    runtime::Pi3PreparedInputs,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pi3PreprocessStage {
    patch_multiple: u32,
    default_pixel_limit: u32,
}

impl Default for Pi3PreprocessStage {
    fn default() -> Self {
        Self {
            patch_multiple: 14,
            default_pixel_limit: 255_000,
        }
    }
}

impl Pi3PreprocessStage {
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<Pi3PreparedInputs> {
        crate::runtime::prepare_pi3_inputs_with_stage(self, source, interval, device)
    }

    pub fn target_size_for(&self, original: SpatialSize, pixel_limit: u32) -> Result<SpatialSize> {
        if original.width == 0 || original.height == 0 {
            return Err(Lux3dError::InvalidInput("original size must be non-zero"));
        }

        let width = f64::from(original.width);
        let height = f64::from(original.height);
        let area = width * height;
        let scale = if area > 0.0 {
            (f64::from(pixel_limit) / area).sqrt()
        } else {
            1.0
        };
        let target_width = width * scale;
        let target_height = height * scale;
        let multiple = f64::from(self.patch_multiple);

        let mut k = (target_width / multiple).round().max(1.0);
        let mut m = (target_height / multiple).round().max(1.0);
        while (k * multiple) * (m * multiple) > f64::from(pixel_limit) {
            if (k / m) > (target_width / target_height) {
                k -= 1.0;
            } else {
                m -= 1.0;
            }
        }

        Ok(SpatialSize::new(
            (k.max(1.0) as u32) * self.patch_multiple,
            (m.max(1.0) as u32) * self.patch_multiple,
        ))
    }

    pub fn rescale_intrinsics(
        &self,
        mut intrinsics: [[f32; 3]; 3],
        original: SpatialSize,
        target: SpatialSize,
    ) -> [[f32; 3]; 3] {
        let scale_x = target.width as f32 / original.width as f32;
        let scale_y = target.height as f32 / original.height as f32;
        intrinsics[0][0] *= scale_x;
        intrinsics[0][2] *= scale_x;
        intrinsics[1][1] *= scale_y;
        intrinsics[1][2] *= scale_y;
        intrinsics
    }
}

impl PreprocessStage<Pi3RawInput, Pi3PreparedBatch> for Pi3PreprocessStage {
    fn preprocess(&self, input: Pi3RawInput) -> Result<Pi3PreparedBatch> {
        if input.interval == 0 {
            return Err(Lux3dError::InvalidInput("interval must be at least 1"));
        }
        let target_size = self.target_size_for(input.original_size, input.pixel_limit)?;
        let sampled_frames = input.frame_count.div_ceil(input.interval);
        let resize_scale = [
            target_size.width as f32 / input.original_size.width as f32,
            target_size.height as f32 / input.original_size.height as f32,
        ];

        Ok(Pi3PreparedBatch {
            family: ModelFamily::Pi3,
            source: input.source,
            sampled_frames,
            original_size: input.original_size,
            target_size,
            resize_scale,
            interval: input.interval,
            pixel_limit: if input.pixel_limit == 0 {
                self.default_pixel_limit
            } else {
                input.pixel_limit
            },
            pixel_range: RgbRange::ZeroToOne,
            optional_conditions: Pi3OptionalConditions::interface_ready(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescales_intrinsics_using_target_size() {
        let stage = Pi3PreprocessStage::default();
        let original = SpatialSize::new(1920, 1080);
        let target = stage
            .target_size_for(original, 255_000)
            .expect("target size");
        let intrinsics = [[500.0, 0.0, 960.0], [0.0, 500.0, 540.0], [0.0, 0.0, 1.0]];
        let scaled = stage.rescale_intrinsics(intrinsics, original, target);

        assert!((scaled[0][0] - 175.0).abs() < 1e-3);
        assert!((scaled[1][1] - 175.0).abs() < 1e-3);
    }
}
