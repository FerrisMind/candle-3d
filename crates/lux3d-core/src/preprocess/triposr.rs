use std::path::Path;

use candle_core::Device;

use crate::{
    Result,
    contracts::{
        ModelFamily, NormalizationStats, PreprocessStage, SpatialSize, TripoPreparedImage,
        TripoRawInput,
    },
    runtime::TripoPreparedInputs,
};

#[derive(Debug, Clone, PartialEq)]
pub struct TripoPreprocessStage {
    cond_image_size: u32,
    background_value: f32,
    normalization: NormalizationStats,
}

impl Default for TripoPreprocessStage {
    fn default() -> Self {
        Self {
            cond_image_size: 512,
            background_value: 0.5,
            normalization: NormalizationStats::dino(),
        }
    }
}

impl TripoPreprocessStage {
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoPreparedInputs> {
        crate::runtime::prepare_triposr_inputs_with_stage(self, source, device)
    }

    pub(crate) const fn target_edge(&self) -> u32 {
        self.cond_image_size
    }

    pub(crate) const fn background_value(&self) -> f32 {
        self.background_value
    }

    pub fn composite_over_gray(&self, rgba: [f32; 4]) -> [f32; 3] {
        let alpha = rgba[3];
        [
            rgba[0] * alpha + (1.0 - alpha) * self.background_value,
            rgba[1] * alpha + (1.0 - alpha) * self.background_value,
            rgba[2] * alpha + (1.0 - alpha) * self.background_value,
        ]
    }

    pub fn normalize_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        [
            (rgb[0] - self.normalization.mean[0]) / self.normalization.std[0],
            (rgb[1] - self.normalization.mean[1]) / self.normalization.std[1],
            (rgb[2] - self.normalization.mean[2]) / self.normalization.std[2],
        ]
    }

    pub fn preprocess_rgb(&self, rgb: [f32; 3], rgba: [f32; 4]) -> Result<TripoPreparedImage> {
        self.preprocess(TripoRawInput::new(SpatialSize::new(1, 1), rgb, Some(rgba)))
    }
}

impl PreprocessStage<TripoRawInput, TripoPreparedImage> for TripoPreprocessStage {
    fn preprocess(&self, input: TripoRawInput) -> Result<TripoPreparedImage> {
        let composited_rgb = input
            .rgba
            .map(|rgba| self.composite_over_gray(rgba))
            .unwrap_or(input.rgb);
        let normalized_rgb = self.normalize_rgb(composited_rgb);

        Ok(TripoPreparedImage {
            family: ModelFamily::TripoSr,
            original_size: input.original_size,
            target_size: SpatialSize::new(self.cond_image_size, self.cond_image_size),
            background_value: self.background_value,
            composited_rgb,
            normalized_rgb,
            normalization: self.normalization.clone(),
        })
    }
}
