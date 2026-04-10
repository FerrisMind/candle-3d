use candle_core::{Device, IndexOp, Tensor};

use crate::runtime::{TripoInferenceOutput, TripoPreparedInputs, TripoSrPipeline};
use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, TensorContract, TripoPreparedImage, TripoSceneCode},
};

#[derive(Debug, Clone, PartialEq)]
pub struct TripoDensityGridCpu {
    pub resolution: u32,
    pub threshold: f32,
    pub radius: f32,
    pub density_values: Vec<f32>,
}

pub type TripoSceneCodes = Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoNeuralStage;

impl NeuralStage<TripoPreparedImage, TripoSceneCode> for TripoNeuralStage {
    fn infer(&self, _input: &TripoPreparedImage) -> Result<TripoSceneCode> {
        Ok(TripoSceneCode {
            family: ModelFamily::TripoSr,
            image_tokenizer: "facebook/dino-vitb16",
            transformer_layers: 16,
            cross_attention_dim: 768,
            triplane: TensorContract::new("triposr.scene_codes", vec![1, 3, 40, 64, 64]),
            radius: 0.87,
            density_activation: "exp",
            density_bias: -1.0,
            samples_per_ray: 128,
            marching_cubes_threshold: 25.0,
        })
    }
}

impl TripoNeuralStage {
    pub fn infer_tensors(
        &self,
        pipeline: &TripoSrPipeline,
        prepared: &TripoPreparedInputs,
    ) -> Result<TripoInferenceOutput> {
        let (
            image_tokens,
            triplane_seed_tokens,
            backbone_tokens,
            detokenized_triplanes,
            scene_codes,
        ) = pipeline.scene_codes(&prepared.preprocessed_image)?;
        Ok(TripoInferenceOutput {
            preprocessed_image: prepared.preprocessed_image.clone(),
            image_tokens,
            triplane_seed_tokens,
            backbone_tokens,
            detokenized_triplanes,
            scene_codes,
            mesh: None,
        })
    }

    pub fn build_density_grid_cpu(
        &self,
        pipeline: &TripoSrPipeline,
        scene_codes: &TripoSceneCodes,
        resolution: u32,
        threshold: f32,
        chunk_size: usize,
    ) -> Result<TripoDensityGridCpu> {
        if resolution < 2 {
            return Err(crate::error::Lux3dError::InvalidInput(
                "TripoSR marching-cubes resolution must be at least 2",
            ));
        }
        let scene_code = scene_codes.i(0).map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to index TripoSR scene code batch: {source}"),
            }
        })?;
        let device = scene_code.device().clone();
        let total_positions = (resolution as usize).pow(3);
        let step = chunk_size.max(1);
        let mut density_values = vec![0f32; total_positions];
        let mut query_positions = Vec::with_capacity(step * 3);
        let mut start = 0usize;
        while start < total_positions {
            let len = (total_positions - start).min(step);
            append_dense_query_grid_chunk_positions(
                &mut query_positions,
                start,
                len,
                resolution,
                0.87,
            );
            let query_chunk =
                Tensor::from_slice(&query_positions, (len, 3), &device).map_err(|source| {
                    crate::error::Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to materialize TripoSR query chunk: {source}"),
                    }
                })?;
            let density_chunk = pipeline
                .query_triplane_density(&scene_code, &query_chunk, len)?
                .contiguous()
                .and_then(|tensor| tensor.to_device(&Device::Cpu))
                .and_then(|tensor| tensor.flatten_all())
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(
                    |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to gather TripoSR density chunk on cpu: {source}"),
                    },
                )?;
            density_values[start..start + len].copy_from_slice(&density_chunk);
            start += len;
        }

        Ok(TripoDensityGridCpu {
            resolution,
            threshold,
            radius: 0.87,
            density_values,
        })
    }

    pub fn sample_vertex_colors_cpu(
        &self,
        pipeline: &TripoSrPipeline,
        scene_codes: &TripoSceneCodes,
        vertices: &[f32],
        chunk_size: usize,
    ) -> Result<Vec<f32>> {
        if vertices.is_empty() {
            return Ok(Vec::new());
        }

        let scene_code = scene_codes.i(0).map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to index TripoSR scene code batch: {source}"),
            }
        })?;
        let step = chunk_size.max(1);
        let vertex_count = vertices.len() / 3;
        let mut colors = vec![0f32; vertices.len()];
        let mut start = 0usize;
        while start < vertex_count {
            let len = (vertex_count - start).min(step);
            let offset = start * 3;
            let end = offset + len * 3;
            let vertex_tensor =
                Tensor::from_slice(&vertices[offset..end], (len, 3), scene_code.device()).map_err(
                    |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "failed to materialize TripoSR vertex chunk for color query: {source}"
                        ),
                    },
                )?;
            let color_chunk = pipeline
                .query_triplane_color(&scene_code, &vertex_tensor, len)?
                .contiguous()
                .and_then(|tensor| tensor.to_device(&Device::Cpu))
                .and_then(|tensor| tensor.flatten_all())
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(
                    |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                        message: format!(
                            "failed to gather TripoSR vertex color chunk on cpu: {source}"
                        ),
                    },
                )?;
            colors[offset..end].copy_from_slice(&color_chunk);
            start += len;
        }

        Ok(colors
            .into_iter()
            .map(|value| (value.clamp(0.0, 1.0) * 255.0).round() / 255.0)
            .collect())
    }
}

fn append_dense_query_grid_chunk_positions(
    positions: &mut Vec<f32>,
    start: usize,
    len: usize,
    resolution: u32,
    radius: f32,
) {
    let resolution = resolution as usize;
    let plane = resolution * resolution;
    let denom = (resolution - 1) as f32;
    positions.clear();
    positions.reserve(len * 3);
    for index in start..start + len {
        let x = index / plane;
        let yz = index % plane;
        let y = yz / resolution;
        let z = yz % resolution;
        positions.push((x as f32 / denom) * 2.0 * radius - radius);
        positions.push((y as f32 / denom) * 2.0 * radius - radius);
        positions.push((z as f32 / denom) * 2.0 * radius - radius);
    }
}
