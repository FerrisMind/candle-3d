use candle_core::Tensor;

use crate::runtime::{Pi3xInferenceOutput, Pi3xPipeline, Pi3xPreparedInputs, Pi3xVoOutput};
use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, Pi3xPreparedBatch, Pi3xSceneCode, TensorContract},
};

pub type Pi3xTensorOutput = Pi3xInferenceOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3xNeuralStage;

impl NeuralStage<Pi3xPreparedBatch, Pi3xSceneCode> for Pi3xNeuralStage {
    fn infer(&self, input: &Pi3xPreparedBatch) -> Result<Pi3xSceneCode> {
        let frames = input.sampled_frames;
        let height = input.target_size.height as usize;
        let width = input.target_size.width as usize;

        Ok(Pi3xSceneCode {
            family: ModelFamily::Pi3x,
            encoder_backbone: "dinov2_vitl14_reg",
            positional_encoding: "rope100 + projective_rope",
            decoder_depth: 36,
            pose_inject_blocks: 5,
            metric_decoder: "ContextOnlyTransformerDecoder + Linear",
            local_points: TensorContract::new(
                "pi3x.local_points",
                vec![1, frames, height, width, 3],
            ),
            confidence_logits: TensorContract::new(
                "pi3x.confidence_logits",
                vec![1, frames, height, width, 1],
            ),
            camera_poses: TensorContract::new("pi3x.camera_poses", vec![1, frames, 4, 4]),
            rays: TensorContract::new("pi3x.rays", vec![1, frames, height, width, 3]),
            metric: TensorContract::new("pi3x.metric", vec![1]),
            optional_conditions: input.optional_conditions,
        })
    }
}

impl Pi3xNeuralStage {
    pub fn infer_tensors(
        &self,
        pipeline: &Pi3xPipeline,
        prepared: &Pi3xPreparedInputs,
    ) -> Result<Pi3xTensorOutput> {
        pipeline.infer_prepared(prepared)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn infer_vo_from_path(
        &self,
        pipeline: &Pi3xPipeline,
        source: &std::path::Path,
        interval: Option<usize>,
        chunk_size: Option<usize>,
        overlap: Option<usize>,
        conf_threshold: Option<f32>,
        inject_conditions: crate::runtime::Pi3xInjectConditions,
        device: &candle_core::Device,
    ) -> Result<Pi3xVoOutput> {
        pipeline.infer_vo_from_path(
            source,
            interval,
            chunk_size,
            overlap,
            conf_threshold,
            inject_conditions,
            device,
        )
    }
}

#[allow(dead_code)]
fn _keep_tensor_import(_: &Tensor) {}
