use crate::runtime::{Pi3Pipeline, Pi3PreparedInputs};
use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, Pi3PreparedBatch, Pi3SceneCode, TensorContract},
};

pub type Pi3TensorOutput = crate::runtime::Pi3InferenceOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3NeuralStage;

impl NeuralStage<Pi3PreparedBatch, Pi3SceneCode> for Pi3NeuralStage {
    fn infer(&self, input: &Pi3PreparedBatch) -> Result<Pi3SceneCode> {
        let frames = input.sampled_frames;
        let height = input.target_size.height as usize;
        let width = input.target_size.width as usize;

        Ok(Pi3SceneCode {
            family: ModelFamily::Pi3,
            encoder_backbone: "dinov2_vitl14_reg",
            positional_encoding: "rope100",
            decoder_depth: 36,
            local_points: TensorContract::new(
                "pi3.local_points",
                vec![1, frames, height, width, 3],
            ),
            confidence_logits: TensorContract::new(
                "pi3.confidence_logits",
                vec![1, frames, height, width, 1],
            ),
            camera_poses: TensorContract::new("pi3.camera_poses", vec![1, frames, 4, 4]),
            optional_conditions: input.optional_conditions,
        })
    }
}

impl Pi3NeuralStage {
    pub fn infer_tensors(
        &self,
        pipeline: &Pi3Pipeline,
        prepared: &Pi3PreparedInputs,
    ) -> Result<Pi3TensorOutput> {
        let frame_count = prepared.rgb_frames.dim(0).map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to read Pi3 frame count: {source}"),
            }
        })?;
        let height = prepared.target_size.height as usize;
        let width = prepared.target_size.width as usize;

        let patch_tokens = pipeline.encode_patch_tokens(&prepared.normalized_frames)?;
        let decoder_hidden = pipeline.decode_hidden(&patch_tokens, frame_count, height, width)?;
        let decoder_positions =
            pipeline.decode_positions_only(&patch_tokens, frame_count, height, width)?;
        let point_hidden = pipeline.point_decoder_hidden(&decoder_hidden, &decoder_positions)?;
        let conf_hidden = pipeline.conf_decoder_hidden(&decoder_hidden, &decoder_positions)?;
        let camera_hidden = pipeline.camera_decoder_hidden(&decoder_hidden, &decoder_positions)?;

        let local_points = pipeline
            .local_points_from_head_output(&pipeline.point_head_output(
                &point_hidden,
                height,
                width,
            )?)?
            .reshape((1, frame_count, height, width, 3))
            .map_err(
                |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to reshape Pi3 local_points: {source}"),
                },
            )?;

        let confidence_logits = pipeline
            .conf_head_output(&conf_hidden, height, width)?
            .reshape((1, frame_count, height, width, 1))
            .map_err(
                |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to reshape Pi3 confidence_logits: {source}"),
                },
            )?;

        let camera_poses = pipeline
            .camera_poses_from_hidden(&camera_hidden, height / 14, width / 14)?
            .reshape((1, frame_count, 4, 4))
            .map_err(
                |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to reshape Pi3 camera_poses: {source}"),
                },
            )?;

        let points = pipeline.world_points(&local_points, &camera_poses)?;
        let export_mask = pipeline.export_mask(&local_points, &confidence_logits)?;

        Ok(Pi3TensorOutput {
            local_points,
            confidence_logits,
            camera_poses,
            points,
            export_mask,
            rgb_frames: prepared.rgb_frames.clone(),
        })
    }
}
