use crate::runtime::{VggtInferenceOutput, VggtPipeline, VggtPreparedInputs};
use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, TensorContract, VggtPreparedBatch, VggtSceneCode},
};

pub type VggtTensorOutput = VggtInferenceOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VggtNeuralStage;

impl NeuralStage<VggtPreparedBatch, VggtSceneCode> for VggtNeuralStage {
    fn infer(&self, input: &VggtPreparedBatch) -> Result<VggtSceneCode> {
        let frames = input.sampled_frames;
        let height = input.target_size.height as usize;
        let width = input.target_size.width as usize;

        Ok(VggtSceneCode {
            family: ModelFamily::Vggt,
            backbone: "dinov2_vitl14_reg",
            aggregator_depth: 24,
            camera_refine_iterations: 4,
            world_points: TensorContract::new(
                "vggt.world_points",
                vec![1, frames, height, width, 3],
            ),
            world_points_conf: TensorContract::new(
                "vggt.world_points_conf",
                vec![1, frames, height, width],
            ),
            depth: TensorContract::new("vggt.depth", vec![1, frames, height, width, 1]),
            depth_conf: TensorContract::new("vggt.depth_conf", vec![1, frames, height, width]),
            pose_enc: TensorContract::new("vggt.camera.pose_enc", vec![1, frames, 9]),
            extrinsics: TensorContract::new("vggt.camera.extrinsics", vec![1, frames, 3, 4]),
            intrinsics: TensorContract::new("vggt.camera.intrinsics", vec![1, frames, 3, 3]),
        })
    }
}

impl VggtNeuralStage {
    pub fn infer_tensors(
        &self,
        pipeline: &VggtPipeline,
        prepared: &VggtPreparedInputs,
    ) -> Result<VggtTensorOutput> {
        pipeline.infer_tensors(prepared)
    }
}
