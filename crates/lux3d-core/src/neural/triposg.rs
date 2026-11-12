//! TripoSG neural stage: the static latent-scene contract for the DINOv2 +
//! DiT rectified-flow stack, plus a thin helper that drives the pipeline for
//! image embeds -> seeded latents -> denoise (no math is duplicated here —
//! everything delegates to [`TripoSgPipeline`]).

use candle_core::Tensor;

use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, TensorContract, TripoSgPreparedImage, TripoSgSceneCode},
    runtime::{
        TRIPOSG_LATENT_CHANNELS, TripoSgInferenceOutput, TripoSgPipeline, TripoSgRunOptions,
    },
};

/// Static description of the TripoSG latent scene produced by the DiT
/// rectified-flow denoise loop. Defaults match the reference pipeline call
/// (`num_inference_steps=50`, `guidance_scale=7.0`) and the shipped configs
/// (`dinov2-large` conditioning, 21 DiT blocks of width 2048,
/// cross-attention dim 1024, `[1, 2048, 64]` latents).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TripoSgNeuralStage;

impl NeuralStage<TripoSgPreparedImage, TripoSgSceneCode> for TripoSgNeuralStage {
    fn infer(&self, _input: &TripoSgPreparedImage) -> Result<TripoSgSceneCode> {
        Ok(TripoSgSceneCode {
            family: ModelFamily::TripoSg,
            image_encoder: "dinov2-large",
            dit_layers: 21,
            dit_width: 2048,
            cross_attention_dim: 1024,
            latent_tokens: 2048,
            latent_channels: 64,
            guidance_scale: 7.0,
            num_inference_steps: 50,
            latents: TensorContract::new("triposg.latents", vec![1, 2048, 64]),
        })
    }
}

impl TripoSgNeuralStage {
    /// Conditioning -> latents: DINOv2 image embeds, the Lux3D-seeded
    /// `[1, num_tokens, 64]` init latents, then the CFG denoise loop.
    /// Delegates every numerical step to the pipeline.
    pub fn infer_latents(
        &self,
        pipeline: &TripoSgPipeline,
        pixel_values: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        let image_embeds = pipeline.image_embeds(pixel_values)?;
        crate::runtime::log_wgpu_alloc(pixel_values.device(), "wgpu-dino");
        let latents_init = TripoSgPipeline::sample_latents(
            options.seed,
            options.num_tokens,
            TRIPOSG_LATENT_CHANNELS,
            pixel_values.device(),
        )?;
        let latents = pipeline.infer_with_latents(&image_embeds, &latents_init, options)?;
        crate::runtime::log_wgpu_alloc(pixel_values.device(), "wgpu-denoise");
        Ok(TripoSgInferenceOutput {
            image_embeds,
            latents_init,
            latents,
        })
    }
}
