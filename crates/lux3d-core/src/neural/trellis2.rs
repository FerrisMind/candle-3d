//! TRELLIS.2 neural stage: minimal static contract for the conditioning
//! stage (DINOv3 ViT-L/16 image features consumed by all five TRELLIS.2
//! flow DiTs) plus a helper that drives the DINOv3 encoder for a prepared
//! image. The DiT/sampler/VAE math lives in later `runtime/trellis2_*`
//! modules; nothing is duplicated here.

use candle_core::Tensor;

use crate::{
    Result,
    contracts::{ModelFamily, NeuralStage, TensorContract, Trellis2PreparedImage},
    runtime::{Trellis2Dino, Trellis2DinoConfig},
};

/// DINOv3 ViT-L/16 hidden size (`cond_channels` of every TRELLIS.2 DiT).
pub const TRELLIS2_COND_CHANNELS: usize = 1024;
/// DINOv3 ViT-L/16 layer count.
pub const TRELLIS2_DINO_LAYERS: usize = 24;
/// DINOv3 ViT-L/16 attention heads.
pub const TRELLIS2_DINO_HEADS: usize = 16;
/// DINOv3 patch size.
pub const TRELLIS2_DINO_PATCH: usize = 16;
/// DINOv3 register tokens (CLS + 4 registers prefix the patch tokens).
pub const TRELLIS2_DINO_REGISTERS: usize = 4;
/// Number of prefix tokens fed to the DiTs (CLS + registers).
pub const TRELLIS2_COND_PREFIX_TOKENS: usize = 1 + TRELLIS2_DINO_REGISTERS;
/// Conditioning token count for a square DINOv3 input side:
/// `5 + (resolution / 16)^2` (1029 at 512, 4101 at 1024).
pub fn trellis2_cond_token_count(resolution: u32) -> usize {
    TRELLIS2_COND_PREFIX_TOKENS + (resolution as usize / TRELLIS2_DINO_PATCH).pow(2)
}

/// Static description of the TRELLIS.2 conditioning tensors produced from a
/// prepared image. `cond` is the DINOv3 final hidden state
/// `[1, tokens, 1024]`; the reference `neg_cond` used by the sampler is a
/// zeros tensor of the same shape (described here as a contract, not
/// materialized).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2ConditioningCode {
    pub family: ModelFamily,
    pub image_encoder: &'static str,
    pub image_encoder_hidden: usize,
    pub image_encoder_layers: usize,
    pub image_encoder_registers: usize,
    pub image_encoder_patch_size: usize,
    pub resolution: u32,
    pub cond_tokens: usize,
    pub cond: TensorContract,
    pub neg_cond: TensorContract,
}

/// TRELLIS.2 conditioning stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trellis2NeuralStage;

impl Trellis2NeuralStage {
    /// Static contract view of the conditioning stage for a DINOv3 input
    /// resolution (512 or 1024). No file I/O and no model run.
    pub fn conditioning_code(&self, resolution: u32) -> Result<Trellis2ConditioningCode> {
        let tokens = trellis2_cond_token_count(resolution);
        Ok(Trellis2ConditioningCode {
            family: ModelFamily::Trellis2,
            image_encoder: "dinov3-vitl16",
            image_encoder_hidden: TRELLIS2_COND_CHANNELS,
            image_encoder_layers: TRELLIS2_DINO_LAYERS,
            image_encoder_registers: TRELLIS2_DINO_REGISTERS,
            image_encoder_patch_size: TRELLIS2_DINO_PATCH,
            resolution,
            cond_tokens: tokens,
            cond: TensorContract::new("trellis2.cond", vec![1, tokens, TRELLIS2_COND_CHANNELS]),
            neg_cond: TensorContract::new(
                "trellis2.neg_cond",
                vec![1, tokens, TRELLIS2_COND_CHANNELS],
            ),
        })
    }

    /// Run the DINOv3 encoder on normalized `[1, 3, res, res]` pixel values
    /// (from `Trellis2Preprocess::normalized_pixel_values`) and return the
    /// conditioning tensor `[1, tokens, 1024]` f32 (the sampler computes
    /// `neg_cond` as zeros of the same shape).
    pub fn infer_cond(&self, dino: &Trellis2Dino, pixel_values: &Tensor) -> Result<Tensor> {
        dino.forward(pixel_values).map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("TRELLIS.2 DINOv3 forward failed: {source}"),
            }
        })
    }
}

impl NeuralStage<Trellis2PreparedImage, Trellis2ConditioningCode> for Trellis2NeuralStage {
    fn infer(&self, input: &Trellis2PreparedImage) -> Result<Trellis2ConditioningCode> {
        let resolution = input.target_size.width;
        self.conditioning_code(resolution)
    }
}

/// Default DINOv3 configuration used by every TRELLIS.2 stage.
pub fn trellis2_dino_config() -> Trellis2DinoConfig {
    Trellis2DinoConfig::vitl16()
}

/// Reference `Trellis2NeuralSpec` facts mirrored as constants so contract
/// JSON writers and runtime stay in sync (T3+ DiT/sampler parameters are
/// filled by their own tasks).
pub const TRELLIS2_NEURAL_FACTS: Trellis2NeuralFacts = Trellis2NeuralFacts {
    image_encoder: "dinov3-vitl16",
    image_encoder_layers: TRELLIS2_DINO_LAYERS,
    image_encoder_hidden: TRELLIS2_COND_CHANNELS,
    image_encoder_patch_size: TRELLIS2_DINO_PATCH,
    image_encoder_registers: TRELLIS2_DINO_REGISTERS,
    flow_dit_layers: 30,
    flow_dit_width: 1536,
    flow_dit_heads: 12,
    flow_dit_head_dim: 128,
    cross_attention_dim: TRELLIS2_COND_CHANNELS,
    positional_encoding: "rope3d",
    sparse_structure_latent_channels: 8,
    sparse_structure_resolution: 16,
    slat_latent_channels: 32,
    default_steps: 12,
    default_guidance_scale: 7.5,
};

/// Compile-time mirror of the flow-DiT facts consumed by `Trellis2NeuralSpec`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Trellis2NeuralFacts {
    pub image_encoder: &'static str,
    pub image_encoder_layers: usize,
    pub image_encoder_hidden: usize,
    pub image_encoder_patch_size: usize,
    pub image_encoder_registers: usize,
    pub flow_dit_layers: usize,
    pub flow_dit_width: usize,
    pub flow_dit_heads: usize,
    pub flow_dit_head_dim: usize,
    pub cross_attention_dim: usize,
    pub positional_encoding: &'static str,
    pub sparse_structure_latent_channels: usize,
    pub sparse_structure_resolution: usize,
    pub slat_latent_channels: usize,
    pub default_steps: usize,
    pub default_guidance_scale: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_counts_match_reference_shapes() {
        assert_eq!(trellis2_cond_token_count(512), 1029);
        assert_eq!(trellis2_cond_token_count(1024), 4101);
    }

    #[test]
    fn conditioning_code_uses_reference_constants() {
        let code = Trellis2NeuralStage
            .conditioning_code(512)
            .expect("conditioning code");
        assert_eq!(code.family, ModelFamily::Trellis2);
        assert_eq!(code.cond_tokens, 1029);
        assert_eq!(code.cond.dims, vec![1, 1029, 1024]);
        assert_eq!(code.neg_cond.dims, vec![1, 1029, 1024]);
        assert_eq!(TRELLIS2_NEURAL_FACTS.image_encoder_hidden, 1024);
        assert_eq!(TRELLIS2_NEURAL_FACTS.flow_dit_width, 1536);
        assert_eq!(trellis2_dino_config(), Trellis2DinoConfig::vitl16());
    }

    #[test]
    fn neural_stage_infers_resolution_from_prepared_image() {
        // A minimal prepared image: the static contract only reads the size
        // fields, so empty pixel buffers are fine here.
        let prepared = Trellis2PreparedImage {
            family: ModelFamily::Trellis2,
            original_size: crate::contracts::SpatialSize::new(100, 100),
            target_size: crate::contracts::SpatialSize::new(512, 512),
            uint8_rgb: Vec::new(),
            point: None,
            pixel_values: TensorContract::new("pixel_values", vec![1, 3, 512, 512]),
        };
        let code = Trellis2NeuralStage.infer(&prepared).expect("code");
        assert_eq!(code.resolution, 512);
        assert_eq!(code.cond_tokens, 1029);
    }
}
