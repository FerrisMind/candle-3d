//! TripoSG DINOv2-large image encoder (standard HF `Dinov2Model`,
//! `facebook/dinov2-large` layout) loaded from the canonical
//! `image_encoder.*` weight subtree.
//!
//! Contract (`image_encoder_dinov2/config.json`): hidden 1024, 24 layers,
//! 16 heads (head dim 64), patch 14, MLP inner 4096, LayerNorm eps 1e-6,
//! qkv bias true, LayerScale `lambda1` per sublayer, final `layernorm`,
//! **no register tokens**, position table `[1, 1370, 1024]` (1 CLS + 37x37).
//!
//! Forward (`pipeline_triposg.py:138-149`): `pixel_values [B,3,224,224] F32`
//! -> `last_hidden_state [B,257,1024] F32` with the CLS token INCLUDED and
//! no pooling/L2 normalization. The 37x37 patch rows of the position table
//! are bicubic-resized (align_corners=False, HF formula) to the actual patch
//! grid (16x16 for the 224 input) in F32, with the CLS row prepended
//! unchanged.

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module, conv2d, layer_norm};
use crate::weights::VarBuilder;
use super::nn_blocks::{
    Linear,
    linear_dense as linear,
};

use super::{attention_math::exact_sdpa_heads, triposr::resize_patch_pos_embed_bicubic};

/// Hidden size (`hidden_size=1024`).
const HIDDEN_SIZE: usize = 1024;
/// Transformer layer count (`num_hidden_layers=24`).
const NUM_LAYERS: usize = 24;
/// Attention heads (`num_attention_heads=16`).
const NUM_HEADS: usize = 16;
/// Head dim (`hidden_size / num_attention_heads = 64`).
const HEAD_DIM: usize = 64;
/// Patch size (`patch_size=14`).
const PATCH_SIZE: usize = 14;
/// MLP inner width (`mlp_ratio=4`).
const MLP_INNER: usize = 4096;
/// LayerNorm epsilon (`layer_norm_eps=1e-6`).
const LN_EPS: f64 = 1e-6;
/// Native patch grid of the position table (`image_size=518`, 1370 = 1 + 37^2).
const SOURCE_GRID: usize = 37;

/// Standard HF `Dinov2Model` backbone for TripoSG conditioning.
///
/// Only `load` and `forward` are public; sibling runtime modules (and the
/// later TripoSG pipeline) drive it through those two entry points.
#[derive(Debug)]
pub struct TripoSgDinoEncoder {
    cls_token: Tensor,
    /// Loaded for key validation but unused at inference (masking only).
    #[allow(dead_code)]
    mask_token: Tensor,
    patch_projection: Conv2d,
    position_embeddings: Tensor,
    layers: Vec<DinoLayer>,
    layernorm: LayerNorm,
}

impl TripoSgDinoEncoder {
    /// Load from a VarBuilder rooted at the canonical `image_encoder.` prefix
    /// (i.e. pass `vb.pp("image_encoder")` from the canonical weight set).
    pub fn load(vb: VarBuilder) -> CandleResult<Self> {
        let embeddings = vb.pp("embeddings");
        let patch_projection = conv2d(
            3,
            HIDDEN_SIZE,
            PATCH_SIZE,
            Conv2dConfig {
                stride: PATCH_SIZE,
                ..Default::default()
            },
            embeddings.pp("patch_embeddings").pp("projection"),
        )?;
        let cls_token = embeddings.get((1, 1, HIDDEN_SIZE), "cls_token")?;
        let mask_token = embeddings.get((1, HIDDEN_SIZE), "mask_token")?;
        let position_embeddings = embeddings.get(
            (1, 1 + SOURCE_GRID * SOURCE_GRID, HIDDEN_SIZE),
            "position_embeddings",
        )?;

        let layers = (0..NUM_LAYERS)
            .map(|idx| {
                let layer = DinoLayer::load(vb.pp("encoder").pp("layer").pp(idx.to_string()))?;
                super::drain_upload_staging(vb.device())?;
                Ok(layer)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let layernorm = layer_norm(HIDDEN_SIZE, LN_EPS, vb.pp("layernorm"))?;

        Ok(Self {
            cls_token,
            mask_token,
            patch_projection,
            position_embeddings,
            layers,
            layernorm,
        })
    }

    /// `pixel_values [B,3,224,224] F32 -> last_hidden_state [B,257,1024] F32`.
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let (batch, channels, height, width) = pixel_values.dims4()?;
        if channels != 3 {
            candle_core::bail!("TripoSG DINOv2 expects 3 input channels, got {channels}");
        }
        if height % PATCH_SIZE != 0 || width % PATCH_SIZE != 0 {
            candle_core::bail!(
                "TripoSG DINOv2 input {height}x{width} is not divisible by patch size {PATCH_SIZE}"
            );
        }
        let grid_h = height / PATCH_SIZE;
        let grid_w = width / PATCH_SIZE;

        // 1. Patch embed: conv2d(3,1024,14,stride 14,bias) -> flatten 16x16 = 256 tokens.
        let patch_tokens = self
            .patch_projection
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let (_batch, seq, embed) = patch_tokens.dims3()?;
        if seq != grid_h * grid_w || embed != HIDDEN_SIZE {
            candle_core::bail!(
                "TripoSG DINOv2 patch-embed shape guard failed: expected sequence {} and \
                 hidden {HIDDEN_SIZE}, got ({seq}, {embed})",
                grid_h * grid_w
            );
        }

        // 2. Prepend the CLS token expanded to batch.
        let cls = self.cls_token.broadcast_as((batch, 1, embed))?;
        let mut hidden = Tensor::cat(&[&cls, &patch_tokens], 1)?;

        // 3. Position embedding: CLS row unchanged, patch rows bicubic-resized
        //    37x37 -> grid_h x grid_w (align_corners=False) in F32.
        let pos = self.interpolated_pos_encoding(grid_h, grid_w, hidden.device())?;
        if pos.dim(1)? != 1 + grid_h * grid_w {
            candle_core::bail!(
                "TripoSG DINOv2 position-embed shape guard failed: expected {} rows for a \
                 {grid_h}x{grid_w} patch grid, got {}",
                1 + grid_h * grid_w,
                pos.dim(1)?
            );
        }
        hidden = hidden.broadcast_add(&pos)?;

        // 4. 24 pre-norm blocks.
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        // 5. Final layernorm over the full 257-token sequence.
        self.layernorm.forward(&hidden)
    }

    /// HF `Dinov2Embeddings._interpolate_pos_encoding`: CLS row prepended
    /// unchanged, patch rows reshaped to the native square grid and resized
    /// bicubic (align_corners=False) in F32.
    fn interpolated_pos_encoding(
        &self,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> CandleResult<Tensor> {
        let (rows, hidden) = (self.position_embeddings.dim(1)?, HIDDEN_SIZE);
        let num_patches = rows - 1;
        let source = (num_patches as f64).sqrt();
        if source.fract() != 0.0 {
            candle_core::bail!(
                "TripoSG DINOv2 position table must hold 1 + N^2 patch rows, got {num_patches} \
                 patch rows"
            );
        }
        let source = source as usize;

        let cls = self.position_embeddings.i((.., ..1, ..))?;
        if grid_h == source && grid_w == source {
            return Ok(self.position_embeddings.clone());
        }

        let flat = self
            .position_embeddings
            .i((.., 1.., ..))?
            .reshape((num_patches, hidden))?
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let resized = resize_patch_pos_embed_bicubic(&flat, source, grid_h, grid_w, hidden);
        let resized = Tensor::from_vec(resized, (1, grid_h * grid_w, hidden), &Device::Cpu)?
            .to_device(device)?;
        Tensor::cat(&[&cls, &resized], 1)
    }
}

#[derive(Debug)]
struct DinoLayer {
    norm1: LayerNorm,
    attention: DinoAttention,
    layer_scale1: Tensor,
    norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    layer_scale2: Tensor,
}

impl DinoLayer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(HIDDEN_SIZE, LN_EPS, vb.pp("norm1"))?,
            attention: DinoAttention::load(vb.pp("attention"))?,
            layer_scale1: vb.get((HIDDEN_SIZE,), "layer_scale1.lambda1")?,
            norm2: layer_norm(HIDDEN_SIZE, LN_EPS, vb.pp("norm2"))?,
            mlp_fc1: linear(HIDDEN_SIZE, MLP_INNER, vb.pp("mlp").pp("fc1"))?,
            mlp_fc2: linear(MLP_INNER, HIDDEN_SIZE, vb.pp("mlp").pp("fc2"))?,
            layer_scale2: vb.get((HIDDEN_SIZE,), "layer_scale2.lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let attn = self.attention.forward(&self.norm1.forward(xs)?)?;
        let xs = (xs + attn.broadcast_mul(&self.layer_scale1)?)?;
        let mlp = self.mlp_fc2.forward(
            &self
                .mlp_fc1
                .forward(&self.norm2.forward(&xs)?)?
                .gelu_erf()?,
        )?;
        xs + mlp.broadcast_mul(&self.layer_scale2)?
    }
}

#[derive(Debug)]
struct DinoAttention {
    attention: DinoSelfAttention,
    output: Linear,
}

impl DinoAttention {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            attention: DinoSelfAttention::load(vb.pp("attention"))?,
            output: linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("output").pp("dense"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.output.forward(&self.attention.forward(xs)?)
    }
}

#[derive(Debug)]
struct DinoSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DinoSelfAttention {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            query: linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("query"))?,
            key: linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("key"))?,
            value: linear(HIDDEN_SIZE, HIDDEN_SIZE, vb.pp("value"))?,
            num_heads: NUM_HEADS,
            head_dim: HEAD_DIM,
        })
    }

    /// Standard head-major split (NOT the TripoSG processor qkv re-split),
    /// scale `1/sqrt(head_dim)`, exact F32 SDPA for numerical parity.
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        let q = self.to_heads(&self.query.forward(xs)?)?;
        let k = self.to_heads(&self.key.forward(xs)?)?;
        let v = self.to_heads(&self.value.forward(xs)?)?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let out = exact_sdpa_heads(&q, &k, &v, scale)?;
        out.transpose(1, 2)?.reshape((batch, seq, dim))
    }

    fn to_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        xs.reshape((batch, seq, self.num_heads, dim / self.num_heads))?
            .transpose(1, 2)
    }
}
