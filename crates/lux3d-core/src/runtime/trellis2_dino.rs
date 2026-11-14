//! TRELLIS.2 DINOv3 ViT-L/16 image-conditioning encoder
//! (transformers `DINOv3ViTModel`, weights
//! `models/TRELLIS.2-4B/external/dinov3/model.safetensors`, 415 tensors, all
//! F32, hidden 1024).
//!
//! Contract (`external/dinov3/config.json`): hidden 1024, 24 layers, 16
//! heads (head dim 64), MLP 1024 -> 4096 -> 1024 with exact-erf GELU, patch
//! 16, 4 register tokens, `layer_norm_eps = 1e-5`, layerscale value 1.0,
//! `rope_theta = 100.0`, query/value/output projections biased, **key
//! projection has no bias** (`key_bias: false`).
//!
//! Forward (reference `DinoV3FeatureExtractor.extract_features`,
//! `trellis2/modules/image_feature_extractor.py:81-92`):
//!
//! 1. `embeddings(pixel_values)`: `Conv2d(3, 1024, 16, stride 16)` + bias,
//!    flatten `H*W` row-major, then tokens = `[CLS, 4 registers, patches]`
//!    (no additive position embedding — DINOv3 uses RoPE);
//! 2. `rope_embeddings(pixel_values)`: 2D rotary embedding over patch
//!    centers, `inv_freq = 1 / theta ** arange(0, 1, 4 / head_dim)` (16
//!    frequencies), angles `= 2*pi * coord * inv_freq` for the `(y, x)`
//!    patch-center coordinates normalized to `[-1, 1]`, flattened to 32 and
//!    tiled twice to 64; computed in f32 with autocast disabled;
//! 3. 24 layers of `x += layer_scale1(attn(norm1(x)))` and
//!    `x += layer_scale2(mlp(norm2(x)))`. RoPE is applied to q/k **patch
//!    tokens only** (prefix CLS/register tokens pass through untouched)
//!    with the `rotate_half` convention `cat(-x2, x1)`; attention scaling
//!    is `head_dim ** -0.5 = 1/8`;
//! 4. final **non-affine** `F.layer_norm` over the last dim — the stored
//!    `norm.weight`/`norm.bias` tensors are intentionally unused by the
//!    reference extractor, so the port reproduces the weight-free norm.
//!
//! Output: `[B, 5 + (res / 16)^2, 1024]` f32 — the conditioning tensor for
//! every TRELLIS.2 DiT (1029 tokens at 512, 4101 at 1024). Per locked
//! decision 3 the VarBuilder is expected to materialize F32
//! (`VarBuilder::from_mmaped_safetensors(.., DType::F32, ..)` casts on
//! access); weights are additionally cast defensively.

use std::path::Path;

use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Module, conv2d, layer_norm};

use super::attention_math::exact_sdpa_heads;
use crate::weights::VarBuilder;

/// DINOv3 ViT-L/16 architecture parameters (TRELLIS.2 conditioning model).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2DinoConfig {
    /// `hidden_size` (1024).
    pub hidden_size: usize,
    /// `num_hidden_layers` (24).
    pub num_hidden_layers: usize,
    /// `num_attention_heads` (16).
    pub num_attention_heads: usize,
    /// `intermediate_size` (4096).
    pub intermediate_size: usize,
    /// `patch_size` (16).
    pub patch_size: usize,
    /// `num_register_tokens` (4).
    pub num_register_tokens: usize,
    /// `layer_norm_eps` (1e-5).
    pub layer_norm_eps: f64,
    /// `rope_theta` (100.0).
    pub rope_theta: f64,
}

impl Default for Trellis2DinoConfig {
    fn default() -> Self {
        Self::vitl16()
    }
}

impl Trellis2DinoConfig {
    /// `facebook/dinov3-vitl16-pretrain-lvd1689m` layout used by TRELLIS.2.
    pub const fn vitl16() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            patch_size: 16,
            num_register_tokens: 4,
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
        }
    }

    /// `hidden_size / num_attention_heads` (64).
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Total token count for a square input: CLS + registers + patches.
    pub fn token_count(&self, resolution: u32) -> usize {
        1 + self.num_register_tokens + (resolution as usize / self.patch_size).pow(2)
    }

    /// Parse the HF `config.json` shipped next to the DINOv3 weights.
    /// Unsupported architecture options (`key_bias`, gated MLP) are rejected
    /// instead of silently mismatching the reference.
    pub fn from_config_json(path: &Path) -> CandleResult<Self> {
        let file = std::fs::File::open(path).map_err(candle_core::Error::from)?;
        let value: serde_json::Value = serde_json::from_reader(file)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        let field = |name: &str| -> CandleResult<f64> {
            value.get(name).and_then(|v| v.as_f64()).ok_or_else(|| {
                candle_core::Error::Msg(format!("DINOv3 config.json is missing `{name}`"))
            })
        };
        if value.get("key_bias").and_then(|v| v.as_bool()) == Some(true) {
            candle_core::bail!("DINOv3 config with key_bias=true is not supported");
        }
        if value.get("use_gated_mlp").and_then(|v| v.as_bool()) == Some(true) {
            candle_core::bail!("DINOv3 config with use_gated_mlp=true is not supported");
        }
        Ok(Self {
            hidden_size: field("hidden_size")? as usize,
            num_hidden_layers: field("num_hidden_layers")? as usize,
            num_attention_heads: field("num_attention_heads")? as usize,
            intermediate_size: field("intermediate_size")? as usize,
            patch_size: field("patch_size")? as usize,
            num_register_tokens: value
                .get("num_register_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            layer_norm_eps: field("layer_norm_eps")?,
            rope_theta: field("rope_theta")?,
        })
    }

    /// 2D RoPE tables (`DINOv3ViTRopePositionEmbedding.forward`), computed
    /// in f32 like the reference (autocast disabled):
    /// - patch centers `coords = (i + 0.5) / n` mapped to `[-1, 1]`,
    ///   meshgrid `indexing="ij"` over `(y, x)`, flattened row-major;
    /// - `inv_freq = 1 / theta ** arange(0, 1, 4 / head_dim)` (f32);
    /// - `angles = (2*pi * coord) * inv_freq` per axis, flattened
    ///   `(P, 2, F) -> (P, 2F)` and `tile(2) -> (P, head_dim)`.
    ///
    /// Returns `(cos, sin)`, each `[P, head_dim]` f32.
    pub fn rope_cos_sin(
        &self,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let head_dim = self.head_dim();
        let freq_count = head_dim / 4;
        if freq_count * 4 != head_dim {
            candle_core::bail!(
                "DINOv3 head_dim {head_dim} is not divisible by 4; RoPE layout undefined"
            );
        }
        let theta = self.rope_theta as f32;
        let mut inv_freq = Vec::with_capacity(freq_count);
        for i in 0..freq_count {
            // torch.arange(0, 1, 4/head_dim, dtype=f32) values are exact
            // (i * 4/head_dim with head_dim % 4 == 0), then `base ** t` in f32.
            let exponent = (i as f32) * (4.0 / head_dim as f32);
            inv_freq.push(1.0 / theta.powf(exponent));
        }
        let two_pi = (std::f64::consts::PI * 2.0) as f32;
        let patches = grid_h * grid_w;
        let mut angles = Vec::with_capacity(patches * head_dim);
        for y in 0..grid_h {
            let coord_y = 2.0 * ((y as f32 + 0.5) / grid_h as f32) - 1.0;
            for x in 0..grid_w {
                let coord_x = 2.0 * ((x as f32 + 0.5) / grid_w as f32) - 1.0;
                // flatten(1, 2) of (P, 2, F): y frequencies then x frequencies.
                for coord in [coord_y, coord_x] {
                    let base = two_pi * coord;
                    for &frequency in &inv_freq {
                        angles.push(base * frequency);
                    }
                }
            }
        }
        // Build (P, 32) then tile to (P, 64) exactly like `angles.tile(2)`.
        let half_len = freq_count * 2;
        let mut tiled = Vec::with_capacity(patches * head_dim);
        for row in 0..patches {
            let start = row * half_len;
            tiled.extend_from_slice(&angles[start..start + half_len]);
            tiled.extend_from_slice(&angles[start..start + half_len]);
        }
        let angles = Tensor::from_vec(tiled, (patches, head_dim), device)?;
        Ok((angles.cos()?, angles.sin()?))
    }
}

/// DINOv3 ViT backbone producing the TRELLIS.2 conditioning tensor. Only
/// `load` and `forward` are public.
#[derive(Debug)]
pub struct Trellis2Dino {
    config: Trellis2DinoConfig,
    cls_token: Tensor,
    register_tokens: Tensor,
    patch_embeddings: Conv2d,
    layers: Vec<DinoLayer>,
    // The stored `norm.weight`/`norm.bias` are unused by the reference
    // extractor (final norm is non-affine); intentionally not loaded.
}

impl Trellis2Dino {
    /// Load from a VarBuilder whose tensor names start at
    /// `embeddings.` / `layer.{i}.` / `norm.` (the raw DINOv3 checkpoint
    /// layout; for the canonical package pass `vb.pp("image_cond")`).
    ///
    /// Weight names: `embeddings.{cls_token,register_tokens,
    /// patch_embeddings.weight,patch_embeddings.bias}`,
    /// `layer.{i}.{norm1,norm2}.{weight,bias}`,
    /// `layer.{i}.attention.{q,k,v,o}_proj.{weight,bias}` (k_proj has no
    /// bias), `layer.{i}.mlp.{up,down}_proj.{weight,bias}`,
    /// `layer.{i}.layer_scale{1,2}.lambda1`.
    pub fn load(vb: VarBuilder, config: &Trellis2DinoConfig) -> CandleResult<Self> {
        let hidden = config.hidden_size;
        let embeddings = vb.pp("embeddings");
        let cls_token = embeddings.get((1, 1, hidden), "cls_token")?;
        let register_tokens =
            embeddings.get((1, config.num_register_tokens, hidden), "register_tokens")?;
        let patch_embeddings = conv2d(
            3,
            hidden,
            config.patch_size,
            Conv2dConfig {
                stride: config.patch_size,
                ..Default::default()
            },
            embeddings.pp("patch_embeddings"),
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(DinoLayer::load(
                vb.pp("layer").pp(index.to_string()),
                config,
            )?);
            super::drain_upload_staging(vb.device())?;
        }

        Ok(Self {
            config: config.clone(),
            cls_token,
            register_tokens,
            patch_embeddings,
            layers,
        })
    }

    /// Architecture parameters in use.
    pub fn config(&self) -> &Trellis2DinoConfig {
        &self.config
    }

    /// `pixel_values [B, 3, H, W]` f32 -> conditioning `[B, 5 + (H/16)^2,
    /// 1024]` f32 (final hidden states after the non-affine layer norm,
    /// CLS/register tokens included).
    pub fn forward(&self, pixel_values: &Tensor) -> CandleResult<Tensor> {
        let (batch, channels, height, width) = pixel_values.dims4()?;
        if channels != 3 {
            candle_core::bail!("DINOv3 expects 3 input channels, got {channels}");
        }
        let patch = self.config.patch_size;
        if height % patch != 0 || width % patch != 0 {
            candle_core::bail!(
                "DINOv3 input {height}x{width} is not divisible by patch size {patch}"
            );
        }
        let grid_h = height / patch;
        let grid_w = width / patch;

        // 1. Patch embedding: conv -> flatten H*W row-major -> [B, P, hidden].
        let patch_tokens = self
            .patch_embeddings
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let (_batch, patches, embed) = patch_tokens.dims3()?;
        if patches != grid_h * grid_w || embed != self.config.hidden_size {
            candle_core::bail!(
                "DINOv3 patch-embed shape guard failed: expected ({}, {}), got ({patches}, {embed})",
                grid_h * grid_w,
                self.config.hidden_size
            );
        }

        // 2. Token order: CLS, registers, patches.
        let cls = self.cls_token.broadcast_as((batch, 1, embed))?;
        let registers =
            self.register_tokens
                .broadcast_as((batch, self.config.num_register_tokens, embed))?;
        let mut hidden = Tensor::cat(&[&cls, &registers, &patch_tokens], 1)?;

        // 3. RoPE tables over the patch grid (f32).
        let (cos, sin) = self
            .config
            .rope_cos_sin(grid_h, grid_w, pixel_values.device())?;

        // 4. 24 pre-norm layers.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &cos, &sin)?;
        }

        // 5. Final non-affine layer norm (the reference calls
        //    F.layer_norm without weight/bias).
        non_affine_layer_norm(&hidden, self.config.layer_norm_eps)
    }
}

#[derive(Debug)]
struct DinoLayer {
    norm1: LayerNorm,
    attention: DinoAttention,
    layer_scale1: Tensor,
    norm2: LayerNorm,
    mlp_up: candle_nn::Linear,
    mlp_down: candle_nn::Linear,
    layer_scale2: Tensor,
}

impl DinoLayer {
    fn load(vb: VarBuilder, config: &Trellis2DinoConfig) -> CandleResult<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            norm1: layer_norm(hidden, config.layer_norm_eps, vb.pp("norm1"))?,
            attention: DinoAttention::load(vb.pp("attention"), config)?,
            layer_scale1: load_f32(vb.pp("layer_scale1"), (hidden,), "lambda1")?,
            norm2: layer_norm(hidden, config.layer_norm_eps, vb.pp("norm2"))?,
            mlp_up: linear_f32(
                vb.pp("mlp").pp("up_proj"),
                hidden,
                config.intermediate_size,
                true,
            )?,
            mlp_down: linear_f32(
                vb.pp("mlp").pp("down_proj"),
                config.intermediate_size,
                hidden,
                true,
            )?,
            layer_scale2: load_f32(vb.pp("layer_scale2"), (hidden,), "lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        // x += layer_scale1(attn(norm1(x)))
        let normed = self.norm1.forward(xs)?;
        let attn = self.attention.forward(&normed, cos, sin)?;
        let xs = (xs + attn.broadcast_mul(&self.layer_scale1)?)?;
        // x += layer_scale2(mlp(norm2(x)))
        let normed2 = self.norm2.forward(&xs)?;
        let mlp = self
            .mlp_down
            .forward(&self.mlp_up.forward(&normed2)?.gelu_erf()?)?;
        xs + mlp.broadcast_mul(&self.layer_scale2)?
    }
}

#[derive(Debug)]
struct DinoAttention {
    query: candle_nn::Linear,
    /// k_proj has NO bias (`key_bias: false`).
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    output: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    num_prefix_tokens: usize,
}

impl DinoAttention {
    fn load(vb: VarBuilder, config: &Trellis2DinoConfig) -> CandleResult<Self> {
        let hidden = config.hidden_size;
        Ok(Self {
            query: linear_f32(vb.pp("q_proj"), hidden, hidden, true)?,
            key: linear_f32(vb.pp("k_proj"), hidden, hidden, false)?,
            value: linear_f32(vb.pp("v_proj"), hidden, hidden, true)?,
            output: linear_f32(vb.pp("o_proj"), hidden, hidden, true)?,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            num_prefix_tokens: 1 + config.num_register_tokens,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        let query = self.to_heads(&self.query.forward(xs)?)?;
        let key = self.to_heads(&self.key.forward(xs)?)?;
        let value = self.to_heads(&self.value.forward(xs)?)?;
        let (query, key) = apply_rotary_to_patches(&query, &key, cos, sin, self.num_prefix_tokens)?;
        // scaling = head_dim ** -0.5 (1/8 for head_dim 64).
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let out = exact_sdpa_heads(&query, &key, &value, scale)?;
        let out = out.transpose(1, 2)?.reshape((batch, seq, dim))?;
        self.output.forward(&out)
    }

    fn to_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        xs.reshape((batch, seq, self.num_heads, dim / self.num_heads))?
            .transpose(1, 2)
    }
}

/// `apply_rotary_pos_emb` (DINOv3): rotate q/k patch tokens only; the
/// leading `num_prefix` CLS/register tokens pass through untouched.
fn apply_rotary_to_patches(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    num_prefix: usize,
) -> CandleResult<(Tensor, Tensor)> {
    let seq = q.dim(2)?;
    let patches = seq - num_prefix;
    let head_dim = q.dim(3)?;
    let cos = cos.reshape((1, 1, patches, head_dim))?;
    let sin = sin.reshape((1, 1, patches, head_dim))?;

    let q_prefix = q.narrow(2, 0, num_prefix)?;
    let k_prefix = k.narrow(2, 0, num_prefix)?;
    let q_patches = q.narrow(2, num_prefix, patches)?.contiguous()?;
    let k_patches = k.narrow(2, num_prefix, patches)?.contiguous()?;

    // q = (q * cos) + (rotate_half(q) * sin)
    let q_rotated = q_patches
        .broadcast_mul(&cos)?
        .add(&rotate_half(&q_patches)?.broadcast_mul(&sin)?)?;
    let k_rotated = k_patches
        .broadcast_mul(&cos)?
        .add(&rotate_half(&k_patches)?.broadcast_mul(&sin)?)?;

    Ok((
        Tensor::cat(&[&q_prefix, &q_rotated], 2)?,
        Tensor::cat(&[&k_prefix, &k_rotated], 2)?,
    ))
}

/// `rotate_half`: split the last dim in half and return `cat(-x2, x1)`.
fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
    let dim = x.dim(D::Minus1)?;
    let half = dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, dim - half)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// `F.layer_norm(x, (hidden,))` without affine parameters.
fn non_affine_layer_norm(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let hidden = xs.dim(D::Minus1)?;
    let device = xs.device();
    let weight = Tensor::ones((hidden,), DType::F32, device)?;
    let bias = Tensor::zeros((hidden,), DType::F32, device)?;
    LayerNorm::new(weight, bias, eps).forward(xs)
}

fn load_f32(
    vb: VarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> CandleResult<Tensor> {
    let shape = shape.into();
    let tensor = vb.get(shape, name)?;
    if tensor.dtype() == DType::F32 {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::F32)
    }
}

/// Dense linear loaded from the stored dtype and cast to F32 (locked
/// decision 3: all compute is F32).
fn linear_f32(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> CandleResult<candle_nn::Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let weight = if weight.dtype() == DType::F32 {
        weight
    } else {
        weight.to_dtype(DType::F32)?
    };
    let bias = if bias {
        let bias = vb.get(out_dim, "bias")?;
        Some(if bias.dtype() == DType::F32 {
            bias
        } else {
            bias.to_dtype(DType::F32)?
        })
    } else {
        None
    };
    Ok(candle_nn::Linear::new(weight, bias))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny config: hidden 16, 2 heads -> head_dim 8, so inv_freq has
    /// head_dim/4 = 2 entries and the angle layout is fully hand-checkable.
    fn tiny_config() -> Trellis2DinoConfig {
        Trellis2DinoConfig {
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 32,
            patch_size: 4,
            num_register_tokens: 2,
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
        }
    }

    #[test]
    fn rope_tables_match_hand_computed_cos_sin() {
        let config = tiny_config();
        // head_dim 8 -> inv_freq = 1 / 100 ** arange(0, 1, 0.5) = [1, 0.1].
        let grid = 2; // 2x2 patch grid -> 4 patch tokens
        let (cos, sin) = config.rope_cos_sin(grid, grid, &Device::Cpu).expect("rope");
        assert_eq!(cos.dims(), &[4, 8]);
        assert_eq!(sin.dims(), &[4, 8]);

        let inv_freq = [1.0_f32, 1.0 / 100.0_f32.powf(0.5)];
        let two_pi = (std::f64::consts::PI * 2.0) as f32;
        let mut expected_rows = Vec::new();
        for y in 0..grid {
            let cy = 2.0 * ((y as f32 + 0.5) / grid as f32) - 1.0;
            for x in 0..grid {
                let cx = 2.0 * ((x as f32 + 0.5) / grid as f32) - 1.0;
                // flatten(y freqs, x freqs) -> 4 angles, then tile(2) -> 8.
                let mut half = Vec::new();
                for coord in [cy, cx] {
                    let base = two_pi * coord;
                    for &f in &inv_freq {
                        half.push(base * f);
                    }
                }
                let mut row = half.clone();
                row.extend_from_slice(&half);
                expected_rows.push(row);
            }
        }
        let cos_values = cos.to_vec2::<f32>().expect("cos");
        let sin_values = sin.to_vec2::<f32>().expect("sin");
        for (row, (actual_row, expected)) in cos_values.iter().zip(expected_rows.iter()).enumerate()
        {
            for (index, (&actual, &want)) in actual_row.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - want.cos()).abs() < 1e-6,
                    "cos[{row}][{index}] {actual} != {}",
                    want.cos()
                );
            }
        }
        for (row, (actual_row, expected)) in sin_values.iter().zip(expected_rows.iter()).enumerate()
        {
            for (index, (&actual, &want)) in actual_row.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - want.sin()).abs() < 1e-6,
                    "sin[{row}][{index}] {actual} != {}",
                    want.sin()
                );
            }
        }
        // y frequencies occupy dims 0..4, x frequencies dims 4..8 (per
        // half); row 0 and row 1 differ only in y (x spans 0..2 per row).
    }

    #[test]
    fn rotate_half_matches_reference_convention() {
        let device = &Device::Cpu;
        let x = Tensor::from_vec((1..9).map(|v| v as f32).collect::<Vec<_>>(), (1, 8), device)
            .expect("x");
        let rotated = rotate_half(&x).expect("rotate");
        let values = rotated
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("values");
        // cat(-x2, x1) with x1 = [1, 2, 3, 4], x2 = [5, 6, 7, 8].
        assert_eq!(values, vec![-5.0, -6.0, -7.0, -8.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_applies_to_patches_only_and_prefix_tokens_pass_through() {
        let device = &Device::Cpu;
        let config = tiny_config();
        let head_dim = config.head_dim(); // 8
        let num_prefix = 1 + config.num_register_tokens; // 3
        let patches = 2; // 2 patch tokens

        // One "head" (batch 1, heads 1), seq = 3 prefix + 2 patches.
        let mut q_values = Vec::new();
        for i in 0..(num_prefix + patches) * head_dim {
            q_values.push(0.1 * (i as f32 + 1.0));
        }
        let q = Tensor::from_vec(
            q_values.clone(),
            (1, 1, num_prefix + patches, head_dim),
            device,
        )
        .expect("q");
        let k =
            Tensor::from_vec(q_values, (1, 1, num_prefix + patches, head_dim), device).expect("k");

        // Constant cos/sin tables (c, s) -> rotation is verifiable by hand:
        // out[i] = q[i] * c + partner * s where partner = -q[i+half] for the
        // first half and q[i-half] for the second.
        let (c, s) = (0.5_f32, 0.5_f32.sqrt());
        let cos = Tensor::from_vec(vec![c; patches * head_dim], (patches, head_dim), device)
            .expect("cos");
        let sin = Tensor::from_vec(vec![s; patches * head_dim], (patches, head_dim), device)
            .expect("sin");

        let (q_rot, _k_rot) =
            apply_rotary_to_patches(&q, &k, &cos, &sin, num_prefix).expect("rope apply");
        let q_rot = q_rot
            .reshape((num_prefix + patches, head_dim))
            .and_then(|t| t.to_vec2::<f32>())
            .expect("values");
        // Prefix tokens untouched.
        for (token, expected_base) in q_rot.iter().take(num_prefix).enumerate() {
            for (dim, &value) in expected_base.iter().enumerate() {
                let want = 0.1 * ((token * head_dim + dim) as f32 + 1.0);
                assert!((value - want).abs() < 1e-6);
            }
        }
        // First patch token (token index num_prefix): flat dims
        // [num_prefix*head_dim, (num_prefix+1)*head_dim), value at flat i is
        // 0.1 * (i + 1).
        let base = 0.1 * (num_prefix * head_dim + 1) as f32;
        let q_patch: Vec<f32> = (0..head_dim).map(|i| base + 0.1 * i as f32).collect();
        let rotated = q_rot[num_prefix].to_vec();
        for dim in 0..head_dim {
            let partner = if dim < head_dim / 2 {
                -q_patch[dim + head_dim / 2]
            } else {
                q_patch[dim - head_dim / 2]
            };
            let want = q_patch[dim] * c + partner * s;
            assert!((rotated[dim] - want).abs() < 1e-5, "dim {dim}");
        }
    }

    #[test]
    fn config_parses_local_dinov3_json_shape_only() {
        // No filesystem access required: parse an inline JSON snippet.
        let dir = std::env::temp_dir().join("trellis2-dino-config-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let path = dir.join("config.json");
        std::fs::write(
            &path,
            r#"{"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16,
                "intermediate_size": 4096, "patch_size": 16, "num_register_tokens": 4,
                "layer_norm_eps": 1e-05, "rope_theta": 100.0, "key_bias": false}"#,
        )
        .expect("write");
        let config = Trellis2DinoConfig::from_config_json(&path).expect("config");
        assert_eq!(config, Trellis2DinoConfig::vitl16());
        assert_eq!(config.token_count(512), 1029);
        assert_eq!(config.token_count(1024), 4101);
        // key_bias=true must be rejected.
        std::fs::write(
            &path,
            r#"{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "intermediate_size": 16, "patch_size": 2, "num_register_tokens": 0,
                "layer_norm_eps": 1e-05, "rope_theta": 100.0, "key_bias": true}"#,
        )
        .expect("write");
        assert!(Trellis2DinoConfig::from_config_json(&path).is_err());
    }
}
