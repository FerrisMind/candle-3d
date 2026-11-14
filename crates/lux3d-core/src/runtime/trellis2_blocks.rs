//! TRELLIS.2 transformer machinery shared by the dense sparse-structure flow
//! DiT (T3) and the sparse SLAT flow DiTs (T4).
//!
//! Reference: `trellis2/modules/attention/rope.py`
//! (`RotaryPositionEmbedder`), `trellis2/models/sparse_structure_flow.py`
//! (`TimestepEmbedder`), `trellis2/modules/transformer/modulated.py`
//! (`ModulatedTransformerBlock` / `ModulatedTransformerCrossBlock`) and
//! `trellis2/modules/attention/modules.py` (`MultiHeadAttention` +
//! `MultiHeadRMSNorm`).
//!
//! Exact math (locked decision 6):
//! - 3D RoPE: `head_dim = 128`, `freq_dim = head_dim/2/3 = 21`,
//!   `freqs[i] = 1 / 10000^(i/21)`. For an integer coordinate `(x, y, z)` the
//!   phase angles are `coord[axis] * freqs[i]` (63 values), zero-padded with
//!   angle 0 (unit complex phase) to `head_dim/2 = 64`. The last dim of q/k is
//!   interpreted as 64 interleaved complex pairs and rotated per pair; phases
//!   are shared across heads.
//! - q/k RMS norm: `F.normalize(x.float(), dim=-1) * gamma[head, head_dim] *
//!   sqrt(head_dim)` (torch `F.normalize` eps 1e-12), computed in F32.
//! - adaLN (`share_mod = true`): a global `t_emb = adaLN(t_emb)` with
//!   `adaLN = SiLU -> Linear(hidden, 6*hidden)`; each block computes
//!   `(block.modulation + t_emb).chunk(6)` = shift_msa, scale_msa, gate_msa,
//!   shift_mlp, scale_mlp, gate_mlp.
//! - Block norms are `LayerNorm32(eps=1e-6)`: `norm1`/`norm3` are
//!   non-affine, `norm2` (before cross attention) is affine.
//! - FFN: `Linear(hidden, hidden*mlp_ratio) -> GELU(tanh) -> Linear(..,
//!   hidden)` (the reference uses `nn.GELU(approximate="tanh")`).
//! - Attention: full SDPA, scale `1/sqrt(head_dim)`, output projection
//!   biased.
//!
//! All compute is F32 (locked decision 3); weights are cast to F32 by the
//! VarBuilder at access time.

use candle_core::{D, DType, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Module, layer_norm};

use super::attention_math::exact_query_chunked_sdpa;
use super::nn_blocks::{Linear, linear};
use crate::weights::VarBuilder;

/// Default self-attention query chunk for the 4096-token dense SS DiT
/// (`LUX3D_TRELLIS2_ATTN_CHUNK` override). Bounds the [heads, q, k] scores
/// buffer (12 x 1024 x 4096 x 4 B = 201 MiB per chunk at the default).
fn attn_chunk_size() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("LUX3D_TRELLIS2_ATTN_CHUNK")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1024)
    })
}

/// Per-axis RoPE frequencies `1 / base^(i/freq_dim)` for
/// `i in 0..freq_dim`, computed in F32 exactly like
/// `RotaryPositionEmbedder.__init__`.
fn rope_axis_frequencies(head_dim: usize, rope_base: f32) -> CandleResult<Vec<f32>> {
    if head_dim % 2 != 0 {
        candle_core::bail!("TRELLIS.2 RoPE head dim must be divisible by 2, got {head_dim}");
    }
    // `freq_dim = head_dim // 2 // dim` (integer division). For head_dim 128
    // this is 21: 3 * 21 = 63 rotated pairs, and the remaining pair
    // (64 - 63 = 1) is padded with unit phases (angle 0, identity) exactly
    // like `RotaryPositionEmbedder.forward`.
    let freq_dim = head_dim / 2 / 3;
    if freq_dim == 0 {
        candle_core::bail!("TRELLIS.2 head_dim {head_dim} is too small for 3-axis RoPE");
    }
    Ok((0..freq_dim)
        .map(|i| {
            let exponent = i as f32 / freq_dim as f32;
            1.0 / rope_base.powf(exponent)
        })
        .collect())
}

/// 3D rotary position embedding (`RotaryPositionEmbedder`, head_dim 128,
/// rope_freq `(1.0, 10000.0)`).
///
/// Phases are represented as `(cos, sin)` tables of shape `[L, head_dim/2]`
/// (one complex entry per rotated pair, axis-major: `[axis * freq_dim + i]`),
/// matching `phases = polar(1, outer(indices, freqs)).reshape(L, -1)` padded
/// with unit phases to `head_dim/2`.
#[derive(Debug, Clone)]
pub struct Trellis2Rope3d {
    head_dim: usize,
    rope_base: f32,
}

impl Trellis2Rope3d {
    pub fn new(head_dim: usize) -> Self {
        Self {
            head_dim,
            rope_base: 10000.0,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Angle table for integer coordinates `coords: [L, 3]` (axis order
    /// `(x, y, z)`; for dense SS grids the token order is row-major
    /// `x*res^2 + y*res + z`). Returns `(cos, sin)`, each `[L, head_dim/2]`
    /// F32.
    pub fn cos_sin(
        &self,
        coords: &[i64],
        device: &candle_core::Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        if coords.len() % 3 != 0 {
            candle_core::bail!(
                "TRELLIS.2 RoPE coords must be [L, 3] flattened, got {} values",
                coords.len()
            );
        }
        let tokens = coords.len() / 3;
        let freqs = rope_axis_frequencies(self.head_dim, self.rope_base)?;
        let freq_dim = freqs.len();
        let half = self.head_dim / 2;
        let mut angles = vec![0.0_f32; tokens * half];
        for (token, coord) in coords.chunks_exact(3).enumerate() {
            let row = &mut angles[token * half..(token + 1) * half];
            // Axis-major layout: [x freqs | y freqs | z freqs | zero pad].
            for (axis, &value) in coord.iter().take(3).enumerate() {
                let value = value as f32;
                for (i, &frequency) in freqs.iter().enumerate() {
                    row[axis * freq_dim + i] = value * frequency;
                }
            }
        }
        let angles = Tensor::from_vec(angles, (tokens, half), device)?;
        Ok((angles.cos()?, angles.sin()?))
    }

    /// Rotate `x: [B, L, H, head_dim]` by the phase tables
    /// `(cos, sin): [L, head_dim/2]` (complex multiply per interleaved pair;
    /// the trailing pad pair has angle 0 and leaves its channels untouched).
    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (batch, tokens, heads, head_dim) = x.dims4()?;
        let half = head_dim / 2;
        if cos.dims() != &[tokens, half] || sin.dims() != &[tokens, half] {
            candle_core::bail!(
                "RoPE tables must be [{tokens}, {half}], got cos {:?} sin {:?}",
                cos.dims(),
                sin.dims()
            );
        }
        // Split the last dim into interleaved (re, im) pairs.
        let pairs = x.reshape((batch, tokens, heads, half, 2))?;
        let re = pairs.narrow(4, 0, 1)?.squeeze(4)?;
        let im = pairs.narrow(4, 1, 1)?.squeeze(4)?;
        let cos = cos.reshape((1, tokens, 1, half))?;
        let sin = sin.reshape((1, tokens, 1, half))?;
        let out_re = re.broadcast_mul(&cos)?.sub(&im.broadcast_mul(&sin)?)?;
        let out_im = re.broadcast_mul(&sin)?.add(&im.broadcast_mul(&cos)?)?;
        // Interleave back: complex view_as_real -> flat last dim.
        let rotated = Tensor::stack(&[&out_re, &out_im], 4)?;
        rotated.reshape((batch, tokens, heads, head_dim))
    }
}

/// Dense-grid token coordinates for a cubic resolution:
/// row-major `index = x * res^2 + y * res + z` with coords `(x, y, z)`
/// (`torch.meshgrid(arange(res), arange(res), arange(res), indexing="ij")`).
pub fn cubic_token_coords(resolution: usize) -> Vec<i64> {
    let mut coords = Vec::with_capacity(resolution * resolution * resolution * 3);
    for x in 0..resolution {
        for y in 0..resolution {
            for z in 0..resolution {
                coords.push(x as i64);
                coords.push(y as i64);
                coords.push(z as i64);
            }
        }
    }
    coords
}

/// Sinusoidal timestep embedding (`TimestepEmbedder.timestep_embedding`,
/// `dim = 256`, `max_period = 10000`): `freqs = exp(-ln(10000) *
/// arange(half)/half)`, `embedding = [cos(t*freqs), sin(t*freqs)]`.
/// Returns `[dim]` F32 on `device`.
pub fn timestep_embedding(
    t: f32,
    dim: usize,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    let half = dim / 2;
    let neg_log_period = -(10000.0_f64.ln()) as f32;
    let freqs: Vec<f32> = (0..half)
        .map(|i| neg_log_period * (i as f32) / half as f32)
        .map(f32::exp)
        .collect();
    let freqs = Tensor::from_vec(freqs, (half,), device)?;
    let args = freqs.affine(t as f64, 0.0)?;
    let embedding = Tensor::cat(&[&args.cos()?, &args.sin()?], D::Minus1)?;
    if dim % 2 != 0 {
        // The reference pads odd dims with zeros; unused by TRELLIS.2.
        let zero = Tensor::zeros(1, DType::F32, device)?;
        return Tensor::cat(&[&embedding, &zero], D::Minus1);
    }
    Ok(embedding)
}

/// `TimestepEmbedder`: `[cos, sin]` table (256) -> MLP
/// `Linear(256, hidden) -> SiLU -> Linear(hidden, hidden)`.
#[derive(Debug)]
pub struct Trellis2TimestepEmbedder {
    fc1: Linear,
    fc2: Linear,
    frequency_embedding_size: usize,
}

impl Trellis2TimestepEmbedder {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> CandleResult<Self> {
        let frequency_embedding_size = 256;
        let mlp = vb.pp("mlp");
        Ok(Self {
            fc1: linear(mlp.pp("0"), frequency_embedding_size, hidden_size, true)?,
            fc2: linear(mlp.pp("2"), hidden_size, hidden_size, true)?,
            frequency_embedding_size,
        })
    }

    /// Scalar flow time already scaled by the sampler (`1000 * t`).
    /// Returns `[1, hidden_size]` on the embedder's device.
    pub fn forward(&self, t_model: f32) -> CandleResult<Tensor> {
        let device = self
            .fc1
            .bias()
            .map(|bias| bias.device().clone())
            .unwrap_or(candle_core::Device::Cpu);
        let table = timestep_embedding(t_model, self.frequency_embedding_size, &device)?;
        let table = table.unsqueeze(0)?;
        self.fc2.forward(&self.fc1.forward(&table)?.silu()?)
    }
}

/// Shared adaLN modulation `SiLU -> Linear(hidden, 6*hidden)`
/// (`SparseStructureFlowModel.adaLN_modulation`, `share_mod = true`).
#[derive(Debug)]
pub struct Trellis2AdaLn {
    linear: Linear,
}

impl Trellis2AdaLn {
    pub fn load(vb: VarBuilder, hidden_size: usize) -> CandleResult<Self> {
        // nn.Sequential(SiLU(), Linear(..)) -> the Linear is submodule "1".
        Ok(Self {
            linear: linear(vb.pp("1"), hidden_size, 6 * hidden_size, true)?,
        })
    }

    /// `t_emb [B, hidden]` -> `[B, 6*hidden]`.
    pub fn forward(&self, t_emb: &Tensor) -> CandleResult<Tensor> {
        self.linear.forward(&t_emb.silu()?)
    }
}

/// Chunk a `[B, 6C]` modulation vector into the six adaLN pieces
/// `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`, each
/// broadcastable `[B, 1, C]`.
pub fn chunk_modulation(mod_vec: &Tensor) -> CandleResult<[Tensor; 6]> {
    let parts = mod_vec.chunk(6, D::Minus1)?;
    let [a, b, c, d, e, f]: [Tensor; 6] = parts
        .try_into()
        .map_err(|_| candle_core::Error::Msg("adaLN modulation must chunk into 6".into()))?;
    Ok([
        a.unsqueeze(1)?,
        b.unsqueeze(1)?,
        c.unsqueeze(1)?,
        d.unsqueeze(1)?,
        e.unsqueeze(1)?,
        f.unsqueeze(1)?,
    ])
}

/// Non-affine `F.layer_norm(x, (C,), eps)` over the last dim.
pub(crate) fn layer_norm_no_affine(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
    let denom = var.affine(1.0, eps)?.sqrt()?;
    centered.broadcast_div(&denom)
}

/// `MultiHeadRMSNorm`: `F.normalize(x.float(), dim=-1) * gamma *
/// sqrt(head_dim)` per head, computed in F32.
#[derive(Debug)]
pub struct Trellis2QkRmsNorm {
    gamma: Tensor,
    scale: f32,
}

impl Trellis2QkRmsNorm {
    pub fn load(vb: VarBuilder, heads: usize, head_dim: usize) -> CandleResult<Self> {
        Ok(Self {
            gamma: load_f32(vb, (heads, head_dim), "gamma")?,
            scale: (head_dim as f32).sqrt(),
        })
    }

    /// `x: [B, L, H, head_dim]` (already F32) -> normalized `[B, L, H,
    /// head_dim]`.
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let norm = xs
            .sqr()?
            .sum_keepdim(D::Minus1)?
            .sqrt()?
            .clamp(1e-12_f32, 1.0e30_f32)?;
        let normalized = xs.broadcast_div(&norm)?;
        let gamma = self
            .gamma
            .reshape((1, 1, self.gamma.dim(0)?, self.gamma.dim(1)?))?;
        normalized
            .broadcast_mul(&gamma)?
            .affine(self.scale as f64, 0.0)
    }
}

fn load_f32(
    vb: VarBuilder,
    shape: impl Into<candle_core::Shape>,
    name: &str,
) -> CandleResult<Tensor> {
    let tensor = vb.get(shape, name)?;
    if tensor.dtype() == DType::F32 {
        Ok(tensor)
    } else {
        tensor.to_dtype(DType::F32)
    }
}

/// Dense multi-head self attention with optional q/k RMS norm and RoPE
/// (`MultiHeadAttention(type="self", attn_mode="full")`).
///
/// `to_qkv: Linear(C, 3C)` -> `[B, L, 3, H, head_dim]` unbind; RMS norm and
/// RoPE are applied to q/k (RMS first, matching the reference order); SDPA
/// with scale `1/sqrt(head_dim)`; `to_out: Linear(C, C)` biased.
#[derive(Debug)]
pub struct Trellis2SelfAttention {
    to_qkv: Linear,
    to_out: Linear,
    q_rms_norm: Option<Trellis2QkRmsNorm>,
    k_rms_norm: Option<Trellis2QkRmsNorm>,
    rope: Trellis2Rope3d,
    heads: usize,
    head_dim: usize,
}

impl Trellis2SelfAttention {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        heads: usize,
        qk_rms_norm: bool,
        use_rope: bool,
    ) -> CandleResult<Self> {
        let head_dim = channels / heads;
        if head_dim * heads != channels {
            candle_core::bail!("channels {channels} not divisible by heads {heads}");
        }
        let rope = Trellis2Rope3d::new(head_dim);
        if !use_rope {
            candle_core::bail!("TRELLIS.2 flow models always use RoPE (pe_mode=\"rope\")");
        }
        Ok(Self {
            to_qkv: linear(vb.pp("to_qkv"), channels, channels * 3, true)?,
            to_out: linear(vb.pp("to_out"), channels, channels, true)?,
            q_rms_norm: if qk_rms_norm {
                Some(Trellis2QkRmsNorm::load(
                    vb.pp("q_rms_norm"),
                    heads,
                    head_dim,
                )?)
            } else {
                None
            },
            k_rms_norm: if qk_rms_norm {
                Some(Trellis2QkRmsNorm::load(
                    vb.pp("k_rms_norm"),
                    heads,
                    head_dim,
                )?)
            } else {
                None
            },
            rope,
            heads,
            head_dim,
        })
    }

    /// Split a fused q/k/v projection `[B, L, 3*C]` into three
    /// `[B, L, H, head_dim]` tensors (reshape `[B, L, 3, H, hd]`, unbind 2).
    pub(crate) fn split_heads(
        qkv: &Tensor,
        batch: usize,
        tokens: usize,
        heads: usize,
        head_dim: usize,
    ) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let qkv = qkv.reshape((batch, tokens, 3, heads, head_dim))?;
        let q = qkv.narrow(2, 0, 1)?.squeeze(2)?;
        let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
        let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
        Ok((q, k, v))
    }

    /// Normalize + rotate q and k, then run full SDPA over
    /// `[B, H, L, head_dim]` views.
    pub(crate) fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let mut q = q.clone();
        let mut k = k.clone();
        if let Some(norm) = &self.q_rms_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_rms_norm {
            k = norm.forward(&k)?;
        }
        q = self.rope.apply(&q, cos, sin)?;
        k = self.rope.apply(&k, cos, sin)?;
        let (batch, tokens, heads, head_dim) = q.dims4()?;
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let out = exact_query_chunked_sdpa(&q, &k, &v, scale, attn_chunk_size())?;
        out.reshape((batch, heads, tokens, head_dim))?
            .permute((0, 2, 1, 3))
    }

    /// `x: [B, L, C]` with RoPE tables `cos/sin: [L, head_dim/2]` ->
    /// `[B, L, C]`.
    pub fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        let qkv = self.to_qkv.forward(xs)?;
        let (q, k, v) = Self::split_heads(&qkv, batch, tokens, self.heads, self.head_dim)?;
        let out = self.attention(&q, &k, &v, cos, sin)?;
        self.to_out.forward(&out.flatten_from(D::Minus2)?)
    }
}

/// Dense multi-head cross attention
/// (`MultiHeadAttention(type="cross")`): `to_q: Linear(C, C)` on the tokens,
/// `to_kv: Linear(cond_channels, 2C)` on the conditioning context, q/k RMS
/// norm, no RoPE, full SDPA, biased `to_out`.
#[derive(Debug)]
pub struct Trellis2CrossAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    q_rms_norm: Option<Trellis2QkRmsNorm>,
    k_rms_norm: Option<Trellis2QkRmsNorm>,
    heads: usize,
    head_dim: usize,
}

impl Trellis2CrossAttention {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        cond_channels: usize,
        heads: usize,
        qk_rms_norm: bool,
    ) -> CandleResult<Self> {
        let head_dim = channels / heads;
        Ok(Self {
            to_q: linear(vb.pp("to_q"), channels, channels, true)?,
            to_kv: linear(vb.pp("to_kv"), cond_channels, channels * 2, true)?,
            to_out: linear(vb.pp("to_out"), channels, channels, true)?,
            q_rms_norm: if qk_rms_norm {
                Some(Trellis2QkRmsNorm::load(
                    vb.pp("q_rms_norm"),
                    heads,
                    head_dim,
                )?)
            } else {
                None
            },
            k_rms_norm: if qk_rms_norm {
                Some(Trellis2QkRmsNorm::load(
                    vb.pp("k_rms_norm"),
                    heads,
                    head_dim,
                )?)
            } else {
                None
            },
            heads,
            head_dim,
        })
    }

    /// `x: [B, L, C]`, `context: [B, Lkv, cond_channels]` -> `[B, L, C]`.
    pub fn forward(&self, xs: &Tensor, context: &Tensor) -> CandleResult<Tensor> {
        let (batch, tokens, _) = xs.dims3()?;
        let kv_tokens = context.dim(1)?;
        let q = self
            .to_q
            .forward(xs)?
            .reshape((batch, tokens, self.heads, self.head_dim))?;
        let kv = self.to_kv.forward(context)?.reshape((
            batch,
            kv_tokens,
            2,
            self.heads,
            self.head_dim,
        ))?;
        let k = kv.narrow(2, 0, 1)?.squeeze(2)?;
        let v = kv.narrow(2, 1, 1)?.squeeze(2)?;
        let q = match &self.q_rms_norm {
            Some(norm) => norm.forward(&q)?,
            None => q,
        };
        let k = match &self.k_rms_norm {
            Some(norm) => norm.forward(&k)?,
            None => k,
        };
        let q = q.permute((0, 2, 1, 3))?;
        let k = k.permute((0, 2, 1, 3))?;
        let v = v.permute((0, 2, 1, 3))?;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let out = exact_query_chunked_sdpa(&q, &k, &v, scale, attn_chunk_size())?;
        let out = out
            .reshape((batch, self.heads, tokens, self.head_dim))?
            .permute((0, 2, 1, 3))?
            .reshape((batch, tokens, self.heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// TRELLIS.2 FFN: `Linear(C, hidden) -> GELU(tanh) -> Linear(hidden, C)`
/// (`FeedForwardNet`, `nn.GELU(approximate="tanh")`).
#[derive(Debug)]
pub struct Trellis2FeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl Trellis2FeedForward {
    pub fn load(vb: VarBuilder, channels: usize, hidden: usize) -> CandleResult<Self> {
        // nn.Sequential(Linear, GELU, Linear) -> submodules "0" and "2".
        Ok(Self {
            fc1: linear(vb.pp("mlp").pp("0"), channels, hidden, true)?,
            fc2: linear(vb.pp("mlp").pp("2"), hidden, channels, true)?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        // nn.GELU(approximate="tanh") == candle Tensor::gelu (tanh based).
        self.fc2.forward(&self.fc1.forward(xs)?.gelu()?)
    }
}

/// Dense modulated transformer cross block
/// (`ModulatedTransformerCrossBlock`, `share_mod = true`): adaLN self
/// attention, plain (affine-norm) cross attention, adaLN FFN.
#[derive(Debug)]
pub struct Trellis2DenseBlock {
    norm2: LayerNorm,
    self_attn: Trellis2SelfAttention,
    cross_attn: Trellis2CrossAttention,
    mlp: Trellis2FeedForward,
    modulation: Tensor,
}

impl Trellis2DenseBlock {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        cond_channels: usize,
        heads: usize,
        mlp_hidden: usize,
        qk_rms_norm: bool,
        qk_rms_norm_cross: bool,
    ) -> CandleResult<Self> {
        Ok(Self {
            norm2: layer_norm(channels, 1e-6, vb.pp("norm2"))?,
            self_attn: Trellis2SelfAttention::load(
                vb.pp("self_attn"),
                channels,
                heads,
                qk_rms_norm,
                true,
            )?,
            cross_attn: Trellis2CrossAttention::load(
                vb.pp("cross_attn"),
                channels,
                cond_channels,
                heads,
                qk_rms_norm_cross,
            )?,
            mlp: Trellis2FeedForward::load(vb.pp("mlp"), channels, mlp_hidden)?,
            modulation: load_f32(vb, 6 * channels, "modulation")?,
        })
    }

    /// `x: [B, L, C]`, `t_emb: [B, 6C]` (already adaLN-projected),
    /// `context: [B, Lkv, cond_channels]`, `cos/sin: [L, head_dim/2]`.
    pub fn forward(
        &self,
        xs: &Tensor,
        t_emb: &Tensor,
        context: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let mod_vec = self.modulation.unsqueeze(0)?.broadcast_add(t_emb)?;
        let [
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ] = chunk_modulation(&mod_vec)?;

        let h = layer_norm_no_affine(xs, 1e-6)?;
        let h = h
            .broadcast_mul(&scale_msa.affine(1.0, 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let h = self.self_attn.forward(&h, cos, sin)?;
        let h = h.broadcast_mul(&gate_msa)?;
        let xs = xs.add(&h)?;

        let h = self.norm2.forward(&xs)?;
        let h = self.cross_attn.forward(&h, context)?;
        let xs = xs.add(&h)?;

        let h = layer_norm_no_affine(&xs, 1e-6)?;
        let h = h
            .broadcast_mul(&scale_mlp.affine(1.0, 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        let h = self.mlp.forward(&h)?;
        let h = h.broadcast_mul(&gate_mlp)?;
        xs.add(&h)
    }
}

/// Sparse modulated transformer block (`ModulatedSparseTransformerBlock`,
/// self attention + FFN only), kept generic over a `[N, C]` token feature
/// tensor so the sparse SLAT DiT (T4) can reuse it: sparse full attention in
/// the reference is block-diagonal over batch elements and TRELLIS.2 samples
/// one batch element at a time, so the attention math is identical to the
/// dense path with `B = 1`.
#[derive(Debug)]
pub struct Trellis2SparseBlock {
    self_attn: Trellis2SelfAttention,
    mlp: Trellis2FeedForward,
    modulation: Tensor,
}

impl Trellis2SparseBlock {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        heads: usize,
        mlp_hidden: usize,
        qk_rms_norm: bool,
    ) -> CandleResult<Self> {
        Ok(Self {
            self_attn: Trellis2SelfAttention::load(
                vb.pp("self_attn"),
                channels,
                heads,
                qk_rms_norm,
                true,
            )?,
            mlp: Trellis2FeedForward::load(vb.pp("mlp"), channels, mlp_hidden)?,
            modulation: load_f32(vb, 6 * channels, "modulation")?,
        })
    }

    /// `feats: [N, C]`, `t_emb: [1, 6C]` (adaLN-projected),
    /// `cos/sin: [N, head_dim/2]` from `coords[:, 1:]` (batch dim dropped).
    pub fn forward(
        &self,
        feats: &Tensor,
        t_emb: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let xs = feats.unsqueeze(0)?; // [1, N, C]
        let mod_vec = self.modulation.unsqueeze(0)?.broadcast_add(t_emb)?;
        let [
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ] = chunk_modulation(&mod_vec)?;

        let h = layer_norm_no_affine(&xs, 1e-6)?;
        let h = h
            .broadcast_mul(&scale_msa.affine(1.0, 1.0)?)?
            .broadcast_add(&shift_msa)?;
        let h = self.self_attn.forward(&h, cos, sin)?;
        let h = h.broadcast_mul(&gate_msa)?;
        let xs = xs.add(&h)?;

        let h = layer_norm_no_affine(&xs, 1e-6)?;
        let h = h
            .broadcast_mul(&scale_mlp.affine(1.0, 1.0)?)?
            .broadcast_add(&shift_mlp)?;
        let h = self.mlp.forward(&h)?;
        let h = h.broadcast_mul(&gate_mlp)?;
        xs.add(&h)?.squeeze(0)
    }
}
