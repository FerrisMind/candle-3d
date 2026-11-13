//! TripoSG DiT transformer (HunyuanDiT-style UViT), the rectified-flow Euler
//! scheduler, and the classifier-free-guidance denoise loop.
//!
//! Reference semantics (`triposg/models/transformers/triposg_transformer.py`,
//! `triposg/models/attention_processor.py`,
//! `triposg/schedulers/scheduling_rectified_flow.py`,
//! `triposg/pipelines/pipeline_triposg.py`):
//!
//! - Sequence = timestep token PREPENDED to `proj_in(latents)` (2049 tokens
//!   for 2048 latent tokens); no adaLN, no RoPE, no LayerScale.
//! - Timestep embedding: sinusoid `half = width/2`,
//!   `exp(-ln(10000) * i / half)`, `cat([sin, cos])` (sin FIRST), then MLP
//!   `width -> 4*width -> width` with exact GELU.
//! - 21 blocks: long skip on blocks 11..20 only (`cat([skip, x])` skip FIRST,
//!   then `skip_linear` then `skip_norm` — the reference's `skip_norm_last`
//!   quirk, replicated verbatim), self-attn, cross-attn, exact-GELU FFN, all
//!   pre-normed with LayerNorm(eps 1e-5, affine).
//! - Attention re-splits the projections EXACTLY like
//!   `TripoSGAttnProcessor2_0`: self-attn concatenates `[q, k, v]` on the
//!   channel dim then re-splits per head (head h takes columns
//!   `[384h : 384h+128)` of each of q/k/v); cross-attn uses a standard
//!   head-major query and re-splits `cat([k, v])` with stride 256. Q and K
//!   get RMSNorm (eps 1e-6, per-head weight `[head_dim]`); SDPA scale is the
//!   default `1/sqrt(head_dim)`.
//! - Final `norm_out` over all 2049 tokens, slice the LAST 2048, `proj_out`
//!   `width -> in_channels` -> rectified-flow velocity `[B, L, 64]`.
//!
//! Everything stays F32 end to end (the canonical weights are F32 and the
//! parity baseline was generated in F32). All ops are backend-generic Candle
//! primitives; self-attention scores are computed in query chunks
//! (`LUX3D_TRIPOSG_ATTN_CHUNK`, default 1024) to bound peak memory on 12 GiB
//! GPUs.

use std::sync::OnceLock;

use super::nn_blocks::{Linear, linear_dense as linear, linear_no_bias_dense as linear_no_bias};
use crate::weights::VarBuilder;
use candle_core::{D, DType, Device, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Module, layer_norm};

use super::attention_math::{exact_query_chunked_sdpa, exact_sdpa_heads};

/// Self-attention query chunk (`LUX3D_TRIPOSG_ATTN_CHUNK`, default 1024).
fn attn_chunk_size() -> usize {
    static CHUNK: OnceLock<usize> = OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("LUX3D_TRIPOSG_ATTN_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1024)
    })
}

/// Parsed `transformer` section of the canonical `resolved_config.json`.
///
/// Defaults replicate `TripoSGDiTModel.__init__` (`num_attention_heads=16`,
/// `width=2048`, `in_channels=64`, `num_layers=21`,
/// `cross_attention_dim=1024`) plus the hardcoded `mlp_ratio=4.0`.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgTransformerConfig {
    pub in_channels: usize,
    pub width: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
    pub cross_attention_dim: usize,
    pub mlp_ratio: f64,
}

impl Default for TripoSgTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            width: 2048,
            num_attention_heads: 16,
            num_layers: 21,
            cross_attention_dim: 1024,
            mlp_ratio: 4.0,
        }
    }
}

impl TripoSgTransformerConfig {
    /// Parse from the canonical `resolved_config.json` document (or directly
    /// from its `transformer` subsection). Unknown/missing fields fall back to
    /// the reference defaults.
    pub fn from_resolved_config(value: &serde_json::Value) -> CandleResult<Self> {
        let source = value.get("transformer").unwrap_or(value);
        let get_usize = |key: &str, default: usize| -> usize {
            source
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let cfg = Self {
            in_channels: get_usize("in_channels", 64),
            width: get_usize("width", 2048),
            num_attention_heads: get_usize("num_attention_heads", 16),
            num_layers: get_usize("num_layers", 21),
            cross_attention_dim: get_usize("cross_attention_dim", 1024),
            mlp_ratio: source
                .get("mlp_ratio")
                .and_then(|v| v.as_f64())
                .unwrap_or(4.0),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> CandleResult<()> {
        if self.in_channels == 0 || self.width == 0 || self.cross_attention_dim == 0 {
            candle_core::bail!(
                "TripoSG DiT config has a zero dimension: in_channels {}, width {}, \
                 cross_attention_dim {}",
                self.in_channels,
                self.width,
                self.cross_attention_dim
            );
        }
        if self.num_attention_heads == 0 || self.width % self.num_attention_heads != 0 {
            candle_core::bail!(
                "TripoSG DiT width {} is not divisible by num_attention_heads {}",
                self.width,
                self.num_attention_heads
            );
        }
        if self.width % 2 != 0 {
            candle_core::bail!("TripoSG DiT width {} must be even", self.width);
        }
        if self.mlp_ratio <= 0.0 || (self.width as f64 * self.mlp_ratio).fract() != 0.0 {
            candle_core::bail!(
                "TripoSG DiT mlp_ratio {} must be positive and yield an integer FFN width",
                self.mlp_ratio
            );
        }
        // Skip topology: blocks 0..(num_layers/2 - 1) push, blocks
        // (num_layers/2 + 1)..pop; both halves must be non-empty.
        if self.num_layers < 3 {
            candle_core::bail!(
                "TripoSG DiT needs at least 3 layers for the UViT skip topology, got {}",
                self.num_layers
            );
        }
        Ok(())
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.width / self.num_attention_heads
    }

    pub(crate) fn ff_inner_dim(&self) -> usize {
        (self.width as f64 * self.mlp_ratio) as usize
    }

    /// `time_embed_dim = 4 * width` (diffusers `_set_time_proj("positional")`
    /// with `time_embedding_dim=None`).
    pub(crate) fn time_embed_dim(&self) -> usize {
        self.width * 4
    }
}

/// TripoSG 21-block DiT. Public for the parity test harness; runtime-internal
/// surface (the pipeline drives it through `denoise`).
#[derive(Debug)]
pub struct TripoSgDiT {
    cfg: TripoSgTransformerConfig,
    proj_in: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    blocks: Vec<DiTBlock>,
    norm_out: LayerNorm,
    proj_out: Linear,
    attn_chunk: usize,
}

impl TripoSgDiT {
    /// Load from a VarBuilder rooted at the canonical `transformer.` prefix
    /// (i.e. pass `vb.pp("transformer")` from the canonical weight set).
    pub fn load(vb: VarBuilder, cfg: &TripoSgTransformerConfig) -> CandleResult<Self> {
        let proj_in = linear(cfg.in_channels, cfg.width, vb.pp("proj_in"))?;
        let time_linear_1 = linear(
            cfg.width,
            cfg.time_embed_dim(),
            vb.pp("time_proj").pp("linear_1"),
        )?;
        let time_linear_2 = linear(
            cfg.time_embed_dim(),
            cfg.width,
            vb.pp("time_proj").pp("linear_2"),
        )?;
        let blocks = (0..cfg.num_layers)
            .map(|index| {
                let block = DiTBlock::load(vb.pp("blocks").pp(index.to_string()), cfg, index)?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let norm_out = layer_norm(cfg.width, 1e-5, vb.pp("norm_out"))?;
        let proj_out = linear(cfg.width, cfg.in_channels, vb.pp("proj_out"))?;

        Ok(Self {
            cfg: cfg.clone(),
            proj_in,
            time_linear_1,
            time_linear_2,
            blocks,
            norm_out,
            proj_out,
            attn_chunk: attn_chunk_size(),
        })
    }

    /// `latents [B, L, in_channels] F32`, `timestep` = the scheduler timestep
    /// (e.g. 1000.0), `cond [B, S, cross_attention_dim]` (for CFG the batch
    /// already carries `[uncond, cond]`) -> velocity `[B, L, in_channels]`.
    pub fn forward(&self, latents: &Tensor, timestep: f32, cond: &Tensor) -> CandleResult<Tensor> {
        let (batch, num_tokens, channels) = latents.dims3()?;
        if channels != self.cfg.in_channels {
            candle_core::bail!(
                "TripoSG DiT expects {} latent channels, got {channels}",
                self.cfg.in_channels
            );
        }
        let (cond_batch, _cond_seq, cond_dim) = cond.dims3()?;
        if cond_batch != batch {
            candle_core::bail!(
                "TripoSG DiT condition batch {} does not match latent batch {batch}",
                cond_batch
            );
        }
        if cond_dim != self.cfg.cross_attention_dim {
            candle_core::bail!(
                "TripoSG DiT expects cross-attention dim {}, got {cond_dim}",
                self.cfg.cross_attention_dim
            );
        }

        // Timestep token: [B, 1, width].
        let temb = self.timestep_embedding(timestep, latents.device(), latents.dtype())?;
        let temb = temb
            .reshape((1, 1, self.cfg.width))?
            .broadcast_as((batch, 1, self.cfg.width))?
            .contiguous()?;

        // Sequence = [temb, proj_in(latents)] -> [B, L + 1, width].
        let projected = self.proj_in.forward(latents)?;
        let mut hidden = Tensor::cat(&[&temb, &projected], 1)?;

        // 21 blocks with the UViT long-skip LIFO stack: blocks 0..(mid-1)
        // push their output, block `mid` neither pushes nor pops, blocks
        // (mid+1).. pop (block mid+1 consumes block mid-1's output).
        let mid = self.cfg.num_layers / 2;
        let mut skips: Vec<Tensor> = Vec::new();
        for (index, block) in self.blocks.iter().enumerate() {
            let skip = if index <= mid {
                None
            } else {
                Some(
                    skips
                        .pop()
                        .expect("TripoSG DiT skip stack underflow is a config bug"),
                )
            };
            hidden = block.forward(&hidden, cond, skip.as_ref(), self.attn_chunk)?;
            if index < mid {
                skips.push(hidden.clone());
            }
        }

        // Final: LayerNorm over ALL tokens, slice the LAST num_tokens, proj_out.
        let normalized = self.norm_out.forward(&hidden)?;
        let seq = normalized.dim(1)?;
        let tail = normalized.narrow(1, seq - num_tokens, num_tokens)?;
        self.proj_out.forward(&tail)
    }

    /// diffusers `Timesteps(width, flip_sin_to_cos=False, shift=0)` ->
    /// `TimestepEmbedding` MLP: `[width]` vector, sin FIRST concat.
    fn timestep_embedding(
        &self,
        timestep: f32,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Tensor> {
        let half = self.cfg.width / 2;
        let mut emb = Vec::with_capacity(half);
        let log_max_period = -(10000.0f32).ln();
        for i in 0..half {
            let freq = (log_max_period * i as f32 / half as f32).exp();
            emb.push(timestep * freq);
        }
        let emb = Tensor::from_vec(emb, (1, half), device)?;
        let emb = emb.to_dtype(dtype)?;
        let sin = emb.sin()?;
        let cos = emb.cos()?;
        let projected = Tensor::cat(&[&sin, &cos], D::Minus1)?; // [1, width]
        let projected = self.time_linear_1.forward(&projected)?.gelu_erf()?;
        self.time_linear_2.forward(&projected)
    }
}

/// One HunyuanDiT-style block: (optional long skip) -> self-attn ->
/// cross-attn -> FFN, all pre-normed, all residual.
#[derive(Debug)]
struct DiTBlock {
    skip_norm: Option<LayerNorm>,
    skip_linear: Option<Linear>,
    norm1: LayerNorm,
    attn1: DiTAttention,
    norm2: LayerNorm,
    attn2: DiTAttention,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl DiTBlock {
    fn load(vb: VarBuilder, cfg: &TripoSgTransformerConfig, index: usize) -> CandleResult<Self> {
        let has_skip = index > cfg.num_layers / 2;
        let (skip_norm, skip_linear) = if has_skip {
            (
                Some(layer_norm(cfg.width, 1e-5, vb.pp("skip_norm"))?),
                Some(linear(2 * cfg.width, cfg.width, vb.pp("skip_linear"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            skip_norm,
            skip_linear,
            norm1: layer_norm(cfg.width, 1e-5, vb.pp("norm1"))?,
            attn1: DiTAttention::load(vb.pp("attn1"), cfg, None)?,
            norm2: layer_norm(cfg.width, 1e-5, vb.pp("norm2"))?,
            attn2: DiTAttention::load(vb.pp("attn2"), cfg, Some(cfg.cross_attention_dim))?,
            norm3: layer_norm(cfg.width, 1e-5, vb.pp("norm3"))?,
            ff_in: linear(
                cfg.width,
                cfg.ff_inner_dim(),
                vb.pp("ff").pp("net").pp("0").pp("proj"),
            )?,
            ff_out: linear(cfg.ff_inner_dim(), cfg.width, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cond: &Tensor,
        skip: Option<&Tensor>,
        attn_chunk: usize,
    ) -> CandleResult<Tensor> {
        let mut hidden = match (self.skip_linear.as_ref(), self.skip_norm.as_ref(), skip) {
            (Some(skip_linear), Some(skip_norm), Some(skip)) => {
                // skip FIRST (skip_concat_front=True), then linear THEN norm
                // (skip_norm_last=True — the reference's documented quirk).
                let cat = Tensor::cat(&[skip, xs], D::Minus1)?;
                skip_norm.forward(&skip_linear.forward(&cat)?)?
            }
            _ => xs.clone(),
        };

        // Self-attention (no rotary embedding in the image pipeline).
        let attn_out = self
            .attn1
            .forward(&self.norm1.forward(&hidden)?, None, attn_chunk)?;
        hidden = (hidden + attn_out)?;

        // Cross-attention on the image embeds, reference scale 1.0.
        let cross_out =
            self.attn2
                .forward(&self.norm2.forward(&hidden)?, Some(cond), attn_chunk)?;
        hidden = (hidden + cross_out)?;

        // FFN: exact GELU between the two projections.
        let ff = self.ff_out.forward(
            &self
                .ff_in
                .forward(&self.norm3.forward(&hidden)?)?
                .gelu_erf()?,
        )?;
        hidden + ff
    }
}

/// diffusers `Attention` with `qk_norm="rms_norm"`, `qkv_bias=False`,
/// `out_bias=True`, loaded with the checkpoint's unusual projection layout.
#[derive(Debug)]
struct DiTAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_q_weight: Tensor,
    norm_k_weight: Tensor,
    heads: usize,
    head_dim: usize,
}

impl DiTAttention {
    /// `kv_input_dim` is `None` for self-attention (k/v see the hidden width)
    /// and `cross_attention_dim` for cross-attention.
    fn load(
        vb: VarBuilder,
        cfg: &TripoSgTransformerConfig,
        kv_input_dim: Option<usize>,
    ) -> CandleResult<Self> {
        let kv_in = kv_input_dim.unwrap_or(cfg.width);
        let head_dim = cfg.head_dim();
        Ok(Self {
            to_q: linear_no_bias(cfg.width, cfg.width, vb.pp("to_q"))?,
            to_k: linear_no_bias(kv_in, cfg.width, vb.pp("to_k"))?,
            to_v: linear_no_bias(kv_in, cfg.width, vb.pp("to_v"))?,
            to_out: linear(cfg.width, cfg.width, vb.pp("to_out").pp("0"))?,
            norm_q_weight: vb.get((head_dim,), "norm_q.weight")?,
            norm_k_weight: vb.get((head_dim,), "norm_k.weight")?,
            heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        encoder: Option<&Tensor>,
        attn_chunk: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        match encoder {
            None => {
                // Self-attention. Replicate TripoSGAttnProcessor2_0 EXACTLY:
                // cat([q,k,v], -1) -> (B, L, heads, 3*head_dim) -> split the
                // last dim into thirds per head (NOT head-major thirds of the
                // fused matrix), then transpose to heads-first.
                let q = self.to_q.forward(xs)?;
                let k = self.to_k.forward(xs)?;
                let v = self.to_v.forward(xs)?;
                let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?.reshape((
                    batch,
                    seq,
                    self.heads,
                    3 * self.head_dim,
                ))?;
                let parts = qkv.chunk(3, D::Minus1)?;
                let q = parts[0]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_q_weight)?;
                let k = parts[1]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_k_weight)?;
                let v = parts[2].transpose(1, 2)?.contiguous()?;
                let out = exact_query_chunked_sdpa(&q, &k, &v, scale, attn_chunk)?;
                let out = out
                    .transpose(1, 2)?
                    .reshape((batch, seq, self.heads * self.head_dim))?;
                self.to_out.forward(&out)
            }
            Some(encoder_states) => {
                // Cross-attention: query is a STANDARD head-major split of
                // to_q(x); k/v come from the encoder states and are re-split
                // from cat([k, v], -1) with per-head stride 2*head_dim.
                let (enc_batch, enc_seq, enc_dim) = encoder_states.dims3()?;
                if enc_batch != batch {
                    candle_core::bail!(
                        "TripoSG cross-attention batch mismatch: {enc_batch} vs {batch}"
                    );
                }
                if enc_dim != self.to_k.in_dim() {
                    candle_core::bail!(
                        "TripoSG cross-attention input dim {} does not match to_k weight {}",
                        enc_dim,
                        self.to_k.in_dim()
                    );
                }
                let q = self
                    .to_q
                    .forward(xs)?
                    .reshape((batch, seq, self.heads, self.head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_q_weight)?;
                let k = self.to_k.forward(encoder_states)?;
                let v = self.to_v.forward(encoder_states)?;
                let kv = Tensor::cat(&[&k, &v], D::Minus1)?.reshape((
                    enc_batch,
                    enc_seq,
                    self.heads,
                    2 * self.head_dim,
                ))?;
                let kv_parts = kv.chunk(2, D::Minus1)?;
                let k = kv_parts[0]
                    .transpose(1, 2)?
                    .contiguous()?
                    .rms_norm_heads(&self.norm_k_weight)?;
                let v = kv_parts[1].transpose(1, 2)?.contiguous()?;
                let out = exact_sdpa_heads(&q, &k, &v, scale)?;
                let out = out
                    .transpose(1, 2)?
                    .reshape((batch, seq, self.heads * self.head_dim))?;
                self.to_out.forward(&out)
            }
        }
    }
}

/// RMSNorm over the last (head) dim with a per-head `[head_dim]` weight:
/// `x * rsqrt(mean(x^2, last) + eps) * weight` (diffusers `RMSNorm`, F32).
trait RmsNormHeads {
    fn rms_norm_heads(&self, weight: &Tensor) -> CandleResult<Tensor>;
}

impl RmsNormHeads for Tensor {
    fn rms_norm_heads(&self, weight: &Tensor) -> CandleResult<Tensor> {
        let eps = 1e-6_f64;
        let variance = self.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
        let normed = self.broadcast_div(&(variance + eps)?.sqrt()?)?;
        normed.broadcast_mul(weight)
    }
}

/// Rectified-flow Euler schedule (`RectifiedFlowScheduler.set_timesteps`).
///
/// `sigmas[i] = shift*s_i / (1 + (shift-1)*s_i)` with
/// `s_i = (1 - i/steps)`, plus a terminal 0.0; `timesteps[i] = sigmas[i] *
/// 1000` (F32, matching the torch `sigmas * num_train_timesteps`). `shift=1`
/// is the identity (the shipped TripoSG config).
#[derive(Debug, Clone)]
pub struct RectifiedFlowSchedule {
    /// Length `steps + 1`; last entry is the terminal 0.0.
    pub sigmas: Vec<f32>,
}

impl RectifiedFlowSchedule {
    pub fn new(steps: usize, shift: f32) -> Self {
        let mut sigmas = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let timestep = (1.0 - i as f64 / steps as f64) * 1000.0;
            let sigma = timestep / 1000.0;
            let sigma = shift as f64 * sigma / (1.0 + (shift as f64 - 1.0) * sigma);
            sigmas.push(sigma as f32);
        }
        sigmas.push(0.0);
        Self { sigmas }
    }

    /// The per-step timesteps fed to the DiT (`sigmas[i] * 1000`, F32).
    pub fn timesteps(&self) -> Vec<f32> {
        self.sigmas[..self.sigmas.len() - 1]
            .iter()
            .map(|sigma| sigma * 1000.0)
            .collect()
    }

    pub fn timestep(&self, index: usize) -> f32 {
        self.sigmas[index] * 1000.0
    }

    /// Euler update `x + (sigma_i - sigma_{i+1}) * v` in F32
    /// (`RectifiedFlowScheduler.step` with the sample upcast to F32).
    pub fn step(
        &self,
        index: usize,
        model_output: &Tensor,
        sample: &Tensor,
    ) -> CandleResult<Tensor> {
        if index + 1 >= self.sigmas.len() {
            candle_core::bail!(
                "RectifiedFlowSchedule step index {index} out of range ({} sigmas)",
                self.sigmas.len()
            );
        }
        let sigma = self.sigmas[index];
        let sigma_next = self.sigmas[index + 1];
        let delta = sigma - sigma_next;
        let sample = sample.to_dtype(DType::F32)?;
        let update = model_output
            .to_dtype(DType::F32)?
            .affine(delta as f64, 0.0)?;
        sample + update
    }
}

/// CFG denoise loop (`TripoSGPipeline.__call__` steps 4-6, denoise part only;
/// no RNG — the initial latents are supplied by the caller).
///
/// `latents_init [B=1, L, 64]`, `cond [B=1, 257, 1024]` conditional image
/// embeds (the uncond branch is `zeros_like(cond)`, batch order
/// `[uncond, cond]`). Public for the parity test harness.
pub fn denoise(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    cond: &Tensor,
    steps: usize,
    guidance: f32,
) -> CandleResult<Tensor> {
    denoise_with_dtype(dit, latents_init, cond, steps, guidance, DType::F32)
}

/// [`denoise`] with an explicit compute dtype. The network runs in `dtype`;
/// the Euler update stays in F32 (`RectifiedFlowScheduler.step` contract) and
/// is cast back for the next network pass. The returned latents are F32.
pub fn denoise_with_dtype(
    dit: &TripoSgDiT,
    latents_init: &Tensor,
    cond: &Tensor,
    steps: usize,
    guidance: f32,
    dtype: DType,
) -> CandleResult<Tensor> {
    let scheduler = RectifiedFlowSchedule::new(steps, 1.0);
    let latents_init = latents_init.to_dtype(dtype)?;
    let cond = cond.to_dtype(dtype)?;
    let uncond_cond = Tensor::zeros_like(&cond)?;
    let mut latents = latents_init;
    for index in 0..steps {
        let batch = Tensor::cat(&[&latents, &latents], 0)?;
        let cond_batch = Tensor::cat(&[&uncond_cond, &cond], 0)?;
        let velocity = dit.forward(&batch, scheduler.timestep(index), &cond_batch)?;
        let parts = velocity.chunk(2, 0)?;
        // v = v_uncond + guidance * (v_cond - v_uncond), batch 0 = uncond.
        let combined = (&parts[0] + (&parts[1] - &parts[0])?.affine(guidance as f64, 0.0)?)?;
        latents = scheduler
            .step(index, &combined, &latents)?
            .to_dtype(dtype)?;
    }
    latents.to_dtype(DType::F32)
}
