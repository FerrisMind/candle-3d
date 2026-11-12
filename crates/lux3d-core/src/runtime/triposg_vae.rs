//! TripoSG VAE SDF decoder: latent tokens -> per-point occupancy logits.
//!
//! Reference semantics (`triposg/models/autoencoders/autoencoder_kl_triposg.py`,
//! `triposg/models/embeddings.py`, `triposg/models/transformers/triposg_transformer.py`
//! `DiTBlock`, `triposg/models/attention_processor.py` `TripoSGAttnProcessor2_0`):
//!
//! - `post_quant` Linear(latent_channels -> width) turns the `[B, 2048, 64]`
//!   latents into the KV memory; decoder blocks 0..15 (self-attention only,
//!   dim 1024, 8 heads, head_dim 128, no QK norm, exact-GELU FFN 4x) run over
//!   the 2048 latent tokens ONCE; their output is the kv cache.
//! - The LAST block (16) is cross-attention ONLY: `norm2` on the query,
//!   `attn2` with `norm_cross` = LayerNorm(1024, eps 1e-5, affine) applied to
//!   the kv cache BEFORE the to_k/to_v projections, `norm3` + FFN.
//! - Query path per chunk: frequency embedder `[x, sin(x*2^i), cos(x*2^i)]`
//!   (i = 0..7 per coordinate, raw input FIRST, no pi -> 51 dims) ->
//!   `proj_query` Linear(51 -> 1024) -> block 16 -> `norm_out` -> `proj_out`
//!   Linear(1024 -> 1) -> the result is NEGATED (reference `logits * -1`).
//!   No output activation: marching cubes runs at level 0 on the negated
//!   logits (positive = inside).
//! - Attention re-splits the projections EXACTLY like the DiT
//!   (`TripoSGAttnProcessor2_0`): self-attention concatenates `[q, k, v]` on
//!   the channel dim then re-splits per head (head h takes columns
//!   `[384h : 384h+128)` of each of q/k/v); the cross block uses a standard
//!   head-major query and re-splits `cat([k, v])` with per-head stride
//!   `2 * head_dim`. No QK norm anywhere in the VAE. SDPA scale is the
//!   default `1/sqrt(head_dim)`.
//!
//! Everything stays F32 end to end (the canonical weights are F32 and the
//! parity baseline was generated in F32). All ops are backend-generic Candle
//! primitives; the cross-attention scores are computed in query chunks
//! (`exact_query_chunked_sdpa`, bounded by `LUX3D_TRIPOSG_ATTN_CHUNK` and the
//! shared SDPA score budget).

use candle_core::{D, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Module, layer_norm};
use crate::weights::VarBuilder;
use super::nn_blocks::{
    Linear,
    linear_dense as linear,
    linear_no_bias_dense as linear_no_bias,
};

use super::attention_math::exact_query_chunked_sdpa;

/// Self-attention query chunk (`LUX3D_TRIPOSG_ATTN_CHUNK`, default 1024) —
/// the same knob the DiT uses.
fn attn_chunk_size() -> usize {
    static CHUNK: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("LUX3D_TRIPOSG_ATTN_CHUNK")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1024)
    })
}

/// Parsed `vae` section of the canonical `resolved_config.json`.
///
/// Defaults replicate `TripoSGVAEModel.__init__` (`latent_channels=64`,
/// `width_decoder=1024`, `num_attention_heads=8`, `num_layers_decoder=16`,
/// `embed_frequency=8`, `embed_include_pi=false`) plus the hardcoded
/// `mlp_ratio=4.0` FFN. `num_layers_decoder` counts the SELF-attention
/// blocks; the single cross-attention block (decoder block 16) is added on
/// top, exactly like the reference `ModuleList`.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgVaeConfig {
    pub in_channels: usize,
    pub latent_channels: usize,
    pub width_encoder: usize,
    pub dim: usize,
    pub num_attention_heads: usize,
    pub num_layers_encoder: usize,
    pub num_layers_decoder: usize,
    pub embed_frequency: usize,
    pub embed_include_pi: bool,
}

impl Default for TripoSgVaeConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 64,
            width_encoder: 512,
            dim: 1024,
            num_attention_heads: 8,
            num_layers_encoder: 8,
            num_layers_decoder: 16,
            embed_frequency: 8,
            embed_include_pi: false,
        }
    }
}

impl TripoSgVaeConfig {
    /// Parse from the canonical `resolved_config.json` document (or directly
    /// from its `vae` subsection). Unknown/missing fields fall back to the
    /// reference defaults.
    pub fn from_resolved_config(value: &serde_json::Value) -> CandleResult<Self> {
        let source = value.get("vae").unwrap_or(value);
        let get_usize = |key: &str, default: usize| -> usize {
            source
                .get(key)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let cfg = Self {
            in_channels: get_usize("in_channels", 3),
            latent_channels: get_usize("latent_channels", 64),
            width_encoder: get_usize("width_encoder", 512),
            dim: get_usize("width_decoder", 1024),
            num_attention_heads: get_usize("num_attention_heads", 8),
            num_layers_encoder: get_usize("num_layers_encoder", 8),
            num_layers_decoder: get_usize("num_layers_decoder", 16),
            embed_frequency: get_usize("embed_frequency", 8),
            embed_include_pi: source
                .get("embed_include_pi")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> CandleResult<()> {
        if self.in_channels == 0
            || self.latent_channels == 0
            || self.width_encoder == 0
            || self.dim == 0
            || self.num_layers_encoder == 0
            || self.num_layers_decoder == 0
        {
            candle_core::bail!(
                "TripoSG VAE config has a zero dimension: in_channels {}, latent_channels {}, \
                 width_encoder {}, width_decoder {}, num_layers_encoder {}, \
                 num_layers_decoder {}",
                self.in_channels,
                self.latent_channels,
                self.width_encoder,
                self.dim,
                self.num_layers_encoder,
                self.num_layers_decoder
            );
        }
        if self.num_attention_heads == 0 || self.dim % self.num_attention_heads != 0 {
            candle_core::bail!(
                "TripoSG VAE width_decoder {} is not divisible by num_attention_heads {}",
                self.dim,
                self.num_attention_heads
            );
        }
        if self.width_encoder % self.num_attention_heads != 0 {
            candle_core::bail!(
                "TripoSG VAE width_encoder {} is not divisible by num_attention_heads {}",
                self.width_encoder,
                self.num_attention_heads
            );
        }
        if self.embed_frequency == 0 || self.embed_frequency > 16 {
            candle_core::bail!(
                "TripoSG VAE embed_frequency {} out of supported range 1..=16",
                self.embed_frequency
            );
        }
        Ok(())
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.num_attention_heads
    }

    /// FFN inner width: `mlp_ratio = 4.0` hardcoded in the reference.
    pub(crate) fn ff_inner_dim(&self) -> usize {
        self.dim * 4
    }

    /// Frequency embedder output width: `in_channels * (2 * num_freqs + 1)`
    /// with the raw input included (51 for the reference config).
    pub(crate) fn embed_out_dim(&self) -> usize {
        self.in_channels * (2 * self.embed_frequency + 1)
    }

    /// Encoder `proj_in` input width: frequency embedding plus raw features
    /// (`in_channels + embed_out_dim`, 54 for the reference config).
    pub fn encoder_input_dim(&self) -> usize {
        self.in_channels + self.embed_out_dim()
    }
}

/// TripoSG VAE decoder (decode path only; the encoder/quant weights stay in
/// the canonical file but are never loaded). Public for the parity test
/// harness; the pipeline (T4) drives it through `decode_latents`.
#[derive(Debug)]
pub struct TripoSgVaeDecoder {
    cfg: TripoSgVaeConfig,
    post_quant: Linear,
    self_blocks: Vec<VaeSelfBlock>,
    cross_block: VaeCrossBlock,
    norm_out: LayerNorm,
    proj_out: Linear,
    attn_chunk: usize,
}

impl TripoSgVaeDecoder {
    /// Load from a VarBuilder rooted at the canonical `vae.` prefix (i.e.
    /// pass `vb.pp("vae")` from the canonical weight set). Only
    /// `post_quant.*` and `decoder.*` keys are touched.
    pub fn load(vb: VarBuilder, cfg: &TripoSgVaeConfig) -> CandleResult<Self> {
        let post_quant = linear(cfg.latent_channels, cfg.dim, vb.pp("post_quant"))?;
        let decoder_vb = vb.pp("decoder");
        let self_blocks = (0..cfg.num_layers_decoder)
            .map(|index| {
                let block = VaeSelfBlock::load(decoder_vb.pp("blocks").pp(index.to_string()), cfg)?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let cross_block = VaeCrossBlock::load(
            decoder_vb
                .pp("blocks")
                .pp(cfg.num_layers_decoder.to_string()),
            decoder_vb.pp("proj_query"),
            cfg,
        )?;
        let norm_out = layer_norm(cfg.dim, 1e-5, decoder_vb.pp("norm_out"))?;
        let proj_out = linear(cfg.dim, 1, decoder_vb.pp("proj_out"))?;

        Ok(Self {
            cfg: cfg.clone(),
            post_quant,
            self_blocks,
            cross_block,
            norm_out,
            proj_out,
            attn_chunk: attn_chunk_size(),
        })
    }

    /// Run decoder blocks 0..15 (self-attention only) over the latent tokens.
    ///
    /// `latents [B, L, latent_channels]` (L = 2048, B = 1 for the pipeline)
    /// -> kv cache `[B, L, dim]`. Computed ONCE per decode call; reuse it for
    /// every query chunk / refinement level.
    pub fn kv_cache(&self, latents: &Tensor) -> CandleResult<Tensor> {
        let (_batch, _seq, channels) = latents.dims3()?;
        if channels != self.cfg.latent_channels {
            candle_core::bail!(
                "TripoSG VAE expects {} latent channels, got {channels}",
                self.cfg.latent_channels
            );
        }
        let mut hidden = self.post_quant.forward(latents)?;
        for block in &self.self_blocks {
            let attn_out = block
                .attn1
                .forward(&block.norm1.forward(&hidden)?, self.attn_chunk)?;
            hidden = (hidden + attn_out)?;
            let ff = block.ff_out.forward(
                &block
                    .ff_in
                    .forward(&block.norm3.forward(&hidden)?)?
                    .gelu_erf()?,
            )?;
            hidden = (hidden + ff)?;
        }
        Ok(hidden)
    }

    /// Query the SDF field at `points`.
    ///
    /// `kv_cache [B, L, dim]` from [`Self::kv_cache`] (the pipeline and the
    /// grid builder use B = 1), `points [N, 3]` world coordinates. Returns
    /// the NEGATED decoder logits as `[N]` F32 (positive = inside; marching
    /// cubes level 0). `chunk` bounds the number of query points per decoder
    /// pass (reference `num_chunks=50000`; the pipeline default is 16384).
    pub fn decode_points(
        &self,
        kv_cache: &Tensor,
        points: &Tensor,
        chunk: usize,
    ) -> CandleResult<Tensor> {
        let kv_dims = kv_cache.dims();
        if kv_dims.len() != 3 || kv_dims[2] != self.cfg.dim {
            candle_core::bail!(
                "TripoSG VAE kv cache must be [B, L, {}], got {kv_dims:?}",
                self.cfg.dim
            );
        }
        if kv_dims[0] != 1 {
            candle_core::bail!(
                "TripoSG VAE decode_points expects a batch-1 kv cache (expand the points \
                 per batch element instead), got batch {}",
                kv_dims[0]
            );
        }
        if points.rank() != 2 || points.dim(1)? != 3 {
            candle_core::bail!(
                "TripoSG VAE query points must be [N, 3], got {:?}",
                points.dims()
            );
        }

        // norm_cross + to_k/to_v depend only on the kv cache: compute the
        // cross-attention K/V ONCE and reuse them for every query chunk
        // (numerically identical to the reference, which re-projects per
        // chunk from the same cache).
        let (k_heads, v_heads) = self.cross_block.attn2.prepare_kv(kv_cache)?;

        let num_points = points.dim(0)?;
        let step = chunk.max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < num_points {
            let len = (num_points - start).min(step);
            let query_points = points.narrow(0, start, len)?.contiguous()?;
            let embedded = self.embed_points(&query_points)?; // [len, 51]

            // Block 16 (cross-attention only): x += attn2(norm2(x), kv);
            // x += ff(norm3(x)). The reference treats the query chunk as a
            // batch-1 sequence ([1, N, 3] -> [1, N, dim]); restore that rank
            // after the flat projection.
            let mut hidden =
                self.cross_block
                    .proj_query
                    .forward(&embedded)?
                    .reshape((1, len, self.cfg.dim))?;
            let attn_out = self.cross_block.attn2.forward(
                &self.cross_block.norm2.forward(&hidden)?,
                &k_heads,
                &v_heads,
                self.attn_chunk,
            )?;
            hidden = (hidden + attn_out)?;
            let ff = self.cross_block.ff_out.forward(
                &self
                    .cross_block
                    .ff_in
                    .forward(&self.cross_block.norm3.forward(&hidden)?)?
                    .gelu_erf()?,
            )?;
            hidden = (hidden + ff)?;

            // norm_out -> proj_out ([len, 1]) -> NEGATE -> [len].
            let logits = self.proj_out.forward(&self.norm_out.forward(&hidden)?)?;
            chunks.push(logits.flatten_all()?.affine(-1.0, 0.0)?);
            start += len;
        }
        Tensor::cat(&chunks, 0)
    }

    /// One-shot decode: `kv_cache(latents)` then `decode_points`.
    pub fn decode_latents(
        &self,
        latents: &Tensor,
        points: &Tensor,
        chunk: usize,
    ) -> CandleResult<Tensor> {
        let kv = self.kv_cache(latents)?;
        self.decode_points(&kv, points, chunk)
    }

    /// Frequency positional embedding of query points, reference
    /// `FrequencyPositionalEmbedding(num_freqs=8, logspace=True,
    /// input_dim=3, include_input=True, include_pi=False)`:
    /// `cat([x, sin(x * 2^i), cos(x * 2^i)])` with the sin / cos terms
    /// grouped BY AXIS (x0*f0..f7, x1*f0..f7, x2*f0..f7), raw input first ->
    /// `[N, 3 * (2 * num_freqs + 1)]` (51 dims for the reference config).
    fn embed_points(&self, points: &Tensor) -> CandleResult<Tensor> {
        let (rows, cols) = points.dims2()?;
        let num_freqs = self.cfg.embed_frequency;
        let mut freqs = Vec::with_capacity(num_freqs);
        for i in 0..num_freqs {
            // logspace: f_i = 2^i (exact in F32 for the supported range).
            let f = 2.0_f32.powi(i as i32);
            freqs.push(if self.cfg.embed_include_pi {
                f * std::f32::consts::PI
            } else {
                f
            });
        }
        let freqs = Tensor::from_vec(freqs, (1, 1, num_freqs), points.device())?;
        let products = points
            .reshape((rows, cols, 1))?
            .broadcast_mul(&freqs)?
            .reshape((rows, cols * num_freqs))?;
        Tensor::cat(&[points, &products.sin()?, &products.cos()?], D::Minus1)
    }
}

/// TripoSG VAE point encoder (`vae.encoder.*` + `vae.quant.*`), the exact
/// counterpart of the PyTorch `TripoSGEncoder`:
///
/// - `proj_in` Linear(54 -> 512) is applied to both the sampled query points
///   and the full key/value point cloud;
/// - block 0 is cross-attention ONLY (`norm2` query LayerNorm, `attn2` with
///   `norm_cross` on the kv memory, FFN);
/// - blocks 1..8 are self-attention + FFN at `width_encoder = 512`;
/// - `norm_out` then `quant` Linear(512 -> 2 * latent_channels).
///
/// The same `TripoSGAttnProcessor2_0` qkv re-split as the decoder is used:
/// self-attention re-splits `cat([q,k,v])` per head, cross-attention uses the
/// standard query split and re-splits `cat([k,v])` per head. No QK norm.
#[derive(Debug)]
pub struct TripoSgVaeEncoder {
    cfg: TripoSgVaeConfig,
    proj_in: Linear,
    cross_block: VaeEncoderCrossBlock,
    self_blocks: Vec<VaeSelfBlock>,
    norm_out: LayerNorm,
    quant: Linear,
    attn_chunk: usize,
}

impl TripoSgVaeEncoder {
    /// Load from a VarBuilder rooted at the canonical `vae.` prefix.
    pub fn load(vb: VarBuilder, cfg: &TripoSgVaeConfig) -> CandleResult<Self> {
        let encoder_vb = vb.pp("encoder");
        let self_blocks = (1..=cfg.num_layers_encoder)
            .map(|index| {
                let block = VaeSelfBlock::load_with(
                    encoder_vb.pp("blocks").pp(index.to_string()),
                    cfg.width_encoder,
                    cfg.num_attention_heads,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            cfg: cfg.clone(),
            proj_in: linear(
                cfg.encoder_input_dim(),
                cfg.width_encoder,
                encoder_vb.pp("proj_in"),
            )?,
            cross_block: VaeEncoderCrossBlock::load(encoder_vb.pp("blocks").pp("0"), cfg)?,
            self_blocks,
            norm_out: layer_norm(cfg.width_encoder, 1e-5, encoder_vb.pp("norm_out"))?,
            quant: linear(cfg.width_encoder, cfg.latent_channels * 2, vb.pp("quant"))?,
            attn_chunk: attn_chunk_size(),
        })
    }

    /// Frequency-embed positions and concatenate raw feature channels.
    /// `x [B, N, C]` where the first `in_channels` channels are positions and
    /// the rest are features (reference `[x, sin(x*2^i), cos(x*2^i)]`).
    pub fn embed_features(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 || dims[2] < self.cfg.in_channels {
            candle_core::bail!(
                "TripoSG VAE encoder input must be [B, N, C] with C >= {}, got {:?}",
                self.cfg.in_channels,
                dims
            );
        }
        let positions = x.narrow(D::Minus1, 0, self.cfg.in_channels)?;
        let features = x.narrow(
            D::Minus1,
            self.cfg.in_channels,
            dims[2] - self.cfg.in_channels,
        )?;
        let embedded = self.embed_positions(&positions)?;
        if features.dim(D::Minus1)? == 0 {
            Ok(embedded)
        } else {
            Tensor::cat(&[&embedded, &features], D::Minus1)
        }
    }

    /// Run the encoder on already-embedded point features.
    ///
    /// `x_q [B, L, 54]`, `x_kv [B, N, 54]` -> `[B, L, 2 * latent_channels]`.
    pub fn encode_embedded(&self, x_q: &Tensor, x_kv: &Tensor) -> CandleResult<Tensor> {
        let q_dims = x_q.dims3()?;
        let kv_dims = x_kv.dims3()?;
        if q_dims.0 != kv_dims.0 {
            candle_core::bail!(
                "TripoSG VAE encoder batch mismatch: query {}, kv {}",
                q_dims.0,
                kv_dims.0
            );
        }
        if q_dims.2 != self.cfg.encoder_input_dim() || kv_dims.2 != self.cfg.encoder_input_dim() {
            candle_core::bail!(
                "TripoSG VAE encoder expected {} feature channels, got query {} / kv {}",
                self.cfg.encoder_input_dim(),
                q_dims.2,
                kv_dims.2
            );
        }
        let mut hidden = self.proj_in.forward(x_q)?;
        let encoder_hidden = self.proj_in.forward(x_kv)?;
        hidden = self
            .cross_block
            .forward(&hidden, &encoder_hidden, self.attn_chunk)?;
        for block in &self.self_blocks {
            hidden = block.forward(&hidden, self.attn_chunk)?;
        }
        let hidden = self.norm_out.forward(&hidden)?;
        self.quant.forward(&hidden)
    }

    /// Encode raw point features with an explicit sampled query subset.
    ///
    /// `x [B, N, C]` (positions + features), `sampled_x [B, L, C]` is the
    /// reference `_sample_features` output. This is the parity/injection entry
    /// point: callers that already have sampling indices can reproduce the
    /// reference selection exactly.
    pub fn encode_sampled(&self, x: &Tensor, sampled_x: &Tensor) -> CandleResult<Tensor> {
        let q = self.embed_features(sampled_x)?;
        let kv = self.embed_features(x)?;
        self.encode_embedded(&q, &kv)
    }

    /// Full reference encode: sample `num_tokens * 4` indices, run FPS at
    /// ratio 1/4, then encode. `seed = None` uses a process-time seed (the
    /// reference uses `np.random.default_rng(None)`).
    pub fn encode(
        &self,
        x: &Tensor,
        num_tokens: usize,
        seed: Option<u64>,
    ) -> CandleResult<TripoSgLatentDistribution> {
        if num_tokens == 0 {
            candle_core::bail!("TripoSG VAE encode num_tokens must be at least 1");
        }
        let sampled = sample_point_features(x, num_tokens, seed)?;
        let parameters = self.encode_sampled(x, &sampled)?;
        TripoSgLatentDistribution::from_parameters(&parameters)
    }

    /// Sample features using an explicit initial NumPy-choice index vector.
    /// Exposed for parity tests; the normal path is [`Self::encode`].
    pub fn sample_with_indices(
        &self,
        x: &Tensor,
        indices: &[usize],
        num_tokens: usize,
    ) -> CandleResult<Tensor> {
        sample_point_features_with_indices(x, indices, num_tokens)
    }

    fn embed_positions(&self, positions: &Tensor) -> CandleResult<Tensor> {
        let dims = positions.dims3()?;
        let rows = dims.0 * dims.1;
        let cols = dims.2;
        let num_freqs = self.cfg.embed_frequency;
        let mut freqs = Vec::with_capacity(num_freqs);
        for i in 0..num_freqs {
            let f = 2.0_f32.powi(i as i32);
            freqs.push(if self.cfg.embed_include_pi {
                f * std::f32::consts::PI
            } else {
                f
            });
        }
        let freqs = Tensor::from_vec(freqs, (1, 1, num_freqs), positions.device())?;
        let products = positions
            .reshape((rows, cols, 1))?
            .broadcast_mul(&freqs)?
            .reshape((rows, cols * num_freqs))?;
        let embedded = Tensor::cat(
            &[
                &positions.reshape((rows, cols))?,
                &products.sin()?,
                &products.cos()?,
            ],
            D::Minus1,
        )?;
        embedded.reshape((dims.0, dims.1, cols * (2 * num_freqs + 1)))
    }
}

/// Encoder block 0: cross-attention only. Same diffusers `Attention`
/// semantics as the decoder cross block, but at `width_encoder`.
#[derive(Debug)]
struct VaeEncoderCrossBlock {
    norm2: LayerNorm,
    attn2: VaeCrossAttention,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl VaeEncoderCrossBlock {
    fn load(vb: VarBuilder, cfg: &TripoSgVaeConfig) -> CandleResult<Self> {
        let dim = cfg.width_encoder;
        Ok(Self {
            norm2: layer_norm(dim, 1e-5, vb.pp("norm2"))?,
            attn2: VaeCrossAttention::load_with(vb.pp("attn2"), dim, cfg.num_attention_heads)?,
            norm3: layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff_in: linear(dim, dim * 4, vb.pp("ff").pp("net").pp("0").pp("proj"))?,
            ff_out: linear(dim * 4, dim, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        encoder_hidden: &Tensor,
        attn_chunk: usize,
    ) -> CandleResult<Tensor> {
        let (k_heads, v_heads) = self.attn2.prepare_kv(encoder_hidden)?;
        let attn =
            self.attn2
                .forward(&self.norm2.forward(hidden)?, &k_heads, &v_heads, attn_chunk)?;
        let hidden = (hidden + attn)?;
        let ff = self.ff_out.forward(
            &self
                .ff_in
                .forward(&self.norm3.forward(&hidden)?)?
                .gelu_erf()?,
        )?;
        hidden + ff
    }
}

/// `DiagonalGaussianDistribution` for the TripoSG VAE latent (feature dim -1).
#[derive(Debug)]
pub struct TripoSgLatentDistribution {
    pub mean: Tensor,
    pub logvar: Tensor,
    pub std: Tensor,
}

impl TripoSgLatentDistribution {
    /// Split the quant output on the last dimension into mean and clamped
    /// log-variance (reference clamp `[-30, 20]`).
    pub fn from_parameters(parameters: &Tensor) -> CandleResult<Self> {
        let channels = parameters.dim(D::Minus1)?;
        if channels == 0 || channels % 2 != 0 {
            candle_core::bail!(
                "TripoSG VAE quant parameters must have an even last dimension, got {channels}"
            );
        }
        let half = channels / 2;
        let mean = parameters.narrow(D::Minus1, 0, half)?.contiguous()?;
        let logvar = parameters
            .narrow(D::Minus1, half, half)?
            .clamp(-30.0_f32, 20.0_f32)?;
        let std = (logvar.affine(0.5, 0.0)?).exp()?;
        Ok(Self { mean, logvar, std })
    }

    /// Reference `posterior.mode()` / the deterministic latent.
    pub fn mode(&self) -> Tensor {
        self.mean.clone()
    }

    /// `mean + std * noise` with an injected noise tensor (same shape).
    pub fn sample_with_noise(&self, noise: &Tensor) -> CandleResult<Tensor> {
        self.mean.add(&self.std.mul(noise)?)
    }

    /// Deterministic seed-based sample. The RNG is SplitMix64 + Box-Muller
    /// (documented Lux3D behavior, not torch's device RNG).
    pub fn sample(&self, seed: u64) -> CandleResult<Tensor> {
        let dims = self.mean.dims().to_vec();
        let count: usize = dims.iter().product();
        let mut rng = EncodeNormalRng::new(seed);
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(rng.next_standard_normal());
        }
        let noise = Tensor::from_vec(values, dims, &candle_core::Device::Cpu)?
            .to_device(self.mean.device())?
            .to_dtype(self.mean.dtype())?;
        self.sample_with_noise(&noise)
    }
}

/// Reference `_sample_features`: draw `num_tokens * 4` indices from NumPy (or
/// a documented Lux3D replacement), then run per-sample farthest-point
/// sampling at ratio 1/4. `random_start=False` matches eval mode.
pub fn sample_point_features(
    x: &Tensor,
    num_tokens: usize,
    seed: Option<u64>,
) -> CandleResult<Tensor> {
    let dims = x.dims3()?;
    if dims.1 == 0 || dims.2 == 0 {
        candle_core::bail!("TripoSG VAE sample features needs non-empty points/channels");
    }
    if num_tokens == 0 {
        candle_core::bail!("TripoSG VAE sample features num_tokens must be at least 1");
    }
    let count = num_tokens
        .checked_mul(4)
        .ok_or_else(|| candle_core::Error::Msg("TripoSG VAE sample count overflow".into()))?;
    let seed = seed.unwrap_or_else(random_process_seed);
    let mut rng = EncodeNormalRng::new(seed);
    let indices = if count > dims.1 {
        rng.choice_with_replacement(dims.1, count)
    } else {
        rng.choice_without_replacement(dims.1, count)
    };
    sample_point_features_with_indices(x, &indices, num_tokens)
}

/// FPS half of [`sample_point_features`] with an explicit initial index
/// vector (the reference `rng.choice` output).
pub fn sample_point_features_with_indices(
    x: &Tensor,
    indices: &[usize],
    num_tokens: usize,
) -> CandleResult<Tensor> {
    let dims = x.dims3()?;
    if indices.is_empty() {
        candle_core::bail!("TripoSG VAE sampling index vector must not be empty");
    }
    if num_tokens == 0 || num_tokens > indices.len() {
        candle_core::bail!(
            "TripoSG VAE FPS wants {num_tokens} tokens from {} selected points",
            indices.len()
        );
    }
    let index_tensor = Tensor::from_vec(
        indices
            .iter()
            .map(|index| *index as i64)
            .collect::<Vec<_>>(),
        (indices.len(),),
        &candle_core::Device::Cpu,
    )?
    .to_device(x.device())?;
    let mut batches = Vec::with_capacity(dims.0);
    for batch in 0..dims.0 {
        let row = x.narrow(0, batch, 1)?.squeeze(0)?;
        let selected = row.index_select(&index_tensor, 0)?;
        let positions = selected
            .narrow(1, 0, 3)?
            .to_device(&candle_core::Device::Cpu)?
            .to_vec2::<f32>()?;
        let fps = fps_indices(&positions, num_tokens);
        let fps_tensor = Tensor::from_vec(
            fps.iter().map(|index| *index as i64).collect::<Vec<_>>(),
            (fps.len(),),
            &candle_core::Device::Cpu,
        )?
        .to_device(x.device())?;
        batches.push(selected.index_select(&fps_tensor, 0)?.unsqueeze(0)?);
    }
    let refs = batches.iter().collect::<Vec<_>>();
    Tensor::cat(&refs, 0)
}

/// Deterministic farthest-point sampling (torch_cluster.fps with
/// `random_start=False`): start at index 0, then repeatedly take the point
/// with the largest squared distance to the nearest selected point. Ties pick
/// the lowest index.
fn fps_indices(points: &[Vec<f32>], count: usize) -> Vec<usize> {
    let n = points.len();
    let count = count.min(n);
    let mut selected = Vec::with_capacity(count);
    let mut min_dist = vec![f32::INFINITY; n];
    let mut current = 0usize;
    for step in 0..count {
        selected.push(current);
        let center = &points[current];
        for (index, point) in points.iter().enumerate() {
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            let dz = point[2] - center[2];
            let dist = dx * dx + dy * dy + dz * dz;
            if dist < min_dist[index] {
                min_dist[index] = dist;
            }
        }
        if step + 1 == count {
            break;
        }
        let mut best = 0usize;
        let mut best_dist = f32::NEG_INFINITY;
        for (index, dist) in min_dist.iter().enumerate() {
            if *dist > best_dist {
                best_dist = *dist;
                best = index;
            }
        }
        current = best;
    }
    selected
}

fn random_process_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ ((std::process::id() as u64) << 32) ^ COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Small SplitMix64 RNG used by the encode sampling API. It is deliberately
/// independent of torch/NumPy: injected indices/sampled points provide exact
/// cross-framework parity, while the public API stays deterministic per seed.
struct EncodeNormalRng {
    state: u64,
    spare: Option<f64>,
}

impl EncodeNormalRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_standard_normal(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare as f32;
        }
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        self.spare = Some(radius * angle.sin());
        (radius * angle.cos()) as f32
    }

    /// Unbiased index in `0..n` via Lemire rejection.
    fn next_index(&mut self, n: usize) -> usize {
        let bound = n as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            let product = (value as u128) * (bound as u128);
            let low = product as u64;
            if low >= threshold {
                return (product >> 64) as usize;
            }
        }
    }

    fn choice_without_replacement(&mut self, n: usize, count: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for index in 0..count {
            let swap = index + self.next_index(n - index);
            pool.swap(index, swap);
        }
        pool.truncate(count);
        pool
    }

    fn choice_with_replacement(&mut self, n: usize, count: usize) -> Vec<usize> {
        (0..count).map(|_| self.next_index(n)).collect()
    }
}

/// Decoder blocks 0..15: self-attention + FFN, both pre-normed, residual.
#[derive(Debug)]
struct VaeSelfBlock {
    norm1: LayerNorm,
    attn1: VaeSelfAttention,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl VaeSelfBlock {
    fn load(vb: VarBuilder, cfg: &TripoSgVaeConfig) -> CandleResult<Self> {
        Self::load_with(vb, cfg.dim, cfg.num_attention_heads)
    }

    fn load_with(vb: VarBuilder, dim: usize, heads: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            attn1: VaeSelfAttention::load_with(vb.pp("attn1"), dim, heads)?,
            norm3: layer_norm(dim, 1e-5, vb.pp("norm3"))?,
            ff_in: linear(dim, dim * 4, vb.pp("ff").pp("net").pp("0").pp("proj"))?,
            ff_out: linear(dim * 4, dim, vb.pp("ff").pp("net").pp("2"))?,
        })
    }

    fn forward(&self, hidden: &Tensor, attn_chunk: usize) -> CandleResult<Tensor> {
        let attn = self
            .attn1
            .forward(&self.norm1.forward(hidden)?, attn_chunk)?;
        let hidden = (hidden + attn)?;
        let ff = self.ff_out.forward(
            &self
                .ff_in
                .forward(&self.norm3.forward(&hidden)?)?
                .gelu_erf()?,
        )?;
        hidden + ff
    }
}

/// Decoder block 16: cross-attention + FFN. `norm_cross` lives INSIDE the
/// diffusers `Attention` and is applied to the kv cache before to_k/to_v.
#[derive(Debug)]
struct VaeCrossBlock {
    proj_query: Linear,
    norm2: LayerNorm,
    attn2: VaeCrossAttention,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl VaeCrossBlock {
    /// `block_vb` is rooted at `decoder.blocks.<num_layers_decoder>`;
    /// `proj_query_vb` at `decoder.proj_query` (the query projection lives on
    /// the decoder, not inside the block — reference `TripoSGDecoder`).
    fn load(
        block_vb: VarBuilder,
        proj_query_vb: VarBuilder,
        cfg: &TripoSgVaeConfig,
    ) -> CandleResult<Self> {
        Ok(Self {
            proj_query: linear(cfg.embed_out_dim(), cfg.dim, proj_query_vb)?,
            norm2: layer_norm(cfg.dim, 1e-5, block_vb.pp("norm2"))?,
            attn2: VaeCrossAttention::load(block_vb.pp("attn2"), cfg)?,
            norm3: layer_norm(cfg.dim, 1e-5, block_vb.pp("norm3"))?,
            ff_in: linear(
                cfg.dim,
                cfg.ff_inner_dim(),
                block_vb.pp("ff").pp("net").pp("0").pp("proj"),
            )?,
            ff_out: linear(
                cfg.ff_inner_dim(),
                cfg.dim,
                block_vb.pp("ff").pp("net").pp("2"),
            )?,
        })
    }
}

/// VAE self-attention: no QK norm, checkpoint re-split
/// `cat([q,k,v]) -> view(B, L, heads, 3*head_dim) -> chunk(3)`.
#[derive(Debug)]
struct VaeSelfAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    head_dim: usize,
}

impl VaeSelfAttention {
    fn load_with(vb: VarBuilder, dim: usize, heads: usize) -> CandleResult<Self> {
        if heads == 0 || dim % heads != 0 {
            candle_core::bail!("VAE attention width {dim} is not divisible by {heads} heads");
        }
        let head_dim = dim / heads;
        Ok(Self {
            to_q: linear_no_bias(dim, dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, dim, vb.pp("to_v"))?,
            to_out: linear(dim, dim, vb.pp("to_out").pp("0"))?,
            heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, attn_chunk: usize) -> CandleResult<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let q = self.to_q.forward(xs)?;
        let k = self.to_k.forward(xs)?;
        let v = self.to_v.forward(xs)?;
        // TripoSGAttnProcessor2_0 re-split: head h takes columns
        // [3*head_dim*h : 3*head_dim*h + head_dim) of each of q/k/v.
        let qkv = Tensor::cat(&[&q, &k, &v], D::Minus1)?.reshape((
            batch,
            seq,
            self.heads,
            3 * self.head_dim,
        ))?;
        let parts = qkv.chunk(3, D::Minus1)?;
        let q = parts[0].transpose(1, 2)?.contiguous()?;
        let k = parts[1].transpose(1, 2)?.contiguous()?;
        let v = parts[2].transpose(1, 2)?.contiguous()?;
        let out = exact_query_chunked_sdpa(&q, &k, &v, scale, attn_chunk)?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}

/// VAE cross-attention: standard head-major query, `norm_cross` LayerNorm on
/// the kv cache BEFORE the to_k/to_v projections, checkpoint re-split
/// `cat([k,v]) -> view(B, S, heads, 2*head_dim) -> chunk(2)`.
#[derive(Debug)]
struct VaeCrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_cross: LayerNorm,
    heads: usize,
    head_dim: usize,
}

impl VaeCrossAttention {
    fn load(vb: VarBuilder, cfg: &TripoSgVaeConfig) -> CandleResult<Self> {
        Self::load_with(vb, cfg.dim, cfg.num_attention_heads)
    }

    fn load_with(vb: VarBuilder, dim: usize, heads: usize) -> CandleResult<Self> {
        if heads == 0 || dim % heads != 0 {
            candle_core::bail!("VAE attention width {dim} is not divisible by {heads} heads");
        }
        let head_dim = dim / heads;
        Ok(Self {
            to_q: linear_no_bias(dim, dim, vb.pp("to_q"))?,
            to_k: linear_no_bias(dim, dim, vb.pp("to_k"))?,
            to_v: linear_no_bias(dim, dim, vb.pp("to_v"))?,
            to_out: linear(dim, dim, vb.pp("to_out").pp("0"))?,
            norm_cross: layer_norm(dim, 1e-5, vb.pp("norm_cross"))?,
            heads,
            head_dim,
        })
    }

    /// Apply `norm_cross` then the k/v projections, and pre-split the heads
    /// (per-head stride `2 * head_dim` over `cat([k, v])`). Called once per
    /// decode call; the result is shared by all query chunks.
    fn prepare_kv(&self, kv_cache: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (batch, seq, _) = kv_cache.dims3()?;
        let kv_normed = self.norm_cross.forward(kv_cache)?;
        let k = self.to_k.forward(&kv_normed)?;
        let v = self.to_v.forward(&kv_normed)?;
        let kv = Tensor::cat(&[&k, &v], D::Minus1)?.reshape((
            batch,
            seq,
            self.heads,
            2 * self.head_dim,
        ))?;
        let parts = kv.chunk(2, D::Minus1)?;
        let k = parts[0].transpose(1, 2)?.contiguous()?;
        let v = parts[1].transpose(1, 2)?.contiguous()?;
        Ok((k, v))
    }

    /// Cross-attention for one query chunk. `xs` is the pre-norm query tokens
    /// (norm2 is applied here, matching `x += attn2(norm2(x), kv)`); the
    /// query uses the STANDARD head-major split (the processor does not
    /// re-split the cross-attention query). `k_heads`/`v_heads` come from
    /// [`Self::prepare_kv`].
    fn forward(
        &self,
        xs: &Tensor,
        k_heads: &Tensor,
        v_heads: &Tensor,
        attn_chunk: usize,
    ) -> CandleResult<Tensor> {
        let (batch, seq, _) = xs.dims3()?;
        let scale = (1.0 / (self.head_dim as f64).sqrt()) as f32;
        let q = self
            .to_q
            .forward(xs)?
            .reshape((batch, seq, self.heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let out = exact_query_chunked_sdpa(&q, k_heads, v_heads, scale, attn_chunk)?;
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * self.head_dim))?;
        self.to_out.forward(&out)
    }
}
