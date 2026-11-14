//! TRELLIS.2 shape sparse VAE decoder (`FlexiDualGridVaeDecoder` /
//! `SparseUnetVaeDecoder`, checkpoint `shape_dec_next_dc_f16c32_fp16`, 292
//! stage tensors in the canonical package under `shape_dec.`).
//!
//! Reference: `trellis2/models/sc_vaes/fdg_vae.py` + `sparse_unet_vae.py`.
//!
//! Architecture (decoder config `shape_vae_next_dc_f16c32_fp16_ft_512.json`):
//!
//! - `from_latent` SparseLinear(32, 1024) (+ bias).
//! - Stages 0..3 (`num_blocks [4, 16, 8, 4]`): `num_blocks[s]` x
//!   `SparseConvNeXtBlock3d` then one `SparseResBlockC2S3d` (the C2S up block
//!   doubles the grid and halves the channels 1024 -> 512 -> 256 -> 128 -> 64).
//!   Stage 4 has 0 blocks (channels stay 64 at full resolution).
//! - `SparseConvNeXtBlock3d`: submanifold conv3x3x3 FIRST, then affine
//!   LayerNorm32 (eps 1e-6), then MLP Linear(C, 4C) -> SiLU -> Linear(4C, C);
//!   output `h + x`.
//! - `SparseResBlockC2S3d`: `subdiv = to_subdiv(x)` (Linear(C, 8) + bias);
//!   `h = norm1(x)` (affine, eps 1e-6) -> SiLU -> conv1(C -> 8 * Cout) ->
//!   C2S(h, subdiv); `x = C2S(x, subdiv)`; `h = norm2(h)` (NON-affine,
//!   eps 1e-6) -> SiLU -> conv2(Cout -> Cout); `h += skip` with
//!   `skip = C2S(x, subdiv)` channels repeated
//!   `repeat_interleave(out_channels // (channels // 8), dim=1)`.
//! - Final: non-affine `F.layer_norm(feats, eps 1e-5)`, then `output_layer`
//!   SparseLinear(64, 7).
//! - Dual-grid head: `dual = 3 * sigmoid(feats[0:3]) - 0.5`
//!   (`voxel_margin = 0.5`), `intersected = feats[3:6] > 0`,
//!   `split_weight = softplus(feats[6])`.
//!
//! All compute is F32 (locked decision 3); the checkpoint stores F16 and the
//! VarBuilder casts on access. Sparse conv and C2S come from
//! [`super::trellis2_sparse`] (T4); nothing is duplicated here.
//!
//! The texture decoder shares the same block set with `out_channels 6`
//! and `pred_subdiv = false`: its C2S blocks take the raw `to_subdiv` logits
//! captured from the shape decoder (`guide_subs`, one `[N_parents, 8]` tensor
//! per C2S stage, in stage order) instead of predicting subdivision
//! themselves — no `to_subdiv` weights exist in the `tex_dec.` checkpoint
//! (284 stage tensors = 292 - 4 x 2). See [`super::trellis2_tex`].

use candle_core::{DType, Result as CandleResult, Tensor};

use super::nn_blocks::{Linear, linear};
use super::trellis2_blocks::layer_norm_no_affine;
use super::trellis2_mesh::{DualGridFields, DualGridMesh, flexible_dual_grid_to_mesh};
use super::trellis2_sparse::{SparseConv3d, channel_to_spatial, repeat_channels};
use crate::weights::VarBuilder;

/// Shape/texture sparse VAE decoder architecture parameters
/// (`shape_vae_next_dc_f16c32_fp16_ft_512.json` decoder args; the texture
/// checkpoint `tex_dec_next_dc_f16c32_fp16.json` shares the topology with
/// `out_channels 6` and `pred_subdiv false`).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2ShapeVaeConfig {
    /// Channel pyramid in decoder order ([1024, 512, 256, 128, 64]).
    pub model_channels: Vec<usize>,
    /// ConvNeXt blocks per stage ([4, 16, 8, 4, 0]; a C2S block follows each
    /// stage except the last).
    pub num_blocks: Vec<usize>,
    /// Latent input channels (32).
    pub latent_channels: usize,
    /// Decoded output channels (7 for shape: 3 dual + 3 intersected logits +
    /// 1 split weight; 6 for texture: base_color 3 + metallic + roughness +
    /// alpha).
    pub out_channels: usize,
    /// ConvNeXt MLP ratio (4.0).
    pub mlp_ratio: f64,
    /// When true (shape decoder) each C2S block carries a `to_subdiv`
    /// Linear(C, 8) and returns the raw logits; when false (texture decoder)
    /// the C2S blocks consume external guide logits.
    pub pred_subdiv: bool,
}

impl Trellis2ShapeVaeConfig {
    /// `FlexiDualGridVaeDecoder` shape checkpoint args (resolution 512
    /// finetune; the resolution only affects mesh extraction, not weights).
    pub fn flexi_dual_shape() -> Self {
        Self {
            model_channels: vec![1024, 512, 256, 128, 64],
            num_blocks: vec![4, 16, 8, 4, 0],
            latent_channels: 32,
            out_channels: 7,
            mlp_ratio: 4.0,
            pred_subdiv: true,
        }
    }

    /// `SparseUnetVaeDecoder` texture checkpoint args
    /// (`tex_dec_next_dc_f16c32_fp16.json`): identical channel ladder and
    /// block counts, 6 output channels, `pred_subdiv = false` (the C2S
    /// blocks take the shape decoder's guide logits).
    pub fn tex_decoder() -> Self {
        Self {
            out_channels: 6,
            pred_subdiv: false,
            ..Self::flexi_dual_shape()
        }
    }

    /// Number of C2S up blocks (= one per stage except the last).
    pub fn c2s_stage_count(&self) -> usize {
        self.num_blocks.len().saturating_sub(1)
    }

    fn stage_has_c2s(&self, stage: usize) -> bool {
        stage + 1 < self.num_blocks.len()
    }

    /// Channels entering stage `stage` (= `model_channels[stage]`; the C2S
    /// block of the previous stage outputs `model_channels[stage]`).
    fn stage_in_channels(&self, stage: usize) -> usize {
        self.model_channels[stage]
    }

    /// The exact tensor names this decoder consumes below its prefix
    /// (292 for the shape checkpoint: 2 + 2 top-level plus
    /// `sum(num_blocks[s] * 8 + 8)` for stages 0..3; 284 for the texture
    /// checkpoint, which drops the four `to_subdiv` weight/bias pairs).
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.push("from_latent.weight".to_string());
        names.push("from_latent.bias".to_string());
        names.push("output_layer.weight".to_string());
        names.push("output_layer.bias".to_string());
        for (stage, &count) in self.num_blocks.iter().enumerate() {
            for block in 0..count {
                let base = format!("blocks.{stage}.{block}.");
                names.push(format!("{base}conv.weight"));
                names.push(format!("{base}conv.bias"));
                names.push(format!("{base}norm.weight"));
                names.push(format!("{base}norm.bias"));
                names.push(format!("{base}mlp.0.weight"));
                names.push(format!("{base}mlp.0.bias"));
                names.push(format!("{base}mlp.2.weight"));
                names.push(format!("{base}mlp.2.bias"));
            }
            if self.stage_has_c2s(stage) {
                let channels = self.stage_in_channels(stage);
                let out_channels = self.model_channels[stage + 1];
                let base = format!("blocks.{stage}.{count}.");
                names.push(format!("{base}conv1.weight"));
                names.push(format!("{base}conv1.bias"));
                names.push(format!("{base}conv2.weight"));
                names.push(format!("{base}conv2.bias"));
                names.push(format!("{base}norm1.weight"));
                names.push(format!("{base}norm1.bias"));
                if self.pred_subdiv {
                    names.push(format!("{base}to_subdiv.weight"));
                    names.push(format!("{base}to_subdiv.bias"));
                }
                debug_assert_eq!(channels % 8, 0);
                debug_assert_eq!(out_channels * 8 % channels, 0);
            }
        }
        names
    }
}

/// Affine LayerNorm over the channel dim of `[N, C]` (`LayerNorm32` on sparse
/// features, biased variance, per-channel weight/bias).
fn channel_layer_norm_affine(
    xs: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> CandleResult<Tensor> {
    let (_rows, channels) = xs.dims2()?;
    let mean = xs.mean_keepdim(candle_core::D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let denom = var.affine(1.0, eps)?.sqrt()?;
    let normalized = centered.broadcast_div(&denom)?;
    normalized
        .broadcast_mul(&weight.reshape((1, channels))?)?
        .broadcast_add(&bias.reshape((1, channels))?)
}

/// `SparseConvNeXtBlock3d`: conv -> affine LN(eps 1e-6) -> MLP -> residual.
#[derive(Debug)]
struct ConvNeXtBlock {
    conv: SparseConv3d,
    norm_weight: Tensor,
    norm_bias: Tensor,
    mlp0: Linear,
    mlp2: Linear,
}

impl ConvNeXtBlock {
    fn load(vb: VarBuilder, channels: usize, mlp_ratio: f64) -> CandleResult<Self> {
        let mlp_hidden = (channels as f64 * mlp_ratio) as usize;
        Ok(Self {
            conv: SparseConv3d::load(vb.pp("conv"), channels, channels, true)?,
            norm_weight: vb.get(channels, "norm.weight")?.to_dtype(DType::F32)?,
            norm_bias: vb.get(channels, "norm.bias")?.to_dtype(DType::F32)?,
            mlp0: linear(vb.pp("mlp").pp("0"), channels, mlp_hidden, true)?,
            mlp2: linear(vb.pp("mlp").pp("2"), mlp_hidden, channels, true)?,
        })
    }

    /// `coords`/`feats`: `[N, C]` in, `[N, C]` out on the same coordinates.
    /// Reference `_forward`: `h = conv(x); h = norm(h); h = mlp(h)` with
    /// `mlp = Linear -> SiLU -> Linear`; residual `h + x`. No activation
    /// between the norm and the first MLP linear.
    fn forward(&self, coords: &[[i32; 3]], feats: &Tensor) -> CandleResult<Tensor> {
        let h = self.conv.forward(coords, feats)?;
        let h = channel_layer_norm_affine(&h, &self.norm_weight, &self.norm_bias, 1e-6)?;
        let h = self.mlp0.forward(&h)?;
        let h = h.silu()?;
        let h = self.mlp2.forward(&h)?;
        h.add(feats)
    }
}

/// `SparseResBlockC2S3d`: the C2S up block. With `pred_subdiv` the block
/// owns a `to_subdiv` Linear(C, 8) and emits the raw logits; otherwise
/// (`pred_subdiv = false`, the texture decoder) the caller supplies the
/// guide logits captured from the shape decoder at the same stage — the
/// block math is otherwise identical (shared implementation, no fork).
#[derive(Debug)]
struct C2SResBlock {
    norm1_weight: Tensor,
    norm1_bias: Tensor,
    conv1: SparseConv3d,
    conv2: SparseConv3d,
    to_subdiv: Option<Linear>,
    channels: usize,
    out_channels: usize,
}

impl C2SResBlock {
    fn load(
        vb: VarBuilder,
        channels: usize,
        out_channels: usize,
        pred_subdiv: bool,
    ) -> CandleResult<Self> {
        if channels % 8 != 0 {
            candle_core::bail!("C2S block channels must be divisible by 8, got {channels}");
        }
        let to_subdiv = if pred_subdiv {
            Some(linear(vb.pp("to_subdiv"), channels, 8, true)?)
        } else {
            None
        };
        Ok(Self {
            norm1_weight: vb.get(channels, "norm1.weight")?.to_dtype(DType::F32)?,
            norm1_bias: vb.get(channels, "norm1.bias")?.to_dtype(DType::F32)?,
            conv1: SparseConv3d::load(vb.pp("conv1"), channels, out_channels * 8, true)?,
            conv2: SparseConv3d::load(vb.pp("conv2"), out_channels, out_channels, true)?,
            to_subdiv,
            channels,
            out_channels,
        })
    }

    /// Shared block body given the raw `[N, 8]` subdivision logits (rows
    /// aligned with `coords`). Returns `(child coords, child feats
    /// [M, out_channels])`.
    fn forward_with_logits(
        &self,
        coords: &[[i32; 3]],
        feats: &Tensor,
        subdiv_logits: &Tensor,
    ) -> CandleResult<(Vec<[i32; 3]>, Tensor)> {
        let rows = feats.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "C2S block coords/feats mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        // h = silu(norm1(x)) -> conv1 (on the parent coords) -> C2S.
        let h = channel_layer_norm_affine(feats, &self.norm1_weight, &self.norm1_bias, 1e-6)?;
        let h = h.silu()?;
        let h = self.conv1.forward(coords, &h)?; // [N, out*8]
        let h_c2s = channel_to_spatial(coords, &h, subdiv_logits)?;
        // x = C2S(x, subdiv): the residual path through the same subdivision.
        let x_c2s = channel_to_spatial(coords, feats, subdiv_logits)?;
        // h = silu(norm2(h)) (non-affine) -> conv2.
        let h = layer_norm_no_affine(&h_c2s.feats, 1e-6)?.silu()?;
        let h = self.conv2.forward(&h_c2s.coords, &h)?; // [M, out]
        // skip = repeat_interleave(C2S(x), out // (channels // 8)).
        let repeats = self.out_channels / (self.channels / 8);
        let skip = repeat_channels(&x_c2s.feats, repeats)?;
        let out = h.add(&skip)?;
        Ok((h_c2s.coords, out))
    }

    /// pred_subdiv path: predict the logits from `feats`, then run the
    /// shared body. Returns the raw logits too.
    fn forward(
        &self,
        coords: &[[i32; 3]],
        feats: &Tensor,
    ) -> CandleResult<(Vec<[i32; 3]>, Tensor, Tensor)> {
        let subdiv_logits = match &self.to_subdiv {
            Some(to_subdiv) => to_subdiv.forward(feats)?,
            None => candle_core::bail!(
                "C2S block was loaded without to_subdiv (pred_subdiv = false); \
                 supply guide logits through forward_guided"
            ),
        };
        let (child_coords, out) = self.forward_with_logits(coords, feats, &subdiv_logits)?;
        Ok((child_coords, out, subdiv_logits))
    }

    /// Guided path (texture decoder): run the identical block body with
    /// externally supplied raw subdivision logits `[N, 8]` (the shape
    /// decoder's `to_subdiv` output at the same stage; `logits > 0`
    /// binarized inside C2S exactly like the reference `guide_subs`).
    fn forward_guided(
        &self,
        coords: &[[i32; 3]],
        feats: &Tensor,
        guide_logits: &Tensor,
    ) -> CandleResult<(Vec<[i32; 3]>, Tensor)> {
        self.forward_with_logits(coords, feats, guide_logits)
    }
}

#[derive(Debug)]
enum VaeStageBlock {
    ConvNeXt(ConvNeXtBlock),
    C2S(C2SResBlock),
}

/// One captured block output (parity-trace support).
#[derive(Debug, Clone)]
pub struct ShapeVaeBlockTrace {
    /// Stage index.
    pub stage: usize,
    /// Block index within the stage.
    pub block: usize,
    /// Output coordinates (parent coords for ConvNeXt, child coords for C2S).
    pub coords: Vec<[i32; 3]>,
    /// Output features `[N, C]`.
    pub feats: Tensor,
    /// Raw subdivision logits `[N, 8]` for C2S blocks.
    pub subdiv: Option<Tensor>,
}

/// Shape VAE decode output.
#[derive(Debug, Clone)]
pub struct ShapeVaeDecodeOutput {
    /// Final active voxel coordinates (children of the last C2S stage).
    pub coords: Vec<[i32; 3]>,
    /// Decoded 7-channel features `[N, out_channels]` (after
    /// `output_layer`; the dual-grid head splits these).
    pub feats: Tensor,
    /// Raw subdivision logits per C2S stage, in stage order (texture-decoder
    /// `guide_subs` for the texture decode).
    pub subdiv_logits: Vec<Tensor>,
    /// Per-block outputs when the decode was run with tracing enabled.
    pub trace: Vec<ShapeVaeBlockTrace>,
}

/// Sparse shape VAE decoder (`FlexiDualGridVaeDecoder`). Loads the 292
/// `shape_dec.` tensors from a VarBuilder rooted at the stage.
#[derive(Debug)]
pub struct Trellis2ShapeVaeDecoder {
    config: Trellis2ShapeVaeConfig,
    from_latent: Linear,
    output_layer: Linear,
    stages: Vec<Vec<VaeStageBlock>>,
}

impl Trellis2ShapeVaeDecoder {
    /// Load from a VarBuilder rooted at the decoder tensors (canonical
    /// package: `vb.pp("shape_dec")`). Fails if any expected tensor is
    /// missing.
    pub fn load(vb: VarBuilder, config: &Trellis2ShapeVaeConfig) -> CandleResult<Self> {
        if config.model_channels.len() != config.num_blocks.len() {
            candle_core::bail!(
                "VAE config model_channels ({}) and num_blocks ({}) lengths differ",
                config.model_channels.len(),
                config.num_blocks.len()
            );
        }
        let from_latent = linear(
            vb.pp("from_latent"),
            config.latent_channels,
            config.model_channels[0],
            true,
        )?;
        let output_layer = linear(
            vb.pp("output_layer"),
            *config.model_channels.last().expect("non-empty channels"),
            config.out_channels,
            true,
        )?;
        let mut stages = Vec::with_capacity(config.num_blocks.len());
        for (stage, &count) in config.num_blocks.iter().enumerate() {
            let channels = config.stage_in_channels(stage);
            let mut blocks = Vec::with_capacity(count + 1);
            for block in 0..count {
                blocks.push(VaeStageBlock::ConvNeXt(ConvNeXtBlock::load(
                    vb.pp("blocks").pp(stage.to_string()).pp(block.to_string()),
                    channels,
                    config.mlp_ratio,
                )?));
            }
            if config.stage_has_c2s(stage) {
                let out_channels = config.model_channels[stage + 1];
                blocks.push(VaeStageBlock::C2S(C2SResBlock::load(
                    vb.pp("blocks").pp(stage.to_string()).pp(count.to_string()),
                    channels,
                    out_channels,
                    config.pred_subdiv,
                )?));
            }
            super::drain_upload_staging(vb.device())?;
            stages.push(blocks);
        }
        Ok(Self {
            config: config.clone(),
            from_latent,
            output_layer,
            stages,
        })
    }

    pub fn config(&self) -> &Trellis2ShapeVaeConfig {
        &self.config
    }

    /// Decode latent features on the given coordinates.
    ///
    /// `coords`: `[N]` active voxel coords; `latent`: `[N, latent_channels]`
    /// F32 (the de-normalized shape SLAT). `collect_trace` captures every
    /// block output for parity tests (memory-heavy; off in production).
    pub fn decode(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        collect_trace: bool,
    ) -> CandleResult<ShapeVaeDecodeOutput> {
        self.decode_inner(coords, latent, None, collect_trace)
    }

    /// Guided decode (texture decoder, `pred_subdiv = false`): identical
    /// block walk, but each C2S stage consumes the raw subdivision logits
    /// captured from the shape decoder at the same stage (`guide_subs`,
    /// stage order, one `[N_parents_stage, 8]` F32 tensor each — rows
    /// aligned with the stage's parent coords). The reference passes them
    /// as `guide_subs` to `SparseUnetVaeDecoder.forward`
    /// (`sparse_unet_vae.py:495`).
    pub fn decode_with_guides(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        guide_subs: &[Tensor],
        collect_trace: bool,
    ) -> CandleResult<ShapeVaeDecodeOutput> {
        if guide_subs.len() != self.config.c2s_stage_count() {
            candle_core::bail!(
                "guide_subs must carry one logits tensor per C2S stage ({}), got {}",
                self.config.c2s_stage_count(),
                guide_subs.len()
            );
        }
        self.decode_inner(coords, latent, Some(guide_subs), collect_trace)
    }

    fn decode_inner(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        guide_subs: Option<&[Tensor]>,
        collect_trace: bool,
    ) -> CandleResult<ShapeVaeDecodeOutput> {
        let rows = latent.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "VAE decode coords/latent mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        let mut subdiv_logits = Vec::new();
        let mut guide_index = 0usize;
        let mut trace = Vec::new();
        let mut h = self.from_latent.forward(latent)?;
        let mut coords = coords.to_vec();
        for (stage, blocks) in self.stages.iter().enumerate() {
            for (block_index, block) in blocks.iter().enumerate() {
                let subdiv;
                match block {
                    VaeStageBlock::ConvNeXt(conv_next) => {
                        h = conv_next.forward(&coords, &h)?;
                        subdiv = None;
                    }
                    VaeStageBlock::C2S(c2s) => {
                        let guided = guide_subs.map(|guides| {
                            let logits = &guides[guide_index];
                            guide_index += 1;
                            logits.clone()
                        });
                        match guided {
                            Some(logits) => {
                                let (child_coords, child_feats) =
                                    c2s.forward_guided(&coords, &h, &logits)?;
                                subdiv_logits.push(logits.clone());
                                coords = child_coords;
                                h = child_feats;
                                subdiv = Some(logits);
                            }
                            None => {
                                let (child_coords, child_feats, sub) = c2s.forward(&coords, &h)?;
                                subdiv_logits.push(sub.clone());
                                coords = child_coords;
                                h = child_feats;
                                subdiv = Some(sub);
                            }
                        }
                    }
                }
                if collect_trace {
                    trace.push(ShapeVaeBlockTrace {
                        stage,
                        block: block_index,
                        coords: coords.clone(),
                        feats: h.clone(),
                        subdiv: subdiv.clone(),
                    });
                }
            }
        }
        // Final non-affine F.layer_norm (eps 1e-5) then output_layer.
        let normed = layer_norm_no_affine(&h, 1e-5)?;
        let feats = self.output_layer.forward(&normed)?;
        Ok(ShapeVaeDecodeOutput {
            coords,
            feats,
            subdiv_logits,
            trace,
        })
    }

    /// Reference `SparseUnetVaeDecoder.upsample(x, upsample_times)`
    /// (`sparse_unet_vae.py:509`): run `from_latent` and the stages
    /// `0..upsample_times` exactly like [`Self::decode`] (the same
    /// ConvNeXt/C2S block objects and forward calls), then return the
    /// coordinate rows the moment stage `upsample_times` is reached — its
    /// blocks (and the final norm/output layer) are never run.
    ///
    /// For the shape checkpoint (`upsample_times = 4`, 5 stages) this yields
    /// the children of the stage-3 C2S block: the active coords on the
    /// `resolution` grid (32^3 parents -> 512 grid after four doublings).
    /// The returned rows are grouped by parent in parent order, children
    /// ascending in `subidx = dx + 2*dy + 4*dz` — the C2S row order.
    ///
    /// `coords`: `[N]` parent coords; `latent`: `[N, latent_channels]` F32
    /// (the de-normalized shape SLAT, identical layout to [`Self::decode`]).
    /// The batch dim the reference keeps in `h.coords` is dropped here.
    pub fn upsample_coords(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        upsample_times: usize,
    ) -> CandleResult<Vec<[i32; 3]>> {
        let rows = latent.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "VAE upsample coords/latent mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        if upsample_times >= self.stages.len() {
            candle_core::bail!(
                "upsample_times {upsample_times} must be below the stage count {} \
                 (the reference returns `h.coords` when stage `upsample_times` is reached)",
                self.stages.len()
            );
        }
        let mut h = self.from_latent.forward(latent)?;
        let mut coords = coords.to_vec();
        for (stage, blocks) in self.stages.iter().enumerate() {
            if stage == upsample_times {
                return Ok(coords);
            }
            for block in blocks {
                match block {
                    VaeStageBlock::ConvNeXt(conv_next) => {
                        h = conv_next.forward(&coords, &h)?;
                    }
                    VaeStageBlock::C2S(c2s) => {
                        let (child_coords, child_feats, _subdiv) = c2s.forward(&coords, &h)?;
                        coords = child_coords;
                        h = child_feats;
                    }
                }
            }
        }
        unreachable!("upsample_times < stages.len() returns inside the loop");
    }
}

/// Numerically safe logistic sigmoid (F32).
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Split the decoded 7-channel features into the dual-grid head fields
/// (`FlexiDualGridVaeDecoder.forward` inference branch):
/// `dual = (1 + 2 * margin) * sigmoid(feats[0:3]) - margin`,
/// `intersected = feats[3:6] > 0`, `split_weight = softplus(feats[6])`.
///
/// Computed on the host in f32 (the arrays feed the mesh extractor, which is
/// host-side; `feats` may live on any device).
pub fn dual_grid_fields(feats: &Tensor, voxel_margin: f32) -> CandleResult<DualGridFields> {
    let (rows, channels) = feats.dims2()?;
    if channels != 7 {
        candle_core::bail!("dual-grid head expects 7 decoded channels, got {channels}");
    }
    let values = feats
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let scale = 1.0 + 2.0 * voxel_margin;
    let mut dual = Vec::with_capacity(rows);
    let mut intersected = Vec::with_capacity(rows);
    let mut split_weight = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * 7;
        dual.push([
            scale * sigmoid(values[base]) - voxel_margin,
            scale * sigmoid(values[base + 1]) - voxel_margin,
            scale * sigmoid(values[base + 2]) - voxel_margin,
        ]);
        intersected.push([
            values[base + 3] > 0.0,
            values[base + 4] > 0.0,
            values[base + 5] > 0.0,
        ]);
        split_weight.push(super::trellis2_mesh::softplus_stable(values[base + 6]));
    }
    Ok(DualGridFields {
        dual,
        intersected,
        split_weight,
    })
}

/// One-shot shape decode -> mesh: decode the latent, split the head fields
/// and run the flexible dual-grid extraction.
pub fn decode_shape_mesh(
    decoder: &Trellis2ShapeVaeDecoder,
    coords: &[[i32; 3]],
    latent: &Tensor,
    resolution: usize,
    voxel_margin: f32,
) -> CandleResult<(ShapeVaeDecodeOutput, DualGridMesh)> {
    let decoded = decoder.decode(coords, latent, false)?;
    let fields = dual_grid_fields(&decoded.feats, voxel_margin)?;
    let mesh = flexible_dual_grid_to_mesh(
        &decoded.coords,
        &fields.dual,
        &fields.intersected,
        &fields.split_weight,
        [resolution, resolution, resolution],
        [[-0.5_f32; 3], [0.5_f32; 3]],
    );
    Ok((decoded, mesh))
}
