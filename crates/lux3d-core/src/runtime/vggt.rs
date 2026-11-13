use std::{path::Path, sync::Arc};

use super::nn_blocks::Linear;
use crate::weights::VarBuilder;
use candle_core::{D, DType, Device, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, layer_norm};

use super::{
    attention_math::{
        Rope2d, RopeEmbeddings, apply_layernorm_rope, apply_rope, exact_query_chunked_sdpa,
        position_getter,
    },
    nn_blocks::{LayerScale, Mlp, linear, linear_fwd},
    vggt_dpt::{VggtDptActivation, VggtDptHead},
};
use crate::{
    CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result,
    error::Lux3dError,
    export::VggtExportStage,
    geometry::VggtGeometryStage,
    load_canonical_weights,
    neural::VggtNeuralStage,
    preprocess::VggtPreprocessStage,
    runtime::{DeviceLocalCache, pi3_encoder::Pi3DinoEncoder, pi3_encoder::load_pi3_encoder},
};

pub use crate::preprocess::VggtPreparedInputs;

const VGGT_EMBED_DIM: usize = 1024;
const VGGT_CAMERA_DIM: usize = 2048;
const VGGT_NUM_HEADS: usize = 16;
const VGGT_DEPTH: usize = 24;
const VGGT_PATCH_START_IDX: usize = 5;
const VGGT_HEAD_DIM: usize = VGGT_EMBED_DIM / VGGT_NUM_HEADS;
/// Aggregator/camera block LayerNorm eps (VGGT blocks use 1e-5, unlike the
/// DINOv2 backbone's 1e-6).
const BLOCK_LAYER_NORM_EPS: f64 = 1e-5;
const CAMERA_LAYER_NORM_EPS: f64 = 1e-5;
const ADALN_EPS: f64 = 1e-6;
const CACHED_LAYER_INDICES: [usize; 4] = [4, 11, 17, 23];

#[derive(Debug, Clone)]
pub struct VggtInferenceOutput {
    pub world_points: Tensor,
    pub world_points_conf: Tensor,
    pub depth: Tensor,
    pub depth_conf: Tensor,
    pub pose_enc: Tensor,
    pub extrinsics: Tensor,
    pub intrinsics: Tensor,
    pub rgb_frames: Tensor,
    pub export_mask: Tensor,
}

#[derive(Debug, Clone)]
pub struct VggtDebugTaps {
    pub patch_tokens: Tensor,
    pub aggregator_layer4: Tensor,
    pub aggregator_layer11: Tensor,
    pub aggregator_layer17: Tensor,
    pub aggregator_layer23: Tensor,
    pub pose_enc_iterations: Vec<Tensor>,
}

#[derive(Debug, Clone)]
pub struct VggtAggregatorOutput {
    pub layers: Vec<Option<Tensor>>,
    pub final_tokens: Tensor,
    pub patch_start_idx: usize,
}

#[derive(Debug, Clone)]
pub struct VggtPipeline {
    pub weights: CanonicalWeightSetPaths,
    pub preprocess: VggtPreprocessStage,
    pub neural: VggtNeuralStage,
    pub geometry: VggtGeometryStage,
    pub export: VggtExportStage,
    /// Weight/activation dtype for the neural network. Public API outputs are
    /// always converted back to F32; F16 is an inference-time storage/compute
    /// mode.
    dtype: DType,
    bundle_cache: Arc<DeviceLocalCache<VggtModelBundle>>,
}

/// Alternating-attention transformer block: `x += ls1(attn(norm1(x)))`,
/// `x += ls2(mlp(norm2(x)))` with LayerNorm eps 1e-5, GELU-erf MLP.
///
/// Used both by the aggregator (dim 1024, QK-norm + 2D RoPE, MLP hidden 4096)
/// and by the camera-head trunk (dim 2048, no QK-norm, no RoPE, hidden 8192).
#[derive(Debug)]
struct VggtBlock {
    norm1: LayerNorm,
    attn: VggtAttention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl VggtBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, qk_norm: bool) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, BLOCK_LAYER_NORM_EPS, vb.pp("norm1"))?,
            attn: VggtAttention::load(vb.pp("attn"), dim, num_heads, qk_norm)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, BLOCK_LAYER_NORM_EPS, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&RopeEmbeddings>) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?, rope)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct VggtAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: Option<LayerNorm>,
    k_norm: Option<LayerNorm>,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VggtAttention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, qk_norm: bool) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(layer_norm(head_dim, BLOCK_LAYER_NORM_EPS, vb.pp("q_norm"))?),
                Some(layer_norm(head_dim, BLOCK_LAYER_NORM_EPS, vb.pp("k_norm"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            q_norm,
            k_norm,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, xs: &Tensor, rope: Option<&RopeEmbeddings>) -> CandleResult<Tensor> {
        let (batch, tokens, channels) = xs.dims3()?;
        // [B, N, 3, H, D] -> [3, B, H, N, D]
        let qkv = linear_fwd(&self.qkv, xs)?
            .reshape((batch, tokens, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?.contiguous()?;

        let q = match (&self.q_norm, rope) {
            (Some(norm), Some(embeddings)) => apply_layernorm_rope(&q, norm, embeddings)?,
            (Some(norm), None) => norm.forward(&q)?.contiguous()?,
            (None, Some(embeddings)) => apply_rope(&q, embeddings)?,
            (None, None) => q.contiguous()?,
        };
        let k = match (&self.k_norm, rope) {
            (Some(norm), Some(embeddings)) => apply_layernorm_rope(&k, norm, embeddings)?,
            (Some(norm), None) => norm.forward(&k)?.contiguous()?,
            (None, Some(embeddings)) => apply_rope(&k, embeddings)?,
            (None, None) => k.contiguous()?,
        };

        let out = exact_query_chunked_sdpa(&q, &k, &v, self.scale as f32, usize::MAX)?;
        let out = out.transpose(1, 2)?.reshape((batch, tokens, channels))?;
        linear_fwd(&self.proj, &out)
    }
}

/// Aggregator: DINOv2 ViT-L/14 patch backbone (reused from Pi3) plus 24
/// alternating frame/global blocks and the four cached layer outputs.
#[derive(Debug)]
struct VggtAggregator {
    encoder: Pi3DinoEncoder,
    camera_token: Tensor,
    register_token: Tensor,
    frame_blocks: Vec<VggtBlock>,
    global_blocks: Vec<VggtBlock>,
}

impl VggtAggregator {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let encoder = load_pi3_encoder(vb.pp("patch_embed"))?;
        let camera_token = vb.get((1, 2, 1, VGGT_EMBED_DIM), "camera_token")?;
        let register_token = vb.get((1, 2, 4, VGGT_EMBED_DIM), "register_token")?;
        let frame_blocks = (0..VGGT_DEPTH)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("frame_blocks").pp(idx.to_string()),
                    VGGT_EMBED_DIM,
                    VGGT_NUM_HEADS,
                    true,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let global_blocks = (0..VGGT_DEPTH)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("global_blocks").pp(idx.to_string()),
                    VGGT_EMBED_DIM,
                    VGGT_NUM_HEADS,
                    true,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            encoder,
            camera_token,
            register_token,
            frame_blocks,
            global_blocks,
        })
    }

    /// `[B, S, 3, H, W]` normalized frames -> post-LayerNorm patch tokens
    /// `[B*S, H*W/patch^2, 1024]`.
    fn encode_patch_tokens(&self, normalized_frames: &Tensor) -> CandleResult<Tensor> {
        let (batch, frames, channels, height, width) = normalized_frames.dims5()?;
        let inputs = normalized_frames.reshape((batch * frames, channels, height, width))?;
        self.encoder.forward_patch_tokens(&inputs)
    }

    /// Expand `[1, 2, X, C]` special tokens to `[B*S, X, C]`: slot 0 for the
    /// first frame, slot 1 for every other frame (`slice_expand_and_flatten`).
    fn expand_special_tokens(token: &Tensor, batch: usize, frames: usize) -> CandleResult<Tensor> {
        let (_, slots, x, channels) = token.dims4()?;
        debug_assert_eq!(slots, 2);
        let mut indices = Vec::with_capacity(frames);
        indices.push(0u32);
        for _ in 1..frames {
            indices.push(1);
        }
        let indices = Tensor::from_vec(indices, (frames,), token.device())?;
        let selected = token.index_select(&indices, 1)?; // [1, S, X, C]
        selected
            .broadcast_as((batch, frames, x, channels))?
            .reshape((batch * frames, x, channels))?
            .contiguous()
    }

    fn forward(
        &self,
        patch_tokens: &Tensor,
        batch: usize,
        frames: usize,
        height: usize,
        width: usize,
    ) -> CandleResult<Vec<Option<Tensor>>> {
        let device = patch_tokens.device();
        let patch_h = height / 14;
        let patch_w = width / 14;
        let patch_count = patch_h * patch_w;
        if patch_tokens.dim(1)? != patch_count {
            candle_core::bail!(
                "VGGT patch token count {} does not match {patch_h}x{patch_w}",
                patch_tokens.dim(1)?
            );
        }

        let camera = Self::expand_special_tokens(&self.camera_token, batch, frames)?;
        let register = Self::expand_special_tokens(&self.register_token, batch, frames)?;
        let mut tokens = Tensor::cat(&[&camera, &register, patch_tokens], 1)?;
        let (bs, token_count, channels) = tokens.dims3()?;

        // Positions: 0 for the special tokens, (y + 1, x + 1) for patches.
        let positions = position_getter(bs, patch_h, patch_w, device)?;
        let special = Tensor::zeros((bs, VGGT_PATCH_START_IDX, 2), DType::I64, device)?;
        let positions = Tensor::cat(&[&special, &positions], 1)?;

        let rope = Rope2d::new(100.0);
        let frame_embeddings = rope.embeddings(&positions, VGGT_HEAD_DIM)?;
        let global_positions = positions.reshape((batch, frames * token_count, 2))?;
        let global_embeddings = rope.embeddings(&global_positions, VGGT_HEAD_DIM)?;

        let mut layers: Vec<Option<Tensor>> = (0..VGGT_DEPTH).map(|_| None).collect();
        for round in 0..VGGT_DEPTH {
            let frame_tokens = self.frame_blocks[round]
                .forward(&tokens, Some(&frame_embeddings))?
                .reshape((batch, frames, token_count, channels))?;
            let global_input = frame_tokens.reshape((batch, frames * token_count, channels))?;
            let global_tokens = self.global_blocks[round]
                .forward(&global_input, Some(&global_embeddings))?
                .reshape((batch, frames, token_count, channels))?;
            if CACHED_LAYER_INDICES.contains(&round) {
                layers[round] = Some(Tensor::cat(&[&frame_tokens, &global_tokens], D::Minus1)?);
            }
            tokens = global_tokens.reshape((bs, token_count, channels))?;
        }
        Ok(layers)
    }
}

#[derive(Debug)]
struct VggtCameraHead {
    token_norm: LayerNorm,
    empty_pose_tokens: Tensor,
    embed_pose: Linear,
    pose_ln_modulation: Linear,
    trunk: Vec<VggtBlock>,
    trunk_norm: LayerNorm,
    pose_branch_fc1: Linear,
    pose_branch_fc2: Linear,
}

impl VggtCameraHead {
    const TARGET_DIM: usize = 9;
    const ITERATIONS: usize = 4;

    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let trunk = (0..4)
            .map(|idx| {
                let block = VggtBlock::load(
                    vb.pp("trunk").pp(idx.to_string()),
                    VGGT_CAMERA_DIM,
                    VGGT_NUM_HEADS,
                    false,
                )?;
                super::drain_upload_staging(vb.device())?;
                Ok(block)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            token_norm: layer_norm(VGGT_CAMERA_DIM, CAMERA_LAYER_NORM_EPS, vb.pp("token_norm"))?,
            empty_pose_tokens: vb.get((1, 1, Self::TARGET_DIM), "empty_pose_tokens")?,
            embed_pose: linear(vb.pp("embed_pose"), Self::TARGET_DIM, VGGT_CAMERA_DIM, true)?,
            pose_ln_modulation: linear(
                vb.pp("poseLN_modulation").pp("1"),
                VGGT_CAMERA_DIM,
                VGGT_CAMERA_DIM * 3,
                true,
            )?,
            trunk,
            trunk_norm: layer_norm(VGGT_CAMERA_DIM, CAMERA_LAYER_NORM_EPS, vb.pp("trunk_norm"))?,
            pose_branch_fc1: linear(
                vb.pp("pose_branch").pp("fc1"),
                VGGT_CAMERA_DIM,
                VGGT_CAMERA_DIM / 2,
                true,
            )?,
            pose_branch_fc2: linear(
                vb.pp("pose_branch").pp("fc2"),
                VGGT_CAMERA_DIM / 2,
                Self::TARGET_DIM,
                true,
            )?,
        })
    }

    /// `[B, S, 1374, 2048]` aggregated tokens -> 4 activated `[B, S, 9]`
    /// pose encodings (one per refinement iteration).
    fn forward(&self, tokens: &Tensor) -> CandleResult<Vec<Tensor>> {
        let pose_tokens = tokens.i((.., .., 0))?;
        let pose_tokens = self.token_norm.forward(&pose_tokens)?;
        let (batch, frames, _channels) = pose_tokens.dims3()?;

        let mut pred_pose_enc: Option<Tensor> = None;
        let mut activated_poses = Vec::with_capacity(Self::ITERATIONS);
        for _ in 0..Self::ITERATIONS {
            let module_input = match &pred_pose_enc {
                None => {
                    let empty = self
                        .empty_pose_tokens
                        .broadcast_as((batch, frames, Self::TARGET_DIM))?
                        .contiguous()?;
                    linear_fwd(&self.embed_pose, &empty)?
                }
                Some(pred) => linear_fwd(&self.embed_pose, pred)?,
            };
            // poseLN_modulation = Sequential(SiLU, Linear).
            let modulation = linear_fwd(&self.pose_ln_modulation, &module_input.silu()?)?;
            let modulation = modulation.chunk(3, D::Minus1)?;
            let shift = &modulation[0];
            let scale = &modulation[1];
            let gate = &modulation[2];

            let normed = adaln_norm(&pose_tokens, ADALN_EPS)?;
            let modulated = gate.broadcast_mul(
                &normed
                    .broadcast_mul(&scale.affine(1.0, 1.0)?)?
                    .broadcast_add(shift)?,
            )?;
            let mut hidden = (&modulated + &pose_tokens)?;
            for block in &self.trunk {
                hidden = block.forward(&hidden, None)?;
            }
            let hidden = self.trunk_norm.forward(&hidden)?;
            let delta = linear_fwd(
                &self.pose_branch_fc2,
                &linear_fwd(&self.pose_branch_fc1, &hidden)?.gelu_erf()?,
            )?;

            let pred = match &pred_pose_enc {
                None => delta,
                Some(previous) => (previous + &delta)?,
            };
            // Only the field of view is ReLU-activated; T and quat stay linear.
            let translation = pred.narrow(D::Minus1, 0, 3)?;
            let quat = pred.narrow(D::Minus1, 3, 4)?;
            let fov = pred.narrow(D::Minus1, 7, 2)?.relu()?;
            let activated = Tensor::cat(&[&translation, &quat, &fov], D::Minus1)?;
            activated_poses.push(activated);
            pred_pose_enc = Some(pred);
        }
        Ok(activated_poses)
    }
}

/// LayerNorm without learnable affine (`elementwise_affine=False`).
fn adaln_norm(xs: &Tensor, eps: f64) -> CandleResult<Tensor> {
    let mean = xs.mean_keepdim(D::Minus1)?;
    let centered = xs.broadcast_sub(&mean)?;
    let variance = centered.sqr()?.mean_keepdim(D::Minus1)?;
    centered.broadcast_div(&variance.affine(1.0, eps)?.sqrt()?)
}

/// `pose_encoding_to_extri_intri` for `absT_quaR_FoV`, returning
/// `(extrinsics [B, S, 3, 4], intrinsics [B, S, 3, 3])`.
fn decode_pose(
    pose_enc: &Tensor,
    image_height: usize,
    image_width: usize,
) -> CandleResult<(Tensor, Tensor)> {
    let translation = pose_enc.narrow(D::Minus1, 0, 3)?;
    let quat = pose_enc.narrow(D::Minus1, 3, 4)?;
    let fov_h = pose_enc.i((.., .., 7))?;
    let fov_w = pose_enc.i((.., .., 8))?;

    let rotation = quat_to_mat(&quat)?;
    let extrinsics = Tensor::cat(&[&rotation, &translation.unsqueeze(3)?], 3)?;

    let half_height = image_height as f64 / 2.0;
    let half_width = image_width as f64 / 2.0;
    // fy = (H / 2) / tan(fov_h / 2); recip() is 1/x, so multiply after.
    let fy = tan_tensor(&fov_h.affine(0.5, 0.0)?)?
        .recip()?
        .affine(half_height, 0.0)?;
    let fx = tan_tensor(&fov_w.affine(0.5, 0.0)?)?
        .recip()?
        .affine(half_width, 0.0)?;
    let zero = fx.zeros_like()?;
    let zero_w = zero.affine(1.0, half_width)?;
    let zero_h = zero.affine(1.0, half_height)?;
    let one = zero.affine(1.0, 1.0)?;
    let row0 = Tensor::stack(&[&fx, &zero, &zero_w], 2)?;
    let row1 = Tensor::stack(&[&zero, &fy, &zero_h], 2)?;
    let row2 = Tensor::stack(&[&zero, &zero, &one], 2)?;
    let intrinsics = Tensor::stack(&[&row0, &row1, &row2], 2)?;
    Ok((extrinsics, intrinsics))
}

fn tan_tensor(xs: &Tensor) -> CandleResult<Tensor> {
    xs.sin()?.broadcast_div(&xs.cos()?)
}

/// Exact `vggt.utils.rotation.quat_to_mat` (quaternion order XYZW, scalar last).
fn quat_to_mat(quaternions: &Tensor) -> CandleResult<Tensor> {
    let i = quaternions.i((.., .., 0))?;
    let j = quaternions.i((.., .., 1))?;
    let k = quaternions.i((.., .., 2))?;
    let r = quaternions.i((.., .., 3))?;
    // two_s = 2 / sum(q^2)
    let two_s = quaternions
        .sqr()?
        .sum(D::Minus1)?
        .recip()?
        .affine(2.0, 0.0)?;

    let o0 = two_s
        .broadcast_mul(&(&j.sqr()? + &k.sqr()?)?)?
        .affine(-1.0, 1.0)?;
    let o1 = two_s.broadcast_mul(&(&(&i * &j)? - &(&k * &r)?)?)?;
    let o2 = two_s.broadcast_mul(&(&(&i * &k)? + &(&j * &r)?)?)?;
    let o3 = two_s.broadcast_mul(&(&(&i * &j)? + &(&k * &r)?)?)?;
    let o4 = two_s
        .broadcast_mul(&(&i.sqr()? + &k.sqr()?)?)?
        .affine(-1.0, 1.0)?;
    let o5 = two_s.broadcast_mul(&(&(&j * &k)? - &(&i * &r)?)?)?;
    let o6 = two_s.broadcast_mul(&(&(&i * &k)? - &(&j * &r)?)?)?;
    let o7 = two_s.broadcast_mul(&(&(&j * &k)? + &(&i * &r)?)?)?;
    let o8 = two_s
        .broadcast_mul(&(&i.sqr()? + &j.sqr()?)?)?
        .affine(-1.0, 1.0)?;

    let (batch, frames, _) = quaternions.dims3()?;
    Tensor::stack(&[&o0, &o1, &o2, &o3, &o4, &o5, &o6, &o7, &o8], D::Minus1)?
        .reshape((batch, frames, 3, 3))
}

#[derive(Debug)]
struct VggtModelBundle {
    aggregator: VggtAggregator,
    camera_head: VggtCameraHead,
    point_head: VggtDptHead,
    depth_head: VggtDptHead,
}

impl VggtModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device, dtype: DType) -> Result<Self> {
        let vb = unsafe {
            weights.var_builder(dtype, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open VGGT canonical weights: {source}"),
                }
            })?
        };
        let aggregator = VggtAggregator::load(vb.pp("aggregator")).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load VGGT aggregator: {source}"),
            }
        })?;
        let camera_head = VggtCameraHead::load(vb.pp("camera_head")).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load VGGT camera head: {source}"),
            }
        })?;
        let point_head = VggtDptHead::load(
            vb.pp("point_head"),
            VGGT_CAMERA_DIM,
            4,
            VggtDptActivation::InvLog,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to load VGGT point head: {source}"),
        })?;
        let depth_head = VggtDptHead::load(
            vb.pp("depth_head"),
            VGGT_CAMERA_DIM,
            2,
            VggtDptActivation::Exp,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to load VGGT depth head: {source}"),
        })?;
        Ok(Self {
            aggregator,
            camera_head,
            point_head,
            depth_head,
        })
    }
}

impl VggtPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Self::load_with_dtype(model_assets, DType::F32)
    }

    /// Load VGGT with an explicit neural-network dtype. `F32` is the default
    /// reference mode; `F16` halves weight memory and runs the aggregator,
    /// camera head and DPT heads in half precision on CUDA/Vulkan/WGPU.
    pub fn load_with_dtype(model_assets: ModelAssetOptions, dtype: DType) -> Result<Self> {
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::Vggt, model_assets)?,
            preprocess: VggtPreprocessStage::default(),
            neural: VggtNeuralStage,
            geometry: Default::default(),
            export: Default::default(),
            dtype,
            bundle_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<VggtModelBundle>> {
        self.bundle_cache.get_or_try_init(device, || {
            VggtModelBundle::load(&self.weights, device, self.dtype)
        })
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        self.preprocess
            .prepare_inputs_from_path(source, interval, device)
    }

    pub fn infer_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtInferenceOutput> {
        let prepared = self.prepare_inputs_from_path(source, interval, device)?;
        self.infer_tensors(&prepared)
    }

    pub fn infer_tensors(&self, prepared: &VggtPreparedInputs) -> Result<VggtInferenceOutput> {
        self.run_forward(prepared, false).map(|(output, _)| output)
    }

    /// Export `world_points_conf > 1.0` (the VGGT demo threshold) as ASCII PLY.
    pub fn export_ply(&self, output: &VggtInferenceOutput, path: &Path) -> Result<()> {
        self.export_ply_with_threshold(output, path, 1.0)
    }

    /// Export `world_points_conf > conf_threshold` (strict) as ASCII PLY.
    pub fn export_ply_with_threshold(
        &self,
        output: &VggtInferenceOutput,
        path: &Path,
        conf_threshold: f32,
    ) -> Result<()> {
        let point_cloud = self.geometry.assemble_cpu(output, conf_threshold)?;
        self.export.write_ply(&point_cloud, path)
    }

    /// Dump the raw inference tensors as `.npz` so a quantized run can be
    /// compared against the float run (Chamfer / PSNR / camera-pose AUC).
    pub fn export_inference_npz(&self, output: &VggtInferenceOutput, path: &Path) -> Result<()> {
        use ndarray::{ArrayD, IxDyn};

        let file = std::fs::File::create(path).map_err(|source| Lux3dError::ModelAssetIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut npz = ndarray_npy::NpzWriter::new(file);
        let fields: [(&str, &Tensor); 7] = [
            ("world_points", &output.world_points),
            ("world_points_conf", &output.world_points_conf),
            ("depth", &output.depth),
            ("depth_conf", &output.depth_conf),
            ("extrinsics", &output.extrinsics),
            ("intrinsics", &output.intrinsics),
            ("pose_enc", &output.pose_enc),
        ];
        for (name, tensor) in fields {
            let tensor = tensor
                .to_device(&Device::Cpu)
                .and_then(|tensor| tensor.to_dtype(DType::F32))
                .and_then(|tensor| tensor.contiguous())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to stage `{name}` for npz: {source}"),
                })?;
            let shape = tensor.dims().to_vec();
            let data = tensor
                .flatten_all()
                .and_then(|tensor| tensor.to_vec1::<f32>())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to read `{name}` for npz: {source}"),
                })?;
            let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|_| Lux3dError::InvalidInput("failed to build npz tensor"))?;
            npz.add_array(name, &array)
                .map_err(|_| Lux3dError::InvalidInput("failed to write npz tensor"))?;
        }
        npz.finish()
            .map_err(|_| Lux3dError::InvalidInput("failed to finish npz"))?;
        Ok(())
    }

    pub fn debug_forward(
        &self,
        prepared: &VggtPreparedInputs,
    ) -> Result<(VggtInferenceOutput, VggtDebugTaps)> {
        let (output, taps) = self.run_forward(prepared, true)?;
        let taps = taps.ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: "VGGT debug forward did not capture taps".to_string(),
        })?;
        Ok((output, taps))
    }

    pub fn encode_patch_tokens(&self, normalized_frames: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(normalized_frames.device())?;
        bundle
            .aggregator
            .encode_patch_tokens(normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })
    }

    pub fn run_aggregator(&self, prepared: &VggtPreparedInputs) -> Result<VggtAggregatorOutput> {
        let normalized_frames =
            prepared
                .normalized_frames
                .to_dtype(self.dtype)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to convert VGGT input to {dtype:?}: {source}",
                        dtype = self.dtype
                    ),
                })?;
        let device = normalized_frames.device();
        let bundle = self.bundle_for(device)?;
        let (batch, frames, _channels, height, width) = input_dims(&normalized_frames)?;
        let patch_tokens = bundle
            .aggregator
            .encode_patch_tokens(&normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })?;
        let layers = bundle
            .aggregator
            .forward(&patch_tokens, batch, frames, height, width)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT aggregator forward failed: {source}"),
            })?;
        let final_tokens = layers[CACHED_LAYER_INDICES[3]].clone().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT aggregator did not produce layer 23".to_string(),
            }
        })?;
        Ok(VggtAggregatorOutput {
            layers,
            final_tokens,
            patch_start_idx: VGGT_PATCH_START_IDX,
        })
    }

    fn run_forward(
        &self,
        prepared: &VggtPreparedInputs,
        capture_taps: bool,
    ) -> Result<(VggtInferenceOutput, Option<VggtDebugTaps>)> {
        let normalized_frames =
            prepared
                .normalized_frames
                .to_dtype(self.dtype)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to convert VGGT input to {dtype:?}: {source}",
                        dtype = self.dtype
                    ),
                })?;
        let device = normalized_frames.device();
        let bundle = self.bundle_for(device)?;
        let (batch, frames, _channels, height, width) = input_dims(&normalized_frames)?;
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let stage = |name: &str, t: &std::time::Instant| {
            if stage_time {
                eprintln!("[stage]   {name}: {:.2}s", t.elapsed().as_secs_f64());
            }
        };

        let t = std::time::Instant::now();
        let patch_tokens = bundle
            .aggregator
            .encode_patch_tokens(&normalized_frames)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT encoder forward failed: {source}"),
            })?;
        stage("encode_patch_tokens", &t);

        let t = std::time::Instant::now();
        let layers = bundle
            .aggregator
            .forward(&patch_tokens, batch, frames, height, width)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT aggregator forward failed: {source}"),
            })?;
        stage("aggregator", &t);

        let t = std::time::Instant::now();
        let layer23 = layers[CACHED_LAYER_INDICES[3]].as_ref().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT aggregator did not produce layer 23".to_string(),
            }
        })?;
        let pose_enc_iterations = bundle.camera_head.forward(layer23).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT camera head forward failed: {source}"),
            }
        })?;
        let pose_enc = pose_enc_iterations.last().cloned().ok_or_else(|| {
            Lux3dError::CanonicalWeightsValidation {
                message: "VGGT camera head produced no pose encodings".to_string(),
            }
        })?;
        stage("camera_head", &t);

        let t = std::time::Instant::now();
        let (extrinsics, intrinsics) = decode_pose(&pose_enc, height, width).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT pose decode failed: {source}"),
            }
        })?;
        stage("pose_decode", &t);

        let chunk_size = dpt_chunk_size();
        let t = std::time::Instant::now();
        let (depth, depth_conf) = bundle
            .depth_head
            .forward(
                &layers,
                VGGT_PATCH_START_IDX,
                batch,
                frames,
                chunk_size,
                height,
                width,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT depth head forward failed: {source}"),
            })?;
        let depth = depth
            .reshape((batch, frames, height, width, 1))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT depth: {source}"),
            })?;
        let depth_conf = depth_conf
            .reshape((batch, frames, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT depth_conf: {source}"),
            })?;
        stage("depth_head", &t);

        let t = std::time::Instant::now();
        let (world_points, world_points_conf) = bundle
            .point_head
            .forward(
                &layers,
                VGGT_PATCH_START_IDX,
                batch,
                frames,
                chunk_size,
                height,
                width,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT point head forward failed: {source}"),
            })?;
        let world_points = world_points
            .reshape((batch, frames, height, width, 3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT world_points: {source}"),
            })?;
        let world_points_conf = world_points_conf
            .reshape((batch, frames, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape VGGT world_points_conf: {source}"),
            })?;
        stage("point_head", &t);

        let t = std::time::Instant::now();
        let export_mask = world_points_conf
            .gt(1.0f32)
            .and_then(|mask| mask.to_dtype(DType::U8))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("VGGT export mask computation failed: {source}"),
            })?;
        stage("postprocess", &t);

        let taps = if capture_taps {
            Some(VggtDebugTaps {
                patch_tokens: patch_tokens.clone(),
                aggregator_layer4: cached_layer(&layers, CACHED_LAYER_INDICES[0])?,
                aggregator_layer11: cached_layer(&layers, CACHED_LAYER_INDICES[1])?,
                aggregator_layer17: cached_layer(&layers, CACHED_LAYER_INDICES[2])?,
                aggregator_layer23: cached_layer(&layers, CACHED_LAYER_INDICES[3])?,
                pose_enc_iterations,
            })
        } else {
            None
        };

        // Public tensors stay F32 regardless of the internal compute dtype so
        // geometry/export code and callers keep the historical contract.
        let to_f32 = |tensor: &Tensor| -> Result<Tensor> {
            tensor
                .to_dtype(DType::F32)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to convert VGGT output to f32: {source}"),
                })
        };
        Ok((
            VggtInferenceOutput {
                world_points: to_f32(&world_points)?,
                world_points_conf: to_f32(&world_points_conf)?,
                depth: to_f32(&depth)?,
                depth_conf: to_f32(&depth_conf)?,
                pose_enc: to_f32(&pose_enc)?,
                extrinsics: to_f32(&extrinsics)?,
                intrinsics: to_f32(&intrinsics)?,
                rgb_frames: prepared.rgb_frames.clone(),
                export_mask,
            },
            taps,
        ))
    }
}

fn cached_layer(layers: &[Option<Tensor>], idx: usize) -> Result<Tensor> {
    layers[idx]
        .clone()
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT aggregator did not produce layer {idx}"),
        })
}

fn input_dims(normalized_frames: &Tensor) -> Result<(usize, usize, usize, usize, usize)> {
    normalized_frames
        .dims5()
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("VGGT normalized frames must have shape [B,S,3,H,W]: {source}"),
        })
}

/// Reference `DPTHead(frames_chunk_size=8)` processes frames in groups of 8;
/// frames are independent, so any chunk size is numerically identical. The
/// default is 1 to bound peak VRAM on 12 GiB cards (a 518x518 256-channel
/// activation is ~275 MB per frame).
fn dpt_chunk_size() -> usize {
    std::env::var("LUX3D_VGGT_DPT_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}
