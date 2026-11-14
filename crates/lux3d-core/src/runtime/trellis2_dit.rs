//! TRELLIS.2 dense sparse-structure flow DiT
//! (`SparseStructureFlowModel`, checkpoint
//! `ckpts/ss_flow_img_dit_1_3B_64_bf16`, 640 tensors).
//!
//! Reference: `trellis2/models/sparse_structure_flow.py`.
//!
//! Forward: `x [B, 8, 16, 16, 16]` -> tokens `[B, 4096, 8]` (token order
//! row-major `x*256 + y*16 + z`) -> `input_layer` Linear(8, 1536) -> 30
//! `ModulatedTransformerCrossBlock`s conditioned on
//! `t_emb = adaLN(t_embedder(1000 * t))` and the DINOv3 cond
//! `[B, Lkv, 1024]` (dense cross context) -> final non-affine
//! `F.layer_norm` (eps 1e-5) -> `out_layer` Linear(1536, 8) -> reshape to
//! `[B, 8, 16, 16, 16]`.
//!
//! RoPE phases are a registered buffer in the reference (not saved in the
//! safetensors file) and are recomputed here from the 16^3 meshgrid
//! coordinates. F32 compute throughout (locked decision 3); the checkpoint
//! is bf16 and the VarBuilder materializes F32 on access.

use candle_core::{Result as CandleResult, Tensor};

use super::nn_blocks::{Linear, linear};
use super::trellis2_blocks::{
    Trellis2AdaLn, Trellis2DenseBlock, Trellis2Rope3d, Trellis2TimestepEmbedder,
    cubic_token_coords, layer_norm_no_affine,
};
use super::trellis2_sparse::SparseFeats;
use crate::weights::VarBuilder;

/// Sparse-structure flow DiT architecture parameters (checkpoint
/// `ss_flow_img_dit_1_3B_64_bf16.json` args).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2SsConfig {
    /// Latent grid resolution per axis (16; token count 4096).
    pub resolution: usize,
    /// Input latent channels (8).
    pub in_channels: usize,
    /// Output channels (8).
    pub out_channels: usize,
    /// Model width (1536).
    pub model_channels: usize,
    /// Cross-attention context channels (1024, DINOv3).
    pub cond_channels: usize,
    /// Transformer block count (30).
    pub num_blocks: usize,
    /// Attention heads (12; head dim 128).
    pub num_heads: usize,
    /// FFN ratio (5.3334 -> hidden 8192).
    pub mlp_ratio: f64,
    /// Global adaLN (true for every TRELLIS.2 flow checkpoint).
    pub share_mod: bool,
    /// q/k RMS norm in self attention (true).
    pub qk_rms_norm: bool,
    /// q/k RMS norm in cross attention (true).
    pub qk_rms_norm_cross: bool,
}

impl Default for Trellis2SsConfig {
    fn default() -> Self {
        Self::trellis_2_4b()
    }
}

impl Trellis2SsConfig {
    /// `ss_flow_img_dit_1_3B_64_bf16` checkpoint args.
    pub const fn trellis_2_4b() -> Self {
        Self {
            resolution: 16,
            in_channels: 8,
            out_channels: 8,
            model_channels: 1536,
            cond_channels: 1024,
            num_blocks: 30,
            num_heads: 12,
            mlp_ratio: 5.3334,
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
        }
    }

    /// `int(model_channels * mlp_ratio)` (torch `int()` truncation).
    pub fn mlp_hidden(&self) -> usize {
        (self.model_channels as f64 * self.mlp_ratio) as usize
    }

    /// `model_channels / num_heads` (128).
    pub fn head_dim(&self) -> usize {
        self.model_channels / self.num_heads
    }

    /// Total token count (`resolution^3`, 4096).
    pub fn token_count(&self) -> usize {
        self.resolution * self.resolution * self.resolution
    }

    /// The exact tensor names this model consumes below `prefix`
    /// (640 for the TRELLIS.2 SS checkpoint: 10 top-level + 30 blocks x 21).
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.push("t_embedder.mlp.0.weight".to_string());
        names.push("t_embedder.mlp.0.bias".to_string());
        names.push("t_embedder.mlp.2.weight".to_string());
        names.push("t_embedder.mlp.2.bias".to_string());
        if self.share_mod {
            names.push("adaLN_modulation.1.weight".to_string());
            names.push("adaLN_modulation.1.bias".to_string());
        }
        names.push("input_layer.weight".to_string());
        names.push("input_layer.bias".to_string());
        names.push("out_layer.weight".to_string());
        names.push("out_layer.bias".to_string());
        for index in 0..self.num_blocks {
            let block = format!("blocks.{index}.");
            for name in [
                "norm2.weight",
                "norm2.bias",
                "modulation",
                "self_attn.to_qkv.weight",
                "self_attn.to_qkv.bias",
                "self_attn.to_out.weight",
                "self_attn.to_out.bias",
                "self_attn.q_rms_norm.gamma",
                "self_attn.k_rms_norm.gamma",
                "cross_attn.to_q.weight",
                "cross_attn.to_q.bias",
                "cross_attn.to_kv.weight",
                "cross_attn.to_kv.bias",
                "cross_attn.to_out.weight",
                "cross_attn.to_out.bias",
                "cross_attn.q_rms_norm.gamma",
                "cross_attn.k_rms_norm.gamma",
                "mlp.mlp.0.weight",
                "mlp.mlp.0.bias",
                "mlp.mlp.2.weight",
                "mlp.mlp.2.bias",
            ] {
                names.push(format!("{block}{name}"));
            }
        }
        names
    }
}

/// Dense sparse-structure flow DiT producing the SS latent velocity field.
#[derive(Debug)]
pub struct Trellis2SsDit {
    config: Trellis2SsConfig,
    t_embedder: Trellis2TimestepEmbedder,
    ada_ln: Trellis2AdaLn,
    input_layer: Linear,
    blocks: Vec<Trellis2DenseBlock>,
    out_layer: Linear,
    rope: Trellis2Rope3d,
    /// Cached RoPE tables over the fixed dense grid, `[token_count, 64]`.
    rope_cos: Tensor,
    rope_sin: Tensor,
}

impl Trellis2SsDit {
    /// Load from a VarBuilder rooted at the model tensors. For the canonical
    /// package pass `vb.pp("ss_flow")`; for the raw checkpoint pass the root
    /// builder. Fails if any of the 640 expected tensors is missing.
    pub fn load(vb: VarBuilder, config: &Trellis2SsConfig) -> CandleResult<Self> {
        if !config.share_mod {
            candle_core::bail!(
                "TRELLIS.2 flow checkpoints all use share_mod=true; per-block \
                 adaLN is not implemented"
            );
        }
        let channels = config.model_channels;
        let t_embedder = Trellis2TimestepEmbedder::load(vb.pp("t_embedder"), channels)?;
        let ada_ln = Trellis2AdaLn::load(vb.pp("adaLN_modulation"), channels)?;
        let input_layer = linear(vb.pp("input_layer"), config.in_channels, channels, true)?;
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for index in 0..config.num_blocks {
            blocks.push(Trellis2DenseBlock::load(
                vb.pp("blocks").pp(index.to_string()),
                channels,
                config.cond_channels,
                config.num_heads,
                config.mlp_hidden(),
                config.qk_rms_norm,
                config.qk_rms_norm_cross,
            )?);
            // Keep GPU staging windows small while loading ~2.6 GiB of weights.
            super::drain_upload_staging(vb.device())?;
        }
        let out_layer = linear(vb.pp("out_layer"), channels, config.out_channels, true)?;

        let rope = Trellis2Rope3d::new(config.head_dim());
        let coords = cubic_token_coords(config.resolution);
        let (rope_cos, rope_sin) = rope.cos_sin(&coords, vb.device())?;

        Ok(Self {
            config: config.clone(),
            t_embedder,
            ada_ln,
            input_layer,
            blocks,
            out_layer,
            rope,
            rope_cos,
            rope_sin,
        })
    }

    pub fn config(&self) -> &Trellis2SsConfig {
        &self.config
    }

    /// Token-level forward (`[B, 4096, 8]` -> `[B, 4096, 8]`) without the
    /// 5D reshape, for tests and stage-level comparisons.
    pub fn forward_tokens(
        &self,
        tokens: &Tensor,
        t_model: f32,
        cond: &Tensor,
    ) -> CandleResult<Tensor> {
        let (_batch, token_count, channels) = tokens.dims3()?;
        if channels != self.config.in_channels {
            candle_core::bail!(
                "SS DiT expects {} token channels, got {channels}",
                self.config.in_channels
            );
        }
        if token_count != self.config.token_count() {
            candle_core::bail!(
                "SS DiT expects {} tokens, got {token_count}",
                self.config.token_count()
            );
        }
        let (cond_batch, cond_tokens, cond_channels) = cond.dims3()?;
        if cond_channels != self.config.cond_channels {
            candle_core::bail!(
                "SS DiT cond must have {} channels, got {cond_channels}",
                self.config.cond_channels
            );
        }
        let _ = (cond_batch, cond_tokens);
        let hidden = self.input_layer.forward(tokens)?;
        let t_emb = self.ada_ln.forward(&self.t_embedder.forward(t_model)?)?;
        let mut xs = hidden;
        for block in &self.blocks {
            xs = block.forward(&xs, &t_emb, cond, &self.rope_cos, &self.rope_sin)?;
        }
        // Final non-affine F.layer_norm with eps 1e-5 (reference default),
        // then the output projection back to latent channels.
        let normed = layer_norm_no_affine(&xs, 1e-5)?;
        self.out_layer.forward(&normed)
    }

    /// Full model forward: `x: [B, 8, 16, 16, 16]`, `t_model = 1000 * t`,
    /// `cond: [B, Lkv, 1024]` -> predicted velocity `[B, 8, 16, 16, 16]`.
    pub fn forward(&self, x: &Tensor, t_model: f32, cond: &Tensor) -> CandleResult<Tensor> {
        let (batch, channels, rx, ry, rz) = x.dims5()?;
        let resolution = self.config.resolution;
        if channels != self.config.in_channels
            || (rx, ry, rz) != (resolution, resolution, resolution)
        {
            candle_core::bail!(
                "SS DiT expects [B, {}, {}, {}, {}, {}], got [{batch}, {channels}, {rx}, {ry}, {rz}]",
                self.config.in_channels,
                resolution,
                resolution,
                resolution,
                resolution
            );
        }
        let (cond_batch, cond_tokens, cond_channels) = cond.dims3()?;
        if cond_batch != batch || cond_channels != self.config.cond_channels {
            candle_core::bail!(
                "SS DiT cond must be [{batch}, Lkv, {}], got [{cond_batch}, {cond_tokens}, \
                 {cond_channels}]",
                self.config.cond_channels
            );
        }
        // x.view(B, 8, -1).permute(0, 2, 1) -> tokens row-major over (x, y, z).
        let tokens = x
            .reshape((batch, channels, self.config.token_count()))?
            .transpose(1, 2)?
            .contiguous()?;
        let out = self.forward_tokens(&tokens, t_model, cond)?;
        out.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            self.config.out_channels,
            resolution,
            resolution,
            resolution,
        ))
    }

    /// RoPE phase tables in use (`[token_count, head_dim/2]`).
    pub fn rope_tables(&self) -> (&Tensor, &Tensor) {
        (&self.rope_cos, &self.rope_sin)
    }

    /// The 3D RoPE embedder in use (head_dim 128, rope_freq `(1, 10000)`).
    pub fn rope(&self) -> &Trellis2Rope3d {
        &self.rope
    }
}

/// Shape SLAT flow DiT architecture parameters (`SLatFlowModel`, checkpoints
/// `slat_flow_img2shape_dit_1_3B_512_bf16` and `..._1024_bf16`, 640 tensors
/// each). Identical block math to the dense SS flow: every block is a
/// `ModulatedSparseTransformerCrossBlock`, and with a single batch element
/// the sparse block-diagonal full attention is mathematically the dense
/// `B = 1` path, so the same `Trellis2DenseBlock` is reused.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2SlatConfig {
    /// Latent grid resolution per axis (32 for the 512 stage, 64 for 1024).
    /// Used to validate the coordinate range; token count varies per sample.
    pub resolution: usize,
    /// Input latent channels (32 shape; 64 texture = noise 32 +
    /// normalized shape SLAT 32).
    pub in_channels: usize,
    /// Output channels (32).
    pub out_channels: usize,
    /// Model width (1536).
    pub model_channels: usize,
    /// Cross-attention context channels (1024, DINOv3).
    pub cond_channels: usize,
    /// Transformer block count (30).
    pub num_blocks: usize,
    /// Attention heads (12; head dim 128).
    pub num_heads: usize,
    /// FFN ratio (5.3334 -> hidden 8192).
    pub mlp_ratio: f64,
    /// Global adaLN (true).
    pub share_mod: bool,
    /// q/k RMS norm in self attention (true).
    pub qk_rms_norm: bool,
    /// q/k RMS norm in cross attention (true).
    pub qk_rms_norm_cross: bool,
}

impl Trellis2SlatConfig {
    /// `slat_flow_img2shape_dit_1_3B_512_bf16` args (32^3 latent grid).
    pub const fn shape_512() -> Self {
        Self {
            resolution: 32,
            in_channels: 32,
            out_channels: 32,
            model_channels: 1536,
            cond_channels: 1024,
            num_blocks: 30,
            num_heads: 12,
            mlp_ratio: 5.3334,
            share_mod: true,
            qk_rms_norm: true,
            qk_rms_norm_cross: true,
        }
    }

    /// `slat_flow_img2shape_dit_1_3B_1024_bf16` args (64^3 latent grid).
    pub const fn shape_1024() -> Self {
        Self {
            resolution: 64,
            ..Self::shape_512()
        }
    }

    /// `slat_flow_imgshape2tex_dit_1_3B_512_bf16` (texture SLAT flow, 512
    /// stage): identical transformer to the shape flow with a 64-channel
    /// input (`sp.sparse_cat([noise32, shape_slat_norm32], dim=-1)`) and a
    /// 32-channel output (the normalized texture SLAT).
    pub const fn tex_512() -> Self {
        Self {
            in_channels: 64,
            ..Self::shape_512()
        }
    }

    /// `slat_flow_imgshape2tex_dit_1_3B_1024_bf16` (texture SLAT flow,
    /// 1024 cascade stage; 64^3 latent grid).
    pub const fn tex_1024() -> Self {
        Self {
            resolution: 64,
            ..Self::tex_512()
        }
    }

    /// `int(model_channels * mlp_ratio)` (torch `int()` truncation).
    pub fn mlp_hidden(&self) -> usize {
        (self.model_channels as f64 * self.mlp_ratio) as usize
    }

    /// `model_channels / num_heads` (128).
    pub fn head_dim(&self) -> usize {
        self.model_channels / self.num_heads
    }

    /// The exact tensor names this model consumes below `prefix` (640 for
    /// both shape checkpoints: 10 top-level + 30 blocks x 21 — the sparse
    /// block has the same tensor set as the dense one because norm1/norm3
    /// are non-affine).
    pub fn expected_tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.push("t_embedder.mlp.0.weight".to_string());
        names.push("t_embedder.mlp.0.bias".to_string());
        names.push("t_embedder.mlp.2.weight".to_string());
        names.push("t_embedder.mlp.2.bias".to_string());
        if self.share_mod {
            names.push("adaLN_modulation.1.weight".to_string());
            names.push("adaLN_modulation.1.bias".to_string());
        }
        names.push("input_layer.weight".to_string());
        names.push("input_layer.bias".to_string());
        names.push("out_layer.weight".to_string());
        names.push("out_layer.bias".to_string());
        for index in 0..self.num_blocks {
            let block = format!("blocks.{index}.");
            for name in [
                "norm2.weight",
                "norm2.bias",
                "modulation",
                "self_attn.to_qkv.weight",
                "self_attn.to_qkv.bias",
                "self_attn.to_out.weight",
                "self_attn.to_out.bias",
                "self_attn.q_rms_norm.gamma",
                "self_attn.k_rms_norm.gamma",
                "cross_attn.to_q.weight",
                "cross_attn.to_q.bias",
                "cross_attn.to_kv.weight",
                "cross_attn.to_kv.bias",
                "cross_attn.to_out.weight",
                "cross_attn.to_out.bias",
                "cross_attn.q_rms_norm.gamma",
                "cross_attn.k_rms_norm.gamma",
                "mlp.mlp.0.weight",
                "mlp.mlp.0.bias",
                "mlp.mlp.2.weight",
                "mlp.mlp.2.bias",
            ] {
                names.push(format!("{block}{name}"));
            }
        }
        names
    }
}

/// Shape SLAT flow DiT (`SLatFlowModel`) operating on sparse voxel tokens.
///
/// Forward: `feats [N, in_channels]` on `coords: [N]` (row order preserved) ->
/// `input_layer` Linear(in, model_channels) -> 30 modulated blocks (adaLN
/// self attention with RoPE phases from the token coords, cross attention to
/// the DINOv3 cond) -> final non-affine `F.layer_norm` (eps 1e-5) ->
/// `out_layer` Linear(model_channels, out). F32 compute throughout.
#[derive(Debug)]
pub struct Trellis2SlatDit {
    config: Trellis2SlatConfig,
    t_embedder: Trellis2TimestepEmbedder,
    ada_ln: Trellis2AdaLn,
    input_layer: Linear,
    blocks: Vec<Trellis2DenseBlock>,
    out_layer: Linear,
    rope: Trellis2Rope3d,
}

impl Trellis2SlatDit {
    /// Load from a VarBuilder rooted at the model tensors. For the canonical
    /// package pass `vb.pp("shape_dit_512")` / `vb.pp("shape_dit_1024")`; for
    /// a raw checkpoint pass the root builder. Fails if any of the 640
    /// expected tensors is missing.
    pub fn load(vb: VarBuilder, config: &Trellis2SlatConfig) -> CandleResult<Self> {
        if !config.share_mod {
            candle_core::bail!(
                "TRELLIS.2 flow checkpoints all use share_mod=true; per-block \
                 adaLN is not implemented"
            );
        }
        let channels = config.model_channels;
        let t_embedder = Trellis2TimestepEmbedder::load(vb.pp("t_embedder"), channels)?;
        let ada_ln = Trellis2AdaLn::load(vb.pp("adaLN_modulation"), channels)?;
        let input_layer = linear(vb.pp("input_layer"), config.in_channels, channels, true)?;
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for index in 0..config.num_blocks {
            blocks.push(Trellis2DenseBlock::load(
                vb.pp("blocks").pp(index.to_string()),
                channels,
                config.cond_channels,
                config.num_heads,
                config.mlp_hidden(),
                config.qk_rms_norm,
                config.qk_rms_norm_cross,
            )?);
            // Keep GPU staging windows small while loading ~2.6 GiB of weights.
            super::drain_upload_staging(vb.device())?;
        }
        let out_layer = linear(vb.pp("out_layer"), channels, config.out_channels, true)?;
        Ok(Self {
            config: config.clone(),
            t_embedder,
            ada_ln,
            input_layer,
            blocks,
            out_layer,
            rope: Trellis2Rope3d::new(config.head_dim()),
        })
    }

    pub fn config(&self) -> &Trellis2SlatConfig {
        &self.config
    }

    /// Validate token features/coords and build the RoPE tables from the
    /// token coords in row order (`coords[:, 1:]` of the reference
    /// SparseTensor, batch dim dropped).
    fn prepare(
        &self,
        feats: &Tensor,
        coords: &[[i32; 3]],
    ) -> CandleResult<(Tensor, Tensor, Tensor)> {
        let (rows, channels) = feats.dims2()?;
        if channels != self.config.in_channels {
            candle_core::bail!(
                "SLAT DiT expects {} token channels, got {channels}",
                self.config.in_channels
            );
        }
        if rows != coords.len() {
            candle_core::bail!(
                "SLAT DiT coords/features mismatch: {} coords vs {rows} rows",
                coords.len()
            );
        }
        let resolution = self.config.resolution as i32;
        let mut flat = Vec::with_capacity(rows * 3);
        for coord in coords {
            for value in coord {
                if *value < 0 || *value >= resolution {
                    candle_core::bail!(
                        "SLAT coord {coord:?} outside the 0..{} grid",
                        self.config.resolution
                    );
                }
                flat.push(*value as i64);
            }
        }
        let (cos, sin) = self.rope.cos_sin(&flat, feats.device())?;
        Ok((feats.clone(), cos, sin))
    }

    /// Token-level forward: `feats: [N, in_channels]` on `coords` ->
    /// predicted velocity `[N, out_channels]` in the same row order.
    /// `t_model = 1000 * t`, `cond: [B, Lkv, cond_channels]`.
    pub fn forward(
        &self,
        feats: &Tensor,
        coords: &[[i32; 3]],
        t_model: f32,
        cond: &Tensor,
    ) -> CandleResult<Tensor> {
        let (_cond_batch, _cond_tokens, cond_channels) = cond.dims3()?;
        if cond_channels != self.config.cond_channels {
            candle_core::bail!(
                "SLAT DiT cond must have {} channels, got {cond_channels}",
                self.config.cond_channels
            );
        }
        let (feats, cos, sin) = self.prepare(feats, coords)?;
        let xs = self.input_layer.forward(&feats)?.unsqueeze(0)?; // [1, N, C]
        let t_emb = self.ada_ln.forward(&self.t_embedder.forward(t_model)?)?;
        let mut xs = xs;
        for block in &self.blocks {
            xs = block.forward(&xs, &t_emb, cond, &cos, &sin)?;
        }
        // Final non-affine F.layer_norm with eps 1e-5 (reference default),
        // then the output projection back to latent channels.
        let normed = layer_norm_no_affine(&xs, 1e-5)?;
        self.out_layer.forward(&normed.squeeze(0)?)
    }

    /// `SparseFeats` convenience wrapper around [`Self::forward`] (crate
    /// internal; the shape VAE decoder in T5 operates on SparseFeats).
    #[allow(dead_code)]
    pub(crate) fn forward_sparse(
        &self,
        x: &SparseFeats,
        t_model: f32,
        cond: &Tensor,
    ) -> CandleResult<Tensor> {
        self.forward(&x.feats, &x.coords, t_model, cond)
    }

    /// RoPE phase tables for arbitrary token coords (`[N, head_dim/2]`); the
    /// dense SS DiT caches tables for its fixed grid, the SLAT grids vary per
    /// sample so they are built per forward.
    pub fn rope_tables_for(
        &self,
        coords: &[[i32; 3]],
        device: &candle_core::Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let mut flat = Vec::with_capacity(coords.len() * 3);
        for coord in coords {
            flat.push(coord[0] as i64);
            flat.push(coord[1] as i64);
            flat.push(coord[2] as i64);
        }
        self.rope.cos_sin(&flat, device)
    }
}
