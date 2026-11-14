//! TRELLIS.2 texture branch (T7): texture SLAT flow DiT wrapper, guided
//! texture VAE decode, PBR postprocess and per-vertex trilinear attribute
//! sampling.
//!
//! Reference: `trellis2/pipelines/trellis2_image_to_3d.py`
//! (`sample_tex_slat` 391-432, `decode_tex_slat` 434-453, `decode_latent`
//! 455-486), `trellis2/models/structured_latent_flow.py`
//! (`SLatFlowModel.forward` concat), `trellis2/representations/mesh/base.py`
//! (`MeshWithVoxel.query_attrs`), and the flex_gemm
//! `ops/grid_sample/grid_sample_torch.py` / CUDA trilinear kernel the
//! reference calls.
//!
//! Texture SLAT flow (`tex_dit_512.` / `tex_dit_1024.`, 640 stage tensors
//! each): the input is `[N, 64]` = `sparse_cat([noise, shape_slat_norm])`
//! (noise channels FIRST, the re-normalized shape SLAT second), the output
//! is `[N, 32]` in normalized texture units, de-normalized with
//! `tex_slat_normalization`. Sampler defaults (`pipeline.json`
//! `tex_slat_sampler`): 12 steps, guidance 1.0 (cond-only path),
//! guidance_rescale 0.0, interval [0.6, 0.9], rescale_t 3.0.
//!
//! Texture VAE (`tex_dec.`, 284 stage tensors): the shape decoder topology
//! with `out_channels 6` and `pred_subdiv = false`; every C2S block is
//! guided by the raw `to_subdiv` logits of the shape decoder's decode
//! (`guide_subs`, stage order), so the tex decode walks exactly the shape
//! decoder's coordinate tree and its output coords equal the shape decode's
//! final coords. The decoded 6 channels are `raw * 0.5 + 0.5` clamped to
//! [0, 1] with the PBR layout `base_color 0:3, metallic 3:4, roughness 4:5,
//! alpha 5:6`.
//!
//! Per-vertex attributes (`MeshWithVoxel.query_vertex_attrs`): the mesh
//! vertices are mapped into the voxel grid (`grid = (vertex - origin) /
//! voxel_size` with `origin = -0.5`, `voxel_size = 1 / resolution`) and
//! sampled with the flex_gemm sparse trilinear semantics: 8 neighbor cells
//! at `floor(q - 0.5) + (dx, dy, dz)` (`subidx = dx + 2*dy + 4*dz` order),
//! each weighted by `prod_axis(1 - |q - cell - 0.5|)`; missing (or
//! out-of-grid) neighbors contribute weight zero and the weights are
//! normalized by the surviving weight mass (`w / max(w_sum, 1e-12)` in the
//! CUDA kernel), i.e. the output is the weighted average over the PRESENT
//! neighbors.

use std::collections::HashMap;

use candle_core::{DType, Result as CandleResult, Tensor};

use super::trellis2_dit::Trellis2SlatDit;
use super::trellis2_vae::Trellis2ShapeVaeDecoder;
use crate::weights::VarBuilder;

/// PBR attribute channel layout (reference `pbr_attr_layout`,
/// `trellis2_image_to_3d.py:73-78`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Trellis2PbrLayout;

impl Trellis2PbrLayout {
    /// Channel count of the decoded texture attributes (6).
    pub const CHANNELS: usize = 6;
    /// `base_color` channel range `[0, 3)`.
    pub const BASE_COLOR: std::ops::Range<usize> = 0..3;
    /// `metallic` channel index (3).
    pub const METALLIC: usize = 3;
    /// `roughness` channel index (4).
    pub const ROUGHNESS: usize = 4;
    /// `alpha` channel index (5).
    pub const ALPHA: usize = 5;
}

/// Texture SLAT flow DiT: [`Trellis2SlatDit`] loaded with the 64-channel
/// texture config. The wrapper validates the channel contract and owns the
/// `noise || shape_slat_norm` concatenation (`sp.sparse_cat([x,
/// concat_cond], dim=-1)`, `structured_latent_flow.py:177-178`).
#[derive(Debug)]
pub struct Trellis2TexDit {
    inner: Trellis2SlatDit,
}

impl Trellis2TexDit {
    /// Load from a VarBuilder rooted at the stage tensors (canonical
    /// package: `vb.pp("tex_dit_512")` / `vb.pp("tex_dit_1024")`). The
    /// config must be a texture config (`in_channels == 64`,
    /// `out_channels == 32`).
    pub fn load(
        vb: VarBuilder,
        config: &super::trellis2_dit::Trellis2SlatConfig,
    ) -> CandleResult<Self> {
        if config.in_channels != 64 || config.out_channels != 32 {
            candle_core::bail!(
                "texture SLAT flow config must have in_channels 64 / out_channels 32, \
                 got {}/{}",
                config.in_channels,
                config.out_channels
            );
        }
        Ok(Self {
            inner: Trellis2SlatDit::load(vb, config)?,
        })
    }

    pub fn config(&self) -> &super::trellis2_dit::Trellis2SlatConfig {
        self.inner.config()
    }

    /// Token-level forward: `noise_feats [N, 32]` (the sampled texture
    /// noise) concatenated with `shape_slat_norm [N, 32]` (the shape SLAT
    /// re-normalized with the shape vectors) along channels -> predicted
    /// velocity `[N, 32]` on the same rows.
    pub fn forward(
        &self,
        noise_feats: &Tensor,
        shape_slat_norm: &Tensor,
        coords: &[[i32; 3]],
        t_model: f32,
        cond: &Tensor,
    ) -> CandleResult<Tensor> {
        let x = concat_tex_input(noise_feats, shape_slat_norm)?;
        self.inner.forward(&x, coords, t_model, cond)
    }
}

/// `sp.sparse_cat([x, concat_cond], dim=-1)`: noise feats first, the
/// normalized shape SLAT second. Row counts and devices must match.
pub fn concat_tex_input(noise_feats: &Tensor, shape_slat_norm: &Tensor) -> CandleResult<Tensor> {
    let (noise_rows, noise_channels) = noise_feats.dims2()?;
    let (cond_rows, cond_channels) = shape_slat_norm.dims2()?;
    if noise_rows != cond_rows {
        candle_core::bail!(
            "texture SLAT input row mismatch: noise {noise_rows} vs shape cond {cond_rows}"
        );
    }
    if noise_channels + cond_channels != 64 {
        candle_core::bail!(
            "texture SLAT input must concatenate to 64 channels, got {noise_channels} + \
             {cond_channels}"
        );
    }
    Tensor::cat(&[noise_feats, shape_slat_norm], 1)
}

/// Texture sparse VAE decoder (`SparseUnetVaeDecoder` with `pred_subdiv =
/// false`, checkpoint `tex_dec_next_dc_f16c32_fp16`, 284 stage tensors).
/// Shares the shape decoder's ConvNeXt/C2S block implementations; the C2S
/// blocks run the identical math with the shape decoder's guide logits.
#[derive(Debug)]
pub struct Trellis2TexDecoder {
    inner: Trellis2ShapeVaeDecoder,
}

/// Guided texture decode output.
#[derive(Debug, Clone)]
pub struct TexDecodeOutput {
    /// Final voxel coordinates (identical to the shape decode's final
    /// coords: every child is selected by the guide).
    pub coords: Vec<[i32; 3]>,
    /// Raw decoded 6-channel features `[N, 6]` (pre `* 0.5 + 0.5`).
    pub feats: Tensor,
}

impl Trellis2TexDecoder {
    /// Load from a VarBuilder rooted at the stage tensors (canonical
    /// package: `vb.pp("tex_dec")`).
    pub fn load(
        vb: VarBuilder,
        config: &super::trellis2_vae::Trellis2ShapeVaeConfig,
    ) -> CandleResult<Self> {
        if config.pred_subdiv {
            candle_core::bail!(
                "texture VAE config must set pred_subdiv = false (guided C2S; \
                 no to_subdiv weights exist in tex_dec)"
            );
        }
        if config.out_channels != Trellis2PbrLayout::CHANNELS {
            candle_core::bail!(
                "texture VAE config must decode {} channels, got {}",
                Trellis2PbrLayout::CHANNELS,
                config.out_channels
            );
        }
        Ok(Self {
            inner: Trellis2ShapeVaeDecoder::load(vb, config)?,
        })
    }

    pub fn config(&self) -> &super::trellis2_vae::Trellis2ShapeVaeConfig {
        self.inner.config()
    }

    /// Guided decode: `guide_subs` are the shape decoder's raw subdivision
    /// logits, one `[N_parents_stage, 8]` tensor per C2S stage in stage
    /// order (reference `decode_tex_slat`: `tex_decoder(slat,
    /// guide_subs=subs)`).
    pub fn decode(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        guide_subs: &[Tensor],
    ) -> CandleResult<TexDecodeOutput> {
        let decoded = self
            .inner
            .decode_with_guides(coords, latent, guide_subs, false)?;
        Ok(TexDecodeOutput {
            coords: decoded.coords,
            feats: decoded.feats,
        })
    }

    /// Guided decode with the per-block trace (parity tests).
    pub fn decode_with_trace(
        &self,
        coords: &[[i32; 3]],
        latent: &Tensor,
        guide_subs: &[Tensor],
    ) -> CandleResult<super::trellis2_vae::ShapeVaeDecodeOutput> {
        self.inner
            .decode_with_guides(coords, latent, guide_subs, true)
    }
}

/// PBR postprocess (`decode_tex_slat`): `attrs = feats * 0.5 + 0.5`,
/// clamped to [0, 1]. Returns the flat `[N, 6]` attribute rows on the host
/// in f32.
pub fn pbr_attrs_from_raw(feats: &Tensor) -> CandleResult<Vec<f32>> {
    let (_rows, channels) = feats.dims2()?;
    if channels != Trellis2PbrLayout::CHANNELS {
        candle_core::bail!(
            "texture decode must produce {} channels, got {channels}",
            Trellis2PbrLayout::CHANNELS
        );
    }
    let values = feats
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .affine(0.5, 0.5)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values.iter().map(|&v| v.clamp(0.0, 1.0)).collect())
}

/// Split flat `[N, 6]` PBR attributes into the named layout slices (host
/// f32). Returns `(base_color [N,3], metallic [N], roughness [N],
/// alpha [N])`.
pub fn split_pbr_attrs(
    attrs: &[f32],
) -> CandleResult<(Vec<[f32; 3]>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    if attrs.len() % Trellis2PbrLayout::CHANNELS != 0 {
        candle_core::bail!(
            "PBR attribute buffer length {} is not a multiple of {}",
            attrs.len(),
            Trellis2PbrLayout::CHANNELS
        );
    }
    let rows = attrs.len() / Trellis2PbrLayout::CHANNELS;
    let mut base_color = Vec::with_capacity(rows);
    let mut metallic = Vec::with_capacity(rows);
    let mut roughness = Vec::with_capacity(rows);
    let mut alpha = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * Trellis2PbrLayout::CHANNELS;
        base_color.push([
            attrs[base + Trellis2PbrLayout::BASE_COLOR.start],
            attrs[base + Trellis2PbrLayout::BASE_COLOR.start + 1],
            attrs[base + Trellis2PbrLayout::BASE_COLOR.start + 2],
        ]);
        metallic.push(attrs[base + Trellis2PbrLayout::METALLIC]);
        roughness.push(attrs[base + Trellis2PbrLayout::ROUGHNESS]);
        alpha.push(attrs[base + Trellis2PbrLayout::ALPHA]);
    }
    Ok((base_color, metallic, roughness, alpha))
}

/// Sparse trilinear attribute sampling with the flex_gemm
/// `grid_sample_3d(mode='trilinear')` semantics
/// (`MeshWithVoxel.query_attrs`).
///
/// * `vertices`: flat `[V, 3]` mesh vertices in the reference frame.
/// * `coords`/`attrs`: the sparse voxel grid, `attrs` flat `[N, channels]`.
/// * `resolution`: the voxel grid resolution (`origin = -0.5`,
///   `voxel_size = 1 / resolution`, so `grid = (vertex + 0.5) *
///   resolution`).
///
/// Returns the flat `[V, channels]` sampled attributes. Missing neighbors
/// contribute weight zero; the remaining weights are normalized by their
/// sum (clamped at 1e-12), so a query surrounded by a single present voxel
/// returns that voxel's value exactly.
pub fn query_vertex_attrs(
    vertices: &[[f32; 3]],
    coords: &[[i32; 3]],
    attrs: &[f32],
    channels: usize,
    resolution: u32,
) -> CandleResult<Vec<f32>> {
    if channels == 0 {
        candle_core::bail!("trilinear sampling needs at least one attribute channel");
    }
    if attrs.len() % channels != 0 {
        candle_core::bail!(
            "attribute buffer length {} is not a multiple of {channels}",
            attrs.len()
        );
    }
    let rows = attrs.len() / channels;
    if rows != coords.len() {
        candle_core::bail!(
            "trilinear coords/attrs mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    let map: HashMap<[i32; 3], u32> = coords
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index as u32))
        .collect();
    let resolution_f = resolution as f32;
    let mut out = vec![0.0_f32; vertices.len() * channels];
    // Corner subidx order dx + 2*dy + 4*dz matches the CUDA kernel's
    // `base + (i & 1, (i >> 1) & 1, (i >> 2) & 1)`.
    for (vertex_index, vertex) in vertices.iter().enumerate() {
        // grid = (vertex - origin) / voxel_size, origin -0.5.
        let q = [
            (vertex[0] + 0.5) * resolution_f,
            (vertex[1] + 0.5) * resolution_f,
            (vertex[2] + 0.5) * resolution_f,
        ];
        let base = [
            (q[0] - 0.5).floor() as i64,
            (q[1] - 0.5).floor() as i64,
            (q[2] - 0.5).floor() as i64,
        ];
        let mut acc = vec![0.0_f64; channels];
        let mut weight_sum = 0.0_f64;
        for subidx in 0..8usize {
            let dx = (subidx & 1) as i64;
            let dy = ((subidx >> 1) & 1) as i64;
            let dz = ((subidx >> 2) & 1) as i64;
            let cell = [base[0] + dx, base[1] + dy, base[2] + dz];
            // Grid bounds (0..resolution per axis) and hashmap occupancy:
            // cells outside the grid cannot be occupied, so both checks
            // drop the same neighbors (the reference bounds-checks first).
            if cell.iter().any(|&c| c < 0 || c >= resolution as i64) {
                continue;
            }
            let cell_i = [cell[0] as i32, cell[1] as i32, cell[2] as i32];
            let Some(&row) = map.get(&cell_i) else {
                continue;
            };
            let mut weight = 1.0_f64;
            for axis in 0..3 {
                let center = cell[axis] as f64 + 0.5;
                weight *= 1.0 - (q[axis] as f64 - center).abs();
            }
            // Zero weight contributes nothing (the reference multiplies the
            // zero factor through and it drops out of the sum).
            weight_sum += weight;
            let attr_base = row as usize * channels;
            for (channel, acc_channel) in acc.iter_mut().enumerate() {
                *acc_channel += weight * attrs[attr_base + channel] as f64;
            }
        }
        let denom = weight_sum.max(1e-12);
        let dst = vertex_index * channels;
        for (channel, acc_channel) in acc.iter().enumerate() {
            out[dst + channel] = (acc_channel / denom) as f32;
        }
    }
    Ok(out)
}

/// Sample the base-color channels (PBR layout 0:3) at every mesh vertex.
/// Returns the flat `[V, 3]` RGB rows in [0, 1] for PLY/GLB vertex colors.
pub fn query_vertex_colors(
    vertices: &[[f32; 3]],
    coords: &[[i32; 3]],
    attrs: &[f32],
    resolution: u32,
) -> CandleResult<Vec<f32>> {
    let channels = Trellis2PbrLayout::CHANNELS;
    let sampled = query_vertex_attrs(vertices, coords, attrs, channels, resolution)?;
    let mut colors = Vec::with_capacity(vertices.len() * 3);
    for row in 0..vertices.len() {
        let base = row * channels;
        colors.push(sampled[base]);
        colors.push(sampled[base + 1]);
        colors.push(sampled[base + 2]);
    }
    Ok(colors)
}
