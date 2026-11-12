use std::{collections::HashMap, sync::Mutex};

use candle_core::{D, Device, DeviceLocation, IndexOp, Module, Result as CandleResult, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, LayerNorm, conv_transpose2d,
    conv2d, conv2d_no_bias, layer_norm,
};
use crate::weights::VarBuilder;

pub(crate) const DPT_PATCH_SIZE: usize = 14;
pub(crate) const DPT_INTERMEDIATE_LAYERS: [usize; 4] = [4, 11, 17, 23];
const DPT_FEATURES: usize = 256;
const DPT_OUT_CHANNELS: [usize; 4] = [256, 512, 1024, 1024];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VggtDptActivation {
    /// `sign(x) * (exp(|x|) - 1)` (reference `inv_log`, point head).
    InvLog,
    /// `exp(x)` (reference `exp`, depth head).
    Exp,
}

#[derive(Debug)]
struct ResidualConvUnit {
    conv1: Conv2d,
    conv2: Conv2d,
}

impl ResidualConvUnit {
    fn load(vb: VarBuilder, features: usize) -> CandleResult<Self> {
        let config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv2d(features, features, 3, config.clone(), vb.pp("conv1"))?,
            conv2: conv2d(features, features, 3, config, vb.pp("conv2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        // The reference uses nn.ReLU(inplace=True); the first in-place ReLU
        // mutates `x`, so the residual added back is relu(x), not x.
        let activated = xs.relu()?;
        let out = self.conv1.forward(&activated)?;
        let out = self.conv2.forward(&out.relu()?)?;
        out + activated
    }
}

#[derive(Debug)]
struct FeatureFusionBlock {
    out_conv: Conv2d,
    res_conf_unit1: Option<ResidualConvUnit>,
    res_conf_unit2: ResidualConvUnit,
}

impl FeatureFusionBlock {
    fn load(vb: VarBuilder, features: usize, has_residual: bool) -> CandleResult<Self> {
        Ok(Self {
            out_conv: conv2d(
                features,
                features,
                1,
                Conv2dConfig::default(),
                vb.pp("out_conv"),
            )?,
            res_conf_unit1: if has_residual {
                Some(ResidualConvUnit::load(vb.pp("resConfUnit1"), features)?)
            } else {
                None
            },
            res_conf_unit2: ResidualConvUnit::load(vb.pp("resConfUnit2"), features)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        skip: Option<&Tensor>,
        size: Option<(usize, usize)>,
    ) -> CandleResult<Tensor> {
        let mut output = xs.clone();
        if let Some(unit) = &self.res_conf_unit1 {
            let skip = skip.ok_or_else(|| {
                candle_core::Error::Msg("VGGT DPT refinenet requires a skip connection".to_string())
            })?;
            output = (&output + &unit.forward(skip)?)?;
        }
        let output = self.res_conf_unit2.forward(&output)?;
        let (_n, _c, height, width) = output.dims4()?;
        let output = match size {
            Some((target_h, target_w)) => output.upsample_bilinear2d(target_h, target_w, true)?,
            None => output.upsample_bilinear2d(height * 2, width * 2, true)?,
        };
        self.out_conv.forward(&output)
    }
}

#[derive(Debug)]
enum ResizeLayer {
    Transpose(ConvTranspose2d),
    Identity,
    Conv(Conv2d),
}

impl ResizeLayer {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Transpose(layer) => layer.forward(xs),
            Self::Identity => Ok(xs.clone()),
            Self::Conv(layer) => layer.forward(xs),
        }
    }
}

/// DPT head used by both the VGGT point head and the VGGT depth head.
///
/// Mirrors `vggt.heads.dpt_head.DPTHead` with `patch_size=14`, `features=256`,
/// Upper bound on cached positional-embedding shapes. Each entry stores only
/// the two separable axis tables, so 16 entries stay in the low single-digit
/// megabytes even at 518x518 with 256 channels.
const POS_EMBED_CACHE_CAP: usize = 16;

/// Separable positional-embedding tables for one `(patch_h, patch_w, dim)`
/// shape: `u` is `[1, dim/2, 1, patch_w]` and `v` is
/// `[1, dim/2, patch_h, 1]`. The reference concatenates the full `u`/`v`
/// grids along channels; adding each half to the corresponding channel slice
/// is algebraically identical and avoids retaining a `[1, C, H, W]` tensor.
#[derive(Debug, Clone)]
struct PosEmbedTables {
    u: Tensor,
    v: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PosEmbedKey {
    patch_w: usize,
    patch_h: usize,
    embed_dim: usize,
    aspect_ratio_bits: u64,
    device: DeviceLocation,
}

#[derive(Debug, Default)]
struct PosEmbedCache {
    entries: Mutex<HashMap<PosEmbedKey, PosEmbedTables>>,
}

impl PosEmbedCache {
    fn get_or_compute(
        &self,
        key: PosEmbedKey,
        compute: impl FnOnce() -> CandleResult<PosEmbedTables>,
    ) -> CandleResult<PosEmbedTables> {
        {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(cached) = entries.get(&key) {
                return Ok(cached.clone());
            }
        }
        let computed = compute()?;
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if entries.len() >= POS_EMBED_CACHE_CAP {
            entries.clear();
        }
        entries.insert(key, computed.clone());
        Ok(computed)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len()
    }
}

/// `out_channels=[256, 512, 1024, 1024]`, `intermediate_layer_idx=[4, 11, 17, 23]`,
/// `pos_embed=true`, `down_ratio=1`.
#[derive(Debug)]
pub(crate) struct VggtDptHead {
    norm: LayerNorm,
    projects: Vec<Conv2d>,
    resize_layers: Vec<ResizeLayer>,
    layer_rn: Vec<Conv2d>,
    refinenet1: FeatureFusionBlock,
    refinenet2: FeatureFusionBlock,
    refinenet3: FeatureFusionBlock,
    refinenet4: FeatureFusionBlock,
    output_conv1: Conv2d,
    output_conv2_0: Conv2d,
    output_conv2_2: Conv2d,
    output_dim: usize,
    activation: VggtDptActivation,
    pos_embed_cache: PosEmbedCache,
}

impl VggtDptHead {
    pub(crate) fn load(
        vb: VarBuilder,
        dim_in: usize,
        output_dim: usize,
        activation: VggtDptActivation,
    ) -> CandleResult<Self> {
        let norm = layer_norm(dim_in, 1e-5, vb.pp("norm"))?;

        let projects = DPT_OUT_CHANNELS
            .iter()
            .enumerate()
            .map(|(idx, channels)| {
                conv2d(
                    dim_in,
                    *channels,
                    1,
                    Conv2dConfig::default(),
                    vb.pp("projects").pp(idx.to_string()),
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;

        let resize_layers = vec![
            ResizeLayer::Transpose(conv_transpose2d(
                DPT_OUT_CHANNELS[0],
                DPT_OUT_CHANNELS[0],
                4,
                ConvTranspose2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("0"),
            )?),
            ResizeLayer::Transpose(conv_transpose2d(
                DPT_OUT_CHANNELS[1],
                DPT_OUT_CHANNELS[1],
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("1"),
            )?),
            ResizeLayer::Identity,
            ResizeLayer::Conv(conv2d(
                DPT_OUT_CHANNELS[3],
                DPT_OUT_CHANNELS[3],
                3,
                Conv2dConfig {
                    padding: 1,
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("resize_layers").pp("3"),
            )?),
        ];

        let rn_config = Conv2dConfig {
            padding: 1,
            stride: 1,
            ..Default::default()
        };
        let scratch = vb.pp("scratch");
        let layer_rn = DPT_OUT_CHANNELS
            .iter()
            .enumerate()
            .map(|(idx, channels)| {
                conv2d_no_bias(
                    *channels,
                    DPT_FEATURES,
                    3,
                    rn_config.clone(),
                    scratch.pp(format!("layer{}_rn", idx + 1)),
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;

        let refinenet1 = FeatureFusionBlock::load(scratch.pp("refinenet1"), DPT_FEATURES, true)?;
        let refinenet2 = FeatureFusionBlock::load(scratch.pp("refinenet2"), DPT_FEATURES, true)?;
        let refinenet3 = FeatureFusionBlock::load(scratch.pp("refinenet3"), DPT_FEATURES, true)?;
        let refinenet4 = FeatureFusionBlock::load(scratch.pp("refinenet4"), DPT_FEATURES, false)?;

        let head_features_1 = DPT_FEATURES;
        let head_features_2 = 32;
        let output_conv1 = conv2d(
            head_features_1,
            head_features_1 / 2,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            scratch.pp("output_conv1"),
        )?;
        let conv2_in_channels = head_features_1 / 2;
        let output_conv2_0 = conv2d(
            conv2_in_channels,
            head_features_2,
            3,
            Conv2dConfig {
                padding: 1,
                stride: 1,
                ..Default::default()
            },
            scratch.pp("output_conv2").pp("0"),
        )?;
        let output_conv2_2 = conv2d(
            head_features_2,
            output_dim,
            1,
            Conv2dConfig::default(),
            scratch.pp("output_conv2").pp("2"),
        )?;

        super::drain_upload_staging(vb.device())?;
        Ok(Self {
            norm,
            projects,
            resize_layers,
            layer_rn,
            refinenet1,
            refinenet2,
            refinenet3,
            refinenet4,
            output_conv1,
            output_conv2_0,
            output_conv2_2,
            output_dim,
            activation,
            pos_embed_cache: PosEmbedCache::default(),
        })
    }

    /// Run the head over a chunk of frames.
    ///
    /// `layers` is the 24-slot aggregator output list; each cached entry is
    /// `[B, S, P, 2C]`. Frames `[frame_start, frame_end)` are processed; the
    /// per-frame compute is independent, so chunking is numerically identical.
    ///
    /// Returns `(preds, conf)` with shapes `[B*n, output_dim, H, W]` and
    /// `[B*n, H, W]` where `n = frame_end - frame_start`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_chunk(
        &self,
        layers: &[Option<Tensor>],
        patch_start_idx: usize,
        frame_start: usize,
        frame_end: usize,
        image_height: usize,
        image_width: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let patch_h = image_height / DPT_PATCH_SIZE;
        let patch_w = image_width / DPT_PATCH_SIZE;

        let mut features = Vec::with_capacity(DPT_INTERMEDIATE_LAYERS.len());
        for (dpt_idx, layer_idx) in DPT_INTERMEDIATE_LAYERS.iter().enumerate() {
            let layer = layers.get(*layer_idx).and_then(|entry| entry.as_ref());
            let layer = layer.ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "VGGT DPT head requires cached aggregator layer {layer_idx}"
                ))
            })?;

            // Drop the 5 special tokens (camera + 4 registers) before the
            // [B*S, 1369, 2048] -> [B*S, 2048, 37, 37] reshape.
            let chunk = layer.i((.., frame_start..frame_end, patch_start_idx..))?;
            let (batch, frames, patches, channels) = chunk.dims4()?;
            if patches != patch_h * patch_w {
                candle_core::bail!(
                    "VGGT DPT patch token count {patches} does not match {patch_h}x{patch_w}"
                );
            }
            let x = chunk.reshape((batch * frames, patches, channels))?;
            let x = self.norm.forward(&x)?;
            debug_stats(&format!("stage{dpt_idx}.norm"), &x);
            let x = x
                .permute((0, 2, 1))?
                .reshape((batch * frames, channels, patch_h, patch_w))?;
            let x = self.projects[dpt_idx].forward(&x)?;
            debug_stats(&format!("stage{dpt_idx}.projects"), &x);
            let x = self.apply_pos_embed(&x, image_width, image_height)?;
            debug_stats(&format!("stage{dpt_idx}.pos"), &x);
            let x = self.resize_layers[dpt_idx].forward(&x)?;
            debug_stats(&format!("stage{dpt_idx}.resize"), &x);
            features.push(x);
        }

        let layer1 = &features[0];
        let layer2 = &features[1];
        let layer3 = &features[2];
        let layer4 = &features[3];

        let layer1_rn = self.layer_rn[0].forward(layer1)?;
        debug_stats("layer1_rn", &layer1_rn);
        let layer2_rn = self.layer_rn[1].forward(layer2)?;
        debug_stats("layer2_rn", &layer2_rn);
        let layer3_rn = self.layer_rn[2].forward(layer3)?;
        debug_stats("layer3_rn", &layer3_rn);
        let layer4_rn = self.layer_rn[3].forward(layer4)?;
        debug_stats("layer4_rn", &layer4_rn);

        let (_, _, layer3_h, layer3_w) = layer3_rn.dims4()?;
        let mut out = self
            .refinenet4
            .forward(&layer4_rn, None, Some((layer3_h, layer3_w)))?;
        debug_stats("refinenet4", &out);
        let (_, _, layer2_h, layer2_w) = layer2_rn.dims4()?;
        out = self
            .refinenet3
            .forward(&out, Some(&layer3_rn), Some((layer2_h, layer2_w)))?;
        debug_stats("refinenet3", &out);
        let (_, _, layer1_h, layer1_w) = layer1_rn.dims4()?;
        out = self
            .refinenet2
            .forward(&out, Some(&layer2_rn), Some((layer1_h, layer1_w)))?;
        debug_stats("refinenet2", &out);
        out = self.refinenet1.forward(&out, Some(&layer1_rn), None)?;
        debug_stats("refinenet1", &out);
        out = self.output_conv1.forward(&out)?;
        debug_stats("output_conv1", &out);
        out = out.upsample_bilinear2d(image_height, image_width, true)?;
        debug_stats("final_upsample", &out);
        out = self.apply_pos_embed(&out, image_width, image_height)?;
        debug_stats("final_pos", &out);
        out = self.output_conv2_0.forward(&out)?.relu()?;
        debug_stats("conv2_relu", &out);
        let out = self.output_conv2_2.forward(&out)?;
        debug_stats("conv2_raw", &out);

        let fmap = out.permute((0, 2, 3, 1))?;
        let fmap_channels = fmap.dim(D::Minus1)?;
        debug_assert_eq!(fmap_channels, self.output_dim);
        let xyz = fmap.narrow(D::Minus1, 0, fmap_channels - 1)?;
        let conf = fmap
            .narrow(D::Minus1, fmap_channels - 1, 1)?
            .squeeze(D::Minus1)?;

        let preds = match self.activation {
            // inverse_log_transform: sign(y) * (exp(|y|) - 1)
            VggtDptActivation::InvLog => {
                let magnitude = xyz.abs()?.exp()?.affine(1.0, -1.0)?;
                (xyz.sign()? * magnitude)?
            }
            VggtDptActivation::Exp => xyz.exp()?,
        };
        // expp1: 1 + exp(conf)
        let conf_out = conf.exp()?.affine(1.0, 1.0)?;

        Ok((preds, conf_out))
    }

    /// Run the head over every frame, chunked to bound peak VRAM.
    pub(crate) fn forward(
        &self,
        layers: &[Option<Tensor>],
        patch_start_idx: usize,
        batch: usize,
        frames: usize,
        chunk_size: usize,
        image_height: usize,
        image_width: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let chunk_size = chunk_size.max(1);
        let mut preds = Vec::new();
        let mut confs = Vec::new();
        let mut start = 0usize;
        while start < frames {
            let end = (start + chunk_size).min(frames);
            let (chunk_preds, chunk_conf) = self.forward_chunk(
                layers,
                patch_start_idx,
                start,
                end,
                image_height,
                image_width,
            )?;
            preds.push(chunk_preds);
            confs.push(chunk_conf);
            start = end;
        }
        let preds = Tensor::cat(&preds, 0)?;
        let confs = Tensor::cat(&confs, 0)?;
        let _ = batch;
        Ok((preds, confs))
    }

    fn apply_pos_embed(
        &self,
        xs: &Tensor,
        image_width: usize,
        image_height: usize,
    ) -> CandleResult<Tensor> {
        let (_n, channels, patch_h, patch_w) = xs.dims4()?;
        let tables = self.pos_embed_for(
            patch_w,
            patch_h,
            channels,
            image_width,
            image_height,
            xs.device(),
        )?;
        add_separable_pos_embed(xs, &tables)
    }

    fn pos_embed_for(
        &self,
        patch_w: usize,
        patch_h: usize,
        embed_dim: usize,
        image_width: usize,
        image_height: usize,
        device: &Device,
    ) -> CandleResult<PosEmbedTables> {
        let aspect_ratio = image_width as f64 / image_height as f64;
        let key = PosEmbedKey {
            patch_w,
            patch_h,
            embed_dim,
            aspect_ratio_bits: aspect_ratio.to_bits(),
            device: device.location(),
        };
        self.pos_embed_cache.get_or_compute(key, || {
            compute_pos_tables(patch_w, patch_h, embed_dim, aspect_ratio, device)
        })
    }
}

/// Adds the separable positional embedding without materializing the full
/// `[1, C, H, W]` grid. `tables.u` covers channels `0..C/2` and `tables.v`
/// covers `C/2..C`, matching the reference's channel concatenation.
fn add_separable_pos_embed(xs: &Tensor, tables: &PosEmbedTables) -> CandleResult<Tensor> {
    let channels = xs.dim(1)?;
    if channels % 2 != 0 {
        candle_core::bail!(
            "VGGT DPT positional embedding requires an even channel count, got {channels}"
        );
    }
    let half = channels / 2;
    let out_u = xs.narrow(1, 0, half)?.broadcast_add(&tables.u)?;
    let out_v = xs.narrow(1, half, half)?.broadcast_add(&tables.v)?;
    Tensor::cat(&[&out_u, &out_v], 1)
}

/// `_apply_pos_embed` tables, without the broadcast add.
///
/// Mirrors `create_uv_grid` + `position_grid_to_embed` exactly, but exploits
/// the separability of `indexing="xy"` meshgrids: the `u` embedding depends
/// only on the x column and the `v` embedding only on the y row, so only
/// `(patch_w + patch_h) * embed_dim / 2` sin/cos values are computed instead
/// of `patch_w * patch_h * embed_dim` (68M at 518x518x128).
///
/// All angles are computed in f64 from f32 linspace coordinates, matching the
/// reference's `torch.einsum("m,d->md", pos_f32, omega_f64)` promotion.
fn compute_pos_tables(
    patch_w: usize,
    patch_h: usize,
    embed_dim: usize,
    aspect_ratio: f64,
    device: &Device,
) -> CandleResult<PosEmbedTables> {
    let diag = (aspect_ratio * aspect_ratio + 1.0).sqrt();
    let span_x = aspect_ratio / diag;
    let span_y = 1.0 / diag;
    let left = -span_x * (patch_w as f64 - 1.0) / patch_w as f64;
    let right = span_x * (patch_w as f64 - 1.0) / patch_w as f64;
    let top = -span_y * (patch_h as f64 - 1.0) / patch_h as f64;
    let bottom = span_y * (patch_h as f64 - 1.0) / patch_h as f64;

    let x_coords = linspace_f32(left, right, patch_w);
    let y_coords = linspace_f32(top, bottom, patch_h);

    // Per-axis embedding width (embed_dim // 2), split into sin / cos halves.
    let axis_dim = embed_dim / 2;
    let n_freq = axis_dim / 2;
    let omega = (0..n_freq)
        .map(|d| 1.0 / 100f64.powf(d as f64 / n_freq as f64))
        .collect::<Vec<_>>();

    let emb_u = pos_axis_embed(&x_coords, axis_dim, &omega);
    let emb_v = pos_axis_embed(&y_coords, axis_dim, &omega);

    // ratio = 0.1; then [axis, dim] -> [1, dim, 1, axis] for `u` and
    // [1, dim, axis, 1] for `v`.
    let u = Tensor::from_vec(emb_u, (patch_w, axis_dim), device)?
        .affine(0.1, 0.0)?
        .t()?
        .unsqueeze(0)?
        .unsqueeze(2)?
        .contiguous()?;
    let v = Tensor::from_vec(emb_v, (patch_h, axis_dim), device)?
        .affine(0.1, 0.0)?
        .t()?
        .unsqueeze(0)?
        .unsqueeze(3)?
        .contiguous()?;
    Ok(PosEmbedTables { u, v })
}

fn pos_axis_embed(coords: &[f32], axis_dim: usize, omega: &[f64]) -> Vec<f32> {
    let n_freq = omega.len();
    let mut emb = vec![0f32; coords.len() * axis_dim];
    for (index, pos) in coords.iter().enumerate() {
        let pos = *pos as f64;
        for (d, freq) in omega.iter().enumerate() {
            let angle = pos * freq;
            emb[index * axis_dim + d] = angle.sin() as f32;
            emb[index * axis_dim + n_freq + d] = angle.cos() as f32;
        }
    }
    emb
}

/// `torch.linspace(start, end, steps, dtype=float32)` on CPU: the step is
/// computed in f64 and each value is `start + i * step` rounded to f32.
fn linspace_f32(start: f64, end: f64, steps: usize) -> Vec<f32> {
    if steps <= 1 {
        return vec![start as f32];
    }
    let step = (end - start) / (steps as f64 - 1.0);
    (0..steps)
        .map(|idx| (start + idx as f64 * step) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn full_pos_from_tables(
        tables: &PosEmbedTables,
        patch_h: usize,
        patch_w: usize,
    ) -> CandleResult<Tensor> {
        let dim = tables.u.dim(1)?;
        let u = tables.u.broadcast_as((1, dim, patch_h, patch_w))?;
        let v = tables.v.broadcast_as((1, dim, patch_h, patch_w))?;
        Tensor::cat(&[&u, &v], 1)
    }

    #[test]
    fn separable_pos_embed_matches_channel_concat_reference() {
        let device = Device::Cpu;
        for (patch_w, patch_h) in [(37, 37), (37, 28), (19, 49)] {
            let dim = 16;
            let aspect_ratio = patch_w as f64 / patch_h as f64;
            let tables = compute_pos_tables(patch_w, patch_h, dim, aspect_ratio, &device)
                .expect("pos tables");
            assert_eq!(tables.u.dims(), &[1, dim / 2, 1, patch_w]);
            assert_eq!(tables.v.dims(), &[1, dim / 2, patch_h, 1]);

            let xs = Tensor::randn(0f32, 1f32, (1, dim, patch_h, patch_w), &device)
                .expect("random activations");
            let merged = add_separable_pos_embed(&xs, &tables).expect("separable add");
            let full = full_pos_from_tables(&tables, patch_h, patch_w).expect("full grid");
            let reference = xs.broadcast_add(&full).expect("reference add");
            let delta = merged
                .sub(&reference)
                .expect("delta")
                .abs()
                .expect("abs")
                .flatten_all()
                .expect("flatten")
                .max(0)
                .expect("max")
                .to_scalar::<f32>()
                .expect("scalar");
            assert!(
                delta < 1e-6,
                "separable pos embed mismatch for {patch_w}x{patch_h}: {delta:e}"
            );
        }
    }

    #[test]
    fn pos_embed_cache_is_bounded() {
        let cache = PosEmbedCache::default();
        let device = Device::Cpu;
        for idx in 0..POS_EMBED_CACHE_CAP + 4 {
            let key = PosEmbedKey {
                patch_w: idx + 1,
                patch_h: idx + 2,
                embed_dim: 4,
                aspect_ratio_bits: idx as u64,
                device: device.location(),
            };
            cache
                .get_or_compute(key, || {
                    Ok(PosEmbedTables {
                        u: Tensor::zeros((1, 2, 1, idx + 1), DType::F32, &device)?,
                        v: Tensor::zeros((1, 2, idx + 2, 1), DType::F32, &device)?,
                    })
                })
                .expect("cache insert");
        }
        assert!(
            cache.len() <= POS_EMBED_CACHE_CAP,
            "pos-embed cache exceeded cap: {}",
            cache.len()
        );
        assert!(cache.len() > 0, "pos-embed cache should retain an entry");
    }
}

/// Opt-in (`LUX3D_VGGT_DPT_DEBUG=1`) stage statistics for parity debugging.
fn debug_stats(name: &str, tensor: &Tensor) {
    if std::env::var_os("LUX3D_VGGT_DPT_DEBUG").is_none() {
        return;
    }
    let Ok(flat) = tensor.flatten_all() else {
        return;
    };
    let min = flat
        .min(0)
        .and_then(|v| v.to_scalar::<f32>())
        .unwrap_or(f32::NAN);
    let max = flat
        .max(0)
        .and_then(|v| v.to_scalar::<f32>())
        .unwrap_or(f32::NAN);
    let mean = flat
        .mean_all()
        .and_then(|v| v.to_scalar::<f32>())
        .unwrap_or(f32::NAN);
    eprintln!("[dpt-diag] {name}: min={min:.6e} max={max:.6e} mean={mean:.6e}");
}
