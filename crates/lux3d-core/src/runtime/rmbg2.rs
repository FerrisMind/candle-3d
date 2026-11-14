//! RMBG-2.0 (BiRefNet with a Swin-L backbone) in candle.
//!
//! This is a direct port of the reference `models/RMBG-2.0/birefnet.py` used by
//! TRELLIS.2 for RGB inputs that do not carry a usable alpha channel. The
//! architecture constants below mirror `Config` in that file:
//!
//! * `mul_scl_ipt = "cat"`: run the backbone on the full and half-size image
//!   and concatenate the upsampled half features to the full ones.
//! * `cxt_num = 3`: inject the three finer encoder levels into stage 4.
//! * `squeeze_block = "BasicDecBlk_x1"`: 5760 -> 3072 channels.
//! * `dec_att = "ASPPDeformable"`, `dec_ipt_split = True`, `ms_supervision`.
//!
//! Deformable convolutions are expressed with regular tensor operations
//! (bilinear gather + 1x1 convolutions) so the model runs on every candle
//! backend, including Vulkan/WGPU where torchvision-style kernels do not exist.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, ModuleT,
    VarBuilder, batch_norm, conv2d, conv2d_no_bias, layer_norm,
    ops::{sigmoid, softmax_last_dim},
};
use image::{RgbImage, RgbaImage};

use crate::preprocess::{
    TRELLIS2_IMAGE_NET_MEAN, TRELLIS2_IMAGE_NET_STD, Trellis2BackgroundRemover, resize_bicubic_u8,
    resize_bilinear_u8,
};
use crate::runtime::nn_blocks::{Linear as LuxLinear, linear, linear_no_bias_dense};
use crate::weights::load_auxiliary_canonical_weights;

const WINDOW_SIZE: usize = 12;
const PATCH_SIZE: usize = 4;
const EMBED_DIM: usize = 192;
const DEPTHS: [usize; 4] = [2, 2, 18, 2];
const HEADS: [usize; 4] = [6, 12, 24, 48];

#[derive(Debug, Clone)]
pub struct Rmbg2Config;

impl Default for Rmbg2Config {
    fn default() -> Self {
        Self
    }
}

#[derive(Debug)]
pub struct Rmbg2Outputs {
    /// Multi-scale supervision heads at 1/4, 1/8, and 1/16 resolution.
    pub m4: Tensor,
    pub m3: Tensor,
    pub m2: Tensor,
    /// Final single-channel logits at the input resolution.
    pub logits: Tensor,
}

impl Rmbg2Outputs {
    /// The reference wrapper consumes `model(input)[-1].sigmoid()`.
    pub fn sigmoid(&self) -> Result<Tensor> {
        sigmoid(&self.logits)
    }
}

#[derive(Debug)]
pub struct Rmbg2Model {
    bb: SwinTransformer,
    squeeze: BasicDecBlk,
    decoder: Decoder,
    device: Device,
    dtype: DType,
}

impl Rmbg2Model {
    pub fn load(cfg: &Rmbg2Config, vb: VarBuilder) -> Result<Self> {
        let _ = cfg;
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let bb = SwinTransformer::load(vb.pp("bb"))?;
        let squeeze = BasicDecBlk::load(5760, 3072, vb.pp("squeeze_module").pp("0"))?;
        let decoder = Decoder::load(&[3072, 1536, 768, 384], vb.pp("decoder"))?;
        Ok(Self {
            bb,
            squeeze,
            decoder,
            device,
            dtype,
        })
    }

    /// Load RMBG-2.0 from a canonical package directory, a standalone
    /// `model.gguf`, a raw `models/RMBG-2.0` directory, or a raw
    /// `model.safetensors` file.
    pub fn load_from_path(path: &std::path::Path, device: &Device) -> crate::Result<Self> {
        let wguf = path.is_file()
            && path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"));
        let has_manifest = path.is_dir() && path.join("manifest.json").is_file();
        let has_gguf = path.is_dir() && path.join("model.gguf").is_file();
        if wguf || has_manifest || (has_gguf && !path.join("model.safetensors").is_file()) {
            let paths = load_auxiliary_canonical_weights(path, "rmbg2")?;
            let vb = unsafe { paths.var_builder(DType::F32, device) }.map_err(|source| {
                crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to open RMBG-2.0 package `{}`: {source}",
                        path.display()
                    ),
                }
            })?;
            return Self::load(&Rmbg2Config, vb).map_err(|source| {
                crate::error::Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load RMBG-2.0: {source}"),
                }
            });
        }
        let weights = if path.is_file() {
            path.to_path_buf()
        } else {
            let candidate = path.join("model.safetensors");
            if !candidate.is_file() {
                return Err(crate::error::Lux3dError::MissingWeightFile { path: candidate });
            }
            candidate
        };
        // SAFETY: the mmap-backed builder's tensors are owned by the returned
        // model; the file is read-only and outlives the load call.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(std::slice::from_ref(&weights), DType::F32, device)
        }
        .map_err(
            |source| crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to open raw RMBG-2.0 weights `{}`: {source}",
                    weights.display()
                ),
            },
        )?;
        Self::load(&Rmbg2Config, vb).map_err(|source| {
            crate::error::Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load RMBG-2.0: {source}"),
            }
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Exact reference wrapper: BILINEAR resize to 1024, ImageNet
    /// normalization, sigmoid, truncating uint8 mask, BICUBIC resize back.
    pub fn alpha_mask(&self, rgb: &RgbImage) -> Result<RgbaImage> {
        let (width, height) = rgb.dimensions();
        let resized =
            resize_bilinear_u8::<3>(rgb.as_raw(), width as usize, height as usize, 1024, 1024);
        let mut values = Vec::with_capacity(3 * 1024 * 1024);
        for channel in 0..3 {
            for index in 0..1024 * 1024 {
                let v = f32::from(resized[index * 3 + channel]);
                values.push(
                    (v / 255.0 - TRELLIS2_IMAGE_NET_MEAN[channel])
                        / TRELLIS2_IMAGE_NET_STD[channel],
                );
            }
        }
        let input =
            Tensor::from_vec(values, (1, 3, 1024, 1024), &self.device)?.to_dtype(self.dtype)?;
        let logits = self.forward(&input)?;
        let pred = candle_nn::ops::sigmoid(&logits)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?;
        let pred = pred.reshape((1024, 1024))?.to_vec2::<f32>()?;
        let mut mask = vec![0_u8; 1024 * 1024];
        for (dst, row) in mask.chunks_exact_mut(1024).zip(pred.iter()) {
            for (dst, &value) in dst.iter_mut().zip(row.iter()) {
                // torchvision `ToPILImage`: mul(255).clamp(0,255).byte().
                *dst = (value * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
        let mask = resize_bicubic_u8::<1>(&mask, 1024, 1024, width as usize, height as usize);
        let mut rgba = vec![0_u8; width as usize * height as usize * 4];
        for (index, pixel) in rgb.pixels().enumerate() {
            rgba[index * 4] = pixel[0];
            rgba[index * 4 + 1] = pixel[1];
            rgba[index * 4 + 2] = pixel[2];
            rgba[index * 4 + 3] = mask[index];
        }
        RgbaImage::from_raw(width, height, rgba)
            .ok_or_else(|| candle_core::Error::Msg("RMBG-2.0 mask buffer mismatch".into()))
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(self.forward_outputs(input)?.logits)
    }

    pub fn forward_outputs(&self, input: &Tensor) -> Result<Rmbg2Outputs> {
        let (enc, cxt) = self.forward_enc(input)?;
        let x4 = self.squeeze.forward(&cxt)?;
        let out = self.decoder.forward(input, &enc, x4)?;
        Ok(Rmbg2Outputs {
            m4: out.m4,
            m3: out.m3,
            m2: out.m2,
            logits: out.logits,
        })
    }

    /// Encoder half of `BiRefNet.forward_enc`, including the multi-scale
    /// concatenation used by `mul_scl_ipt == "cat"` and the context tensor.
    pub(crate) fn forward_enc(&self, input: &Tensor) -> Result<([Tensor; 4], Tensor)> {
        let (b, _, h, w) = input.dims4()?;
        let full = self.bb.forward(input)?;
        let half = self
            .bb
            .forward(&input.upsample_bilinear2d(h / 2, w / 2, true)?)?;

        let enc = [
            concat_mul_scale(&full[0], &half[0])?,
            concat_mul_scale(&full[1], &half[1])?,
            concat_mul_scale(&full[2], &half[2])?,
            concat_mul_scale(&full[3], &half[3])?,
        ];
        let (c4_h, c4_w) = (enc[3].dim(2)?, enc[3].dim(3)?);
        let cxt = Tensor::cat(
            &[
                enc[0].upsample_bilinear2d(c4_h, c4_w, true)?,
                enc[1].upsample_bilinear2d(c4_h, c4_w, true)?,
                enc[2].upsample_bilinear2d(c4_h, c4_w, true)?,
                enc[3].clone(),
            ],
            1,
        )?;
        let _ = b;
        Ok((enc, cxt))
    }
}

fn concat_mul_scale(full: &Tensor, half: &Tensor) -> Result<Tensor> {
    let (h, w) = (full.dim(2)?, full.dim(3)?);
    Tensor::cat(&[full, &half.upsample_bilinear2d(h, w, true)?], 1)
}

// ---------------------------------------------------------------------------
// Swin-L backbone
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct SwinTransformer {
    patch_embed: PatchEmbed,
    layers: Vec<BasicLayer>,
    norms: Vec<LayerNorm>,
}

impl SwinTransformer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let patch_embed = PatchEmbed::load(vb.pp("patch_embed"))?;
        let mut layers = Vec::with_capacity(4);
        for (i, (&depth, &heads)) in DEPTHS.iter().zip(HEADS.iter()).enumerate() {
            layers.push(BasicLayer::load(
                EMBED_DIM << i,
                depth,
                heads,
                i + 1 < DEPTHS.len(),
                vb.pp("layers").pp(i),
            )?);
        }
        let mut norms = Vec::with_capacity(4);
        for i in 0..4 {
            norms.push(layer_norm(
                EMBED_DIM << i,
                LayerNormConfig::default(),
                vb.pp(format!("norm{i}")),
            )?);
        }
        Ok(Self {
            patch_embed,
            layers,
            norms,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<[Tensor; 4]> {
        let (mut x, mut h, mut w) = self.patch_embed.forward(input)?;
        let mut outs: Vec<Tensor> = Vec::with_capacity(4);
        for (i, layer) in self.layers.iter().enumerate() {
            let (x_out, out_h, out_w, x_down, down_h, down_w) = layer.forward(&x, h, w)?;
            let n = self.norms[i].forward(&x_out)?;
            let b = n.dim(0)?;
            let c = n.dim(2)?;
            outs.push(
                n.reshape((b, out_h, out_w, c))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?,
            );
            x = x_down;
            h = down_h;
            w = down_w;
        }
        Ok([
            outs.remove(0),
            outs.remove(0),
            outs.remove(0),
            outs.remove(0),
        ])
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: Conv2d,
    norm: LayerNorm,
}

impl PatchEmbed {
    fn load(vb: VarBuilder) -> Result<Self> {
        let proj = conv2d(
            3,
            EMBED_DIM,
            PATCH_SIZE,
            Conv2dConfig {
                stride: PATCH_SIZE,
                ..Default::default()
            },
            vb.pp("proj"),
        )?;
        let norm = layer_norm(EMBED_DIM, LayerNormConfig::default(), vb.pp("norm"))?;
        Ok(Self { proj, norm })
    }

    fn forward(&self, input: &Tensor) -> Result<(Tensor, usize, usize)> {
        let (_, _, h, w) = input.dims4()?;
        let mut x = input.clone();
        if w % PATCH_SIZE != 0 {
            x = x.pad_with_zeros(3, 0, PATCH_SIZE - w % PATCH_SIZE)?;
        }
        if h % PATCH_SIZE != 0 {
            x = x.pad_with_zeros(2, 0, PATCH_SIZE - h % PATCH_SIZE)?;
        }
        let x = self.proj.forward(&x)?;
        let (_, c, ph, pw) = x.dims4()?;
        let x = x.reshape((x.dim(0)?, c, ph * pw))?.transpose(1, 2)?;
        let x = self.norm.forward(&x)?;
        Ok((x, ph, pw))
    }
}

#[derive(Debug)]
struct BasicLayer {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
    window_size: usize,
    shift_size: usize,
}

impl BasicLayer {
    fn load(
        dim: usize,
        depth: usize,
        heads: usize,
        downsample: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let shift = if i % 2 == 0 { 0 } else { WINDOW_SIZE / 2 };
            blocks.push(SwinBlock::load(dim, heads, shift, vb.pp("blocks").pp(i))?);
        }
        let downsample = if downsample {
            Some(PatchMerging::load(dim, vb.pp("downsample"))?)
        } else {
            None
        };
        Ok(Self {
            blocks,
            downsample,
            window_size: WINDOW_SIZE,
            shift_size: WINDOW_SIZE / 2,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, usize, usize, Tensor, usize, usize)> {
        let mask = swin_mask(h, w, self.window_size, self.shift_size, x.device())?;
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(
                &x,
                h,
                w,
                if block.shift_size > 0 {
                    Some(&mask)
                } else {
                    None
                },
            )?;
        }
        let x_out = x.clone();
        match &self.downsample {
            Some(down) => {
                let x_down = down.forward(&x, h, w)?;
                Ok((x_out, h, w, x_down, h.div_ceil(2), w.div_ceil(2)))
            }
            None => Ok((x_out, h, w, x, h, w)),
        }
    }
}

#[derive(Debug)]
struct SwinBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    window_size: usize,
    shift_size: usize,
}

impl SwinBlock {
    fn load(dim: usize, heads: usize, shift_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm1"))?,
            attn: WindowAttention::load(dim, heads, vb.pp("attn"))?,
            norm2: layer_norm(dim, LayerNormConfig::default(), vb.pp("norm2"))?,
            mlp: Mlp::load(dim, dim * 4, vb.pp("mlp"))?,
            window_size: WINDOW_SIZE,
            shift_size,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        assert_eq!(l, h * w, "swin token count does not match H*W");
        let shortcut = x.clone();
        let y = self.norm1.forward(x)?.reshape((b, h, w, c))?;

        let pad_r = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b = (self.window_size - h % self.window_size) % self.window_size;
        let mut y = if pad_r > 0 {
            y.pad_with_zeros(2, 0, pad_r)?
        } else {
            y
        };
        if pad_b > 0 {
            y = y.pad_with_zeros(1, 0, pad_b)?;
        }
        let hp = h + pad_b;
        let wp = w + pad_r;
        if self.shift_size > 0 {
            y = y
                .roll(-(self.shift_size as i32), 1)?
                .roll(-(self.shift_size as i32), 2)?;
        }
        let windows = window_partition(&y, self.window_size)?;
        let windows = windows.reshape((windows.dim(0)?, self.window_size * self.window_size, c))?;
        let attn = self.attn.forward(&windows, mask)?;
        let attn = attn.reshape((attn.dim(0)?, self.window_size, self.window_size, c))?;
        let shifted = window_reverse(&attn, self.window_size, b, hp, wp)?;
        let mut y = if self.shift_size > 0 {
            shifted
                .roll(self.shift_size as i32, 1)?
                .roll(self.shift_size as i32, 2)?
        } else {
            shifted
        };
        if pad_r > 0 || pad_b > 0 {
            y = y.narrow(1, 0, h)?.narrow(2, 0, w)?.contiguous()?;
        }
        let y = y.reshape((b, h * w, c))?;
        let x = shortcut.add(&y)?;
        let mlp = self.mlp.forward(&self.norm2.forward(&x)?)?;
        x.add(&mlp)
    }
}

#[derive(Debug)]
struct WindowAttention {
    qkv: LuxLinear,
    proj: LuxLinear,
    table: Tensor,
    index: Tensor,
    heads: usize,
    scale: f64,
}

impl WindowAttention {
    fn load(dim: usize, heads: usize, vb: VarBuilder) -> Result<Self> {
        let table = vb.get(
            ((2 * WINDOW_SIZE - 1) * (2 * WINDOW_SIZE - 1), heads),
            "relative_position_bias_table",
        )?;
        let index = relative_position_index(WINDOW_SIZE, table.device())?;
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            table,
            index,
            heads,
            scale: ((dim / heads) as f64).powf(-0.5),
        })
    }

    fn relative_bias(&self) -> Result<Tensor> {
        let n = WINDOW_SIZE * WINDOW_SIZE;
        self.table
            .index_select(&self.index, 0)?
            .reshape((n, n, self.heads))?
            .permute((2, 0, 1))?
            .unsqueeze(0)
    }

    fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (b_, n, c) = x.dims3()?;
        let qkv = self
            .qkv
            .forward(x)?
            .reshape((b_, n, 3, self.heads, c / self.heads))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.get(0)?.affine(self.scale, 0.0)?.contiguous()?;
        let k = qkv.get(1)?.contiguous()?;
        let v = qkv.get(2)?.contiguous()?;
        let mut attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        attn = attn.broadcast_add(&self.relative_bias()?)?;
        if let Some(mask) = mask {
            let nw = mask.dim(0)?;
            attn = attn
                .reshape((b_ / nw, nw, self.heads, n, n))?
                .broadcast_add(&mask.unsqueeze(1)?.unsqueeze(0)?)?
                .reshape((b_, self.heads, n, n))?;
        }
        let attn = softmax_last_dim(&attn)?;
        let out = attn
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b_, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct Mlp {
    fc1: LuxLinear,
    fc2: LuxLinear,
}

impl Mlp {
    fn load(dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(vb.pp("fc1"), dim, hidden, true)?,
            fc2: linear(vb.pp("fc2"), hidden, dim, true)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.fc1.forward(x)?.gelu_erf()?)
    }
}

#[derive(Debug)]
struct PatchMerging {
    norm: LayerNorm,
    reduction: LuxLinear,
    dim: usize,
}

impl PatchMerging {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm: layer_norm(4 * dim, LayerNormConfig::default(), vb.pp("norm"))?,
            reduction: linear_no_bias_dense(4 * dim, 2 * dim, vb.pp("reduction"))?,
            dim,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        assert_eq!(l, h * w, "patch merging token count does not match H*W");
        assert_eq!(c, self.dim);
        let mut x = x.reshape((b, h, w, c))?;
        if w % 2 == 1 {
            x = x.pad_with_zeros(2, 0, 1)?;
        }
        if h % 2 == 1 {
            x = x.pad_with_zeros(1, 0, 1)?;
        }
        let hp = h.div_ceil(2);
        let wp = w.div_ceil(2);
        let x = x.reshape((b, hp, 2, wp, 2, c))?;
        let x0 = x.narrow(2, 0, 1)?.narrow(4, 0, 1)?;
        let x1 = x.narrow(2, 1, 1)?.narrow(4, 0, 1)?;
        let x2 = x.narrow(2, 0, 1)?.narrow(4, 1, 1)?;
        let x3 = x.narrow(2, 1, 1)?.narrow(4, 1, 1)?;
        let x = Tensor::cat(&[&x0, &x1, &x2, &x3], 5)?.reshape((b, hp * wp, 4 * c))?;
        let x = self.norm.forward(&x)?;
        self.reduction.forward(&x)
    }
}

fn window_partition(x: &Tensor, window: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    x.reshape((b, h / window, window, w / window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * (h / window) * (w / window), window, window, c))
}

fn window_reverse(windows: &Tensor, window: usize, b: usize, h: usize, w: usize) -> Result<Tensor> {
    let c = windows.dim(3)?;
    windows
        .reshape((b, h / window, w / window, window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, h, w, c))
}

fn relative_position_index(window: usize, device: &candle_core::Device) -> Result<Tensor> {
    let n = window * window;
    let mut index = vec![0u32; n * n];
    for i in 0..n {
        let ih = i / window;
        let iw = i % window;
        for j in 0..n {
            let jh = j / window;
            let jw = j % window;
            let rh = (ih as i64 - jh as i64 + window as i64 - 1) as u32;
            let rw = (iw as i64 - jw as i64 + window as i64 - 1) as u32;
            index[i * n + j] = rh * (2 * window as u32 - 1) + rw;
        }
    }
    Tensor::from_vec(index, (n * n,), device)
}

fn swin_mask(
    h: usize,
    w: usize,
    window: usize,
    shift: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let hp = h.div_ceil(window) * window;
    let wp = w.div_ceil(window) * window;
    let mut img = vec![0f32; hp * wp];
    let h_slices = [
        py_slice(hp, 0, Some(-(window as isize))),
        py_slice(hp, -(window as isize), Some(-(shift as isize))),
        py_slice(hp, -(shift as isize), None),
    ];
    let w_slices = [
        py_slice(wp, 0, Some(-(window as isize))),
        py_slice(wp, -(window as isize), Some(-(shift as isize))),
        py_slice(wp, -(shift as isize), None),
    ];
    let mut cnt = 0f32;
    for (h0, h1) in h_slices {
        for (w0, w1) in w_slices {
            for y in h0..h1 {
                for x in w0..w1 {
                    img[y * wp + x] = cnt;
                }
            }
            cnt += 1.0;
        }
    }
    let nh = hp / window;
    let nw = wp / window;
    let n = window * window;
    let mut mask = vec![0f32; nh * nw * n * n];
    for wy in 0..nh {
        for wx in 0..nw {
            let wi = wy * nw + wx;
            for i in 0..n {
                let iy = wy * window + i / window;
                let ix = wx * window + i % window;
                for j in 0..n {
                    let jy = wy * window + j / window;
                    let jx = wx * window + j % window;
                    let same = img[iy * wp + ix] == img[jy * wp + jx];
                    mask[wi * n * n + i * n + j] = if same { 0.0 } else { -100.0 };
                }
            }
        }
    }
    Tensor::from_vec(mask, (nh * nw, n, n), device)
}

fn py_slice(len: usize, start: isize, end: Option<isize>) -> (usize, usize) {
    let norm = |v: isize| -> usize {
        if v < 0 {
            (len as isize + v).max(0) as usize
        } else {
            (v as usize).min(len)
        }
    };
    let start = norm(start);
    let end = end.map(norm).unwrap_or(len);
    (start, end.max(start))
}

// ---------------------------------------------------------------------------
// Deformable convolution
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct DeformConv {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    weight: Tensor,
    bias: Option<Tensor>,
    kernel: usize,
    padding: usize,
    stride: usize,
}

impl DeformConv {
    fn load(
        deform_vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding,
            stride: 1,
            ..Default::default()
        };
        let offset_conv = conv2d(
            in_c,
            2 * kernel * kernel,
            kernel,
            cfg,
            deform_vb.pp("offset_conv"),
        )?;
        let modulator_conv = conv2d(
            in_c,
            kernel * kernel,
            kernel,
            cfg,
            deform_vb.pp("modulator_conv"),
        )?;
        let reg_vb = deform_vb.pp("regular_conv");
        let weight = reg_vb.get((out_c, in_c, kernel, kernel), "weight")?;
        let bias = if bias {
            Some(reg_vb.get(out_c, "bias")?)
        } else {
            None
        };
        Ok(Self {
            offset_conv,
            modulator_conv,
            weight,
            bias,
            kernel,
            padding,
            stride: 1,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, ih, iw) = x.dims4()?;
        let offset = self.offset_conv.forward(x)?;
        let mask = sigmoid(&self.modulator_conv.forward(x)?)?.affine(2.0, 0.0)?;
        let (_, _, oh, ow) = offset.dims4()?;
        let n = oh * ow;
        let k2 = self.kernel * self.kernel;

        let x_flat = x.reshape((b, c, ih * iw))?;
        let mut out: Option<Tensor> = None;

        for ki in 0..k2 {
            let kh = ki / self.kernel;
            let kw = ki % self.kernel;
            let y_base = base_coords(
                oh,
                ow,
                self.stride,
                self.padding,
                kh,
                true,
                x.device(),
                x.dtype(),
            )?;
            let x_base = base_coords(
                oh,
                ow,
                self.stride,
                self.padding,
                kw,
                false,
                x.device(),
                x.dtype(),
            )?;
            let dy = offset
                .narrow(1, 2 * ki, 1)?
                .narrow(2, 0, oh)?
                .narrow(3, 0, ow)?
                .reshape((b, n))?;
            let dx = offset
                .narrow(1, 2 * ki + 1, 1)?
                .narrow(2, 0, oh)?
                .narrow(3, 0, ow)?
                .reshape((b, n))?;
            let y = dy.broadcast_add(&y_base)?;
            let xc = dx.broadcast_add(&x_base)?;
            let (iy0, iy1, wy0, wy1) = bilinear_axis(y.as_ref(), ih)?;
            let (ix0, ix1, wx0, wx1) = bilinear_axis(xc.as_ref(), iw)?;

            // Index ops stay in F32 until the final cast: Vulkan does not
            // implement `affine` for U32 tensors.
            let i00 = flat_index(&iy0, &ix0, iw)?;
            let i01 = flat_index(&iy0, &ix1, iw)?;
            let i10 = flat_index(&iy1, &ix0, iw)?;
            let i11 = flat_index(&iy1, &ix1, iw)?;

            let v00 = gather_flat(&x_flat, &i00, b, c, n)?.reshape((b, c, oh, ow))?;
            let v01 = gather_flat(&x_flat, &i01, b, c, n)?.reshape((b, c, oh, ow))?;
            let v10 = gather_flat(&x_flat, &i10, b, c, n)?.reshape((b, c, oh, ow))?;
            let v11 = gather_flat(&x_flat, &i11, b, c, n)?.reshape((b, c, oh, ow))?;

            let weight_shape = (b, 1, oh, ow);
            let w00 = wy0.broadcast_mul(&wx0)?.reshape(weight_shape)?;
            let w01 = wy0.broadcast_mul(&wx1)?.reshape(weight_shape)?;
            let w10 = wy1.broadcast_mul(&wx0)?.reshape(weight_shape)?;
            let w11 = wy1.broadcast_mul(&wx1)?.reshape(weight_shape)?;
            let sampled = v00
                .broadcast_mul(&w00)?
                .add(&v01.broadcast_mul(&w01)?)?
                .add(&v10.broadcast_mul(&w10)?)?
                .add(&v11.broadcast_mul(&w11)?)?;
            let sampled = sampled.broadcast_mul(&mask.narrow(1, ki, 1)?)?;

            let weight_k = self
                .weight
                .narrow(2, kh, 1)?
                .narrow(3, kw, 1)?
                .contiguous()?;
            let term = sampled.conv2d(&weight_k, 0, 1, 1, 1)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let mut out = out.expect("deformable convolution kernel is non-empty");
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.reshape((1, bias.dim(0)?, 1, 1))?)?;
        }
        Ok(out)
    }
}

#[derive(Debug)]
struct DeformBranch {
    deform: DeformConv,
    bn: BatchNorm,
}

impl DeformBranch {
    fn load(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Self> {
        Ok(Self {
            deform: DeformConv::load(vb.pp("atrous_conv"), in_c, out_c, kernel, padding, bias)?,
            bn: batch_norm(out_c, BatchNormConfig::default(), vb.pp("bn"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.bn.forward_t(&self.deform.forward(x)?, false)?.relu()
    }
}

#[allow(clippy::too_many_arguments)]
fn base_coords(
    oh: usize,
    ow: usize,
    stride: usize,
    padding: usize,
    kernel_pos: usize,
    is_y: bool,
    device: &candle_core::Device,
    dtype: DType,
) -> Result<Tensor> {
    let mut v = Vec::with_capacity(oh * ow);
    for oy in 0..oh {
        for ox in 0..ow {
            let base = if is_y { oy * stride } else { ox * stride };
            v.push(base as f32 - padding as f32 + kernel_pos as f32);
        }
    }
    Tensor::from_vec(v, (1, oh * ow), device)?.to_dtype(dtype)
}

/// Returns integer low/high indices and bilinear weights along one axis.
/// The base convolution padding is applied by subtracting it from sampled
/// coordinates before calling this helper.
fn bilinear_axis(coord: &Tensor, len: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    let low = coord.floor()?;
    let frac = coord.sub(&low)?;
    let max = (len.saturating_sub(1)) as f64;
    let low_idx = low.clamp(0.0, max)?.to_dtype(DType::U32)?;
    let high_idx = low
        .affine(1.0, 1.0)?
        .clamp(0.0, max)?
        .to_dtype(DType::U32)?;
    let low_valid = in_range(&low, len)?;
    let high_valid = in_range(&low.affine(1.0, 1.0)?, len)?;
    let w_low = frac.affine(-1.0, 1.0)?.broadcast_mul(&low_valid)?;
    let w_high = frac.broadcast_mul(&high_valid)?;
    Ok((low_idx, high_idx, w_low, w_high))
}

fn flat_index(iy: &Tensor, ix: &Tensor, width: usize) -> Result<Tensor> {
    let row = iy.to_dtype(DType::F32)?.affine(width as f64, 0.0)?;
    row.add(&ix.to_dtype(DType::F32)?)?.to_dtype(DType::U32)
}

fn gather_flat(x_flat: &Tensor, index: &Tensor, b: usize, c: usize, n: usize) -> Result<Tensor> {
    let index = index
        .reshape((b, 1, n))?
        .broadcast_as((b, c, n))?
        .contiguous()?;
    x_flat.gather(&index, 2)
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

#[allow(dead_code)] // p* heads are multi-scale supervision outputs used by the trace tests.
struct DecoderOutputs {
    p4: Tensor,
    p3: Tensor,
    p2: Tensor,
    p1: Tensor,
    m4: Tensor,
    m3: Tensor,
    m2: Tensor,
    logits: Tensor,
}

#[derive(Debug)]
struct Decoder {
    ipt_blk5: SimpleConvs,
    ipt_blk4: SimpleConvs,
    ipt_blk3: SimpleConvs,
    ipt_blk2: SimpleConvs,
    ipt_blk1: SimpleConvs,
    decoder_block4: BasicDecBlk,
    decoder_block3: BasicDecBlk,
    decoder_block2: BasicDecBlk,
    decoder_block1: BasicDecBlk,
    conv_out1: Conv2d,
    lateral_block4: Conv2d,
    lateral_block3: Conv2d,
    lateral_block2: Conv2d,
    conv_ms_spvn_4: Conv2d,
    conv_ms_spvn_3: Conv2d,
    conv_ms_spvn_2: Conv2d,
    gdt_convs_4: GdtConv,
    gdt_convs_3: GdtConv,
    gdt_convs_2: GdtConv,
    gdt_attn_4: Conv2d,
    gdt_attn_3: Conv2d,
    gdt_attn_2: Conv2d,
}

impl Decoder {
    fn load(channels: &[usize; 4], vb: VarBuilder) -> Result<Self> {
        let out5 = channels[0] / 8;
        let out4 = channels[0] / 8;
        let out3 = channels[1] / 8;
        let out2 = channels[2] / 8;
        let out1 = channels[3] / 8;
        let conv1x1 = |in_c: usize, out_c: usize, vb: VarBuilder| {
            conv2d(in_c, out_c, 1, Conv2dConfig::default(), vb)
        };
        Ok(Self {
            ipt_blk5: SimpleConvs::load(3072, out5, vb.pp("ipt_blk5"))?,
            ipt_blk4: SimpleConvs::load(768, out4, vb.pp("ipt_blk4"))?,
            ipt_blk3: SimpleConvs::load(192, out3, vb.pp("ipt_blk3"))?,
            ipt_blk2: SimpleConvs::load(48, out2, vb.pp("ipt_blk2"))?,
            ipt_blk1: SimpleConvs::load(3, out1, vb.pp("ipt_blk1"))?,
            decoder_block4: BasicDecBlk::load(
                channels[0] + out5,
                channels[1],
                vb.pp("decoder_block4"),
            )?,
            decoder_block3: BasicDecBlk::load(
                channels[1] + out4,
                channels[2],
                vb.pp("decoder_block3"),
            )?,
            decoder_block2: BasicDecBlk::load(
                channels[2] + out3,
                channels[3],
                vb.pp("decoder_block2"),
            )?,
            decoder_block1: BasicDecBlk::load(
                channels[3] + out2,
                channels[3] / 2,
                vb.pp("decoder_block1"),
            )?,
            conv_out1: conv2d(
                channels[3] / 2 + out1,
                1,
                1,
                Conv2dConfig::default(),
                vb.pp("conv_out1").pp("0"),
            )?,
            lateral_block4: conv1x1(channels[1], channels[1], vb.pp("lateral_block4").pp("conv"))?,
            lateral_block3: conv1x1(channels[2], channels[2], vb.pp("lateral_block3").pp("conv"))?,
            lateral_block2: conv1x1(channels[3], channels[3], vb.pp("lateral_block2").pp("conv"))?,
            conv_ms_spvn_4: conv1x1(channels[1], 1, vb.pp("conv_ms_spvn_4"))?,
            conv_ms_spvn_3: conv1x1(channels[2], 1, vb.pp("conv_ms_spvn_3"))?,
            conv_ms_spvn_2: conv1x1(channels[3], 1, vb.pp("conv_ms_spvn_2"))?,
            gdt_convs_4: GdtConv::load(channels[1], vb.pp("gdt_convs_4"))?,
            gdt_convs_3: GdtConv::load(channels[2], vb.pp("gdt_convs_3"))?,
            gdt_convs_2: GdtConv::load(channels[3], vb.pp("gdt_convs_2"))?,
            gdt_attn_4: conv1x1(16, 1, vb.pp("gdt_convs_attn_4").pp("0"))?,
            gdt_attn_3: conv1x1(16, 1, vb.pp("gdt_convs_attn_3").pp("0"))?,
            gdt_attn_2: conv1x1(16, 1, vb.pp("gdt_convs_attn_2").pp("0"))?,
        })
    }

    fn forward(&self, image: &Tensor, enc: &[Tensor; 4], mut x4: Tensor) -> Result<DecoderOutputs> {
        let (x1, x2, x3, _) = (&enc[0], &enc[1], &enc[2], &enc[3]);
        let ipt5 = self.ipt_blk5.forward(&patches_batch(image, &x4)?)?;
        x4 = Tensor::cat(&[x4, ipt5], 1)?;
        let p4 = self.decoder_block4.forward(&x4)?;
        let m4 = self.conv_ms_spvn_4.forward(&p4)?;
        let attn4 = sigmoid(&self.gdt_attn_4.forward(&self.gdt_convs_4.forward(&p4)?)?)?;
        let p4_gdt = p4.broadcast_mul(&attn4)?;

        let (h3, w3) = (x3.dim(2)?, x3.dim(3)?);
        let p3 = p4_gdt
            .upsample_bilinear2d(h3, w3, true)?
            .add(&self.lateral_block4.forward(x3)?)?;
        let ipt4 = self.ipt_blk4.forward(&patches_batch(image, &p3)?)?;
        let p3_in = Tensor::cat(&[p3, ipt4], 1)?;
        let p3 = self.decoder_block3.forward(&p3_in)?;
        let m3 = self.conv_ms_spvn_3.forward(&p3)?;
        let attn3 = sigmoid(&self.gdt_attn_3.forward(&self.gdt_convs_3.forward(&p3)?)?)?;
        let p3_gdt = p3.broadcast_mul(&attn3)?;

        let (h2, w2) = (x2.dim(2)?, x2.dim(3)?);
        let p2 = p3_gdt
            .upsample_bilinear2d(h2, w2, true)?
            .add(&self.lateral_block3.forward(x2)?)?;
        let ipt3 = self.ipt_blk3.forward(&patches_batch(image, &p2)?)?;
        let p2_in = Tensor::cat(&[p2, ipt3], 1)?;
        let p2 = self.decoder_block2.forward(&p2_in)?;
        let m2 = self.conv_ms_spvn_2.forward(&p2)?;
        let attn2 = sigmoid(&self.gdt_attn_2.forward(&self.gdt_convs_2.forward(&p2)?)?)?;
        let p2_gdt = p2.broadcast_mul(&attn2)?;

        let (h1, w1) = (x1.dim(2)?, x1.dim(3)?);
        let p1 = p2_gdt
            .upsample_bilinear2d(h1, w1, true)?
            .add(&self.lateral_block2.forward(x1)?)?;
        let p1 = Tensor::cat(
            &[
                p1.clone(),
                self.ipt_blk2.forward(&patches_batch(image, &p1)?)?,
            ],
            1,
        )?;
        let p1 = self.decoder_block1.forward(&p1)?;
        let (ih, iw) = (image.dim(2)?, image.dim(3)?);
        let p1_up = p1.upsample_bilinear2d(ih, iw, true)?;
        let p1_out = Tensor::cat(
            &[
                p1_up.clone(),
                self.ipt_blk1.forward(&patches_batch(image, &p1_up)?)?,
            ],
            1,
        )?;
        let logits = self.conv_out1.forward(&p1_out)?;
        Ok(DecoderOutputs {
            p4,
            p3,
            p2,
            p1,
            m4,
            m3,
            m2,
            logits,
        })
    }
}

/// Reference `Decoder.get_patches_batch` for `dec_ipt_split = True`.
///
/// The image is cut into non-overlapping patches of the target spatial size,
/// column-major and then row-major, and the patches are concatenated on the
/// channel dimension. All patched inputs in BiRefNet have exact divisors at
/// 1024 (and at 128 in the parity fixture); reject anything else instead of
/// silently changing the semantics.
fn patches_batch(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = image.dims4()?;
    let (ph, pw) = (target.dim(2)?, target.dim(3)?);
    if ph == 0 || pw == 0 || h % ph != 0 || w % pw != 0 {
        candle_core::bail!("RMBG-2.0 patch injection requires {h}x{w} to divide by {ph}x{pw}")
    }
    let rows = h / ph;
    let cols = w / pw;
    let x = image.reshape((b, c, rows, ph, cols, pw))?;
    // [B, C, rows, ph, cols, pw] -> [B, cols, rows, C, ph, pw]
    let x = x.permute((0, 4, 2, 1, 3, 5))?.contiguous()?;
    x.reshape((b, cols * rows * c, ph, pw))
}

#[derive(Debug)]
struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv1: conv2d(in_c, 64, 3, cfg, vb.pp("conv1"))?,
            conv_out: conv2d(64, out_c, 3, cfg, vb.pp("conv_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv_out.forward(&self.conv1.forward(x)?)
    }
}

#[derive(Debug)]
struct GdtConv {
    conv: Conv2d,
    bn: BatchNorm,
}

impl GdtConv {
    fn load(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv: conv2d(in_c, 16, 3, cfg, vb.pp("0"))?,
            bn: batch_norm(16, BatchNormConfig::default(), vb.pp("1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.bn.forward_t(&self.conv.forward(x)?, false)?.relu()
    }
}

#[derive(Debug)]
struct BasicDecBlk {
    conv_in: Conv2d,
    bn_in: BatchNorm,
    dec_att: ASPPDeformable,
    conv_out: Conv2d,
    bn_out: BatchNorm,
}

impl BasicDecBlk {
    fn load(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        Ok(Self {
            conv_in: conv2d(in_c, 64, 3, cfg, vb.pp("conv_in"))?,
            bn_in: batch_norm(64, BatchNormConfig::default(), vb.pp("bn_in"))?,
            dec_att: ASPPDeformable::load(64, vb.pp("dec_att"))?,
            conv_out: conv2d(64, out_c, 3, cfg, vb.pp("conv_out"))?,
            bn_out: batch_norm(out_c, BatchNormConfig::default(), vb.pp("bn_out"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_in.forward(x)?;
        let x = self.bn_in.forward_t(&x, false)?.relu()?;
        let x = self.dec_att.forward(&x)?;
        let x = self.conv_out.forward(&x)?;
        self.bn_out.forward_t(&x, false)
    }
}

#[derive(Debug)]
struct ASPPDeformable {
    aspp1: DeformBranch,
    aspp_deforms: Vec<DeformBranch>,
    global_pool_conv: Conv2d,
    global_pool_bn: BatchNorm,
    conv1: Conv2d,
    bn1: BatchNorm,
}

impl ASPPDeformable {
    fn load(in_c: usize, vb: VarBuilder) -> Result<Self> {
        let mut aspp_deforms = Vec::new();
        for i in 0..3 {
            aspp_deforms.push(DeformBranch::load(
                vb.pp("aspp_deforms").pp(i),
                in_c,
                256,
                [1, 3, 7][i],
                [1, 3, 7][i] / 2,
                false,
            )?);
        }
        Ok(Self {
            aspp1: DeformBranch::load(vb.pp("aspp1"), in_c, 256, 1, 0, false)?,
            aspp_deforms,
            global_pool_conv: conv2d_no_bias(
                in_c,
                256,
                1,
                Conv2dConfig::default(),
                vb.pp("global_avg_pool").pp("1"),
            )?,
            global_pool_bn: batch_norm(
                256,
                BatchNormConfig::default(),
                vb.pp("global_avg_pool").pp("2"),
            )?,
            conv1: conv2d_no_bias(1280, in_c, 1, Conv2dConfig::default(), vb.pp("conv1"))?,
            bn1: batch_norm(in_c, BatchNormConfig::default(), vb.pp("bn1"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x1 = self.aspp1.forward(x)?;
        let mut parts = Vec::with_capacity(5);
        parts.push(x1.clone());
        for block in &self.aspp_deforms {
            parts.push(block.forward(x)?);
        }
        let pooled = x.mean_keepdim((2, 3))?;
        let pooled = self.global_pool_conv.forward(&pooled)?;
        let pooled = self.global_pool_bn.forward_t(&pooled, false)?.relu()?;
        let (oh, ow) = (x1.dim(2)?, x1.dim(3)?);
        parts.push(pooled.upsample_bilinear2d(oh, ow, true)?);
        let x = Tensor::cat(&parts, 1)?;
        let x = self.conv1.forward(&x)?;
        self.bn1.forward_t(&x, false)?.relu()
    }
}

impl Trellis2BackgroundRemover for Rmbg2Model {
    fn remove_background(&self, rgb: &RgbImage) -> crate::Result<RgbaImage> {
        self.alpha_mask(rgb)
            .map_err(|err| crate::error::Lux3dError::ContractValidation {
                message: format!("RMBG-2.0 background removal failed: {err}"),
            })
    }
}

fn in_range(values: &Tensor, len: usize) -> Result<Tensor> {
    // Torchvision's bilinear sampler treats a corner outside [0, len) as a
    // zero contribution, so mask the corresponding weight rather than only
    // clamping the gather index.
    let dtype = values.dtype();
    values
        .ge(0.0)?
        .to_dtype(dtype)?
        .broadcast_mul(&values.lt(len as f64)?.to_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn load_golden(
        npz: &mut ndarray_npy::NpzReader<std::fs::File>,
        name: &str,
    ) -> ndarray::ArrayD<f32> {
        npz.by_name(name)
            .unwrap_or_else(|err| panic!("missing golden tensor {name}: {err}"))
    }

    fn tensor(array: ndarray::ArrayD<f32>, device: &candle_core::Device) -> Tensor {
        let dims = array.shape().to_vec();
        let (data, _) = array.into_raw_vec_and_offset();
        Tensor::from_vec(data, dims.as_slice(), device).unwrap()
    }

    /// Device for opt-in full-model parity tests:
    /// `LUX3D_RMBG2_DEVICE=cpu|cuda|vulkan|wgpu` (default cpu).
    fn parity_device() -> candle_core::Device {
        let name = std::env::var("LUX3D_RMBG2_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        match name.as_str() {
            "cpu" => candle_core::Device::Cpu,
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    candle_core::Device::new_cuda(0).expect("cuda device")
                }
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("build with `--features cuda` to run the CUDA parity test")
                }
            }
            "vulkan" => {
                #[cfg(feature = "vulkan")]
                {
                    candle_core::Device::new_vulkan(0).expect("vulkan device")
                }
                #[cfg(not(feature = "vulkan"))]
                {
                    panic!("build with `--features vulkan` to run the Vulkan parity test")
                }
            }
            "wgpu" => {
                #[cfg(feature = "wgpu")]
                {
                    candle_core::Device::new_wgpu(0).expect("wgpu device")
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    panic!("build with `--features wgpu` to run the WGPU parity test")
                }
            }
            other => panic!("unknown LUX3D_RMBG2_DEVICE `{other}`"),
        }
    }

    #[test]
    fn rmbg2_matches_reference_network_traces() {
        let model_dir = Path::new("../../models/RMBG-2.0");
        let npz_path = Path::new("../../.swarm/results/trellis2-rmbg2-golden/network_golden.npz");
        if !model_dir.join("model.safetensors").exists() || !npz_path.exists() {
            eprintln!("skipping RMBG-2.0 parity: fixture or weights missing");
            return;
        }
        let device = parity_device();
        eprintln!("RMBG-2.0 network parity device: {device:?}");
        let model = Rmbg2Model::load_from_path(model_dir, &device).expect("RMBG-2.0");
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz_path).unwrap()).unwrap();

        let input = tensor(load_golden(&mut npz, "input"), &device);
        let (_, _, h, w) = input.dims4().unwrap();

        // Backbone stages, full and half-size branch.
        let full = model.bb.forward(&input).unwrap();
        for (i, got) in full.iter().enumerate() {
            let want = tensor(
                load_golden(&mut npz, &format!("bb_full_x{}", i + 1)),
                &device,
            );
            let d = max_abs_diff(got, &want);
            eprintln!("bb_full_x{} max abs diff {d:e}", i + 1);
            assert!(d < 1e-3, "bb_full_x{} diff {d}", i + 1);
        }
        let half_input = input.upsample_bilinear2d(h / 2, w / 2, true).unwrap();
        let half = model.bb.forward(&half_input).unwrap();
        for (i, got) in half.iter().enumerate() {
            let want = tensor(
                load_golden(&mut npz, &format!("bb_half_x{}", i + 1)),
                &device,
            );
            let d = max_abs_diff(got, &want);
            eprintln!("bb_half_x{} max abs diff {d:e}", i + 1);
            assert!(d < 1e-3, "bb_half_x{} diff {d}", i + 1);
        }

        let (enc, cxt) = model.forward_enc(&input).unwrap();
        for (i, got) in enc.iter().enumerate() {
            let want = tensor(load_golden(&mut npz, &format!("enc_x{}", i + 1)), &device);
            let d = max_abs_diff(got, &want);
            eprintln!("enc_x{} max abs diff {d:e}", i + 1);
            assert!(d < 1e-3, "enc_x{} diff {d}", i + 1);
        }
        let want = tensor(load_golden(&mut npz, "x4_cxt"), &device);
        assert!(max_abs_diff(&cxt, &want) < 1e-3);

        let squeezed = model.squeeze.forward(&cxt).unwrap();
        let want = tensor(load_golden(&mut npz, "squeeze"), &device);
        let d = max_abs_diff(&squeezed, &want);
        eprintln!("squeeze max abs diff {d:e}");
        assert!(d < 1e-3, "squeeze diff {d}");

        let ipt5 = model
            .decoder
            .ipt_blk5
            .forward(&patches_batch(&input, &squeezed).unwrap())
            .unwrap();
        for (name, got) in [
            ("ipt5_patches", &patches_batch(&input, &squeezed).unwrap()),
            ("ipt5", &ipt5),
        ] {
            let want = tensor(load_golden(&mut npz, name), &device);
            eprintln!("{name} diff {:e}", max_abs_diff(got, &want));
        }
        let dec4_input = Tensor::cat(&[squeezed.clone(), ipt5], 1).unwrap();
        let want = tensor(load_golden(&mut npz, "dec4_input"), &device);
        eprintln!("dec4_input diff {:e}", max_abs_diff(&dec4_input, &want));
        let outputs = model.decoder.forward(&input, &enc, squeezed).unwrap();
        for (name, got, tol) in [
            ("p4", &outputs.p4, 2e-3),
            ("p3", &outputs.p3, 2e-3),
            ("p2", &outputs.p2, 2e-3),
            ("p1", &outputs.p1, 2e-3),
            ("logits", &outputs.logits, 2e-2),
        ] {
            let want = tensor(load_golden(&mut npz, name), &device);
            let d = max_abs_diff(got, &want);
            eprintln!("{name} max abs diff {d:e}");
            assert!(d < tol, "{name} diff {d}");
        }
        let (ih, iw) = (input.dim(2).unwrap(), input.dim(3).unwrap());
        let p1_up = outputs.p1.upsample_bilinear2d(ih, iw, true).unwrap();
        let ipt1 = model
            .decoder
            .ipt_blk1
            .forward(&patches_batch(&input, &p1_up).unwrap())
            .unwrap();
        let want = tensor(load_golden(&mut npz, "ipt1"), &device);
        eprintln!("ipt1 diff {:e}", max_abs_diff(&ipt1, &want));
        for (i, got) in [&outputs.m4, &outputs.m3, &outputs.m2, &outputs.logits]
            .iter()
            .enumerate()
        {
            let want = tensor(load_golden(&mut npz, &format!("full_out_{i}")), &device);
            let d = max_abs_diff(got, &want);
            eprintln!("full_out_{i} max abs diff {d:e}");
            assert!(d < 1e-2, "full_out_{i} diff {d}");
        }
        let final_out = tensor(load_golden(&mut npz, "full_out_last"), &device);
        let d = max_abs_diff(&outputs.logits, &final_out);
        eprintln!("full_out_last max abs diff {d:e}");
        assert!(d < 2e-2);
        drop(model);
    }

    #[test]
    fn rmbg2_squeeze_stage_trace() {
        let model_dir = Path::new("../../models/RMBG-2.0");
        let trace_path = Path::new("../../.swarm/results/trellis2-rmbg2-golden/squeeze_trace.npz");
        if !model_dir.join("model.safetensors").exists() || !trace_path.exists() {
            eprintln!("skipping RMBG-2.0 squeeze trace: fixture or weights missing");
            return;
        }
        let device = candle_core::Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                DType::F32,
                &device,
            )
            .unwrap()
        };
        let model = Rmbg2Model::load(&Rmbg2Config, vb).unwrap();
        let mut npz =
            ndarray_npy::NpzReader::new(std::fs::File::open(trace_path).unwrap()).unwrap();
        let input = tensor(load_golden(&mut npz, "input"), &device);
        let block = &model.squeeze;
        let x = block.conv_in.forward(&input).unwrap();
        let d = max_abs_diff(&x, &tensor(load_golden(&mut npz, "conv_in"), &device));
        eprintln!("conv_in diff {d:e}");
        let x = block.bn_in.forward_t(&x, false).unwrap().relu().unwrap();
        let d = max_abs_diff(&x, &tensor(load_golden(&mut npz, "relu_in"), &device));
        eprintln!("relu_in diff {d:e}");
        let att = &block.dec_att;
        let a1 = att.aspp1.forward(&x).unwrap();
        let d = max_abs_diff(&a1, &tensor(load_golden(&mut npz, "aspp1"), &device));
        eprintln!("aspp1 diff {d:e}");
        for (i, aspp) in att.aspp_deforms.iter().enumerate() {
            let got = aspp.forward(&x).unwrap();
            let d = max_abs_diff(
                &got,
                &tensor(load_golden(&mut npz, &format!("aspp_deform{i}")), &device),
            );
            eprintln!("aspp_deform{i} diff {d:e}");
        }
        let pooled = x.mean_keepdim((2, 3)).unwrap();
        let pooled = att.global_pool_conv.forward(&pooled).unwrap();
        let pooled = att
            .global_pool_bn
            .forward_t(&pooled, false)
            .unwrap()
            .relu()
            .unwrap();
        let d = max_abs_diff(&pooled, &tensor(load_golden(&mut npz, "global"), &device));
        eprintln!("global diff {d:e}");
        drop(model);
    }

    #[test]
    fn rmbg2_alpha_matches_reference_preprocess_golden() {
        if std::env::var("LUX3D_RMBG2_FULL_TEST").as_deref() != Ok("1") {
            eprintln!("skipping full RMBG-2.0 alpha parity (set LUX3D_RMBG2_FULL_TEST=1)");
            return;
        }
        let model_dir = Path::new("../../models/RMBG-2.0");
        let npz_path =
            Path::new("../../.swarm/results/trellis2-rmbg2-golden/preprocess_golden.npz");
        if !model_dir.join("model.safetensors").exists() || !npz_path.exists() {
            eprintln!("skipping full RMBG-2.0 alpha parity: fixture or weights missing");
            return;
        }
        let device = parity_device();
        eprintln!("RMBG-2.0 full parity device: {device:?}");
        let model = Rmbg2Model::load_from_path(model_dir, &device).expect("RMBG-2.0");
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz_path).unwrap()).unwrap();
        let source = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("downscaled_rgb")
            .expect("downscaled_rgb");
        let expected = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix2>("rembg_alpha")
            .expect("rembg_alpha");
        let (h, w, _) = (source.shape()[0], source.shape()[1], source.shape()[2]);
        let rgb = RgbImage::from_raw(w as u32, h as u32, source.as_slice().unwrap().to_vec())
            .expect("rgb buffer");
        let rgba = model.alpha_mask(&rgb).unwrap();
        let got: Vec<u8> = rgba.pixels().map(|pixel| pixel[3]).collect();
        let mut max_diff = 0i32;
        let mut mismatches = 0usize;
        for (&a, &b) in got.iter().zip(expected.as_slice().unwrap()) {
            let d = (i32::from(a) - i32::from(b)).abs();
            max_diff = max_diff.max(d);
            if d != 0 {
                mismatches += 1;
            }
        }
        eprintln!(
            "RMBG-2.0 alpha max |diff|={max_diff}, mismatched={mismatches}/{}",
            got.len()
        );
        assert!(max_diff <= 1, "RMBG-2.0 alpha drift {max_diff}");
        assert!(
            mismatches * 100 <= got.len(),
            "more than 1% of alpha pixels differ: {mismatches}/{}",
            got.len()
        );
    }

    /// Full reference preprocessing on the RGB fixture: RMBG-2.0 mask ->
    /// crop bbox -> premultiplied crop -> LANCZOS DINOv3 inputs. Opt-in
    /// because it runs one full BiRefNet forward.
    #[test]
    fn rmbg2_preprocess_crop_matches_reference_golden() {
        if std::env::var("LUX3D_RMBG2_FULL_TEST").as_deref() != Ok("1") {
            eprintln!("skipping RMBG-2.0 preprocess parity (set LUX3D_RMBG2_FULL_TEST=1)");
            return;
        }
        let model_dir = Path::new("../../models/RMBG-2.0");
        let npz_path =
            Path::new("../../.swarm/results/trellis2-rmbg2-golden/preprocess_golden.npz");
        if !model_dir.join("model.safetensors").exists() || !npz_path.exists() {
            eprintln!("skipping RMBG-2.0 preprocess parity: fixture or weights missing");
            return;
        }
        let device = candle_core::Device::Cpu;
        let model = Rmbg2Model::load_from_path(model_dir, &device).expect("RMBG-2.0");
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(npz_path).unwrap()).unwrap();
        let source = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("source_rgb")
            .expect("source_rgb");
        let (h, w) = (source.shape()[0], source.shape()[1]);
        let rgb = RgbImage::from_raw(
            w as u32,
            h as u32,
            source.as_slice().expect("contiguous").to_vec(),
        )
        .expect("source rgb");
        let stage = crate::preprocess::Trellis2Preprocess::new(512).expect("preprocess");
        let crop = stage
            .prepare_crop_from_dynamic_image(&image::DynamicImage::ImageRgb8(rgb), Some(&model))
            .expect("preprocess crop");

        let bbox = npz
            .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix1>("crop_bbox")
            .expect("crop_bbox");
        let bbox = bbox.as_slice().expect("contiguous");
        assert_eq!(
            (
                bbox[0] as u32,
                bbox[1] as u32,
                bbox[2] as u32,
                bbox[3] as u32
            ),
            crop.crop_info().bbox,
            "bbox mismatch"
        );
        let crop_box = npz
            .by_name::<ndarray::OwnedRepr<i64>, ndarray::Ix1>("crop_box")
            .expect("crop_box");
        let crop_box = crop_box.as_slice().expect("contiguous");
        assert_eq!(
            (crop_box[0], crop_box[1], crop_box[2], crop_box[3]),
            crop.crop_info().crop_box,
            "crop box mismatch"
        );

        let expected = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("final_rgb")
            .expect("final_rgb");
        let expected = expected.as_slice().expect("contiguous");
        let got = crop.rgb().as_raw();
        let mut max_diff = 0i32;
        let mut mismatches = 0usize;
        for (&a, &b) in got.iter().zip(expected.iter()) {
            let d = (i32::from(a) - i32::from(b)).abs();
            max_diff = max_diff.max(d);
            if d != 0 {
                mismatches += 1;
            }
        }
        eprintln!(
            "RMBG-2.0 crop max |diff|={max_diff}, mismatched={mismatches}/{}",
            got.len()
        );
        assert!(max_diff <= 1, "RMBG-2.0 crop drift {max_diff}");
        assert!(
            mismatches * 1000 <= got.len(),
            "more than 0.1% of crop bytes differ: {mismatches}/{}",
            got.len()
        );

        let mean = [0.485_f32, 0.456, 0.406];
        let std = [0.229_f32, 0.224, 0.225];
        for res in [512_u32, 1024] {
            let stage = crate::preprocess::Trellis2Preprocess::new(res).expect("stage");
            let prepared = stage.resize_crop(&crop).expect("resize crop");
            let expected = npz
                .by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix4>(&format!("dino_input_{res}"))
                .expect("dino_input");
            let expected = expected.as_slice().expect("contiguous");
            let mut max_diff = 0.0_f32;
            let plane = res as usize * res as usize;
            for channel in 0..3 {
                for index in 0..plane {
                    let byte = prepared.uint8_rgb[index * 3 + channel];
                    let got = (f32::from(byte) / 255.0 - mean[channel]) / std[channel];
                    max_diff = max_diff.max((got - expected[channel * plane + index]).abs());
                }
            }
            // The RMBG alpha mask differs by at most 1 LSB at 87/205932 crop
            // bytes; LANCZOS (verified byte-exact against Pillow above) can
            // turn that into at most a 2-byte difference => 2/255/std.
            eprintln!("RMBG-2.0 dino_input_{res} max |diff|={max_diff:e}");
            assert!(
                max_diff <= 3.0 / 255.0 / 0.229 + 1e-6,
                "DINOv3 {res} input drift {max_diff:e}"
            );
        }
    }

    /// Cross-device comparison at the production 1024x1024 input size.
    /// `LUX3D_RMBG2_CROSS_TEST=1`, with `LUX3D_RMBG2_DEVICE` selecting the
    /// accelerator. Compares encoder/context, squeeze and head tensors.
    #[test]
    fn rmbg2_cross_device_1024_stages() {
        if std::env::var("LUX3D_RMBG2_CROSS_TEST").as_deref() != Ok("1") {
            eprintln!("skipping RMBG-2.0 cross-device test (set LUX3D_RMBG2_CROSS_TEST=1)");
            return;
        }
        let model_dir = Path::new("../../models/RMBG-2.0");
        if !model_dir.join("model.safetensors").exists() {
            eprintln!("skipping RMBG-2.0 cross-device test: weights missing");
            return;
        }
        let gpu = parity_device();
        if gpu.is_cpu() {
            eprintln!("skipping RMBG-2.0 cross-device test: device is CPU");
            return;
        }
        let cpu = candle_core::Device::Cpu;
        let model_cpu = Rmbg2Model::load_from_path(model_dir, &cpu).expect("cpu model");
        let model_gpu = Rmbg2Model::load_from_path(model_dir, &gpu).expect("gpu model");

        let mut values = Vec::with_capacity(3 * 1024 * 1024);
        for channel in 0..3 {
            for index in 0..1024 * 1024 {
                let v = (index as f32 * 0.0001 + channel as f32).sin() * 2.0;
                values.push(
                    (v / 255.0 - TRELLIS2_IMAGE_NET_MEAN[channel])
                        / TRELLIS2_IMAGE_NET_STD[channel],
                );
            }
        }
        let t_cpu = Tensor::from_vec(values.clone(), (1, 3, 1024, 1024), &cpu).unwrap();
        let t_gpu = Tensor::from_vec(values, (1, 3, 1024, 1024), &gpu).unwrap();

        let (enc_cpu, cxt_cpu) = model_cpu.forward_enc(&t_cpu).unwrap();
        let (enc_gpu, cxt_gpu) = model_gpu.forward_enc(&t_gpu).unwrap();
        let enc_gpu_host = enc_gpu
            .clone()
            .map(|tensor| tensor.to_device(&cpu).unwrap());
        let cxt_gpu_host = cxt_gpu.to_device(&cpu).unwrap();
        for i in 0..4 {
            eprintln!(
                "enc[{i}] cross-device max diff {}",
                max_abs_diff(&enc_cpu[i], &enc_gpu_host[i])
            );
        }
        eprintln!(
            "cxt cross-device max diff {}",
            max_abs_diff(&cxt_cpu, &cxt_gpu_host)
        );
        let sq_cpu = model_cpu.squeeze.forward(&cxt_cpu).unwrap();
        let sq_gpu = model_gpu.squeeze.forward(&cxt_gpu).unwrap();
        eprintln!(
            "squeeze cross-device max diff {}",
            max_abs_diff(&sq_cpu, &sq_gpu.to_device(&cpu).unwrap())
        );
        let out_cpu = model_cpu.decoder.forward(&t_cpu, &enc_cpu, sq_cpu).unwrap();
        let out_gpu = model_gpu.decoder.forward(&t_gpu, &enc_gpu, sq_gpu).unwrap();
        let attn2_cpu = candle_nn::ops::sigmoid(
            &model_cpu
                .decoder
                .gdt_attn_2
                .forward(&model_cpu.decoder.gdt_convs_2.forward(&out_cpu.p2).unwrap())
                .unwrap(),
        )
        .unwrap();
        let attn2_gpu = candle_nn::ops::sigmoid(
            &model_gpu
                .decoder
                .gdt_attn_2
                .forward(&model_gpu.decoder.gdt_convs_2.forward(&out_gpu.p2).unwrap())
                .unwrap(),
        )
        .unwrap();
        eprintln!(
            "gdt_attn2 cross-device max diff {}",
            max_abs_diff(&attn2_cpu, &attn2_gpu.to_device(&cpu).unwrap())
        );
        let p2g_cpu = out_cpu.p2.broadcast_mul(&attn2_cpu).unwrap();
        let p2g_gpu = out_gpu.p2.broadcast_mul(&attn2_gpu).unwrap();
        eprintln!(
            "p2_gdt cross-device max diff {}",
            max_abs_diff(&p2g_cpu, &p2g_gpu.to_device(&cpu).unwrap())
        );
        let lat_cpu = model_cpu
            .decoder
            .lateral_block2
            .forward(&enc_cpu[0])
            .unwrap();
        let lat_gpu = model_gpu
            .decoder
            .lateral_block2
            .forward(&enc_gpu[0])
            .unwrap();
        eprintln!(
            "lateral2 cross-device max diff {}",
            max_abs_diff(&lat_cpu, &lat_gpu.to_device(&cpu).unwrap())
        );
        let p1pre_cpu = p2g_cpu
            .upsample_bilinear2d(enc_cpu[0].dim(2).unwrap(), enc_cpu[0].dim(3).unwrap(), true)
            .unwrap()
            .add(&lat_cpu)
            .unwrap();
        let p1pre_gpu = p2g_gpu
            .upsample_bilinear2d(enc_gpu[0].dim(2).unwrap(), enc_gpu[0].dim(3).unwrap(), true)
            .unwrap()
            .add(&lat_gpu)
            .unwrap();
        eprintln!(
            "p1_pre_decoder cross-device max diff {}",
            max_abs_diff(&p1pre_cpu, &p1pre_gpu.to_device(&cpu).unwrap())
        );
        let patches2_cpu = patches_batch(&t_cpu, &p1pre_cpu).unwrap();
        let patches2_gpu = patches_batch(&t_gpu, &p1pre_gpu).unwrap();
        eprintln!(
            "patches2 cross-device max diff {}",
            max_abs_diff(&patches2_cpu, &patches2_gpu.to_device(&cpu).unwrap())
        );
        let ipt2_cpu = model_cpu.decoder.ipt_blk2.forward(&patches2_cpu).unwrap();
        let ipt2_gpu = model_gpu.decoder.ipt_blk2.forward(&patches2_gpu).unwrap();
        eprintln!(
            "ipt2 cross-device max diff {}",
            max_abs_diff(&ipt2_cpu, &ipt2_gpu.to_device(&cpu).unwrap())
        );
        let dec1_in_cpu = Tensor::cat(&[p1pre_cpu.clone(), ipt2_cpu], 1).unwrap();
        let dec1_in_gpu = Tensor::cat(&[p1pre_gpu.clone(), ipt2_gpu], 1).unwrap();
        let dec1_cpu = model_cpu
            .decoder
            .decoder_block1
            .forward(&dec1_in_cpu)
            .unwrap();
        let dec1_gpu = model_gpu
            .decoder
            .decoder_block1
            .forward(&dec1_in_gpu)
            .unwrap();
        eprintln!(
            "decoder_block1 cross-device max diff {}",
            max_abs_diff(&dec1_cpu, &dec1_gpu.to_device(&cpu).unwrap())
        );
        let (ih, iw) = (t_cpu.dim(2).unwrap(), t_cpu.dim(3).unwrap());
        let p1up_cpu = out_cpu.p1.upsample_bilinear2d(ih, iw, true).unwrap();
        let p1up_gpu = out_gpu.p1.upsample_bilinear2d(ih, iw, true).unwrap();
        eprintln!(
            "p1_up cross-device max diff {}",
            max_abs_diff(&p1up_cpu, &p1up_gpu.to_device(&cpu).unwrap())
        );
        let ipt1_cpu = model_cpu
            .decoder
            .ipt_blk1
            .forward(&patches_batch(&t_cpu, &p1up_cpu).unwrap())
            .unwrap();
        let ipt1_gpu = model_gpu
            .decoder
            .ipt_blk1
            .forward(&patches_batch(&t_gpu, &p1up_gpu).unwrap())
            .unwrap();
        eprintln!(
            "ipt1 cross-device max diff {}",
            max_abs_diff(&ipt1_cpu, &ipt1_gpu.to_device(&cpu).unwrap())
        );
        for (name, cpu_t, gpu_t) in [
            ("p1", out_cpu.p1.clone(), out_gpu.p1.clone()),
            ("m4", out_cpu.m4, out_gpu.m4),
            ("m3", out_cpu.m3, out_gpu.m3),
            ("m2", out_cpu.m2, out_gpu.m2),
            ("logits", out_cpu.logits, out_gpu.logits),
        ] {
            let gpu_t = gpu_t.to_device(&cpu).unwrap();
            eprintln!(
                "{name} cross-device max diff {}",
                max_abs_diff(&cpu_t, &gpu_t)
            );
        }
    }

    #[test]
    fn rmbg2_cross_device_gather_output_chunks() {
        if std::env::var("LUX3D_RMBG2_CROSS_TEST").as_deref() != Ok("1") {
            eprintln!("skipping gather output-chunk probe (set LUX3D_RMBG2_CROSS_TEST=1)");
            return;
        }
        let gpu = parity_device();
        if gpu.is_cpu() {
            return;
        }
        let cpu = candle_core::Device::Cpu;
        let (b, c, total, n) = (1usize, 4usize, 262144usize, 262144usize);
        let values: Vec<f32> = (0..b * c * total)
            .map(|i| ((i * 13 + 7) % 1009) as f32)
            .collect();
        let indices: Vec<u32> = (0..n).map(|i| ((i * 7919 + 13) % total) as u32).collect();
        let x_cpu = Tensor::from_vec(values.clone(), (b, c, total), &cpu).unwrap();
        let x_gpu = Tensor::from_vec(values, (b, c, total), &gpu).unwrap();
        let idx_cpu = Tensor::from_vec(indices.clone(), (b, 1, n), &cpu)
            .unwrap()
            .broadcast_as((b, c, n))
            .unwrap()
            .contiguous()
            .unwrap();
        let idx_gpu = Tensor::from_vec(indices, (b, 1, n), &gpu)
            .unwrap()
            .broadcast_as((b, c, n))
            .unwrap()
            .contiguous()
            .unwrap();
        let expected = x_cpu.gather(&idx_cpu, 2).unwrap();
        let raw = x_gpu.gather(&idx_gpu, 2).unwrap().to_device(&cpu).unwrap();
        let diff = max_abs_diff(&expected, &raw);
        eprintln!("gather total={total} n={n} raw max diff {diff}");
        assert_eq!(diff, 0.0, "raw gather drift");
        for chunk in [16384usize, 32768] {
            let mut parts = Vec::new();
            for start in (0..n).step_by(chunk) {
                let len = chunk.min(n - start);
                let ids = idx_gpu.narrow(2, start, len).unwrap().contiguous().unwrap();
                parts.push(x_gpu.gather(&ids, 2).unwrap());
            }
            let out = Tensor::cat(&parts, 2).unwrap().to_device(&cpu).unwrap();
            let diff = max_abs_diff(&expected, &out);
            eprintln!("gather total={total} n={n} chunk={chunk} max diff {diff}");
            assert_eq!(diff, 0.0, "chunked gather drift at chunk {chunk}");
        }
    }

    #[test]
    fn rmbg2_cross_device_conv_out_sizes() {
        if std::env::var("LUX3D_RMBG2_CROSS_TEST").as_deref() != Ok("1") {
            eprintln!("skipping RMBG-2.0 conv-size probe (set LUX3D_RMBG2_CROSS_TEST=1)");
            return;
        }
        let model_dir = Path::new("../../models/RMBG-2.0");
        if !model_dir.join("model.safetensors").exists() {
            eprintln!("skipping RMBG-2.0 conv-size probe: weights missing");
            return;
        }
        let gpu = parity_device();
        if gpu.is_cpu() {
            return;
        }
        let cpu = candle_core::Device::Cpu;
        let model_cpu = Rmbg2Model::load_from_path(model_dir, &cpu).expect("cpu model");
        let model_gpu = Rmbg2Model::load_from_path(model_dir, &gpu).expect("gpu model");
        for size in [64usize, 128, 256, 512, 1024] {
            let n = 240 * size * size;
            let values: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
            let t_cpu = Tensor::from_vec(values.clone(), (1, 240, size, size), &cpu).unwrap();
            let t_gpu = Tensor::from_vec(values, (1, 240, size, size), &gpu).unwrap();
            let out_cpu = model_cpu.decoder.conv_out1.forward(&t_cpu).unwrap();
            let out_gpu = model_gpu
                .decoder
                .conv_out1
                .forward(&t_gpu)
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            eprintln!(
                "conv_out1 {size} cross-device max diff {}",
                max_abs_diff(&out_cpu, &out_gpu)
            );
        }
    }

    #[test]
    fn rmbg2_lanczos_resize_matches_pillow_bytes() {
        let golden = Path::new("../../.swarm/results/trellis2-rmbg2-golden/lanczos_golden.npz");
        let source_path =
            Path::new("../../.swarm/results/trellis2-rmbg2-golden/preprocess_golden.npz");
        if !golden.exists() || !source_path.exists() {
            eprintln!("skipping LANCZOS byte parity: golden missing");
            return;
        }
        let mut source_npz =
            ndarray_npy::NpzReader::new(std::fs::File::open(source_path).unwrap()).unwrap();
        let source = source_npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("final_rgb")
            .expect("final_rgb");
        let (h, w) = (source.shape()[0], source.shape()[1]);
        let source = source.as_slice().expect("contiguous");
        let mut golden_npz =
            ndarray_npy::NpzReader::new(std::fs::File::open(golden).unwrap()).unwrap();
        for res in [512_usize, 1024] {
            let expected = golden_npz
                .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>(&format!("resized_{res}"))
                .expect("resized golden");
            let got = crate::preprocess::resize_lanczos_u8::<3>(source, w, h, res, res);
            let expected = expected.as_slice().expect("contiguous");
            let max_diff = got
                .iter()
                .zip(expected.iter())
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .max()
                .unwrap_or(0);
            let mismatches = got
                .iter()
                .zip(expected.iter())
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "LANCZOS {res} max |diff|={max_diff}, mismatched={mismatches}/{}",
                got.len()
            );
            assert_eq!(0, max_diff, "LANCZOS {res} differs from Pillow bytes");
        }
    }

    /// Quantized packages must reproduce the F32 canonical alpha mask. This
    /// runs three full 1024x1024 BiRefNet forwards, so it is opt-in.
    #[test]
    fn rmbg2_quantized_packages_match_canonical_alpha() {
        if std::env::var("LUX3D_RMBG2_QUANT_TEST").as_deref() != Ok("1") {
            eprintln!("skipping RMBG-2.0 quantization parity (set LUX3D_RMBG2_QUANT_TEST=1)");
            return;
        }
        let canonical = Path::new("../../3d/canonical-weights/rmbg-2.0");
        let variants = [
            ("f16", Path::new("../../3d/gguf-weights/rmbg-2.0-f16")),
            ("q8_0", Path::new("../../3d/gguf-weights/rmbg-2.0-q8_0")),
        ];
        if !canonical.join("model.safetensors").exists()
            || variants
                .iter()
                .any(|(_, path)| !path.join("model.gguf").exists())
        {
            eprintln!("skipping RMBG-2.0 quantization parity: packages missing");
            return;
        }
        let device = candle_core::Device::Cpu;
        let golden_path =
            Path::new("../../.swarm/results/trellis2-rmbg2-golden/preprocess_golden.npz");
        let rgb = if golden_path.exists() {
            let mut npz =
                ndarray_npy::NpzReader::new(std::fs::File::open(golden_path).expect("golden npz"))
                    .expect("golden npz reader");
            let source = npz
                .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("downscaled_rgb")
                .expect("downscaled_rgb");
            let (h, w) = (source.shape()[0], source.shape()[1]);
            RgbImage::from_raw(
                w as u32,
                h as u32,
                source.as_slice().expect("contiguous").to_vec(),
            )
            .expect("rgb buffer")
        } else {
            let mut rgb = RgbImage::new(64, 64);
            for (x, y, pixel) in rgb.enumerate_pixels_mut() {
                *pixel = image::Rgb([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8]);
            }
            rgb
        };
        let alpha_for = |path: &Path| -> Vec<u8> {
            let model = Rmbg2Model::load_from_path(path, &device).expect("load package");
            model
                .alpha_mask(&rgb)
                .expect("alpha mask")
                .pixels()
                .map(|pixel| pixel[3])
                .collect()
        };
        let baseline = alpha_for(canonical);
        for (label, path) in variants {
            let got = alpha_for(path);
            let mut max_diff = 0i32;
            let (mut mismatches, mut over_2, mut over_8, mut threshold_crossings) = (0, 0, 0, 0);
            for (&a, &b) in baseline.iter().zip(got.iter()) {
                let d = (i32::from(a) - i32::from(b)).abs();
                max_diff = max_diff.max(d);
                if d != 0 {
                    mismatches += 1;
                }
                if d > 2 {
                    over_2 += 1;
                }
                if d > 8 {
                    over_8 += 1;
                }
                if (a >= 204) != (b >= 204) {
                    threshold_crossings += 1;
                }
            }
            eprintln!(
                "RMBG-2.0 {label} alpha max |diff|={max_diff}, mismatch={mismatches}, >2={over_2}, >8={over_8}, threshold-crossings={threshold_crossings}/{}",
                got.len()
            );
            let allowed_max = if label == "f16" { 2 } else { 16 };
            assert!(
                max_diff <= allowed_max,
                "RMBG-2.0 {label} alpha drift {max_diff} > {allowed_max}"
            );
            // The 0.8*255 foreground threshold drives the crop bbox; require
            // at most 0.01% of pixels to flip across it.
            assert!(
                threshold_crossings * 10_000 <= got.len(),
                "RMBG-2.0 {label}: {threshold_crossings}/{} pixels cross the foreground threshold",
                got.len()
            );
            assert!(
                mismatches * 100 <= got.len(),
                "RMBG-2.0 {label}: more than 1% of alpha pixels differ: {mismatches}/{}",
                got.len()
            );
        }
    }
}
