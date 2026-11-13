//! BriaRMBG / IS-Net inference for TripoSG background removal.
//!
//! Reference: `models/RMBG-1.4/briarmbg.py` (the same module is vendored by
//! TripoSG as `scripts/briarmbg.py`). The reference preprocessing path is
//! implemented in `preprocess/triposg.rs`; this module only owns the network:
//!
//! - `conv_in` is a stride-2 3x3 convolution with bias and **no** activation
//!   (the first activation lives inside `stage1.rebnconvin`);
//! - `stage1..stage4` are RSU-7/6/5/4 with ceil-mode 2x2 max pools between the
//!   first `d - 2` encoder blocks; `stage5/6` are RSU-4F with dilations
//!   `1,2,4,8` and no pooling;
//! - decoder stages mirror the encoders; every `_upsample_like` is bilinear
//!   `align_corners=false` with no antialias filter (reference uses
//!   `F.interpolate(..., mode="bilinear")`);
//! - the six side heads are 3x3 convolutions; `forward` returns the first
//!   sigmoid side output (`result[0][0]` in the reference), while
//!   `forward_sides` exposes all six for parity tests.
//!
//! BatchNorm runs in eval mode with the stored running statistics and eps
//! `1e-5`. `num_batches_tracked` is a training-only integer buffer and is not
//! read. The loader accepts an F16 VarBuilder for cross-backend precision
//! experiments, but RMBG is sensitive to half-precision activation storage:
//! F16 backends agree with each other to ~1e-3, while the F16 mask differs from
//! F32 by ~1e-1 rel_l2 (about 0.27% binary flips). Prefer F32 or F16-class
//! reduced-precision matmul when the mask must match F32.

use crate::weights::VarBuilder;
use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT, batch_norm, conv2d};

/// RMBG model configuration (the checkpoint only stores the channel counts).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmbgConfig {
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Default for RmbgConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            out_channels: 1,
        }
    }
}

impl RmbgConfig {
    pub fn from_resolved_config(value: &serde_json::Value) -> CandleResult<Self> {
        let in_channels = value.get("in_ch").and_then(|v| v.as_u64()).unwrap_or(3) as usize;
        let out_channels = value.get("out_ch").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        if in_channels == 0 || out_channels == 0 {
            candle_core::bail!(
                "RMBG config has zero channels: in_ch={in_channels}, out_ch={out_channels}"
            );
        }
        Ok(Self {
            in_channels,
            out_channels,
        })
    }
}

/// BriaRMBG network. Public so parity tests can drive individual side heads.
#[derive(Debug)]
pub struct RmbgModel {
    conv_in: Conv2d,
    stages: Vec<Rsu>,
    decoders: Vec<Rsu>,
    sides: Vec<Conv2d>,
}

impl RmbgModel {
    /// Load from a VarBuilder rooted at the RMBG canonical model root. The key
    /// layout is exactly `conv_in.*`, `stage1.*`, ..., `stage1d.*`, `side1.*`
    /// etc.; no prefix is expected.
    pub fn load(vb: VarBuilder, cfg: &RmbgConfig) -> CandleResult<Self> {
        let conv_in = conv2d(
            cfg.in_channels,
            64,
            3,
            Conv2dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv_in"),
        )?;

        let stages = vec![
            Rsu::load_pooled(7, 64, 32, 64, vb.pp("stage1"))?,
            Rsu::load_pooled(6, 64, 32, 128, vb.pp("stage2"))?,
            Rsu::load_pooled(5, 128, 64, 256, vb.pp("stage3"))?,
            Rsu::load_pooled(4, 256, 128, 512, vb.pp("stage4"))?,
            Rsu::load_dilated(4, 512, 256, 512, vb.pp("stage5"))?,
            Rsu::load_dilated(4, 512, 256, 512, vb.pp("stage6"))?,
        ];
        let decoders = vec![
            Rsu::load_dilated(4, 1024, 256, 512, vb.pp("stage5d"))?,
            Rsu::load_pooled(4, 1024, 128, 256, vb.pp("stage4d"))?,
            Rsu::load_pooled(5, 512, 64, 128, vb.pp("stage3d"))?,
            Rsu::load_pooled(6, 256, 32, 64, vb.pp("stage2d"))?,
            Rsu::load_pooled(7, 128, 16, 64, vb.pp("stage1d"))?,
        ];
        let sides = (1..=6)
            .map(|index| {
                let in_channels = [64, 64, 128, 256, 512, 512][index - 1];
                conv2d(
                    in_channels,
                    cfg.out_channels,
                    3,
                    Conv2dConfig {
                        padding: 1,
                        ..Default::default()
                    },
                    vb.pp(format!("side{index}")),
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;

        Ok(Self {
            conv_in,
            stages,
            decoders,
            sides,
        })
    }

    /// Full reference forward: all six sigmoid side outputs, coarsest-first
    /// (reference `[d1, d2, d3, d4, d5, d6]`).
    pub fn forward_sides(&self, input: &Tensor) -> CandleResult<Vec<Tensor>> {
        let (_batch, channels, height, width) = input.dims4()?;
        if channels != self.stages[0].in_channels() && channels != 3 {
            // The stage1 validator below reports the precise mismatch; this
            // check only guards against silent broadcast surprises.
            candle_core::bail!("RMBG input must have 3 channels, got {channels}");
        }

        let hxin = self.conv_in.forward(input)?;
        let hx1 = self.stages[0].forward(&hxin)?;
        let hx2 = self.stages[1].forward(&max_pool_ceil(&hx1)?)?;
        let hx3 = self.stages[2].forward(&max_pool_ceil(&hx2)?)?;
        let hx4 = self.stages[3].forward(&max_pool_ceil(&hx3)?)?;
        let hx5 = self.stages[4].forward(&max_pool_ceil(&hx4)?)?;
        let hx6 = self.stages[5].forward(&max_pool_ceil(&hx5)?)?;

        let hx6up = upsample_like(&hx6, &hx5)?;
        let hx5d = self.decoders[0].forward(&Tensor::cat(&[&hx6up, &hx5], 1)?)?;
        let hx5dup = upsample_like(&hx5d, &hx4)?;
        let hx4d = self.decoders[1].forward(&Tensor::cat(&[&hx5dup, &hx4], 1)?)?;
        let hx4dup = upsample_like(&hx4d, &hx3)?;
        let hx3d = self.decoders[2].forward(&Tensor::cat(&[&hx4dup, &hx3], 1)?)?;
        let hx3dup = upsample_like(&hx3d, &hx2)?;
        let hx2d = self.decoders[3].forward(&Tensor::cat(&[&hx3dup, &hx2], 1)?)?;
        let hx2dup = upsample_like(&hx2d, &hx1)?;
        let hx1d = self.decoders[4].forward(&Tensor::cat(&[&hx2dup, &hx1], 1)?)?;

        let hidden = [&hx1d, &hx2d, &hx3d, &hx4d, &hx5d, &hx6];
        let mut outputs = Vec::with_capacity(6);
        for (side, hidden) in self.sides.iter().zip(hidden.iter()) {
            let logits = side.forward(hidden)?;
            let logits = upsample_like(&logits, input)?;
            outputs.push(candle_nn::ops::sigmoid(&logits)?);
        }

        // The side heads are upsampled to the exact input spatial size.
        let output_dims = outputs[0].dims4()?;
        if (output_dims.2, output_dims.3) != (height, width) {
            candle_core::bail!(
                "RMBG side output has spatial size {}x{}, expected {}x{}",
                output_dims.2,
                output_dims.3,
                height,
                width
            );
        }
        Ok(outputs)
    }

    /// Reference `rmbg()` output: the first (`side1`) sigmoid map.
    pub fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let mut sides = self.forward_sides(input)?;
        Ok(sides.remove(0))
    }
}

#[derive(Debug)]
enum Rsu {
    Pooled(RsuPooled),
    Dilated(RsuDilated),
}

impl Rsu {
    fn load_pooled(
        depth: usize,
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        if depth < 2 {
            candle_core::bail!("RSU depth must be at least 2, got {depth}");
        }
        let rebnconvin = RebnConv::load(in_channels, out_channels, 1, vb.pp("rebnconvin"))?;
        let mut encoders = Vec::with_capacity(depth);
        for index in 1..=depth {
            let (input, output, dilation) = if index == 1 {
                (out_channels, mid_channels, 1)
            } else if index == depth {
                (mid_channels, mid_channels, 2)
            } else {
                (mid_channels, mid_channels, 1)
            };
            encoders.push(RebnConv::load(
                input,
                output,
                dilation,
                vb.pp(format!("rebnconv{index}")),
            )?);
        }
        let mut decoders = Vec::with_capacity(depth - 1);
        for index in (1..depth).rev() {
            let output = if index == 1 {
                out_channels
            } else {
                mid_channels
            };
            decoders.push(RebnConv::load(
                mid_channels * 2,
                output,
                1,
                vb.pp(format!("rebnconv{index}d")),
            )?);
        }
        Ok(Self::Pooled(RsuPooled {
            in_channels,
            rebnconvin,
            encoders,
            decoders,
        }))
    }

    fn load_dilated(
        depth: usize,
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let rebnconvin = RebnConv::load(in_channels, out_channels, 1, vb.pp("rebnconvin"))?;
        let mut encoders = Vec::with_capacity(depth);
        let mut dilation = 1;
        for index in 1..=depth {
            let (input, output) = if index == 1 {
                (out_channels, mid_channels)
            } else if index == depth {
                (mid_channels, mid_channels)
            } else {
                (mid_channels, mid_channels)
            };
            encoders.push(RebnConv::load(
                input,
                output,
                dilation,
                vb.pp(format!("rebnconv{index}")),
            )?);
            dilation *= 2;
        }
        let mut decoders = Vec::with_capacity(depth - 1);
        dilation = 1 << (depth - 2);
        for index in (1..depth).rev() {
            let output = if index == 1 {
                out_channels
            } else {
                mid_channels
            };
            decoders.push(RebnConv::load(
                mid_channels * 2,
                output,
                dilation,
                vb.pp(format!("rebnconv{index}d")),
            )?);
            dilation /= 2;
        }
        Ok(Self::Dilated(RsuDilated {
            in_channels,
            rebnconvin,
            encoders,
            decoders,
        }))
    }

    fn in_channels(&self) -> usize {
        match self {
            Self::Pooled(stage) => stage.in_channels,
            Self::Dilated(stage) => stage.in_channels,
        }
    }

    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        match self {
            Self::Pooled(stage) => stage.forward(input),
            Self::Dilated(stage) => stage.forward(input),
        }
    }
}

#[derive(Debug)]
struct RsuPooled {
    in_channels: usize,
    rebnconvin: RebnConv,
    encoders: Vec<RebnConv>,
    decoders: Vec<RebnConv>,
}

impl RsuPooled {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let hxin = self.rebnconvin.forward(input)?;
        let depth = self.encoders.len();
        let mut levels: Vec<Tensor> = Vec::with_capacity(depth);
        let mut hidden = hxin.clone();
        for (index, conv) in self.encoders.iter().enumerate() {
            hidden = conv.forward(&hidden)?;
            levels.push(hidden.clone());
            if index + 2 < depth {
                hidden = max_pool_ceil(&hidden)?;
            }
        }

        // Deepest level first; decoder `j` corresponds to `rebnconv{d-1-j}d`
        // and merges the upsampled previous output with level `d-2-j`.
        let mut current = levels[depth - 1].clone();
        for (decoder_index, decoder) in self.decoders.iter().enumerate() {
            let level = depth - 2 - decoder_index;
            let upsampled = upsample_like(&current, &levels[level])?;
            let merged = Tensor::cat(&[&upsampled, &levels[level]], 1)?;
            current = decoder.forward(&merged)?;
        }
        current.add(&hxin)
    }
}

#[derive(Debug)]
struct RsuDilated {
    in_channels: usize,
    rebnconvin: RebnConv,
    encoders: Vec<RebnConv>,
    decoders: Vec<RebnConv>,
}

impl RsuDilated {
    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let hxin = self.rebnconvin.forward(input)?;
        let mut levels: Vec<Tensor> = Vec::with_capacity(self.encoders.len());
        let mut hidden = hxin.clone();
        for conv in &self.encoders {
            hidden = conv.forward(&hidden)?;
            levels.push(hidden.clone());
        }
        let depth = levels.len();
        let mut current = levels[depth - 1].clone();
        for (decoder_index, decoder) in self.decoders.iter().enumerate() {
            let level = depth - 2 - decoder_index;
            let merged = Tensor::cat(&[&current, &levels[level]], 1)?;
            current = decoder.forward(&merged)?;
        }
        current.add(&hxin)
    }
}

/// Reference `REBNCONV`: conv3x3(padding=dilation) -> BatchNorm(eval) -> ReLU.
#[derive(Debug)]
struct RebnConv {
    conv: Conv2d,
    bn: BatchNorm,
}

impl RebnConv {
    fn load(
        in_channels: usize,
        out_channels: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> CandleResult<Self> {
        let conv = conv2d(
            in_channels,
            out_channels,
            3,
            Conv2dConfig {
                padding: dilation,
                dilation,
                ..Default::default()
            },
            vb.pp("conv_s1"),
        )?;
        let bn = batch_norm(out_channels, 1e-5, vb.pp("bn_s1"))?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, input: &Tensor) -> CandleResult<Tensor> {
        let x = self.conv.forward(input)?;
        let x = self.bn.forward_t(&x, false)?;
        x.relu()
    }
}

/// `_upsample_like(src, target)`: bilinear resize to the target spatial size,
/// `align_corners=false`, no antialias.
fn upsample_like(source: &Tensor, target: &Tensor) -> CandleResult<Tensor> {
    let source_dims = source.dims4()?;
    let target_dims = target.dims4()?;
    if source_dims.2 == target_dims.2 && source_dims.3 == target_dims.3 {
        return Ok(source.clone());
    }
    source.upsample_bilinear2d(target_dims.2, target_dims.3, false)
}

/// `nn.MaxPool2d(2, stride=2, ceil_mode=True)` with no padding. Candle's
/// max-pool kernel only supports exact windows, so when a dimension is odd we
/// append one `-inf` row/column first; the implicit ceil-mode window outside
/// the tensor has the same effect.
fn max_pool_ceil(input: &Tensor) -> CandleResult<Tensor> {
    let (batch, channels, height, width) = input.dims4()?;
    let mut padded = input.clone();
    if height % 2 == 1 {
        let row = Tensor::full(
            f32::NEG_INFINITY,
            (batch, channels, 1, width),
            input.device(),
        )?
        .to_dtype(input.dtype())?;
        padded = Tensor::cat(&[&padded, &row], 2)?;
    }
    if width % 2 == 1 {
        let column = Tensor::full(
            f32::NEG_INFINITY,
            (batch, channels, padded.dim(2)?, 1),
            input.device(),
        )?
        .to_dtype(input.dtype())?;
        padded = Tensor::cat(&[&padded, &column], 3)?;
    }
    padded.max_pool2d_with_stride(2, 2)
}
