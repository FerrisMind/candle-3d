use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Module;

use crate::weights::VarBuilder;

/// Linear layer that accepts either dense or GGUF-packed weights.
///
/// Safetensors packages load a regular [`candle_nn::Linear`]. A quantized
/// GGUF package keeps Q8_0/Q4_0/k-quant weights packed in a [`QMatMul`], so
/// the matmul runs the backend's integer-dot-product kernels and the weights
/// stay compressed in VRAM. F32/F16 GGUF entries are dequantized to a dense
/// layer, which keeps F16 storage as a container-only option for now.
#[derive(Debug, Clone)]
pub(crate) struct Linear {
    inner: LinearInner,
}

#[derive(Debug, Clone)]
enum LinearInner {
    Dense(candle_nn::Linear),
    Quant {
        weight: QMatMul,
        bias: Option<Tensor>,
        in_dim: usize,
    },
}

impl Linear {
    pub(crate) fn in_dim(&self) -> usize {
        match &self.inner {
            LinearInner::Dense(l) => l.weight().dim(1).unwrap_or(0),
            LinearInner::Quant { in_dim, .. } => *in_dim,
        }
    }

    pub(crate) fn bias(&self) -> Option<&Tensor> {
        match &self.inner {
            LinearInner::Dense(l) => l.bias(),
            LinearInner::Quant { bias, .. } => bias.as_ref(),
        }
    }

    fn dense_weight(&self) -> Option<&Tensor> {
        match &self.inner {
            LinearInner::Dense(l) => Some(l.weight()),
            LinearInner::Quant { .. } => None,
        }
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        match &self.inner {
            LinearInner::Dense(l) => l.forward(xs),
            LinearInner::Quant { weight, bias, .. } => {
                // The quantized matmul kernels across cuda/vulkan/wgpu require a
                // contiguous activation; chunked GeGLU halves and reshaped
                // attention states arrive strided.
                let xs = xs.contiguous()?;
                let ys = weight.forward(&xs)?;
                match bias {
                    Some(bias) => ys.broadcast_add(bias),
                    None => Ok(ys),
                }
            }
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        Linear::forward(self, xs)
    }
}

fn load_linear(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> CandleResult<Linear> {
    let bias = if bias { Some(vb.get(out_dim, "bias")?) } else { None };
    if let Some(qt) = vb.get_quantized("weight")? {
        if let Some(linear) = quantized_linear(qt, &vb, in_dim, out_dim, bias.clone())? {
            return Ok(linear);
        }
    }
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear {
        inner: LinearInner::Dense(candle_nn::Linear::new(weight, bias)),
    })
}

/// Keep packed weights for the quantized dtypes; dequantize raw F16/F32 GGUF
/// entries so they behave exactly like the dense safetensors path.
fn quantized_linear(
    qt: QTensor,
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: Option<Tensor>,
) -> CandleResult<Option<Linear>> {
    if qt.shape().dims2()? != (out_dim, in_dim) {
        candle_core::bail!(
            "quantized linear weight shape {:?} does not match expected ({out_dim}, {in_dim})",
            qt.shape()
        );
    }
    if matches!(
        qt.dtype(),
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
    ) {
        let weight = qt.dequantize(vb.device())?.to_dtype(DType::F32)?;
        return Ok(Some(Linear {
            inner: LinearInner::Dense(candle_nn::Linear::new(weight, bias)),
        }));
    }
    Ok(Some(Linear {
        inner: LinearInner::Quant {
            weight: QMatMul::from_qtensor(qt)?,
            bias,
            in_dim,
        },
    }))
}

/// Load a linear layer from the legacy `(vb, in, out, bias)` call convention
/// used by most runtime blocks.
pub(crate) fn linear(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, bias)
}

/// `candle_nn::linear`-compatible constructor.
pub(crate) fn linear_dense(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, true)
}

/// `candle_nn::linear_no_bias`-compatible constructor.
pub(crate) fn linear_no_bias_dense(
    in_dim: usize,
    out_dim: usize,
    vb: VarBuilder,
) -> CandleResult<Linear> {
    load_linear(vb, in_dim, out_dim, false)
}

/// Linear forward. The fused bias-epilogue GEMM (MUL_MAT_ADD,
/// `candle_nn::ops::mul_mat_add`) is correct on every shape but measured
/// NET-NEGATIVE on a 12 GiB WDDM card for the dense pi3x shapes (warm
/// criterion 5.13s -> 8.85s median, OOM risk: removing the add dispatches
/// shifts the batch/allocator dynamics over the VRAM edge), so it is
/// opt-in via CANDLE_LUX3D_FUSED_LINEAR=1. Default: plain unfused path.
/// Quantized (QMatMul) layers always take the plain path.
pub(crate) fn linear_fwd(l: &Linear, xs: &Tensor) -> CandleResult<Tensor> {
    match (l.bias(), l.dense_weight()) {
        (Some(bias), Some(weight))
            if std::env::var("CANDLE_LUX3D_FUSED_LINEAR").as_deref() == Ok("1") =>
        {
            candle_nn::ops::mul_mat_add(xs, &weight.t()?, bias)
        }
        _ => l.forward(xs),
    }
}

#[derive(Debug)]
pub(crate) struct LayerScale {
    gamma: Tensor,
}

impl LayerScale {
    pub(crate) fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            gamma: vb.get(dim, "gamma")?,
        })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        xs.broadcast_mul(&self.gamma)
    }
}

#[derive(Debug)]
pub(crate) struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    pub(crate) fn new(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            fc1: linear(vb.pp("fc1"), dim, dim * 4, true)?,
            fc2: linear(vb.pp("fc2"), dim * 4, dim, true)?,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let xs = linear_fwd(&self.fc1, xs)?.gelu_erf()?;
        linear_fwd(&self.fc2, &xs)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct GeGlu {
    proj: Linear,
}

impl GeGlu {
    pub(crate) fn load(vb: VarBuilder, dim_in: usize, dim_out: usize) -> CandleResult<Self> {
        Ok(Self {
            proj: linear(vb.pp("proj"), dim_in, dim_out * 2, true)?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let parts = linear_fwd(&self.proj, xs)?.chunk(2, candle_core::D::Minus1)?;
        parts[0].broadcast_mul(&parts[1].gelu_erf()?)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct FeedForward {
    geglu: GeGlu,
    out: Linear,
}

impl FeedForward {
    pub(crate) fn load(vb: VarBuilder, dim: usize) -> CandleResult<Self> {
        Ok(Self {
            geglu: GeGlu::load(vb.pp("net").pp("0"), dim, dim * 4)?,
            out: linear(vb.pp("net").pp("2"), dim * 4, dim, true)?,
        })
    }

    pub(crate) fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.out.forward(&self.geglu.forward(xs)?)
    }
}
