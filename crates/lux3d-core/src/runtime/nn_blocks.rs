use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

pub(crate) fn linear(
    vb: VarBuilder,
    in_dim: usize,
    out_dim: usize,
    bias: bool,
) -> CandleResult<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

/// Linear forward. The fused bias-epilogue GEMM (MUL_MAT_ADD,
/// `candle_nn::ops::mul_mat_add`) is correct on every shape but measured
/// NET-NEGATIVE on a 12 GiB WDDM card for the dense pi3x shapes (warm
/// criterion 5.13s -> 8.85s median, OOM risk: removing the add dispatches
/// shifts the batch/allocator dynamics over the VRAM edge), so it is
/// opt-in via CANDLE_LUX3D_FUSED_LINEAR=1. Default: plain unfused path.
pub(crate) fn linear_fwd(l: &Linear, xs: &Tensor) -> CandleResult<Tensor> {
    match l.bias() {
        Some(bias)
            if std::env::var("CANDLE_LUX3D_FUSED_LINEAR").as_deref() == Ok(&"1") =>
        {
            candle_nn::ops::mul_mat_add(xs, &l.weight().t()?, bias)
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
