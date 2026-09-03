use candle_core::{D, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, attention::AttnMask, layer_norm};

use super::{
    attention_math::{Rope2d, RopeEmbeddings, exact_query_chunked_sdpa, position_getter},
    nn_blocks::{LayerScale, Mlp, linear},
};

#[derive(Debug)]
struct RopeAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    rope: Rope2d,
}

impl RopeAttention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            q_norm: layer_norm(head_dim, 1e-5, vb.pp("q_norm"))?,
            k_norm: layer_norm(head_dim, 1e-5, vb.pp("k_norm"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            rope,
        })
    }
}

impl Module for RopeAttention {
    fn forward(&self, _xs: &Tensor) -> CandleResult<Tensor> {
        candle_core::bail!("RopeAttention requires positions")
    }
}

impl RopeAttention {
    fn forward_with_cache(&self, xs: &Tensor, rope_cache: &RopeEmbeddings) -> CandleResult<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .transpose(1, 3)?;
        let q = self.q_norm.forward(&qkv.i((.., .., 0))?)?.contiguous()?;
        let k = self.k_norm.forward(&qkv.i((.., .., 1))?)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.contiguous()?;

        let q = self
            .rope
            .apply_with_embeddings(&q, rope_cache)?
            .contiguous()?;
        let k = self
            .rope
            .apply_with_embeddings(&k, rope_cache)?
            .contiguous()?;
        let out = match xs.device() {
            candle_core::Device::Cpu => candle_nn::attention::flash_attn::<f32>(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                self.scale as f32,
                AttnMask::None,
                None,
                None,
            )?,
            _ => exact_query_chunked_sdpa(&q, &k, &v, self.scale as f32, 128)?,
        }
        .transpose(1, 2)?
        .reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct RopeBlock {
    norm1: LayerNorm,
    attn: RopeAttention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl RopeBlock {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: RopeAttention::new(vb.pp("attn"), dim, num_heads, rope)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, rope_cache: &RopeEmbeddings) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.ls1.forward(
            &self
                .attn
                .forward_with_cache(&self.norm1.forward(xs)?, rope_cache)?,
        )?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
pub struct Pi3Decoder {
    register_token: Tensor,
    blocks: Vec<RopeBlock>,
    patch_size: usize,
}

impl Pi3Decoder {
    pub fn new(vb: VarBuilder) -> CandleResult<Self> {
        let rope = Rope2d::new(100.0);
        let decoder_vb = vb.pp("decoder");
        let blocks = (0..36)
            .map(|idx| RopeBlock::new(decoder_vb.pp(idx.to_string()), 1024, 16, rope))
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            register_token: vb.get((1, 1, 5, 1024), "register_token")?,
            blocks,
            patch_size: 14,
        })
    }

    pub fn decode(
        &self,
        hidden: &Tensor,
        num_views: usize,
        height: usize,
        width: usize,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (bn, hw, dim) = hidden.dims3()?;
        let batch = bn / num_views;
        let hidden = hidden.reshape((batch * num_views, hw, dim))?;
        let register = self
            .register_token
            .broadcast_as((batch, num_views, 5, 1024))?
            .reshape((batch * num_views, 5, 1024))?;
        let mut hidden = Tensor::cat(&[&register, &hidden], 1)?;
        let hw_with_registers = hidden.dim(1)?;

        let mut positions =
            self.decoder_positions(batch, num_views, height, width, hidden.device())?;
        let positions_even = positions.clone();
        let positions_odd = positions.reshape((batch, num_views * hw_with_registers, 2))?;
        let rope = Rope2d::new(100.0);
        let even_cache = rope.embeddings(&positions_even, 64)?;
        let odd_cache = rope.embeddings(&positions_odd, 64)?;

        let mut final_outputs = Vec::with_capacity(2);
        for (idx, block) in self.blocks.iter().enumerate() {
            if idx % 2 == 0 {
                hidden = hidden.reshape((batch * num_views, hw_with_registers, 1024))?;
                positions = positions_even.clone();
                hidden = block.forward(&hidden, &even_cache)?;
            } else {
                hidden = hidden.reshape((batch, num_views * hw_with_registers, 1024))?;
                positions = positions_odd.clone();
                hidden = block.forward(&hidden, &odd_cache)?;
            }
            if idx + 1 >= self.blocks.len() - 1 {
                final_outputs.push(hidden.reshape((batch * num_views, hw_with_registers, 1024))?);
            }
        }

        let positions = positions.reshape((batch * num_views, hw_with_registers, 2))?;
        let hidden = Tensor::cat(&[&final_outputs[0], &final_outputs[1]], D::Minus1)?;
        Ok((hidden, positions))
    }

    pub fn decoder_positions(
        &self,
        batch: usize,
        num_views: usize,
        height: usize,
        width: usize,
        device: &candle_core::Device,
    ) -> CandleResult<Tensor> {
        let positions = position_getter(
            batch * num_views,
            height / self.patch_size,
            width / self.patch_size,
            device,
        )?;
        let special = Tensor::zeros((batch * num_views, 5, 2), candle_core::DType::I64, device)?;
        Tensor::cat(&[&special, &positions], 1)
    }
}

pub fn load_pi3_decoder(vb: VarBuilder) -> CandleResult<Pi3Decoder> {
    Pi3Decoder::new(vb)
}

#[derive(Debug)]
struct BranchAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    rope: Rope2d,
}

impl BranchAttention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
            rope,
        })
    }

    fn forward_with_cache(&self, xs: &Tensor, rope_cache: &RopeEmbeddings) -> CandleResult<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .transpose(1, 3)?;
        let q = self
            .rope
            .apply_with_embeddings(&qkv.i((.., .., 0))?, rope_cache)?;
        let k = self
            .rope
            .apply_with_embeddings(&qkv.i((.., .., 1))?, rope_cache)?;
        let v = qkv.i((.., .., 2))?.contiguous()?;
        let out = match xs.device() {
            candle_core::Device::Cpu => candle_nn::attention::flash_attn::<f32>(
                &q.transpose(1, 2)?,
                &k.transpose(1, 2)?,
                &v.transpose(1, 2)?,
                self.scale as f32,
                AttnMask::None,
                None,
                None,
            )?,
            _ => exact_query_chunked_sdpa(&q, &k, &v, self.scale as f32, 128)?,
        }
        .transpose(1, 2)?
        .reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct BranchBlock {
    norm1: LayerNorm,
    attn: BranchAttention,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl BranchBlock {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: BranchAttention::new(vb.pp("attn"), dim, num_heads, rope)?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, rope_cache: &RopeEmbeddings) -> CandleResult<Tensor> {
        let xs = (xs
            + self
                .attn
                .forward_with_cache(&self.norm1.forward(xs)?, rope_cache)?)?;
        let residual = &xs;
        let xs = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        xs + residual
    }
}

#[derive(Debug)]
pub struct Pi3BranchDecoder {
    projects: Linear,
    blocks: Vec<BranchBlock>,
    linear_out: Linear,
}

impl Pi3BranchDecoder {
    fn new(vb: VarBuilder, out_dim: usize) -> CandleResult<Self> {
        let rope = Rope2d::new(100.0);
        let blocks = (0..5)
            .map(|idx| BranchBlock::new(vb.pp("blocks").pp(idx.to_string()), 1024, 16, rope))
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            projects: linear(vb.pp("projects"), 2048, 1024, true)?,
            blocks,
            linear_out: linear(vb.pp("linear_out"), 1024, out_dim, true)?,
        })
    }

    pub fn forward(&self, hidden: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let mut hidden = self.projects.forward(hidden)?;
        let rope = Rope2d::new(100.0);
        let cache = rope.embeddings(positions, 64)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &cache)?;
        }
        self.linear_out.forward(&hidden)
    }
}

pub fn load_pi3_point_decoder(vb: VarBuilder) -> CandleResult<Pi3BranchDecoder> {
    Pi3BranchDecoder::new(vb.pp("point_decoder"), 1024)
}

pub fn load_pi3_conf_decoder(vb: VarBuilder) -> CandleResult<Pi3BranchDecoder> {
    Pi3BranchDecoder::new(vb.pp("conf_decoder"), 1024)
}

pub fn load_pi3_camera_decoder(vb: VarBuilder) -> CandleResult<Pi3BranchDecoder> {
    Pi3BranchDecoder::new(vb.pp("camera_decoder"), 512)
}
