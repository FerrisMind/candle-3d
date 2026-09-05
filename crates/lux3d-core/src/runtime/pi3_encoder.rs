use std::{collections::HashMap, sync::Mutex};

use candle_core::{D, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm};

use super::{
    nn_blocks::{LayerScale, Mlp, linear},
    resampling::compute_aa_cubic_weights,
};

#[derive(Debug)]
struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f64,
}

impl Attention {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> CandleResult<Self> {
        let qkv = linear(vb.pp("qkv"), dim, dim * 3, true)?;
        let proj = linear(vb.pp("proj"), dim, dim, true)?;
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale: 1.0 / ((dim / num_heads) as f64).sqrt(),
        })
    }
}

impl Module for Attention {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .transpose(0, 1)?
            .transpose(2, 3)?;
        let q = (qkv.i(0)? * self.scale)?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let attn = candle_nn::ops::softmax_last_dim(&q.matmul(&k.t()?)?)?;
        let attn = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&attn)
    }
}

#[derive(Debug)]
struct Block {
    norm1: LayerNorm,
    attn: Attention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Block {
    fn new(vb: VarBuilder, dim: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: Attention::new(vb.pp("attn"), dim, num_heads)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self
            .ls1
            .forward(&self.attn.forward(&self.norm1.forward(xs)?)?)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?;
        xs + residual
    }
}

#[derive(Debug)]
struct PatchEmbed {
    proj: candle_nn::Conv2d,
    patch_size: usize,
}

impl PatchEmbed {
    fn new(
        vb: VarBuilder,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
    ) -> CandleResult<Self> {
        let config = candle_nn::Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        Ok(Self {
            proj: candle_nn::conv2d(in_channels, embed_dim, patch_size, config, vb.pp("proj"))?,
            patch_size,
        })
    }
}

impl Module for PatchEmbed {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        if h % self.patch_size != 0 || w % self.patch_size != 0 {
            candle_core::bail!("image size must be divisible by patch size");
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
pub struct Pi3DinoEncoder {
    patch_embed: PatchEmbed,
    cls_token: Tensor,
    register_tokens: Tensor,
    pos_embed: Tensor,
    interpolated_pos_encoding_cache: Mutex<HashMap<(usize, usize), Tensor>>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    patch_size: usize,
}

impl Pi3DinoEncoder {
    pub fn new(vb: VarBuilder) -> CandleResult<Self> {
        Self::new_with_in_channels(vb, 3)
    }

    pub fn new_with_in_channels(vb: VarBuilder, in_channels: usize) -> CandleResult<Self> {
        let embed_dim = 1024;
        let patch_size = 14;
        let patch_embed =
            PatchEmbed::new(vb.pp("patch_embed"), patch_size, in_channels, embed_dim)?;
        let cls_token = vb.get((1, 1, embed_dim), "cls_token")?;
        let register_tokens = vb.get((1, 4, embed_dim), "register_tokens")?;
        let pos_embed = vb.get((1, 1370, embed_dim), "pos_embed")?;
        let norm = layer_norm(embed_dim, 1e-6, vb.pp("norm"))?;
        let vb_b = vb.pp("blocks");
        let blocks = (0..24)
            .map(|idx| Block::new(vb_b.pp(idx.to_string()), embed_dim, 16))
            .collect::<CandleResult<Vec<_>>>()?;

        Ok(Self {
            patch_embed,
            cls_token,
            register_tokens,
            pos_embed,
            interpolated_pos_encoding_cache: Mutex::new(HashMap::new()),
            blocks,
            norm,
            patch_size,
        })
    }

    fn interpolate_pos_encoding(
        &self,
        xs: &Tensor,
        width: usize,
        height: usize,
    ) -> CandleResult<Tensor> {
        let npatch = xs.dim(1)? - 1;
        let n = self.pos_embed.dim(1)? - 1;
        if npatch == n && width == height {
            return Ok(self.pos_embed.clone());
        }

        let class_pos_embed = self.pos_embed.i((.., ..1))?;
        let patch_pos_embed = self.pos_embed.i((.., 1..))?;
        let dim = xs.dim(D::Minus1)?;
        let target_h = width / self.patch_size;
        let target_w = height / self.patch_size;
        let cache_key = (target_h, target_w);
        if let Some(cached) = self
            .interpolated_pos_encoding_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .get(&cache_key)
            .cloned()
        {
            return Ok(cached);
        }
        let sqrt_n = (n as f64).sqrt() as usize;
        let patch_pos_embed = resize_patch_pos_embed(
            &patch_pos_embed,
            sqrt_n,
            target_h,
            target_w,
            dim,
            xs.device(),
        )?;
        let interpolated = Tensor::cat(&[&class_pos_embed, &patch_pos_embed], 1)?;
        self.interpolated_pos_encoding_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .insert(cache_key, interpolated.clone());
        Ok(interpolated)
    }

    fn prepare_tokens(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, _channels, width, height) = xs.dims4()?;
        let xs = self.patch_embed.forward(xs)?;
        let cls_token = self
            .cls_token
            .broadcast_as((batch, 1, self.cls_token.dim(D::Minus1)?))?;
        let xs = Tensor::cat(&[&cls_token, &xs], 1)?;
        let xs = xs.broadcast_add(&self.interpolate_pos_encoding(&xs, width, height)?)?;
        let reg_tokens =
            self.register_tokens
                .broadcast_as((batch, 4, self.register_tokens.dim(D::Minus1)?))?;
        Tensor::cat(&[&xs.i((.., ..1))?, &reg_tokens, &xs.i((.., 1..))?], 1)
    }

    pub fn forward_prepared_tokens(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.prepare_tokens(xs)
    }

    pub fn forward_patch_tokens(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
        let mut xs = self.prepare_tokens(xs)?;
        for (idx, block) in self.blocks.iter().enumerate() {
            let t = std::time::Instant::now();
            xs = block.forward(&xs)?;
            if stage_time && (idx < 3 || idx + 1 == self.blocks.len()) {
                eprintln!("[stage]     enc.block{idx:02}: {:.3}s", t.elapsed().as_secs_f64());
            }
        }
        let xs = self.norm.forward(&xs)?;
        xs.i((.., 5..))
    }
}

pub fn load_pi3_encoder(vb: VarBuilder) -> CandleResult<Pi3DinoEncoder> {
    Pi3DinoEncoder::new(vb)
}

pub fn load_pi3_depth_encoder(vb: VarBuilder) -> CandleResult<Pi3DinoEncoder> {
    Pi3DinoEncoder::new_with_in_channels(vb, 2)
}

impl Pi3DinoEncoder {
    #[cfg(test)]
    pub(crate) fn cached_pos_encoding_entries(&self) -> usize {
        self.interpolated_pos_encoding_cache
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .len()
    }
}

fn resize_patch_pos_embed(
    patch_pos_embed: &Tensor,
    source_hw: usize,
    target_h: usize,
    target_w: usize,
    dim: usize,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    let cpu = candle_core::Device::Cpu;
    let dtype = patch_pos_embed.dtype();
    let flat = patch_pos_embed
        .to_device(&cpu)?
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let horizontal = compute_aa_cubic_weights(source_hw, target_w);
    let vertical = compute_aa_cubic_weights(source_hw, target_h);
    let mut resized = vec![0f32; target_h * target_w * dim];
    for channel in 0..dim {
        let mut temp = vec![0f32; source_hw * target_w];
        for y in 0..source_hw {
            for out_x in 0..target_w {
                let spec = &horizontal[out_x];
                let mut value = 0f32;
                for (offset, weight) in spec.weights.iter().enumerate() {
                    let src_x = spec.start + offset;
                    value += flat[(y * source_hw + src_x) * dim + channel] * *weight;
                }
                temp[y * target_w + out_x] = value;
            }
        }

        for out_y in 0..target_h {
            let spec = &vertical[out_y];
            for out_x in 0..target_w {
                let mut value = 0f32;
                for (offset, weight) in spec.weights.iter().enumerate() {
                    let src_y = spec.start + offset;
                    value += temp[src_y * target_w + out_x] * *weight;
                }
                resized[(out_y * target_w + out_x) * dim + channel] = value;
            }
        }
    }

    Tensor::from_vec(resized, (1, target_h * target_w, dim), device)?.to_dtype(dtype)
}
