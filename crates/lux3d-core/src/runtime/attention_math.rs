use candle_core::{D, DType, IndexOp, Result as CandleResult, Tensor};

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rope2d {
    base: f32,
}

impl Rope2d {
    pub(crate) fn new(base: f32) -> Self {
        Self { base }
    }

    pub(crate) fn embeddings(
        &self,
        positions: &Tensor,
        token_dim: usize,
    ) -> CandleResult<RopeEmbeddings> {
        let positions = positions.to_dtype(DType::U32)?;
        let half = token_dim / 2;
        let max_position = positions.flatten_all()?.max(0)?.to_scalar::<u32>()? as usize + 1;

        let (cos, sin) = self.cos_sin(half, max_position, positions.device())?;
        let pos_y = positions.i((.., .., 0))?;
        let pos_x = positions.i((.., .., 1))?;
        Ok(RopeEmbeddings {
            cos_y: self.apply_embedding(&pos_y, &cos)?,
            sin_y: self.apply_embedding(&pos_y, &sin)?,
            cos_x: self.apply_embedding(&pos_x, &cos)?,
            sin_x: self.apply_embedding(&pos_x, &sin)?,
        })
    }

    pub(crate) fn apply_with_embeddings(
        &self,
        tokens: &Tensor,
        embeddings: &RopeEmbeddings,
    ) -> CandleResult<Tensor> {
        let (_b, _heads, _n, d) = tokens.dims4()?;
        let half = d / 2;
        let y = tokens.i((.., .., .., ..half))?;
        let x = tokens.i((.., .., .., half..))?;
        let y = self.apply_rope_embedded(&y, &embeddings.cos_y, &embeddings.sin_y)?;
        let x = self.apply_rope_embedded(&x, &embeddings.cos_x, &embeddings.sin_x)?;
        Tensor::cat(&[&y, &x], D::Minus1)
    }

    fn cos_sin(
        &self,
        dim: usize,
        seq_len: usize,
        device: &candle_core::Device,
    ) -> CandleResult<(Tensor, Tensor)> {
        let mut inv_freq = Vec::with_capacity(dim);
        for i in (0..dim).step_by(2) {
            let freq = 1.0f32 / self.base.powf(i as f32 / dim as f32);
            inv_freq.push(freq);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        let t = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    fn apply_embedding(&self, positions: &Tensor, table: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq_len) = positions.dims2()?;
        let pos = positions.to_dtype(DType::U32)?.flatten_all()?;
        table
            .embedding(&pos)?
            .reshape((batch, seq_len, table.dim(D::Minus1)?))?
            .unsqueeze(1)
    }

    fn apply_rope_embedded(
        &self,
        tokens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> CandleResult<Tensor> {
        let rotated = rotate_half(tokens)?;
        tokens.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RopeEmbeddings {
    cos_y: Tensor,
    sin_y: Tensor,
    cos_x: Tensor,
    sin_x: Tensor,
}

pub(crate) fn rotate_half(xs: &Tensor) -> CandleResult<Tensor> {
    let d = xs.dim(D::Minus1)?;
    let half = d / 2;
    let x1 = xs.i((.., .., .., ..half))?;
    let x2 = xs.i((.., .., .., half..))?;
    let neg_x2 = x2.affine(-1.0, 0.0)?;
    Tensor::cat(&[&neg_x2, &x1], D::Minus1)
}

pub(crate) fn position_getter(
    batch: usize,
    h: usize,
    w: usize,
    device: &candle_core::Device,
) -> CandleResult<Tensor> {
    let mut positions = Vec::with_capacity(batch * h * w * 2);
    for _ in 0..batch {
        for y in 0..h {
            for x in 0..w {
                positions.push(y as i64);
                positions.push(x as i64);
            }
        }
    }
    Tensor::from_vec(positions, (batch, h * w, 2), device)
}

pub(crate) fn exact_query_chunked_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    chunk_size: usize,
) -> CandleResult<Tensor> {
    let q = q.affine(scale as f64, 0.0)?;
    let q_seq = q.dim(2)?;
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let mut outputs = Vec::new();

    let mut start = 0usize;
    while start < q_seq {
        let len = (q_seq - start).min(chunk_size);
        let q_chunk = q.narrow(2, start, len)?.contiguous()?;
        let scores = q_chunk.matmul(&k_t)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        outputs.push(attn.matmul(v)?);
        start += len;
    }

    let output_refs = outputs.iter().collect::<Vec<_>>();
    Tensor::cat(&output_refs, 2)
}

pub(crate) fn exact_sdpa_heads(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
) -> CandleResult<Tensor> {
    let in_dtype = q.dtype();
    let q = q.to_dtype(DType::F32)?.contiguous()?;
    let k = k.to_dtype(DType::F32)?.contiguous()?;
    let v = v.to_dtype(DType::F32)?.contiguous()?;
    let scores = q
        .matmul(&k.transpose(2, 3)?.contiguous()?)?
        .affine(scale as f64, 0.0)?;
    let attn = candle_nn::ops::softmax_last_dim(&scores)?;
    attn.matmul(&v)?.to_dtype(in_dtype)
}
