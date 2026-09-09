use candle_core::{D, DType, IndexOp, Result as CandleResult, Tensor};
use candle_nn::Module;

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
        if half % 2 != 0 {
            candle_core::bail!("rope token_dim must be divisible by 4, got {token_dim}");
        }
        let max_position = positions.flatten_all()?.max(0)?.to_scalar::<u32>()? as usize + 1;

        let (cos, sin) = self.cos_sin(half, max_position, positions.device())?;
        let pos_y = positions.i((.., .., 0))?;
        let pos_x = positions.i((.., .., 1))?;
        let cos_y = self.apply_embedding(&pos_y, &cos)?;
        let sin_y = self.apply_embedding(&pos_y, &sin)?;
        let cos_x = self.apply_embedding(&pos_x, &cos)?;
        let sin_x = self.apply_embedding(&pos_x, &sin)?;
        let quarter = half / 2;
        // Sign-flipped sin tables fold the rotate_half negation into the
        // precomputed tables (exact in IEEE: -(t*s) == t*(-s)).
        let sin_alt_y = Tensor::cat(
            &[
                &sin_y.i((.., .., .., ..quarter))?.affine(-1.0, 0.0)?,
                &sin_y.i((.., .., .., quarter..))?,
            ],
            D::Minus1,
        )?;
        let sin_alt_x = Tensor::cat(
            &[
                &sin_x.i((.., .., .., ..quarter))?.affine(-1.0, 0.0)?,
                &sin_x.i((.., .., .., quarter..))?,
            ],
            D::Minus1,
        )?;
        // The fused rope kernels index the tables with flat (b, n) rows, so
        // both must be contiguous; sin_alt_full comes back non-contiguous
        // from the cat/affine chain on GPU backends (the old broadcast path
        // never cared).
        Ok(RopeEmbeddings {
            cos_full: Tensor::cat(&[&cos_y, &cos_x], D::Minus1)?.contiguous()?,
            sin_alt_full: Tensor::cat(&[&sin_alt_y, &sin_alt_x], D::Minus1)?.contiguous()?,
        })
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
}

#[derive(Debug, Clone)]
pub(crate) struct RopeEmbeddings {
    /// cos tables for both halves concatenated: (b, 1, n, token_dim).
    cos_full: Tensor,
    /// sin tables for both halves with the rotate_half negation baked in:
    /// [-sin_y2, sin_y1, -sin_x2, sin_x1], (b, 1, n, token_dim).
    sin_alt_full: Tensor,
}

/// Whole-tensor form of the per-half rope
/// (`out = cat(rotate_half_pair(y)*sin + y*cos, rotate_half_pair(x)*sin + x*cos)`).
///
/// `pairs_rot` gathers the rotated partners `[y2, y1, x2, x1]` and
/// `sin_alt_full` carries the negations, so this is bit-identical to the
/// old path in 8 dispatches instead of ~17. Rope runs per attention per
/// layer, so the saved dispatches cut straight into the WDDM submission tax.
fn rope_apply(tokens: &Tensor, embeddings: &RopeEmbeddings) -> CandleResult<Tensor> {
    let d = tokens.dim(D::Minus1)?;
    let half = d / 2;
    let quarter = half / 2;
    let y2 = tokens.i((.., .., .., quarter..half))?;
    let y1 = tokens.i((.., .., .., ..quarter))?;
    let x2 = tokens.i((.., .., .., (half + quarter)..))?;
    let x1 = tokens.i((.., .., .., half..(half + quarter)))?;
    let pairs_rot = Tensor::cat(&[&y2, &y1, &x2, &x1], D::Minus1)?;
    tokens.broadcast_mul(&embeddings.cos_full)?
        + pairs_rot.broadcast_mul(&embeddings.sin_alt_full)?
}

/// LayerNorm(+affine) -> rope in ONE dispatch on vulkan/wgpu
/// (candle_nn::ops::layernorm_rope_fused): the fused kernel reads the
/// (possibly strided) qkv head view directly, so the norm's slow tensor-op
/// path, both surrounding .contiguous() copies, and the 8-op rope chain all
/// collapse into a single kernel. Output is contiguous (b, h, n, head_dim).
/// Everything else (cpu/cuda, other head dims) keeps the composed path,
/// which is exactly the previous op sequence.
pub(crate) fn apply_layernorm_rope(
    q: &Tensor,
    q_norm: &candle_nn::LayerNorm,
    embeddings: &RopeEmbeddings,
) -> CandleResult<Tensor> {
    let fused_ok = (q.device().is_vulkan() || q.device().is_wgpu())
        && q.rank() == 4
        && q.dim(3)? == 64
        && q.stride()[3] == 1
        && q.dtype() == DType::F32
        && q_norm.bias().is_some();
    if fused_ok {
        return candle_nn::ops::layernorm_rope_fused(
            q,
            q_norm.weight(),
            q_norm.bias().unwrap(),
            q_norm.eps() as f32,
            &embeddings.cos_full,
            &embeddings.sin_alt_full,
        );
    }
    let q = q_norm.forward(q)?.contiguous()?;
    rope_apply(&q, embeddings)
}

/// RoPE-only fused variant (no norm), same single kernel with apply_norm=0.
/// Accepts strided head views (last dim contiguous), so call sites can drop
/// the leading .contiguous() too. Output is contiguous.
pub(crate) fn apply_rope(tokens: &Tensor, embeddings: &RopeEmbeddings) -> CandleResult<Tensor> {
    let fused_ok = (tokens.device().is_vulkan() || tokens.device().is_wgpu())
        && tokens.rank() == 4
        && tokens.dim(3)? == 64
        && tokens.stride()[3] == 1
        && tokens.dtype() == DType::F32;
    if fused_ok {
        return candle_nn::ops::rope_fused(
            tokens,
            &embeddings.cos_full,
            &embeddings.sin_alt_full,
        );
    }
    rope_apply(tokens, embeddings)
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
                positions.push((y + 1) as i64);
                positions.push((x + 1) as i64);
            }
        }
    }
    Tensor::from_vec(positions, (batch, h * w, 2), device)
}

/// Peak bytes allowed for one attention scores buffer. A pi3 decoder odd-block
/// single-shot scores tensor (heads × 8244² × f32) is ~4 GiB; allocating it
/// fragments the CUDA memory pool (+4 GiB reserved step) and OOMs wgpu on
/// 12 GiB cards. Override with LUX3D_MAX_SDPA_SCORES_BYTES.
fn max_sdpa_scores_bytes() -> usize {
    static BUDGET: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("LUX3D_MAX_SDPA_SCORES_BYTES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(768 * 1024 * 1024)
    })
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
    // Softmax over the last dim is row-independent, so chunking only trades
    // peak score-buffer memory for more kernel launches; a single shot keeps
    // the GEMMs big (much faster on every GPU backend) — but only while the
    // scores buffer fits the budget, otherwise the memory pools fragment or
    // OOM (see max_sdpa_scores_bytes).
    let kv_len = k_t.dim(D::Minus1)?;
    let scores_elem_bytes = if q.dtype() == DType::F32 { 4 } else { 2 };
    let scores_bytes_per_row = q.dim(0)? * q.dim(1)? * kv_len * scores_elem_bytes;
    let budget_chunk = (max_sdpa_scores_bytes() / scores_bytes_per_row.max(1)).max(1);
    let effective_chunk = chunk_size.min(budget_chunk).min(q_seq).max(1);
    if effective_chunk >= q_seq {
        let scores = q.matmul(&k_t)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        return attn.matmul(v);
    }
    let mut outputs = Vec::new();

    let mut start = 0usize;
    while start < q_seq {
        let len = (q_seq - start).min(effective_chunk);
        let q_chunk = q.narrow(2, start, len)?.contiguous()?;
        let scores = q_chunk.matmul(&k_t)?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
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
