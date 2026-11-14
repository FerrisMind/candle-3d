//! TRELLIS.2 sparse tensor primitives: coordinate hashmap, submanifold 3x3x3
//! sparse convolution, C2S/S2C coordinate ops and exact sparse full attention
//! helpers, shared by the shape SLAT flow DiT (T4) and later sparse VAEs (T5+).
//!
//! Reference: `trellis2/modules/sparse/basic.py` (`SparseTensor`), and
//! `trellis2/modules/sparse/attention/full_attn.py`
//! (`sparse_scaled_dot_product_attention`).
//!
//! Sparse tensor model (single batch element, `num_samples = 1`):
//! - `SparseFeats { coords: Vec<[i32; 3]>, feats: Tensor }` with row `i` of
//!   `feats` (`[N, C]`, F32) corresponding to `coords[i] = (x, y, z)`.
//! - Coordinates are non-negative, below the grid resolution and unique.
//! - Row order is preserved by every op here except `spatial_to_channel`,
//!   which sorts parents row-major exactly like the reference
//!   `SparseSpatial2Channel` (`code.unique()` sorts ascending).
//!
//! Sparse convolution (locked decision 7): submanifold 3x3x3, stride 1,
//! padding 1, output coords == input coords:
//! ```text
//! out[i, o] = bias[o] + sum_{kd,kh,kw} sum_c W[o,kd,kh,kw,c] * x[coord_i + (kd-1,kh-1,kw-1), c]
//! ```
//! with missing neighbors contributing zero. The canonical weights store the
//! kernel in the flex_gemm layout `[out, kd, kh, kw, in]` (NOT the PyTorch
//! `[out, in, kd, kh, kw]`); `SparseConv3d::load` transposes it into
//! `[27, Cout, Cin]` (`k = kd*9 + kh*3 + kw`, kw fastest, matching the
//! in-memory flattening of `(kd, kh, kw)`) and the forward is 27
//! hashmap-gather + dense GEMM pairs accumulated in F32.

use std::collections::HashMap;

use candle_core::{DType, Result as CandleResult, Tensor};

use super::attention_math::exact_query_chunked_sdpa;
use crate::weights::VarBuilder;

/// A single-batch sparse feature tensor: row `i` of `feats` belongs to
/// `coords[i]`.
#[derive(Debug, Clone)]
pub(crate) struct SparseFeats {
    /// `(x, y, z)` integer latent coordinates, batch dim dropped.
    pub coords: Vec<[i32; 3]>,
    /// `[N, C]` F32 features.
    pub feats: Tensor,
}

impl SparseFeats {
    /// Crate-internal constructor; consumed by the shape VAE decoder (T5).
    #[allow(dead_code)]
    pub fn new(coords: Vec<[i32; 3]>, feats: Tensor) -> CandleResult<Self> {
        let rows = feats.dim(0)?;
        if rows != coords.len() {
            candle_core::bail!(
                "SparseFeats coords/features mismatch: {} coords vs {} feature rows",
                coords.len(),
                rows
            );
        }
        if feats.dtype() != DType::F32 {
            candle_core::bail!("SparseFeats feats must be F32, got {:?}", feats.dtype());
        }
        Ok(Self { coords, feats })
    }
}

/// Coordinate hashmap `coord -> row index`, built once per grid and reused
/// across the 27 conv offsets and mesh extraction.
#[derive(Debug, Clone)]
pub struct CoordMap {
    map: HashMap<[i32; 3], u32>,
}

impl CoordMap {
    /// Build from unique coordinates; later duplicates overwrite earlier
    /// entries (inputs are required to be duplicate-free).
    pub fn new(coords: &[[i32; 3]]) -> Self {
        let mut map = HashMap::with_capacity(coords.len());
        for (index, coord) in coords.iter().enumerate() {
            map.insert(*coord, index as u32);
        }
        Self { map }
    }

    pub fn get(&self, coord: [i32; 3]) -> Option<u32> {
        self.map.get(&coord).copied()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Kernel offset `(dx, dy, dz)` for flat index `k = kd*9 + kh*3 + kw`
/// (kw fastest), i.e. `delta = (kd-1, kh-1, kw-1)` for a 3^3 kernel with
/// padding 1.
pub fn kernel_offset_delta(k: usize) -> [i32; 3] {
    debug_assert!(k < 27);
    [
        (k / 9) as i32 - 1,
        ((k % 9) / 3) as i32 - 1,
        (k % 3) as i32 - 1,
    ]
}

/// Vulkan compute dispatches are limited to 65535 workgroups per dimension
/// (spec minimum; NVIDIA reports 65535 for Y/Z). candle's Vulkan
/// `index_select` dispatches the index count as the Y workgroup count
/// without clamping, so gathers with more than 65535 indices are silently
/// corrupt on that backend (measured on an RTX 3060: 65535 indices exact,
/// 65536 indices ~96% wrong rows). The sparse ops below gather with
/// production-scale index counts (conv pair counts and C2S child counts
/// exceed 65535 at resolution 512), so gathers on Vulkan are split into
/// limit-sized chunks. A gather is row-wise and exact, so chunking is
/// bit-identical on every backend; CPU/CUDA keep the single-dispatch path.
const INDEX_SELECT_DISPATCH_CHUNK: usize = 65_535;

fn index_select_rows(feats: &Tensor, ids: &Tensor) -> CandleResult<Tensor> {
    let rows = ids.dim(0)?;
    if rows <= INDEX_SELECT_DISPATCH_CHUNK || !feats.device().is_vulkan() {
        return feats.index_select(ids, 0);
    }
    let mut parts = Vec::new();
    let mut start = 0usize;
    while start < rows {
        let len = (rows - start).min(INDEX_SELECT_DISPATCH_CHUNK);
        parts.push(feats.index_select(&ids.narrow(0, start, len)?, 0)?);
        start += len;
    }
    Tensor::cat(&parts, 0)
}

/// Submanifold 3x3x3 sparse convolution (`SparseConv3d`): output coords equal
/// input coords; the kernel is stored as `[27, Cout, Cin]` (flex_gemm layout
/// transposed by the loader).
#[derive(Debug, Clone)]
pub struct SparseConv3d {
    /// `[27, Cout, Cin]`, index `k = kd*9 + kh*3 + kw` (kw fastest).
    weight: Tensor,
    bias: Option<Tensor>,
    in_channels: usize,
    out_channels: usize,
}

impl SparseConv3d {
    /// Load from a VarBuilder rooted at the conv tensors. The checkpoint
    /// stores `weight` as `[Cout, kd, kh, kw, Cin]` (flex_gemm layout); it is
    /// transposed into `[27, Cout, Cin]` in F32.
    pub fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        bias: bool,
    ) -> CandleResult<Self> {
        // GGUF packages store the rank-5 kernel flattened (candle's GGUF
        // reader caps tensor rank at 4); the helper reshapes it back.
        let weight = super::nn_blocks::get_weight_shaped(
            &vb,
            (out_channels, 3, 3, 3, in_channels),
            "weight",
        )?;
        // Flatten (kd, kh, kw) in memory order (kw fastest) -> k = kd*9 +
        // kh*3 + kw, then move the kernel axis in front.
        let weight = weight
            .to_dtype(DType::F32)?
            .reshape((out_channels, 27, in_channels))?
            .permute((1, 0, 2))?
            .contiguous()?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias")?.to_dtype(DType::F32)?)
        } else {
            None
        };
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
        })
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// The transposed kernel `[27, Cout, Cin]`.
    pub fn weight27(&self) -> &Tensor {
        &self.weight
    }

    /// `coords: [N]`, `feats: [N, Cin]` -> `[N, Cout]` F32. 27 gather + GEMM
    /// pairs accumulated in F32; missing neighbors contribute zero.
    pub fn forward(&self, coords: &[[i32; 3]], feats: &Tensor) -> CandleResult<Tensor> {
        let (rows, channels) = feats.dims2()?;
        if rows != coords.len() || channels != self.in_channels {
            candle_core::bail!(
                "SparseConv3d expects [N, {}] with N = {} coords, got [{rows}, {channels}]",
                self.in_channels,
                coords.len()
            );
        }
        let device = feats.device();
        let map = CoordMap::new(coords);
        let mut out = Tensor::zeros((rows, self.out_channels), DType::F32, device)?;
        for k in 0..27 {
            let delta = kernel_offset_delta(k);
            // Destination rows (input rows with an active neighbor at this
            // offset) and the matching source rows.
            let mut dest: Vec<u32> = Vec::with_capacity(rows);
            let mut src: Vec<u32> = Vec::with_capacity(rows);
            for (index, coord) in coords.iter().enumerate() {
                if let Some(source) = map.get([
                    coord[0] + delta[0],
                    coord[1] + delta[1],
                    coord[2] + delta[2],
                ]) {
                    dest.push(index as u32);
                    src.push(source);
                }
            }
            if dest.is_empty() {
                continue;
            }
            let pairs = dest.len();
            let dest = Tensor::from_vec(dest, (pairs,), device)?;
            let src = Tensor::from_vec(src, (pairs,), device)?;
            let gathered = index_select_rows(feats, &src)?; // [n, Cin]
            let kernel = self.weight.narrow(0, k, 1)?.squeeze(0)?; // [Cout, Cin]
            let product = gathered.matmul(&kernel.transpose(0, 1)?)?; // [n, Cout]
            out = out.index_add(&dest, &product, 0)?;
        }
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(bias)?;
        }
        Ok(out)
    }
}

/// Binarized subdivision: `leaf = sub > 0` per parent (`[N, 8]` logits ->
/// `[N, 8]` bools in row-major subidx order).
fn subdivision_leaves(subdiv_logits: &Tensor, parents: usize) -> CandleResult<Vec<[bool; 8]>> {
    let (rows, channels) = subdiv_logits.dims2()?;
    if rows != parents || channels != 8 {
        candle_core::bail!("subdivision logits must be [{parents}, 8], got [{rows}, {channels}]");
    }
    let values = subdiv_logits
        .to_device(&candle_core::Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(values
        .chunks_exact(8)
        .map(|chunk| {
            let mut leaf = [false; 8];
            for (i, &v) in chunk.iter().enumerate() {
                leaf[i] = v > 0.0;
            }
            leaf
        })
        .collect())
}

/// Output of `channel_to_spatial`.
#[derive(Debug, Clone)]
pub struct Channel2SpatialOutput {
    /// Child coords `parent * 2 + (dx, dy, dz)`, grouped by parent in parent
    /// order, children ascending in `subidx`.
    pub coords: Vec<[i32; 3]>,
    /// `[M, child_channels]` selected features.
    pub feats: Tensor,
    /// Binarized subdivision per parent, `[N, 8]` F32 (row-major subidx).
    pub sub_binary: Tensor,
}

/// C2S (`SparseChannel2Spatial`, factor 2): subdivision logits `[N, 8]`
/// (`leaf = logits > 0`) select, per parent, which children survive.
///
/// `feats: [N, C]` with `C` a multiple of 8, channel layout
/// `[child, channel]`: reshaped to `[N*8, C/8]`, output row `j` is
/// `feats[parent*8 + subidx]` with `subidx = dx + 2*dy + 4*dz`. Children are
/// emitted grouped by parent in parent order, ascending `subidx` (row-major
/// nonzero order, exactly `sub.nonzero()[:, -1]` in the reference).
pub fn channel_to_spatial(
    coords: &[[i32; 3]],
    feats: &Tensor,
    subdiv_logits: &Tensor,
) -> CandleResult<Channel2SpatialOutput> {
    let (rows, channels) = feats.dims2()?;
    if rows != coords.len() {
        candle_core::bail!(
            "channel_to_spatial coords/features mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    if channels % 8 != 0 {
        candle_core::bail!(
            "channel_to_spatial expects channels divisible by 8 (layout [child, channel]), got {channels}"
        );
    }
    let leaves = subdivision_leaves(subdiv_logits, rows)?;
    let child_channels = channels / 8;
    let mut out_coords: Vec<[i32; 3]> = Vec::new();
    let mut src_rows: Vec<u32> = Vec::new();
    for (parent, coord) in coords.iter().enumerate() {
        for subidx in 0..8usize {
            if leaves[parent][subidx] {
                let dx = (subidx % 2) as i32;
                let dy = ((subidx / 2) % 2) as i32;
                let dz = (subidx / 4) as i32;
                out_coords.push([coord[0] * 2 + dx, coord[1] * 2 + dy, coord[2] * 2 + dz]);
                src_rows.push((parent * 8 + subidx) as u32);
            }
        }
    }
    let device = feats.device();
    let flat = feats.reshape((rows * 8, child_channels))?;
    let src = Tensor::from_vec(src_rows, (out_coords.len(),), device)?;
    let out_feats = index_select_rows(&flat, &src)?;
    let sub_binary = Tensor::from_vec(
        leaves
            .iter()
            .flat_map(|leaf| leaf.iter().map(|&l| l as u8 as f32).collect::<Vec<_>>())
            .collect(),
        (rows, 8),
        device,
    )?;
    Ok(Channel2SpatialOutput {
        coords: out_coords,
        feats: out_feats,
        sub_binary,
    })
}

/// The C2S skip ("skin") connection: repeat every channel `repeats` times
/// (`x.feats.repeat_interleave(repeats, dim=1)`) so a `[N, Cin]` tensor
/// becomes `[N, 8 * out_channels]` before `channel_to_spatial`.
/// `repeats = out_channels // (Cin / 8)` at the call site.
pub fn repeat_channels(feats: &Tensor, repeats: usize) -> CandleResult<Tensor> {
    if repeats == 1 {
        return Ok(feats.clone());
    }
    let (rows, channels) = feats.dims2()?;
    feats
        .unsqueeze(2)? // [N, C, 1]
        .broadcast_as((rows, channels, repeats))?
        .contiguous()?
        .reshape((rows, channels * repeats))
}

/// Output of `spatial_to_channel`.
#[derive(Debug, Clone)]
pub struct Spatial2ChannelOutput {
    /// Parent coords `child // 2`, row-major sorted (ascending `code`, the
    /// reference `code.unique()` order).
    pub coords: Vec<[i32; 3]>,
    /// `[Np, 8 * C]` F32 with channel layout `[parent, child, channel]`
    /// (empty child slots are zero, matching the reference zero-init).
    pub feats: Tensor,
}

/// S2C (`SparseSpatial2Channel`, factor 2, DIM 3): each child contributes its
/// features to parent `child // 2` at slot `subidx = dx + 2*dy + 4*dz` where
/// `(dx, dy, dz) = child % 2`. Parents are sorted row-major (lexicographic
/// `(x, y, z)` — identical to the reference row-major code order for a single
/// batch). Encoder-side op; kept for tests and round-trip validation.
pub fn spatial_to_channel(
    coords: &[[i32; 3]],
    feats: &Tensor,
) -> CandleResult<Spatial2ChannelOutput> {
    let (rows, channels) = feats.dims2()?;
    if rows != coords.len() {
        candle_core::bail!(
            "spatial_to_channel coords/features mismatch: {} coords vs {rows} rows",
            coords.len()
        );
    }
    // Dedupe parents and sort row-major (lexicographic (x, y, z) == the
    // reference ascending row-major `code` for one batch element).
    let mut parents: Vec<[i32; 3]> = coords
        .iter()
        .map(|c| [c[0] / 2, c[1] / 2, c[2] / 2])
        .collect();
    parents.sort_unstable();
    parents.dedup();
    let parent_index: HashMap<[i32; 3], usize> =
        parents.iter().enumerate().map(|(i, p)| (*p, i)).collect();
    let mut out = vec![0.0_f32; parents.len() * 8 * channels];
    let feats = feats.to_device(&candle_core::Device::Cpu)?;
    let values = feats.flatten_all()?.to_vec1::<f32>()?;
    for (child, coord) in coords.iter().enumerate() {
        let parent = [coord[0] / 2, coord[1] / 2, coord[2] / 2];
        let slot = parent_index[&parent];
        let dx = (coord[0] % 2) as usize;
        let dy = (coord[1] % 2) as usize;
        let dz = (coord[2] % 2) as usize;
        let subidx = dx + 2 * dy + 4 * dz;
        let dst = (slot * 8 + subidx) * channels;
        out[dst..dst + channels].copy_from_slice(&values[child * channels..(child + 1) * channels]);
    }
    let device = feats.device();
    let parent_count = parents.len();
    Ok(Spatial2ChannelOutput {
        coords: parents,
        feats: Tensor::from_vec(out, (parent_count, 8 * channels), device)?,
    })
}

/// Exact block-diagonal full attention for a single batch element: self
/// attention over `qkv: [T, 3, H, head_dim]` -> `[T, H, head_dim]`, scale
/// `1/sqrt(head_dim)`, softmax over all T keys (identical to the dense path
/// with `B = 1`). Queries are chunked; keys are never chunked, so softmax
/// stays exact and no `[H, T, T]` buffer larger than `[H, chunk, T]` is
/// materialized.
pub fn sparse_self_attention(qkv: &Tensor, chunk_size: usize) -> CandleResult<Tensor> {
    let (tokens, fused, heads, head_dim) = qkv.dims4()?;
    if fused != 3 {
        candle_core::bail!("sparse_self_attention expects [T, 3, H, D], got {qkv:?}");
    }
    let qkv = qkv.unsqueeze(0)?; // [1, T, 3, H, D]
    let q = qkv.narrow(2, 0, 1)?.squeeze(2)?; // [1, T, H, D]
    let k = qkv.narrow(2, 1, 1)?.squeeze(2)?;
    let v = qkv.narrow(2, 2, 1)?.squeeze(2)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out = exact_query_chunked_sdpa(
        &q.permute((0, 2, 1, 3))?, // [1, H, T, D]
        &k.permute((0, 2, 1, 3))?,
        &v.permute((0, 2, 1, 3))?,
        scale,
        chunk_size,
    )?;
    Ok(out
        .permute((0, 2, 1, 3))?
        .reshape((tokens, heads, head_dim))?)
}

/// Exact cross attention for a single batch element: `q: [T, H, head_dim]`
/// from the sparse tokens, `kv: [Tkv, 2, H, head_dim]` from the conditioning
/// context -> `[T, H, head_dim]`, scale `1/sqrt(head_dim)`, softmax over all
/// `Tkv` context keys.
pub fn sparse_cross_attention(q: &Tensor, kv: &Tensor, chunk_size: usize) -> CandleResult<Tensor> {
    let (tokens, heads, head_dim) = q.dims3()?;
    let (_kv_tokens, fused, kv_heads, kv_head_dim) = kv.dims4()?;
    if fused != 2 || kv_heads != heads || kv_head_dim != head_dim {
        candle_core::bail!(
            "sparse_cross_attention expects kv [Tkv, 2, H, D], got {kv:?} for q {q:?}"
        );
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let out = exact_query_chunked_sdpa(
        &q.unsqueeze(0)?.permute((0, 2, 1, 3))?, // [1, H, T, D]
        &kv.narrow(1, 0, 1)?
            .squeeze(1)?
            .unsqueeze(0)?
            .permute((0, 2, 1, 3))?, // k
        &kv.narrow(1, 1, 1)?
            .squeeze(1)?
            .unsqueeze(0)?
            .permute((0, 2, 1, 3))?, // v
        scale,
        chunk_size,
    )?;
    Ok(out
        .permute((0, 2, 1, 3))?
        .reshape((tokens, heads, head_dim))?)
}
