//! Hierarchical SDF grid for the TripoSG VAE decoder.
//!
//! Reference semantics (`triposg/inference_utils.py`
//! `generate_dense_grid_points_gpu`, `find_candidates_band`,
//! `expand_edge_region_fast`, `hierarchical_extract_geometry`; the non-flash
//! path — `use_flash_decoder=False`):
//!
//! - Dense pass at depth `d`: `n = 2^d` coordinates per axis from
//!   `linspace(bounds_min, bounds_max, n)`. The reference builds these
//!   coordinates in torch.float16, so the linspace is replicated EXACTLY
//!   through fp16 arithmetic (endpoints, step and every product/sum are
//!   rounded through fp16, first half filled upward from `start`, second half
//!   downward from `end` — verified bit-identical against torch CPU and CUDA
//!   for every n in {2, 4, 8, ..., 1024}). All n^3 points are queried in
//!   x-fastest order (idx = x + y*n + z*n*n) and the RAW decoder logits
//!   (already negated by the decoder) are stored.
//! - Refinement levels d+1..=hierarchical_depth (N = 2n): the high grid is
//!   initialized by nearest upsample (`high[h] = low[h/2]`, matching
//!   `F.interpolate(scale_factor=2)` default nearest mode); candidate band =
//!   interior low cells with `abs(sigmoid(v) * 2 - 1) < 1.0`
//!   (`find_candidates_band`); optionally culled by 3x3x3 sign-change check
//!   (env `LUX3D_TRIPOSG_REFINE_ALL=1` restores the reference-exact full
//!   band); dilated by Chebyshev radius 2 for n < 512 and radius 1 for
//!   n >= 512 (`expand_edge_region_fast` max_pool kernel 5 / 3); every marked
//!   low cell expands to its 8 high children, which are re-queried at world
//!   coordinate `bounds_min + high_index * (bounds_size / N)` and scattered
//!   into the high grid.
//! - Marching cubes later runs at level 0 on the final grid with the vertex
//!   mapping `v / N * bounds_size + bounds_min` (mcubes scale `bounds_size /
//!   N`), matching the reference `vertices / 2^hierarchical_octree_depth *
//!   bbox_size + bbox_min` — NOT `bounds_size / (N - 1)`.
//!
//! All math is F32 (the canonical parity baseline was generated in F32; the
//! reference's fp16 GRID-VALUE casts only matter for mesh extraction and are
//! applied by the caller when needed).

use candle_core::{DType, Device, Result as CandleResult, Tensor};

use super::triposg_vae::TripoSgVaeDecoder;

/// A decoded SDF field on a regular grid, x-fastest:
/// `values[x + y * n + z * n * n]` is the (negated) decoder logit at the grid
/// point with axis coordinates `(x, y, z)` — the packing order the `mcubes`
/// crate expects (`get_value(x, y, z)`).
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgSdfGrid {
    /// Octree depth the grid was produced at (`dims == 2^depth`).
    pub depth: u32,
    /// Grid side length `n = 2^depth`.
    pub dims: usize,
    /// `[xmin, ymin, zmin, xmax, ymax, zmax]` world bounds.
    pub bounds: [f32; 6],
    /// Raw (negated) decoder logits, x-fastest.
    pub values: Vec<f32>,
}

/// Hard cap: reject `hierarchical_depth > 10` (grid side > 1024).
const MAX_HIERARCHICAL_DEPTH: u32 = 10;

/// Decode the hierarchical SDF grid for `latents [1, 2048, 64]`.
///
/// `bounds` is `[xmin, ymin, zmin, xmax, ymax, zmax]`; the pipeline default
/// is `[-1.005, -1.005, -1.005, 1.005, 1.005, 1.005]`. `chunk` bounds the
/// number of query points per decoder pass (reference `num_chunks=50000`,
/// pipeline default 16384).
///
/// When `hierarchical_depth <= dense_depth` the dense grid is returned
/// directly with `depth = dense_depth`; otherwise every level from
/// `dense_depth + 1` to `hierarchical_depth` is refined and the final grid
/// carries `depth = hierarchical_depth`, `dims = 2^hierarchical_depth`.
pub fn hierarchical_sdf_grid(
    decoder: &TripoSgVaeDecoder,
    latents: &Tensor,
    bounds: [f32; 6],
    dense_depth: u32,
    hierarchical_depth: u32,
    chunk: usize,
) -> CandleResult<TripoSgSdfGrid> {
    if dense_depth == 0 {
        candle_core::bail!(
            "invalid input: TripoSG dense_octree_depth must be at least 1, got {dense_depth}"
        );
    }
    if hierarchical_depth > MAX_HIERARCHICAL_DEPTH || (1usize << hierarchical_depth) > 1024 {
        candle_core::bail!(
            "invalid input: TripoSG hierarchical_octree_depth {hierarchical_depth} exceeds \
             the hard cap of {MAX_HIERARCHICAL_DEPTH} (grid side {} > 1024)",
            1usize << hierarchical_depth
        );
    }
    if bounds[0] >= bounds[3] || bounds[1] >= bounds[4] || bounds[2] >= bounds[5] {
        candle_core::bail!(
            "invalid input: TripoSG SDF grid bounds {bounds:?} must have min < max per axis"
        );
    }

    // The kv cache is computed ONCE and shared by the dense pass and every
    // refinement level (the reference lambda re-enters vae.decode per call,
    // which recomputes it — numerically identical, ours is just cheaper).
    let kv = decoder.kv_cache(latents)?;

    // ---- Dense pass -------------------------------------------------------
    let n = 1usize << dense_depth;
    let coords = fp16_linspace_grid(bounds, n, latents.device())?;
    let total = n * n * n;
    let mut values = Vec::with_capacity(total);
    let step = chunk.max(1).min(total);
    let mut start = 0usize;
    while start < total {
        let len = (total - start).min(step);
        let mut points = Vec::with_capacity(len * 3);
        for m in start..start + len {
            // x-fastest linear index (x fastest, z slowest).
            let x = m % n;
            let y = (m / n) % n;
            let z = m / (n * n);
            points.push(coords[0][x]);
            points.push(coords[1][y]);
            points.push(coords[2][z]);
        }
        let points = Tensor::from_vec(points, (len, 3), latents.device())?;
        let out = decoder.decode_points(&kv, &points, chunk)?;
        values.extend(out.to_vec1::<f32>()?);
        start += len;
    }
    let mut grid = TripoSgSdfGrid {
        depth: dense_depth,
        dims: n,
        bounds,
        values,
    };

    // ---- Refinement levels ------------------------------------------------
    for level in (dense_depth + 1)..=hierarchical_depth {
        grid = refine_level(decoder, &kv, grid, bounds, level, chunk, latents.device())?;
    }
    Ok(grid)
}

/// One refinement level: low grid (side n, depth `level - 1`) -> high grid
/// (side N = 2n, depth `level`).
fn refine_level(
    decoder: &TripoSgVaeDecoder,
    kv: &Tensor,
    low: TripoSgSdfGrid,
    bounds: [f32; 6],
    level: u32,
    chunk: usize,
    device: &Device,
) -> CandleResult<TripoSgSdfGrid> {
    let n = low.dims;
    let big = 1usize << level; // == 2 * n

    // 1. Nearest upsample: high[hx,hy,hz] = low[hx/2,hy/2,hz/2]
    //    (F.interpolate scale_factor=2 default nearest mode).
    let mut high = vec![0.0_f32; big * big * big];
    for hz in 0..big {
        for hy in 0..big {
            let low_row = (hz / 2) * n * n + (hy / 2) * n;
            let dst = hz * big * big + hy * big;
            for hx in 0..big {
                high[dst + hx] = low.values[low_row + hx / 2];
            }
        }
    }

    // 2. Candidate band on the LOW grid interior (find_candidates_band):
    //    |sigmoid(v) * 2 - 1| < 1.0 on f32 sigmoid values; saturated logits
    //    (sigmoid == exactly 1.0) are excluded.
    let refine_all = std::env::var("LUX3D_TRIPOSG_REFINE_ALL")
        .map(|value| value == "1")
        .unwrap_or(false);
    let mut marked = vec![false; n * n * n];
    for z in 1..n - 1 {
        for y in 1..n - 1 {
            for x in 1..n - 1 {
                let v = low.values[x + y * n + z * n * n];
                let sigmoid = 1.0 / (1.0 + (-v).exp());
                if (sigmoid * 2.0 - 1.0).abs() < 1.0
                    && (refine_all || neighborhood_has_sign_change(&low.values, n, x, y, z))
                {
                    marked[x + y * n + z * n * n] = true;
                }
            }
        }
    }

    // 3. Dilate the marker grid by Chebyshev radius 2 (n < 512) or 1
    //    (n >= 512), clamped at the borders (expand_edge_region_fast
    //    max_pool3d kernel 5 pad 2 / kernel 3 pad 1 with implicit -inf
    //    padding).
    let radius = if n < 512 { 2usize } else { 1usize };
    let mut dilated = vec![false; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                if !marked[x + y * n + z * n * n] {
                    continue;
                }
                let z0 = z.saturating_sub(radius);
                let z1 = (z + radius).min(n - 1);
                let y0 = y.saturating_sub(radius);
                let y1 = (y + radius).min(n - 1);
                let x0 = x.saturating_sub(radius);
                let x1 = (x + radius).min(n - 1);
                for wz in z0..=z1 {
                    for wy in y0..=y1 {
                        for wx in x0..=x1 {
                            dilated[wx + wy * n + wz * n * n] = true;
                        }
                    }
                }
            }
        }
    }

    // 4. Expand every marked low cell to its 8 high children, re-query the
    //    decoder at `bounds_min + high_index * (bounds_size / N)` and scatter
    //    the logits into the high grid.
    let size = [
        (bounds[3] - bounds[0]) as f64,
        (bounds[4] - bounds[1]) as f64,
        (bounds[5] - bounds[2]) as f64,
    ];
    let mins = [bounds[0] as f64, bounds[1] as f64, bounds[2] as f64];
    let cells_per_batch = (chunk / 8).max(1);
    let mut marked_coords = Vec::new();
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                if dilated[x + y * n + z * n * n] {
                    marked_coords.push((x, y, z));
                }
            }
        }
    }
    for batch in marked_coords.chunks(cells_per_batch) {
        let len = batch.len() * 8;
        let mut points = Vec::with_capacity(len * 3);
        for &(x, y, z) in batch {
            for &(cx, cy, cz) in &[
                (2 * x, 2 * y, 2 * z),
                (2 * x, 2 * y, 2 * z + 1),
                (2 * x, 2 * y + 1, 2 * z),
                (2 * x, 2 * y + 1, 2 * z + 1),
                (2 * x + 1, 2 * y, 2 * z),
                (2 * x + 1, 2 * y, 2 * z + 1),
                (2 * x + 1, 2 * y + 1, 2 * z),
                (2 * x + 1, 2 * y + 1, 2 * z + 1),
            ] {
                points.push((mins[0] + cx as f64 * size[0] / big as f64) as f32);
                points.push((mins[1] + cy as f64 * size[1] / big as f64) as f32);
                points.push((mins[2] + cz as f64 * size[2] / big as f64) as f32);
            }
        }
        let points = Tensor::from_vec(points, (len, 3), device)?;
        let out = decoder.decode_points(kv, &points, chunk)?;
        let out = out.to_vec1::<f32>()?;
        for (cell_index, &(x, y, z)) in batch.iter().enumerate() {
            for (child, &(cx, cy, cz)) in [
                (2 * x, 2 * y, 2 * z),
                (2 * x, 2 * y, 2 * z + 1),
                (2 * x, 2 * y + 1, 2 * z),
                (2 * x, 2 * y + 1, 2 * z + 1),
                (2 * x + 1, 2 * y, 2 * z),
                (2 * x + 1, 2 * y, 2 * z + 1),
                (2 * x + 1, 2 * y + 1, 2 * z),
                (2 * x + 1, 2 * y + 1, 2 * z + 1),
            ]
            .iter()
            .enumerate()
            {
                high[cx + cy * big + cz * big * big] = out[cell_index * 8 + child];
            }
        }
    }

    Ok(TripoSgSdfGrid {
        depth: level,
        dims: big,
        bounds,
        values: high,
    })
}

/// True when the 3x3x3 clamped neighborhood of `(x, y, z)` contains BOTH a
/// negative and a non-negative value. Cells whose whole low-res neighborhood
/// is uniform cannot contain a level-0 crossing on the parent grid, so their
/// children normally reproduce the parent sign. The refined decoder is not
/// strictly sign-monotone at half-cell offsets, so culling can in principle
/// omit a patch the reference's full-band query would find; the impact is
/// bounded empirically by the golden mesh parity, and
/// `LUX3D_TRIPOSG_REFINE_ALL=1` restores the reference-exact full band.
fn neighborhood_has_sign_change(values: &[f32], n: usize, x: usize, y: usize, z: usize) -> bool {
    let mut has_negative = false;
    let mut has_non_negative = false;
    for dz in -1i64..=1 {
        let wz = (z as i64 + dz).clamp(0, n as i64 - 1) as usize;
        for dy in -1i64..=1 {
            let wy = (y as i64 + dy).clamp(0, n as i64 - 1) as usize;
            for dx in -1i64..=1 {
                let wx = (x as i64 + dx).clamp(0, n as i64 - 1) as usize;
                if values[wx + wy * n + wz * n * n] < 0.0 {
                    has_negative = true;
                } else {
                    has_non_negative = true;
                }
                if has_negative && has_non_negative {
                    return true;
                }
            }
        }
    }
    false
}

/// `torch.linspace(start, end, n, dtype=torch.float16)` replicated EXACTLY.
///
/// The torch kernels (CPU `RangeFactoriesKernel.cpp` and the CUDA variant
/// agree bit-for-bit here) quantize the endpoints AND the step to fp16, then
/// fill the first half upward as `f16(start + f16(step * i))` and the second
/// half downward as `f16(end - f16(step * (n - 1 - i)))` — every product and
/// sum rounds through fp16. A naive f32 linspace rounded once differs in ~1/3
/// of entries by one fp16 ulp, which is visible in the decoded logits, so the
/// exact scheme is replicated: fp16 arithmetic = f32 math with an fp16
/// round-trip after every operation.
fn fp16_linspace_grid(bounds: [f32; 6], n: usize, device: &Device) -> CandleResult<[Vec<f32>; 3]> {
    let starts = [bounds[0] as f64, bounds[1] as f64, bounds[2] as f64];
    let ends = [bounds[3] as f64, bounds[4] as f64, bounds[5] as f64];
    Ok([
        fp16_linspace_axis(starts[0], ends[0], n, device)?,
        fp16_linspace_axis(starts[1], ends[1], n, device)?,
        fp16_linspace_axis(starts[2], ends[2], n, device)?,
    ])
}

fn fp16_linspace_axis(start: f64, end: f64, n: usize, device: &Device) -> CandleResult<Vec<f32>> {
    if n == 0 {
        candle_core::bail!("invalid input: fp16 linspace needs at least one sample");
    }
    if n == 1 {
        // torch steps == 1 fills with `start`.
        return round_f16_vec(vec![start as f32], device);
    }
    let s = start as f32;
    let e = end as f32;
    // Endpoints and step quantized through fp16 first.
    let endpoints = round_f16_vec(vec![s, e], device)?;
    let s = endpoints[0];
    let e = endpoints[1];
    let step_prep = round_f16_vec(vec![(e - s) / (n as f32 - 1.0)], device)?;
    let step = step_prep[0];

    let halfway = n / 2;
    // Pass 1: fp16-round the products (upward index i, downward index n-1-i).
    let mut products = vec![0.0_f32; n];
    for (i, product) in products.iter_mut().enumerate() {
        let index = if i < halfway { i } else { n - i - 1 };
        *product = step * index as f32;
    }
    let products = round_f16_vec(products, device)?;
    // Pass 2: fp16-round the sums/differences.
    let mut out = vec![0.0_f32; n];
    for (i, value) in out.iter_mut().enumerate() {
        *value = if i < halfway {
            s + products[i]
        } else {
            e - products[i]
        };
    }
    round_f16_vec(out, device)
}

/// Round an F32 vector through fp16 and back (round-to-nearest-even,
/// identical to torch's f32 -> f16 conversion).
fn round_f16_vec(values: Vec<f32>, device: &Device) -> CandleResult<Vec<f32>> {
    let len = values.len();
    let rounded = Tensor::from_vec(values, (len,), device)?
        .to_dtype(DType::F16)?
        .to_dtype(DType::F32)?;
    rounded.to_vec1::<f32>()
}
