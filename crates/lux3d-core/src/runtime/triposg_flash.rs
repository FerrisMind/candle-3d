//! TripoSG flash decoder: hierarchical near-surface SDF extraction plus a
//! clean-room dual marching cubes (DMC) implementation.
//!
//! Reference semantics (`triposg/inference_utils.py::flash_extract_geometry`,
//! `generate_dense_grid_points_2`, `extract_near_surface_volume_fn`,
//! `expand_edge_region_fast`):
//!
//! - resolutions start at `2^depth`, are halved while `>= 63`, reversed, then
//!   the first level is snapped to `round(r / 4) * 4 - 1` and every next level
//!   doubles it (`depth 9 -> 63, 126, 252, 504`);
//! - the dense level queries every point of an `(r + 1)^3` float32 linspace;
//! - each refinement marks low-grid voxels that have a sign change among their
//!   six replicated neighbours, adds `|logit| < 0.95`, dilates the marker by
//!   two (three when not at the final level), maps marked cells to `2 * idx`
//!   in the high grid, dilates the high marker (`1` step normally, `2` steps at
//!   the final level), and queries every resulting high-grid point;
//! - unqueried high-grid values stay at `-10000` and become `NaN` before DMC;
//! - DMC consumes `sdf = -logits` (negative inside) and maps grid vertices as
//!   `v / 2^depth * bbox_size + bbox_min`, reversing every triangle.
//!
//! The DMC kernel is a clean-room Rust implementation. Its MC cycle table
//! comes from the MIT/Apache-licensed `isomesh` crate (publicly derived MC
//! connectivity), not from Diso's CC BY-NC CUDA source. Vertex placement,
//! dual quads, quad triangulation and world mapping follow the documented DMC
//! algorithm. Face ordering/topology is expected to match on unambiguous
//! cells; the reference's private ambiguous-face flips are not copied.

use std::collections::HashMap;

use candle_core::{Result as CandleResult, Tensor};

use super::triposg_vae::TripoSgVaeDecoder;

/// Flash SDF grid (z-fastest, matching both NumPy `meshgrid(indexing="ij")`
/// C-order and Diso's `gA(x,y,z) = z + dims[2] * (y + dims[1] * x)`).
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgFlashGrid {
    /// Requested octree depth (`2^depth` is the final world-mapping divisor).
    pub depth: u32,
    /// Grid side length (`last_resolution + 1`).
    pub dims: usize,
    pub bounds: [f32; 6],
    /// Raw (positive = inside) decoder logits, z-fastest; unqueried final
    /// cells are `NaN`.
    pub values: Vec<f32>,
}

/// CPU mesh from the flash DMC path. Vertices are already in world space;
/// faces are zero-based and already winding-reversed like the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgDmcMesh {
    pub vertices: Vec<f32>,
    pub faces: Vec<i64>,
}

/// Resolutions used by the reference flash path.
///
/// `depth 9 -> [63, 126, 252, 504]`, `depth 8 -> [31, 62, 124, 248]`,
/// `depth 5 -> [31]`.
pub fn flash_resolutions(octree_depth: u32) -> Vec<usize> {
    let mut octree_resolution = 1usize << octree_depth;
    let mut resolutions = Vec::new();
    if octree_resolution < 63 {
        resolutions.push(octree_resolution);
    }
    while octree_resolution >= 63 {
        resolutions.push(octree_resolution);
        octree_resolution /= 2;
    }
    resolutions.reverse();
    resolutions[0] = ((resolutions[0] as f64 / 4.0).round() as usize) * 4 - 1;
    for index in 0..resolutions.len().saturating_sub(1) {
        resolutions[index + 1] = resolutions[0] * (1usize << (index + 1));
    }
    resolutions
}

/// Decode the flash hierarchical SDF grid. `octree_depth` is the reference
/// `flash_octree_depth` (default 9).
pub fn flash_sdf_grid(
    decoder: &TripoSgVaeDecoder,
    latents: &Tensor,
    bounds: [f32; 6],
    octree_depth: u32,
    chunk: usize,
) -> CandleResult<TripoSgFlashGrid> {
    if octree_depth == 0 || octree_depth > 10 {
        candle_core::bail!(
            "invalid input: TripoSG flash_octree_depth {octree_depth} out of range 1..=10"
        );
    }
    if bounds[0] >= bounds[3] || bounds[1] >= bounds[4] || bounds[2] >= bounds[5] {
        candle_core::bail!(
            "invalid input: TripoSG flash bounds {bounds:?} must have min < max per axis"
        );
    }
    let resolutions = flash_resolutions(octree_depth);
    let kv = decoder.kv_cache(latents)?;
    let mut grid = dense_flash_level(
        decoder,
        &kv,
        bounds,
        resolutions[0],
        chunk,
        latents.device(),
    )?;

    for (index, &resolution) in resolutions[1..].iter().enumerate() {
        let final_level = index + 2 == resolutions.len();
        grid = refine_flash_level_with_final(
            decoder,
            &kv,
            &grid,
            bounds,
            resolution,
            chunk,
            latents.device(),
            final_level,
        )?;
    }
    for value in &mut grid.values {
        if *value == -10000.0 {
            *value = f32::NAN;
        }
    }
    Ok(TripoSgFlashGrid {
        depth: octree_depth,
        dims: resolutions[resolutions.len() - 1] + 1,
        bounds,
        values: grid.values,
    })
}

/// Dense first level: `(r + 1)^3` NumPy-float32 linspace points, z-fastest.
fn dense_flash_level(
    decoder: &TripoSgVaeDecoder,
    kv: &Tensor,
    bounds: [f32; 6],
    resolution: usize,
    chunk: usize,
    device: &candle_core::Device,
) -> CandleResult<TripoSgFlashGrid> {
    let dims = resolution + 1;
    let axes = [
        np_float32_linspace(bounds[0], bounds[3], dims),
        np_float32_linspace(bounds[1], bounds[4], dims),
        np_float32_linspace(bounds[2], bounds[5], dims),
    ];
    let mut values = Vec::with_capacity(dims * dims * dims);
    let mut points = Vec::new();
    let step = chunk.max(1);
    for x in 0..dims {
        for y in 0..dims {
            for z in 0..dims {
                points.push(axes[0][x]);
                points.push(axes[1][y]);
                points.push(axes[2][z]);
                if points.len() / 3 == step {
                    values.extend(decode_points(decoder, kv, &points, device, chunk)?);
                    points.clear();
                }
            }
        }
    }
    if !points.is_empty() {
        values.extend(decode_points(decoder, kv, &points, device, chunk)?);
    }
    Ok(TripoSgFlashGrid {
        depth: resolution.trailing_zeros(),
        dims,
        bounds,
        values,
    })
}

#[allow(clippy::too_many_arguments)]
fn refine_flash_level_with_final(
    decoder: &TripoSgVaeDecoder,
    kv: &Tensor,
    low: &TripoSgFlashGrid,
    bounds: [f32; 6],
    resolution: usize,
    chunk: usize,
    device: &candle_core::Device,
    final_level: bool,
) -> CandleResult<TripoSgFlashGrid> {
    let low_dims = low.dims;
    let dims = resolution + 1;
    let mut marked = near_surface_mask(&low.values, low_dims, 0.0);
    for (index, value) in low.values.iter().enumerate() {
        if value.abs() < 0.95 {
            marked[index] = 1;
        }
    }
    if !final_level {
        dilate_mask(&mut marked, low_dims, 2);
    }

    let mut high_marker = vec![0u8; dims * dims * dims];
    for x in 0..low_dims {
        for y in 0..low_dims {
            for z in 0..low_dims {
                if marked[x * low_dims * low_dims + y * low_dims + z] == 0 {
                    continue;
                }
                let hx = 2 * x;
                let hy = 2 * y;
                let hz = 2 * z;
                if hx < dims && hy < dims && hz < dims {
                    high_marker[hx * dims * dims + hy * dims + hz] = 1;
                }
            }
        }
    }
    dilate_mask(&mut high_marker, dims, if final_level { 2 } else { 1 });

    let size = [
        (bounds[3] - bounds[0]) as f64,
        (bounds[4] - bounds[1]) as f64,
        (bounds[5] - bounds[2]) as f64,
    ];
    let mins = [bounds[0], bounds[1], bounds[2]];
    let steps = [
        (size[0] / resolution as f64) as f32,
        (size[1] / resolution as f64) as f32,
        (size[2] / resolution as f64) as f32,
    ];
    let mut coordinates = Vec::new();
    let mut indices = Vec::new();
    for x in 0..dims {
        for y in 0..dims {
            for z in 0..dims {
                let index = x * dims * dims + y * dims + z;
                if high_marker[index] == 0 {
                    continue;
                }
                indices.push(index);
                coordinates.push(x as f32 * steps[0] + mins[0]);
                coordinates.push(y as f32 * steps[1] + mins[1]);
                coordinates.push(z as f32 * steps[2] + mins[2]);
            }
        }
    }
    let mut values = vec![-10000.0_f32; dims * dims * dims];
    let step = chunk.max(1);
    let mut start = 0usize;
    while start < indices.len() {
        let len = (indices.len() - start).min(step);
        let points = &coordinates[start * 3..(start + len) * 3];
        let decoded = decode_points(decoder, kv, points, device, chunk)?;
        for (offset, value) in decoded.into_iter().enumerate() {
            values[indices[start + offset]] = value;
        }
        start += len;
    }
    Ok(TripoSgFlashGrid {
        depth: low.depth,
        dims,
        bounds,
        values,
    })
}

fn decode_points(
    decoder: &TripoSgVaeDecoder,
    kv: &Tensor,
    coordinates: &[f32],
    device: &candle_core::Device,
    chunk: usize,
) -> CandleResult<Vec<f32>> {
    if coordinates.is_empty() {
        return Ok(Vec::new());
    }
    let len = coordinates.len() / 3;
    let points = Tensor::from_vec(coordinates.to_vec(), (len, 3), device)?;
    decoder.decode_points(kv, &points, chunk)?.to_vec1::<f32>()
}

/// `np.linspace(start, stop, count, dtype=np.float32)` (verified bit-exact
/// against NumPy: double-precision step, each point rounded to f32, last point
/// replaced by the stop rounded to f32).
fn np_float32_linspace(start: f32, stop: f32, count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![start];
    }
    let start64 = start as f64;
    let stop64 = stop as f64;
    let step = (stop64 - start64) / (count - 1) as f64;
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        values.push((start64 + index as f64 * step) as f32);
    }
    values[count - 1] = stop;
    values
}

/// `extract_near_surface_volume_fn(..., alpha=0)`: a voxel whose six
/// replicated-neighbour signs are not all equal, masked by `value > -9000`.
fn near_surface_mask(values: &[f32], dims: usize, alpha: f32) -> Vec<u8> {
    let mut out = vec![0u8; values.len()];
    let plane = dims * dims;
    for x in 0..dims {
        for y in 0..dims {
            for z in 0..dims {
                let index = x * plane + y * dims + z;
                let val = values[index] + alpha;
                if !(val > -9000.0) {
                    continue;
                }
                let mut same = true;
                for (dx, dy, dz) in [
                    (-1_i32, 0_i32, 0_i32),
                    (1, 0, 0),
                    (0, -1, 0),
                    (0, 1, 0),
                    (0, 0, -1),
                    (0, 0, 1),
                ] {
                    let nx = (x as i32 + dx).clamp(0, dims as i32 - 1) as usize;
                    let ny = (y as i32 + dy).clamp(0, dims as i32 - 1) as usize;
                    let nz = (z as i32 + dz).clamp(0, dims as i32 - 1) as usize;
                    let neighbour = values[nx * plane + ny * dims + nz] + alpha;
                    let neighbour = if neighbour > -9000.0 { neighbour } else { val };
                    if sign(val) != sign(neighbour) {
                        same = false;
                        break;
                    }
                }
                if !same {
                    out[index] = 1;
                }
            }
        }
    }
    out
}

fn sign(value: f32) -> i8 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

/// In-place binary dilation with a `(2 * radius + 1)^3` all-ones kernel
/// (zero padding, `> 0` threshold). Repeating this `radius` times matches one
/// convolution with that kernel.
fn dilate_mask(mask: &mut [u8], dims: usize, radius: usize) {
    if radius == 0 {
        return;
    }
    let mut current = mask.to_vec();
    for _ in 0..radius {
        let mut next = vec![0u8; current.len()];
        let plane = dims * dims;
        for x in 0..dims {
            for y in 0..dims {
                for z in 0..dims {
                    let index = x * plane + y * dims + z;
                    if current[index] == 0 {
                        continue;
                    }
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(dims - 1);
                    let y0 = y.saturating_sub(1);
                    let y1 = (y + 1).min(dims - 1);
                    let z0 = z.saturating_sub(1);
                    let z1 = (z + 1).min(dims - 1);
                    for nx in x0..=x1 {
                        for ny in y0..=y1 {
                            for nz in z0..=z1 {
                                next[nx * plane + ny * dims + nz] = 1;
                            }
                        }
                    }
                }
            }
        }
        current = next;
    }
    mask.copy_from_slice(&current);
}

// ---------------------------------------------------------------------------
// Clean-room dual marching cubes
// ---------------------------------------------------------------------------

/// Diso cube corners: bit 0 = x, bit 1 = y, bit 2 = z.
const CORNER_OFFSETS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

/// Diso edge numbering (`mcEdgeLocations`): [base x, base y, base z, axis].
const EDGE_BASES: [[i32; 4]; 12] = [
    [0, 0, 0, 0],
    [1, 0, 0, 2],
    [0, 0, 1, 0],
    [0, 0, 0, 2],
    [0, 1, 0, 0],
    [1, 1, 0, 2],
    [0, 1, 1, 0],
    [0, 1, 0, 2],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 1, 1],
];

/// isomesh edge id -> Diso edge id. isomesh numbers x edges first, then y,
/// then z, with corners in `i & 1, (i >> 1) & 1, (i >> 2) & 1` order.
const ISO_TO_DISO: [usize; 12] = [0, 4, 2, 6, 8, 9, 11, 10, 3, 1, 7, 5];

/// Quad neighbours and local edge ids for the three positive-axis edges,
/// copied from the documented DMC quad topology (not from Diso code): each
/// primal edge is surrounded by four cells.
const QUAD_NEIGHBOURS: [[(i32, i32, i32, usize); 4]; 3] = [
    [(0, 0, 0, 0), (0, -1, 0, 4), (0, -1, -1, 6), (0, 0, -1, 2)],
    [(0, 0, 0, 8), (0, 0, -1, 11), (-1, 0, -1, 10), (-1, 0, 0, 9)],
    [(0, 0, 0, 3), (-1, 0, 0, 1), (-1, -1, 0, 5), (0, -1, 0, 7)],
];

/// Reproduce the final mesh the reference pipeline builds from raw DiffDMC
/// output with `trimesh.Trimesh(..., process=True, validate=False)`:
///
/// 1. merge bit-identical vertices (`merge_vertices`);
/// 2. drop faces with non-finite vertices (`remove_infinite_values`);
/// 3. drop degenerate faces (`nondegenerate_faces`);
/// 4. drop vertices no face references (`remove_unreferenced_vertices`).
///
/// Step 4 matters most: NaN-padded boundary shells at depth >= 7 produce
/// thousands of unreferenced vertices. Step 1 is also observable (depth 9:
/// 279,092 referenced raw vertices -> 279,090 final).
///
/// Trimesh's default `merge_vertices` quantizes at `1e-8`, but our DMC
/// arithmetic can differ from the reference in the last ulp on ~0.05% of
/// vertices. Quantizing those would merge distinct vertices and delete four
/// otherwise valid depth-9 triangles. The raw duplicates that the reference
/// actually merges are bit-identical here, so exact-bit merging reproduces the
/// reference final counts (279,090 / 558,116) without false degenerates.
pub fn compact_dmc_mesh(mesh: &TripoSgDmcMesh) -> TripoSgDmcMesh {
    let vertex_count = mesh.vertices.len() / 3;

    let mut merged = Vec::with_capacity(vertex_count);
    let mut merge_remap = vec![u32::MAX; vertex_count];
    let mut canonical: HashMap<[u32; 3], u32> = HashMap::with_capacity(vertex_count);
    for index in 0..vertex_count {
        let base = index * 3;
        let coords = [
            mesh.vertices[base],
            mesh.vertices[base + 1],
            mesh.vertices[base + 2],
        ];
        if !coords.iter().all(|value| value.is_finite()) {
            continue;
        }
        let key = [
            coords[0].to_bits(),
            coords[1].to_bits(),
            coords[2].to_bits(),
        ];
        let target = *canonical.entry(key).or_insert_with(|| {
            let target = (merged.len() / 3) as u32;
            merged.extend_from_slice(&coords);
            target
        });
        merge_remap[index] = target;
    }

    let merged_count = merged.len() / 3;
    let mut used = vec![false; merged_count];
    let mut faces = Vec::with_capacity(mesh.faces.len());
    for face in mesh.faces.chunks_exact(3) {
        if face
            .iter()
            .any(|&index| index < 0 || index as usize >= vertex_count)
        {
            continue;
        }
        let mapped = [
            merge_remap[face[0] as usize],
            merge_remap[face[1] as usize],
            merge_remap[face[2] as usize],
        ];
        if mapped.contains(&u32::MAX)
            || mapped[0] == mapped[1]
            || mapped[1] == mapped[2]
            || mapped[0] == mapped[2]
        {
            continue;
        }
        for &index in &mapped {
            used[index as usize] = true;
        }
        faces.extend(mapped.iter().map(|&index| i64::from(index)));
    }

    let mut compact_remap = vec![u32::MAX; merged_count];
    let mut vertices = Vec::with_capacity(used.iter().filter(|&&is_used| is_used).count() * 3);
    for (index, &is_used) in used.iter().enumerate() {
        if is_used {
            compact_remap[index] = (vertices.len() / 3) as u32;
            let base = index * 3;
            vertices.extend_from_slice(&merged[base..base + 3]);
        }
    }
    for face in &mut faces {
        *face = i64::from(compact_remap[*face as usize]);
    }
    TripoSgDmcMesh { vertices, faces }
}

/// Run clean-room DMC on `grid` and map vertices to world space.
pub fn flash_dual_marching_cubes(grid: &TripoSgFlashGrid) -> CandleResult<TripoSgDmcMesh> {
    let dims = grid.dims;
    if dims == 0 || grid.values.len() != dims * dims * dims {
        candle_core::bail!("TripoSG flash grid length does not match dims^3");
    }
    // The reference converts positive-inside decoder logits to SDF
    // (`sdf = -grid_logits / octree_resolution`) before DMC. The scale is
    // irrelevant at interior iso-0 crossings, but NOT at the padded shell
    // (padding is the fixed value iso + 1). `octree_resolution` is the
    // schedule-loop variable: it ends at 32 whenever `2^depth >= 63`, and at
    // `2^depth` below that.
    let divisor = (1u64 << grid.depth.min(5)) as f32;
    let sdf: Vec<f32> = grid.values.iter().map(|value| -*value / divisor).collect();
    let padded = pad_grid(&sdf, dims, 1.0);
    let pd = dims + 2;
    let cells = pd - 1;
    let cell_count = cells * cells * cells;

    // First pass: one dual vertex per MC cycle, plus a dense cell -> first
    // patch map (u32::MAX for unused cells).
    let mut vertices = Vec::new();
    let mut cell_patch_base = vec![u32::MAX; cell_count];
    for x in 0..cells {
        for y in 0..cells {
            for z in 0..cells {
                let cell = x * cells * cells + y * cells + z;
                let code = cell_code(&padded, pd, x, y, z, 0.0);
                if code == 0 || code == 255 {
                    continue;
                }
                cell_patch_base[cell] = vertices.len() as u32 / 3;
                create_cell_patches(&padded, pd, x, y, z, code, &mut vertices);
            }
        }
    }

    // Second pass: one quad per positive-axis cut edge.
    let mut quads: Vec<[u32; 4]> = Vec::new();
    for x in 0..cells {
        for y in 0..cells {
            for z in 0..cells {
                let cell = x * cells * cells + y * cells + z;
                if cell_patch_base[cell] == u32::MAX {
                    continue;
                }
                for axis in 0..3 {
                    let (d0, d1) = edge_values(&padded, pd, x, y, z, axis, 0.0);
                    let entering = d0 < 0.0 && d1 >= 0.0;
                    let exiting = d1 < 0.0 && d0 >= 0.0;
                    if !entering && !exiting {
                        continue;
                    }
                    let base_order = QUAD_NEIGHBOURS[axis];
                    // Exiting edges use the base order. Entering edges are the
                    // same cycle traversed backwards, rotated so the current
                    // cell's own edge stays in slot 0 (Diso's `dmcQuad`
                    // entering rows). The rotation matters: the triangulation
                    // rule compares the diagonals named by slots 0..3.
                    let order = if exiting {
                        base_order
                    } else {
                        let reversed = [base_order[3], base_order[2], base_order[1], base_order[0]];
                        let start = reversed
                            .iter()
                            .position(|entry| entry.0 == 0 && entry.1 == 0 && entry.2 == 0)
                            .expect("quad order contains the current cell");
                        [
                            reversed[start],
                            reversed[(start + 1) % 4],
                            reversed[(start + 2) % 4],
                            reversed[(start + 3) % 4],
                        ]
                    };
                    let mut quad = [0u32; 4];
                    let mut valid = true;
                    for (slot, &(dx, dy, dz, neighbour_edge)) in order.iter().enumerate() {
                        let nx = x as i32 + dx;
                        let ny = y as i32 + dy;
                        let nz = z as i32 + dz;
                        if nx < 0
                            || ny < 0
                            || nz < 0
                            || nx >= cells as i32
                            || ny >= cells as i32
                            || nz >= cells as i32
                        {
                            valid = false;
                            break;
                        }
                        let neighbour =
                            nx as usize * cells * cells + ny as usize * cells + nz as usize;
                        let base = cell_patch_base[neighbour];
                        if base == u32::MAX {
                            valid = false;
                            break;
                        }
                        let Some(local) = edge_patch_index(
                            &padded,
                            pd,
                            nx as usize,
                            ny as usize,
                            nz as usize,
                            neighbour_edge,
                            0.0,
                        ) else {
                            valid = false;
                            break;
                        };
                        quad[slot] = base + local as u32;
                    }
                    if !valid {
                        continue;
                    }
                    quads.push(quad);
                }
            }
        }
    }

    // DiffDMC.forward subtracts 1 from every vertex before triangulating.
    for value in &mut vertices {
        *value -= 1.0;
    }
    let faces = triangulate_quads(&vertices, &quads);

    // flash_extract_geometry: vertices / 2^depth * bbox_size + bbox_min, then
    // faces[:, ::-1].
    let denom = (1u64 << grid.depth) as f64;
    let size = [
        (grid.bounds[3] - grid.bounds[0]) as f64,
        (grid.bounds[4] - grid.bounds[1]) as f64,
        (grid.bounds[5] - grid.bounds[2]) as f64,
    ];
    let mins = [
        grid.bounds[0] as f64,
        grid.bounds[1] as f64,
        grid.bounds[2] as f64,
    ];
    for vertex in vertices.chunks_exact_mut(3) {
        for axis in 0..3 {
            vertex[axis] = (vertex[axis] as f64 / denom * size[axis] + mins[axis]) as f32;
        }
    }
    let mut reversed = Vec::with_capacity(faces.len());
    for face in faces.chunks_exact(3) {
        reversed.extend([face[2], face[1], face[0]]);
    }
    Ok(TripoSgDmcMesh {
        vertices,
        faces: reversed,
    })
}

fn pad_grid(values: &[f32], dims: usize, fill: f32) -> Vec<f32> {
    let pd = dims + 2;
    let mut out = vec![fill; pd * pd * pd];
    for x in 0..dims {
        for y in 0..dims {
            for z in 0..dims {
                out[(x + 1) * pd * pd + (y + 1) * pd + (z + 1)] =
                    values[x * dims * dims + y * dims + z];
            }
        }
    }
    out
}

fn cell_code(values: &[f32], dims: usize, x: usize, y: usize, z: usize, iso: f32) -> u8 {
    let mut code = 0u8;
    for (corner, offset) in CORNER_OFFSETS.iter().enumerate() {
        let value = values[(x as i32 + offset[0]) as usize * dims * dims
            + (y as i32 + offset[1]) as usize * dims
            + (z as i32 + offset[2]) as usize];
        if value >= iso {
            code |= 1 << corner;
        }
    }
    code
}

fn edge_corners(edge: usize) -> (usize, usize) {
    let base = EDGE_BASES[edge];
    let axis = base[3] as usize;
    let c0 = 0usize
        | ((base[0] as usize & 1) << 0)
        | ((base[1] as usize & 1) << 1)
        | ((base[2] as usize & 1) << 2);
    let c1 = c0 | (1 << axis);
    (c0, c1)
}

fn edge_values(
    values: &[f32],
    dims: usize,
    x: usize,
    y: usize,
    z: usize,
    axis: usize,
    _iso: f32,
) -> (f32, f32) {
    let edge = [0usize, 8, 3][axis];
    let base = EDGE_BASES[edge];
    let v0 = values[(x as i32 + base[0]) as usize * dims * dims
        + (y as i32 + base[1]) as usize * dims
        + (z as i32 + base[2]) as usize];
    let mut x1 = base[0];
    let mut y1 = base[1];
    let mut z1 = base[2];
    match axis {
        0 => x1 += 1,
        1 => y1 += 1,
        _ => z1 += 1,
    }
    let v1 = values[(x as i32 + x1) as usize * dims * dims
        + (y as i32 + y1) as usize * dims
        + (z as i32 + z1) as usize];
    (v0, v1)
}

/// Append one dual vertex per MC cycle for this cell.
fn create_cell_patches(
    values: &[f32],
    dims: usize,
    x: usize,
    y: usize,
    z: usize,
    code: u8,
    out: &mut Vec<f32>,
) {
    let next = isomesh::marching_cubes::table::segment_links(code, 0);
    let mut visited = [false; 12];
    for start in 0..12 {
        if next[start] == isomesh::marching_cubes::table::NO_EDGE || visited[start] {
            continue;
        }
        let mut sum = [0.0f32; 3];
        let mut count = 0u32;
        let mut current = start as u8;
        loop {
            let edge = ISO_TO_DISO[current as usize];
            visited[current as usize] = true;
            let point = crossing_point(values, dims, x, y, z, edge);
            sum[0] += point[0];
            sum[1] += point[1];
            sum[2] += point[2];
            count += 1;
            current = next[current as usize];
            if current == isomesh::marching_cubes::table::NO_EDGE || visited[current as usize] {
                break;
            }
        }
        if count > 0 {
            // Diso accumulates in `Scalar` (f32) and multiplies by the f32
            // reciprocal; keep that arithmetic so diagonal-choice comparisons
            // below stay bit-comparable on near-coplanar quads.
            let inv = 1.0f32 / count as f32;
            out.push(sum[0] * inv);
            out.push(sum[1] * inv);
            out.push(sum[2] * inv);
        }
    }
}

/// Local patch index containing `edge_id` (Diso numbering), for a used cell.
fn edge_patch_index(
    values: &[f32],
    dims: usize,
    x: usize,
    y: usize,
    z: usize,
    edge_id: usize,
    _iso: f32,
) -> Option<u8> {
    let code = cell_code(values, dims, x, y, z, 0.0);
    let next = isomesh::marching_cubes::table::segment_links(code, 0);
    let target_iso = DISO_TO_ISO[edge_id];
    let mut visited = [false; 12];
    let mut patch = 0u8;
    for start in 0..12 {
        if next[start] == isomesh::marching_cubes::table::NO_EDGE || visited[start] {
            continue;
        }
        let mut current = start as u8;
        loop {
            visited[current as usize] = true;
            if current as usize == target_iso {
                return Some(patch);
            }
            current = next[current as usize];
            if current == isomesh::marching_cubes::table::NO_EDGE || visited[current as usize] {
                break;
            }
        }
        patch += 1;
    }
    None
}

const DISO_TO_ISO: [usize; 12] = [0, 9, 2, 8, 1, 11, 3, 10, 4, 5, 7, 6];

/// Crossing point of one local edge, in padded-grid coordinates (`f64`
/// accumulation matches the reference's scalar vertex average).
fn crossing_point(
    values: &[f32],
    dims: usize,
    x: usize,
    y: usize,
    z: usize,
    edge: usize,
) -> [f32; 3] {
    let base = EDGE_BASES[edge];
    let (c0, c1) = edge_corners(edge);
    let _ = c1;
    let axis = base[3] as usize;
    let v0 = values[(x as i32 + base[0]) as usize * dims * dims
        + (y as i32 + base[1]) as usize * dims
        + (z as i32 + base[2]) as usize];
    let mut offset = [base[0], base[1], base[2]];
    offset[axis] += 1;
    let v1 = values[(x as i32 + offset[0]) as usize * dims * dims
        + (y as i32 + offset[1]) as usize * dims
        + (z as i32 + offset[2]) as usize];
    let _ = c0;
    let t = if v1 != v0 {
        let raw = (0.0 - v0) / (v1 - v0);
        // Diso's min/max are `a < b ? a : b` / `a > b ? a : b`, so
        // clamp(NaN, 0, 1) == 1: an edge touching an unqueried cell snaps to
        // its upper endpoint instead of producing a NaN vertex. Those patches
        // are unreferenced by faces but are part of the reference vertex array.
        if raw.is_nan() {
            1.0
        } else {
            raw.clamp(0.0, 1.0)
        }
    } else {
        0.5
    };
    let p0 = [
        (x as i32 + base[0]) as f32,
        (y as i32 + base[1]) as f32,
        (z as i32 + base[2]) as f32,
    ];
    let p1 = [
        (x as i32 + offset[0]) as f32,
        (y as i32 + offset[1]) as f32,
        (z as i32 + offset[2]) as f32,
    ];
    [
        p0[0] + (p1[0] - p0[0]) * t,
        p0[1] + (p1[1] - p0[1]) * t,
        p0[2] + (p1[2] - p0[2]) * t,
    ]
}

fn triangulate_quads(vertices: &[f32], quads: &[[u32; 4]]) -> Vec<i64> {
    let mut selected = Vec::with_capacity(quads.len());
    for quad in quads {
        let points = quad.map(|index| {
            let base = index as usize * 3;
            [vertices[base], vertices[base + 1], vertices[base + 2]]
        });
        let angles1 = config_cos(&points, [[0, 1, 3], [1, 2, 3]]);
        let angles2 = config_cos(&points, [[0, 1, 2], [0, 2, 3]]);
        selected.push(angles1 < angles2);
    }
    let mut faces = Vec::with_capacity(quads.len() * 2);
    for (quad, &use_first) in quads.iter().zip(selected.iter()) {
        if use_first {
            for [a, b, c] in [[0usize, 1, 3], [1, 2, 3]] {
                faces.extend([quad[a] as i64, quad[b] as i64, quad[c] as i64]);
            }
        }
    }
    for (quad, &use_first) in quads.iter().zip(selected.iter()) {
        if !use_first {
            for [a, b, c] in [[0usize, 1, 2], [0, 2, 3]] {
                faces.extend([quad[a] as i64, quad[b] as i64, quad[c] as i64]);
            }
        }
    }
    faces
}

fn config_cos(points: &[[f32; 3]; 4], triangles: [[usize; 3]; 2]) -> f32 {
    let mut max_cos = f32::NEG_INFINITY;
    for [a, b, c] in triangles {
        let cos1 = cosine(points[b], points[a], points[c], points[a]);
        let cos2 = cosine(points[c], points[b], points[a], points[b]);
        let cos3 = cosine(points[a], points[c], points[b], points[c]);
        max_cos = max_cos.max(cos1.max(cos2).max(cos3));
    }
    max_cos
}

fn cosine(center: [f32; 3], p1: [f32; 3], center2: [f32; 3], p2: [f32; 3]) -> f32 {
    let unit = |vector: [f32; 3]| {
        let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
            .sqrt()
            .max(1e-12);
        [vector[0] / norm, vector[1] / norm, vector[2] / norm]
    };
    let a = unit([p1[0] - center[0], p1[1] - center[1], p1[2] - center[2]]);
    let b = unit([p2[0] - center2[0], p2[1] - center2[1], p2[2] - center2[2]]);
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
