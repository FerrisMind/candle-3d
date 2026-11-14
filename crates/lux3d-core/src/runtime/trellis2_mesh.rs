//! TRELLIS.2 O-Voxel flexible dual-grid mesh extraction
//! (`o_voxel/convert/flexible_dual_grid.py::flexible_dual_grid_to_mesh`,
//! inference branch `train = False` with `split_weight` provided).
//!
//! Reference behavior (port verbatim in behavior; locked decision 9):
//!
//! - One vertex per active voxel, in coordinate-row order, including voxels
//!   that end up unused by any face:
//!   `vertex = (coord_f32 + dual) * voxel_size + aabb[0]` with
//!   `voxel_size = (aabb[1] - aabb[0]) / grid_size` per axis.
//! - A coordinate hashmap (`coord -> voxel row index`) answers neighbor
//!   lookups; the reference CUDA kernel inserts the row index as the value.
//! - For each voxel `i` and axis `a` with `intersected[i][a]`, the quad
//!   corners are the four voxels at
//!   `i + edge_neighbor_voxel_offset[a][0..4]`:
//!   x axis `[[0,0,0],[0,0,1],[0,1,1],[0,1,0]]`,
//!   y axis `[[0,0,0],[1,0,0],[1,0,1],[0,0,1]]`,
//!   z axis `[[0,0,0],[0,1,0],[1,1,0],[1,0,0]]`.
//!   All four neighbors must exist (any missing -> the quad is skipped).
//! - Quad split: `w0 * w2 > w1 * w3` (per-voxel `split_weight` rows, the
//!   decoder's `softplus(feats[6])`) selects `quad_split_1 = [0,1,2,0,2,3]`,
//!   otherwise `quad_split_2 = [0,1,3,3,1,2]`. Quads are emitted in
//!   `(voxel, axis)` row-major order, two triangles each.
//!
//! No hole filling, decimation, remeshing or orientation unification (the
//! reference `Mesh.fill_holes()` post-step is a documented exclusion).

use std::collections::HashMap;

/// Quad corner voxel offsets per axis (x, y, z), reference
/// `flexible_dual_grid_to_mesh.edge_neighbor_voxel_offset`.
const EDGE_NEIGHBOR_VOXEL_OFFSET: [[[i32; 3]; 4]; 3] = [
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]], // x axis
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], // y axis
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], // z axis
];

/// `quad_split_1`: diagonal 0-2 (`[0,1,2,0,2,3]`).
const QUAD_SPLIT_1: [usize; 6] = [0, 1, 2, 0, 2, 3];
/// `quad_split_2`: diagonal 1-3 (`[0,1,3,3,1,2]`).
const QUAD_SPLIT_2: [usize; 6] = [0, 1, 3, 3, 1, 2];

/// Mesh extracted by [`flexible_dual_grid_to_mesh`].
#[derive(Debug, Clone, PartialEq)]
pub struct DualGridMesh {
    /// Flat `[x, y, z, ...]` f32 positions, one per active voxel (row order
    /// matches the input coordinates).
    pub vertices: Vec<f32>,
    /// Flat zero-based triangle indices.
    pub faces: Vec<i64>,
}

/// Decoded dual-grid head fields for one sample: per-voxel dual offset
/// (`3 * sigmoid(feats[0:3]) - voxel_margin` with `voxel_margin = 0.5`),
/// per-axis intersection flags (`feats[3:6] > 0`) and per-voxel quad split
/// weight (`softplus(feats[6])`).
#[derive(Debug, Clone, PartialEq)]
pub struct DualGridFields {
    /// `[N, 3]` dual offsets in voxel units.
    pub dual: Vec<[f32; 3]>,
    /// `[N, 3]` per-axis quad emission flags.
    pub intersected: Vec<[bool; 3]>,
    /// `[N]` quad split weights.
    pub split_weight: Vec<f32>,
}

/// Stable softplus: `x > 20 -> x`, else `ln(1 + exp(x))` (F32, matching
/// `F.softplus` semantics for the values the decoder produces).
pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// Extract the dual-grid mesh.
///
/// * `coords`: `[N]` active voxel coordinates `(x, y, z)`, unique.
/// * `dual_vertices`: `[N, 3]` dual offsets in voxel units.
/// * `intersected`: `[N, 3]` per-axis quad emission flags.
/// * `split_weight`: `[N]` per-voxel quad split weights (the inference path
///   always provides them; the `split_weight = None` normal-alignment branch
///   and the training branch of the reference are not ported).
/// * `grid_size`: voxel grid resolution per axis (the reference
///   `set_resolution` value; voxel size is `aabb_size / grid_size`).
/// * `aabb`: `[min, max]` corner, `[[x0,y0,z0],[x1,y1,z1]]`.
#[allow(clippy::too_many_arguments)]
pub fn flexible_dual_grid_to_mesh(
    coords: &[[i32; 3]],
    dual_vertices: &[[f32; 3]],
    intersected: &[[bool; 3]],
    split_weight: &[f32],
    grid_size: [usize; 3],
    aabb: [[f32; 3]; 2],
) -> DualGridMesh {
    let n = coords.len();
    assert_eq!(dual_vertices.len(), n, "dual vertex count mismatch");
    assert_eq!(intersected.len(), n, "intersected flag count mismatch");
    assert_eq!(split_weight.len(), n, "split weight count mismatch");
    let voxel_size = [
        (aabb[1][0] - aabb[0][0]) / grid_size[0] as f32,
        (aabb[1][1] - aabb[0][1]) / grid_size[1] as f32,
        (aabb[1][2] - aabb[0][2]) / grid_size[2] as f32,
    ];

    // One vertex per active voxel, including unused ones.
    let mut vertices = Vec::with_capacity(n * 3);
    for (index, coord) in coords.iter().enumerate() {
        for axis in 0..3 {
            vertices.push(
                (coord[axis] as f32 + dual_vertices[index][axis]) * voxel_size[axis]
                    + aabb[0][axis],
            );
        }
    }

    // Coord -> row index hashmap (reference `hashmap_insert_3d_idx_as_val`).
    let map: HashMap<[i32; 3], u32> = coords
        .iter()
        .enumerate()
        .map(|(index, coord)| (*coord, index as u32))
        .collect();

    let mut faces: Vec<i64> = Vec::new();
    for (index, coord) in coords.iter().enumerate() {
        for axis in 0..3 {
            if !intersected[index][axis] {
                continue;
            }
            let mut quad = [0u32; 4];
            let mut valid = true;
            for (corner, offset) in EDGE_NEIGHBOR_VOXEL_OFFSET[axis].iter().enumerate() {
                match map.get(&[
                    coord[0] + offset[0],
                    coord[1] + offset[1],
                    coord[2] + offset[2],
                ]) {
                    Some(&row) => quad[corner] = row,
                    None => {
                        valid = false;
                        break;
                    }
                }
            }
            if !valid {
                continue;
            }
            let rows: [usize; 4] = [
                quad[0] as usize,
                quad[1] as usize,
                quad[2] as usize,
                quad[3] as usize,
            ];
            let (w0, w1, w2, w3) = (
                split_weight[rows[0]],
                split_weight[rows[1]],
                split_weight[rows[2]],
                split_weight[rows[3]],
            );
            let split = if w0 * w2 > w1 * w3 {
                &QUAD_SPLIT_1
            } else {
                &QUAD_SPLIT_2
            };
            for corner in split {
                faces.push(quad[*corner] as i64);
            }
        }
    }

    DualGridMesh { vertices, faces }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_coords(side: i32) -> Vec<[i32; 3]> {
        let mut coords = Vec::new();
        for x in 0..side {
            for y in 0..side {
                for z in 0..side {
                    coords.push([x, y, z]);
                }
            }
        }
        coords
    }

    #[test]
    fn softplus_stable_matches_reference() {
        // Softplus continuity at the branch point and large-x passthrough.
        for value in [-30.0_f32, -1.0, 0.0, 1.0, 19.999, 20.0, 20.001, 100.0] {
            let expected = if value > 20.0 {
                value
            } else {
                (1.0 + value.exp()).ln()
            };
            assert!((softplus_stable(value) - expected).abs() < 1e-6);
        }
        // ln(1 + e^0) = ln 2.
        assert!((softplus_stable(0.0) - 2.0_f32.ln()).abs() < 1e-6);
    }

    /// Closed 2x2x2 cube: 8 active voxels, every axis intersected, dual
    /// offsets 0 -> one quad per face plane per axis = 6 quads / 12
    /// triangles, all 8 vertices used, bbox exactly the voxel cube.
    #[test]
    fn closed_cube_has_twelve_faces() {
        let coords = cube_coords(2);
        let n = coords.len();
        let dual = vec![[0.0_f32; 3]; n];
        let intersected = vec![[true, true, true]; n];
        let split = vec![1.0_f32; n];
        let mesh = flexible_dual_grid_to_mesh(
            &coords,
            &dual,
            &intersected,
            &split,
            [2, 2, 2],
            [[0.0; 3], [1.0; 3]],
        );
        assert_eq!(mesh.vertices.len(), n * 3);
        assert_eq!(mesh.faces.len(), 36, "12 triangles = 6 quads x 2");
        // Face index range within the 8 vertices.
        let max_index = mesh.faces.iter().copied().max().unwrap_or(-1);
        assert_eq!(max_index, 7);
        // Vertex bbox: dual 0 puts each vertex at its voxel center, so the
        // extent is the cell centers {0, 0.5} per axis (voxel size 0.5).
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for vertex in mesh.vertices.chunks_exact(3) {
            for axis in 0..3 {
                min[axis] = min[axis].min(vertex[axis]);
                max[axis] = max[axis].max(vertex[axis]);
            }
        }
        assert_eq!(min, [0.0; 3]);
        assert_eq!(max, [0.5; 3]);
        // Every vertex is referenced by some face (closed surface, no
        // unused vertices).
        let mut used = vec![false; n];
        for index in &mesh.faces {
            used[*index as usize] = true;
        }
        assert!(used.iter().all(|&u| u));
    }

    /// Open 2x2 plane (z axis intersected only): exactly one quad -> two
    /// triangles; the x/y axes emit nothing (their neighbor quads need
    /// voxels off the plane).
    #[test]
    fn open_plane_emits_one_quad() {
        let coords = vec![
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ];
        let n = coords.len();
        let dual = vec![[0.25_f32; 3]; n];
        let intersected = vec![[false, false, true]; n];
        let split = vec![0.5_f32; n];
        let mesh = flexible_dual_grid_to_mesh(
            &coords,
            &dual,
            &intersected,
            &split,
            [2, 2, 2],
            [[0.0; 3], [1.0; 3]],
        );
        // z-axis quads exist only for voxels with y+1 and z+1 in range:
        // (0,0,0), (1,0,0) at plane z=0... z-axis offsets move x and y, so
        // the plane is x = coord.x; each 2x2 (y,z) cell emits one quad from
        // its min corner: (0,0,0) and (1,0,0).
        assert_eq!(mesh.faces.len(), 12, "2 quads x 2 triangles");
        assert_eq!(mesh.vertices.len(), n * 3);
        // Vertices carry the dual offset: (0 + 0.25) * 0.5 = 0.125.
        assert!((mesh.vertices[0] - 0.125).abs() < 1e-7);
        assert!((mesh.vertices[4] - 0.125).abs() < 1e-7);
    }

    /// Single voxel with all axes intersected: every quad needs a neighbor
    /// outside the grid -> no faces, one (unused) vertex.
    #[test]
    fn isolated_voxel_produces_no_faces() {
        let mesh = flexible_dual_grid_to_mesh(
            &[[0, 0, 0]],
            &[[0.1, 0.2, 0.3]],
            &[[true, true, true]],
            &[1.0],
            [4, 4, 4],
            [[-0.5; 3], [0.5; 3]],
        );
        assert!(mesh.faces.is_empty());
        assert_eq!(mesh.vertices.len(), 3);
        assert!((mesh.vertices[0] - (-0.5 + 0.1 * 0.25)).abs() < 1e-7);
    }

    /// Quad skipping on partial neighbor sets: a 2x2x1 slab with the x axis
    /// intersected only emits the quad whose 4 voxels all exist.
    #[test]
    fn partial_neighbor_set_skips_quad() {
        // Voxels (0,0,0) and (0,1,0) only; x-axis quad needs
        // (0,0,0),(0,0,1),(0,1,1),(0,1,0) -> (0,0,1) missing -> skipped.
        let coords = vec![[0, 0, 0], [0, 1, 0]];
        let mesh = flexible_dual_grid_to_mesh(
            &coords,
            &[[0.0; 3]; 2],
            &[[true, true, true]; 2],
            &[1.0, 1.0],
            [2, 2, 2],
            [[0.0; 3], [1.0; 3]],
        );
        assert!(mesh.faces.is_empty());
    }

    /// The split rule: `w0*w2 > w1*w3` picks `quad_split_1`; the opposite
    /// picks `quad_split_2`; equality picks split 2 (reference `torch.where`
    /// takes the else branch when the condition is false).
    #[test]
    fn split_rule_selects_diagonal_by_weights() {
        // 2x2 plane at z=0 listed in the z-axis corner order
        // [[0,0,0],[0,1,0],[1,1,0],[1,0,0]]: the z-axis quad from voxel
        // (0,0,0) has corners exactly rows [0, 1, 2, 3].
        let coords = vec![[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]];
        let dual = vec![[0.0_f32; 3]; 4];
        let intersected = vec![[false, false, true]; 4];
        // split1 [0,1,2,0,2,3] applied to quad [0,1,2,3] -> (0,1,2),(0,2,3)
        // split2 [0,1,3,3,1,2] applied to quad [0,1,2,3] -> (0,1,3),(3,1,2)
        let cases = [
            // (w0, w1, w2, w3, expected triangles)
            ([5.0, 0.0, 5.0, 0.0], vec![0, 1, 2, 0, 2, 3]),
            ([0.0, 5.0, 0.0, 5.0], vec![0, 1, 3, 3, 1, 2]),
            ([1.0, 1.0, 1.0, 1.0], vec![0, 1, 3, 3, 1, 2]),
        ];
        for (weights, expected) in cases {
            let mesh = flexible_dual_grid_to_mesh(
                &coords,
                &dual,
                &intersected,
                &weights,
                [2, 2, 1],
                [[0.0; 3], [1.0; 3]],
            );
            assert_eq!(mesh.faces, expected, "weights {weights:?}");
        }
    }

    /// Vertex ordering matches the coordinate row order regardless of faces.
    #[test]
    fn vertices_follow_coordinate_row_order() {
        let coords = vec![[3, 1, 2], [0, 0, 0], [1, 5, 7]];
        let dual = vec![[0.5; 3]; 3];
        let mesh = flexible_dual_grid_to_mesh(
            &coords,
            &dual,
            &[[false; 3]; 3],
            &[1.0; 3],
            [8, 8, 8],
            [[-1.0; 3], [1.0; 3]],
        );
        for (row, coord) in coords.iter().enumerate() {
            for axis in 0..3 {
                let expected = (coord[axis] as f32 + 0.5) * (2.0 / 8.0) - 1.0;
                assert!((mesh.vertices[row * 3 + axis] - expected).abs() < 1e-6);
            }
        }
    }
}
