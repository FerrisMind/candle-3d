//! TRELLIS.2 dual-grid mesh extraction tests (integration surface):
//! closed-cube / open-plane fixtures, face counts, bounding boxes, the
//! `w0*w2 > w1*w3` split rule through the head-field path, and quad-skip
//! behavior on partial neighbor sets.

use candle_core::{Device, Tensor};
use lux3d_core::runtime::{
    DualGridMesh, TRELLIS2_VOXEL_MARGIN, dual_grid_fields, flexible_dual_grid_to_mesh,
};

/// Closed 2x2x2 cube: 8 voxels, all axes intersected, zero dual offsets.
/// Each axis emits one quad per 2x2 slice (anchored at the min corner), so
/// 6 quads / 12 triangles; dual vertices sit at the voxel centers and form
/// a closed box surface.
#[test]
fn closed_cube_fixture() {
    let mut coords = Vec::new();
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                coords.push([x, y, z]);
            }
        }
    }
    let n = coords.len();
    let mesh = flexible_dual_grid_to_mesh(
        &coords,
        &vec![[0.0_f32; 3]; n],
        &vec![[true, true, true]; n],
        &vec![1.0; n],
        [2, 2, 2],
        [[0.0; 3], [1.0; 3]],
    );
    assert_eq!(mesh.vertices.len() / 3, n);
    assert_eq!(mesh.faces.len() / 3, 12, "6 quads -> 12 triangles");
    // Dual 0 -> vertex = voxel center: (c + 0) * 0.5.
    let (min, max) = vertex_bbox(&mesh);
    assert_eq!(min, [0.0; 3]);
    assert_eq!(max, [0.5; 3]);
    // Closed surface: every undirected edge is used exactly twice.
    let mut edge_count: std::collections::HashMap<[i64; 2], usize> =
        std::collections::HashMap::new();
    for face in mesh.faces.chunks_exact(3) {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            let key = if a < b { [a, b] } else { [b, a] };
            *edge_count.entry(key).or_insert(0) += 1;
        }
    }
    assert!(
        edge_count.values().all(|&count| count == 2),
        "closed mesh: {edge_count:?}"
    );
}

/// Open 2x2 plane at z = 0 (z-axis flags only): exactly one quad -> two
/// triangles; x/y axes emit nothing because their neighbor voxels are
/// missing. The dual offset shifts vertices onto the z = 0 plane.
#[test]
fn open_plane_fixture() {
    // Listed in the z-axis corner order so the single quad is rows 0..3.
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]];
    let n = coords.len();
    let dual = vec![[0.5_f32; 3]; n];
    let mesh = flexible_dual_grid_to_mesh(
        &coords,
        &dual,
        &vec![[false, false, true]; n],
        &vec![0.5; n],
        [2, 2, 1],
        [[-1.0; 3], [1.0; 3]],
    );
    assert_eq!(mesh.faces.len() / 3, 2, "1 quad -> 2 triangles");
    // Vertex z = (0 + 0.5) * (2.0 / 1) + (-1.0) = 0 for every row.
    for vertex in mesh.vertices.chunks_exact(3) {
        assert!((vertex[2] - 0.0).abs() < 1e-6, "z {vertex:?}");
    }
    let (min, max) = vertex_bbox(&mesh);
    assert!(min[2] >= -1.0 && max[2] <= 1.0);
}

/// Split rule through the head-field path: a `[N, 7]` decoded tensor drives
/// `dual_grid_fields` (sigmoid dual, `> 0` intersected, softplus split) and
/// the extractor picks the diagonal by `w0*w2 > w1*w3`.
#[test]
fn head_fields_and_split_rule() {
    // 2x2 plane at z=0 in z-axis corner order: the single z-quad is rows
    // [0, 1, 2, 3].
    let coords: Vec<[i32; 3]> = vec![[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]];
    let n = coords.len();

    let run = |feats: Vec<f32>| -> DualGridMesh {
        let tensor = Tensor::from_vec(feats, (n, 7), &Device::Cpu).expect("feats");
        let fields = dual_grid_fields(&tensor, TRELLIS2_VOXEL_MARGIN).expect("fields");
        flexible_dual_grid_to_mesh(
            &coords,
            &fields.dual,
            &fields.intersected,
            &fields.split_weight,
            [2, 2, 1],
            [[-0.5; 3], [0.5; 3]],
        )
    };
    let base_feats = |split_logit: f32| -> Vec<f32> {
        let mut feats = vec![0.0_f32; n * 7];
        for row in 0..n {
            // dual logits 0 -> dual = 2*sigmoid(0) - 0.5 = 0.5.
            // intersected logits: -5 -> false, +5 -> true.
            feats[row * 7 + 3] = -5.0;
            feats[row * 7 + 4] = -5.0;
            feats[row * 7 + 5] = 5.0;
            feats[row * 7 + 6] = split_logit;
        }
        feats
    };

    // Head-field sanity: dual logits 0 -> dual = 2*sigmoid(0) - 0.5 = 0.5,
    // z intersected, split = softplus(1).
    let fields = {
        let tensor = Tensor::from_vec(base_feats(1.0), (n, 7), &Device::Cpu).expect("feats");
        dual_grid_fields(&tensor, TRELLIS2_VOXEL_MARGIN).expect("fields")
    };
    assert!((fields.dual[0][0] - 0.5).abs() < 1e-6);
    assert_eq!(fields.intersected[0], [false, false, true]);
    assert!((fields.split_weight[0] - (1.0_f32.exp() + 1.0).ln()).abs() < 1e-5);

    // Split 1: w0*w2 > w1*w3 (rows 0 and 2 heavy) -> (0,1,2),(0,2,3).
    let mut feats = base_feats(0.0);
    feats[0 * 7 + 6] = 10.0;
    feats[2 * 7 + 6] = 10.0;
    feats[1 * 7 + 6] = -10.0;
    feats[3 * 7 + 6] = -10.0;
    let mesh_split1 = run(feats);
    let faces1: Vec<[i64; 3]> = mesh_split1
        .faces
        .chunks_exact(3)
        .map(|f| [f[0], f[1], f[2]])
        .collect();
    assert_eq!(faces1.len(), 2);
    assert_eq!(faces1[0], [0, 1, 2]);
    assert_eq!(faces1[1], [0, 2, 3]);

    // Split 2: w0*w2 < w1*w3 (rows 1 and 3 heavy) -> (0,1,3),(3,1,2).
    let mut feats = base_feats(0.0);
    feats[0 * 7 + 6] = -10.0;
    feats[2 * 7 + 6] = -10.0;
    feats[1 * 7 + 6] = 10.0;
    feats[3 * 7 + 6] = 10.0;
    let mesh_split2 = run(feats);
    let faces2: Vec<[i64; 3]> = mesh_split2
        .faces
        .chunks_exact(3)
        .map(|f| [f[0], f[1], f[2]])
        .collect();
    assert_eq!(faces2[0], [0, 1, 3]);
    assert_eq!(faces2[1], [3, 1, 2]);
}

/// Face indices always reference existing vertices, even when quads are
/// skipped around the boundary; unused vertices are still emitted.
#[test]
fn face_indices_stay_in_range_with_skipped_quads() {
    // Two far-apart voxels: no quads, vertices still emitted.
    let mesh = flexible_dual_grid_to_mesh(
        &[[0, 0, 0], [10, 10, 10]],
        &[[0.1; 3], [0.2; 3]],
        &[[true; 3]; 2],
        &[1.0, 2.0],
        [16, 16, 16],
        [[-0.5; 3], [0.5; 3]],
    );
    assert_eq!(mesh.vertices.len() / 3, 2);
    assert!(mesh.faces.is_empty());

    // L-shaped 3-voxel strip: no axis finds its full 2x2 neighbor patch ->
    // zero faces, no out-of-range indices.
    let mesh = flexible_dual_grid_to_mesh(
        &[[0, 0, 0], [0, 1, 0], [0, 0, 1]],
        &[[0.0; 3]; 3],
        &[[true; 3]; 3],
        &[1.0; 3],
        [2, 2, 2],
        [[0.0; 3], [1.0; 3]],
    );
    assert!(mesh.faces.is_empty());
    for &index in &mesh.faces {
        assert!(index >= 0 && (index as usize) < mesh.vertices.len() / 3);
    }
}

fn vertex_bbox(mesh: &DualGridMesh) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for vertex in mesh.vertices.chunks_exact(3) {
        for axis in 0..3 {
            min[axis] = min[axis].min(vertex[axis]);
            max[axis] = max[axis].max(vertex[axis]);
        }
    }
    (min, max)
}
