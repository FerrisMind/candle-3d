//! TRELLIS.2 1024 cascade tests (T6):
//!
//! 1. Non-gated: the HR quantization `((coord + 0.5) / 512 * (hr // 16)).int()`
//!    against hand-computed values (edges 0/511, midpoint flips, the excluded
//!    512 input, monotonicity over the whole 0..512 range), duplicate
//!    collisions + lexicographic `torch.unique(dim=0)` ordering, the
//!    token-budget loop (boundary at `== max_num_tokens`, the
//!    1024 -> 896 -> 768 decrement sequence through the test-only stop
//!    resolution, and the production invariant that the 1024 cascade always
//!    breaks at 1024), and a tiny synthetic two-stage shape VAE whose
//!    `upsample_coords` doubles coordinates and preserves the parent/child
//!    (subidx) order.
//! 2. Raw-dir gated: `upsample_coords(slat, 4)` on the real shape VAE must
//!    equal the stage-3 C2S trace of a full `decode` (same block objects),
//!    stay inside the 512 grid, and quantize to `hr_resolution == 1024`
//!    with `<= 49152` unique rows.
//! 3. Env-gated smoke (`LUX3D_TRELLIS2_RUN_CASCADE_SMOKE=1`): the full
//!    cascade shape stage (LR DiT -> upsample -> quantize -> HR DiT) on real
//!    weights with injected zero conditioning and the golden latent.
//! 4. Env-gated CUDA e2e (`LUX3D_TRELLIS2_RUN_CASCADE=1`, `cuda` feature):
//!    the full image -> mesh 1024 cascade; asserts `hr_resolution == 1024`,
//!    `<= 49152` HR tokens and a non-empty mesh, and records peak VRAM
//!    (polled through `nvidia-smi`) and wall time.
//!
//! The smoke test runs the reference step count by default; override with
//! `LUX3D_TRELLIS2_CASCADE_STEPS` (T5's E2E convention). Example CUDA run:
//! `LUX3D_TRELLIS2_RUN_CASCADE=1 cargo test --release -p lux3d-core --features cuda \
//!  --test trellis2_cascade cascade_full_cuda_e2e -- --nocapture --test-threads 1`.

use std::collections::HashSet;
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::runtime::{
    TRELLIS2_CASCADE_HR_RESOLUTION, TRELLIS2_CASCADE_LR_RESOLUTION,
    TRELLIS2_CASCADE_UPSAMPLE_TIMES, Trellis2Pipeline, quantize_hr_coords,
    quantize_hr_coords_with_stop,
};
use lux3d_core::test_support::workspace_root;

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-vae-golden")
}

fn raw_dir() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_RAW_DIR") {
        let path = PathBuf::from(path);
        return path.is_dir().then_some(path);
    }
    let path = workspace_root().join("models").join("TRELLIS.2-4B");
    path.is_dir().then_some(path)
}

fn steps_override() -> usize {
    std::env::var("LUX3D_TRELLIS2_CASCADE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12)
}

// ---------------------------------------------------------------------------
// 1. Quantization + token budget (pure unit tests)
// ---------------------------------------------------------------------------

/// Hand-computed quantization table for `lr_resolution = 512`:
/// `quant(c) = trunc((c + 0.5) / 512 * (hr_resolution / 16))`.
#[test]
fn quantize_hand_computed_mapping_and_edges() {
    // hr_resolution 1024 -> scale 64: quant = (c + 0.5) / 8 (exact in f32).
    let table: [(i32, i32); 8] = [
        (0, 0),
        (1, 0),    // 1.5 / 8 = 0.1875
        (3, 0),    // 3.5 / 8 = 0.4375
        (7, 0),    // 7.5 / 8 = 0.9375 (just under the first cell edge)
        (8, 1),    // 8.5 / 8 = 1.0625 (just above)
        (256, 32), // 256.5 / 8 = 32.0625
        (504, 63), // 504.5 / 8 = 63.0625
        (511, 63), // 511.5 / 8 = 63.9375 (the top edge maps to 63, not 64)
    ];
    for (coord, expected) in table {
        let (coords, hr) = quantize_hr_coords(
            &[[coord, coord, coord]],
            TRELLIS2_CASCADE_LR_RESOLUTION,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            49152,
        );
        assert_eq!(hr, 1024, "the 1024 cascade must always decode at 1024");
        assert_eq!(
            coords,
            vec![[expected, expected, expected]],
            "coord {coord}"
        );
    }

    // A 512 input (which the upsample can never emit) would truncate to the
    // out-of-grid cell 64 — documents the truncation boundary; the excluded
    // value cannot occur because four C2S doublings of a < 32 grid stay < 512.
    let (coords, _) = quantize_hr_coords(
        &[[512, 512, 512]],
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49152,
    );
    assert_eq!(coords, vec![[64, 64, 64]]);

    // The whole legitimate range maps into 0..64, monotonically, with the
    // exact cell-edge flip points (8k - 7 -> cell k for k >= 1).
    for coord in 0..512i32 {
        let expected = ((coord as f32 + 0.5) / 512.0 * 64.0) as i32;
        assert!((0..64).contains(&expected), "coord {coord} -> {expected}");
        let (coords, _) = quantize_hr_coords(
            &[[coord, 0, 0]],
            TRELLIS2_CASCADE_LR_RESOLUTION,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            49152,
        );
        assert_eq!(coords, vec![[expected, 0, 0]], "coord {coord}");
    }
}

/// Duplicate rows collapse and the output is lexicographically sorted
/// (`torch.unique(dim=0)` semantics).
#[test]
fn quantize_dedups_and_sorts_lexicographically() {
    let hr_coords: Vec<[i32; 3]> = vec![
        [10, 0, 0],
        [0, 10, 0],
        [10, 0, 0], // exact duplicate
        [0, 0, 10],
        [7, 7, 7], // quantizes onto (0,0,0) — collides with (0,0,0)
        [0, 0, 0],
        [15, 0, 0], // quantizes onto (1,0,0) — collides with (10,0,0)
    ];
    let (coords, hr) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49152,
    );
    assert_eq!(hr, 1024);
    // Cells: (0,0,0) from [0,0,0]/[7,7,7]; (0,0,1) from [0,0,10];
    // (0,1,0) from [0,10,0]; (1,0,0) from [10,0,0]/[15,0,0].
    assert_eq!(
        coords,
        vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]],
        "unique rows must be deduplicated and sorted"
    );
    // Sorted + unique invariants for arbitrary inputs.
    let mut sorted = coords.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(coords, sorted);
}

/// The production loop (stop resolution 1024) can never reduce the
/// resolution: `hr_resolution == 1024` breaks the loop even when the token
/// count is at or above the budget.
#[test]
fn budget_loop_always_breaks_at_1024() {
    // 8 distinct 64-grid cells.
    let hr_coords: Vec<[i32; 3]> = (0..8).map(|k| [0, k * 9, 0]).collect();
    let expected: Vec<[i32; 3]> = (0..8i32).map(|k| [0, k, 0]).collect();
    for max_num_tokens in [0usize, 1, 7, 8] {
        let (coords, hr) = quantize_hr_coords(
            &hr_coords,
            TRELLIS2_CASCADE_LR_RESOLUTION,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            max_num_tokens,
        );
        assert_eq!(hr, 1024, "max_num_tokens {max_num_tokens}");
        assert_eq!(coords, expected);
    }
}

/// The budget boundary is strict (`num_tokens < max_num_tokens` breaks) and
/// the decrement path walks 1024 -> 896 -> 768 (exercised through the
/// test-only stop resolution; the reference hardcodes the 1024 stop).
///
/// Coord family `[0, k * 9, 0]`, k = 0..8, quantized per grid:
/// - 1024 (scale 64): y cells {0,1,2,3,4,5,6,7} -> 8 unique rows,
/// - 896 (scale 56):  y cells {0,1,2,3,3,4,5,6} -> 7 unique rows,
/// - 768 (scale 48):  y cells {0,0,1,2,3,4,5,5} -> 6 unique rows.
#[test]
fn budget_boundary_and_decrement_sequence_1024_896_768() {
    let hr_coords: Vec<[i32; 3]> = (0..8).map(|k| [0, k * 9, 0]).collect();
    let stop = 768u32;

    // max = 9: 8 < 9 at the first pass -> break at 1024, 64-grid coords.
    let (coords, hr) =
        quantize_hr_coords_with_stop(&hr_coords, TRELLIS2_CASCADE_LR_RESOLUTION, 1024, 9, stop);
    assert_eq!(hr, 1024);
    assert_eq!(coords, (0..8i32).map(|k| [0, k, 0]).collect::<Vec<_>>());

    // max = 8: 8 < 8 is false (the == boundary is OVER budget) -> 896;
    // 7 < 8 -> break at 896 with the 56-grid coords (36 collides onto 3).
    let (coords, hr) =
        quantize_hr_coords_with_stop(&hr_coords, TRELLIS2_CASCADE_LR_RESOLUTION, 1024, 8, stop);
    assert_eq!(hr, 896);
    assert_eq!(
        coords,
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [0, 4, 0],
            [0, 5, 0],
            [0, 6, 0]
        ]
    );

    // max = 7: over budget at 896 as well (7 < 7 false) -> 768 == stop
    // breaks unconditionally with the 48-grid coords (9 and 63 collapse).
    let (coords, hr) =
        quantize_hr_coords_with_stop(&hr_coords, TRELLIS2_CASCADE_LR_RESOLUTION, 1024, 7, stop);
    assert_eq!(hr, 768);
    assert_eq!(
        coords,
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 2, 0],
            [0, 3, 0],
            [0, 4, 0],
            [0, 5, 0]
        ]
    );
}

/// The real 49152 budget: a coord set whose 1024-pass quantization is
/// EXACTLY 49152 unique rows is over budget (`<` is strict), while one less
/// row breaks immediately.
#[test]
fn budget_boundary_at_reference_49152() {
    // 64 * 64 * 12 = 49152 parents whose 64-grid cells (i, j, k) are all
    // distinct: coord = (8i, 8j, 8k) -> ((8i + 0.5) / 8) = i + 0.0625 -> i.
    let mut hr_coords = Vec::with_capacity(49152);
    for i in 0..64i32 {
        for j in 0..64i32 {
            for k in 0..12i32 {
                hr_coords.push([8 * i, 8 * j, 8 * k]);
            }
        }
    }
    assert_eq!(hr_coords.len(), 49152);

    // One over the boundary: 49152 < 49153 -> break at 1024 with all rows.
    let (coords, hr) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49153,
    );
    assert_eq!(hr, 1024);
    assert_eq!(coords.len(), 49152);
    assert_eq!(coords[0], [0, 0, 0]);
    assert_eq!(coords[49151], [63, 63, 11]);

    // Exactly at the boundary: over budget -> the loop decrements. With the
    // test stop at 768 the second pass (56-grid) collapses the axes
    // (<= 56 * 56 * 11 = 34496 rows) and breaks under budget at 896.
    let (coords, hr) =
        quantize_hr_coords_with_stop(&hr_coords, TRELLIS2_CASCADE_LR_RESOLUTION, 1024, 49152, 768);
    assert_eq!(hr, 896);
    assert!(coords.len() < 49152, "{} rows at 896", coords.len());
    // Sorted + unique.
    let mut sorted = coords.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(coords, sorted);

    // With the production stop (1024) the same over-budget set still
    // decodes at 1024 — the cascade's resolution never drops.
    let (coords, hr) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49152,
    );
    assert_eq!(hr, 1024);
    assert_eq!(coords.len(), 49152);
}

// ---------------------------------------------------------------------------
// 2. Tiny synthetic shape VAE: upsample order
// ---------------------------------------------------------------------------

const C: usize = 8; // tiny child-channel count

fn const_tensor(shape: &[usize], value: f32) -> Tensor {
    let len: usize = shape.iter().product();
    Tensor::from_vec(vec![value; len], shape.to_vec(), &Device::Cpu).expect("tensor")
}

fn subdiv_bias(pattern: &[f32; 8]) -> Tensor {
    Tensor::from_vec(pattern.to_vec(), (8,), &Device::Cpu).expect("subdiv bias")
}

/// Entries for one C2S block: zeroed convs, identity norm1, and the given
/// subdivision bias pattern (weight zero -> every parent subdivides alike).
fn c2s_entries(
    stage: usize,
    block: usize,
    in_channels: usize,
    bias: &Tensor,
) -> Vec<(String, Tensor)> {
    let base = format!("blocks.{stage}.{block}.");
    vec![
        (
            format!("{base}conv1.weight"),
            const_tensor(&[8 * C, 3, 3, 3, in_channels], 0.0),
        ),
        (format!("{base}conv1.bias"), const_tensor(&[8 * C], 0.0)),
        (
            format!("{base}conv2.weight"),
            const_tensor(&[C, 3, 3, 3, C], 0.0),
        ),
        (format!("{base}conv2.bias"), const_tensor(&[C], 0.0)),
        (
            format!("{base}norm1.weight"),
            const_tensor(&[in_channels], 1.0),
        ),
        (
            format!("{base}norm1.bias"),
            const_tensor(&[in_channels], 0.0),
        ),
        (
            format!("{base}to_subdiv.weight"),
            const_tensor(&[8, in_channels], 0.0),
        ),
        (format!("{base}to_subdiv.bias"), bias.clone()),
    ]
}

/// Two-C2S-stage decoder config (`upsample_times` 1 and 2 are exercisable).
fn tiny_two_stage_config() -> lux3d_core::runtime::Trellis2ShapeVaeConfig {
    lux3d_core::runtime::Trellis2ShapeVaeConfig {
        model_channels: vec![16, C, C, C, C],
        num_blocks: vec![0, 0, 0, 0, 0],
        latent_channels: 4,
        out_channels: 7,
        mlp_ratio: 4.0,
        pred_subdiv: true,
    }
}

fn tiny_two_stage_decoder(bias: &Tensor) -> lux3d_core::runtime::Trellis2ShapeVaeDecoder {
    use lux3d_core::runtime::Trellis2ShapeVaeDecoder;
    let mut entries: Vec<(String, Tensor)> = vec![
        (
            "from_latent.weight".to_string(),
            const_tensor(&[16, 4], 0.0),
        ),
        ("from_latent.bias".to_string(), const_tensor(&[16], 0.0)),
        (
            "output_layer.weight".to_string(),
            const_tensor(&[7, C], 0.0),
        ),
        ("output_layer.bias".to_string(), const_tensor(&[7], 0.0)),
    ];
    entries.extend(c2s_entries(0, 0, 16, bias));
    entries.extend(c2s_entries(1, 0, C, bias));
    entries.extend(c2s_entries(2, 0, C, bias));
    entries.extend(c2s_entries(3, 0, C, bias));
    let map: std::collections::HashMap<String, Tensor> = entries.into_iter().collect();
    let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    Trellis2ShapeVaeDecoder::load(vb, &tiny_two_stage_config()).expect("tiny decoder load")
}

/// All-children subdivision: two stages double 1 -> 8 -> 64 coords, parents
/// in input order, children ascending in `subidx = dx + 2*dy + 4*dz`.
#[test]
fn upsample_two_stage_doubles_and_preserves_parent_child_order() {
    let ones = subdiv_bias(&[1.0; 8]);
    let decoder = tiny_two_stage_decoder(&ones);

    let coords = vec![[0i32, 0, 0]];
    let latent = const_tensor(&[1, 4], 0.0);

    // One C2S stage: the 8 children of (0,0,0) in subidx order.
    let level1 = decoder
        .upsample_coords(&coords, &latent, 1)
        .expect("upsample 1");
    assert_eq!(
        level1,
        vec![
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    );

    // Two C2S stages: 64 grandchildren grouped by parent in parent order.
    let level2 = decoder
        .upsample_coords(&coords, &latent, 2)
        .expect("upsample 2");
    assert_eq!(level2.len(), 64);
    let mut expected = Vec::with_capacity(64);
    for parent in &level1 {
        for subidx in 0..8usize {
            let dx = (subidx % 2) as i32;
            let dy = ((subidx / 2) % 2) as i32;
            let dz = (subidx / 4) as i32;
            expected.push([parent[0] * 2 + dx, parent[1] * 2 + dy, parent[2] * 2 + dz]);
        }
    }
    assert_eq!(level2, expected);
    // First grandchild block is the (0,0,0) subtree, second the (1,0,0) one.
    assert_eq!(
        &level2[0..8],
        &[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ][..]
    );
    assert_eq!(
        &level2[8..16],
        &[
            [2, 0, 0],
            [3, 0, 0],
            [2, 1, 0],
            [3, 1, 0],
            [2, 0, 1],
            [3, 0, 1],
            [2, 1, 1],
            [3, 1, 1]
        ][..]
    );
    assert_eq!(level2[63], [3, 3, 3]);

    // Four stages from one parent: 8^4 = 4096 leaves (the reference call
    // shape with upsample_times = 4 on a 5-stage decoder).
    let level4 = decoder
        .upsample_coords(&coords, &latent, 4)
        .expect("upsample 4");
    assert_eq!(level4.len(), 4096);

    // upsample_times must stay below the stage count (the reference returns
    // `h.coords` when stage `upsample_times` is reached).
    assert!(decoder.upsample_coords(&coords, &latent, 5).is_err());

    // Empty parents -> empty coords.
    let empty: Vec<[i32; 3]> = Vec::new();
    let empty_latent = const_tensor(&[0, 4], 0.0);
    assert!(
        decoder
            .upsample_coords(&empty, &empty_latent, 2)
            .expect("empty upsample")
            .is_empty()
    );
}

/// Even/odd selection (`subdiv > 0`) plus parent-order preservation with
/// unsorted parents, and equality with the decode trace (the upsample must
/// reuse the exact decode block paths).
#[test]
fn upsample_selection_parent_order_and_decode_trace_equality() {
    let alternating = subdiv_bias(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    let decoder = tiny_two_stage_decoder(&alternating);

    // Two unsorted parents; even subidx survive (dx + 2dy + 4dz even
    // <=> dz = 0 and (dx, dy) in {(0,0), (1,1)}... enumerated below).
    let coords = vec![[3i32, 1, 2], [0, 0, 0]];
    let latent = const_tensor(&[2, 4], 0.0);
    let level1 = decoder
        .upsample_coords(&coords, &latent, 1)
        .expect("upsample 1");
    assert_eq!(
        level1,
        vec![
            // Parent (3,1,2) first (input order), even subidx children.
            [6, 2, 4], // subidx 0
            [6, 3, 4], // subidx 2
            [6, 2, 5], // subidx 4
            [6, 3, 5], // subidx 6
            // Parent (0,0,0) second.
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
        ]
    );

    // Same rows through a full decode: the C2S trace blocks must match the
    // upsample levels exactly (identical block objects, identical order).
    let decoded = decoder.decode(&coords, &latent, true).expect("decode");
    assert_eq!(decoded.trace.len(), 4, "all-zero num_blocks -> C2S only");
    assert_eq!(
        decoded.trace[0].coords, level1,
        "stage-0 C2S == upsample(1)"
    );
    let level2 = decoder
        .upsample_coords(&coords, &latent, 2)
        .expect("upsample 2");
    assert_eq!(
        decoded.trace[1].coords, level2,
        "stage-1 C2S == upsample(2)"
    );
    // The decode runs all four C2S stages, so its final coords are the
    // upsample(4) rows (four doublings), and the trace at every level
    // matches the corresponding upsample prefix.
    let level4 = decoder
        .upsample_coords(&coords, &latent, 4)
        .expect("upsample 4");
    assert_eq!(decoded.coords, level4, "final decode coords == upsample(4)");
    assert_eq!(
        decoded.trace[2].coords,
        decoder
            .upsample_coords(&coords, &latent, 3)
            .expect("upsample 3")
    );
}

// ---------------------------------------------------------------------------
// 3. Raw-dir gated: real shape VAE upsample + quantize
// ---------------------------------------------------------------------------

fn load_golden_latent() -> Option<(Vec<[i32; 3]>, Tensor)> {
    let golden = golden_dir();
    if !golden.join("slat_coords.npy").is_file() {
        return None;
    }
    use ndarray::Array2;
    let slat_coords: Array2<i32> = ndarray_npy::read_npy(golden.join("slat_coords.npy")).ok()?;
    let slat_feats: Array2<f32> = ndarray_npy::read_npy(golden.join("slat_feats.npy")).ok()?;
    assert_eq!(slat_coords.dim(), (64, 3));
    assert_eq!(slat_feats.dim(), (64, 32));
    let coords: Vec<[i32; 3]> = (0..64)
        .map(|i| {
            [
                slat_coords[[i, 0]],
                slat_coords[[i, 1]],
                slat_coords[[i, 2]],
            ]
        })
        .collect();
    let latent = Tensor::from_vec(
        slat_feats.iter().copied().collect::<Vec<f32>>(),
        (64, 32),
        &Device::Cpu,
    )
    .expect("golden latent tensor");
    Some((coords, latent))
}

/// On the real shape VAE, `upsample_coords(slat, 4)` must equal the stage-3
/// C2S trace of a full decode (the upsample reuses the exact decode block
/// paths), stay inside the 512 grid, and quantize under the production
/// budget loop to `hr_resolution == 1024` with `<= 49152` sorted unique rows.
#[test]
fn cascade_upsample_and_quantize_on_real_shape_vae() {
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let Some((coords, latent)) = load_golden_latent() else {
        eprintln!(
            "SKIP: VAE golden fixture missing at {}",
            golden_dir().display()
        );
        return;
    };
    for coord in &coords {
        assert!(
            coord.iter().all(|&v| (0..32).contains(&v)),
            "LR coord {coord:?}"
        );
    }

    let shape_dec = raw.join("ckpts/shape_dec_next_dc_f16c32_fp16.safetensors");
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[shape_dec], DType::F32, &Device::Cpu)
    }
    .expect("mmaped shape_dec");
    let decoder = lux3d_core::runtime::Trellis2ShapeVaeDecoder::load(
        vb,
        &lux3d_core::runtime::Trellis2ShapeVaeConfig::flexi_dual_shape(),
    )
    .expect("shape_dec load");

    let started = std::time::Instant::now();
    let hr_coords = decoder
        .upsample_coords(&coords, &latent, TRELLIS2_CASCADE_UPSAMPLE_TIMES)
        .expect("upsample(slat, 4)");
    println!(
        "upsample(slat, 4): {} -> {} coords in {:.1}s",
        coords.len(),
        hr_coords.len(),
        started.elapsed().as_secs_f64()
    );
    assert!(!hr_coords.is_empty());
    for coord in &hr_coords {
        assert!(
            coord.iter().all(|&v| (0..512).contains(&v)),
            "HR coord {coord:?} outside the 512 grid"
        );
    }

    // Same rows as the full decode's last C2S trace block (stage 3, its last
    // block index) — identical block objects and child ordering.
    let decoded = decoder.decode(&coords, &latent, true).expect("decode");
    let last_c2s = decoded.trace.last().expect("trace");
    assert_eq!(last_c2s.stage, 3);
    assert_eq!(
        last_c2s.coords, hr_coords,
        "upsample(slat, 4) == decode stage-3 C2S coords"
    );

    // Production quantization + budget: 1024 cascade always decodes at 1024.
    let (quant, hr_resolution) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49152,
    );
    assert_eq!(hr_resolution, 1024);
    assert!(!quant.is_empty());
    assert!(quant.len() <= 49152, "{} quantized rows", quant.len());
    for coord in &quant {
        assert!(
            coord.iter().all(|&v| (0..64).contains(&v)),
            "quant {coord:?}"
        );
    }
    let mut sorted = quant.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(quant, sorted, "quantized rows must be sorted and unique");
    println!("quantized HR coords: {}", quant.len());
}

// ---------------------------------------------------------------------------
// 4. Env-gated smoke: full cascade shape stage on real weights (CPU-friendly)
// ---------------------------------------------------------------------------

/// LR DiT (golden 64-coord latent) -> upsample -> quantize -> HR DiT ->
/// decode at 1024, all on real weights with zero conditioning. Gated behind
/// `LUX3D_TRELLIS2_RUN_CASCADE_SMOKE=1` (loads both 1.3B DiTs; minutes on
/// CPU). `LUX3D_TRELLIS2_CASCADE_STEPS` overrides the step count.
#[test]
fn cascade_shape_stage_smoke_real_weights() {
    if std::env::var("LUX3D_TRELLIS2_RUN_CASCADE_SMOKE")
        .ok()
        .as_deref()
        != Some("1")
    {
        eprintln!("SKIP: set LUX3D_TRELLIS2_RUN_CASCADE_SMOKE=1 to run the cascade shape smoke");
        return;
    }
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let Some((coords, _latent)) = load_golden_latent() else {
        eprintln!(
            "SKIP: VAE golden fixture missing at {}",
            golden_dir().display()
        );
        return;
    };

    let pipeline =
        Trellis2Pipeline::load_from_raw_dir(&raw, &Device::Cpu).expect("raw-dir pipeline");
    let steps = steps_override();
    // Zero conditioning (matches the reference neg_cond; the cond VALUES do
    // not matter for a smoke run, only the token counts).
    let cond_lr = Tensor::zeros((1, 1029, 1024), DType::F32, &Device::Cpu).expect("cond lr");
    let cond_hr = Tensor::zeros((1, 4101, 1024), DType::F32, &Device::Cpu).expect("cond hr");

    let started = std::time::Instant::now();
    let cascade = pipeline
        .sample_shape_slat_cascade(
            &cond_lr,
            &cond_hr,
            &coords,
            None,
            42,
            steps,
            49152,
            &Device::Cpu,
        )
        .expect("cascade shape stage");
    println!(
        "cascade shape stage in {:.1}s",
        started.elapsed().as_secs_f64()
    );

    assert_eq!(cascade.lr_slat.dim(0).expect("lr rows"), coords.len());
    assert_eq!(cascade.hr_resolution, 1024);
    assert!(!cascade.quant_coords.is_empty());
    assert!(
        cascade.quant_coords.len() <= 49152,
        "{} HR tokens",
        cascade.quant_coords.len()
    );
    assert_eq!(
        cascade.hr_slat.dim(0).expect("hr rows"),
        cascade.quant_coords.len()
    );
    assert_eq!(cascade.hr_slat.dim(1).expect("hr channels"), 32);
    for coord in &cascade.quant_coords {
        assert!(
            coord.iter().all(|&v| (0..64).contains(&v)),
            "quant {coord:?}"
        );
    }

    // Decode the HR slat at hr_resolution through the pipeline surface.
    let decode_started = std::time::Instant::now();
    let stages = pipeline
        .decode_shape_slat(
            &cascade.quant_coords,
            &cascade.hr_slat,
            cascade.hr_resolution,
            &Device::Cpu,
        )
        .expect("HR decode");
    println!(
        "HR decode in {:.1}s",
        decode_started.elapsed().as_secs_f64()
    );
    let mesh = stages.mesh.expect("mesh");
    mesh.validate().expect("mesh buffers valid");
    assert_eq!(mesh.resolution, 1024);
    assert!(mesh.face_count() > 0, "decode must produce faces");
    assert!(!mesh.has_non_finite_vertex());
    println!(
        "cascade smoke mesh: {} vertices / {} faces at resolution {}",
        mesh.vertex_count(),
        mesh.face_count(),
        mesh.resolution
    );
}

// ---------------------------------------------------------------------------
// 5. Env-gated CUDA e2e: full image -> mesh 1024 cascade
// ---------------------------------------------------------------------------

/// Peak-VRAM sampler: polls `nvidia-smi` in a background thread.
#[cfg(feature = "cuda")]
struct VramPoller {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    peak_mib: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "cuda")]
impl VramPoller {
    fn start() -> Option<Self> {
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.used")
            .arg("--format=csv,noheader,nounits")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let peak = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let stop_thread = stop.clone();
        let peak_thread = peak.clone();
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(std::sync::atomic::Ordering::Relaxed) {
                if let Ok(output) = std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=memory.used")
                    .arg("--format=csv,noheader,nounits")
                    .output()
                {
                    if let Ok(text) = String::from_utf8(output.stdout) {
                        if let Ok(mib) = text.trim().parse::<usize>() {
                            peak_thread.fetch_max(mib, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(500));
            }
        });
        Some(Self {
            stop,
            peak_mib: peak,
            handle: Some(handle),
        })
    }

    fn peak_mib(&self) -> usize {
        self.peak_mib.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(feature = "cuda")]
impl Drop for VramPoller {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Full 1024 cascade image -> mesh on CUDA real weights. Gated behind
/// `LUX3D_TRELLIS2_RUN_CASCADE=1` and the `cuda` feature; runs the reference
/// 12 steps unless `LUX3D_TRELLIS2_CASCADE_STEPS` overrides (use 2 for a
/// smoke pass). Peak VRAM is sampled through `nvidia-smi` every 500 ms.
#[cfg(feature = "cuda")]
#[test]
fn cascade_full_cuda_e2e() {
    if std::env::var("LUX3D_TRELLIS2_RUN_CASCADE").ok().as_deref() != Some("1") {
        eprintln!("SKIP: set LUX3D_TRELLIS2_RUN_CASCADE=1 to run the CUDA 1024 cascade e2e");
        return;
    }
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: TRELLIS.2 raw checkpoint dir not found (set LUX3D_TRELLIS2_RAW_DIR)");
        return;
    };
    let device = Device::new_cuda(0).expect("CUDA device");

    // Synthetic RGBA input: opaque rounded square on transparent background.
    let temp = std::env::temp_dir().join(format!("lux3d-trellis2-cascade-{}", std::process::id()));
    std::fs::create_dir_all(&temp).expect("temp dir");
    let image_path = temp.join("input.png");
    {
        let mut image = image::RgbaImage::new(96, 96);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            let inside = (16..80).contains(&x) && (16..80).contains(&y);
            *pixel = if inside {
                image::Rgba([180, 120, 60, 255])
            } else {
                image::Rgba([0, 0, 0, 0])
            };
        }
        image.save(&image_path).expect("save input png");
    }

    let pipeline = Trellis2Pipeline::load_from_raw_dir(&raw, &device).expect("raw-dir pipeline");
    let steps = steps_override();
    let options = lux3d_core::runtime::Trellis2RunOptions {
        pipeline_type: lux3d_core::contracts::Trellis2PipelineType::Cascade1024,
        ss_steps: steps,
        shape_steps: steps,
        ..lux3d_core::runtime::Trellis2RunOptions::default()
    };

    let vram = VramPoller::start();
    let started = std::time::Instant::now();
    let stages = pipeline
        .run_from_path_with_stages(&image_path, &device, &options)
        .expect("cascade e2e run");
    let elapsed = started.elapsed().as_secs_f64();
    let mesh = stages.mesh.expect("cascade mesh");
    println!(
        "cascade e2e: {steps} steps, {} HR tokens, hr_resolution {:?}, {} vertices, {} faces, \
         {elapsed:.1}s, peak VRAM {} MiB",
        stages.quant_coords.as_ref().map(|c| c.len()).unwrap_or(0),
        stages.hr_resolution,
        mesh.vertex_count(),
        mesh.face_count(),
        vram.as_ref().map(|p| p.peak_mib()).unwrap_or(0),
    );

    // Acceptance: HR tokens within the budget, hr_resolution 1024 (the
    // reference loop cannot reduce the 1024 cascade), non-empty finite mesh.
    let quant = stages.quant_coords.as_ref().expect("cascade stages");
    assert!(!quant.is_empty());
    assert!(quant.len() <= 49152, "{} HR tokens", quant.len());
    assert_eq!(stages.hr_resolution, Some(1024));
    assert_eq!(stages.shape_coords.len(), quant.len());
    assert_eq!(mesh.resolution, 1024);
    assert!(mesh.face_count() > 0, "decode must produce faces");
    mesh.validate().expect("mesh buffers valid");
    assert!(!mesh.has_non_finite_vertex());

    // Export the produced mesh.
    let voxel_count = mesh.voxel_coords.len() / 3;
    let attr_channels = if voxel_count == 0 || mesh.voxel_attrs.is_empty() {
        7
    } else {
        mesh.voxel_attrs.len() / voxel_count
    };
    let buffers = lux3d_core::geometry::Trellis2MeshBuffers {
        resolution: mesh.resolution,
        vertices: mesh.vertices.clone(),
        faces: mesh.faces.clone(),
        voxel_coords: mesh.voxel_coords.clone(),
        voxel_attrs: mesh.voxel_attrs.clone(),
        voxel_attr_channels: attr_channels,
        vertex_colors: mesh.vertex_colors.clone(),
    };
    let obj_path = temp.join("cascade_mesh.obj");
    pipeline.export_mesh(&buffers, &obj_path).expect("export");

    // Export matrix: PLY + GLB always; with the texture stage on (default)
    // the mesh carries 6-channel PBR attrs + vertex colors, so the PBR
    // sidecar pair must round-trip too.
    let stage = lux3d_core::export::Trellis2ExportStage;
    let ply_path = temp.join("cascade_mesh.ply");
    let glb_path = temp.join("cascade_mesh.glb");
    stage.write_mesh(&buffers, &ply_path).expect("export ply");
    stage.write_mesh(&buffers, &glb_path).expect("export glb");
    assert!(ply_path.is_file() && ply_path.metadata().unwrap().len() > 0);
    assert!(glb_path.is_file() && glb_path.metadata().unwrap().len() > 0);
    let colors = buffers.vertex_colors.as_ref();
    println!(
        "cascade export: obj={} bytes, ply={} bytes, glb={} bytes, vertex_colors={}",
        obj_path.metadata().unwrap().len(),
        ply_path.metadata().unwrap().len(),
        glb_path.metadata().unwrap().len(),
        colors.is_some()
    );
    if buffers.voxel_attr_channels == 6 {
        let sidecar = stage
            .write_mesh_with_pbr(&buffers, &glb_path)
            .expect("export pbr pair");
        let bin_bytes = std::fs::read(&sidecar).expect("sidecar bytes");
        assert_eq!(&bin_bytes[..8], lux3d_core::export::pbrvox::MAGIC.as_slice());
        let count = u64::from_le_bytes(bin_bytes[16..24].try_into().expect("count"));
        assert_eq!(count, buffers.voxel_count() as u64);
        let json_path = temp.join("cascade_mesh.pbr.json");
        let json: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&json_path).expect("pbr json"),
        )
        .expect("pbr json parse");
        assert_eq!(json["resolution"], buffers.resolution);
        assert_eq!(json["voxel_count"], buffers.voxel_count());
        println!("cascade pbr sidecar: {} voxels, {}", count, sidecar.display());
    }
    drop(vram);
    let _ = std::fs::remove_dir_all(&temp);
}

/// Non-CUDA builds: the CUDA e2e is documented as a skip so the acceptance
/// matrix is explicit in default CI runs.
#[cfg(not(feature = "cuda"))]
#[test]
fn cascade_full_cuda_e2e_requires_cuda_feature() {
    if std::env::var("LUX3D_TRELLIS2_RUN_CASCADE").ok().as_deref() != Some("1") {
        return;
    }
    eprintln!(
        "SKIP: LUX3D_TRELLIS2_RUN_CASCADE=1 requires the `cuda` feature \
         (`cargo test --release -p lux3d-core --features cuda --test trellis2_cascade \
         cascade_full_cuda_e2e -- --nocapture --test-threads 1`)"
    );
}

/// The unused-import set below is intentionally minimal: keep `HashSet`
/// referenced (used by the sorted-unique checks if extended) without
/// triggering dead-code warnings.
#[test]
fn quantize_output_is_a_set() {
    let hr_coords: Vec<[i32; 3]> = vec![[0, 0, 0], [511, 511, 511], [0, 0, 0]];
    let (coords, _) = quantize_hr_coords(
        &hr_coords,
        TRELLIS2_CASCADE_LR_RESOLUTION,
        TRELLIS2_CASCADE_HR_RESOLUTION,
        49152,
    );
    let set: HashSet<[i32; 3]> = coords.iter().copied().collect();
    assert_eq!(set.len(), coords.len());
    assert_eq!(coords, vec![[0, 0, 0], [63, 63, 63]]);
}
