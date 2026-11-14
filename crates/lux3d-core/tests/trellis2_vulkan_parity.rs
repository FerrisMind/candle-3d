//! Gated Vulkan-vs-CPU numeric parity regression for the TRELLIS.2 sparse
//! ops and shape VAE decode.
//!
//! Background (T9b, 2026-11-14): candle's Vulkan `index_select` dispatches
//! the index count as the Y workgroup count without clamping to the device
//! limit (`vulkan_backend.rs::run_index_select_f32`; NVIDIA reports
//! maxComputeWorkGroupCount Y/Z = 65535). Any gather with more than 65535
//! indices is silently corrupt there — measured on an RTX 3060: 65535
//! indices exact, 65536 indices ~96% wrong rows. The TRELLIS.2 shape VAE
//! gathers exceed that at resolution 512 (sparse-conv pair counts and C2S
//! child counts reach ~500k), which collapsed the decoded `intersected`
//! flags and lost ~2/3 of the mesh quads (z-axis worst). The lux3d-side fix
//! is `index_select_rows` in `runtime/trellis2_sparse.rs`: Vulkan gathers
//! are split into 65535-index chunks (bit-identical by construction).
//!
//! This suite guards that fix and documents the upstream defect:
//! 1. Golden-fixture VAE decode (64-voxel latent, canonical weights) on CPU
//!    vs Vulkan: coords exact, per-block feats parity, mesh face multiset
//!    equal, per-axis quad counts equal.
//! 2. Production-shape synthetic SparseConv3d (200k coords x 128 ch — pair
//!    counts above the dispatch limit) CPU vs Vulkan, twice for run-to-run
//!    drift. This is the regression guard for the chunked gather.
//! 3. Synthetic `Tensor::index_add` accumulation (500k rows, duplicate
//!    destinations, 27 passes) — must stay exact (CAS-atomic scatter).
//! 4. Raw `Tensor::index_select` boundary probes (65535/65536) — print-only
//!    documentation of the upstream candle defect; they are expected to
//!    FAIL parity at 65536 until candle clamps the dispatch.
//!
//! Gated by `LUX3D_TRELLIS2_VULKAN_PARITY=1`; requires a Vulkan device and
//! the `vulkan` cargo feature. `LUX3D_TRELLIS2_VULKAN_PARITY_COOPMAT`
//! (default `0`) pins `CANDLE_VULKAN_F32_UNALIGNED_COOPMAT` before any
//! Vulkan dispatch: the product CLI sets that variable only when absent, so
//! an explicit value overrides the shipped TF32-class coopmat default and
//! the suite always exercises exact-fp32 matmul.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

fn golden_dir() -> PathBuf {
    lux3d_core::test_support::workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-vae-golden")
}

fn canonical_model_file() -> PathBuf {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_CANONICAL_MODEL") {
        return PathBuf::from(path);
    }
    lux3d_core::test_support::canonical_package_dir(lux3d_core::ModelFamily::Trellis2)
        .join("model.safetensors")
}

/// Deterministic pattern value (index-derived, identical on every device).
fn pattern_value(i: usize, seed: usize) -> f32 {
    (((i * 7 + seed * 3) % 23) as f32 - 11.0) * 0.05
}

struct Stats {
    max_abs: f64,
    mean_abs: f64,
    rel: f64,
}

fn stats(a: &[f32], b: &[f32]) -> Stats {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut sum_b_sq = 0.0_f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x - *y).abs() as f64;
        max_abs = max_abs.max(d);
        sum_abs += d;
        sum_sq += d * d;
        sum_b_sq += (*y as f64) * (*y as f64);
    }
    let n = a.len() as f64;
    let rms = (sum_sq / n).sqrt();
    let ref_rms = (sum_b_sq / n).sqrt().max(1e-30);
    Stats {
        max_abs,
        mean_abs: sum_abs / n,
        rel: rms / ref_rms,
    }
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .expect("to cpu")
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec")
}

/// Per-axis quad counts the extractor would emit for these head fields
/// (diagnostic re-implementation of the emission loop).
fn quad_counts(coords: &[[i32; 3]], intersected: &[[bool; 3]]) -> [usize; 3] {
    const OFFSETS: [[[i32; 3]; 4]; 3] = [
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
    ];
    let map: HashMap<[i32; 3], u32> = coords
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i as u32))
        .collect();
    let mut counts = [0usize; 3];
    for (i, coord) in coords.iter().enumerate() {
        for axis in 0..3 {
            if !intersected[i][axis] {
                continue;
            }
            if OFFSETS[axis].iter().all(|o| {
                map.contains_key(&[coord[0] + o[0], coord[1] + o[1], coord[2] + o[2]])
            }) {
                counts[axis] += 1;
            }
        }
    }
    counts
}

fn face_multiset(mesh: &lux3d_core::runtime::DualGridMesh) -> HashSet<[i64; 3]> {
    let mut set = HashSet::new();
    for face in mesh.faces.chunks_exact(3) {
        let mut t = [face[0], face[1], face[2]];
        t.sort_unstable();
        set.insert(t);
    }
    set
}

fn to_mesh(
    decoded: &lux3d_core::runtime::ShapeVaeDecodeOutput,
) -> (
    lux3d_core::runtime::DualGridMesh,
    lux3d_core::runtime::DualGridFields,
) {
    let fields = lux3d_core::runtime::dual_grid_fields(
        &decoded.feats,
        lux3d_core::runtime::TRELLIS2_VOXEL_MARGIN,
    )
    .expect("fields");
    let mesh = lux3d_core::runtime::flexible_dual_grid_to_mesh(
        &decoded.coords,
        &fields.dual,
        &fields.intersected,
        &fields.split_weight,
        [128, 128, 128],
        [[-0.5_f32; 3], [0.5_f32; 3]],
    );
    (mesh, fields)
}

#[test]
fn trellis2_vulkan_sparse_ops_and_vae_parity() {
    if std::env::var("LUX3D_TRELLIS2_VULKAN_PARITY").as_deref() != Ok("1") {
        eprintln!(
            "SKIP: set LUX3D_TRELLIS2_VULKAN_PARITY=1 to run the Vulkan parity regression"
        );
        return;
    }
    // Pin the coopmat opt-in before any Vulkan dispatch (explicit values
    // win over the product default because the CLI only fills the variable
    // when absent).
    let coop =
        std::env::var("LUX3D_TRELLIS2_VULKAN_PARITY_COOPMAT").unwrap_or_else(|_| "0".to_string());
    if std::env::var_os("CANDLE_VULKAN_F32_UNALIGNED_COOPMAT").is_none() {
        // SAFETY: single-threaded test setup, before any reader of the var.
        unsafe { std::env::set_var("CANDLE_VULKAN_F32_UNALIGNED_COOPMAT", &coop) };
    }
    println!("coopmat opt-in = {coop}");

    let vulkan = match Device::new_vulkan(0) {
        Ok(device) => device,
        Err(err) => {
            eprintln!("SKIP: no Vulkan device: {err}");
            return;
        }
    };
    let golden = golden_dir();
    if !golden.join("shape_mesh.npz").is_file() {
        eprintln!("SKIP: VAE oracle golden missing at {}", golden.display());
        return;
    }
    let model_file = canonical_model_file();
    if !model_file.is_file() {
        eprintln!(
            "SKIP: canonical TRELLIS.2 weights not found at {}",
            model_file.display()
        );
        return;
    }

    use ndarray::Array2;
    let slat_coords: Array2<i32> =
        ndarray_npy::read_npy(golden.join("slat_coords.npy")).expect("slat_coords");
    let slat_feats: Array2<f32> =
        ndarray_npy::read_npy(golden.join("slat_feats.npy")).expect("slat_feats");
    let coords: Vec<[i32; 3]> = (0..slat_coords.dim().0)
        .map(|i| [slat_coords[[i, 0]], slat_coords[[i, 1]], slat_coords[[i, 2]]])
        .collect();

    let decode_on = |device: &Device| {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_file], DType::F32, device) }
                .expect("mmaped canonical weights");
        let decoder = lux3d_core::runtime::Trellis2ShapeVaeDecoder::load(
            vb.pp("shape_dec"),
            &lux3d_core::runtime::Trellis2ShapeVaeConfig::flexi_dual_shape(),
        )
        .expect("shape_dec load");
        let latent = Tensor::from_vec(
            slat_feats.iter().copied().collect::<Vec<f32>>(),
            (slat_feats.dim().0, slat_feats.dim().1),
            device,
        )
        .expect("latent");
        decoder.decode(&coords, &latent, true).expect("decode")
    };

    let cpu = decode_on(&Device::Cpu);
    let vk1 = decode_on(&vulkan);
    let vk2 = decode_on(&vulkan);

    // Vulkan must be run-to-run deterministic here (the index_add CAS
    // scatter is order-nondeterministic in principle; at this scale the
    // sums are far from any threshold, so expect bit-identical output).
    {
        let a = flat(&vk1.feats);
        let b = flat(&vk2.feats);
        let s = stats(&a, &b);
        println!(
            "vulkan run-to-run feats: max_abs {:.3e} mean_abs {:.3e} rel {:.3e}",
            s.max_abs, s.mean_abs, s.rel
        );
        assert!(
            s.max_abs <= 1e-5,
            "vulkan run-to-run drift {:.3e}",
            s.max_abs
        );
    }

    // Per-block parity CPU vs Vulkan: coords exact, feats ~fp32-order noise.
    let mut worst = (0.0_f64, String::new());
    assert_eq!(cpu.trace.len(), vk1.trace.len(), "block count");
    for (cb, vb) in cpu.trace.iter().zip(vk1.trace.iter()) {
        assert_eq!(
            cb.coords, vb.coords,
            "stage {} block {} coords",
            cb.stage, cb.block
        );
        let a = flat(&vb.feats);
        let b = flat(&cb.feats);
        let s = stats(&a, &b);
        println!(
            "blocks.{}.{}: n={} ch={} max_abs {:.3e} mean_abs {:.3e} rel {:.3e}",
            cb.stage,
            cb.block,
            cb.coords.len(),
            cb.feats.dim(1).expect("channels"),
            s.max_abs,
            s.mean_abs,
            s.rel
        );
        if s.max_abs > worst.0 {
            worst = (s.max_abs, format!("blocks.{}.{}", cb.stage, cb.block));
        }
    }
    println!("worst block: {} max_abs {:.3e}", worst.1, worst.0);
    assert!(
        worst.0 <= 5e-4,
        "vulkan/cpu block parity worst max_abs {:.3e}",
        worst.0
    );

    // Mesh parity: identical quad counts per axis and identical face set.
    let (cpu_mesh, cpu_fields) = to_mesh(&cpu);
    let (vk_mesh, vk_fields) = to_mesh(&vk1);
    let cpu_quads = quad_counts(&cpu.coords, &cpu_fields.intersected);
    let vk_quads = quad_counts(&vk1.coords, &vk_fields.intersected);
    println!(
        "cpu:    coords {} quads x {} y {} z {} faces {}",
        cpu.coords.len(),
        cpu_quads[0],
        cpu_quads[1],
        cpu_quads[2],
        cpu_mesh.faces.len() / 3
    );
    println!(
        "vulkan: coords {} quads x {} y {} z {} faces {}",
        vk1.coords.len(),
        vk_quads[0],
        vk_quads[1],
        vk_quads[2],
        vk_mesh.faces.len() / 3
    );
    assert_eq!(cpu_quads, vk_quads, "per-axis quad counts must match");
    let cpu_faces = face_multiset(&cpu_mesh);
    let vk_faces = face_multiset(&vk_mesh);
    assert_eq!(cpu_faces, vk_faces, "face multisets must match");

    // Production-shape synthetic SparseConv3d: pair counts above the 65535
    // dispatch limit (the T9b defect) must stay exact through the chunked
    // gather and be run-to-run stable.
    println!("--- synthetic SparseConv3d (dense-cube coords) ---");
    sparse_conv_probe(&vulkan, 200_000, 128, 128, 123);
    sparse_conv_probe(&vulkan, 60_000, 1024, 1024, 321);
    // index_add accumulation with duplicate destinations: exact and stable.
    println!("--- synthetic index_add accumulation ---");
    index_add_probe(&vulkan, 500_000, 128);
    // Raw index_select boundary probes: documentation of the upstream candle
    // defect (print-only; 65536+ ids is silently corrupt until candle clamps
    // the Y dispatch).
    println!("--- raw index_select boundary probes (upstream defect, print-only) ---");
    for (rows, width) in [
        (60_000usize, 128usize),
        (65_535, 128),
        (65_536, 128),
        (70_000, 128),
        (200_000, 8),
    ] {
        raw_index_select_probe(&vulkan, rows, width);
    }
}

/// SparseConv3d forward on CPU vs Vulkan (twice): dense-cube coords,
/// deterministic pattern activations/weights. Asserts parity and stability.
fn sparse_conv_probe(vulkan: &Device, rows: usize, cin: usize, cout: usize, seed: usize) {
    let side = (rows as f64).cbrt().ceil() as i32;
    let mut coords = Vec::with_capacity(rows);
    for x in 0..side {
        for y in 0..side {
            for z in 0..side {
                if coords.len() < rows {
                    coords.push([x, y, z]);
                }
            }
        }
    }
    let rows = coords.len();
    let feats_cpu = Tensor::from_vec(
        (0..rows * cin).map(|i| pattern_value(i, seed)).collect(),
        (rows, cin),
        &Device::Cpu,
    )
    .expect("feats");
    let weight_cpu = Tensor::from_vec(
        (0..cout * 3 * 3 * 3 * cin)
            .map(|i| pattern_value(i, seed + 1) * 0.1)
            .collect(),
        (cout, 3, 3, 3, cin),
        &Device::Cpu,
    )
    .expect("weight");
    let bias_cpu = Tensor::from_vec(
        (0..cout).map(|i| pattern_value(i, seed + 2)).collect(),
        (cout,),
        &Device::Cpu,
    )
    .expect("bias");

    let run = |device: &Device| {
        let map: HashMap<String, Tensor> = HashMap::from([
            (
                "weight".to_string(),
                weight_cpu.to_device(device).expect("w"),
            ),
            ("bias".to_string(), bias_cpu.to_device(device).expect("b")),
        ]);
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        let conv =
            lux3d_core::runtime::SparseConv3d::load(vb, cin, cout, true).expect("conv load");
        let feats = feats_cpu.to_device(device).expect("x");
        conv.forward(&coords, &feats).expect("conv forward")
    };
    let cpu_out = flat(&run(&Device::Cpu));
    let vk1 = flat(&run(vulkan));
    let vk2 = flat(&run(vulkan));
    let s1 = stats(&vk1, &cpu_out);
    let s12 = stats(&vk2, &vk1);
    println!(
        "conv n={rows} cin={cin} cout={cout}: vk-vs-cpu max_abs {:.3e} mean_abs {:.3e} rel {:.3e} | vk-run-to-run max_abs {:.3e}",
        s1.max_abs, s1.mean_abs, s1.rel, s12.max_abs
    );
    assert!(
        s1.rel <= 1e-5,
        "vulkan conv parity n={rows} cin={cin}: rel {:.3e} (chunked gather regressed?)",
        s1.rel
    );
    assert!(
        s12.max_abs <= 1e-5,
        "vulkan conv run-to-run drift n={rows} cin={cin}: max_abs {:.3e}",
        s12.max_abs
    );
}

/// index_add probe: 27 sequential passes emulating SparseConv3d accumulation
/// with 2-way duplicate destinations, `rows` x `width` F32. Asserts parity.
fn index_add_probe(vulkan: &Device, rows: usize, width: usize) {
    let run = |device: &Device| {
        let ids = Tensor::from_vec(
            (0..rows).map(|i| (i % (rows / 2)) as u32).collect(),
            (rows,),
            device,
        )
        .expect("ids");
        let src = Tensor::from_vec(
            (0..rows * width).map(|i| pattern_value(i, 9)).collect(),
            (rows, width),
            device,
        )
        .expect("src");
        let mut dst = Tensor::zeros((rows / 2, width), DType::F32, device).expect("dst");
        for _ in 0..27 {
            dst = dst.index_add(&ids, &src, 0).expect("index_add");
        }
        dst
    };
    let cpu_out = flat(&run(&Device::Cpu));
    let vk1 = flat(&run(vulkan));
    let vk2 = flat(&run(vulkan));
    let s1 = stats(&vk1, &cpu_out);
    let s12 = stats(&vk2, &vk1);
    println!(
        "index_add rows={rows} width={width} passes=27 (2-way dup): vk-vs-cpu max_abs {:.3e} mean_abs {:.3e} rel {:.3e} | vk-run-to-run max_abs {:.3e}",
        s1.max_abs, s1.mean_abs, s1.rel, s12.max_abs
    );
    assert!(
        s1.max_abs <= 1e-4,
        "vulkan index_add parity rows={rows}: max_abs {:.3e}",
        s1.max_abs
    );
    assert!(s12.max_abs <= 1e-4, "vulkan index_add run-to-run drift");
}

/// Raw `Tensor::index_select` boundary probe (bypasses the lux3d chunked
/// helper): documents the upstream candle Vulkan defect — print-only.
fn raw_index_select_probe(vulkan: &Device, rows: usize, width: usize) {
    let run = |device: &Device| {
        let table = Tensor::from_vec(
            (0..(rows + 4096) * width)
                .map(|i| pattern_value(i, 17))
                .collect(),
            (rows + 4096, width),
            device,
        )
        .expect("table");
        let ids = Tensor::from_vec(
            (0..rows)
                .map(|i| ((i * 7919) % (rows + 4096)) as u32)
                .collect(),
            (rows,),
            device,
        )
        .expect("ids");
        table.index_select(&ids, 0).expect("index_select")
    };
    let cpu_out = flat(&run(&Device::Cpu));
    let vk1 = flat(&run(vulkan));
    let s1 = stats(&vk1, &cpu_out);
    let bad_rows = vk1
        .chunks_exact(width)
        .zip(cpu_out.chunks_exact(width))
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-5))
        .count();
    println!(
        "index_select rows={rows} width={width}: vk-vs-cpu max_abs {:.3e} rel {:.3e} | bad rows {bad_rows}/{rows}",
        s1.max_abs, s1.rel
    );
}
