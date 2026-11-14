//! Golden tests for the TRELLIS.2 preprocess + DINOv3 ViT-L/16
//! conditioning stage.
//!
//! Golden source: `.swarm/results/trellis2-dino-golden/` produced by
//! `.swarm/scripts/trellis2_dino_golden.py` (transformers 5.17.0 on CPU,
//! reusing the SS-oracle preprocess artifacts of the T.png example). The
//! DINOv3 test additionally requires the real weights at
//! `models/TRELLIS.2-4B/external/dinov3` (or `LUX3D_TRELLIS2_DINO_DIR`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::Device;
use lux3d_core::preprocess::{
    TRELLIS2_IMAGE_NET_MEAN, TRELLIS2_IMAGE_NET_STD, Trellis2Preprocess,
    normalized_pixel_values_from_u8,
};
use lux3d_core::runtime::{Trellis2Dino, Trellis2DinoConfig};
use lux3d_core::test_support::workspace_root;

fn golden_dir() -> PathBuf {
    workspace_root()
        .join(".swarm")
        .join("results")
        .join("trellis2-dino-golden")
}

fn dinov3_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("LUX3D_TRELLIS2_DINO_DIR") {
        return PathBuf::from(path);
    }
    workspace_root()
        .join("models")
        .join("TRELLIS.2-4B")
        .join("external")
        .join("dinov3")
}

fn read_bytes(path: &Path) -> Option<Vec<u8>> {
    std::fs::read(path).ok()
}

fn read_f32(path: &Path) -> Option<Vec<f32>> {
    read_bytes(path).map(|bytes| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    })
}

/// Byte-exact preprocess parity: the Rust alpha path must reproduce the
/// Python golden `image_512_uint8` (the LANCZOS 512x512 resize of the
/// premultiplied square crop) and the intermediate square crop exactly.
#[test]
fn trellis2_preprocess_matches_python_golden() {
    let golden = golden_dir();
    let expected_image = match read_bytes(&golden.join("image_512_u8_rgb.bin")) {
        Some(bytes) => bytes,
        None => {
            eprintln!("SKIP: golden not generated (run .swarm/scripts/trellis2_dino_golden.py)");
            return;
        }
    };
    let source = PathBuf::from(
        "G:/jobs/candle-based-projects/candle-3d-refs/TRELLIS.2/assets/example_image/T.png",
    );
    if !source.is_file() {
        eprintln!("SKIP: reference T.png not present at {source:?}");
        return;
    }

    let stage = Trellis2Preprocess::new(512).expect("stage");
    let prepared = stage
        .prepare_inputs_from_path(&source)
        .expect("prepared image");

    assert_eq!(prepared.target_size.width, 512);
    assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);
    let mismatches: Vec<usize> = prepared
        .uint8_rgb
        .iter()
        .zip(expected_image.iter())
        .enumerate()
        .filter_map(|(index, (a, b))| (a != b).then_some(index))
        .take(8)
        .collect();
    assert!(
        prepared.uint8_rgb == expected_image,
        "512x512 DINOv3 input pixels must match the Python golden exactly; first \
         mismatches (index, actual, expected): {mismatches:?}"
    );
}

/// Real-weights golden comparison: DINOv3 ViT-L/16 forward on the golden
/// 512x512 input must reproduce the Python cond tensor. Target <= 1e-3
/// relative (vs the max |cond| norm).
#[test]
fn trellis2_dino_forward_matches_python_golden() {
    let golden = golden_dir();
    let expected_cond = match read_f32(&golden.join("cond_cpu_f32.bin")) {
        Some(values) => values,
        None => {
            eprintln!("SKIP: golden not generated (run .swarm/scripts/trellis2_dino_golden.py)");
            return;
        }
    };
    let golden_image = match read_bytes(&golden.join("image_512_u8_rgb.bin")) {
        Some(bytes) => bytes,
        None => {
            eprintln!("SKIP: golden image missing");
            return;
        }
    };
    let weights_dir = dinov3_dir();
    if !weights_dir.join("model.safetensors").is_file() {
        eprintln!(
            "SKIP: DINOv3 weights not found at {} (set LUX3D_TRELLIS2_DINO_DIR)",
            weights_dir.display()
        );
        return;
    }

    let device = Device::Cpu;
    let config = Trellis2DinoConfig::vitl16();
    assert_eq!(config.token_count(512), 1029);

    let load_started = Instant::now();
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[weights_dir.join("model.safetensors")],
            candle_core::DType::F32,
            &device,
        )
    }
    .expect("mmaped safetensors");
    let dino = Trellis2Dino::load(vb, &config).expect("dino load");
    let load_seconds = load_started.elapsed().as_secs_f64();

    let pixel_values =
        normalized_pixel_values_from_u8(&golden_image, 512, &device).expect("normalize");
    assert_eq!(pixel_values.dims(), &[1, 3, 512, 512]);

    let forward_started = Instant::now();
    let cond = dino.forward(&pixel_values).expect("dino forward");
    let forward_seconds = forward_started.elapsed().as_secs_f64();
    assert_eq!(cond.dims(), &[1, 1029, 1024]);

    let actual = cond
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("vec");
    assert_eq!(actual.len(), expected_cond.len());

    let mut max_abs = 0.0_f64;
    let mut golden_max = 0.0_f64;
    let mut sum_sq_diff = 0.0_f64;
    let mut sum_sq_golden = 0.0_f64;
    let mut rel_worst = 0.0_f64;
    for (&a, &b) in actual.iter().zip(expected_cond.iter()) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(f64::from(diff));
        golden_max = golden_max.max(f64::from(b.abs()));
        sum_sq_diff += f64::from(diff) * f64::from(diff);
        sum_sq_golden += f64::from(b) * f64::from(b);
        if b.abs() > 0.5 {
            rel_worst = rel_worst.max(f64::from(diff) / f64::from(b.abs()));
        }
    }
    let rel_vs_max = max_abs / golden_max;
    let rms_rel = (sum_sq_diff / sum_sq_golden).sqrt();
    println!(
        "trellis2_dino @512 CPU: load {load_seconds:.1}s, forward {forward_seconds:.1}s, \
         max_abs={max_abs:.3e}, rel_vs_max_abs={rel_vs_max:.3e}, rms_rel={rms_rel:.3e}, \
         elementwise_rel(|golden|>0.5)={rel_worst:.3e}"
    );
    assert!(
        rel_vs_max <= 1e-3,
        "cond relative error {rel_vs_max:.3e} exceeds the 1e-3 target \
         (max_abs={max_abs:.3e})"
    );
    assert!(
        rms_rel <= 1e-3,
        "cond RMS relative error {rms_rel:.3e} exceeds the 1e-3 target"
    );

    // Secondary reference: the SS-oracle CUDA cond. Compute-path differences
    // (CUDA eager vs CPU f32) make this looser; failures here are reported
    // and asserted at the same 1e-3 bound, which the Python-side CPU-vs-CUDA
    // gap already satisfies.
    if let Some(expected_cuda) = read_f32(&golden.join("cond_cuda_f32.bin")) {
        let mut cuda_max_abs = 0.0_f64;
        let mut cuda_golden_max = 0.0_f64;
        for (&a, &b) in actual.iter().zip(expected_cuda.iter()) {
            cuda_max_abs = cuda_max_abs.max(f64::from((a - b).abs()));
            cuda_golden_max = cuda_golden_max.max(f64::from(b.abs()));
        }
        let cuda_rel = cuda_max_abs / cuda_golden_max;
        println!(
            "trellis2_dino vs CUDA oracle: max_abs={cuda_max_abs:.3e}, rel_vs_max_abs={cuda_rel:.3e}"
        );
        assert!(
            cuda_rel <= 1e-3,
            "CUDA-oracle relative error {cuda_rel:.3e}"
        );
    }
}

/// Normalization sanity: the pixel values fed to the model must equal
/// `(v / 255 - mean) / std` per channel in CHW order.
#[test]
fn trellis2_normalization_matches_imagenet_formula() {
    if golden_dir().join("image_512_u8_rgb.bin").is_file() {
        let golden_image = read_bytes(&golden_dir().join("image_512_u8_rgb.bin")).expect("image");
        let tensor =
            normalized_pixel_values_from_u8(&golden_image, 512, &Device::Cpu).expect("tensor");
        let values = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("vec");
        // First pixel (row 0, col 0), channel R.
        let expected_r = (f32::from(golden_image[0]) / 255.0 - TRELLIS2_IMAGE_NET_MEAN[0])
            / TRELLIS2_IMAGE_NET_STD[0];
        assert!((values[0] - expected_r).abs() < 1e-6);
        // Plane stride: channel G of the same pixel lives at 512*512.
        let expected_g = (f32::from(golden_image[1]) / 255.0 - TRELLIS2_IMAGE_NET_MEAN[1])
            / TRELLIS2_IMAGE_NET_STD[1];
        assert!((values[512 * 512] - expected_g).abs() < 1e-6);
    }
}
