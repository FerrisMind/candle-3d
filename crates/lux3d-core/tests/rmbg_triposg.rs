//! RMBG-1.4 parity against the reference PyTorch preprocessing baseline.
//!
//! The baseline (`tools/python_baseline/rmbg_dump.py`) captures the exact
//! `scripts/image_process.py` path TripoSG uses when an image has no valid
//! alpha channel: 1024x1024 torchvision bilinear-antialias resize, the local
//! `(x - 0.5) / 1.0` normalization wrapper, BriaRMBG, then the sigmoid side-1
//! output.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use lux3d_core::test_support::runtime_root;
use lux3d_core::{RmbgConfig, RmbgModel, load_auxiliary_canonical_weights};

fn rmbg_package_dir() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("canonical-weights")
        .join("rmbg-1.4")
}

fn baseline_path() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("rmbg")
        .join("smoke")
        .join("artifacts.safetensors")
}

fn baseline_tensor(name: &str, device: &Device) -> Tensor {
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(baseline_path())
            .expect("RMBG baseline safetensors")
    }
    .load(name, device)
    .expect("baseline tensor")
}

#[test]
fn rmbg_auxiliary_package_manifest_and_checksums_validate() {
    let package_dir = rmbg_package_dir();
    let package = load_auxiliary_canonical_weights(&package_dir, "rmbg")
        .expect("load RMBG auxiliary package");
    assert_eq!(package.manifest.family, "rmbg");
    assert_eq!(package.manifest.tensor_count, 798);
    assert_eq!(
        package.manifest.dtype_histogram.get("float32").copied(),
        Some(686)
    );
    assert_eq!(
        package.manifest.dtype_histogram.get("int64").copied(),
        Some(112)
    );
    assert_eq!(
        package.manifest.raw_files.as_ref(),
        &[
            "model.safetensors".to_string(),
            "config.json".to_string(),
            "preprocessor_config.json".to_string()
        ]
    );
    assert_eq!(
        package.manifest.source_checksums.get("model.safetensors"),
        Some(&"46ef7fe46f2ae284d8f1aaa24bfa5fca5ef25a34e2c7caa890a0029eb100e87f".to_string())
    );
    let tensor_count = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&package.canonical_file)
            .expect("RMBG canonical safetensors")
    }
    .tensors()
    .len();
    assert_eq!(tensor_count, package.manifest.tensor_count);

    let error = load_auxiliary_canonical_weights(&package_dir, "triposg")
        .expect_err("family mismatch must fail");
    assert!(
        error.to_string().contains("does not match requested"),
        "unexpected error: {error}"
    );

    let missing = package_dir
        .parent()
        .expect("package parent")
        .join("rmbg-1.4-does-not-exist");
    let error =
        load_auxiliary_canonical_weights(&missing, "rmbg").expect_err("missing package must fail");
    assert!(
        error.to_string().contains("missing canonical artifact")
            || error.to_string().contains("canonical artifact"),
        "unexpected error: {error}"
    );
}

#[test]
fn rmbg_forward_matches_reference_sigmoid() {
    let device = Device::Cpu;
    let package = load_auxiliary_canonical_weights(rmbg_package_dir(), "rmbg")
        .expect("load RMBG auxiliary package");
    assert_eq!(package.manifest.tensor_count, 798);
    let resolved = package
        .read_resolved_config_json()
        .expect("RMBG resolved config");
    let config = RmbgConfig::from_resolved_config(&resolved).expect("RMBG config");
    let vb = unsafe { package.var_builder(DType::F32, &device) }.expect("RMBG VarBuilder");
    let model = RmbgModel::load(vb, &config).expect("load RMBG model");

    let input = baseline_tensor("rmbg.model_input", &device);
    assert_eq!(input.dims(), &[1, 3, 1024, 1024]);
    let actual = model.forward(&input).expect("RMBG forward");
    assert_eq!(actual.dims(), &[1, 1, 1024, 1024]);

    let expected = baseline_tensor("rmbg.sigmoid", &device);
    let delta = (actual - expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    println!("RMBG sigmoid max abs delta: {delta:.6e}");
    assert!(
        delta <= 2e-4,
        "RMBG sigmoid max abs delta {delta:.6e} exceeds 2e-4"
    );
}

#[test]
fn triposg_rmbg_preprocess_matches_reference() {
    use lux3d_core::SpatialSize;
    use lux3d_core::preprocess::TripoSgPreprocessStage;

    let device = Device::Cpu;
    let package = load_auxiliary_canonical_weights(rmbg_package_dir(), "rmbg")
        .expect("load RMBG auxiliary package");
    let resolved = package
        .read_resolved_config_json()
        .expect("RMBG resolved config");
    let config = RmbgConfig::from_resolved_config(&resolved).expect("RMBG config");
    let vb = unsafe { package.var_builder(DType::F32, &device) }.expect("RMBG VarBuilder");
    let model = RmbgModel::load(vb, &config).expect("load RMBG model");

    let stage = TripoSgPreprocessStage::default();
    let prepared = stage
        .prepare_inputs_from_path_with_rmbg(
            &runtime_root()
                .join("candle-3d")
                .join("test-assets")
                .join("mesh-input-rgb.png"),
            &device,
            Some(&model),
        )
        .expect("TripoSG RMBG preprocess");

    assert_eq!(prepared.target_size, SpatialSize::new(224, 224));
    let expected_u8 = baseline_tensor("rmbg.preprocessed_uint8", &device)
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let expected_u8: Vec<u8> = expected_u8
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect();
    let mismatches: Vec<(usize, u8, u8)> = expected_u8
        .iter()
        .zip(prepared.uint8_rgb.iter())
        .enumerate()
        .filter_map(|(index, (expected, actual))| {
            (expected != actual).then_some((index, *expected, *actual))
        })
        .take(16)
        .collect();
    assert!(
        expected_u8 == prepared.uint8_rgb,
        "RMBG conditioning uint8 mismatch; first (index, expected, actual): {mismatches:?}"
    );

    let expected_pv = baseline_tensor("rmbg.pixel_values", &device)
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let actual_pv = prepared
        .pixel_values
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let max_delta = expected_pv
        .iter()
        .zip(actual_pv.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    println!("RMBG pixel_values max abs delta: {max_delta:.6e}");
    assert!(max_delta <= 1e-5, "RMBG pixel_values delta {max_delta}");
}
