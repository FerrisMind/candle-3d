//! TripoSG preprocess parity tests against the python-baseline smoke sample
//! plus synthetic coverage of the reference alpha path.

use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use lux3d_core::SpatialSize;
use lux3d_core::preprocess::TripoSgPreprocessStage;
use lux3d_core::test_support::runtime_root;

fn repo_root() -> PathBuf {
    runtime_root().join("candle-3d")
}

fn smoke_baseline_path() -> PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposg")
        .join("smoke")
        .join("artifacts.safetensors")
}

fn load_baseline_tensor(name: &str) -> Tensor {
    let path = smoke_baseline_path();
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&path).expect("smoke baseline safetensors")
    }
    .load(name, &Device::Cpu)
    .expect("baseline tensor")
}

fn chw_index(channel: usize, y: usize, x: usize) -> usize {
    channel * 224 * 224 + y * 224 + x
}

/// Golden parity on CPU: `uint8_rgb` must equal
/// `triposg.preprocessed_uint8` exactly (each stored f32 rounded/clamped to
/// u8), and `pixel_values` max abs delta must be <= 1e-5.
#[test]
fn triposg_preprocess_matches_triposg_smoke_baseline() {
    let stage = TripoSgPreprocessStage::default();
    let prepared = stage
        .prepare_inputs_from_path(
            &repo_root().join("test-assets").join("mesh-input.png"),
            &Device::Cpu,
        )
        .expect("triposg prepared inputs");

    assert_eq!(prepared.target_size, SpatialSize::new(224, 224));
    assert_eq!(prepared.uint8_rgb.len(), 3 * 224 * 224);

    let expected_u8_f32 = load_baseline_tensor("triposg.preprocessed_uint8")
        .to_dtype(DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert_eq!(expected_u8_f32.len(), prepared.uint8_rgb.len());
    let expected_u8: Vec<u8> = expected_u8_f32
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
        .take(8)
        .collect();
    assert!(
        expected_u8 == prepared.uint8_rgb,
        "uint8_rgb must match the baseline exactly; first mismatches (index, expected, actual): \
         {mismatches:?}"
    );
    println!(
        "uint8_rgb parity: exact match on {} values",
        expected_u8.len()
    );

    let expected_pv = load_baseline_tensor("triposg.pixel_values")
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
    assert_eq!(expected_pv.len(), actual_pv.len());
    let delta = expected_pv
        .iter()
        .zip(actual_pv.iter())
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    println!("pixel_values max abs delta: {delta:.3e}");
    assert!(delta <= 1e-5, "pixel_values delta too large: {delta}");
}

/// Synthetic RGBA: opaque red left half, transparent right half. The valid
/// alpha branch crops to the red bbox, square-pads it (pad_lr=1, pad_tb=0),
/// and the transparent padding pixels composite to white.
#[test]
fn triposg_synthetic_rgba_white_composite_square_pad_and_chw() {
    let mut rgba = RgbaImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            if x < 2 {
                rgba.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            } else {
                rgba.put_pixel(x, y, Rgba([0, 0, 0, 0]));
            }
        }
    }

    let stage = TripoSgPreprocessStage::default();
    // Reference square-pad arithmetic for the 2x4 bbox: w < h, so
    // pad_tb = int(4 * 0.1) = 0 and pad_lr = int(0 + (4 - 2) / 2) = 1.
    assert_eq!(stage.square_pad_extents(2, 4), (1, 0));
    // Golden-sample bbox 454x250: pad_lr = int(45.4) = 45,
    // pad_tb = int(45 + 102) = 147 -> 544x544 square.
    assert_eq!(stage.square_pad_extents(454, 250), (45, 147));

    let prepared = stage
        .prepare_dynamic_image(&DynamicImage::ImageRgba8(rgba), &Device::Cpu)
        .expect("prepared synthetic rgba");
    assert_eq!(prepared.original_size, SpatialSize::new(4, 4));
    assert_eq!(prepared.target_size, SpatialSize::new(224, 224));
    assert_eq!(prepared.uint8_rgb.len(), 3 * 224 * 224);

    // The 4x4 padded square upscales 64x; source pixel centers land on exact
    // output pixels. Padded source (0, 0) is white (transparent composited
    // over white); padded source (1, 1) is opaque red.
    let white = chw_index(0, 15, 15);
    assert_eq!(
        [
            prepared.uint8_rgb[white],
            prepared.uint8_rgb[chw_index(1, 15, 15)],
            prepared.uint8_rgb[chw_index(2, 15, 15)]
        ],
        [255, 255, 255],
        "transparent pixels must composite to white"
    );
    let red = chw_index(0, 112, 112);
    assert_eq!(
        [
            prepared.uint8_rgb[red],
            prepared.uint8_rgb[chw_index(1, 112, 112)],
            prepared.uint8_rgb[chw_index(2, 112, 112)]
        ],
        [255, 0, 0],
        "opaque red pixels must survive unchanged"
    );

    // CHW ordering: channel planes are contiguous. R stays 255 everywhere
    // (red and white share R=255); G and B planes are identical (red and
    // white both have G == B) and span the full 0..255 range.
    let plane = 224 * 224;
    assert!(
        prepared.uint8_rgb[0..plane]
            .iter()
            .all(|value| *value == 255),
        "R plane must be all 255"
    );
    assert_eq!(
        prepared.uint8_rgb[plane..2 * plane],
        prepared.uint8_rgb[2 * plane..3 * plane]
    );
    assert_eq!(prepared.uint8_rgb[plane..2 * plane].iter().min(), Some(&0));
    assert_eq!(
        prepared.uint8_rgb[plane..2 * plane].iter().max(),
        Some(&255)
    );

    // pixel_values must be the normalized uint8 planes.
    let mean = [0.485_f32, 0.456, 0.406];
    let std = [0.229_f32, 0.224, 0.225];
    let values = prepared
        .pixel_values
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    for channel in 0..3 {
        for index in 0..plane {
            let expected = (f32::from(prepared.uint8_rgb[channel * plane + index]) / 255.0
                - mean[channel])
                / std[channel];
            assert!(
                (values[channel * plane + index] - expected).abs() <= 1e-6,
                "pixel_values must match (v/255 - mean)/std"
            );
        }
    }
}

/// RGB input without an alpha channel must fail unless BriaRMBG is supplied:
/// the reference `load_image` always constructs and calls RMBG for this branch.
#[test]
fn triposg_rgb_input_without_rmbg_is_rejected() {
    let rgb = RgbImage::from_pixel(4, 4, Rgb([255, 0, 0]));
    let stage = TripoSgPreprocessStage::default();
    let error = stage
        .prepare_dynamic_image(&DynamicImage::ImageRgb8(rgb), &Device::Cpu)
        .expect_err("RGB input requires BriaRMBG");
    assert!(
        error.to_string().contains("BriaRMBG"),
        "unexpected error: {error}"
    );

    // The contract-only trait view does not run the network; it still reports
    // the opaque composite metadata for a single RGB pixel.
    use lux3d_core::contracts::{PreprocessStage, TripoSgRawInput};
    let prepared_pixel = stage
        .preprocess(TripoSgRawInput {
            original_size: SpatialSize::new(1, 1),
            rgb: Some([1.0, 0.0, 0.0]),
            rgba: None,
        })
        .expect("single-pixel rgb prepared");
    assert_eq!(prepared_pixel.uint8_rgb, vec![255, 0, 0]);
}

/// RGBA whose alpha histogram fails the reference `is_valid_alpha` check (all
/// pixels at 128 fall in neither the 0 nor the 255 bin) takes the RMBG branch;
/// without a model it must be rejected rather than silently treated as opaque.
#[test]
fn triposg_rgba_with_invalid_alpha_requires_rmbg() {
    let rgba = RgbaImage::from_pixel(4, 4, Rgba([255, 0, 0, 128]));
    let stage = TripoSgPreprocessStage::default();
    let error = stage
        .prepare_dynamic_image(&DynamicImage::ImageRgba8(rgba), &Device::Cpu)
        .expect_err("invalid alpha requires BriaRMBG");
    assert!(
        error.to_string().contains("BriaRMBG"),
        "unexpected error: {error}"
    );

    // The contract-only trait path still composites a single opaque pixel.
    use lux3d_core::contracts::{PreprocessStage, TripoSgRawInput};
    let prepared_pixel = stage
        .preprocess(TripoSgRawInput {
            original_size: SpatialSize::new(1, 1),
            rgb: None,
            rgba: Some([1.0, 0.0, 0.0, 1.0]),
        })
        .expect("single-pixel rgba prepared");
    assert_eq!(prepared_pixel.uint8_rgb, vec![255, 0, 0]);
}

/// Large-input path: 2001x1000 exceeds the 2000 px cap, so the pipeline
/// downscales first (area average, alpha included) and computes alpha
/// validity plus the foreground bbox on the downscaled image. The transparent
/// left half is excluded by the bbox, leaving a red-dominant crop.
#[test]
fn triposg_large_input_downscale_and_alpha_bbox() {
    let mut rgba = RgbaImage::new(2001, 1000);
    for y in 0..1000 {
        for x in 0..2001 {
            if x >= 1001 {
                rgba.put_pixel(x, y, Rgba([255, 0, 0, 255]));
            }
        }
    }

    let stage = TripoSgPreprocessStage::default();
    // Reference downscale dimensions: scale = 2000/2001, new size
    // (int(2001 * scale), int(1000 * scale)) = (2000, 999).
    assert_eq!(stage.capped_input_dimensions(2001, 1000), (2000, 999));
    assert_eq!(stage.capped_input_dimensions(614, 614), (614, 614));

    let prepared = stage
        .prepare_dynamic_image(&DynamicImage::ImageRgba8(rgba), &Device::Cpu)
        .expect("prepared large rgba");
    assert_eq!(prepared.original_size, SpatialSize::new(2001, 1000));
    assert_eq!(prepared.target_size, SpatialSize::new(224, 224));
    assert_eq!(prepared.uint8_rgb.len(), 3 * 224 * 224);

    let plane = 224 * 224;
    let mean = |values: &[u8]| -> f32 {
        values.iter().map(|value| f32::from(*value)).sum::<f32>() / values.len() as f32
    };
    let mean_r = mean(&prepared.uint8_rgb[0..plane]);
    let mean_g = mean(&prepared.uint8_rgb[plane..2 * plane]);
    println!("large-input mean R: {mean_r:.2}, mean G: {mean_g:.2}");
    // Valid-alpha branch: the bbox excludes the transparent left half, so the
    // crop is red-dominant. Had the alpha been ignored (opaque treatment over
    // the full image), half the crop would be black and mean_r would be ~127.
    assert!(mean_r > 250.0, "crop must be red-dominant, mean R {mean_r}");
    assert!(
        mean_g < 60.0,
        "crop must exclude the transparent half, mean G {mean_g}"
    );
    // Red (255,0,0) and white both satisfy G == B; the per-channel resize
    // keeps the planes identical elementwise.
    assert_eq!(
        prepared.uint8_rgb[plane..2 * plane],
        prepared.uint8_rgb[2 * plane..3 * plane]
    );
}
