//! TRELLIS.2 preprocess stage tests: synthetic coverage of the reference
//! alpha path geometry (bbox, Pillow banker's-rounded square crop, opaque
//! fallback, <=1024 downscale) plus the DINOv3 input normalization contract.

use image::{Rgba, RgbaImage};
use lux3d_core::contracts::{ModelFamily, SpatialSize, TensorContract, Trellis2RawInput};
use lux3d_core::preprocess::{
    TRELLIS2_ALPHA_THRESHOLD, Trellis2Preprocess, alpha_foreground_bbox, reference_crop_box,
    square_crop_premultiplied,
};

fn rgba_image(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> RgbaImage {
    let mut image = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            image.put_pixel(x, y, Rgba(fill(x, y)));
        }
    }
    image
}

/// The alpha threshold is exactly `alpha > 0.8 * 255` on uint8 values
/// (204.00000000000003 in f64), i.e. `alpha >= 205`.
#[test]
fn alpha_threshold_matches_reference_comparison() {
    let image = rgba_image(2, 1, |x, _| [255, 0, 0, if x == 0 { 204 } else { 205 }]);
    // Only the 205 pixel passes; 204 is below the threshold.
    assert_eq!(alpha_foreground_bbox(&image), Some((1, 0, 1, 0)));
    assert_eq!(TRELLIS2_ALPHA_THRESHOLD, 205);
}

/// A fully-opaque input behaves like the reference `has_alpha == false`
/// branch without RMBG: bbox spans the whole (downscaled) image and the
/// Pillow banker's rounding of the `.5` center shrinks a 100x100 opaque
/// source to a 98x98 crop (reference arithmetic, not a port bug).
#[test]
fn opaque_input_uses_whole_image_bbox_and_reference_rounding() {
    let stage = Trellis2Preprocess::new(512).expect("stage");
    let image = rgba_image(100, 100, |x, y| [x as u8, y as u8, 32, 255]);
    let (prepared, crop) = stage
        .prepare_rgba8_with_crop(&image, false)
        .expect("prepared");
    assert!(!crop.has_alpha);
    assert_eq!(crop.bbox, (0, 0, 99, 99));
    assert_eq!(crop.crop_box, (0, 0, 98, 98));
    assert_eq!(crop.crop_size, 98);
    assert_eq!(prepared.original_size, SpatialSize::new(100, 100));
    assert_eq!(prepared.target_size, SpatialSize::new(512, 512));
    assert_eq!(prepared.family, ModelFamily::Trellis2);
    assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);
    assert_eq!(
        prepared.pixel_values,
        TensorContract::new("pixel_values", vec![1, 3, 512, 512])
    );
}

/// RGBA input: the square crop wraps the foreground bbox exactly like the
/// reference (`size = max(extent_x, extent_y)`, centered, `size // 2`).
#[test]
fn alpha_input_square_crop_geometry() {
    let stage = Trellis2Preprocess::new(512).expect("stage");
    let image = rgba_image(40, 30, |x, y| {
        if (10..25).contains(&x) && (5..24).contains(&y) {
            [200, 100, 50, 255]
        } else {
            [0, 0, 0, 0]
        }
    });
    let (prepared, crop) = stage
        .prepare_rgba8_with_crop(&image, true)
        .expect("prepared");
    assert!(crop.has_alpha);
    assert_eq!(crop.bbox, (10, 5, 24, 23));
    assert_eq!(crop.crop_box, (8, 5, 26, 23));
    assert_eq!(crop.crop_size, 18);
    assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);
}

/// Premultiplication matches the reference f32 sequence
/// `(rgb / 255) * (alpha / 255) * 255` truncated to uint8, and the square
/// crop zero-fills out-of-bounds pixels (Pillow transparent black).
#[test]
fn premultiply_and_padding_match_reference() {
    let image = rgba_image(2, 2, |x, y| [255, 0, 17 * (x + y) as u8, 128]);
    // Crop box (-1, -1, 3, 3) extends one pixel beyond each edge.
    let crop = square_crop_premultiplied(&image, (-1, -1, 3, 3), true);
    assert_eq!(crop.width(), 4);
    assert_eq!(crop.height(), 4);
    // Out-of-bounds pixels are transparent black after premultiply.
    assert_eq!(*crop.get_pixel(0, 0), image::Rgb([0, 0, 0]));
    // In-bounds premultiply follows the exact f32 rounding sequence
    // `(rgb / 255) * (alpha / 255) * 255` truncated to uint8. Output (1, 1)
    // maps to image pixel (0, 0) (crop origin (-1, -1)); output (2, 2) maps
    // to image pixel (1, 1) whose blue is 17 * 2 = 34.
    let expected_a = (f32::from(255_u8) / 255.0 * (f32::from(128_u8) / 255.0) * 255.0) as u8;
    let expected_b = (f32::from(0_u8) / 255.0 * (f32::from(128_u8) / 255.0) * 255.0) as u8;
    let expected_c = (f32::from(34_u8) / 255.0 * (f32::from(128_u8) / 255.0) * 255.0) as u8;
    assert_eq!(
        *crop.get_pixel(1, 1),
        image::Rgb([expected_a, expected_b, 0])
    );
    assert_eq!(
        *crop.get_pixel(2, 2),
        image::Rgb([expected_a, expected_b, expected_c])
    );
    // Opaque path passes RGB through unchanged.
    let opaque = square_crop_premultiplied(&image, (0, 0, 2, 2), false);
    assert_eq!(*opaque.get_pixel(0, 0), image::Rgb([255, 0, 0]));
}

/// The <= 1024 downscale is identity below the cap and truncates
/// `(w * scale, h * scale)` like Python `int()`.
#[test]
fn downscale_cap_dimensions() {
    let stage = Trellis2Preprocess::new(512).expect("stage");
    let small = rgba_image(614, 614, |_, _| [9, 9, 9, 255]);
    let (_, crop) = stage
        .prepare_rgba8_with_crop(&small, false)
        .expect("prepared");
    assert_eq!(crop.downscaled_size, (614, 614));

    // 2000x1000 -> scale 1024/2000 -> (1024, 512).
    let wide = rgba_image(2000, 1000, |_, _| [0, 0, 0, 255]);
    let (_, crop_wide) = stage
        .prepare_rgba8_with_crop(&wide, false)
        .expect("prepared");
    assert_eq!(crop_wide.downscaled_size, (1024, 512));
}

/// Fully transparent alpha is an error (the reference would run RMBG, which
/// is excluded by contract).
#[test]
fn fully_transparent_alpha_is_rejected() {
    let stage = Trellis2Preprocess::new(512).expect("stage");
    let image = rgba_image(8, 8, |_, _| [255, 255, 255, 0]);
    assert!(stage.prepare_rgba8_with_crop(&image, true).is_err());
}

/// Contract-style `Trellis2RawInput` path quantizes f32 [0, 1] buffers and
/// rejects inconsistent inputs; the stage only accepts 512/1024.
#[test]
fn raw_input_contract_path() {
    let stage = Trellis2Preprocess::new(512).expect("stage");
    let raw = Trellis2RawInput {
        original_size: SpatialSize::new(3, 3),
        rgba: Some(vec![[1.0, 0.5, 0.25, 1.0]; 9]),
        rgb: None,
    };
    let prepared = stage.preprocess(raw).expect("prepared");
    assert_eq!(prepared.target_size, SpatialSize::new(512, 512));
    assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);

    let mismatched = Trellis2RawInput {
        original_size: SpatialSize::new(3, 3),
        rgba: Some(vec![[0.0, 0.0, 0.0, 1.0]; 4]),
        rgb: None,
    };
    assert!(stage.preprocess(mismatched).is_err());
    let empty = Trellis2RawInput {
        original_size: SpatialSize::new(3, 3),
        rgba: None,
        rgb: None,
    };
    assert!(stage.preprocess(empty).is_err());
    assert!(Trellis2Preprocess::new(256).is_err());
    assert!(Trellis2Preprocess::new(1024).is_ok());
}

/// Reference crop-box arithmetic, including the Pillow banker's rounding
/// corner cases.
#[test]
fn crop_box_reference_arithmetic() {
    // .5 centers on both axes with odd size shrink to the even 2*half side.
    assert_eq!(reference_crop_box((0, 0, 99, 99)), (0, 0, 98, 98));
    // Integer centers preserve the exact size.
    assert_eq!(reference_crop_box((0, 0, 98, 98)), (0, 0, 98, 98));
    assert_eq!(reference_crop_box((2, 5, 40, 50)), (-1, 6, 43, 50));
    assert_eq!(reference_crop_box((1, 1, 6, 6)), (2, 2, 6, 6));
    assert_eq!(reference_crop_box((4, 7, 4, 7)), (4, 7, 4, 7));
}

/// DINOv3 input normalization contract: `[1, 3, res, res]` CHW with
/// `(v / 255 - mean) / std`.
#[test]
fn normalized_pixel_values_chw_layout() {
    use lux3d_core::preprocess::{
        TRELLIS2_IMAGE_NET_MEAN, TRELLIS2_IMAGE_NET_STD, normalized_pixel_values_from_u8,
    };
    let mut pixels = vec![0_u8; 4 * 4 * 3];
    pixels[0] = 255; // R of pixel (0, 0) in the HWC buffer
    pixels[1] = 128; // G of pixel (0, 0) in the HWC buffer
    let tensor =
        normalized_pixel_values_from_u8(&pixels, 4, &candle_core::Device::Cpu).expect("tensor");
    assert_eq!(tensor.dims(), &[1, 3, 4, 4]);
    let values = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .expect("values");
    let expected_r = (1.0_f32 - TRELLIS2_IMAGE_NET_MEAN[0]) / TRELLIS2_IMAGE_NET_STD[0];
    assert!((values[0] - expected_r).abs() < 1e-6);
    let expected_g =
        (f32::from(128_u8) / 255.0 - TRELLIS2_IMAGE_NET_MEAN[1]) / TRELLIS2_IMAGE_NET_STD[1];
    assert!((values[16] - expected_g).abs() < 1e-6);
}
