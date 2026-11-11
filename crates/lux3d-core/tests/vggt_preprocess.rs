use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{DType, Device};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use lux3d_core::contracts::{
    ModelFamily, Pi3InputSource, PreprocessStage, RgbRange, SpatialSize, VggtRawInput,
};
use lux3d_core::preprocess::VggtPreprocessStage;
use lux3d_core::test_support::runtime_root;

fn temp_dir(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path.push(format!(
        "lux3d-vggt-preprocess-{label}-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn max_abs_delta(lhs: &candle_core::Tensor, rhs: &candle_core::Tensor) -> f32 {
    lhs.to_dtype(DType::F32)
        .expect("lhs dtype")
        .sub(rhs)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar")
}

#[test]
fn vggt_preprocess_matches_pi3_frames_golden_baseline() {
    let device = Device::Cpu;
    let stage = VggtPreprocessStage::default();
    let source = runtime_root()
        .join("candle-3d")
        .join("test-assets")
        .join("pi3-frames");
    let prepared = stage
        .prepare_inputs_from_path(&source, None, &device)
        .expect("vggt prepared inputs");

    let baseline_path = runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("vggt")
        .join("pi3-frames-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("vggt baseline safetensors")
    };
    let expected_rgb = baseline
        .load("vggt.loader.rgb_frames", &Device::Cpu)
        .expect("baseline rgb frames");
    let expected_normalized = baseline
        .load("vggt.model.normalized_frames", &Device::Cpu)
        .expect("baseline normalized frames");

    assert_eq!(prepared.interval, 1);
    assert_eq!(prepared.original_size, SpatialSize::new(512, 512));
    assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
    assert_eq!(prepared.rgb_frames.dims(), expected_rgb.dims());
    assert_eq!(
        prepared.normalized_frames.dims(),
        expected_normalized.dims()
    );

    let rgb_delta = max_abs_delta(&prepared.rgb_frames, &expected_rgb);
    let normalized_delta = max_abs_delta(&prepared.normalized_frames, &expected_normalized);
    eprintln!("vggt preprocess rgb max-abs delta: {rgb_delta:e}");
    eprintln!("vggt preprocess normalized max-abs delta: {normalized_delta:e}");

    assert!(
        rgb_delta <= 1e-5,
        "vggt preprocess rgb delta too large: {rgb_delta:e} (expected <= 1e-5)"
    );
    assert!(
        normalized_delta <= 1e-5,
        "vggt preprocess normalized delta too large: {normalized_delta:e} (expected <= 1e-5)"
    );
}

#[test]
fn target_size_for_matches_reference_pre_crop_formula() {
    let stage = VggtPreprocessStage::default();

    // 512 * (518 / 512) / 14 == 37 exactly.
    assert_eq!(
        stage
            .target_size_for(SpatialSize::new(512, 512), 518)
            .expect("square target"),
        SpatialSize::new(518, 518)
    );

    // 480 * (518 / 640) / 14 == 27.75 -> 28 * 14 == 392.
    assert_eq!(
        stage
            .target_size_for(SpatialSize::new(640, 480), 518)
            .expect("landscape target"),
        SpatialSize::new(518, 392)
    );

    // 640 * (518 / 480) / 14 == 49.333... -> 49 * 14 == 686.
    assert_eq!(
        stage
            .target_size_for(SpatialSize::new(480, 640), 518)
            .expect("portrait target"),
        SpatialSize::new(518, 686)
    );

    // 4 * (518 / 8) / 14 == 18.5 -> ties-to-even rounds to 18 * 14 == 252.
    assert_eq!(
        stage
            .target_size_for(SpatialSize::new(8, 4), 518)
            .expect("ties-to-even target"),
        SpatialSize::new(518, 252)
    );

    assert!(
        stage
            .target_size_for(SpatialSize::new(0, 512), 518)
            .is_err()
    );
    assert!(
        stage
            .target_size_for(SpatialSize::new(512, 512), 0)
            .is_err()
    );
}

#[test]
fn preprocess_stage_fills_vggt_prepared_batch() {
    let stage = VggtPreprocessStage::default();

    let batch = stage
        .preprocess(VggtRawInput::directory(5, 640, 480, 2))
        .expect("vggt prepared batch");
    assert_eq!(batch.family, ModelFamily::Vggt);
    assert_eq!(batch.source, Pi3InputSource::Directory);
    assert_eq!(batch.sampled_frames, 3);
    assert_eq!(batch.original_size, SpatialSize::new(640, 480));
    assert_eq!(batch.target_size, SpatialSize::new(518, 392));
    assert_eq!(batch.interval, 2);
    assert_eq!(batch.pixel_range, RgbRange::ZeroToOne);

    assert!(
        stage
            .preprocess(VggtRawInput::directory(5, 512, 512, 0))
            .is_err()
    );
}

fn normalize_rgb(rgb: [f32; 3]) -> [f32; 3] {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    [
        (rgb[0] - MEAN[0]) / STD[0],
        (rgb[1] - MEAN[1]) / STD[1],
        (rgb[2] - MEAN[2]) / STD[2],
    ]
}

fn assert_uniform_frame(frame: &candle_core::Tensor, expected: [f32; 3]) {
    for (channel, expected) in expected.into_iter().enumerate() {
        let plane = frame.get(channel).expect("channel");
        let values = plane
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(
            values.iter().all(|value| (*value - expected).abs() <= 1e-6),
            "normalized channel {channel} expected {expected}, got min {} max {}",
            values.iter().copied().fold(f32::INFINITY, f32::min),
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );
    }
}

fn assert_row_is(frame: &candle_core::Tensor, y: usize, expected: f32) {
    let row = frame
        .get(y)
        .expect("row")
        .to_vec1::<f32>()
        .expect("row vec");
    assert!(
        row.iter().all(|value| (*value - expected).abs() <= 1e-6),
        "row {y} expected {expected}, got min {} max {}",
        row.iter().copied().fold(f32::INFINITY, f32::min),
        row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
}

#[test]
fn preprocesses_rgba_over_white_and_center_pads_mismatched_shapes() {
    let dir = temp_dir("rgba-pad");

    // 2x2 fully translucent black: composites to white beforehand, so the
    // uniform frame stays white after the resize.
    let mut rgba = RgbaImage::new(2, 2);
    for pixel in rgba.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 128]);
    }
    rgba.save(dir.join("000.png")).expect("save rgba frame");

    // 2x2 uniform RGB; bicubic resize of a constant image stays constant.
    let mut rgb = RgbImage::new(2, 2);
    for pixel in rgb.pixels_mut() {
        *pixel = Rgb([10, 20, 30]);
    }
    rgb.save(dir.join("001.png")).expect("save rgb frame");

    // 4x2 uniform black: 518x252 after crop-mode sizing, then white-padded.
    let wide = RgbImage::from_pixel(4, 2, Rgb([0, 0, 0]));
    wide.save(dir.join("002.png")).expect("save wide frame");

    let prepared = VggtPreprocessStage::default()
        .prepare_inputs_from_path(&dir, None, &Device::Cpu)
        .expect("prepared rgba/pad inputs");
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(prepared.interval, 1);
    assert_eq!(prepared.original_size, SpatialSize::new(2, 2));
    assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
    assert_eq!(prepared.rgb_frames.dims(), &[3, 3, 518, 518]);
    assert_eq!(prepared.normalized_frames.dims(), &[1, 3, 3, 518, 518]);

    // Frame 0: RGBA(0, 0, 0, 128) over white -> 127 for every channel.
    let frame0 = prepared.rgb_frames.get(0).expect("frame 0");
    let expected_gray = 127.0_f32 / 255.0;
    let values = frame0
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    assert!(
        values
            .iter()
            .all(|value| (*value - expected_gray).abs() <= 1e-6),
        "transparent frame did not composite to 127"
    );
    let normalized0 = prepared
        .normalized_frames
        .get(0)
        .expect("normalized batch")
        .get(0)
        .expect("normalized frame 0");
    assert_uniform_frame(
        &normalized0,
        normalize_rgb([expected_gray, expected_gray, expected_gray]),
    );

    // Frame 1: uniform RGB(10, 20, 30) survives the resize.
    let frame1 = prepared.rgb_frames.get(1).expect("frame 1");
    for (channel, expected) in [10.0_f32, 20.0, 30.0].into_iter().enumerate() {
        let plane = frame1.get(channel).expect("channel");
        let values = plane
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("vec");
        assert!(
            values
                .iter()
                .all(|value| (*value - expected / 255.0).abs() <= 1e-6),
            "rgb frame channel {channel} changed under constant resize"
        );
    }
    let normalized1 = prepared
        .normalized_frames
        .get(0)
        .expect("normalized batch")
        .get(1)
        .expect("normalized frame 1");
    assert_uniform_frame(
        &normalized1,
        normalize_rgb([10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]),
    );

    // Frame 2: 518x252 content center-padded with 133 white rows top/bottom.
    let frame2 = prepared.rgb_frames.get(2).expect("frame 2");
    let channel0 = frame2.get(0).expect("channel 0");
    for y in [0usize, 132] {
        assert_row_is(&channel0, y, 1.0);
    }
    for y in [133usize, 384] {
        assert_row_is(&channel0, y, 0.0);
    }
    for y in [385usize, 517] {
        assert_row_is(&channel0, y, 1.0);
    }

    let normalized2 = prepared
        .normalized_frames
        .get(0)
        .expect("normalized batch")
        .get(2)
        .expect("normalized frame 2");
    let white = normalize_rgb([1.0, 1.0, 1.0]);
    let black = normalize_rgb([0.0, 0.0, 0.0]);
    for channel in 0..3 {
        let plane = normalized2.get(channel).expect("normalized channel");
        for y in [0usize, 132, 385, 517] {
            assert_row_is(&plane, y, white[channel]);
        }
        for y in [133usize, 384] {
            assert_row_is(&plane, y, black[channel]);
        }
    }
}

#[test]
fn portrait_frames_are_center_cropped_to_the_max_size() {
    let dir = temp_dir("portrait-crop");

    // 64x128 -> scaled height 1036, center-cropped back to 518.
    let portrait = RgbImage::from_pixel(64, 128, Rgb([200, 40, 10]));
    portrait
        .save(dir.join("000.png"))
        .expect("save portrait frame");

    let prepared = VggtPreprocessStage::default()
        .prepare_inputs_from_path(&dir, None, &Device::Cpu)
        .expect("prepared portrait inputs");
    let _ = fs::remove_dir_all(&dir);

    assert_eq!(prepared.original_size, SpatialSize::new(64, 128));
    assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
    assert_eq!(prepared.rgb_frames.dims(), &[1, 3, 518, 518]);
    assert_eq!(prepared.normalized_frames.dims(), &[1, 1, 3, 518, 518]);
}
