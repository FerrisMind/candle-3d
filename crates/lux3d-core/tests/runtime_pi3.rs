use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;

use candle_core::{DType, Device};
use lux3d_core::ModelFamily;
use lux3d_core::runtime::Pi3Pipeline;
use lux3d_core::test_support::{GpuTestLock, model_asset_options, runtime_root};
use safetensors::SafeTensors;

fn repo_root() -> PathBuf {
    runtime_root()
}

fn accel_device() -> Device {
    Device::new_cuda(0).expect("runtime_pi3 tests require CUDA")
}

#[test]
fn pi3_preprocess_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            None,
            &accel_device(),
        )
        .expect("pi3 prepared inputs");

    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let rgb_frames = baseline
        .load("pi3.loader.rgb_frames", &accel_device())
        .expect("rgb frames");
    let normalized = baseline
        .load("pi3.model.normalized_frames", &accel_device())
        .expect("normalized frames");

    assert_eq!(prepared.rgb_frames.dims(), rgb_frames.dims());
    assert_eq!(prepared.normalized_frames.dims(), normalized.dims());

    let rgb_delta = prepared
        .rgb_frames
        .to_dtype(DType::F32)
        .expect("rgb dtype")
        .sub(&rgb_frames)
        .expect("rgb delta")
        .abs()
        .expect("rgb abs")
        .flatten_all()
        .expect("rgb flatten")
        .max(0)
        .expect("rgb max")
        .to_scalar::<f32>()
        .expect("rgb scalar");

    let normalized_delta = prepared
        .normalized_frames
        .to_dtype(DType::F32)
        .expect("norm dtype")
        .sub(&normalized)
        .expect("norm delta")
        .abs()
        .expect("norm abs")
        .flatten_all()
        .expect("norm flatten")
        .max(0)
        .expect("norm max")
        .to_scalar::<f32>()
        .expect("norm scalar");

    assert!(rgb_delta <= 1e-5, "rgb delta too large: {rgb_delta}");
    assert!(
        normalized_delta <= 1e-5,
        "normalized delta too large: {normalized_delta}"
    );
}

#[test]
fn pi3_video_preprocess_matches_skating_smoke_summary() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            &accel_device(),
        )
        .expect("pi3 prepared video inputs");

    assert_eq!(10, prepared.interval);
    assert_eq!(&[11, 3, 364, 700], prepared.rgb_frames.dims());
    assert_eq!(&[1, 11, 3, 364, 700], prepared.normalized_frames.dims());
}

#[test]
fn pi3_prepared_tokens_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            None,
            &accel_device(),
        )
        .expect("pi3 prepared inputs");

    let prepared_tokens = pipeline
        .prepare_encoder_tokens(&prepared.normalized_frames)
        .expect("pi3 prepared tokens");

    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let expected = baseline
        .load("pi3.encoder.prepared_tokens", &accel_device())
        .expect("prepared tokens");

    assert_eq!(prepared_tokens.dims(), expected.dims());

    let delta = prepared_tokens
        .to_dtype(DType::F32)
        .expect("prepared dtype")
        .sub(&expected)
        .expect("prepared delta")
        .abs()
        .expect("prepared abs")
        .flatten_all()
        .expect("prepared flatten")
        .max(0)
        .expect("prepared max")
        .to_scalar::<f32>()
        .expect("prepared scalar");

    assert!(delta <= 1e-5, "prepared token delta too large: {delta}");
}

#[test]
fn pi3_encoder_patch_tokens_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            None,
            &accel_device(),
        )
        .expect("pi3 prepared inputs");

    let patch_tokens = pipeline
        .encode_patch_tokens(&prepared.normalized_frames)
        .expect("pi3 patch tokens");

    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let expected = baseline
        .load("pi3.encoder.patch_tokens", &accel_device())
        .expect("encoder patch tokens");

    assert_eq!(patch_tokens.dims(), expected.dims());

    let delta = patch_tokens
        .to_dtype(DType::F32)
        .expect("patch dtype")
        .sub(&expected)
        .expect("patch delta")
        .abs()
        .expect("patch abs")
        .flatten_all()
        .expect("patch flatten")
        .max(0)
        .expect("patch max")
        .to_scalar::<f32>()
        .expect("patch scalar");

    assert!(delta <= 1e-3, "patch token delta too large: {delta}");
}

#[test]
fn pi3_decoder_hidden_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let expected = baseline
        .load("pi3.decoder.hidden", &accel_device())
        .expect("decoder hidden");
    let patch_tokens = baseline
        .load("pi3.encoder.patch_tokens", &accel_device())
        .expect("patch tokens");
    let decoded = pipeline
        .decode_hidden(&patch_tokens, 8, 378, 672)
        .expect("pi3 decode hidden");

    assert_eq!(decoded.dims(), expected.dims());

    let delta = decoded
        .to_dtype(DType::F32)
        .expect("decoder dtype")
        .sub(&expected)
        .expect("decoder delta")
        .abs()
        .expect("decoder abs")
        .flatten_all()
        .expect("decoder flatten")
        .max(0)
        .expect("decoder max")
        .to_scalar::<f32>()
        .expect("decoder scalar");

    assert!(delta <= 1e-3, "decoder hidden delta too large: {delta}");
}

#[test]
fn pi3_decoder_positions_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            None,
            &accel_device(),
        )
        .expect("pi3 prepared inputs");
    let patch_tokens = pipeline
        .encode_patch_tokens(&prepared.normalized_frames)
        .expect("pi3 patch tokens");
    let positions = pipeline
        .decode_positions_only(&patch_tokens, 8, 378, 672)
        .expect("pi3 decode positions");

    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let expected = baseline
        .load("pi3.decoder.positions", &accel_device())
        .expect("decoder positions");

    assert_eq!(positions.dims(), expected.dims());

    let delta = positions
        .to_dtype(DType::F32)
        .expect("positions dtype")
        .sub(&expected)
        .expect("positions delta")
        .abs()
        .expect("positions abs")
        .flatten_all()
        .expect("positions flatten")
        .max(0)
        .expect("positions max")
        .to_scalar::<f32>()
        .expect("positions scalar");

    assert!(delta <= 1e-6, "decoder positions delta too large: {delta}");
}

#[test]
fn pi3_point_decoder_hidden_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.decoder.hidden", &accel_device())
        .expect("decoder hidden");
    let positions = baseline
        .load("pi3.decoder.positions", &accel_device())
        .expect("decoder positions");
    let expected = baseline
        .load("pi3.point_decoder.hidden", &accel_device())
        .expect("point decoder hidden");

    let actual = pipeline
        .point_decoder_hidden(&hidden, &positions)
        .expect("point decoder hidden");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("point hidden dtype")
        .sub(&expected)
        .expect("point hidden delta")
        .abs()
        .expect("point hidden abs")
        .flatten_all()
        .expect("point hidden flatten")
        .max(0)
        .expect("point hidden max")
        .to_scalar::<f32>()
        .expect("point hidden scalar");

    assert!(
        delta <= 1e-3,
        "point decoder hidden delta too large: {delta}"
    );
}

#[test]
fn pi3_conf_decoder_hidden_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.decoder.hidden", &accel_device())
        .expect("decoder hidden");
    let positions = baseline
        .load("pi3.decoder.positions", &accel_device())
        .expect("decoder positions");
    let expected = baseline
        .load("pi3.conf_decoder.hidden", &accel_device())
        .expect("conf decoder hidden");

    let actual = pipeline
        .conf_decoder_hidden(&hidden, &positions)
        .expect("conf decoder hidden");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("conf hidden dtype")
        .sub(&expected)
        .expect("conf hidden delta")
        .abs()
        .expect("conf hidden abs")
        .flatten_all()
        .expect("conf hidden flatten")
        .max(0)
        .expect("conf hidden max")
        .to_scalar::<f32>()
        .expect("conf hidden scalar");

    assert!(
        delta <= 1e-3,
        "conf decoder hidden delta too large: {delta}"
    );
}

#[test]
fn pi3_camera_decoder_hidden_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.decoder.hidden", &accel_device())
        .expect("decoder hidden");
    let positions = baseline
        .load("pi3.decoder.positions", &accel_device())
        .expect("decoder positions");
    let expected = baseline
        .load("pi3.camera_decoder.hidden", &accel_device())
        .expect("camera decoder hidden");

    let actual = pipeline
        .camera_decoder_hidden(&hidden, &positions)
        .expect("camera decoder hidden");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("camera hidden dtype")
        .sub(&expected)
        .expect("camera hidden delta")
        .abs()
        .expect("camera hidden abs")
        .flatten_all()
        .expect("camera hidden flatten")
        .max(0)
        .expect("camera hidden max")
        .to_scalar::<f32>()
        .expect("camera hidden scalar");

    assert!(
        delta <= 1e-3,
        "camera decoder hidden delta too large: {delta}"
    );
}

#[test]
fn pi3_local_points_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.point_decoder.hidden", &accel_device())
        .expect("point decoder hidden");
    let expected = baseline
        .load("pi3.local_points", &accel_device())
        .expect("local points");

    let point_head = pipeline
        .point_head_output(&hidden, 378, 672)
        .expect("point head output");
    let actual = pipeline
        .local_points_from_head_output(&point_head)
        .expect("local points")
        .reshape((1, 8, 378, 672, 3))
        .expect("local points reshape");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("local points dtype")
        .sub(&expected)
        .expect("local points delta")
        .abs()
        .expect("local points abs")
        .flatten_all()
        .expect("local points flatten")
        .max(0)
        .expect("local points max")
        .to_scalar::<f32>()
        .expect("local points scalar");

    assert!(delta <= 1e-3, "local points delta too large: {delta}");
}

#[test]
fn pi3_confidence_logits_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.conf_decoder.hidden", &accel_device())
        .expect("conf decoder hidden");
    let expected = baseline
        .load("pi3.confidence_logits", &accel_device())
        .expect("confidence logits");

    let actual = pipeline
        .conf_head_output(&hidden, 378, 672)
        .expect("conf head output")
        .reshape((1, 8, 378, 672, 1))
        .expect("confidence reshape");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("confidence dtype")
        .sub(&expected)
        .expect("confidence delta")
        .abs()
        .expect("confidence abs")
        .flatten_all()
        .expect("confidence flatten")
        .max(0)
        .expect("confidence max")
        .to_scalar::<f32>()
        .expect("confidence scalar");

    assert!(delta <= 1e-3, "confidence logits delta too large: {delta}");
}

#[test]
fn pi3_camera_poses_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let hidden = baseline
        .load("pi3.camera_decoder.hidden", &accel_device())
        .expect("camera decoder hidden");
    let expected = baseline
        .load("pi3.camera_poses", &accel_device())
        .expect("camera poses");

    let actual = pipeline
        .camera_poses_from_hidden(&hidden, 27, 48)
        .expect("camera poses")
        .reshape((1, 8, 4, 4))
        .expect("camera poses reshape");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("camera poses dtype")
        .sub(&expected)
        .expect("camera poses delta")
        .abs()
        .expect("camera poses abs")
        .flatten_all()
        .expect("camera poses flatten")
        .max(0)
        .expect("camera poses max")
        .to_scalar::<f32>()
        .expect("camera poses scalar");

    assert!(delta <= 1e-3, "camera poses delta too large: {delta}");
}

#[test]
fn pi3_points_match_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let local_points = baseline
        .load("pi3.local_points", &accel_device())
        .expect("local points");
    let camera_poses = baseline
        .load("pi3.camera_poses", &accel_device())
        .expect("camera poses");
    let expected = baseline
        .load("pi3.points", &accel_device())
        .expect("points");

    let actual = pipeline
        .world_points(&local_points, &camera_poses)
        .expect("world points");

    let delta = actual
        .to_dtype(DType::F32)
        .expect("points dtype")
        .sub(&expected)
        .expect("points delta")
        .abs()
        .expect("points abs")
        .flatten_all()
        .expect("points flatten")
        .max(0)
        .expect("points max")
        .to_scalar::<f32>()
        .expect("points scalar");

    assert!(delta <= 1e-3, "points delta too large: {delta}");
}

#[test]
fn pi3_export_mask_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let local_points = baseline
        .load("pi3.local_points", &accel_device())
        .expect("local points");
    let confidence_logits = baseline
        .load("pi3.confidence_logits", &accel_device())
        .expect("confidence logits");
    let expected = read_bool_safetensor(&baseline_path, "pi3.export_mask");

    let actual = pipeline
        .export_mask(&local_points, &confidence_logits)
        .expect("export mask");

    assert_eq!(actual.dims(), &[1, 8, 378, 672]);
    assert_eq!(
        actual
            .to_dtype(DType::U8)
            .expect("actual u8")
            .flatten_all()
            .expect("actual flatten")
            .to_vec1::<u8>()
            .expect("actual vec"),
        expected
    );
}

#[test]
fn pi3_non_edge_mask_matches_house_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("artifacts.safetensors");
    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    };
    let local_points = baseline
        .load("pi3.local_points", &accel_device())
        .expect("local points");
    let expected = read_bool_safetensor(&baseline_path, "pi3.non_edge_mask");

    let actual = pipeline
        .non_edge_mask(&local_points)
        .expect("non-edge mask");

    assert_eq!(actual.dims(), &[1, 8, 378, 672]);
    assert_eq!(
        actual
            .to_dtype(DType::U8)
            .expect("actual u8")
            .flatten_all()
            .expect("actual flatten")
            .to_vec1::<u8>()
            .expect("actual vec"),
        expected
    );
}

#[test]
fn pi3_export_ply_end_to_end_emits_vendor_aligned_header() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
    let output = pipeline
        .infer_from_path_with_interval(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            None,
            &accel_device(),
        )
        .expect("pi3 inference");

    let output_path =
        std::env::temp_dir().join(format!("lux3d-pi3-export-{}.ply", std::process::id()));
    pipeline
        .export_ply(&output, &output_path)
        .expect("export ply");
    let expected_vertex_count = output
        .export_mask
        .to_dtype(DType::U8)
        .expect("mask u8")
        .flatten_all()
        .expect("mask flatten")
        .to_vec1::<u8>()
        .expect("mask vec")
        .into_iter()
        .filter(|value| *value != 0)
        .count();

    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3")
        .join("house-golden")
        .join("point_cloud.ply");
    let (baseline_vertex_count, _) =
        read_pi3_binary_ply_header(&mut File::open(&baseline_path).expect("open baseline ply"));
    let sample_indices = sample_pi3_vertex_indices(baseline_vertex_count);
    let snapshot = read_ascii_pi3_ply_snapshot(&output_path, &sample_indices, 16);
    let _ = fs::remove_file(&output_path);

    assert_eq!(snapshot.header[0], "ply");
    assert_eq!(snapshot.header[1], "format ascii 1.0");
    assert!(
        snapshot
            .header
            .iter()
            .any(|line| line == "property float nx")
    );
    assert!(
        snapshot
            .header
            .iter()
            .any(|line| line == "property float ny")
    );
    assert!(
        snapshot
            .header
            .iter()
            .any(|line| line == "property float nz")
    );
    assert_eq!(snapshot.header[12], "end_header");
    assert_eq!(snapshot.vertex_count, expected_vertex_count);

    let vertex_count_delta = expected_vertex_count.abs_diff(baseline_vertex_count);
    assert!(
        vertex_count_delta <= 4,
        "baseline/runtime Pi3 export vertex count drifted by {vertex_count_delta} \
         (runtime={expected_vertex_count}, baseline={baseline_vertex_count})"
    );
    let (_, sampled_rows) = read_pi3_binary_ply_samples(&baseline_path, &sample_indices);
    for (index, expected_row) in sampled_rows {
        assert!(
            snapshot
                .sample_windows
                .iter()
                .find(|(sample_index, _)| *sample_index == index)
                .expect("sample window")
                .1
                .iter()
                .map(|line| parse_ascii_pi3_ply_record(line))
                .any(|actual: Pi3PlyVertex| actual.approx_eq(&expected_row, 1e-5)),
            "baseline PLY vertex at sample index {index} did not match any exported row"
        );
    }
}
fn read_bool_safetensor(path: &std::path::Path, key: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("read safetensors bytes");
    let safetensors = SafeTensors::deserialize(&bytes).expect("deserialize safetensors");
    let view = safetensors.tensor(key).expect("tensor view");
    view.data().to_vec()
}

fn sample_pi3_vertex_indices(vertex_count: usize) -> Vec<usize> {
    let mut indices = Vec::with_capacity(5);
    for candidate in [
        0usize,
        vertex_count / 4,
        vertex_count / 2,
        (vertex_count * 3) / 4,
        vertex_count.saturating_sub(1),
    ] {
        if candidate < vertex_count && !indices.contains(&candidate) {
            indices.push(candidate);
        }
    }
    indices
}

#[derive(Debug)]
struct Pi3AsciiPlySnapshot {
    header: Vec<String>,
    vertex_count: usize,
    sample_windows: Vec<(usize, Vec<String>)>,
}

fn read_ascii_pi3_ply_snapshot(
    path: &std::path::Path,
    sample_indices: &[usize],
    window_radius: usize,
) -> Pi3AsciiPlySnapshot {
    let file = File::open(path).expect("open ascii ply");
    let reader = BufReader::new(file);
    let mut header = Vec::with_capacity(13);
    let mut sample_windows = sample_indices
        .iter()
        .copied()
        .map(|index| (index, Vec::new()))
        .collect::<Vec<_>>();
    let mut vertex_count = 0usize;

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.expect("read ascii ply line");
        if line_index < 13 {
            header.push(line);
            continue;
        }

        for (sample_index, lines) in &mut sample_windows {
            let start = sample_index.saturating_sub(window_radius);
            let end = *sample_index + window_radius;
            if vertex_count >= start && vertex_count <= end {
                lines.push(line.clone());
            }
        }
        vertex_count += 1;
    }

    Pi3AsciiPlySnapshot {
        header,
        vertex_count,
        sample_windows,
    }
}

#[derive(Debug, Clone, Copy)]
struct Pi3PlyVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [u8; 3],
}

impl Pi3PlyVertex {
    fn approx_eq(&self, other: &Self, tolerance: f32) -> bool {
        self.position
            .iter()
            .zip(other.position.iter())
            .all(|(lhs, rhs)| (lhs - rhs).abs() <= tolerance)
            && self
                .normal
                .iter()
                .zip(other.normal.iter())
                .all(|(lhs, rhs)| (lhs - rhs).abs() <= tolerance)
            && self.color == other.color
    }
}

fn read_pi3_binary_ply_samples(
    path: &std::path::Path,
    sample_indices: &[usize],
) -> (usize, Vec<(usize, Pi3PlyVertex)>) {
    const RECORD_SIZE: u64 = 27;

    let mut file = File::open(path).expect("open baseline ply");
    let (vertex_count, data_offset) = read_pi3_binary_ply_header(&mut file);
    let mut sampled_rows = Vec::with_capacity(sample_indices.len());
    for &index in sample_indices {
        assert!(index < vertex_count, "sample index out of bounds");
        let offset = data_offset + (index as u64 * RECORD_SIZE);
        file.seek(SeekFrom::Start(offset))
            .expect("seek baseline ply record");
        let mut record = [0u8; RECORD_SIZE as usize];
        file.read_exact(&mut record)
            .expect("read baseline ply record");
        sampled_rows.push((index, parse_binary_pi3_ply_record(&record)));
    }
    (vertex_count, sampled_rows)
}

fn read_pi3_binary_ply_header(file: &mut File) -> (usize, u64) {
    let mut header = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        file.read_exact(&mut byte)
            .expect("read baseline ply header");
        header.push(byte[0]);
        if header.ends_with(b"end_header\n") || header.ends_with(b"end_header\r\n") {
            break;
        }
    }

    let text = String::from_utf8(header.clone()).expect("baseline ply header must be utf8");
    let vertex_count = text
        .lines()
        .find_map(|line| {
            line.strip_prefix("element vertex ")
                .map(|value| value.parse::<usize>().expect("parse vertex count"))
        })
        .expect("baseline ply must declare vertex count");
    (vertex_count, header.len() as u64)
}

fn parse_binary_pi3_ply_record(record: &[u8; 27]) -> Pi3PlyVertex {
    Pi3PlyVertex {
        position: [
            f32::from_le_bytes(record[0..4].try_into().expect("x bytes")),
            f32::from_le_bytes(record[4..8].try_into().expect("y bytes")),
            f32::from_le_bytes(record[8..12].try_into().expect("z bytes")),
        ],
        normal: [
            f32::from_le_bytes(record[12..16].try_into().expect("nx bytes")),
            f32::from_le_bytes(record[16..20].try_into().expect("ny bytes")),
            f32::from_le_bytes(record[20..24].try_into().expect("nz bytes")),
        ],
        color: [record[24], record[25], record[26]],
    }
}

fn parse_ascii_pi3_ply_record(line: &str) -> Pi3PlyVertex {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    assert_eq!(parts.len(), 9, "unexpected ascii PLY row format");
    Pi3PlyVertex {
        position: [
            parts[0].parse::<f32>().expect("parse ascii x"),
            parts[1].parse::<f32>().expect("parse ascii y"),
            parts[2].parse::<f32>().expect("parse ascii z"),
        ],
        normal: [
            parts[3].parse::<f32>().expect("parse ascii nx"),
            parts[4].parse::<f32>().expect("parse ascii ny"),
            parts[5].parse::<f32>().expect("parse ascii nz"),
        ],
        color: [
            parts[6].parse::<u8>().expect("parse ascii red"),
            parts[7].parse::<u8>().expect("parse ascii green"),
            parts[8].parse::<u8>().expect("parse ascii blue"),
        ],
    }
}
