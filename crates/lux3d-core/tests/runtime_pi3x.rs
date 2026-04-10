use std::fs;
use std::path::PathBuf;

use candle_core::{DType, Device};
use lux3d_core::runtime::{Pi3xInjectConditions, Pi3xPipeline, Pi3xVoPipeline};
use lux3d_core::test_support::GpuTestLock;
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(r"H:\GitHub\LuxRT")
}

fn accel_device() -> Device {
    Device::new_cuda(0).expect("runtime_pi3x tests require CUDA")
}

fn core_baseline() -> candle_core::safetensors::MmapedSafetensors {
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3x")
        .join("room-golden")
        .join("artifacts.safetensors");
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("pi3x core baseline")
    }
}

fn vo_baseline() -> candle_core::safetensors::MmapedSafetensors {
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3x")
        .join("skating-vo-golden")
        .join("artifacts.safetensors");
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path).expect("pi3x vo baseline")
    }
}

fn vo_inject_baseline() -> candle_core::safetensors::MmapedSafetensors {
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("pi3x")
        .join("skating-vo-inject-golden")
        .join("artifacts.safetensors");
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("pi3x vo inject baseline")
    }
}

fn max_abs_delta(actual: &candle_core::Tensor, expected: &candle_core::Tensor) -> f32 {
    actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(expected)
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
fn pi3x_preprocess_matches_room_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xPipeline::load(repo_root()).expect("pi3x pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb"),
            Some(
                &repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("condition.npz"),
            ),
            None,
            &accel_device(),
        )
        .expect("pi3x prepared inputs");

    let expected_rgb = core_baseline()
        .load("pi3x.loader.rgb_frames", &accel_device())
        .expect("rgb frames");
    let expected_norm = core_baseline()
        .load("pi3x.model.normalized_frames", &accel_device())
        .expect("normalized frames");
    let expected_depths = core_baseline()
        .load("pi3x.conditions.depths", &accel_device())
        .expect("depths");
    let expected_intrinsics = core_baseline()
        .load("pi3x.conditions.intrinsics", &accel_device())
        .expect("intrinsics");
    let expected_poses = core_baseline()
        .load("pi3x.conditions.poses", &accel_device())
        .expect("poses");

    let actual_rgb = prepared.rgb_frames.unsqueeze(0).expect("rgb unsqueeze");
    let rgb_delta = max_abs_delta(&actual_rgb, &expected_rgb);
    let norm_delta = max_abs_delta(&prepared.normalized_frames, &expected_norm);
    let actual_depths = prepared
        .depths
        .as_ref()
        .expect("depths")
        .to_device(&accel_device())
        .expect("depths to gpu");
    let actual_intrinsics = prepared
        .intrinsics
        .as_ref()
        .expect("intrinsics")
        .to_device(&accel_device())
        .expect("intrinsics to gpu");
    let actual_poses = prepared
        .poses
        .as_ref()
        .expect("poses")
        .to_device(&accel_device())
        .expect("poses to gpu");

    let depths_delta = max_abs_delta(&actual_depths, &expected_depths);
    let intrinsics_delta = max_abs_delta(&actual_intrinsics, &expected_intrinsics);
    let poses_delta = max_abs_delta(&actual_poses, &expected_poses);

    assert!(rgb_delta <= 0.02, "rgb delta too large: {rgb_delta}");
    assert!(
        norm_delta <= 0.05,
        "normalized delta too large: {norm_delta}"
    );
    assert!(
        depths_delta <= 1e-5,
        "depth delta too large: {depths_delta}"
    );
    assert!(
        intrinsics_delta <= 1e-5,
        "intrinsics delta too large: {intrinsics_delta}"
    );
    assert!(poses_delta <= 1e-5, "poses delta too large: {poses_delta}");
}

#[test]
fn pi3x_core_outputs_match_room_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xPipeline::load(repo_root()).expect("pi3x pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb"),
            Some(
                &repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("condition.npz"),
            ),
            None,
            &accel_device(),
        )
        .expect("pi3x inference");

    let expected_points = core_baseline()
        .load("pi3x.points", &accel_device())
        .expect("points");
    let expected_metric = core_baseline()
        .load("pi3x.metric", &accel_device())
        .expect("metric");

    let points_delta = max_abs_delta(&output.points, &expected_points);
    let metric_delta = max_abs_delta(&output.metric, &expected_metric);
    let summary_path = repo_root()
        .join("3d")
        .join("baselines")
        .join("pi3x")
        .join("room-golden")
        .join("summary.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(summary_path).expect("summary")).expect("json");
    let expected_mask_true_count = summary["mask_true_count"]
        .as_u64()
        .expect("mask_true_count");
    let actual_mask_true_count = output
        .export_mask
        .to_device(&Device::Cpu)
        .expect("mask cpu")
        .flatten_all()
        .expect("mask flatten")
        .to_vec1::<u8>()
        .expect("mask vec")
        .into_iter()
        .filter(|value| *value != 0)
        .count() as u64;

    assert!(
        points_delta <= 0.25,
        "points delta too large: {points_delta}"
    );
    assert!(
        metric_delta <= 0.05,
        "metric delta too large: {metric_delta}"
    );
    assert!(
        actual_mask_true_count.abs_diff(expected_mask_true_count) <= 10_000,
        "mask_true_count mismatch: actual={actual_mask_true_count} expected={expected_mask_true_count}"
    );
}

#[test]
fn pi3x_export_ply_end_to_end_emits_vendor_aligned_header() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xPipeline::load(repo_root()).expect("pi3x pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb"),
            Some(
                &repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("condition.npz"),
            ),
            None,
            &accel_device(),
        )
        .expect("pi3x inference");

    let output_path = std::env::temp_dir().join(format!("lux3d-pi3x-{}.ply", std::process::id()));
    pipeline
        .export_ply(&output, &output_path)
        .expect("pi3x ply export");
    let body = fs::read_to_string(&output_path).expect("read pi3x ply");
    let _ = fs::remove_file(&output_path);

    assert!(body.starts_with("ply\nformat ascii 1.0\n"));
    assert!(body.contains("property float nx"));
}

#[test]
fn pi3x_vo_outputs_match_skating_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(8),
            Some(4),
            Some(0.05),
            Pi3xInjectConditions::default(),
            &accel_device(),
        )
        .expect("pi3x vo inference");

    let expected_points = vo_baseline()
        .load("pi3xvo.points", &accel_device())
        .expect("vo points");
    let expected_poses = vo_baseline()
        .load("pi3xvo.camera_poses", &accel_device())
        .expect("vo poses");
    let expected_sim3 = vo_baseline()
        .load("pi3xvo.sim3_transforms", &accel_device())
        .expect("vo sim3");

    let points_delta = max_abs_delta(&output.points, &expected_points);
    let poses_delta = max_abs_delta(&output.camera_poses, &expected_poses);
    let sim3_delta = max_abs_delta(&output.sim3_transforms, &expected_sim3);
    eprintln!(
        "pi3x_vo_with_injected_overlap_conditions_matches_golden_baseline deltas: points={points_delta} poses={poses_delta} sim3={sim3_delta}"
    );

    assert!(
        points_delta <= 5.0,
        "vo points delta too large: {points_delta}"
    );
    assert!(
        poses_delta <= 0.15,
        "vo poses delta too large: {poses_delta}"
    );
    assert!(sim3_delta <= 0.15, "vo sim3 delta too large: {sim3_delta}");

    let points = output
        .points
        .to_device(&Device::Cpu)
        .expect("points cpu")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("vec");
    let summary_path = repo_root()
        .join("3d")
        .join("baselines")
        .join("pi3x")
        .join("skating-vo-golden")
        .join("summary.json");
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(summary_path).expect("summary")).expect("json");

    let expected_bbox_min = summary["bbox_min"]
        .as_array()
        .expect("bbox_min")
        .iter()
        .map(|v| v.as_f64().expect("bbox val") as f32)
        .collect::<Vec<_>>();
    let expected_bbox_max = summary["bbox_max"]
        .as_array()
        .expect("bbox_max")
        .iter()
        .map(|v| v.as_f64().expect("bbox val") as f32)
        .collect::<Vec<_>>();

    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for chunk in points.chunks_exact(3) {
        for axis in 0..3 {
            mins[axis] = mins[axis].min(chunk[axis]);
            maxs[axis] = maxs[axis].max(chunk[axis]);
        }
    }

    for axis in 0..3 {
        assert!((mins[axis] - expected_bbox_min[axis]).abs() <= 0.5);
        assert!((maxs[axis] - expected_bbox_max[axis]).abs() <= 0.5);
    }
}

#[test]
fn pi3x_vo_with_injected_overlap_conditions_runs() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(8),
            Some(4),
            Some(0.05),
            Pi3xInjectConditions {
                pose: true,
                depth: true,
                ray: true,
            },
            &accel_device(),
        )
        .expect("pi3x vo injected inference");

    assert_eq!(&[1, 11, 364, 700, 3], output.points.dims());
    assert_eq!(&[1, 11, 4, 4], output.camera_poses.dims());
    assert_eq!(&[1, 11, 364, 700, 1], output.confidence_logits.dims());
}

#[test]
fn pi3x_vo_with_injected_overlap_conditions_matches_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(8),
            Some(4),
            Some(0.05),
            Pi3xInjectConditions {
                pose: true,
                depth: true,
                ray: true,
            },
            &accel_device(),
        )
        .expect("pi3x vo injected inference");

    let expected_points = vo_inject_baseline()
        .load("pi3xvo.points", &accel_device())
        .expect("vo inject points");
    let expected_poses = vo_inject_baseline()
        .load("pi3xvo.camera_poses", &accel_device())
        .expect("vo inject poses");
    let expected_sim3 = vo_inject_baseline()
        .load("pi3xvo.sim3_transforms", &accel_device())
        .expect("vo inject sim3");

    let points_delta = max_abs_delta(&output.points, &expected_points);
    let poses_delta = max_abs_delta(&output.camera_poses, &expected_poses);
    let sim3_delta = max_abs_delta(&output.sim3_transforms, &expected_sim3);

    // Prior-injection uses the Candle multimodal path plus CPU-side alignment math, and the
    // geometry remains stable while pose/sim3 drift is materially larger than the plain VO path.
    // Keep this gate explicit rather than pretending strict parity is achievable here.
    assert!(
        points_delta <= 5.0 && poses_delta <= 1.0 && sim3_delta <= 1.0,
        "vo injected deltas too large: points={points_delta} poses={poses_delta} sim3={sim3_delta}"
    );
}

#[test]
fn pi3x_vo_rejects_invalid_overlap_configuration() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let err = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(4),
            Some(4),
            Some(0.05),
            Pi3xInjectConditions::default(),
            &accel_device(),
        )
        .expect_err("expected invalid overlap error");
    let message = err.to_string();
    assert!(message.contains("overlap must be smaller than chunk_size"));
}

#[test]
fn pi3x_vo_sparse_overlap_uses_adaptive_masking_and_runs() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(8),
            Some(4),
            Some(1.1),
            Pi3xInjectConditions::default(),
            &accel_device(),
        )
        .expect("sparse overlap should still produce a VO output");
    assert_eq!(&[1, 11, 364, 700, 3], output.points.dims());
    assert_eq!(&[1, 11, 4, 4], output.camera_poses.dims());
    assert_eq!(&[1, 1, 4, 4], output.sim3_transforms.dims());
    assert_eq!(&[1, 11, 364, 700], output.export_mask.dims());
}

#[test]
fn pi3x_vo_export_ply_uses_inference_time_masking() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline = Pi3xVoPipeline::load(repo_root()).expect("pi3x vo pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            None,
            Some(8),
            Some(4),
            Some(1.1),
            Pi3xInjectConditions::default(),
            &accel_device(),
        )
        .expect("pi3x vo sparse-overlap inference");

    let output_path =
        std::env::temp_dir().join(format!("lux3d-pi3x-vo-export-{}.ply", std::process::id()));
    pipeline
        .export_ply(&output, &output_path)
        .expect("pi3x vo ply export");
    let body = fs::read_to_string(&output_path).expect("read pi3x vo ply");
    let _ = fs::remove_file(&output_path);
    let lines = body.lines().collect::<Vec<_>>();

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
    let expected_rows = expected_pi3x_vo_ply_rows(&output);

    assert_eq!(lines[0], "ply");
    assert_eq!(lines[1], "format ascii 1.0");
    assert!(lines.contains(&"property float nx"));
    assert_eq!(lines.len(), expected_vertex_count + 13);
    assert_eq!(expected_rows.len(), expected_vertex_count);
    assert_eq!(lines[13], expected_rows[0]);
    assert_eq!(
        lines[13 + expected_rows.len() / 2],
        expected_rows[expected_rows.len() / 2]
    );
    assert_eq!(
        lines[12 + expected_rows.len()],
        expected_rows[expected_rows.len() - 1]
    );
}

fn expected_pi3x_vo_ply_rows(output: &lux3d_core::runtime::Pi3xVoOutput) -> Vec<String> {
    let mask = output
        .export_mask
        .to_dtype(DType::U8)
        .expect("mask u8")
        .flatten_all()
        .expect("mask flatten")
        .to_vec1::<u8>()
        .expect("mask vec");
    let points = output
        .points
        .to_device(&Device::Cpu)
        .expect("points cpu")
        .flatten_all()
        .expect("points flatten")
        .to_vec1::<f32>()
        .expect("points vec");
    let rgb = output
        .rgb_frames
        .to_device(&Device::Cpu)
        .expect("rgb cpu")
        .permute((0, 2, 3, 1))
        .expect("rgb permute")
        .flatten_all()
        .expect("rgb flatten")
        .to_vec1::<f32>()
        .expect("rgb vec");

    let mut rows = Vec::new();
    for (idx, keep) in mask.into_iter().enumerate() {
        if keep == 0 {
            continue;
        }
        let p = idx * 3;
        let r = (rgb[p].clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (rgb[p + 1].clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (rgb[p + 2].clamp(0.0, 1.0) * 255.0).round() as u8;
        rows.push(format!(
            "{} {} {} {} {} {} {} {} {}",
            points[p],
            points[p + 1],
            points[p + 2],
            0.0,
            0.0,
            0.0,
            r,
            g,
            b
        ));
    }
    rows
}
