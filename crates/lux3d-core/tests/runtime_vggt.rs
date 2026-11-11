use candle_core::{DType, Device, Tensor};
use lux3d_core::ModelFamily;
use lux3d_core::contracts::{NeuralStage, SpatialSize, VggtPreparedBatch};
use lux3d_core::runtime::VggtPipeline;
use lux3d_core::test_support::{GpuTestLock, model_asset_options, runtime_root};

fn accel_device() -> Option<Device> {
    Device::new_cuda(0).ok()
}

fn baseline_path() -> std::path::PathBuf {
    runtime_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("vggt")
        .join("pi3-frames-golden")
        .join("artifacts.safetensors")
}

fn as_f32(tensor: &Tensor) -> Tensor {
    tensor.to_dtype(DType::F32).expect("cast tensor to f32")
}

/// Elementwise max |a - b|.
fn max_abs_delta(lhs: &Tensor, rhs: &Tensor) -> f32 {
    let lhs = as_f32(lhs);
    let rhs = as_f32(rhs);
    lhs.sub(&rhs)
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

/// Elementwise max |a - b| / max(|b|, 1e-3).
fn max_rel_delta(lhs: &Tensor, rhs: &Tensor) -> f32 {
    let lhs = as_f32(lhs);
    let rhs = as_f32(rhs);
    let delta = lhs.sub(&rhs).expect("delta").abs().expect("abs");
    let denominator = rhs
        .abs()
        .expect("rhs abs")
        .clamp(1e-3f32, f32::MAX)
        .expect("clamp");
    (delta / denominator)
        .expect("relative delta")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar")
}

fn report(name: &str, dims: &[usize], max_abs: f32, rel: Option<f32>) {
    match rel {
        Some(rel) => {
            eprintln!("[vggt-parity] {name} dims={dims:?} max_abs={max_abs:.6e} rel_max={rel:.6e}")
        }
        None => eprintln!("[vggt-parity] {name} dims={dims:?} max_abs={max_abs:.6e} rel_max=n/a"),
    }
}

#[test]
fn vggt_runtime_matches_pi3_frames_golden_baseline() {
    let Some(device) = accel_device() else {
        eprintln!("skipping VGGT CUDA parity test: no CUDA device available");
        return;
    };
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        VggtPipeline::load(model_asset_options(ModelFamily::Vggt)).expect("vggt pipeline");

    let source = runtime_root()
        .join("candle-3d")
        .join("test-assets")
        .join("pi3-frames");
    let prepared = pipeline
        .prepare_inputs_from_path(&source, None, &device)
        .expect("vggt prepared inputs");
    assert_eq!(prepared.target_size, SpatialSize::new(518, 518));
    eprintln!(
        "[vggt-parity] prepared target={:?} original={:?} interval={}",
        prepared.target_size, prepared.original_size, prepared.interval
    );

    let baseline = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path())
            .expect("vggt baseline safetensors")
    };
    let load_baseline = |name: &str| -> Tensor {
        baseline
            .load(name, &device)
            .unwrap_or_else(|source| panic!("baseline tensor {name}: {source}"))
    };

    let (output, taps) = pipeline
        .debug_forward(&prepared)
        .expect("vggt debug forward");

    // 1. DINOv2 patch tokens.
    let expected = load_baseline("vggt.encoder.patch_tokens");
    assert_eq!(taps.patch_tokens.dims(), expected.dims());
    let delta = max_abs_delta(&taps.patch_tokens, &expected);
    report(
        "vggt.encoder.patch_tokens",
        taps.patch_tokens.dims(),
        delta,
        None,
    );
    assert!(delta <= 2e-3, "patch token delta {delta:e} > 2e-3");

    // 2. Aggregator cached layers 4/11/17/23.
    let layer_taps = [
        ("vggt.aggregator.layer4", &taps.aggregator_layer4),
        ("vggt.aggregator.layer11", &taps.aggregator_layer11),
        ("vggt.aggregator.layer17", &taps.aggregator_layer17),
        ("vggt.aggregator.layer23", &taps.aggregator_layer23),
    ];
    for (name, tap) in layer_taps {
        let expected = load_baseline(name);
        assert_eq!(tap.dims(), expected.dims(), "{name} shape mismatch");
        let delta = max_abs_delta(tap, &expected);
        report(name, tap.dims(), delta, None);
        assert!(delta <= 5e-3, "{name} delta {delta:e} > 5e-3");
    }

    // 3. Camera-head refinement iterations (activated pose encodings).
    assert_eq!(taps.pose_enc_iterations.len(), 4);
    for (idx, tap) in taps.pose_enc_iterations.iter().enumerate() {
        let name = format!("vggt.camera.pose_enc.{idx}");
        let expected = load_baseline(&name);
        assert_eq!(tap.dims(), expected.dims(), "{name} shape mismatch");
        let delta = max_abs_delta(tap, &expected);
        report(&name, tap.dims(), delta, None);
        assert!(delta <= 1e-3, "{name} delta {delta:e} > 1e-3");
    }

    // 4. Depth (relative tolerance) + absolute delta.
    let expected = load_baseline("vggt.depth");
    assert_eq!(output.depth.dims(), expected.dims());
    let depth_abs = max_abs_delta(&output.depth, &expected);
    let depth_rel = max_rel_delta(&output.depth, &expected);
    report(
        "vggt.depth",
        output.depth.dims(),
        depth_abs,
        Some(depth_rel),
    );
    assert!(
        depth_rel <= 5e-3,
        "depth relative delta {depth_rel:e} > 5e-3"
    );

    // 5. Depth and world-point confidence.
    let expected = load_baseline("vggt.depth_conf");
    assert_eq!(output.depth_conf.dims(), expected.dims());
    let delta = max_abs_delta(&output.depth_conf, &expected);
    report("vggt.depth_conf", output.depth_conf.dims(), delta, None);
    assert!(delta <= 5e-3, "depth_conf delta {delta:e} > 5e-3");

    let expected = load_baseline("vggt.world_points_conf");
    assert_eq!(output.world_points_conf.dims(), expected.dims());
    let delta = max_abs_delta(&output.world_points_conf, &expected);
    report(
        "vggt.world_points_conf",
        output.world_points_conf.dims(),
        delta,
        None,
    );
    assert!(delta <= 5e-3, "world_points_conf delta {delta:e} > 5e-3");

    // 6. World points (relative tolerance) + absolute delta.
    let expected = load_baseline("vggt.world_points");
    assert_eq!(output.world_points.dims(), expected.dims());
    let points_abs = max_abs_delta(&output.world_points, &expected);
    let points_rel = max_rel_delta(&output.world_points, &expected);
    report(
        "vggt.world_points",
        output.world_points.dims(),
        points_abs,
        Some(points_rel),
    );
    assert!(
        points_rel <= 5e-3,
        "world_points relative delta {points_rel:e} > 5e-3"
    );

    // 7. Pose decoding.
    let expected = load_baseline("vggt.camera.extrinsics");
    assert_eq!(output.extrinsics.dims(), expected.dims());
    let delta = max_abs_delta(&output.extrinsics, &expected);
    report(
        "vggt.camera.extrinsics",
        output.extrinsics.dims(),
        delta,
        None,
    );
    assert!(delta <= 2e-3, "extrinsics delta {delta:e} > 2e-3");

    let expected = load_baseline("vggt.camera.intrinsics");
    assert_eq!(output.intrinsics.dims(), expected.dims());
    let delta = max_abs_delta(&output.intrinsics, &expected);
    report(
        "vggt.camera.intrinsics",
        output.intrinsics.dims(),
        delta,
        None,
    );
    assert!(delta <= 2e-3, "intrinsics delta {delta:e} > 2e-3");

    // 8. Output shapes.
    assert_eq!(output.export_mask.dims(), &[1, 3, 518, 518]);
    assert_eq!(output.export_mask.dtype(), DType::U8);
    assert_eq!(output.rgb_frames.dims(), &[3, 3, 518, 518]);
    assert_eq!(output.world_points.dims(), &[1, 3, 518, 518, 3]);
    assert_eq!(output.world_points_conf.dims(), &[1, 3, 518, 518]);
    assert_eq!(output.depth.dims(), &[1, 3, 518, 518, 1]);
    assert_eq!(output.depth_conf.dims(), &[1, 3, 518, 518]);
    assert_eq!(output.pose_enc.dims(), &[1, 3, 9]);
    assert_eq!(output.extrinsics.dims(), &[1, 3, 3, 4]);
    assert_eq!(output.intrinsics.dims(), &[1, 3, 3, 3]);

    // 9. Export mask is exactly `world_points_conf > 1.0` and non-empty.
    let expected_mask = output
        .world_points_conf
        .gt(1.0f32)
        .expect("conf > 1.0")
        .to_dtype(DType::U8)
        .expect("mask dtype");
    let mask_delta = max_abs_delta(&output.export_mask, &expected_mask);
    let vertices = expected_mask
        .flatten_all()
        .expect("mask flatten")
        .to_dtype(DType::U32)
        .expect("mask u32")
        .sum_all()
        .expect("mask sum")
        .to_scalar::<u32>()
        .expect("mask count");
    report(
        "vggt.export_mask",
        output.export_mask.dims(),
        mask_delta,
        None,
    );
    eprintln!(
        "[vggt-parity] export vertices above threshold 1.0: {vertices} / {}",
        expected_mask.elem_count()
    );
    assert_eq!(mask_delta, 0.0, "export_mask differs from conf > 1.0");
    assert!(vertices > 0, "export mask is empty at threshold 1.0");

    // 10. PLY export reuses `output` and writes the same vertex set.
    let export_vertices = output
        .export_mask
        .flatten_all()
        .and_then(|mask| mask.to_dtype(DType::U32))
        .and_then(|mask| mask.sum_all())
        .and_then(|mask| mask.to_scalar::<u32>())
        .expect("export mask vertex count");
    assert_eq!(
        export_vertices, vertices,
        "export_mask vertex count differs from conf > 1.0"
    );
    assert!(export_vertices > 0, "PLY export would be empty");

    let ply_path = std::env::temp_dir().join(format!(
        "lux3d-vggt-export-runtime-{}.ply",
        std::process::id()
    ));
    pipeline
        .export_ply_with_threshold(&output, &ply_path, 1.0)
        .expect("vggt export_ply_with_threshold");
    assert!(ply_path.exists(), "PLY export file was not created");
    let ply_contents = std::fs::read_to_string(&ply_path).expect("read exported PLY");
    std::fs::remove_file(&ply_path).expect("remove exported PLY");

    assert!(
        ply_contents.starts_with("ply\n"),
        "exported PLY must start with the ply magic line"
    );
    let vertex_line = format!("element vertex {export_vertices}");
    assert!(
        ply_contents.lines().any(|line| line == vertex_line),
        "exported PLY must declare `{vertex_line}`"
    );
}

#[test]
fn vggt_neural_stage_contract_uses_corrected_shapes() {
    let stage = lux3d_core::neural::VggtNeuralStage;
    let batch = VggtPreparedBatch {
        family: ModelFamily::Vggt,
        source: lux3d_core::contracts::Pi3InputSource::Directory,
        sampled_frames: 3,
        original_size: SpatialSize::new(512, 512),
        target_size: SpatialSize::new(518, 518),
        interval: 1,
        pixel_range: lux3d_core::contracts::RgbRange::ZeroToOne,
    };
    let scene = stage.infer(&batch).expect("vggt neural contract");

    assert_eq!(scene.family, ModelFamily::Vggt);
    assert_eq!(scene.backbone, "dinov2_vitl14_reg");
    assert_eq!(scene.aggregator_depth, 24);
    assert_eq!(scene.camera_refine_iterations, 4);
    assert_eq!(scene.world_points.name, "vggt.world_points");
    assert_eq!(scene.world_points.dims, vec![1, 3, 518, 518, 3]);
    assert_eq!(scene.world_points_conf.dims, vec![1, 3, 518, 518]);
    assert_eq!(scene.depth.dims, vec![1, 3, 518, 518, 1]);
    assert_eq!(scene.depth_conf.dims, vec![1, 3, 518, 518]);
    assert_eq!(scene.pose_enc.name, "vggt.camera.pose_enc");
    assert_eq!(scene.pose_enc.dims, vec![1, 3, 9]);
    assert_eq!(scene.extrinsics.name, "vggt.camera.extrinsics");
    assert_eq!(scene.extrinsics.dims, vec![1, 3, 3, 4]);
    assert_eq!(scene.intrinsics.name, "vggt.camera.intrinsics");
    assert_eq!(scene.intrinsics.dims, vec![1, 3, 3, 3]);
}
