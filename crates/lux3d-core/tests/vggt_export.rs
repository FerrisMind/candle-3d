use candle_core::{DType, Device, Tensor};
use lux3d_core::contracts::{
    ExportStage, GeometryStage, ModelFamily, TensorContract, VggtSceneCode,
};
use lux3d_core::export::{PointCloudFormat, VggtExportStage};
use lux3d_core::geometry::VggtGeometryStage;
use lux3d_core::runtime::VggtInferenceOutput;

/// Minimal synthetic inference output: one frame, 2x2 points, confidences
/// `[1.5, 0.5, 2.0, 1.0]` so the strict `> 1.0` mask keeps only indices 0 and 2.
fn synthetic_output() -> VggtInferenceOutput {
    let device = Device::Cpu;
    let world_points = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0,
        ],
        (1, 1, 2, 2, 3),
        &device,
    )
    .expect("world points");
    let world_points_conf = Tensor::from_vec(vec![1.5f32, 0.5, 2.0, 1.0], (1, 1, 2, 2), &device)
        .expect("world points conf");
    let export_mask = world_points_conf
        .gt(1.0f32)
        .expect("conf > 1.0")
        .to_dtype(DType::U8)
        .expect("mask dtype");

    VggtInferenceOutput {
        world_points,
        world_points_conf,
        depth: Tensor::zeros((1, 1, 2, 2, 1), DType::F32, &device).expect("depth"),
        depth_conf: Tensor::zeros((1, 1, 2, 2), DType::F32, &device).expect("depth conf"),
        pose_enc: Tensor::zeros((1, 1, 9), DType::F32, &device).expect("pose enc"),
        extrinsics: Tensor::zeros((1, 1, 3, 4), DType::F32, &device).expect("extrinsics"),
        intrinsics: Tensor::zeros((1, 1, 3, 3), DType::F32, &device).expect("intrinsics"),
        // [S, C, H, W]: distinct per-channel values; flat pixels are
        // (1, 0, 0.5), (0, 1, 0.5), (0, 1, 0.5), (1, 0, 0.5).
        rgb_frames: Tensor::from_vec(
            vec![
                1.0f32, 0.0, 0.0, 1.0, // red
                0.0, 1.0, 1.0, 0.0, // green
                0.5, 0.5, 0.5, 0.5, // blue
            ],
            (1, 3, 2, 2),
            &device,
        )
        .expect("rgb frames"),
        export_mask,
    }
}

fn synthetic_scene() -> VggtSceneCode {
    VggtSceneCode {
        family: ModelFamily::Vggt,
        backbone: "dinov2_vitl14_reg",
        aggregator_depth: 24,
        camera_refine_iterations: 4,
        world_points: TensorContract::new("vggt.world_points", vec![1, 1, 2, 2, 3]),
        world_points_conf: TensorContract::new("vggt.world_points_conf", vec![1, 1, 2, 2]),
        depth: TensorContract::new("vggt.depth", vec![1, 1, 2, 2, 1]),
        depth_conf: TensorContract::new("vggt.depth_conf", vec![1, 1, 2, 2]),
        pose_enc: TensorContract::new("vggt.pose_enc", vec![1, 1, 9]),
        extrinsics: TensorContract::new("vggt.camera.extrinsics", vec![1, 1, 3, 4]),
        intrinsics: TensorContract::new("vggt.camera.intrinsics", vec![1, 1, 3, 3]),
    }
}

#[test]
fn vggt_geometry_descriptor_uses_world_point_contract() {
    let scene = synthetic_scene();
    let point_cloud = VggtGeometryStage
        .assemble(&scene)
        .expect("vggt geometry descriptor");

    assert_eq!(point_cloud.family, ModelFamily::Vggt);
    assert_eq!(point_cloud.points.name, "vggt.points");
    assert_eq!(point_cloud.points.dims, vec![4, 3]);
    assert_eq!(point_cloud.confidence, scene.world_points_conf);
    assert_eq!(point_cloud.confidence.dims, vec![1, 1, 2, 2]);
    assert_eq!(
        point_cloud.assembly_path,
        "world_points = point_head(aggregated_tokens)"
    );
    assert!(point_cloud.supports_depth_unprojection);
}

#[test]
fn vggt_assemble_cpu_applies_strict_confidence_threshold() {
    let output = synthetic_output();
    let point_cloud = VggtGeometryStage
        .assemble_cpu(&output, 1.0)
        .expect("vggt cpu point cloud");

    // 1.5 and 2.0 pass; 0.5 and the exact-threshold 1.0 are excluded.
    assert_eq!(point_cloud.vertex_count, 2);
    assert_eq!(point_cloud.points, vec![1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);
    assert_eq!(point_cloud.colors, vec![255, 0, 128, 0, 255, 128]);

    // A confidence exactly equal to the threshold is never exported.
    let none_above = VggtGeometryStage
        .assemble_cpu(&output, 2.0)
        .expect("vggt cpu point cloud at threshold 2.0");
    assert_eq!(none_above.vertex_count, 0);
    assert!(none_above.points.is_empty());
    assert!(none_above.colors.is_empty());
}

#[test]
fn vggt_write_ply_emits_pi3_ascii_format() {
    let output = synthetic_output();
    let point_cloud = VggtGeometryStage
        .assemble_cpu(&output, 1.0)
        .expect("vggt cpu point cloud");
    let path = std::env::temp_dir().join(format!("lux3d-vggt-export-{}.ply", std::process::id()));
    VggtExportStage
        .write_ply(&point_cloud, &path)
        .expect("write vggt ply");

    let contents = std::fs::read_to_string(&path).expect("read vggt ply");
    std::fs::remove_file(&path).expect("remove vggt ply");

    let mut lines = contents.lines();
    for expected in [
        "ply",
        "format ascii 1.0",
        "element vertex 2",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ] {
        assert_eq!(lines.next(), Some(expected), "unexpected PLY header line");
    }
    assert_eq!(lines.next(), Some("1 2 3 0 0 0 255 0 128"));
    assert_eq!(lines.next(), Some("7 8 9 0 0 0 0 255 128"));
    assert_eq!(lines.next(), None, "unexpected extra PLY vertex lines");
}

#[test]
fn vggt_write_point_cloud_dispatches_xyz() {
    let output = synthetic_output();
    let point_cloud = VggtGeometryStage
        .assemble_cpu(&output, 1.0)
        .expect("vggt cpu point cloud");
    let path = std::env::temp_dir().join(format!("lux3d-vggt-export-{}.xyz", std::process::id()));
    assert_eq!(
        VggtExportStage
            .write_point_cloud(&point_cloud, &path)
            .expect("write vggt xyz"),
        PointCloudFormat::Xyz
    );

    let contents = std::fs::read_to_string(&path).expect("read vggt xyz");
    std::fs::remove_file(&path).expect("remove vggt xyz");
    assert_eq!(contents, "1 2 3 255 0 128\n7 8 9 0 255 128\n");
}

#[test]
fn vggt_export_plan_documents_demo_export() {
    let scene = synthetic_scene();
    let point_cloud = VggtGeometryStage
        .assemble(&scene)
        .expect("vggt geometry descriptor");
    let plan = VggtExportStage
        .export(&point_cloud)
        .expect("vggt export plan");

    assert_eq!(plan.primary_extension, "ply");
    assert_eq!(plan.alternate_extensions, vec!["xyz".to_string()]);
    assert_eq!(
        plan.utility_notes,
        vec![
            "world_points_conf > 1.0".to_string(),
            "colors from preprocessed RGB frames".to_string(),
            "write_ply(points, rgb); .xyz is a plain-text alternative".to_string(),
        ]
    );
}
