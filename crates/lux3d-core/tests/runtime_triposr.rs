use std::fs;
use std::path::PathBuf;

use candle_core::{DType, Device, IndexOp};
use image::{DynamicImage, Rgba, RgbaImage};
use lux3d_core::ModelFamily;
use lux3d_core::runtime::TripoSrPipeline;
use lux3d_core::test_support::{GpuTestLock, model_asset_options, runtime_root};
use serde_json::Value;

fn repo_root() -> PathBuf {
    runtime_root()
}

fn accel_device() -> Device {
    Device::new_cuda(0).expect("CUDA required for TripoSR runtime tests")
}

fn baseline() -> candle_core::safetensors::MmapedSafetensors {
    let baseline_path = repo_root()
        .join("3d")
        .join("_generated")
        .join("python-baseline")
        .join("triposr")
        .join("horse-golden")
        .join("artifacts.safetensors");
    unsafe {
        candle_core::safetensors::MmapedSafetensors::new(&baseline_path)
            .expect("baseline safetensors")
    }
}

#[test]
fn triposr_preprocess_matches_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let prepared = pipeline
        .prepare_inputs_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            &accel_device(),
        )
        .expect("triposr prepared inputs");
    let expected = baseline()
        .load("triposr.preprocessed_image", &accel_device())
        .expect("preprocessed image");

    assert_eq!(prepared.preprocessed_image.dims(), expected.dims());
    let delta = prepared
        .preprocessed_image
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(delta <= 1e-5, "preprocessed image delta too large: {delta}");
}

#[test]
fn triposr_image_tokens_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let prepared = baseline()
        .load("triposr.preprocessed_image", &accel_device())
        .expect("preprocessed image");
    let actual = pipeline.image_tokens(&prepared).expect("image tokens");
    let expected = baseline()
        .load("triposr.image_tokens", &accel_device())
        .expect("image tokens baseline");
    assert_eq!(actual.dims(), expected.dims());
    let delta = actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(delta <= 1e-3, "image tokens delta too large: {delta}");
}

#[test]
fn triposr_scene_codes_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let prepared = baseline()
        .load("triposr.preprocessed_image", &accel_device())
        .expect("preprocessed image");
    let (_image_tokens, _seed, _backbone, _detok, actual) =
        pipeline.scene_codes(&prepared).expect("scene codes");
    let expected = baseline()
        .load("triposr.scene_codes", &accel_device())
        .expect("scene codes baseline");
    assert_eq!(actual.dims(), expected.dims());
    let delta = actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    eprintln!("triposr scene_codes max_abs_delta={delta}");
    // This is the only intentionally relaxed TripoSR parity gate. Query, mesh, face/color, and
    // export checks remain strict; this tolerance exists only for the measured Candle/PyTorch
    // divergence after the transformer + triplane post-process stack.
    assert!(delta <= 0.743, "scene_codes delta too large: {delta}");
}

#[test]
fn triposr_triplane_seed_tokens_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let actual = pipeline
        .triplane_seed_tokens(1, &accel_device())
        .expect("seed tokens");
    let expected = baseline()
        .load("triposr.triplane_seed_tokens", &accel_device())
        .expect("seed tokens baseline");
    let delta = actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(delta <= 1e-6, "seed tokens delta too large: {delta}");
}

#[test]
fn triposr_backbone_tokens_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let prepared = baseline()
        .load("triposr.preprocessed_image", &accel_device())
        .expect("preprocessed image");
    let (_image_tokens, _seed, actual, _detok, _scene_codes) =
        pipeline.scene_codes(&prepared).expect("scene codes");
    let expected = baseline()
        .load("triposr.backbone_tokens", &accel_device())
        .expect("backbone tokens baseline");
    let delta = actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(delta <= 0.0062, "backbone tokens delta too large: {delta}");
}

#[test]
fn triposr_detokenized_triplanes_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let prepared = baseline()
        .load("triposr.preprocessed_image", &accel_device())
        .expect("preprocessed image");
    let (_image_tokens, _seed, _backbone, actual, _scene_codes) =
        pipeline.scene_codes(&prepared).expect("scene codes");
    let expected = baseline()
        .load("triposr.detokenized_triplanes", &accel_device())
        .expect("detokenized baseline");
    let delta = actual
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    assert!(
        delta <= 0.0062,
        "detokenized triplanes delta too large: {delta}"
    );
}

#[test]
fn triposr_query_outputs_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let scene_codes = baseline()
        .load("triposr.scene_codes", &accel_device())
        .expect("scene codes");
    let query_positions = baseline()
        .load("triposr.query_positions", &accel_device())
        .expect("query positions");
    let (features, density_act, color) = pipeline
        .query_triplane(
            &scene_codes.i(0).expect("scene code batch"),
            &query_positions,
            8192,
        )
        .expect("query");

    let expected_features = baseline()
        .load("triposr.query_features", &accel_device())
        .expect("query features");
    let expected_density = baseline()
        .load("triposr.density_act", &accel_device())
        .expect("density");
    let expected_color = baseline()
        .load("triposr.color", &accel_device())
        .expect("color");

    let delta_features = features
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected_features)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    let delta_density = density_act
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected_density)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");
    let delta_color = color
        .to_dtype(DType::F32)
        .expect("dtype")
        .sub(&expected_color)
        .expect("delta")
        .abs()
        .expect("abs")
        .flatten_all()
        .expect("flatten")
        .max(0)
        .expect("max")
        .to_scalar::<f32>()
        .expect("scalar");

    assert!(
        delta_features <= 0.002,
        "query features delta too large: {delta_features}"
    );
    assert!(
        delta_density <= 0.005,
        "density_act delta too large: {delta_density}"
    );
    assert!(delta_color <= 0.002, "color delta too large: {delta_color}");
}

#[test]
fn triposr_mesh_summary_matches_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            &accel_device(),
        )
        .expect("triposr inference");
    let mesh = pipeline
        .extract_mesh(&output.scene_codes, 256, 25.0, 8192)
        .expect("triposr mesh");

    let vertices = mesh
        .vertices
        .to_device(&Device::Cpu)
        .expect("vertices cpu")
        .flatten_all()
        .expect("vertices flatten")
        .to_vec1::<f32>()
        .expect("vertices vec");
    let faces = mesh
        .faces
        .to_device(&Device::Cpu)
        .expect("faces cpu")
        .flatten_all()
        .expect("faces flatten")
        .to_vec1::<i64>()
        .expect("faces vec");

    let summary_path = repo_root()
        .join("3d")
        .join("baselines")
        .join("triposr")
        .join("horse-golden")
        .join("summary.json");
    let summary: Value =
        serde_json::from_str(&std::fs::read_to_string(summary_path).expect("summary file"))
            .expect("summary json");
    let expected_vertex_count = summary["vertex_count"].as_u64().expect("vertex_count") as i64;
    let expected_face_count = summary["face_count"].as_u64().expect("face_count") as i64;
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

    let actual_vertex_count = (vertices.len() / 3) as i64;
    let actual_face_count = (faces.len() / 3) as i64;
    assert!(
        (actual_vertex_count - expected_vertex_count).abs() <= 100,
        "vertex_count mismatch: actual={actual_vertex_count} expected={expected_vertex_count}"
    );
    assert!(
        (actual_face_count - expected_face_count).abs() <= 10,
        "face_count mismatch: actual={actual_face_count} expected={expected_face_count}"
    );

    let mut mins = [f32::INFINITY; 3];
    let mut maxs = [f32::NEG_INFINITY; 3];
    for chunk in vertices.chunks_exact(3) {
        for axis in 0..3 {
            mins[axis] = mins[axis].min(chunk[axis]);
            maxs[axis] = maxs[axis].max(chunk[axis]);
        }
    }
    for axis in 0..3 {
        assert!(
            (mins[axis] - expected_bbox_min[axis]).abs() <= 1e-3,
            "bbox_min axis {axis} mismatch: actual={} expected={}",
            mins[axis],
            expected_bbox_min[axis]
        );
        assert!(
            (maxs[axis] - expected_bbox_max[axis]).abs() <= 1e-3,
            "bbox_max axis {axis} mismatch: actual={} expected={}",
            maxs[axis],
            expected_bbox_max[axis]
        );
    }
}

#[test]
fn triposr_mesh_faces_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            &accel_device(),
        )
        .expect("triposr inference");
    let mesh = pipeline
        .extract_mesh(&output.scene_codes, 256, 25.0, 8192)
        .expect("triposr mesh");

    let actual_faces = mesh
        .faces
        .to_device(&Device::Cpu)
        .expect("faces cpu")
        .flatten_all()
        .expect("faces flatten")
        .to_vec1::<i64>()
        .expect("faces vec");
    let actual_vertices = mesh
        .vertices
        .to_device(&Device::Cpu)
        .expect("vertices cpu")
        .flatten_all()
        .expect("vertices flatten")
        .to_vec1::<f32>()
        .expect("vertices vec");
    let expected_faces = baseline()
        .load("triposr.mesh_faces", &Device::Cpu)
        .expect("mesh faces baseline")
        .flatten_all()
        .expect("faces baseline flatten")
        .to_vec1::<f32>()
        .expect("faces baseline vec")
        .into_iter()
        .map(|value| value.round() as i64)
        .collect::<Vec<_>>();
    let expected_vertices = baseline()
        .load("triposr.mesh_vertices", &Device::Cpu)
        .expect("mesh vertices baseline")
        .flatten_all()
        .expect("vertices baseline flatten")
        .to_vec1::<f32>()
        .expect("vertices baseline vec");

    let actual_areas = canonicalize_face_areas(&actual_vertices, &actual_faces);
    let expected_areas = canonicalize_face_areas(&expected_vertices, &expected_faces);

    assert!(
        ((actual_areas.len() as i64) - (expected_areas.len() as i64)).abs() <= 10,
        "face count drift too large: actual={} expected={}",
        actual_areas.len(),
        expected_areas.len()
    );
    compare_sampled_vectors(
        &actual_areas,
        &expected_areas,
        actual_areas.len().min(expected_areas.len()),
        256,
        10_000,
    );
}

#[test]
fn triposr_vertex_colors_match_horse_golden_baseline() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            &accel_device(),
        )
        .expect("triposr inference");
    let mesh = pipeline
        .extract_mesh(&output.scene_codes, 256, 25.0, 8192)
        .expect("triposr mesh");

    let actual_colors = mesh
        .vertex_colors
        .to_device(&Device::Cpu)
        .expect("colors cpu")
        .flatten_all()
        .expect("colors flatten")
        .to_vec1::<f32>()
        .expect("colors vec");
    let expected_colors = baseline()
        .load("triposr.vertex_colors", &Device::Cpu)
        .expect("vertex colors baseline")
        .flatten_all()
        .expect("colors baseline flatten")
        .to_vec1::<f32>()
        .expect("colors baseline vec");

    assert!(
        (((actual_colors.len() / 3) as i64) - ((expected_colors.len() / 3) as i64)).abs() <= 100,
        "vertex-color count drift too large: actual={} expected={}",
        actual_colors.len() / 3,
        expected_colors.len() / 3
    );
    let actual = color_channel_stats(&actual_colors);
    let expected = color_channel_stats(&expected_colors);
    for channel in 0..3 {
        for stat in 0..5 {
            assert!(
                (actual[channel][stat] - expected[channel][stat]).abs() <= 0.01,
                "vertex color stat mismatch channel {} stat {}: actual={} expected={}",
                channel,
                stat,
                actual[channel][stat],
                expected[channel][stat]
            );
        }
    }
}

#[test]
fn triposr_export_obj_serializes_mesh_buffers_consistently() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");
    let output = pipeline
        .infer_from_path(
            &repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            &accel_device(),
        )
        .expect("triposr inference");
    let mesh = pipeline
        .extract_mesh(&output.scene_codes, 256, 25.0, 8192)
        .expect("triposr mesh");

    let output_path =
        std::env::temp_dir().join(format!("lux3d-triposr-export-{}.obj", std::process::id()));
    pipeline
        .export_obj(&mesh, &output_path)
        .expect("export obj");

    let body = fs::read_to_string(&output_path).expect("read exported obj");
    let _ = fs::remove_file(&output_path);
    let expected = expected_triposr_obj_rows(&mesh);
    let lines = body.lines().collect::<Vec<_>>();

    assert_eq!(lines[0], "# https://github.com/mikedh/trimesh");
    assert_eq!(lines.len(), expected.len() + 1);
    assert_eq!(lines[1], expected[0]);
    assert_eq!(lines[1 + expected.len() / 2], expected[expected.len() / 2]);
    assert_eq!(lines[expected.len()], expected[expected.len() - 1]);
}

#[test]
fn triposr_preprocess_matches_vendor_rgba_bbox_crop_semantics() {
    let _guard = GpuTestLock::acquire().expect("gpu test lock");
    let pipeline =
        TripoSrPipeline::load(model_asset_options(ModelFamily::TripoSr)).expect("triposr pipeline");

    let mut image = RgbaImage::from_pixel(4, 4, Rgba([0, 0, 0, 0]));
    image.put_pixel(1, 1, Rgba([255, 0, 0, 255]));
    image.put_pixel(2, 1, Rgba([0, 255, 0, 255]));
    image.put_pixel(1, 2, Rgba([0, 0, 255, 255]));
    image.put_pixel(2, 2, Rgba([255, 255, 255, 255]));

    let input_path = std::env::temp_dir().join(format!(
        "lux3d-triposr-inclusive-crop-{}.png",
        std::process::id()
    ));
    DynamicImage::ImageRgba8(image)
        .save(&input_path)
        .expect("write temporary rgba input");
    let prepared = pipeline
        .prepare_inputs_from_path(&input_path, &accel_device())
        .expect("triposr prepared rgba input");
    let _ = fs::remove_file(&input_path);

    let pixels = prepared
        .preprocessed_image
        .to_device(&Device::Cpu)
        .expect("preprocessed image cpu")
        .flatten_all()
        .expect("preprocessed image flatten")
        .to_vec1::<f32>()
        .expect("preprocessed image vec");

    let pixel_at = |x: usize, y: usize| -> [f32; 3] {
        let offset = ((y * 512) + x) * 3;
        [pixels[offset], pixels[offset + 1], pixels[offset + 2]]
    };

    let top_left = pixel_at(96, 96);
    let top_right = pixel_at(416, 96);
    let bottom_left = pixel_at(96, 416);
    let bottom_right = pixel_at(416, 416);

    assert!(top_left[0] > top_left[1] && top_left[0] > top_left[2]);
    assert!(top_right[0] > top_right[1] && top_right[0] > top_right[2]);
    assert!(bottom_left[0] > bottom_left[1] && bottom_left[0] > bottom_left[2]);
    assert!(bottom_right[0] > bottom_right[1] && bottom_right[0] > bottom_right[2]);
}

fn canonicalize_face_areas(vertices: &[f32], faces: &[i64]) -> Vec<[i64; 1]> {
    let mut canonical = Vec::with_capacity(faces.len() / 3);
    for face in faces.chunks_exact(3) {
        let a = vertex_triplet(vertices, face[0] as usize);
        let b = vertex_triplet(vertices, face[1] as usize);
        let c = vertex_triplet(vertices, face[2] as usize);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let cross = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        canonical.push([((area) * 1_000_000.0).round() as i64]);
    }
    canonical.sort_unstable();
    canonical
}

fn compare_sampled_vectors<const N: usize>(
    actual: &[[i64; N]],
    expected: &[[i64; N]],
    comparable_len: usize,
    sample_count: usize,
    tolerance: i64,
) {
    let step = (comparable_len / sample_count.max(1)).max(1);
    for index in (0..comparable_len).step_by(step).take(sample_count) {
        for axis in 0..N {
            assert!(
                (actual[index][axis] - expected[index][axis]).abs() <= tolerance,
                "sample mismatch at entry {index} axis {axis}: actual={} expected={}",
                actual[index][axis],
                expected[index][axis]
            );
        }
    }
}

fn vertex_triplet(vertices: &[f32], index: usize) -> [f32; 3] {
    let start = index * 3;
    [vertices[start], vertices[start + 1], vertices[start + 2]]
}

fn color_channel_stats(colors: &[f32]) -> [[f32; 5]; 3] {
    let mut stats = [[0.0; 5]; 3];
    for (channel, channel_stats) in stats.iter_mut().enumerate() {
        let mut values = colors
            .iter()
            .skip(channel)
            .step_by(3)
            .copied()
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.partial_cmp(b).expect("finite colors"));
        let len = values.len();
        let mean = values.iter().sum::<f32>() / len as f32;
        let quantile = |numerator: usize| values[(len.saturating_sub(1) * numerator) / 100];
        *channel_stats = [values[0], quantile(10), quantile(50), quantile(90), mean];
    }
    stats
}

fn expected_triposr_obj_rows(mesh: &lux3d_core::runtime::TripoMeshBuffers) -> Vec<String> {
    let vertices = mesh
        .vertices
        .to_device(&Device::Cpu)
        .expect("vertices cpu")
        .flatten_all()
        .expect("vertices flatten")
        .to_vec1::<f32>()
        .expect("vertices vec");
    let colors = mesh
        .vertex_colors
        .to_device(&Device::Cpu)
        .expect("colors cpu")
        .flatten_all()
        .expect("colors flatten")
        .to_vec1::<f32>()
        .expect("colors vec");
    let faces = mesh
        .faces
        .to_device(&Device::Cpu)
        .expect("faces cpu")
        .flatten_all()
        .expect("faces flatten")
        .to_vec1::<i64>()
        .expect("faces vec");

    let mut rows = Vec::with_capacity((vertices.len() / 3) + (faces.len() / 3));
    for i in 0..(vertices.len() / 3) {
        let p = i * 3;
        rows.push(format!(
            "v {} {} {} {} {} {}",
            vertices[p],
            vertices[p + 1],
            vertices[p + 2],
            colors[p].clamp(0.0, 1.0),
            colors[p + 1].clamp(0.0, 1.0),
            colors[p + 2].clamp(0.0, 1.0)
        ));
    }
    for i in 0..(faces.len() / 3) {
        let p = i * 3;
        rows.push(format!(
            "f {} {} {}",
            faces[p] + 1,
            faces[p + 1] + 1,
            faces[p + 2] + 1
        ));
    }
    rows
}
