use std::{
    fs,
    path::{Path, PathBuf},
};

use lux3d_core::test_support::{resolve_raw_model_dir_for_tests, workspace_root};
use lux3d_core::{
    CanonicalizationPlan, FutureWeightLoader, ModelAssetOptions, ModelFamily, WeightLocator,
    ensure_canonical_weights, load_canonical_weights,
};

fn required_raw_filenames(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Pi3 => &["model.safetensors"],
        ModelFamily::Pi3x => &["model.safetensors", "config.json"],
        ModelFamily::TripoSr => &["model.ckpt", "config.yaml"],
        ModelFamily::Vggt => &["model.safetensors", "config.json"],
    }
}

fn temp_fixture_root(label: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("current time")
        .as_nanos();
    std::env::temp_dir().join(format!("lux3d-{label}-{unique}"))
}

fn create_raw_fixture(family: ModelFamily) -> (PathBuf, PathBuf, PathBuf) {
    let root = temp_fixture_root(family.as_str());
    let raw_dir = root.join("raw");
    let canonical_dir = root.join("canonical").join(family.as_str());
    fs::create_dir_all(&raw_dir).expect("raw fixture dir");
    for filename in required_raw_filenames(family) {
        fs::write(raw_dir.join(filename), b"fixture").expect("raw fixture file");
    }
    (root, raw_dir, canonical_dir)
}

fn cleanup_dir(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

fn temp_output_dir(family: ModelFamily) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("current time")
        .as_nanos();
    std::env::temp_dir().join(format!("lux3d-normalizer-{}-{unique}", family.as_str()))
}

#[test]
fn vggt_canonical_package_repo_is_the_oxide_lab_mirror() {
    // The public `facebook/VGGT-1B` repository only ships the raw checkpoint;
    // the canonical four-file package lives in the oxide-lab mirror.
    assert_eq!("oxide-lab/VGGT-1B", ModelFamily::Vggt.huggingface_repo_id());
}

#[test]
fn pi3_locator_finds_raw_safetensors_and_canonical_target() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Pi3);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights: CanonicalizationPlan = locator.locate(ModelFamily::Pi3).expect("pi3 weights");

    assert_eq!(ModelFamily::Pi3, weights.family);
    assert_eq!(
        vec![raw_model_dir.join("model.safetensors")],
        weights.raw_files.to_vec()
    );
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    cleanup_dir(&root);
}

#[test]
fn pi3x_locator_finds_raw_safetensors_and_canonical_target() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Pi3x);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights: CanonicalizationPlan = locator.locate(ModelFamily::Pi3x).expect("pi3x weights");

    assert_eq!(ModelFamily::Pi3x, weights.family);
    assert_eq!(
        vec![
            raw_model_dir.join("model.safetensors"),
            raw_model_dir.join("config.json")
        ],
        weights.raw_files.to_vec()
    );
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    cleanup_dir(&root);
}

#[test]
fn vggt_locator_finds_raw_safetensors_and_canonical_target() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Vggt);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights: CanonicalizationPlan = locator.locate(ModelFamily::Vggt).expect("vggt weights");

    assert_eq!(ModelFamily::Vggt, weights.family);
    assert_eq!(
        vec![
            raw_model_dir.join("model.safetensors"),
            raw_model_dir.join("config.json")
        ],
        weights.raw_files.to_vec()
    );
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    cleanup_dir(&root);
}

fn create_vggt_safetensors_fixture(raw_dir: &Path) -> Vec<String> {
    fs::create_dir_all(raw_dir).expect("vggt raw fixture dir");
    let value = 1.0f32.to_le_bytes();
    let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &value)
        .expect("vggt fixture tensor view");
    safetensors::serialize_to_file(
        [("aggregator.patch_embed.cls_token".to_string(), view)],
        None,
        &raw_dir.join("model.safetensors"),
    )
    .expect("write vggt safetensors fixture");
    fs::write(
        raw_dir.join("config.json"),
        br#"{"embed_dim": 1024, "img_size": 518, "patch_size": 14}"#,
    )
    .expect("write vggt config fixture");
    vec!["model.safetensors".to_string(), "config.json".to_string()]
}

#[test]
fn vggt_python_normalizer_produces_loadable_canonical_weights() {
    let root = temp_fixture_root("vggt-normalizer");
    let raw_model_dir = root.join("raw");
    let output_dir = root.join("canonical").join("vggt");
    let raw_files = create_vggt_safetensors_fixture(&raw_model_dir);

    run_python_normalizer(ModelFamily::Vggt, &raw_model_dir, &output_dir);

    let manifest_path = output_dir.join("manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&manifest_path).expect("read vggt manifest"))
            .expect("parse vggt manifest");
    assert_eq!("vggt", manifest["family"]);
    assert_eq!(serde_json::json!(raw_files), manifest["raw_files"]);
    assert_eq!("model.safetensors", manifest["canonical_file"]);
    assert_eq!("resolved_config.json", manifest["resolved_config_file"]);
    assert_eq!(1, manifest["tensor_count"]);

    let checksums_path = output_dir.join("checksums.json");
    let checksums: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&checksums_path).expect("read vggt checksums"))
            .expect("parse vggt checksums");
    let mut names = checksums["files"]
        .as_array()
        .expect("checksums files array")
        .iter()
        .map(|entry| {
            assert_eq!(64, entry["sha256"].as_str().expect("sha256").len());
            assert!(entry["size_bytes"].as_u64().expect("size_bytes") > 0);
            entry["relative_path"].as_str().expect("relative_path")
        })
        .collect::<Vec<_>>();
    names.sort_unstable();
    assert_eq!(
        vec!["manifest.json", "model.safetensors", "resolved_config.json"],
        names
    );

    let locator = WeightLocator::new(raw_model_dir, output_dir);
    let plan = locator.locate(ModelFamily::Vggt).expect("vggt weights");
    let canonical = ensure_canonical_weights(&plan).expect("vggt canonical package");
    assert_eq!(ModelFamily::Vggt, canonical.manifest.family);
    assert_eq!(raw_files.as_slice(), canonical.manifest.raw_files.as_ref());
    assert_eq!(1, canonical.manifest.tensor_count);
    assert!(canonical.canonical_file.is_file());
    assert!(canonical.resolved_config_file.is_file());

    cleanup_dir(&root);
}

#[test]
fn triposr_locator_finds_raw_checkpoint_and_canonical_target() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::TripoSr);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights = locator
        .locate(ModelFamily::TripoSr)
        .expect("triposr weights");

    assert_eq!(
        vec![
            raw_model_dir.join("model.ckpt"),
            raw_model_dir.join("config.yaml")
        ],
        weights.raw_files.to_vec()
    );
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    cleanup_dir(&root);
}

#[test]
fn pi3_canonical_weights_are_required_before_runtime_load() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Pi3);
    let locator = WeightLocator::new(raw_model_dir, canonical_dir);
    let mut plan = locator.locate(ModelFamily::Pi3).expect("pi3 weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
    cleanup_dir(&root);
}

#[test]
fn pi3x_canonical_weights_are_required_before_runtime_load() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Pi3x);
    let locator = WeightLocator::new(raw_model_dir, canonical_dir);
    let mut plan = locator.locate(ModelFamily::Pi3x).expect("pi3x weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
    cleanup_dir(&root);
}

#[test]
fn triposr_canonical_weights_are_required_before_runtime_load() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::TripoSr);
    let locator = WeightLocator::new(raw_model_dir, canonical_dir);
    let mut plan = locator
        .locate(ModelFamily::TripoSr)
        .expect("triposr weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
    cleanup_dir(&root);
}

#[test]
fn pi3_python_normalizer_produces_loadable_canonical_weights() {
    let raw_model_dir =
        resolve_raw_model_dir_for_tests(ModelFamily::Pi3).expect("resolve pi3 raw model dir");
    let output_dir = temp_output_dir(ModelFamily::Pi3);
    run_python_normalizer(ModelFamily::Pi3, &raw_model_dir, &output_dir);

    let weights = load_canonical_weights(
        ModelFamily::Pi3,
        ModelAssetOptions {
            canonical_dir: Some(output_dir.clone()),
            cache_dir: None,
        },
    )
    .expect("pi3 canonical weights");
    assert_eq!(1210, weights.manifest.tensor_count);
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn pi3x_python_normalizer_produces_loadable_canonical_weights() {
    let raw_model_dir =
        resolve_raw_model_dir_for_tests(ModelFamily::Pi3x).expect("resolve pi3x raw model dir");
    let output_dir = temp_output_dir(ModelFamily::Pi3x);
    run_python_normalizer(ModelFamily::Pi3x, &raw_model_dir, &output_dir);

    let weights = load_canonical_weights(
        ModelFamily::Pi3x,
        ModelAssetOptions {
            canonical_dir: Some(output_dir.clone()),
            cache_dir: None,
        },
    )
    .expect("pi3x canonical weights");
    assert_eq!(1873, weights.manifest.tensor_count);
    assert_eq!(2, weights.manifest.raw_files.len());
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn triposr_python_normalizer_produces_loadable_canonical_weights() {
    let raw_model_dir = resolve_raw_model_dir_for_tests(ModelFamily::TripoSr)
        .expect("resolve triposr raw model dir");
    let output_dir = temp_output_dir(ModelFamily::TripoSr);
    run_python_normalizer(ModelFamily::TripoSr, &raw_model_dir, &output_dir);

    let weights = load_canonical_weights(
        ModelFamily::TripoSr,
        ModelAssetOptions {
            canonical_dir: Some(output_dir.clone()),
            cache_dir: None,
        },
    )
    .expect("triposr canonical weights");
    assert_eq!(549, weights.manifest.tensor_count);
    assert_eq!(2, weights.manifest.raw_files.len());
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn pi3_python_normalizer_is_idempotent() {
    let raw_model_dir =
        resolve_raw_model_dir_for_tests(ModelFamily::Pi3).expect("resolve pi3 raw model dir");
    let output_dir = temp_output_dir(ModelFamily::Pi3);
    run_python_normalizer(ModelFamily::Pi3, &raw_model_dir, &output_dir);
    let manifest_path = output_dir.join("manifest.json");
    let checksums_path = output_dir.join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("pi3 manifest before");
    let checksums_before = std::fs::read_to_string(&checksums_path).expect("pi3 checksums before");
    run_python_normalizer(ModelFamily::Pi3, &raw_model_dir, &output_dir);
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("pi3 manifest after");
    let checksums_after = std::fs::read_to_string(&checksums_path).expect("pi3 checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn pi3x_python_normalizer_is_idempotent() {
    let raw_model_dir =
        resolve_raw_model_dir_for_tests(ModelFamily::Pi3x).expect("resolve pi3x raw model dir");
    let output_dir = temp_output_dir(ModelFamily::Pi3x);
    run_python_normalizer(ModelFamily::Pi3x, &raw_model_dir, &output_dir);
    let manifest_path = output_dir.join("manifest.json");
    let checksums_path = output_dir.join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("pi3x manifest before");
    let checksums_before = std::fs::read_to_string(&checksums_path).expect("pi3x checksums before");
    run_python_normalizer(ModelFamily::Pi3x, &raw_model_dir, &output_dir);
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("pi3x manifest after");
    let checksums_after = std::fs::read_to_string(&checksums_path).expect("pi3x checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
    let _ = std::fs::remove_dir_all(output_dir);
}

#[test]
fn triposr_python_normalizer_is_idempotent() {
    let raw_model_dir = resolve_raw_model_dir_for_tests(ModelFamily::TripoSr)
        .expect("resolve triposr raw model dir");
    let output_dir = temp_output_dir(ModelFamily::TripoSr);
    run_python_normalizer(ModelFamily::TripoSr, &raw_model_dir, &output_dir);
    let manifest_path = output_dir.join("manifest.json");
    let checksums_path = output_dir.join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("triposr manifest before");
    let checksums_before =
        std::fs::read_to_string(&checksums_path).expect("triposr checksums before");
    run_python_normalizer(ModelFamily::TripoSr, &raw_model_dir, &output_dir);
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("triposr manifest after");
    let checksums_after =
        std::fs::read_to_string(&checksums_path).expect("triposr checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
    let _ = std::fs::remove_dir_all(output_dir);
}

fn run_python_normalizer(family: ModelFamily, raw_model_dir: &Path, output_dir: &PathBuf) {
    let repo_root = workspace_root();
    let family_name = family.as_str();
    let script = repo_root
        .join("tools")
        .join("python_baseline")
        .join("normalize_weights.py");

    let output = std::process::Command::new("python")
        .arg("-I")
        .arg(&script)
        .arg("--repo-root")
        .arg(&repo_root)
        .arg("--family")
        .arg(family_name)
        .arg("--raw-model-dir")
        .arg(raw_model_dir)
        .arg("--output-dir")
        .arg(output_dir)
        .output()
        .expect("python normalizer process");

    assert!(
        output.status.success(),
        "python normalizer failed for `{family_name}`:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
