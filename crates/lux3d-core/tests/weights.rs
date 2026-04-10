use std::{
    fs,
    path::{Path, PathBuf},
};

use lux3d_core::test_support::{resolve_raw_model_dir_for_tests, runtime_root};
use lux3d_core::{
    CanonicalizationPlan, FutureWeightLoader, ModelAssetOptions, ModelFamily, WeightLocator,
    ensure_canonical_weights, load_canonical_weights,
};

fn required_raw_filenames(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Pi3 => &["model.safetensors"],
        ModelFamily::Pi3x => &["model.safetensors", "config.json"],
        ModelFamily::TripoSr => &["model.ckpt", "config.yaml"],
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
    let repo_root = runtime_root();
    let family_name = family.as_str();
    let python = repo_root
        .join("3d")
        .join("_generated")
        .join("python-envs")
        .join(family_name)
        .join("Scripts")
        .join("python.exe");
    let script = repo_root
        .join("3d-rs")
        .join("tools")
        .join("python_baseline")
        .join("normalize_weights.py");

    let output = std::process::Command::new(&python)
        .arg(&script)
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
