use std::path::PathBuf;

use lux3d_core::{
    CanonicalizationPlan, FutureWeightLoader, ModelFamily, WeightLocator, load_canonical_weights,
};

#[test]
fn pi3_locator_finds_raw_safetensors_and_canonical_target() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let weights: CanonicalizationPlan = locator.locate(ModelFamily::Pi3).expect("pi3 weights");

    assert_eq!(ModelFamily::Pi3, weights.family);
    assert!(weights.raw_files[0].ends_with(r"3d\models\yyfz233-Pi3\model.safetensors"));
    assert!(
        weights
            .canonical_root
            .ends_with(r"3d\canonical-weights\pi3")
    );
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
}

#[test]
fn pi3x_locator_finds_raw_safetensors_and_canonical_target() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let weights: CanonicalizationPlan = locator.locate(ModelFamily::Pi3x).expect("pi3x weights");

    assert_eq!(ModelFamily::Pi3x, weights.family);
    assert!(weights.raw_files[0].ends_with(r"3d\models\yyfz233-Pi3X\model.safetensors"));
    assert!(weights.raw_files[1].ends_with(r"3d\models\yyfz233-Pi3X\config.json"));
    assert!(
        weights
            .canonical_root
            .ends_with(r"3d\canonical-weights\pi3x")
    );
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
}

#[test]
fn triposr_locator_finds_raw_checkpoint_and_canonical_target() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let weights = locator
        .locate(ModelFamily::TripoSr)
        .expect("triposr weights");

    assert!(weights.raw_files[0].ends_with(r"3d\models\stabilityai-TripoSR\model.ckpt"));
    assert!(
        weights
            .canonical_root
            .ends_with(r"3d\canonical-weights\triposr")
    );
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
}

#[test]
fn pi3_canonical_weights_are_required_before_runtime_load() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let mut plan = locator.locate(ModelFamily::Pi3).expect("pi3 weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = lux3d_core::ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
}

#[test]
fn pi3x_canonical_weights_are_required_before_runtime_load() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let mut plan = locator.locate(ModelFamily::Pi3x).expect("pi3x weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = lux3d_core::ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
}

#[test]
fn triposr_canonical_weights_are_required_before_runtime_load() {
    let locator = WeightLocator::new(PathBuf::from(r"H:\GitHub\LuxRT"));
    let mut plan = locator
        .locate(ModelFamily::TripoSr)
        .expect("triposr weights");
    plan.canonical_root = plan.canonical_root.join("__missing_test__");

    let manifest = lux3d_core::ensure_canonical_weights(&plan);
    assert!(
        manifest.is_err(),
        "canonical weights should be missing before the canonical normalizer runs"
    );
}

#[test]
fn pi3_python_normalizer_produces_loadable_canonical_weights() {
    run_python_normalizer("pi3");

    let weights = load_canonical_weights(ModelFamily::Pi3, PathBuf::from(r"H:\GitHub\LuxRT"))
        .expect("pi3 canonical weights");
    assert_eq!(1210, weights.manifest.tensor_count);
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
}

#[test]
fn pi3x_python_normalizer_produces_loadable_canonical_weights() {
    run_python_normalizer("pi3x");

    let weights = load_canonical_weights(ModelFamily::Pi3x, PathBuf::from(r"H:\GitHub\LuxRT"))
        .expect("pi3x canonical weights");
    assert_eq!(1873, weights.manifest.tensor_count);
    assert_eq!(2, weights.manifest.raw_files.len());
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
}

#[test]
fn triposr_python_normalizer_produces_loadable_canonical_weights() {
    run_python_normalizer("triposr");

    let weights = load_canonical_weights(ModelFamily::TripoSr, PathBuf::from(r"H:\GitHub\LuxRT"))
        .expect("triposr canonical weights");
    assert_eq!(549, weights.manifest.tensor_count);
    assert_eq!(2, weights.manifest.raw_files.len());
    assert!(weights.canonical_file.is_file());
    assert!(weights.resolved_config_file.is_file());
}

#[test]
fn pi3_python_normalizer_is_idempotent() {
    run_python_normalizer("pi3");
    let repo_root = PathBuf::from(r"H:\GitHub\LuxRT");
    let manifest_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("pi3")
        .join("manifest.json");
    let checksums_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("pi3")
        .join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("pi3 manifest before");
    let checksums_before = std::fs::read_to_string(&checksums_path).expect("pi3 checksums before");
    run_python_normalizer("pi3");
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("pi3 manifest after");
    let checksums_after = std::fs::read_to_string(&checksums_path).expect("pi3 checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
}

#[test]
fn pi3x_python_normalizer_is_idempotent() {
    run_python_normalizer("pi3x");
    let repo_root = PathBuf::from(r"H:\GitHub\LuxRT");
    let manifest_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("pi3x")
        .join("manifest.json");
    let checksums_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("pi3x")
        .join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("pi3x manifest before");
    let checksums_before = std::fs::read_to_string(&checksums_path).expect("pi3x checksums before");
    run_python_normalizer("pi3x");
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("pi3x manifest after");
    let checksums_after = std::fs::read_to_string(&checksums_path).expect("pi3x checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
}

#[test]
fn triposr_python_normalizer_is_idempotent() {
    run_python_normalizer("triposr");
    let repo_root = PathBuf::from(r"H:\GitHub\LuxRT");
    let manifest_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("triposr")
        .join("manifest.json");
    let checksums_path = repo_root
        .join("3d")
        .join("canonical-weights")
        .join("triposr")
        .join("checksums.json");

    let manifest_before = std::fs::read_to_string(&manifest_path).expect("triposr manifest before");
    let checksums_before =
        std::fs::read_to_string(&checksums_path).expect("triposr checksums before");
    run_python_normalizer("triposr");
    let manifest_after = std::fs::read_to_string(&manifest_path).expect("triposr manifest after");
    let checksums_after =
        std::fs::read_to_string(&checksums_path).expect("triposr checksums after");

    assert_eq!(manifest_before, manifest_after);
    assert_eq!(checksums_before, checksums_after);
}

fn run_python_normalizer(family: &str) {
    let repo_root = PathBuf::from(r"H:\GitHub\LuxRT");
    let python = repo_root
        .join("3d")
        .join("_generated")
        .join("python-envs")
        .join(family)
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
        .arg(family)
        .arg("--repo-root")
        .arg(&repo_root)
        .output()
        .expect("python normalizer process");

    assert!(
        output.status.success(),
        "python normalizer failed for `{family}`:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}
