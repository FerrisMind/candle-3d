use std::{collections::BTreeSet, fs, path::PathBuf};

use lux3d_core::{BaselineManifest, BaselineSampleKind, ModelFamily, ModelSpec};
use serde_json::Value;

fn repo_root() -> PathBuf {
    PathBuf::from(r"H:\GitHub\LuxRT")
}

fn baseline_manifest_path(family: &str, sample_id: &str) -> PathBuf {
    repo_root()
        .join("3d")
        .join("baselines")
        .join(family)
        .join(sample_id)
        .join("manifest.json")
}

fn summary_path(family: &str, sample_id: &str) -> PathBuf {
    repo_root()
        .join("3d")
        .join("baselines")
        .join(family)
        .join(sample_id)
        .join("summary.json")
}

fn load_manifest(family: &str, sample_id: &str) -> BaselineManifest {
    let path = baseline_manifest_path(family, sample_id);
    BaselineManifest::from_path(&path)
        .unwrap_or_else(|err| panic!("failed to read {path:?}: {err}"))
}

#[test]
fn golden_manifests_cover_declared_parity_taps() {
    let pi3 = load_manifest("pi3", "house-golden");
    let pi3_spec = ModelSpec::inspect(repo_root(), ModelFamily::Pi3).expect("pi3 spec");
    pi3.validate_against_model_spec(&pi3_spec)
        .expect("pi3 golden manifest validation");

    let pi3x = load_manifest("pi3x", "room-golden");
    let pi3x_spec = ModelSpec::inspect(repo_root(), ModelFamily::Pi3x).expect("pi3x spec");
    pi3x.validate_against_model_spec(&pi3x_spec)
        .expect("pi3x golden manifest validation");

    let triposr = load_manifest("triposr", "horse-golden");
    let triposr_spec = ModelSpec::inspect(repo_root(), ModelFamily::TripoSr).expect("triposr spec");
    triposr
        .validate_against_model_spec(&triposr_spec)
        .expect("triposr golden manifest validation");
}

#[test]
fn golden_tensors_match_contract_dtypes_and_ranks() {
    for (family, sample_id, model_family) in [
        ("pi3", "house-golden", ModelFamily::Pi3),
        ("pi3x", "room-golden", ModelFamily::Pi3x),
        ("triposr", "horse-golden", ModelFamily::TripoSr),
    ] {
        let manifest = load_manifest(family, sample_id);
        let spec = ModelSpec::inspect(repo_root(), model_family).expect("model spec");
        let parity_taps = spec
            .baseline_parity_taps
            .iter()
            .map(|tap| (tap.name.as_str(), tap))
            .collect::<std::collections::BTreeMap<_, _>>();

        for artifact in manifest.tensor_artifacts.iter() {
            if let Some(tap) = parity_taps.get(artifact.name.as_str()) {
                assert_eq!(tap.semantic, artifact.semantic);
                assert_eq!(tap.storage_dtype, artifact.dtype);
                assert_eq!(tap.observed_shape.as_slice(), artifact.shape.as_ref());
            }
        }
    }
}

#[test]
fn smoke_manifests_are_summary_only() {
    let pi3 = load_manifest("pi3", "skating-smoke");
    assert_eq!(BaselineSampleKind::Smoke, pi3.sample_kind);
    assert!(pi3.tensor_artifacts.is_empty());

    let triposr = load_manifest("triposr", "chair-smoke");
    assert_eq!(BaselineSampleKind::Smoke, triposr.sample_kind);
    assert!(triposr.tensor_artifacts.is_empty());

    let pi3_summary: Value = serde_json::from_str(
        &fs::read_to_string(summary_path("pi3", "skating-smoke")).expect("pi3 smoke summary"),
    )
    .expect("pi3 smoke summary json");
    let triposr_summary: Value = serde_json::from_str(
        &fs::read_to_string(summary_path("triposr", "chair-smoke")).expect("triposr smoke summary"),
    )
    .expect("triposr smoke summary json");

    let pi3_keys = pi3_summary
        .as_object()
        .expect("pi3 smoke summary object")
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();
    let triposr_keys = triposr_summary
        .as_object()
        .expect("triposr smoke summary object")
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>();

    let allowed_pi3 = BTreeSet::from([
        "bbox_max".to_string(),
        "bbox_min".to_string(),
        "device".to_string(),
        "family".to_string(),
        "first_color_samples".to_string(),
        "first_point_samples".to_string(),
        "interval".to_string(),
        "mask_true_count".to_string(),
        "point_count".to_string(),
        "sample_id".to_string(),
        "sample_kind".to_string(),
        "sampled_frames".to_string(),
        "source_path".to_string(),
        "target_size".to_string(),
        "weight_files".to_string(),
    ]);
    let allowed_triposr = BTreeSet::from([
        "bbox_max".to_string(),
        "bbox_min".to_string(),
        "device".to_string(),
        "face_count".to_string(),
        "family".to_string(),
        "first_color_samples".to_string(),
        "first_vertex_samples".to_string(),
        "mc_resolution".to_string(),
        "mc_threshold".to_string(),
        "sample_id".to_string(),
        "sample_kind".to_string(),
        "scene_codes_shape".to_string(),
        "source_path".to_string(),
        "vertex_count".to_string(),
        "weight_files".to_string(),
    ]);

    assert_eq!(allowed_pi3, pi3_keys);
    assert_eq!(allowed_triposr, triposr_keys);
}

#[test]
fn manifests_reference_expected_local_weight_files() {
    for (family, sample_id, model_family) in [
        ("pi3", "house-golden", ModelFamily::Pi3),
        ("pi3", "skating-smoke", ModelFamily::Pi3),
        ("pi3x", "room-golden", ModelFamily::Pi3x),
        ("pi3x", "skating-vo-golden", ModelFamily::Pi3x),
        ("pi3x", "skating-vo-inject-golden", ModelFamily::Pi3x),
        ("triposr", "horse-golden", ModelFamily::TripoSr),
        ("triposr", "chair-smoke", ModelFamily::TripoSr),
    ] {
        let manifest = load_manifest(family, sample_id);
        let spec = ModelSpec::inspect(repo_root(), model_family).expect("model spec");
        assert_eq!(
            spec.weight_plan.raw_files.as_ref(),
            manifest.weight_files.as_ref()
        );
    }
}

#[test]
fn pi3x_vo_manifests_have_expected_artifacts() {
    for sample_id in ["skating-vo-golden", "skating-vo-inject-golden"] {
        let manifest = load_manifest("pi3x", sample_id);
        assert_eq!(BaselineSampleKind::Golden, manifest.sample_kind);
        assert_eq!(ModelFamily::Pi3x, manifest.family);
        assert_eq!(2, manifest.weight_files.len());
        let artifact_names = manifest
            .tensor_artifacts
            .iter()
            .map(|artifact| artifact.name.as_str())
            .collect::<BTreeSet<_>>();
        let expected = BTreeSet::from([
            "pi3xvo.points",
            "pi3xvo.camera_poses",
            "pi3xvo.confidence_logits",
            "pi3xvo.sim3_transforms",
        ]);
        assert_eq!(expected, artifact_names);
    }
}
