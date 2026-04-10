use std::path::PathBuf;

use lux3d_core::{ModelAssetOptions, ModelFamily, load_canonical_weights};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

fn runtime_root() -> PathBuf {
    workspace_root()
        .parent()
        .expect("runtime root")
        .to_path_buf()
}

#[test]
fn loads_pi3_from_explicit_canonical_package_dir() {
    let canonical_dir = runtime_root()
        .join("3d")
        .join("canonical-weights")
        .join("pi3");

    let weights = load_canonical_weights(
        ModelFamily::Pi3,
        ModelAssetOptions {
            canonical_dir: Some(canonical_dir.clone()),
            cache_dir: None,
        },
    )
    .expect("pi3 canonical package");

    assert_eq!(
        canonical_dir.join("model.safetensors"),
        weights.canonical_file
    );
    assert_eq!(
        canonical_dir.join("resolved_config.json"),
        weights.resolved_config_file
    );
}

#[test]
fn supports_cache_dir_override_for_auto_resolved_packages() {
    let cache_dir = std::env::temp_dir().join("lux3d-test-cache-contract");

    let options = ModelAssetOptions {
        canonical_dir: None,
        cache_dir: Some(cache_dir.clone()),
    };

    assert_eq!(Some(cache_dir), options.cache_dir);
    assert!(options.canonical_dir.is_none());
}

#[test]
fn reuses_existing_cached_package_without_explicit_model_path() {
    let cache_root = std::env::temp_dir().join(format!("lux3d-cache-reuse-{}", std::process::id()));
    let package_dir = cache_root.join("pi3");
    std::fs::create_dir_all(&package_dir).expect("cache package dir");

    for filename in [
        "model.safetensors",
        "manifest.json",
        "checksums.json",
        "resolved_config.json",
    ] {
        std::fs::copy(
            runtime_root()
                .join("3d")
                .join("canonical-weights")
                .join("pi3")
                .join(filename),
            package_dir.join(filename),
        )
        .expect("copy cached asset");
    }

    let weights = load_canonical_weights(
        ModelFamily::Pi3,
        ModelAssetOptions {
            canonical_dir: None,
            cache_dir: Some(cache_root.clone()),
        },
    )
    .expect("cached pi3 package");

    assert_eq!(
        package_dir.join("model.safetensors"),
        weights.canonical_file
    );
    let _ = std::fs::remove_dir_all(cache_root);
}
