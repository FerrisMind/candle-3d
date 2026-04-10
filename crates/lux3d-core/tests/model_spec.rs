use std::path::PathBuf;

use lux3d_core::test_support::runtime_root;
use lux3d_core::{
    ExportSpec, FutureWeightLoader, GeometrySpec, ModelFamily, ModelSpec, NeuralSpec,
    PreprocessSpec, RuntimeGeometry, SourceDisposition,
};

fn assert_path_free_weight_plan(
    spec: &ModelSpec,
    family: ModelFamily,
    expected_raw_files: &[&str],
) {
    assert_eq!(
        expected_raw_files
            .iter()
            .map(|name| PathBuf::from(name))
            .collect::<Vec<_>>(),
        spec.weight_plan.raw_files.to_vec()
    );
    assert_eq!(
        PathBuf::from(family.as_str()),
        spec.weight_plan.canonical_root
    );
    for path in spec
        .weight_plan
        .raw_files
        .iter()
        .chain(std::iter::once(&spec.weight_plan.canonical_root))
    {
        let rendered = path.to_string_lossy();
        for forbidden in [
            "3d/models",
            "3d/canonical-weights",
            "yyfz233-Pi3",
            "yyfz233-Pi3X",
            "stabilityai-TripoSR",
        ] {
            assert!(
                !rendered.contains(forbidden),
                "weight plan must stay path-free, found `{forbidden}` in `{rendered}`"
            );
        }
    }
}

#[test]
fn pi3_model_spec_is_source_truth_backed() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::Pi3).expect("pi3 model spec");

    match &spec.preprocess {
        PreprocessSpec::Pi3(preprocess) => {
            assert_eq!(255_000, preprocess.pixel_limit);
            assert_eq!(14, preprocess.patch_multiple);
            assert_eq!(1, preprocess.image_default_interval);
            assert_eq!(10, preprocess.video_default_interval);
        }
        other => panic!("expected Pi3 preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::Pi3(neural) => {
            assert_eq!("dinov2_vitl14_reg", neural.encoder_backbone);
            assert_eq!("rope100", neural.positional_encoding);
            assert_eq!("large", neural.decoder_size);
            assert_eq!(36, neural.decoder_depth);
        }
        other => panic!("expected Pi3 neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::Pi3(geometry) => {
            assert_eq!("point_cloud", geometry.artifact_kind);
            assert!(!geometry.supports_marching_cubes);
            assert!(!geometry.supports_sim3_fusion_extension);
        }
        other => panic!("expected Pi3 geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::Pi3(export) => {
            assert_eq!("ply", export.primary_extension);
            assert!(export.alternate_extensions.is_empty());
        }
        other => panic!("expected Pi3 export spec, got {other:?}"),
    }

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_path_free_weight_plan(&spec, ModelFamily::Pi3, &["model.safetensors"]);
    assert!(
        spec.license_policy
            .entries
            .iter()
            .any(|entry| entry.disposition == SourceDisposition::ReferenceOnly)
    );
    assert_eq!("house-golden", spec.golden_baseline_sample);
    assert!(
        spec.exclusions
            .iter()
            .any(|entry| entry.item == "pi3x_multimodal_conditioning")
    );
}

#[test]
fn pi3x_model_spec_is_source_truth_backed() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::Pi3x).expect("pi3x model spec");

    match &spec.preprocess {
        PreprocessSpec::Pi3x(preprocess) => {
            assert_eq!(255_000, preprocess.pixel_limit);
            assert_eq!(14, preprocess.patch_multiple);
            assert!(preprocess.supports_conditions_npz);
        }
        other => panic!("expected Pi3x preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::Pi3x(neural) => {
            assert_eq!("dinov2_vitl14_reg", neural.encoder_backbone);
            assert_eq!("rope100 + projective_rope", neural.positional_encoding);
            assert_eq!(5, neural.pose_inject_blocks);
        }
        other => panic!("expected Pi3x neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::Pi3x(geometry) => {
            assert_eq!("point_cloud", geometry.artifact_kind);
            assert!(geometry.supports_sim3_fusion_extension);
        }
        other => panic!("expected Pi3x geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::Pi3x(export) => {
            assert_eq!("ply", export.primary_extension);
            assert!(export.alternate_extensions.is_empty());
        }
        other => panic!("expected Pi3x export spec, got {other:?}"),
    }

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_path_free_weight_plan(
        &spec,
        ModelFamily::Pi3x,
        &["model.safetensors", "config.json"],
    );
    assert_eq!("room-golden", spec.golden_baseline_sample);
    assert!(
        spec.notes
            .iter()
            .any(|note| note.contains("use_multimodal=true"))
    );
}

#[test]
fn triposr_model_spec_is_source_truth_backed() {
    let spec =
        ModelSpec::inspect(runtime_root(), ModelFamily::TripoSr).expect("triposr model spec");

    match &spec.preprocess {
        PreprocessSpec::TripoSr(preprocess) => {
            assert_eq!(512, preprocess.target_size.width);
            assert_eq!(512, preprocess.target_size.height);
            assert_eq!(0.5, preprocess.background_value);
            assert!(preprocess.optional_rgba_compositing);
        }
        other => panic!("expected TripoSR preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::TripoSr(neural) => {
            assert_eq!("facebook/dino-vitb16", neural.image_tokenizer);
            assert_eq!(16, neural.transformer_layers);
            assert_eq!(128, neural.samples_per_ray);
            assert_eq!(0.87, neural.radius);
        }
        other => panic!("expected TripoSR neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::TripoSr(geometry) => {
            assert_eq!("marching_cubes", geometry.surface_extractor);
            assert_eq!(256, geometry.default_resolution);
            assert!(!geometry.supports_texture_baking_extension);
        }
        other => panic!("expected TripoSR geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::TripoSr(export) => {
            assert_eq!("obj", export.primary_extension);
            assert!(export.alternate_extensions.is_empty());
        }
        other => panic!("expected TripoSR export spec, got {other:?}"),
    }

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_path_free_weight_plan(&spec, ModelFamily::TripoSr, &["model.ckpt", "config.yaml"]);
    assert_eq!("horse-golden", spec.golden_baseline_sample);

    let RuntimeGeometry::TripoSr(runtime) = &spec.runtime_geometry else {
        panic!("expected triposr runtime geometry");
    };
    assert_eq!(256, runtime.default_resolution);
    assert_eq!(25.0, runtime.threshold);
}

#[test]
fn license_policy_covers_every_used_source_file() {
    for family in [ModelFamily::Pi3, ModelFamily::Pi3x, ModelFamily::TripoSr] {
        let spec = ModelSpec::inspect(runtime_root(), family).expect("source-backed model spec");

        for source_path in spec.used_source_paths().iter() {
            assert!(
                spec.license_policy.covers_source(source_path),
                "missing license classification for {source_path}"
            );
        }
    }
}

#[test]
fn vendor_sources_have_fingerprints_and_license_entries() {
    for family in [ModelFamily::Pi3, ModelFamily::Pi3x, ModelFamily::TripoSr] {
        let spec = ModelSpec::inspect(runtime_root(), family).expect("source-backed model spec");

        for vendor in &spec.vendor_sources {
            assert!(
                vendor.sha256.len() == 64 && vendor.sha256.chars().all(|ch| ch.is_ascii_hexdigit()),
                "invalid sha256 for {}",
                vendor.source_path
            );
            assert!(
                spec.license_policy.covers_source(&vendor.source_path),
                "missing license entry for {}",
                vendor.source_path
            );
        }
    }
}
