use std::{
    fs,
    path::{Path, PathBuf},
};

use lux3d_core::test_support::{resolve_raw_model_dir_for_tests, workspace_root};
use lux3d_core::{
    CanonicalizationPlan, FutureWeightLoader, ModelAssetOptions, ModelFamily, RawWeightFormat,
    WeightLocator, ensure_canonical_weights, load_canonical_weights,
};

fn required_raw_filenames(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Pi3 => &["model.safetensors"],
        ModelFamily::Pi3x => &["model.safetensors", "config.json"],
        ModelFamily::TripoSr => &["model.ckpt", "config.yaml"],
        ModelFamily::TripoSg => &[
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "image_encoder_dinov2/model.safetensors",
            "model_index.json",
            "feature_extractor_dinov2/preprocessor_config.json",
            "image_encoder_dinov2/config.json",
            "scheduler/scheduler_config.json",
            "transformer/config.json",
            "vae/config.json",
        ],
        ModelFamily::Vggt => &["model.safetensors", "config.json"],
        ModelFamily::Trellis2 => &[
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.json",
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.json",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.json",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
            "ckpts/shape_dec_next_dc_f16c32_fp16.json",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/tex_dec_next_dc_f16c32_fp16.json",
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.json",
            "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            "external/dinov3/config.json",
            "external/dinov3/model.safetensors",
            "pipeline.json",
        ],
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
        let path = raw_dir.join(filename);
        // Required filenames may be nested (TripoSG sharded layout).
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("raw fixture parent dir");
        }
        fs::write(&path, b"fixture").expect("raw fixture file");
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
fn triposg_canonical_package_repo_is_the_oxide_lab_mirror() {
    // The raw checkpoint lives at `VAST-AI/TripoSG` (raw shards,
    // diffusers layout); the canonical four-file package is the oxide-lab
    // mirror (public, files verified byte-identical 2026-11-12).
    assert_eq!(
        "oxide-lab/TripoSG",
        ModelFamily::TripoSg.huggingface_repo_id()
    );
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

fn triposg_raw_relative_files() -> [&'static str; 9] {
    [
        "transformer/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "image_encoder_dinov2/model.safetensors",
        "model_index.json",
        "feature_extractor_dinov2/preprocessor_config.json",
        "image_encoder_dinov2/config.json",
        "scheduler/scheduler_config.json",
        "transformer/config.json",
        "vae/config.json",
    ]
}

#[test]
fn triposg_locator_finds_raw_shards_in_exact_order() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::TripoSg);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights = locator
        .locate(ModelFamily::TripoSg)
        .expect("triposg weights");

    assert_eq!(ModelFamily::TripoSg, weights.family);
    assert_eq!(RawWeightFormat::SafetensorsDirect, weights.raw_format);
    let expected = triposg_raw_relative_files()
        .into_iter()
        .map(|relative| raw_model_dir.join(relative))
        .collect::<Vec<_>>();
    assert_eq!(expected, weights.raw_files.to_vec());
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    assert!(weights.license_note.contains("MIT"));
    cleanup_dir(&root);
}

#[test]
fn triposg_locator_finds_the_local_raw_checkpoint_dir() {
    let raw_model_dir = workspace_root().join("models").join("TripoSG");
    if !raw_model_dir.is_dir() {
        eprintln!(
            "skipping: local TripoSG raw dir {} is absent",
            raw_model_dir.display()
        );
        return;
    }

    let canonical_dir = temp_fixture_root("triposg-local-canonical").join("triposg");
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir);
    let weights = locator
        .locate(ModelFamily::TripoSg)
        .expect("triposg weights from the local raw checkpoint dir");

    let expected = triposg_raw_relative_files()
        .into_iter()
        .map(|relative| raw_model_dir.join(relative))
        .collect::<Vec<_>>();
    assert_eq!(expected, weights.raw_files.to_vec());
    assert_eq!(RawWeightFormat::SafetensorsDirect, weights.raw_format);
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert!(weights.license_note.contains("MIT"));
}

fn trellis2_ckpt_stems() -> [&'static str; 7] {
    [
        "ss_flow_img_dit_1_3B_64_bf16",
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
        "shape_dec_next_dc_f16c32_fp16",
        "tex_dec_next_dc_f16c32_fp16",
    ]
}

#[test]
fn trellis2_locator_finds_raw_component_files_in_locator_order() {
    let (root, raw_model_dir, canonical_dir) = create_raw_fixture(ModelFamily::Trellis2);
    let locator = WeightLocator::new(raw_model_dir.clone(), canonical_dir.clone());
    let weights = locator
        .locate(ModelFamily::Trellis2)
        .expect("trellis2 weights");

    assert_eq!(ModelFamily::Trellis2, weights.family);
    assert_eq!(RawWeightFormat::SafetensorsDirect, weights.raw_format);
    let expected: Vec<PathBuf> = required_raw_filenames(ModelFamily::Trellis2)
        .iter()
        .map(|relative| raw_model_dir.join(relative))
        .collect();
    assert_eq!(19, expected.len());
    assert_eq!(expected, weights.raw_files.to_vec());
    assert_eq!(canonical_dir, weights.canonical_root);
    assert_eq!("model.safetensors", weights.canonical_filename);
    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        weights.future_loader
    );
    assert!(weights.license_note.contains("MIT (Microsoft)"));
    assert!(weights.license_note.contains("DINOv3"));

    // Missing one component must fail loudly (no partial plans).
    let broken_root = temp_fixture_root("trellis2-broken");
    let broken_raw = broken_root.join("raw");
    let broken_canonical = broken_root.join("canonical").join("trellis2");
    fs::create_dir_all(&broken_raw).expect("broken raw dir");
    fs::write(broken_raw.join("pipeline.json"), b"{}").expect("pipeline fixture");
    let broken_locator = WeightLocator::new(broken_raw, broken_canonical);
    assert!(
        broken_locator.locate(ModelFamily::Trellis2).is_err(),
        "locator must reject a raw tree missing the ckpt components"
    );
    cleanup_dir(&broken_root);
    cleanup_dir(&root);
}

/// Exact real-tree tensor counts per stage (from the weights inventory); the
/// normalizer hard-fails on a mismatch, so synthetic fixtures must match.
fn trellis2_stage_tensor_counts() -> [(&'static str, usize); 9] {
    [
        ("image_cond", 415),
        ("ss_flow", 640),
        ("ss_dec", 74),
        ("shape_dit_512", 640),
        ("shape_dit_1024", 640),
        ("shape_dec", 292),
        ("tex_dit_512", 640),
        ("tex_dit_1024", 640),
        ("tex_dec", 284),
    ]
}

fn write_trellis2_safetensors_fixture(raw_dir: &Path) {
    use std::collections::BTreeMap;

    let tensor_data: Vec<u8> = (0..64u32).flat_map(|v| v.to_le_bytes()).collect();
    let write_shard =
        |relative: &str, count: usize, dtype: safetensors::Dtype, shape: Vec<usize>| {
            let needed: usize =
                shape.iter().product::<usize>() * ((dtype.bitsize() as usize).div_ceil(8));
            assert!(
                needed <= tensor_data.len(),
                "fixture data slice must cover {needed} bytes"
            );
            let tensors: BTreeMap<String, safetensors::tensor::TensorView> = (0..count)
                .map(|index| {
                    (
                        format!("t{index}"),
                        safetensors::tensor::TensorView::new(
                            dtype,
                            shape.clone(),
                            &tensor_data[..needed],
                        )
                        .expect("fixture tensor view"),
                    )
                })
                .collect();
            let path = raw_dir.join(relative);
            fs::create_dir_all(path.parent().expect("shard parent")).expect("shard dir");
            safetensors::serialize_to_file(tensors, None, &path).expect("write shard fixture");
        };

    write_shard(
        "external/dinov3/model.safetensors",
        415,
        safetensors::Dtype::F32,
        vec![1, 1, 2],
    );
    for stem in [
        "ss_flow_img_dit_1_3B_64_bf16",
        "slat_flow_img2shape_dit_1_3B_512_bf16",
        "slat_flow_img2shape_dit_1_3B_1024_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
        "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
    ] {
        write_shard(
            &format!("ckpts/{stem}.safetensors"),
            640,
            safetensors::Dtype::BF16,
            vec![2, 1],
        );
    }
    write_shard(
        "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        74,
        safetensors::Dtype::F16,
        vec![1],
    );
    write_shard(
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
        292,
        safetensors::Dtype::F16,
        vec![2],
    );
    write_shard(
        "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
        284,
        safetensors::Dtype::F16,
        vec![2],
    );

    for stem in trellis2_ckpt_stems() {
        let path = raw_dir.join("ckpts").join(format!("{stem}.json"));
        fs::write(&path, format!(r#"{{"class_name": "{stem}"}}"#)).expect("ckpt config fixture");
    }
    fs::create_dir_all(raw_dir.join("external").join("ss_dec").join("ckpts")).expect("ss_dec dir");
    fs::write(
        raw_dir
            .join("external")
            .join("ss_dec")
            .join("ckpts")
            .join("ss_dec_conv3d_16l8_fp16.json"),
        r#"{"out_channels": 1}"#,
    )
    .expect("ss_dec config fixture");
    fs::create_dir_all(raw_dir.join("external").join("dinov3")).expect("dinov3 dir");
    fs::write(
        raw_dir.join("external").join("dinov3").join("config.json"),
        r#"{"hidden_size": 1024}"#,
    )
    .expect("dinov3 config fixture");
    fs::write(
        raw_dir.join("pipeline.json"),
        r#"{"name": "Trellis2ImageTo3DPipeline", "args": {
            "sparse_structure_sampler": {"steps": 12},
            "shape_slat_normalization": {"mean": [0.781296], "std": [5.972266]},
            "tex_slat_normalization": {"mean": [3.501659], "std": [2.665652]}
        }}"#,
    )
    .expect("pipeline fixture");
}

#[test]
fn trellis2_python_normalizer_merges_stages_and_provenance_keys() {
    let root = temp_fixture_root("trellis2-normalizer");
    let raw_model_dir = root.join("raw");
    let output_dir = root.join("canonical").join("trellis2");
    fs::create_dir_all(&raw_model_dir).expect("raw fixture dir");
    write_trellis2_safetensors_fixture(&raw_model_dir);

    run_python_normalizer(ModelFamily::Trellis2, &raw_model_dir, &output_dir);

    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("manifest.json")).expect("read trellis2 manifest"),
    )
    .expect("parse trellis2 manifest");
    assert_eq!("trellis2", manifest["family"]);
    // Every raw basename is unique, so provenance keys stay bare and follow
    // the locator order exactly.
    let expected_raw_files: Vec<String> = required_raw_filenames(ModelFamily::Trellis2)
        .iter()
        .map(|relative| {
            Path::new(relative)
                .file_name()
                .and_then(|name| name.to_str())
                .expect("basename")
                .to_string()
        })
        .collect();
    assert_eq!(serde_json::json!(expected_raw_files), manifest["raw_files"]);
    assert_eq!(19, manifest["raw_files"].as_array().expect("array").len());
    assert_eq!(4265, manifest["tensor_count"]);
    assert_eq!(
        serde_json::json!({"BF16": 3200, "F16": 650, "F32": 415}),
        manifest["dtype_histogram"]
    );
    assert_eq!(
        19,
        manifest["source_checksums"].as_object().expect("map").len()
    );

    let canonical_bytes =
        fs::read(output_dir.join("model.safetensors")).expect("read canonical fixture");
    let tensors =
        safetensors::SafeTensors::deserialize(&canonical_bytes).expect("deserialize fixture");
    assert_eq!(4265, tensors.len());
    for (prefix, count) in trellis2_stage_tensor_counts() {
        let stage_keys = tensors
            .names()
            .iter()
            .filter(|key| key.starts_with(&format!("{prefix}.")))
            .count();
        assert_eq!(count, stage_keys, "stage `{prefix}` tensor count");
    }
    let probe = tensors.tensor("image_cond.t0").expect("image_cond probe");
    assert_eq!(safetensors::Dtype::F32, probe.dtype());
    let probe = tensors.tensor("ss_flow.t0").expect("ss_flow probe");
    assert_eq!(safetensors::Dtype::BF16, probe.dtype());
    let probe = tensors.tensor("shape_dec.t291").expect("shape_dec probe");
    assert_eq!(safetensors::Dtype::F16, probe.dtype());

    let resolved: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("resolved_config.json")).expect("read resolved config"),
    )
    .expect("parse resolved config");
    assert_eq!("trellis2", resolved["family"]);
    assert_eq!(
        serde_json::json!(12),
        resolved["pipeline_args"]["sparse_structure_sampler"]["steps"]
    );
    assert_eq!(
        serde_json::json!([0.781296]),
        resolved["normalization"]["shape_slat"]["mean"]
    );
    for prefix in [
        "image_cond",
        "ss_flow",
        "ss_dec",
        "shape_dit_512",
        "shape_dit_1024",
        "shape_dec",
        "tex_dit_512",
        "tex_dit_1024",
        "tex_dec",
    ] {
        assert!(
            resolved["stage_configs"].get(prefix).is_some(),
            "missing stage config `{prefix}`"
        );
    }

    // Locator plan plus the Python-written package must validate together.
    let locator = WeightLocator::new(raw_model_dir.clone(), output_dir.clone());
    let plan = locator
        .locate(ModelFamily::Trellis2)
        .expect("trellis2 weights");
    let canonical = ensure_canonical_weights(&plan).expect("trellis2 canonical package");
    assert_eq!(4265, canonical.manifest.tensor_count);
    assert_eq!(19, canonical.manifest.raw_files.len());
    assert_eq!(ModelFamily::Trellis2, canonical.manifest.family);

    // Second run must hit the idempotency fast path.
    let manifest_before = fs::read_to_string(output_dir.join("manifest.json")).expect("before");
    run_python_normalizer(ModelFamily::Trellis2, &raw_model_dir, &output_dir);
    let manifest_after = fs::read_to_string(output_dir.join("manifest.json")).expect("after");
    assert_eq!(manifest_before, manifest_after);

    cleanup_dir(&root);
}

#[test]
fn triposg_provenance_keys_disambiguate_duplicate_basenames() {
    use sha2::{Digest, Sha256};

    let sha256 = |path: &Path| -> String {
        let bytes = fs::read(path).expect("read fixture for sha256");
        let mut digest = Sha256::new();
        digest.update(&bytes);
        digest
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    };

    let root = temp_fixture_root("triposg-provenance");
    let raw_dir = root.join("raw");
    let canonical_dir = root.join("canonical").join("triposg");

    // Two raw shards share the basename `model.safetensors`; `config.json` is
    // unique. Expected provenance keys: `a/model.safetensors`,
    // `b/model.safetensors`, and the bare `config.json`.
    for relative in ["a/model.safetensors", "b/model.safetensors", "config.json"] {
        let path = raw_dir.join(relative);
        fs::create_dir_all(path.parent().expect("fixture parent")).expect("fixture dir");
        fs::write(&path, format!("payload-{relative}")).expect("fixture payload");
    }

    let plan = CanonicalizationPlan {
        family: ModelFamily::TripoSg,
        raw_format: RawWeightFormat::SafetensorsDirect,
        raw_files: vec![
            raw_dir.join("a/model.safetensors"),
            raw_dir.join("b/model.safetensors"),
            raw_dir.join("config.json"),
        ]
        .into_boxed_slice(),
        canonical_root: canonical_dir.clone(),
        canonical_filename: "model.safetensors".to_string(),
        future_loader: FutureWeightLoader::CandleMmapSafetensors,
        license_note: "test fixture".to_string(),
    };

    fs::create_dir_all(&canonical_dir).expect("canonical fixture dir");
    fs::write(canonical_dir.join("model.safetensors"), b"canonical").expect("canonical fixture");
    fs::write(canonical_dir.join("resolved_config.json"), b"{}").expect("config fixture");

    let write_package = |manifest: serde_json::Value| {
        let manifest_path = canonical_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            serde_json::to_string_pretty(&manifest).expect("serialize manifest"),
        )
        .expect("manifest fixture");
        let mut files = vec!["model.safetensors", "resolved_config.json", "manifest.json"]
            .into_iter()
            .map(|relative| {
                let path = canonical_dir.join(relative);
                serde_json::json!({
                    "relative_path": relative,
                    "sha256": sha256(&path),
                    "size_bytes": fs::metadata(&path).expect("metadata").len(),
                })
            })
            .collect::<Vec<_>>();
        files.sort_unstable_by(|a, b| {
            a["relative_path"]
                .as_str()
                .cmp(&b["relative_path"].as_str())
        });
        let checksums = serde_json::json!({ "files": files });
        fs::write(
            canonical_dir.join("checksums.json"),
            serde_json::to_string_pretty(&checksums).expect("serialize checksums"),
        )
        .expect("checksums fixture");
    };

    let manifest = serde_json::json!({
        "family": "triposg",
        "normalizer_version": 1,
        "raw_files": ["a/model.safetensors", "b/model.safetensors", "config.json"],
        "canonical_file": "model.safetensors",
        "resolved_config_file": "resolved_config.json",
        "tensor_count": 0,
        "dtype_histogram": {},
        "source_checksums": {
            "a/model.safetensors": sha256(&raw_dir.join("a/model.safetensors")),
            "b/model.safetensors": sha256(&raw_dir.join("b/model.safetensors")),
            "config.json": sha256(&raw_dir.join("config.json")),
        }
    });
    write_package(manifest);

    let canonical =
        ensure_canonical_weights(&plan).expect("duplicate-basename provenance keys must validate");
    assert_eq!(
        vec!["a/model.safetensors", "b/model.safetensors", "config.json"],
        canonical.manifest.raw_files.to_vec()
    );
    assert_eq!(3, canonical.manifest.source_checksums.len());

    // Bare `model.safetensors` keys must be rejected now that the raw
    // basenames are duplicated: the parent-prefixed form is required.
    let bare_manifest = serde_json::json!({
        "family": "triposg",
        "normalizer_version": 1,
        "raw_files": ["model.safetensors", "model.safetensors", "config.json"],
        "canonical_file": "model.safetensors",
        "resolved_config_file": "resolved_config.json",
        "tensor_count": 0,
        "dtype_histogram": {},
        "source_checksums": {
            "model.safetensors": sha256(&raw_dir.join("a/model.safetensors")),
            "config.json": sha256(&raw_dir.join("config.json")),
        }
    });
    write_package(bare_manifest);

    let rejected = ensure_canonical_weights(&plan);
    assert!(
        rejected.is_err(),
        "bare duplicate-basename provenance keys must be rejected"
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

#[test]
fn triposg_python_normalizer_merges_shards_and_provenance_keys() {
    let root = temp_fixture_root("triposg-normalizer");
    let raw_model_dir = root.join("raw");
    let output_dir = root.join("canonical").join("triposg");

    // Synthetic raw repo: three single-tensor shards (two share a basename)
    // plus the six config files of the real layout.
    fs::create_dir_all(&raw_model_dir).expect("raw fixture dir");
    let write_shard = |relative: &str, key: &str| {
        let path = raw_model_dir.join(relative);
        fs::create_dir_all(path.parent().expect("shard parent")).expect("shard dir");
        let value = 1.0f32.to_le_bytes();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], &value)
            .expect("fixture tensor view");
        safetensors::serialize_to_file([(key.to_string(), view)], None, &path)
            .expect("write shard fixture");
    };
    write_shard(
        "transformer/diffusion_pytorch_model.safetensors",
        "proj_in.weight",
    );
    write_shard(
        "vae/diffusion_pytorch_model.safetensors",
        "post_quant.weight",
    );
    write_shard(
        "image_encoder_dinov2/model.safetensors",
        "embeddings.cls_token",
    );
    let write_json = |relative: &str, body: &str| {
        let path = raw_model_dir.join(relative);
        fs::create_dir_all(path.parent().expect("config parent")).expect("config dir");
        fs::write(path, body).expect("write config fixture");
    };
    write_json("model_index.json", r#"{"_class_name": "TripoSGPipeline"}"#);
    write_json(
        "feature_extractor_dinov2/preprocessor_config.json",
        r#"{"crop_size": 224}"#,
    );
    write_json(
        "image_encoder_dinov2/config.json",
        r#"{"hidden_size": 1024}"#,
    );
    write_json("scheduler/scheduler_config.json", r#"{"shift": 1}"#);
    write_json("transformer/config.json", r#"{"num_layers": 21}"#);
    write_json("vae/config.json", r#"{"latent_channels": 64}"#);

    run_python_normalizer(ModelFamily::TripoSg, &raw_model_dir, &output_dir);

    let manifest: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(output_dir.join("manifest.json")).expect("read triposg manifest"),
    )
    .expect("parse triposg manifest");
    assert_eq!("triposg", manifest["family"]);
    // Duplicate basenames are parent-prefixed; unique basenames stay bare.
    assert_eq!(
        serde_json::json!([
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.safetensors",
            "model.safetensors",
            "model_index.json",
            "preprocessor_config.json",
            "image_encoder_dinov2/config.json",
            "scheduler_config.json",
            "transformer/config.json",
            "vae/config.json"
        ]),
        manifest["raw_files"]
    );
    assert_eq!(3, manifest["tensor_count"]);
    let canonical_bytes =
        fs::read(output_dir.join("model.safetensors")).expect("read canonical fixture");
    let tensors =
        safetensors::SafeTensors::deserialize(&canonical_bytes).expect("deserialize fixture");
    let mut keys = tensors.names();
    keys.sort();
    assert_eq!(
        vec![
            "image_encoder.embeddings.cls_token",
            "transformer.proj_in.weight",
            "vae.post_quant.weight"
        ],
        keys
    );

    // The locator plan plus the Python-written package must validate together.
    let locator = WeightLocator::new(raw_model_dir.clone(), output_dir.clone());
    let plan = locator
        .locate(ModelFamily::TripoSg)
        .expect("triposg weights");
    let canonical = ensure_canonical_weights(&plan).expect("triposg canonical package");
    assert_eq!(3, canonical.manifest.tensor_count);
    assert_eq!(9, canonical.manifest.raw_files.len());

    // Second run must hit the idempotency fast path.
    let manifest_before = fs::read_to_string(output_dir.join("manifest.json")).expect("before");
    run_python_normalizer(ModelFamily::TripoSg, &raw_model_dir, &output_dir);
    let manifest_after = fs::read_to_string(output_dir.join("manifest.json")).expect("after");
    assert_eq!(manifest_before, manifest_after);

    cleanup_dir(&root);
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

// T10: the standalone TRELLIS.2 GGUF package (DiT-only quantization policy)
// must load on its own: embedded resolved config, `lux3d.family` metadata,
// a Q8_0-packed DiT linear for the `QMatMul` path, and F16 dense fallback
// tensors for the dense-read loaders (DINOv3 / VAEs). Gated on the artifact
// produced by `weights quantize --family trellis2`; skipped when absent.
#[test]
fn trellis2_standalone_gguf_package_loads() {
    let gguf = lux3d_core::test_support::runtime_root()
        .join("3d")
        .join("huggingface")
        .join("TRELLIS.2-4B-GGUF")
        .join("trellis2-q8_0.gguf");
    if !gguf.is_file() {
        eprintln!(
            "skipping trellis2_standalone_gguf_package_loads: no package at {}",
            gguf.display()
        );
        return;
    }

    let weights = load_canonical_weights(
        ModelFamily::Trellis2,
        ModelAssetOptions {
            canonical_dir: Some(gguf.clone()),
            cache_dir: None,
        },
    )
    .expect("standalone trellis2 gguf package");
    assert_eq!(weights.canonical_file, gguf);
    let config = weights
        .read_resolved_config_json()
        .expect("embedded resolved config");
    assert!(
        config.get("pipeline_args").is_some() && config.get("stage_configs").is_some(),
        "trellis2 resolved config must carry the pipeline defaults and stage configs"
    );

    let content = {
        let mut file = fs::File::open(&gguf).expect("open gguf");
        candle_core::quantized::gguf_file::Content::read(&mut file).expect("gguf header")
    };
    match content.metadata.get("lux3d.family") {
        Some(candle_core::quantized::gguf_file::Value::String(family)) => {
            assert_eq!(family, "trellis2")
        }
        other => panic!("missing lux3d.family metadata: {other:?}"),
    }
    assert!(content.metadata.contains_key("lux3d.resolved_config"));

    // DiT linear matrices only; every other tensor stays in the F16 fallback.
    // 1075 rank-2 DiT `*.weight` matrices minus `ss_flow.input_layer.weight`
    // (row length 8 does not divide into the 32-wide Q8_0 block).
    let quantized = content
        .tensor_infos
        .values()
        .filter(|info| info.ggml_dtype == candle_core::quantized::GgmlDType::Q8_0)
        .count();
    assert_eq!(
        quantized, 1074,
        "expected the 1074 Q8_0 DiT linear matrices"
    );
    assert!(content.tensor_infos.values().all(|info| info.ggml_dtype
        == candle_core::quantized::GgmlDType::Q8_0
        || info.ggml_dtype == candle_core::quantized::GgmlDType::F16));

    // The runtime load path: quantized DiT linear via `get_quantized`, dense
    // fallback tensor via the regular F32 VarBuilder.
    let vb = unsafe {
        weights
            .var_builder(candle_core::DType::F32, &candle_core::Device::Cpu)
            .expect("gguf var builder")
    };
    let qkv = vb
        .pp("ss_flow.blocks.0.self_attn")
        .get_quantized("to_qkv.weight")
        .expect("quantized DiT linear lookup")
        .expect("DiT linear must be Q8_0-packed");
    assert_eq!(qkv.dtype(), candle_core::quantized::GgmlDType::Q8_0);
    assert_eq!(qkv.shape().dims(), &[4608, 1536]);
    let cls = vb
        .pp("image_cond.embeddings")
        .get((1, 1, 1024), "cls_token")
        .expect("dense fallback tensor");
    assert_eq!(cls.dtype(), candle_core::DType::F32);
    assert_eq!(cls.dims(), &[1, 1, 1024]);
}
