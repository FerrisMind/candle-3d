use std::{
    fs,
    path::{Path, PathBuf},
};

use lux3d_core::test_support::runtime_root;
use lux3d_core::{
    ExportSpec, FutureWeightLoader, GeometrySpec, ModelFamily, ModelSpec, NeuralSpec,
    Pi3InputSource, PreprocessSpec, RgbRange, RuntimeGeometry, SourceDisposition, TensorDType,
};

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

fn assert_path_free_weight_plan(
    spec: &ModelSpec,
    family: ModelFamily,
    expected_raw_files: &[&str],
) {
    assert_eq!(
        expected_raw_files
            .iter()
            .map(PathBuf::from)
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
            assert_eq!(export.alternate_extensions, vec!["xyz".to_string()]);
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
            assert_eq!(export.alternate_extensions, vec!["xyz".to_string()]);
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
            assert_eq!("glb", export.primary_extension);
            assert_eq!(
                export.alternate_extensions,
                vec!["obj".to_string(), "ply".to_string(), "stl".to_string()]
            );
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
fn vggt_model_spec_is_source_truth_backed() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::Vggt).expect("vggt model spec");

    match &spec.preprocess {
        PreprocessSpec::Vggt(preprocess) => {
            assert_eq!(
                vec![Pi3InputSource::Directory, Pi3InputSource::Video],
                preprocess.accepted_sources
            );
            assert_eq!("NCHW", preprocess.input_layout);
            assert_eq!(RgbRange::ZeroToOne, preprocess.input_range);
            assert_eq!(14, preprocess.patch_multiple);
            assert_eq!(518, preprocess.target_width);
            assert_eq!(518, preprocess.max_image_size);
            assert_eq!("crop", preprocess.resize_mode);
            assert_eq!("bicubic", preprocess.resize_filter);
            assert_eq!(1.0, preprocess.background_value);
            assert!(preprocess.optional_rgba_compositing);
            assert_eq!(
                [0.485, 0.456, 0.406],
                preprocess.internal_normalization.mean
            );
            assert_eq!([0.229, 0.224, 0.225], preprocess.internal_normalization.std);
        }
        other => panic!("expected Vggt preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::Vggt(neural) => {
            assert_eq!("dinov2_vitl14_reg", neural.patch_embed_backbone);
            assert_eq!(24, neural.aggregator_depth);
            assert_eq!("frame_global_alternating", neural.attention_pattern);
            assert_eq!(100.0, neural.rope_frequency);
            assert!(neural.qk_norm);
            assert_eq!(4, neural.register_tokens);
            assert_eq!("absT_quaR_FoV", neural.pose_encoding);
            assert_eq!(4, neural.camera_refine_iterations);
            assert_eq!("dpt_inv_log_expp1", neural.point_head);
            assert_eq!("dpt_exp_expp1", neural.depth_head);
            assert!(neural.track_head.is_none());
        }
        other => panic!("expected Vggt neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::Vggt(geometry) => {
            assert_eq!("point_cloud", geometry.artifact_kind);
            assert_eq!(
                "world_points = point_head(aggregated_tokens)",
                geometry.assembly_formula
            );
            assert!(!geometry.supports_marching_cubes);
            assert!(geometry.supports_depth_unprojection);
        }
        other => panic!("expected Vggt geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::Vggt(export) => {
            assert_eq!("ply", export.primary_extension);
            assert_eq!(export.alternate_extensions, vec!["xyz".to_string()]);
            for expected_note in [
                "world_points_conf > 1.0",
                "colors from preprocessed RGB frames",
                "write_ply(points, rgb); .xyz is a plain-text alternative",
            ] {
                assert!(
                    export
                        .utility_notes
                        .iter()
                        .any(|note| note == expected_note),
                    "missing export utility note `{expected_note}`"
                );
            }
        }
        other => panic!("expected Vggt export spec, got {other:?}"),
    }

    let RuntimeGeometry::Vggt(runtime) = &spec.runtime_geometry else {
        panic!("expected vggt runtime geometry");
    };
    assert_eq!(5, runtime.outputs.len());
    for (name, dims) in [
        ("vggt.world_points", ["1", "S", "H", "W", "3"].as_slice()),
        ("vggt.world_points_conf", ["1", "S", "H", "W"].as_slice()),
        ("vggt.depth", ["1", "S", "H", "W", "1"].as_slice()),
        ("vggt.camera.extrinsics", ["1", "S", "3", "4"].as_slice()),
        ("vggt.camera.intrinsics", ["1", "S", "3", "3"].as_slice()),
    ] {
        let output = runtime
            .outputs
            .iter()
            .find(|output| output.name == name)
            .unwrap_or_else(|| panic!("missing vggt runtime output `{name}`"));
        assert_eq!(
            dims,
            output.dims.iter().map(String::as_str).collect::<Vec<_>>(),
            "unexpected dims for `{name}`"
        );
        assert_eq!(
            TensorDType::F32,
            output.runtime_dtype,
            "unexpected runtime dtype for `{name}`"
        );
    }

    let pose = spec
        .runtime_tensors
        .iter()
        .find(|tensor| tensor.name == "vggt.camera.pose_enc")
        .expect("vggt contract must expose `vggt.camera.pose_enc`");
    assert_eq!(
        vec!["1", "S", "9"],
        pose.dims.iter().map(String::as_str).collect::<Vec<_>>()
    );
    assert_eq!(TensorDType::F32, pose.runtime_dtype);

    let export_mask = spec
        .runtime_tensors
        .iter()
        .find(|tensor| tensor.name == "vggt.export_mask")
        .expect("vggt contract must expose `vggt.export_mask`");
    assert_eq!(
        vec!["1", "S", "H", "W"],
        export_mask
            .dims
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        TensorDType::U8,
        export_mask.runtime_dtype,
        "vggt export mask is materialized as U8"
    );

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_path_free_weight_plan(
        &spec,
        ModelFamily::Vggt,
        &["model.safetensors", "config.json"],
    );
    assert_eq!("pi3-frames-golden", spec.golden_baseline_sample);
    for excluded in ["vggt_track_head", "vggt_commercial_checkpoint"] {
        assert!(
            spec.exclusions.iter().any(|entry| entry.item == excluded),
            "missing contract exclusion `{excluded}`"
        );
    }
    assert_eq!("CC-BY-NC-4.0", spec.license_policy.repo_license);
    assert!(spec.license_policy.weight_license.contains("CC-BY-NC-4.0"));
    for source_path in spec.used_source_paths().iter() {
        assert!(
            spec.license_policy.covers_source(source_path),
            "missing license classification for {source_path}"
        );
    }
}

#[test]
fn triposg_model_spec_is_source_truth_backed() {
    let spec =
        ModelSpec::inspect(runtime_root(), ModelFamily::TripoSg).expect("triposg model spec");

    match &spec.preprocess {
        PreprocessSpec::TripoSg(preprocess) => {
            assert_eq!("NCHW", preprocess.input_layout);
            assert_eq!("NCHW", preprocess.tokenizer_input_layout);
            assert_eq!(224, preprocess.target_size.width);
            assert_eq!(224, preprocess.target_size.height);
            assert_eq!(1.0, preprocess.background_value);
            assert_eq!(0.1, preprocess.foreground_padding_ratio);
            assert!(preprocess.optional_rgba_compositing);
            assert_eq!([0.485, 0.456, 0.406], preprocess.normalization.mean);
            assert_eq!([0.229, 0.224, 0.225], preprocess.normalization.std);
            assert_eq!("BriaRMBG 1.4 (IS-Net/BriaRMBG)", preprocess.rmbg.model);
            assert_eq!("rmbg-1.4", preprocess.rmbg.canonical_root);
            assert_eq!([1024, 1024], preprocess.rmbg.input_size);
            assert_eq!([0.5, 0.5, 0.5], preprocess.rmbg.normalization.mean);
            assert_eq!([1.0, 1.0, 1.0], preprocess.rmbg.normalization.std);
            assert!(preprocess.rmbg.license.contains("BRIA"));
        }
        other => panic!("expected TripoSG preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::TripoSg(neural) => {
            assert_eq!("facebook/dinov2-large", neural.image_encoder);
            assert_eq!(24, neural.image_encoder_layers);
            assert_eq!(1024, neural.image_encoder_hidden);
            assert_eq!(14, neural.image_encoder_patch_size);
            assert_eq!(257, neural.image_embeds_tokens);
            assert_eq!(21, neural.dit_layers);
            assert_eq!(2048, neural.dit_width);
            assert_eq!(16, neural.dit_heads);
            assert_eq!(1024, neural.cross_attention_dim);
            assert_eq!(2048, neural.latent_tokens);
            assert_eq!(64, neural.latent_channels);
            assert_eq!("RectifiedFlowScheduler", neural.scheduler);
            assert_eq!(50, neural.default_steps);
            assert_eq!(7.0, neural.default_guidance_scale);
            assert_eq!(16, neural.vae_decoder_layers);
            assert_eq!(1024, neural.vae_decoder_width);
            assert_eq!(8, neural.vae_embed_frequency);
        }
        other => panic!("expected TripoSG neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::TripoSg(geometry) => {
            assert_eq!("mesh", geometry.artifact_kind);
            assert_eq!("flash_diffdmc", geometry.surface_extractor);
            assert_eq!(8, geometry.default_dense_octree_depth);
            assert_eq!(9, geometry.default_hierarchical_octree_depth);
            assert!(geometry.default_use_flash_decoder);
            assert_eq!(9, geometry.default_flash_octree_depth);
            assert_eq!(0.0, geometry.sdf_level);
            assert_eq!(
                [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005],
                geometry.bounds
            );
        }
        other => panic!("expected TripoSG geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::TripoSg(export) => {
            assert_eq!("glb", export.primary_extension);
            assert_eq!(
                export.alternate_extensions,
                vec!["obj".to_string(), "ply".to_string(), "stl".to_string()]
            );
            assert!(
                export
                    .utility_notes
                    .iter()
                    .any(|note| note.contains("1-based indices")),
                "export notes must document the 1-based face indices"
            );
        }
        other => panic!("expected TripoSG export spec, got {other:?}"),
    }

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_path_free_weight_plan(&spec, ModelFamily::TripoSg, &triposg_raw_relative_files());
    assert_eq!("mesh-input-golden", spec.golden_baseline_sample);

    assert!(
        spec.exclusions.is_empty(),
        "TripoSG contract must not exclude reference features: {:?}",
        spec.exclusions
    );

    assert_eq!("MIT", spec.license_policy.repo_license);
    assert!(spec.license_policy.weight_license.contains("MIT"));
    for source_path in spec.used_source_paths().iter() {
        assert!(
            spec.license_policy.covers_source(source_path),
            "missing license classification for {source_path}"
        );
    }
}

#[test]
fn trellis2_contract_path_follows_the_family_name() {
    // Pure path computation; the full inspect of the contract document lives
    // in `trellis2_model_spec_is_source_truth_backed`.
    let path = ModelSpec::contract_path(Path::new("runtime"), ModelFamily::Trellis2);
    let rendered = path.to_string_lossy().replace('\\', "/");
    assert_eq!("runtime/3d/contracts/trellis2.json", rendered);
}

#[test]
fn trellis2_model_spec_is_source_truth_backed() {
    let spec =
        ModelSpec::inspect(runtime_root(), ModelFamily::Trellis2).expect("trellis2 model spec");

    match &spec.preprocess {
        PreprocessSpec::Trellis2(preprocess) => {
            assert_eq!("HWC", preprocess.input_layout);
            assert!(preprocess.premultiply_alpha);
            assert_eq!(vec![512, 1024], preprocess.target_sizes);
            assert_eq!(512, preprocess.default_target_size);
            assert_eq!([0.485, 0.456, 0.406], preprocess.normalization.mean);
            assert_eq!([0.229, 0.224, 0.225], preprocess.normalization.std);
            assert!(
                preprocess.no_alpha_behavior.contains("RMBG-2.0"),
                "no-alpha behavior must document the RMBG-2.0 path: {}",
                preprocess.no_alpha_behavior
            );
        }
        other => panic!("expected trellis2 preprocess spec, got {other:?}"),
    }

    match &spec.neural {
        NeuralSpec::Trellis2(neural) => {
            assert!(neural.image_encoder.contains("DINOv3"));
            assert_eq!(24, neural.image_encoder_layers);
            assert_eq!(1024, neural.image_encoder_hidden);
            assert_eq!(16, neural.image_encoder_patch_size);
            assert_eq!(4, neural.image_encoder_registers);
            assert_eq!(30, neural.flow_dit_layers);
            assert_eq!(1536, neural.flow_dit_width);
            assert_eq!(12, neural.flow_dit_heads);
            assert_eq!(128, neural.flow_dit_head_dim);
            assert_eq!(1024, neural.cross_attention_dim);
            assert_eq!(8, neural.sparse_structure_latent_channels);
            assert_eq!(32, neural.sparse_structure_resolution);
            assert_eq!(32, neural.slat_latent_channels);
            assert_eq!(
                [1024, 512, 256, 128, 64].as_slice(),
                neural.shape_vae_decoder_channels
            );
            assert_eq!(6, neural.texture_vae_decoder_output_channels);
            assert_eq!("FlowEulerGuidanceIntervalSampler", neural.sampler);
            assert_eq!(12, neural.default_steps);
            assert_eq!(7.5, neural.default_guidance_scale);
        }
        other => panic!("expected trellis2 neural spec, got {other:?}"),
    }

    match &spec.geometry {
        GeometrySpec::Trellis2(geometry) => {
            assert_eq!("mesh", geometry.artifact_kind);
            assert_eq!("flexible_dual_grid", geometry.surface_extractor);
            assert_eq!(7, geometry.voxel_output_channels);
            assert!(!geometry.supports_fill_holes);
            assert!(geometry.quad_split_rule.contains("w0*w2 > w1*w3"));
        }
        other => panic!("expected trellis2 geometry spec, got {other:?}"),
    }

    match &spec.export {
        ExportSpec::Trellis2(export) => {
            assert_eq!("glb", export.primary_extension);
            assert_eq!(
                export.alternate_extensions,
                vec!["obj".to_string(), "ply".to_string()]
            );
            assert!(
                export
                    .utility_notes
                    .iter()
                    .any(|note| note.contains("pbrvox.bin")),
                "export notes must document the PBR sidecar"
            );
        }
        other => panic!("expected trellis2 export spec, got {other:?}"),
    }

    let RuntimeGeometry::Trellis2(runtime) = &spec.runtime_geometry else {
        panic!("expected trellis2 runtime geometry");
    };
    assert_eq!("mesh", runtime.artifact_kind);
    assert_eq!(2, runtime.outputs.len());

    assert_eq!(
        FutureWeightLoader::CandleMmapSafetensors,
        spec.weight_plan.future_loader
    );
    assert_eq!(19, spec.weight_plan.raw_files.len());
    assert_path_free_weight_plan(
        &spec,
        ModelFamily::Trellis2,
        &[
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
    );

    for excluded in [
        "trellis2_mesh_postprocess",
        "trellis2_texture_atlas_uv",
        "trellis2_1536_cascade_and_single_1024",
        "trellis2_multi_sample",
    ] {
        assert!(
            spec.exclusions.iter().any(|entry| entry.item == excluded),
            "missing contract exclusion `{excluded}`"
        );
    }
    assert!(
        !spec
            .exclusions
            .iter()
            .any(|entry| entry.item == "trellis2_rmbg_background_removal"),
        "RMBG-2.0 background removal must not be excluded"
    );
    assert_eq!("MIT", spec.license_policy.repo_license);
    assert!(spec.license_policy.weight_license.contains("TRELLIS.2-4B"));
    for source_path in spec.used_source_paths().iter() {
        assert!(
            spec.license_policy.covers_source(source_path),
            "missing license classification for {source_path}"
        );
    }
}

#[test]
fn canonical_family_contract_list_is_unaffected_by_trellis2() {
    // Adding the `trellis2` family must not disturb the five existing
    // contract documents: their paths follow the family names and the family
    // tags inside each present document are unchanged. Full `ModelSpec::inspect`
    // (with vendor-file hashing) is covered by the per-family tests and the
    // vggt contract, which exists in this checkout.
    let runtime = runtime_root();
    let mut checked = 0;
    for family in [
        ModelFamily::Pi3,
        ModelFamily::Pi3x,
        ModelFamily::TripoSr,
        ModelFamily::TripoSg,
        ModelFamily::Vggt,
        ModelFamily::Trellis2,
    ] {
        let contract_path = ModelSpec::contract_path(&runtime, family);
        let rendered = contract_path.to_string_lossy().replace('\\', "/");
        assert!(
            rendered.ends_with(&format!("3d/contracts/{}.json", family.as_str())),
            "unexpected contract path for `{family}`: {rendered}"
        );
        if !contract_path.is_file() {
            // pi3/pi3x/triposr/trellis2 JSONs may be absent from this
            // checkout; that is a documentation gap owned by the contract
            // document task, not a regression of the family enum.
            eprintln!("note: contract file absent: {}", contract_path.display());
            continue;
        }
        let body = fs::read_to_string(&contract_path).expect("read contract body");
        let value: serde_json::Value = serde_json::from_str(&body).expect("parse contract body");
        assert_eq!(
            family.as_str(),
            value["family"].as_str().expect("family tag"),
            "contract document for `{family}` must keep its family tag"
        );
        assert_eq!(
            family.as_str(),
            value["weight_plan"]["family"]
                .as_str()
                .expect("weight plan family"),
            "weight plan for `{family}` must keep its family tag"
        );
        checked += 1;
    }
    assert!(
        checked >= 1,
        "expected at least one existing contract document on disk to verify"
    );
}

#[test]
fn license_policy_covers_every_used_source_file() {
    for family in [
        ModelFamily::Pi3,
        ModelFamily::Pi3x,
        ModelFamily::TripoSr,
        ModelFamily::TripoSg,
        ModelFamily::Vggt,
        ModelFamily::Trellis2,
    ] {
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
    for family in [
        ModelFamily::Pi3,
        ModelFamily::Pi3x,
        ModelFamily::TripoSr,
        ModelFamily::TripoSg,
        ModelFamily::Vggt,
        ModelFamily::Trellis2,
    ] {
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
