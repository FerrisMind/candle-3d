use lux3d_core::contracts::{NormalizationStats, TensorContract};
use lux3d_core::test_support::runtime_root;
use lux3d_core::{
    ExportSpec, GeometrySpec, ModelFamily, ModelSpec, NeuralSpec, PreprocessSpec, RuntimeGeometry,
    SpatialSize, TensorDType, Trellis2Mesh, Trellis2PipelineType, Trellis2PreparedImage,
    Trellis2RawInput,
};

#[test]
fn trellis2_family_round_trips_and_keeps_the_reserved_repo_id() {
    assert_eq!("trellis2", ModelFamily::Trellis2.as_str());

    let serialized = serde_json::to_string(&ModelFamily::Trellis2).expect("serialize family");
    assert_eq!(r#""trellis2""#, serialized);
    let deserialized: ModelFamily = serde_json::from_str(&serialized).expect("deserialize family");
    assert_eq!(ModelFamily::Trellis2, deserialized);

    // Reserved canonical-package mirror; must not be uploaded to.
    assert_eq!(
        "oxide-lab/TRELLIS.2-4B",
        ModelFamily::Trellis2.huggingface_repo_id()
    );

    // Existing families keep their serde names.
    for (family, name) in [
        (ModelFamily::Pi3, "pi3"),
        (ModelFamily::Pi3x, "pi3x"),
        (ModelFamily::TripoSr, "triposr"),
        (ModelFamily::TripoSg, "triposg"),
        (ModelFamily::Vggt, "vggt"),
    ] {
        assert_eq!(
            name,
            serde_json::to_string(&family)
                .expect("family name")
                .trim_matches('"')
        );
    }
}

#[test]
fn trellis2_contract_types_round_trip_independently_of_candle() {
    let raw_input = Trellis2RawInput {
        original_size: SpatialSize::new(1024, 768),
        rgba: Some(vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.25, 1.0]]),
        rgb: None,
    };
    let body = serde_json::to_string(&raw_input).expect("serialize raw input");
    assert_eq!(raw_input, serde_json::from_str(&body).expect("round trip"));

    let prepared = Trellis2PreparedImage {
        family: ModelFamily::Trellis2,
        original_size: SpatialSize::new(1024, 768),
        target_size: SpatialSize::new(518, 518),
        uint8_rgb: vec![0, 128, 255],
        point: None,
        pixel_values: TensorContract::new("trellis2.pixel_values", vec![1, 3, 518, 518]),
    };
    let body = serde_json::to_string(&prepared).expect("serialize prepared image");
    assert_eq!(prepared, serde_json::from_str(&body).expect("round trip"));

    let mesh = Trellis2Mesh {
        family: ModelFamily::Trellis2,
        vertices: vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0],
        faces: vec![0, 1, 2],
        voxel_coords: vec![3, 4, 5],
        voxel_attrs: vec![0.25; 7],
        resolution: 64,
    };
    let body = serde_json::to_string(&mesh).expect("serialize mesh");
    assert_eq!(mesh, serde_json::from_str(&body).expect("round trip"));
}

#[test]
fn trellis2_pipeline_type_uses_reference_config_strings() {
    for (variant, name) in [
        (Trellis2PipelineType::R512, "512"),
        (Trellis2PipelineType::R1024, "1024"),
        (Trellis2PipelineType::Cascade1024, "1024_cascade"),
    ] {
        assert_eq!(variant.as_str(), name);
        let body = serde_json::to_string(&variant).expect("serialize pipeline type");
        assert_eq!(format!(r#""{name}""#), body);
        let back: Trellis2PipelineType =
            serde_json::from_str(&body).expect("deserialize pipeline type");
        assert_eq!(variant, back);
    }
}

#[test]
fn trellis2_spec_enums_tag_the_family_and_expose_source_paths() {
    let source_paths = vec!["trellis2/pipelines/trellis2_image_to_3d.py".to_string()];

    let preprocess = PreprocessSpec::Trellis2(lux3d_core::Trellis2PreprocessSpec {
        input_layout: "HWC".to_string(),
        alpha_path: "rgba foreground bbox -> square crop -> premultiplied rgb".to_string(),
        no_alpha_behavior: "treated as fully opaque; RMBG-2.0 excluded in this slice".to_string(),
        foreground_crop: "alpha_bbox_square_crop".to_string(),
        premultiply_alpha: true,
        target_sizes: vec![512, 1024],
        default_target_size: 512,
        normalization: NormalizationStats {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        },
        source_paths: source_paths.clone(),
    });
    let body = serde_json::to_string(&preprocess).expect("serialize preprocess spec");
    assert!(
        body.contains(r#""family":"trellis2""#),
        "preprocess spec must tag family trellis2: {body}"
    );
    let back: PreprocessSpec = serde_json::from_str(&body).expect("round trip");
    assert_eq!(preprocess, back);
    assert_eq!(source_paths, preprocess.source_paths());

    let neural = NeuralSpec::Trellis2(lux3d_core::Trellis2NeuralSpec {
        image_encoder: "facebook/dinov3-vitl16-pretrain-lvd1689m".to_string(),
        image_encoder_layers: 24,
        image_encoder_hidden: 1024,
        image_encoder_patch_size: 16,
        image_encoder_registers: 4,
        flow_dit_layers: 30,
        flow_dit_width: 1536,
        flow_dit_heads: 12,
        flow_dit_head_dim: 128,
        cross_attention_dim: 1024,
        positional_encoding: "rope3d_latent_indices".to_string(),
        sparse_structure_latent_channels: 8,
        sparse_structure_resolution: 16,
        slat_latent_channels: 32,
        shape_vae_decoder: "FlexiDualGridVaeDecoder".to_string(),
        shape_vae_decoder_channels: vec![1024, 512, 256, 128, 64],
        texture_vae_decoder: "SparseUnetVaeDecoder".to_string(),
        texture_vae_decoder_output_channels: 6,
        sampler: "FlowEulerGuidanceIntervalSampler".to_string(),
        default_steps: 12,
        default_guidance_scale: 7.5,
        source_paths: source_paths.clone(),
    });
    let body = serde_json::to_string(&neural).expect("serialize neural spec");
    assert!(body.contains(r#""family":"trellis2""#));
    let back: NeuralSpec = serde_json::from_str(&body).expect("round trip");
    assert_eq!(neural, back);
    assert_eq!(source_paths, neural.source_paths());

    let geometry = GeometrySpec::Trellis2(lux3d_core::Trellis2GeometrySpec {
        artifact_kind: "mesh".to_string(),
        surface_extractor: "flexible_dual_grid".to_string(),
        voxel_output_channels: 7,
        coordinate_frame: "[-0.5, 0.5]^3".to_string(),
        quad_split_rule: "w0*w2 > w1*w3".to_string(),
        supports_fill_holes: false,
        source_paths: source_paths.clone(),
    });
    let body = serde_json::to_string(&geometry).expect("serialize geometry spec");
    assert!(body.contains(r#""family":"trellis2""#));
    let back: GeometrySpec = serde_json::from_str(&body).expect("round trip");
    assert_eq!(geometry, back);
    assert_eq!(source_paths, geometry.source_paths());

    let export = ExportSpec::Trellis2(lux3d_core::Trellis2ExportSpec {
        primary_extension: "obj".to_string(),
        alternate_extensions: vec!["ply".to_string(), "glb".to_string()],
        utility_notes: vec!["pbr sidecar <stem>.pbrvox.bin".to_string()],
        source_paths: source_paths.clone(),
    });
    let body = serde_json::to_string(&export).expect("serialize export spec");
    assert!(body.contains(r#""family":"trellis2""#));
    let back: ExportSpec = serde_json::from_str(&body).expect("round trip");
    assert_eq!(export, back);
    assert_eq!(source_paths, export.source_paths());

    let runtime = RuntimeGeometry::Trellis2(lux3d_core::Trellis2RuntimeGeometry {
        artifact_kind: "mesh".to_string(),
        surface_extractor: "flexible_dual_grid".to_string(),
        outputs: vec![],
        source_paths: source_paths.clone(),
        notes: vec!["minimal T1 scaffolding".to_string()],
    });
    let body = serde_json::to_string(&runtime).expect("serialize runtime geometry");
    assert!(body.contains(r#""family":"trellis2""#));
    let back: RuntimeGeometry = serde_json::from_str(&body).expect("round trip");
    assert_eq!(runtime, back);
    assert_eq!(source_paths, runtime.source_paths());
}

#[test]
fn triposg_contract_is_surface_only_and_tracked() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::TripoSg).expect("triposg contract");

    // Vendor sources must pin the ported TripoSG reference files.
    for expected_source in [
        "triposg/pipelines/pipeline_triposg.py",
        "triposg/models/transformers/triposg_transformer.py",
        "triposg/models/autoencoders/autoencoder_kl_triposg.py",
        "triposg/inference_utils.py",
        "triposg/schedulers/scheduling_rectified_flow.py",
    ] {
        assert!(
            spec.vendor_sources
                .iter()
                .any(|entry| entry.source_path.ends_with(expected_source)),
            "missing vendor source `{expected_source}`"
        );
    }

    // Runtime geometry outputs: F32 vertices and I64 faces; the baseline
    // stores faces as f32 (same convention as TripoSR).
    let RuntimeGeometry::TripoSg(runtime) = &spec.runtime_geometry else {
        panic!("expected triposg runtime geometry");
    };
    let vertices = runtime
        .outputs
        .iter()
        .find(|output| output.name == "triposg.mesh_vertices")
        .expect("mesh vertices output");
    assert_eq!(TensorDType::F32, vertices.runtime_dtype);
    assert_eq!(None, vertices.baseline_storage_dtype);
    let faces = runtime
        .outputs
        .iter()
        .find(|output| output.name == "triposg.mesh_faces")
        .expect("mesh faces output");
    assert_eq!(TensorDType::I64, faces.runtime_dtype);
    assert_eq!(Some(TensorDType::F32), faces.baseline_storage_dtype);

    // Surface-only geometry contract: the geometry stage exposes exactly the
    // two mesh buffers. VAE-encode parameters and the flash SDF grid are
    // parity taps, not exported geometry outputs; background removal lives in
    // the separately licensed RMBG package.
    assert_eq!(2, runtime.outputs.len());
    assert!(
        runtime
            .outputs
            .iter()
            .all(|output| output.name.starts_with("triposg.mesh")),
        "triposg runtime geometry must only expose mesh buffers"
    );

    // The RNG is NOT a parity tap (`latents_init` is baseline-injected in
    // parity tests because the public seeded sampler is intentionally not
    // cross-framework identical): the declared tap set is exactly these six.
    let tap_names = spec
        .baseline_parity_taps
        .iter()
        .map(|tap| tap.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        vec![
            "triposg.pixel_values",
            "triposg.image_embeds",
            "triposg.latents_step_final",
            "triposg.sdf_values",
            "triposg.flash_sdf_grid",
            "triposg.vae_encode_parameters"
        ],
        tap_names,
        "baseline parity taps must be exactly the declared set (no RNG/latents_init tap)"
    );

    // RMBG, the flash decoder and the VAE encoder are all implemented; the
    // contract no longer excludes any reference feature.
    assert!(
        spec.exclusions.is_empty(),
        "TripoSG contract must not exclude reference features: {:?}",
        spec.exclusions
    );
    for tensor in ["triposg.flash_sdf_grid", "triposg.vae_encode_parameters"] {
        assert!(
            spec.runtime_tensors
                .iter()
                .any(|entry| entry.name == tensor),
            "missing parity tensor `{tensor}`"
        );
    }
    assert!(
        spec.runtime_tensors
            .iter()
            .all(|tensor| !tensor.name.contains("rmbg")),
        "RMBG tensors belong to the separate canonical package"
    );
}

#[test]
fn pi3_and_pi3x_contracts_are_separately_scoped() {
    let pi3 = ModelSpec::inspect(runtime_root(), ModelFamily::Pi3).expect("pi3 contract");
    let pi3x = ModelSpec::inspect(runtime_root(), ModelFamily::Pi3x).expect("pi3x contract");

    assert!(
        pi3.vendor_sources
            .iter()
            .any(|entry| entry.source_path.ends_with("pi3/models/pi3.py"))
    );
    assert!(
        pi3x.vendor_sources
            .iter()
            .any(|entry| entry.source_path.ends_with("pi3/models/pi3x.py"))
    );
    assert!(
        pi3x.vendor_sources
            .iter()
            .any(|entry| entry.source_path.ends_with("pi3/pipe/pi3x_vo.py"))
    );
    assert_ne!(pi3.family, pi3x.family);
}

#[test]
fn triposr_runtime_geometry_keeps_integer_faces_with_storage_override() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::TripoSr).expect("triposr contract");

    let RuntimeGeometry::TripoSr(runtime) = &spec.runtime_geometry else {
        panic!("expected triposr runtime geometry");
    };

    let faces = runtime
        .outputs
        .iter()
        .find(|output| output.name == "triposr.mesh.faces")
        .expect("mesh faces output");
    assert_eq!(lux3d_core::TensorDType::I64, faces.runtime_dtype);
    assert_eq!(
        Some(lux3d_core::TensorDType::F32),
        faces.baseline_storage_dtype
    );
}

#[test]
fn vggt_contract_is_track_excluded_and_point_cloud_only() {
    let spec = ModelSpec::inspect(runtime_root(), ModelFamily::Vggt).expect("vggt contract");

    for expected_source in [
        "vggt/models/vggt.py",
        "vggt/heads/dpt_head.py",
        "vggt/layers/attention.py",
    ] {
        assert!(
            spec.vendor_sources
                .iter()
                .any(|entry| entry.source_path.ends_with(expected_source)),
            "missing vendor source `{expected_source}`"
        );
    }

    let RuntimeGeometry::Vggt(runtime) = &spec.runtime_geometry else {
        panic!("expected vggt runtime geometry");
    };
    for (name, dims) in [
        ("vggt.world_points", ["1", "S", "H", "W", "3"].as_slice()),
        ("vggt.world_points_conf", ["1", "S", "H", "W"].as_slice()),
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
        assert_eq!(TensorDType::F32, output.runtime_dtype);
    }
    assert!(
        runtime
            .outputs
            .iter()
            .all(|output| !output.name.starts_with("vggt.track")),
        "vggt contract must not expose track outputs"
    );

    let NeuralSpec::Vggt(neural) = &spec.neural else {
        panic!("expected vggt neural spec");
    };
    assert_eq!(None, neural.track_head);
    assert!(
        spec.exclusions
            .iter()
            .any(|entry| entry.item == "vggt_track_head"),
        "vggt_track_head must stay excluded"
    );
}
