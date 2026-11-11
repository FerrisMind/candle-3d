use lux3d_core::test_support::runtime_root;
use lux3d_core::{ModelFamily, ModelSpec, NeuralSpec, RuntimeGeometry, TensorDType};

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
