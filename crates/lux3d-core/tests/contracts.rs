use std::path::PathBuf;

use lux3d_core::{ModelFamily, ModelSpec, RuntimeGeometry};

#[test]
fn pi3_and_pi3x_contracts_are_separately_scoped() {
    let pi3 = ModelSpec::inspect(PathBuf::from(r"H:\GitHub\LuxRT"), ModelFamily::Pi3)
        .expect("pi3 contract");
    let pi3x = ModelSpec::inspect(PathBuf::from(r"H:\GitHub\LuxRT"), ModelFamily::Pi3x)
        .expect("pi3x contract");

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
    let spec = ModelSpec::inspect(PathBuf::from(r"H:\GitHub\LuxRT"), ModelFamily::TripoSr)
        .expect("triposr contract");

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
