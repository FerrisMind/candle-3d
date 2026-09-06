pub mod contracts;
pub mod error;
pub mod export;
pub mod geometry;
pub mod neural;
pub mod preprocess;
pub mod runtime;
pub mod test_support;
pub mod weights;

pub use contracts::{
    BaselineGeometrySummary, BaselineManifest, BaselineParityTap, BaselinePreviewArtifact,
    BaselineSampleKind, BaselineTensorArtifact, ContractExclusion, ContractSourceOfTruth,
    ContractStage, ExportPlan, ExportSpec, GeometryBufferContract, GeometrySpec, LicenseEntry,
    LicensePolicy, ModelFamily, ModelSpec, NeuralSpec, Pi3ExportSpec, Pi3GeometrySpec,
    Pi3InputSource, Pi3NeuralSpec, Pi3PreprocessSpec, Pi3xExportSpec, Pi3xGeometrySpec,
    Pi3xNeuralSpec, Pi3xPreprocessSpec, PreprocessSpec, RgbRange, RuntimeArchitecture,
    RuntimeGeometry, RuntimeTensorContract, SourceDisposition, SpatialSize, TensorDType,
    TripoExportSpec, TripoGeometrySpec, TripoNeuralSpec, TripoPreprocessSpec, VendorSource,
};
pub use error::{Lux3dError, Result};
pub use weights::{
    CanonicalChecksumEntry, CanonicalChecksums, CanonicalWeightSet, CanonicalWeightSetPaths,
    CanonicalWeightsManifest, CanonicalizationPlan, FutureWeightLoader, ModelAssetOptions,
    RawWeightFormat, WeightLocator, ensure_canonical_weights, load_canonical_weights,
};

/// Enable candle's reduced-precision GPU GEMM opt-ins (TF32-class coopmat on
/// vulkan, f16-MMA coop on wgpu). Measured 1.9–2.5× on attention/linear GEMM
/// shapes; the pi3/pi3x warm timings and mesh-compare 6/6 were validated with
/// these paths active. Candle now defaults both to OFF so plain DType::F32
/// matmul keeps exact-fp32 accumulation — this product opts back in. Explicit
/// user environment overrides win.
pub fn enable_reduced_precision_gemm_opt_ins() {
    for (key, value) in [
        ("CANDLE_VULKAN_F32_UNALIGNED_COOPMAT", "1"),
        ("CANDLE_WGPU_COOP_MATMUL", "1"),
    ] {
        if std::env::var_os(key).is_none() {
            // SAFETY: called before any threads are spawned and before any
            // other thread reads the environment; no concurrent access.
            unsafe { std::env::set_var(key, value) };
        }
    }
}

