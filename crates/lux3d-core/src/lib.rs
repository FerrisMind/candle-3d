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
