mod baseline;
mod common;
mod family;
mod pi3;
mod pi3x;
mod spec;
mod stages;
mod triposr;

pub use baseline::{
    BaselineGeometrySummary, BaselineManifest, BaselinePreviewArtifact, BaselineSampleKind,
    BaselineTensorArtifact,
};
pub use common::{
    ContractSourceOfTruth, ContractStage, ExportPlan, NormalizationStats, Pi3InputSource, RgbRange,
    RuntimeArchitecture, SpatialSize, TensorContract, TensorDType,
};
pub use family::ModelFamily;
pub use pi3::{Pi3OptionalConditions, Pi3PointCloud, Pi3PreparedBatch, Pi3RawInput, Pi3SceneCode};
pub use pi3x::{Pi3xPointCloud, Pi3xPreparedBatch, Pi3xSceneCode, Pi3xVoPointCloud};
pub use spec::{
    BaselineParityTap, ContractExclusion, ExportSpec, GeometryBufferContract, GeometrySpec,
    LicenseEntry, LicensePolicy, ModelSpec, NeuralSpec, Pi3ExportSpec, Pi3GeometrySpec,
    Pi3NeuralSpec, Pi3PreprocessSpec, Pi3xExportSpec, Pi3xGeometrySpec, Pi3xNeuralSpec,
    Pi3xPreprocessSpec, PreprocessSpec, RuntimeGeometry, RuntimeTensorContract, SourceDisposition,
    TripoExportSpec, TripoGeometrySpec, TripoNeuralSpec, TripoPreprocessSpec, VendorSource,
};
pub use stages::{ExportStage, GeometryStage, NeuralStage, PreprocessStage};
pub use triposr::{TripoMesh, TripoPreparedImage, TripoRawInput, TripoSceneCode};
