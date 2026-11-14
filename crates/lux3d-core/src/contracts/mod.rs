mod baseline;
mod common;
mod family;
mod pi3;
mod pi3x;
mod spec;
mod stages;
mod trellis2;
mod triposg;
mod triposr;
mod vggt;

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
    Trellis2ExportSpec, Trellis2GeometrySpec, Trellis2NeuralSpec, Trellis2PreprocessSpec,
    Trellis2RuntimeGeometry, TripoExportSpec, TripoGeometrySpec, TripoNeuralSpec,
    TripoPreprocessSpec, TripoSgExportSpec, TripoSgGeometrySpec, TripoSgNeuralSpec,
    TripoSgPreprocessSpec, TripoSgRmbgSpec, TripoSgRuntimeGeometry, VendorSource, VggtExportSpec,
    VggtGeometrySpec, VggtNeuralSpec, VggtPreprocessSpec, VggtRuntimeGeometry,
};
pub use stages::{ExportStage, GeometryStage, NeuralStage, PreprocessStage};
pub use trellis2::{Trellis2Mesh, Trellis2PipelineType, Trellis2PreparedImage, Trellis2RawInput};
pub use triposg::{TripoSgMesh, TripoSgPreparedImage, TripoSgRawInput, TripoSgSceneCode};
pub use triposr::{TripoMesh, TripoPreparedImage, TripoRawInput, TripoSceneCode};
pub use vggt::{VggtPointCloud, VggtPreparedBatch, VggtRawInput, VggtSceneCode};
