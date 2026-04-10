use crate::Result;

pub trait PreprocessStage<Raw, Prepared> {
    fn preprocess(&self, input: Raw) -> Result<Prepared>;
}

pub trait NeuralStage<Prepared, SceneCode> {
    fn infer(&self, input: &Prepared) -> Result<SceneCode>;
}

pub trait GeometryStage<SceneCode, GeometryArtifact> {
    fn assemble(&self, scene: &SceneCode) -> Result<GeometryArtifact>;
}

pub trait ExportStage<GeometryArtifact> {
    type Output;

    fn export(&self, artifact: &GeometryArtifact) -> Result<Self::Output>;
}
