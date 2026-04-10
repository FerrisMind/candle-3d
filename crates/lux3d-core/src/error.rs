use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Lux3dError {
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),
    #[error("missing required weight file at `{path}`")]
    MissingWeightFile { path: PathBuf },
    #[error("failed to read contract file `{path}`: {source}")]
    ContractFileIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse contract file `{path}`: {source}")]
    ContractFileJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("contract validation failed: {message}")]
    ContractValidation { message: String },
    #[error("failed to read baseline manifest `{path}`: {source}")]
    BaselineManifestIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse baseline manifest `{path}`: {source}")]
    BaselineManifestJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("baseline validation failed: {message}")]
    BaselineValidation { message: String },
    #[error("missing required canonical artifact at `{path}`")]
    MissingCanonicalArtifact { path: PathBuf },
    #[error("failed to read canonical manifest `{path}`: {source}")]
    CanonicalManifestIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse canonical manifest `{path}`: {source}")]
    CanonicalManifestJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to read canonical checksums `{path}`: {source}")]
    CanonicalChecksumsIo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse canonical checksums `{path}`: {source}")]
    CanonicalChecksumsJson {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("canonical weights validation failed: {message}")]
    CanonicalWeightsValidation { message: String },
}

pub type Result<T> = std::result::Result<T, Lux3dError>;
