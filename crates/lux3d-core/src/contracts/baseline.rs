use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{Result, error::Lux3dError};

use super::{ModelFamily, ModelSpec, SpatialSize, TensorDType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BaselineSampleKind {
    Golden,
    Smoke,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselineTensorArtifact {
    pub name: String,
    pub semantic: String,
    pub dtype: TensorDType,
    pub shape: Box<[usize]>,
    pub storage_relpath: PathBuf,
    pub storage_key: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BaselineGeometrySummary {
    pub device: String,
    pub sampled_frames: Option<usize>,
    pub interval: Option<usize>,
    pub target_size: Option<SpatialSize>,
    pub scene_codes_shape: Option<Box<[usize]>>,
    pub point_count: Option<usize>,
    pub mask_true_count: Option<usize>,
    pub vertex_count: Option<usize>,
    pub face_count: Option<usize>,
    pub bbox_min: Option<[f32; 3]>,
    pub bbox_max: Option<[f32; 3]>,
    pub first_point_samples: Box<[[f32; 3]]>,
    pub first_color_samples: Box<[[f32; 3]]>,
    pub first_vertex_samples: Box<[[f32; 3]]>,
    pub mc_resolution: Option<u32>,
    pub mc_threshold: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BaselinePreviewArtifact {
    pub label: String,
    pub relative_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BaselineManifest {
    pub family: ModelFamily,
    pub sample_id: String,
    pub sample_kind: BaselineSampleKind,
    pub source_path: PathBuf,
    pub light_root: PathBuf,
    pub heavy_root: Option<PathBuf>,
    pub weight_files: Box<[PathBuf]>,
    pub tensor_artifacts: Box<[BaselineTensorArtifact]>,
    pub geometry_summary: BaselineGeometrySummary,
    pub summary_relpath: PathBuf,
    pub checksums_relpath: PathBuf,
    pub previews: Box<[BaselinePreviewArtifact]>,
    pub notes: Box<[String]>,
}

impl BaselineManifest {
    pub fn from_path(path: &Path) -> Result<Self> {
        let body = fs::read_to_string(path).map_err(|source| Lux3dError::BaselineManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&body).map_err(|source| Lux3dError::BaselineManifestJson {
            path: path.to_path_buf(),
            source,
        })
    }

    pub fn validate_against_model_spec(&self, spec: &ModelSpec) -> Result<()> {
        if self.family != spec.family {
            return Err(Lux3dError::BaselineValidation {
                message: format!(
                    "manifest family `{}` does not match spec family `{}`",
                    self.family, spec.family
                ),
            });
        }

        if normalize_weight_files(self.weight_files.as_ref())
            != normalize_weight_files(spec.weight_plan.raw_files.as_ref())
        {
            return Err(Lux3dError::BaselineValidation {
                message: format!(
                    "manifest weight files {:?} do not match spec {:?}",
                    self.weight_files, spec.weight_plan.raw_files
                ),
            });
        }

        if self.sample_kind == BaselineSampleKind::Smoke {
            if !self.tensor_artifacts.is_empty() {
                return Err(Lux3dError::BaselineValidation {
                    message: format!(
                        "smoke manifest `{}` must not contain tensor artifacts",
                        self.sample_id
                    ),
                });
            }
            return Ok(());
        }

        if self.sample_id != spec.golden_baseline_sample {
            return Err(Lux3dError::BaselineValidation {
                message: format!(
                    "golden manifest `{}` does not match contract golden sample `{}`",
                    self.sample_id, spec.golden_baseline_sample
                ),
            });
        }

        let by_name = self
            .tensor_artifacts
            .iter()
            .map(|artifact| (artifact.name.as_str(), artifact))
            .collect::<BTreeMap<_, _>>();

        for tap in spec.baseline_parity_taps.iter() {
            let Some(artifact) = by_name.get(tap.name.as_str()) else {
                return Err(Lux3dError::BaselineValidation {
                    message: format!(
                        "golden manifest `{}` is missing tensor artifact `{}`",
                        self.sample_id, tap.name
                    ),
                });
            };

            if artifact.semantic != tap.semantic {
                return Err(Lux3dError::BaselineValidation {
                    message: format!(
                        "artifact `{}` semantic `{}` does not match tap `{}`",
                        artifact.name, artifact.semantic, tap.semantic
                    ),
                });
            }

            if artifact.dtype != tap.storage_dtype {
                return Err(Lux3dError::BaselineValidation {
                    message: format!(
                        "artifact `{}` dtype `{:?}` does not match tap `{:?}`",
                        artifact.name, artifact.dtype, tap.storage_dtype
                    ),
                });
            }

            if artifact.shape.as_ref() != tap.observed_shape.as_slice() {
                return Err(Lux3dError::BaselineValidation {
                    message: format!(
                        "artifact `{}` shape `{:?}` does not match tap shape `{:?}`",
                        artifact.name, artifact.shape, tap.observed_shape
                    ),
                });
            }
        }

        Ok(())
    }
}

fn normalize_weight_files(paths: &[PathBuf]) -> Vec<String> {
    paths
        .iter()
        .map(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| path.to_string_lossy().into_owned())
        })
        .collect()
}
