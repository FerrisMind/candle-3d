use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, contracts::ModelFamily, error::Lux3dError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawWeightFormat {
    SafetensorsDirect,
    CheckpointFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FutureWeightLoader {
    CandleMmapSafetensors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalizationPlan {
    pub family: ModelFamily,
    pub raw_format: RawWeightFormat,
    pub raw_files: Box<[PathBuf]>,
    pub canonical_root: PathBuf,
    pub canonical_filename: String,
    pub future_loader: FutureWeightLoader,
    pub license_note: String,
}

pub type CanonicalWeightSet = CanonicalizationPlan;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalChecksumEntry {
    pub relative_path: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalChecksums {
    pub files: Vec<CanonicalChecksumEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalWeightsManifest {
    pub family: ModelFamily,
    pub normalizer_version: u32,
    pub raw_files: Box<[String]>,
    pub canonical_file: String,
    pub resolved_config_file: String,
    pub tensor_count: usize,
    pub dtype_histogram: BTreeMap<String, usize>,
    pub source_checksums: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalWeightSetPaths {
    pub manifest: CanonicalWeightsManifest,
    pub canonical_file: PathBuf,
    pub resolved_config_file: PathBuf,
    pub manifest_file: PathBuf,
    pub checksums_file: PathBuf,
}

impl CanonicalWeightSetPaths {
    /// # Safety
    ///
    /// The returned builder is backed by a memory-mapped safetensors file.
    /// Callers must ensure the mapped file outlives every tensor/module created from the builder
    /// and that the file is not mutated while the mapping is in use.
    pub unsafe fn var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<VarBuilder<'static>> {
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.canonical_file),
                dtype,
                device,
            )
        }
    }

    pub fn read_resolved_config_json(&self) -> Result<serde_json::Value> {
        let body = fs::read_to_string(&self.resolved_config_file).map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: self.resolved_config_file.clone(),
                source,
            }
        })?;
        serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalManifestJson {
            path: self.resolved_config_file.clone(),
            source,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightLocator {
    repo_root: PathBuf,
}

impl WeightLocator {
    pub fn new(repo_root: PathBuf) -> Self {
        Self { repo_root }
    }

    pub fn locate(&self, family: ModelFamily) -> Result<CanonicalizationPlan> {
        match family {
            ModelFamily::Pi3 => self.locate_pi3(),
            ModelFamily::Pi3x => self.locate_pi3x(),
            ModelFamily::TripoSr => self.locate_triposr(),
        }
    }

    fn locate_pi3(&self) -> Result<CanonicalizationPlan> {
        let raw = self.ensure_exists(
            self.repo_root
                .join("3d")
                .join("models")
                .join("yyfz233-Pi3")
                .join("model.safetensors"),
        )?;
        Ok(CanonicalizationPlan {
            family: ModelFamily::Pi3,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![raw].into_boxed_slice(),
            canonical_root: self
                .repo_root
                .join("3d")
                .join("canonical-weights")
                .join(ModelFamily::Pi3.as_str()),
            canonical_filename: "model.safetensors".to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "Pi3 code is BSD-3-Clause; local weights remain non-commercial."
                .to_string(),
        })
    }

    fn locate_pi3x(&self) -> Result<CanonicalizationPlan> {
        let raw = self.ensure_exists(
            self.repo_root
                .join("3d")
                .join("models")
                .join("yyfz233-Pi3X")
                .join("model.safetensors"),
        )?;
        let config = self.ensure_exists(
            self.repo_root
                .join("3d")
                .join("models")
                .join("yyfz233-Pi3X")
                .join("config.json"),
        )?;
        Ok(CanonicalizationPlan {
            family: ModelFamily::Pi3x,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![raw, config].into_boxed_slice(),
            canonical_root: self
                .repo_root
                .join("3d")
                .join("canonical-weights")
                .join(ModelFamily::Pi3x.as_str()),
            canonical_filename: "model.safetensors".to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "Pi3X code is BSD-3-Clause; local weights remain non-commercial."
                .to_string(),
        })
    }

    fn locate_triposr(&self) -> Result<CanonicalizationPlan> {
        let model_root = self
            .repo_root
            .join("3d")
            .join("models")
            .join("stabilityai-TripoSR");
        let ckpt = self.ensure_exists(model_root.join("model.ckpt"))?;
        let config = self.ensure_exists(model_root.join("config.yaml"))?;

        Ok(CanonicalizationPlan {
            family: ModelFamily::TripoSr,
            raw_format: RawWeightFormat::CheckpointFirst,
            raw_files: vec![ckpt, config].into_boxed_slice(),
            canonical_root: self
                .repo_root
                .join("3d")
                .join("canonical-weights")
                .join(ModelFamily::TripoSr.as_str()),
            canonical_filename: "model.safetensors".to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note:
                "TripoSR code and weights are MIT; canonical safetensors stay outside vendor tree."
                    .to_string(),
        })
    }

    fn ensure_exists(&self, path: PathBuf) -> Result<PathBuf> {
        if path.is_file() {
            Ok(path)
        } else {
            Err(Lux3dError::MissingWeightFile { path })
        }
    }
}

pub fn ensure_canonical_weights(plan: &CanonicalizationPlan) -> Result<CanonicalWeightSetPaths> {
    let repo_root = repo_root_from_plan(plan)?;
    let manifest_file = plan.canonical_root.join("manifest.json");
    let checksums_file = plan.canonical_root.join("checksums.json");
    let canonical_file = plan.canonical_root.join(&plan.canonical_filename);
    let resolved_config_file = plan.canonical_root.join("resolved_config.json");

    for required in [
        manifest_file.clone(),
        checksums_file.clone(),
        canonical_file.clone(),
        resolved_config_file.clone(),
    ] {
        if !required.is_file() {
            return Err(Lux3dError::MissingCanonicalArtifact { path: required });
        }
    }

    let manifest = read_manifest(&manifest_file)?;
    validate_manifest(plan, &manifest)?;

    let checksums = read_checksums(&checksums_file)?;
    validate_checksums(&repo_root, &checksums)?;

    Ok(CanonicalWeightSetPaths {
        manifest,
        canonical_file,
        resolved_config_file,
        manifest_file,
        checksums_file,
    })
}

pub fn load_canonical_weights(
    family: ModelFamily,
    repo_root: PathBuf,
) -> Result<CanonicalWeightSetPaths> {
    let locator = WeightLocator::new(repo_root);
    let plan = locator.locate(family)?;
    ensure_canonical_weights(&plan)
}

fn repo_root_from_plan(plan: &CanonicalizationPlan) -> Result<PathBuf> {
    plan.canonical_root
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .ok_or_else(|| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "could not derive repo root from canonical root `{}`",
                plan.canonical_root.display()
            ),
        })
}

fn read_manifest(path: &Path) -> Result<CanonicalWeightsManifest> {
    let body = fs::read_to_string(path).map_err(|source| Lux3dError::CanonicalManifestIo {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalManifestJson {
        path: path.to_path_buf(),
        source,
    })
}

fn read_checksums(path: &Path) -> Result<CanonicalChecksums> {
    let body = fs::read_to_string(path).map_err(|source| Lux3dError::CanonicalChecksumsIo {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalChecksumsJson {
        path: path.to_path_buf(),
        source,
    })
}

fn validate_manifest(
    plan: &CanonicalizationPlan,
    manifest: &CanonicalWeightsManifest,
) -> Result<()> {
    if manifest.family != plan.family {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest family `{}` does not match plan family `{}`",
                manifest.family, plan.family
            ),
        });
    }

    let expected_raw_files = plan
        .raw_files
        .iter()
        .map(|path| relative_to_repo(path, &repo_root_from_plan(plan)?))
        .collect::<Result<Vec<_>>>()?;
    if manifest.raw_files.as_ref() != expected_raw_files.as_slice() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest raw files {:?} do not match plan {:?}",
                manifest.raw_files, expected_raw_files
            ),
        });
    }

    let expected_canonical = format!(
        "3d/canonical-weights/{}/{}",
        plan.family.as_str(),
        plan.canonical_filename
    );
    if manifest.canonical_file != expected_canonical {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest canonical file `{}` does not match expected `{}`",
                manifest.canonical_file, expected_canonical
            ),
        });
    }

    let expected_config = format!(
        "3d/canonical-weights/{}/resolved_config.json",
        plan.family.as_str()
    );
    if manifest.resolved_config_file != expected_config {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest resolved config `{}` does not match expected `{}`",
                manifest.resolved_config_file, expected_config
            ),
        });
    }

    Ok(())
}

fn validate_checksums(repo_root: &Path, checksums: &CanonicalChecksums) -> Result<()> {
    for entry in &checksums.files {
        let absolute_path = repo_root.join(PathBuf::from(&entry.relative_path));
        if !absolute_path.is_file() {
            return Err(Lux3dError::MissingCanonicalArtifact {
                path: absolute_path,
            });
        }
        let metadata =
            fs::metadata(&absolute_path).map_err(|source| Lux3dError::CanonicalChecksumsIo {
                path: absolute_path.clone(),
                source,
            })?;
        if metadata.len() != entry.size_bytes {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "size mismatch for `{}`: manifest={}, actual={}",
                    entry.relative_path,
                    entry.size_bytes,
                    metadata.len()
                ),
            });
        }
        let digest = sha256_file(&absolute_path)?;
        if digest != entry.sha256 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "sha256 mismatch for `{}`: manifest={}, actual={}",
                    entry.relative_path, entry.sha256, digest
                ),
            });
        }
    }
    Ok(())
}

fn relative_to_repo(path: &Path, repo_root: &Path) -> Result<String> {
    path.strip_prefix(repo_root)
        .map(|relative| relative.to_string_lossy().replace('\\', "/"))
        .map_err(|_| Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "path `{}` does not live under repo root `{}`",
                path.display(),
                repo_root.display()
            ),
        })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    let bytes = fs::read(path).map_err(|source| Lux3dError::CanonicalChecksumsIo {
        path: path.to_path_buf(),
        source,
    })?;
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}
