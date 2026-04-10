use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, contracts::ModelFamily, error::Lux3dError};

const CANONICAL_FILENAME: &str = "model.safetensors";
const RESOLVED_CONFIG_FILENAME: &str = "resolved_config.json";
const MANIFEST_FILENAME: &str = "manifest.json";
const CHECKSUMS_FILENAME: &str = "checksums.json";
const REQUIRED_PACKAGE_FILES: [&str; 4] = [
    CANONICAL_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    MANIFEST_FILENAME,
    CHECKSUMS_FILENAME,
];

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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ModelAssetOptions {
    pub canonical_dir: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
}

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
    raw_model_dir: PathBuf,
    canonical_root: PathBuf,
}

impl WeightLocator {
    pub fn new(raw_model_dir: PathBuf, canonical_root: PathBuf) -> Self {
        Self {
            raw_model_dir,
            canonical_root,
        }
    }

    pub fn locate(&self, family: ModelFamily) -> Result<CanonicalizationPlan> {
        match family {
            ModelFamily::Pi3 => self.locate_pi3(),
            ModelFamily::Pi3x => self.locate_pi3x(),
            ModelFamily::TripoSr => self.locate_triposr(),
        }
    }

    fn locate_pi3(&self) -> Result<CanonicalizationPlan> {
        let raw = self.ensure_exists(self.raw_model_dir.join(CANONICAL_FILENAME))?;
        Ok(CanonicalizationPlan {
            family: ModelFamily::Pi3,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![raw].into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "Pi3 code is BSD-3-Clause; local weights remain non-commercial."
                .to_string(),
        })
    }

    fn locate_pi3x(&self) -> Result<CanonicalizationPlan> {
        let raw = self.ensure_exists(self.raw_model_dir.join(CANONICAL_FILENAME))?;
        let config = self.ensure_exists(self.raw_model_dir.join("config.json"))?;
        Ok(CanonicalizationPlan {
            family: ModelFamily::Pi3x,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![raw, config].into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "Pi3X code is BSD-3-Clause; local weights remain non-commercial."
                .to_string(),
        })
    }

    fn locate_triposr(&self) -> Result<CanonicalizationPlan> {
        let ckpt = self.ensure_exists(self.raw_model_dir.join("model.ckpt"))?;
        let config = self.ensure_exists(self.raw_model_dir.join("config.yaml"))?;

        Ok(CanonicalizationPlan {
            family: ModelFamily::TripoSr,
            raw_format: RawWeightFormat::CheckpointFirst,
            raw_files: vec![ckpt, config].into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
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
    load_canonical_package(plan.family, plan.canonical_root.clone(), Some(plan))
}

pub fn load_canonical_weights(
    family: ModelFamily,
    options: ModelAssetOptions,
) -> Result<CanonicalWeightSetPaths> {
    if let Some(dir) = options.canonical_dir {
        return load_canonical_package(family, dir, None);
    }

    let cache_root = cache_root(&options)?;
    let package_dir = package_dir(&cache_root, family);

    if let Ok(existing) = load_canonical_package(family, package_dir.clone(), None) {
        return Ok(existing);
    }

    download_canonical_package(family, &cache_root, &package_dir)?;
    load_canonical_package(family, package_dir, None)
}

fn cache_root(options: &ModelAssetOptions) -> Result<PathBuf> {
    if let Some(cache_dir) = &options.cache_dir {
        return Ok(cache_dir.clone());
    }

    dirs::cache_dir()
        .map(|root| root.join("LuxRT").join("models"))
        .ok_or(Lux3dError::CacheDirUnavailable)
}

fn package_dir(cache_root: &Path, family: ModelFamily) -> PathBuf {
    cache_root.join(family.as_str())
}

fn download_canonical_package(
    family: ModelFamily,
    cache_root: &Path,
    package_dir: &Path,
) -> Result<()> {
    fs::create_dir_all(package_dir).map_err(|source| Lux3dError::ModelAssetIo {
        path: package_dir.to_path_buf(),
        source,
    })?;

    let hub_cache = cache_root.join("hf-hub");
    fs::create_dir_all(&hub_cache).map_err(|source| Lux3dError::ModelAssetIo {
        path: hub_cache.clone(),
        source,
    })?;

    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let api = ApiBuilder::from_env()
        .with_cache_dir(hub_cache)
        .with_token(token)
        .with_user_agent("ferrismind", "LuxRT")
        .with_progress(false)
        .build()
        .map_err(|source| Lux3dError::ModelAssetResolution {
            message: format!(
                "failed to initialize Hugging Face API client for `{}`: {source}",
                family.huggingface_repo_id()
            ),
        })?;
    let repo = api.model(family.huggingface_repo_id().to_string());

    for filename in REQUIRED_PACKAGE_FILES {
        let fetched = repo
            .get(filename)
            .map_err(|source| Lux3dError::ModelAssetResolution {
                message: format!(
                    "failed to download `{filename}` from `{}`: {source}",
                    family.huggingface_repo_id()
                ),
            })?;
        copy_asset(&fetched, &package_dir.join(filename))?;
    }

    Ok(())
}

fn copy_asset(source: &Path, target: &Path) -> Result<()> {
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|source_err| Lux3dError::ModelAssetIo {
            path: parent.to_path_buf(),
            source: source_err,
        })?;
    }
    fs::copy(source, target).map_err(|source_err| Lux3dError::ModelAssetIo {
        path: target.to_path_buf(),
        source: source_err,
    })?;
    Ok(())
}

fn load_canonical_package(
    family: ModelFamily,
    package_dir: PathBuf,
    expected_plan: Option<&CanonicalizationPlan>,
) -> Result<CanonicalWeightSetPaths> {
    let manifest_file = package_dir.join(MANIFEST_FILENAME);
    let checksums_file = package_dir.join(CHECKSUMS_FILENAME);
    let canonical_file = package_dir.join(CANONICAL_FILENAME);
    let resolved_config_file = package_dir.join(RESOLVED_CONFIG_FILENAME);

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
    validate_runtime_manifest(expected_plan, family, &manifest)?;

    let checksums = read_checksums(&checksums_file)?;
    validate_checksums(&package_dir, &checksums)?;

    Ok(CanonicalWeightSetPaths {
        manifest,
        canonical_file,
        resolved_config_file,
        manifest_file,
        checksums_file,
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

fn validate_runtime_manifest(
    expected_plan: Option<&CanonicalizationPlan>,
    family: ModelFamily,
    manifest: &CanonicalWeightsManifest,
) -> Result<()> {
    if manifest.family != family {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest family `{}` does not match requested family `{}`",
                manifest.family, family
            ),
        });
    }

    let normalized_canonical = normalize_manifest_path(&manifest.canonical_file);
    if normalized_canonical != CANONICAL_FILENAME {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest canonical file `{}` does not point to `{}`",
                manifest.canonical_file, CANONICAL_FILENAME
            ),
        });
    }

    let normalized_config = normalize_manifest_path(&manifest.resolved_config_file);
    if normalized_config != RESOLVED_CONFIG_FILENAME {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest resolved config `{}` does not point to `{}`",
                manifest.resolved_config_file, RESOLVED_CONFIG_FILENAME
            ),
        });
    }

    if let Some(plan) = expected_plan {
        validate_plan_provenance(plan, manifest)?;
    }

    Ok(())
}

fn validate_plan_provenance(
    plan: &CanonicalizationPlan,
    manifest: &CanonicalWeightsManifest,
) -> Result<()> {
    let expected_raw_files = plan
        .raw_files
        .iter()
        .map(|path| normalize_path_file_name(path))
        .collect::<Vec<_>>();
    let manifest_raw_files = manifest
        .raw_files
        .iter()
        .map(|path| normalize_manifest_path(path))
        .collect::<Vec<_>>();

    if manifest_raw_files != expected_raw_files {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest raw files {:?} do not match plan {:?}",
                manifest.raw_files, expected_raw_files
            ),
        });
    }

    let expected_canonical = plan
        .canonical_root
        .join(&plan.canonical_filename)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(CANONICAL_FILENAME)
        .to_string();
    if expected_canonical != normalize_manifest_path(&manifest.canonical_file) {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest canonical file `{}` does not match plan `{}`",
                manifest.canonical_file, expected_canonical
            ),
        });
    }

    let mut expected_checksums = BTreeMap::new();
    for raw_file in plan.raw_files.iter() {
        let key = normalize_path_file_name(raw_file);
        let digest = sha256_file(raw_file)?;
        expected_checksums.insert(key, digest);
    }

    let manifest_checksums = manifest
        .source_checksums
        .iter()
        .map(|(path, digest)| (normalize_manifest_path(path), digest.clone()))
        .collect::<BTreeMap<_, _>>();

    if manifest_checksums != expected_checksums {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "manifest source checksums {:?} do not match plan {:?}",
                manifest.source_checksums, expected_checksums
            ),
        });
    }

    Ok(())
}

fn validate_checksums(package_dir: &Path, checksums: &CanonicalChecksums) -> Result<()> {
    for entry in &checksums.files {
        let normalized = normalize_manifest_path(&entry.relative_path);
        let absolute_path = package_dir.join(&normalized);
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

fn normalize_manifest_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.replace('\\', "/"))
}

fn normalize_path_file_name(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.to_string_lossy().replace('\\', "/"))
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
