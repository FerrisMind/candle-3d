mod auxiliary;
mod quantize;
pub use auxiliary::{
    AuxiliaryWeightSetPaths, AuxiliaryWeightsManifest, load_auxiliary_canonical_weights,
};
pub(crate) use candle_nn::VarBuilder;
pub use quantize::{
    QuantizeOptions, QuantizeReport, dtype_from_name, dtype_name, quantize_auxiliary_checkpoint,
    quantize_checkpoint,
};

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device, quantized::gguf_file};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Result, contracts::ModelFamily, error::Lux3dError};

pub(crate) const CANONICAL_FILENAME: &str = "model.safetensors";
pub(crate) const GGUF_FILENAME: &str = "model.gguf";
pub(crate) const GGUF_FAMILY_KEY: &str = "lux3d.family";
pub(crate) const GGUF_RESOLVED_CONFIG_KEY: &str = "lux3d.resolved_config";
pub(crate) const RESOLVED_CONFIG_FILENAME: &str = "resolved_config.json";
pub(crate) const MANIFEST_FILENAME: &str = "manifest.json";
pub(crate) const CHECKSUMS_FILENAME: &str = "checksums.json";
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
    /// `Some` for a standalone `model.gguf` that carries `resolved_config`
    /// in its metadata; `None` for safetensors packages.
    pub embedded_resolved_config: Option<serde_json::Value>,
}

impl CanonicalWeightSetPaths {
    /// # Safety
    ///
    /// The returned builder is backed by a memory-mapped safetensors file.
    /// Callers must ensure the mapped file outlives every tensor/module created from the builder
    /// and that the file is not mutated while the mapping is in use.
    ///
    /// A sibling `model.gguf` takes precedence: GGUF containers are read
    /// lazily through [`VarBuilder::from_gguf`], which keeps quantized tensors
    /// packed for `get_quantized` and dequantizes the rest on demand. For a
    /// standalone package [`Self::canonical_file`] is the GGUF itself.
    pub unsafe fn var_builder(
        &self,
        dtype: DType,
        device: &Device,
    ) -> candle_core::Result<VarBuilder<'static>> {
        let gguf_file = if is_gguf_path(&self.canonical_file) {
            self.canonical_file.clone()
        } else {
            self.canonical_file.with_file_name(GGUF_FILENAME)
        };
        if gguf_file.is_file() {
            return VarBuilder::from_gguf(&gguf_file, dtype, device);
        }
        unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&self.canonical_file),
                dtype,
                device,
            )
        }
    }

    pub fn read_resolved_config_json(&self) -> Result<serde_json::Value> {
        if let Some(config) = &self.embedded_resolved_config {
            return Ok(config.clone());
        }
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
            ModelFamily::TripoSg => self.locate_triposg(),
            ModelFamily::Vggt => self.locate_vggt(),
            ModelFamily::Trellis2 => self.locate_trellis2(),
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

    fn locate_vggt(&self) -> Result<CanonicalizationPlan> {
        let raw = self.ensure_exists(self.raw_model_dir.join(CANONICAL_FILENAME))?;
        let config = self.ensure_exists(self.raw_model_dir.join("config.json"))?;
        Ok(CanonicalizationPlan {
            family: ModelFamily::Vggt,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![raw, config].into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "VGGT code and original weights are CC-BY-NC-4.0 (non-commercial); the commercial checkpoint is distributed separately."
                .to_string(),
        })
    }

    /// TripoSG ships as a diffusers-style repo with three safetensors shards
    /// that share filenames across sibling directories. The raw file order
    /// below is the canonical merge order consumed by the Python normalizer.
    fn locate_triposg(&self) -> Result<CanonicalizationPlan> {
        let transformer = self.ensure_exists(
            self.raw_model_dir
                .join("transformer")
                .join("diffusion_pytorch_model.safetensors"),
        )?;
        let vae = self.ensure_exists(
            self.raw_model_dir
                .join("vae")
                .join("diffusion_pytorch_model.safetensors"),
        )?;
        let image_encoder = self.ensure_exists(
            self.raw_model_dir
                .join("image_encoder_dinov2")
                .join("model.safetensors"),
        )?;
        let model_index = self.ensure_exists(self.raw_model_dir.join("model_index.json"))?;
        let preprocessor = self.ensure_exists(
            self.raw_model_dir
                .join("feature_extractor_dinov2")
                .join("preprocessor_config.json"),
        )?;
        let image_encoder_config = self.ensure_exists(
            self.raw_model_dir
                .join("image_encoder_dinov2")
                .join("config.json"),
        )?;
        let scheduler_config = self.ensure_exists(
            self.raw_model_dir
                .join("scheduler")
                .join("scheduler_config.json"),
        )?;
        let transformer_config =
            self.ensure_exists(self.raw_model_dir.join("transformer").join("config.json"))?;
        let vae_config = self.ensure_exists(self.raw_model_dir.join("vae").join("config.json"))?;

        Ok(CanonicalizationPlan {
            family: ModelFamily::TripoSg,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: vec![
                transformer,
                vae,
                image_encoder,
                model_index,
                preprocessor,
                image_encoder_config,
                scheduler_config,
                transformer_config,
                vae_config,
            ]
            .into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note:
                "TripoSG code and weights are MIT; canonical safetensors stay outside vendor tree."
                    .to_string(),
        })
    }

    /// TRELLIS.2-4B ships as a flat per-component layout: each stage is a
    /// `<name>.safetensors` + sibling `<name>.json` config pair under `ckpts/`,
    /// the dense sparse-structure decoder lives under `external/ss_dec/ckpts/`,
    /// and the DINOv3 conditioner under `external/dinov3/`. `pipeline.json`
    /// carries the sampler params, model table and normalization vectors.
    /// The raw file order below is the canonical merge order consumed by the
    /// Python normalizer (`tools/python_baseline/normalize_weights.py`).
    fn locate_trellis2(&self) -> Result<CanonicalizationPlan> {
        let ckpt_pairs = [
            "ss_flow_img_dit_1_3B_64_bf16",
            "slat_flow_img2shape_dit_1_3B_512_bf16",
            "slat_flow_img2shape_dit_1_3B_1024_bf16",
            "slat_flow_imgshape2tex_dit_1_3B_512_bf16",
            "slat_flow_imgshape2tex_dit_1_3B_1024_bf16",
            "shape_dec_next_dc_f16c32_fp16",
            "tex_dec_next_dc_f16c32_fp16",
        ];

        let mut raw_files = Vec::new();
        for stem in ckpt_pairs {
            raw_files.push(
                self.ensure_exists(
                    self.raw_model_dir
                        .join("ckpts")
                        .join(format!("{stem}.json")),
                )?,
            );
            raw_files.push(
                self.ensure_exists(
                    self.raw_model_dir
                        .join("ckpts")
                        .join(format!("{stem}.safetensors")),
                )?,
            );
        }
        raw_files.push(
            self.ensure_exists(
                self.raw_model_dir
                    .join("external")
                    .join("ss_dec")
                    .join("ckpts")
                    .join("ss_dec_conv3d_16l8_fp16.json"),
            )?,
        );
        raw_files.push(
            self.ensure_exists(
                self.raw_model_dir
                    .join("external")
                    .join("ss_dec")
                    .join("ckpts")
                    .join("ss_dec_conv3d_16l8_fp16.safetensors"),
            )?,
        );
        raw_files.push(
            self.ensure_exists(
                self.raw_model_dir
                    .join("external")
                    .join("dinov3")
                    .join("config.json"),
            )?,
        );
        raw_files.push(
            self.ensure_exists(
                self.raw_model_dir
                    .join("external")
                    .join("dinov3")
                    .join("model.safetensors"),
            )?,
        );
        raw_files.push(self.ensure_exists(self.raw_model_dir.join("pipeline.json"))?);

        Ok(CanonicalizationPlan {
            family: ModelFamily::Trellis2,
            raw_format: RawWeightFormat::SafetensorsDirect,
            raw_files: raw_files.into_boxed_slice(),
            canonical_root: self.canonical_root.clone(),
            canonical_filename: CANONICAL_FILENAME.to_string(),
            future_loader: FutureWeightLoader::CandleMmapSafetensors,
            license_note: "TRELLIS.2 code and the microsoft/TRELLIS.2-4B model card are MIT; the bundled DINOv3 encoder is a separately licensed gated Meta model; RMBG-2.0/BiRefNet ships as the separate `rmbg-2.0` package under CC-BY-NC-4.0."
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
    if let Some(path) = options.canonical_dir {
        if path.is_file() {
            if is_gguf_path(&path) {
                return load_standalone_gguf(family, path);
            }
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "`--model-path` file `{}` is not a GGUF; pass a package directory or a `.gguf` file",
                    path.display()
                ),
            });
        }
        return load_canonical_package(family, path, None);
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
    let gguf_file = package_dir.join(GGUF_FILENAME);

    // A standalone quantized package contains only `model.gguf`; it embeds
    // the resolved config in GGUF metadata. Normalization plans still require
    // the safetensors source.
    if !canonical_file.is_file() {
        if gguf_file.is_file() && expected_plan.is_none() {
            return load_standalone_gguf(family, gguf_file);
        }
        return Err(Lux3dError::MissingCanonicalArtifact {
            path: canonical_file,
        });
    }

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
        embedded_resolved_config: None,
    })
}

fn load_standalone_gguf(
    family: ModelFamily,
    gguf_file: PathBuf,
) -> Result<CanonicalWeightSetPaths> {
    let standalone = read_standalone_gguf(&gguf_file, family.as_str())?;
    let manifest = CanonicalWeightsManifest {
        family,
        normalizer_version: 0,
        raw_files: Box::default(),
        canonical_file: GGUF_FILENAME.to_string(),
        resolved_config_file: GGUF_FILENAME.to_string(),
        tensor_count: standalone.tensor_count,
        dtype_histogram: standalone.dtype_histogram,
        source_checksums: BTreeMap::new(),
    };

    Ok(CanonicalWeightSetPaths {
        manifest,
        canonical_file: gguf_file.clone(),
        resolved_config_file: gguf_file.clone(),
        manifest_file: gguf_file.clone(),
        checksums_file: gguf_file,
        embedded_resolved_config: Some(standalone.resolved_config),
    })
}

pub(crate) struct StandaloneGgufMetadata {
    pub resolved_config: serde_json::Value,
    pub tensor_count: usize,
    pub dtype_histogram: BTreeMap<String, usize>,
}

/// Read the metadata required to use a GGUF as a standalone package.
pub(crate) fn read_standalone_gguf(
    gguf_file: &Path,
    expected_family: &str,
) -> Result<StandaloneGgufMetadata> {
    let mut file = fs::File::open(gguf_file).map_err(|source| Lux3dError::ModelAssetIo {
        path: gguf_file.to_path_buf(),
        source,
    })?;
    let content = gguf_file::Content::read(&mut file).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "failed to read standalone GGUF `{}`: {source}",
                gguf_file.display()
            ),
        }
    })?;
    drop(file);

    if let Some(stated) = gguf_metadata_string(&content, GGUF_FAMILY_KEY) {
        if stated != expected_family {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "GGUF `{}` is for family `{stated}`, requested `{expected_family}`",
                    gguf_file.display()
                ),
            });
        }
    }
    let Some(resolved) = gguf_metadata_string(&content, GGUF_RESOLVED_CONFIG_KEY) else {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "standalone GGUF `{}` has no embedded `{GGUF_RESOLVED_CONFIG_KEY}`; \
                 quantize from a canonical package that contains `{RESOLVED_CONFIG_FILENAME}`",
                gguf_file.display()
            ),
        });
    };
    let resolved_config: serde_json::Value = serde_json::from_str(resolved).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "GGUF `{}` has invalid embedded resolved config: {source}",
                gguf_file.display()
            ),
        }
    })?;

    let mut dtype_histogram = BTreeMap::new();
    for info in content.tensor_infos.values() {
        *dtype_histogram
            .entry(quantize::dtype_name(info.ggml_dtype).to_string())
            .or_insert(0usize) += 1;
    }

    Ok(StandaloneGgufMetadata {
        resolved_config,
        tensor_count: content.tensor_infos.len(),
        dtype_histogram,
    })
}

fn gguf_metadata_string<'a>(content: &'a gguf_file::Content, key: &str) -> Option<&'a str> {
    match content.metadata.get(key) {
        Some(gguf_file::Value::String(value)) => Some(value.as_str()),
        _ => None,
    }
}

pub(crate) fn is_gguf_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
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

pub(crate) fn read_checksums(path: &Path) -> Result<CanonicalChecksums> {
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
    let expected_raw_files = provenance_keys(plan.raw_files.iter().map(|path| path.as_path()));
    let manifest_raw_files = provenance_keys(
        manifest
            .raw_files
            .iter()
            .map(|entry| Path::new(entry.as_str())),
    );

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
    for (key, raw_file) in expected_raw_files.iter().zip(plan.raw_files.iter()) {
        let digest = sha256_file(raw_file)?;
        expected_checksums.insert(key.clone(), digest);
    }

    let manifest_checksum_keys = provenance_keys(
        manifest
            .source_checksums
            .keys()
            .map(|key| Path::new(key.as_str())),
    );
    let manifest_checksums = manifest
        .source_checksums
        .values()
        .zip(manifest_checksum_keys)
        .map(|(digest, key)| (key, digest.clone()))
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

/// Compute provenance keys for raw weight files.
///
/// Files whose basename is unique across the list keep the historical bare
/// `<basename>` key. Basenames that occur more than once are keyed as
/// `<parent-dir-name>/<basename>` so sharded raw layouts (e.g. the TripoSG
/// `transformer/` and `vae/` shards that share a filename) stay
/// distinguishable. Every pre-existing family has unique basenames, so this is
/// a strict generalization of the old basename-only keys; keys for existing
/// manifests are byte-identical.
pub(crate) fn provenance_keys<'a, I>(raw_files: I) -> Vec<String>
where
    I: IntoIterator<Item = &'a Path>,
{
    let files: Vec<&Path> = raw_files.into_iter().collect();
    let mut basename_counts: BTreeMap<String, usize> = BTreeMap::new();
    for path in &files {
        *basename_counts.entry(raw_file_basename(path)).or_default() += 1;
    }
    files
        .iter()
        .map(|path| {
            let basename = raw_file_basename(path);
            if basename_counts.get(&basename).copied().unwrap_or_default() > 1 {
                match path
                    .parent()
                    .and_then(|parent| parent.file_name())
                    .and_then(|value| value.to_str())
                {
                    Some(parent) => format!("{parent}/{basename}"),
                    None => basename,
                }
            } else {
                basename
            }
        })
        .collect()
}

fn raw_file_basename(path: &Path) -> String {
    path.file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.to_string_lossy().replace('\\', "/"))
}

pub(crate) fn validate_checksums(package_dir: &Path, checksums: &CanonicalChecksums) -> Result<()> {
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

pub(crate) fn normalize_manifest_path(path: &str) -> String {
    Path::new(path)
        .file_name()
        .and_then(|value| value.to_str())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| path.replace('\\', "/"))
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut digest = Sha256::new();
    let bytes = fs::read(path).map_err(|source| Lux3dError::CanonicalChecksumsIo {
        path: path.to_path_buf(),
        source,
    })?;
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}
