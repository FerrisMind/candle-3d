//! Auxiliary canonical weight packages.
//!
//! Some runtime dependencies are not output model families. RMBG-1.4 is the
//! image-mattening network used by TripoSG's reference preprocessing; its BRIA
//! source-available/non-commercial license differs from TripoSG's MIT license,
//! so it lives in a separate canonical directory and is loaded through this
//! small auxiliary manifest schema instead of `ModelFamily`.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use candle_core::DType;
use serde::{Deserialize, Serialize};

use crate::{Result, error::Lux3dError};

use super::{
    CANONICAL_FILENAME, CHECKSUMS_FILENAME, GGUF_FILENAME, MANIFEST_FILENAME,
    RESOLVED_CONFIG_FILENAME, VarBuilder, is_gguf_path, normalize_manifest_path, read_checksums,
    read_standalone_gguf, validate_checksums,
};

/// Manifest for an auxiliary (non-`ModelFamily`) canonical package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuxiliaryWeightsManifest {
    pub family: String,
    pub normalizer_version: u32,
    pub raw_files: Box<[String]>,
    pub canonical_file: String,
    pub resolved_config_file: String,
    pub tensor_count: usize,
    pub dtype_histogram: BTreeMap<String, usize>,
    pub source_checksums: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuxiliaryWeightSetPaths {
    pub manifest: AuxiliaryWeightsManifest,
    pub canonical_file: PathBuf,
    pub resolved_config_file: PathBuf,
    pub manifest_file: PathBuf,
    pub checksums_file: PathBuf,
    /// `Some` for a standalone `model.gguf` with embedded config metadata.
    pub embedded_resolved_config: Option<serde_json::Value>,
}

impl AuxiliaryWeightSetPaths {
    /// # Safety
    ///
    /// The returned builder is backed by a memory-mapped safetensors file.
    /// Callers must ensure the mapped file outlives every tensor/module created
    /// from the builder and that the file is not mutated while mapped.
    pub unsafe fn var_builder(
        &self,
        dtype: DType,
        device: &candle_core::Device,
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

/// Load and validate an auxiliary canonical package at `package_dir`.
///
/// `expected_family` is the manifest family discriminator (`"rmbg"`). The
/// package must contain `model.safetensors`, `resolved_config.json`,
/// `manifest.json`, and `checksums.json`, and every checksum entry is verified.
/// A standalone `model.gguf` with embedded config metadata is accepted instead.
pub fn load_auxiliary_canonical_weights(
    package_dir: impl AsRef<Path>,
    expected_family: &str,
) -> Result<AuxiliaryWeightSetPaths> {
    let package_dir = package_dir.as_ref();
    if package_dir.is_file() {
        if is_gguf_path(package_dir) {
            return load_standalone_aux_gguf(package_dir.to_path_buf(), expected_family);
        }
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "auxiliary `{}` is not a GGUF; pass a package directory or a `.gguf` file",
                package_dir.display()
            ),
        });
    }
    let manifest_file = package_dir.join(MANIFEST_FILENAME);
    let checksums_file = package_dir.join(CHECKSUMS_FILENAME);
    let canonical_file = package_dir.join(CANONICAL_FILENAME);
    let resolved_config_file = package_dir.join(RESOLVED_CONFIG_FILENAME);
    let gguf_file = package_dir.join(GGUF_FILENAME);

    if !canonical_file.is_file() {
        if gguf_file.is_file() {
            return load_standalone_aux_gguf(gguf_file, expected_family);
        }
        return Err(Lux3dError::MissingCanonicalArtifact { path: canonical_file });
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

    let body =
        fs::read_to_string(&manifest_file).map_err(|source| Lux3dError::CanonicalManifestIo {
            path: manifest_file.clone(),
            source,
        })?;
    let manifest: AuxiliaryWeightsManifest =
        serde_json::from_str(&body).map_err(|source| Lux3dError::CanonicalManifestJson {
            path: manifest_file.clone(),
            source,
        })?;

    if manifest.family != expected_family {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "auxiliary manifest family `{}` does not match requested `{}`",
                manifest.family, expected_family
            ),
        });
    }
    if normalize_manifest_path(&manifest.canonical_file) != CANONICAL_FILENAME {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "auxiliary manifest canonical file `{}` does not point to `{}`",
                manifest.canonical_file, CANONICAL_FILENAME
            ),
        });
    }
    if normalize_manifest_path(&manifest.resolved_config_file) != RESOLVED_CONFIG_FILENAME {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "auxiliary manifest resolved config `{}` does not point to `{}`",
                manifest.resolved_config_file, RESOLVED_CONFIG_FILENAME
            ),
        });
    }

    let checksums = read_checksums(&checksums_file)?;
    validate_checksums(package_dir, &checksums)?;

    Ok(AuxiliaryWeightSetPaths {
        manifest,
        canonical_file,
        resolved_config_file,
        manifest_file,
        checksums_file,
        embedded_resolved_config: None,
    })
}

fn load_standalone_aux_gguf(
    gguf_file: PathBuf,
    expected_family: &str,
) -> Result<AuxiliaryWeightSetPaths> {
    let standalone = read_standalone_gguf(&gguf_file, expected_family)?;
    let manifest = AuxiliaryWeightsManifest {
        family: expected_family.to_string(),
        normalizer_version: 0,
        raw_files: Box::default(),
        canonical_file: GGUF_FILENAME.to_string(),
        resolved_config_file: GGUF_FILENAME.to_string(),
        tensor_count: standalone.tensor_count,
        dtype_histogram: standalone.dtype_histogram,
        source_checksums: BTreeMap::new(),
    };
    Ok(AuxiliaryWeightSetPaths {
        manifest,
        canonical_file: gguf_file.clone(),
        resolved_config_file: gguf_file.clone(),
        manifest_file: gguf_file.clone(),
        checksums_file: gguf_file,
        embedded_resolved_config: Some(standalone.resolved_config),
    })
}
