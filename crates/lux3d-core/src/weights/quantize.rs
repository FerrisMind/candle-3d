//! Offline safetensors -> GGUF quantization for canonical packages.
//!
//! Quantizes rank-2 `*.weight` tensors (the linear layers) and stores
//! everything else — norms, biases, embeddings, convolutions — in the
//! fallback dtype. The resulting `model.gguf` sits next to the canonical
//! `model.safetensors`; [`super::CanonicalWeightSetPaths::var_builder`] picks
//! it up automatically.

use std::fs;
use std::path::{Path, PathBuf};

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{Device, Tensor};

use crate::{
    error::Lux3dError,
    contracts::ModelFamily,
    weights::{GGUF_FAMILY_KEY, GGUF_RESOLVED_CONFIG_KEY, RESOLVED_CONFIG_FILENAME},
    Result,
};

/// Quantization policy for [`quantize_checkpoint`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeOptions {
    /// Dtype used for rank-2 `*.weight` tensors whose row length is a multiple
    /// of the dtype block size.
    pub dtype: GgmlDType,
    /// Dtype for everything else (norms, biases, embeddings, convolutions,
    /// and weights whose row length does not divide into blocks).
    pub fallback: GgmlDType,
    /// Quantize every rank-2 tensor instead of only `*.weight`.
    pub all_2d: bool,
    /// Optional name prefixes: when non-empty, only matching tensors are
    /// quantized. Useful for isolating a sub-model (`transformer.`,
    /// `vae.`, `image_encoder.`) during accuracy bisection.
    pub prefixes: Vec<String>,
}

impl Default for QuantizeOptions {
    fn default() -> Self {
        Self {
            dtype: GgmlDType::Q8_0,
            fallback: GgmlDType::F16,
            all_2d: false,
            prefixes: Vec::new(),
        }
    }
}

/// Summary of a completed quantization run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizeReport {
    pub output: PathBuf,
    pub dtype: GgmlDType,
    pub fallback: GgmlDType,
    pub tensors: usize,
    pub quantized: usize,
    pub fallback_tensors: usize,
    /// Rank-0 tensors (e.g. `num_batches_tracked`) cannot be represented in
    /// GGML and are omitted from the GGUF.
    pub skipped_tensors: usize,
    pub source_bytes: u64,
    pub output_bytes: u64,
    /// Whether `resolved_config.json` was embedded for standalone loading.
    pub embedded_resolved_config: bool,
}

impl QuantizeReport {
    /// Human-readable one-line summary for the CLI.
    pub fn render(&self) -> String {
        let skipped = if self.skipped_tensors == 0 {
            String::new()
        } else {
            format!(", {} scalars skipped", self.skipped_tensors)
        };
        format!(
            "{}: {} tensors ({} {}, {} {}){skipped}, {:.2} GiB -> {:.2} GiB{}",
            self.output.display(),
            self.tensors,
            self.quantized,
            dtype_name(self.dtype),
            self.fallback_tensors,
            dtype_name(self.fallback),
            self.source_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            self.output_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            if self.embedded_resolved_config {
                ", resolved config embedded"
            } else {
                ""
            },
        )
    }
}

/// Quantize a canonical `model.safetensors` into `output` (`.gguf`).
pub fn quantize_checkpoint(
    source: &Path,
    output: &Path,
    family: ModelFamily,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    quantize_checkpoint_with_family(source, output, family.as_str(), options)
}

/// Quantize an auxiliary package (for example `rmbg`) whose family is not a
/// [`ModelFamily`]. The resulting GGUF is standalone when the source package
/// has a sibling `resolved_config.json`.
pub fn quantize_auxiliary_checkpoint(
    source: &Path,
    output: &Path,
    family: &str,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    if family.trim().is_empty() {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: "auxiliary family label must not be empty".to_string(),
        });
    }
    quantize_checkpoint_with_family(source, output, family.trim(), options)
}

fn quantize_checkpoint_with_family(
    source: &Path,
    output: &Path,
    family: &str,
    options: QuantizeOptions,
) -> Result<QuantizeReport> {
    if !source.is_file() {
        return Err(Lux3dError::MissingCanonicalArtifact {
            path: source.to_path_buf(),
        });
    }
    if matches!(options.dtype, GgmlDType::Q8K | GgmlDType::Q8_1) {
        return Err(Lux3dError::CanonicalWeightsValidation {
            message: format!(
                "{} is an activation-side GGML type, not a weight quantization; use q8_0",
                dtype_name(options.dtype)
            ),
        });
    }
    let source_bytes = fs::metadata(source)
        .map_err(|err| Lux3dError::ModelAssetIo {
            path: source.to_path_buf(),
            source: err,
        })?
        .len();

    // SAFETY: the canonical file is read-only for the duration of this
    // function and the mapping is dropped before returning.
    let mmap = unsafe { MmapedSafetensors::multi(std::slice::from_ref(&source)) }.map_err(
        |err| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to open {}: {err}", source.display()),
        },
    )?;

    let mut tensors = Vec::with_capacity(mmap.tensors().len());
    let mut quantized = 0usize;
    let mut fallback = 0usize;
    let mut skipped = 0usize;
    for (name, view) in mmap.tensors() {
        let tensor = mmap
            .load(&name, &Device::Cpu)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load tensor `{name}`: {source}"),
            })?;
        // GGML has no scalar dtype representation; training-only buffers such
        // as BatchNorm `num_batches_tracked` are skipped.
        if tensor.rank() == 0 {
            skipped += 1;
            continue;
        }
        let use_quant = should_quantize(&name, &tensor, &options);
        let dtype = if use_quant {
            quantized += 1;
            options.dtype
        } else {
            fallback += 1;
            options.fallback
        };
        let qtensor = QTensor::quantize(&tensor, dtype).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to quantize tensor `{name}` {:?} to {dtype:?}: {source}",
                    view.shape()
                ),
            }
        })?;
        tensors.push((name, qtensor));
    }

    // Embed the sibling resolved config so the GGUF can be used on its own,
    // without `model.safetensors` / manifest / checksums. `general.architecture`
    // mirrors the config's top-level `architecture`, then `family`, then the
    // requested model family.
    let resolved_config_path = source.with_file_name(RESOLVED_CONFIG_FILENAME);
    let (embedded_resolved_config, architecture) = if resolved_config_path.is_file() {
        let body =
            fs::read_to_string(&resolved_config_path).map_err(|source_err| {
                Lux3dError::ModelAssetIo {
                    path: resolved_config_path.clone(),
                    source: source_err,
                }
            })?;
        let parsed: serde_json::Value =
            serde_json::from_str(&body).map_err(|source_err| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to parse `{}`: {source_err}",
                        resolved_config_path.display()
                    ),
                }
            })?;
        let architecture = resolve_architecture(Some(&parsed), family);
        let serialized = serde_json::to_string(&parsed).map_err(|source_err| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "failed to re-serialize `{}`: {source_err}",
                    resolved_config_path.display()
                ),
            }
        })?;
        (Some(serialized), architecture)
    } else {
        (None, resolve_architecture(None, family))
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|source| Lux3dError::ModelAssetIo {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let mut file = fs::File::create(output).map_err(|source| Lux3dError::ModelAssetIo {
        path: output.to_path_buf(),
        source,
    })?;
    let mut metadata = vec![
        (
            "general.architecture",
            gguf_file::Value::String(architecture),
        ),
        (
            GGUF_FAMILY_KEY,
            gguf_file::Value::String(family.to_string()),
        ),
        (
            "lux3d.quantization",
            gguf_file::Value::String(dtype_name(options.dtype).into()),
        ),
        (
            "lux3d.fallback_dtype",
            gguf_file::Value::String(dtype_name(options.fallback).into()),
        ),
        (
            "lux3d.tensor_count",
            gguf_file::Value::U32(tensors.len().try_into().unwrap_or(u32::MAX)),
        ),
    ];
    if let Some(config) = embedded_resolved_config.as_ref() {
        metadata.push((
            GGUF_RESOLVED_CONFIG_KEY,
            gguf_file::Value::String(config.clone()),
        ));
    }
    let metadata_refs: Vec<(&str, &gguf_file::Value)> =
        metadata.iter().map(|(name, value)| (*name, value)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();
    gguf_file::write(&mut file, &metadata_refs, &tensor_refs).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to write {}: {source}", output.display()),
        }
    })?;
    drop(file);

    let output_bytes = fs::metadata(output)
        .map_err(|source| Lux3dError::ModelAssetIo {
            path: output.to_path_buf(),
            source,
        })?
        .len();
    Ok(QuantizeReport {
        output: output.to_path_buf(),
        dtype: options.dtype,
        fallback: options.fallback,
        tensors: tensors.len(),
        quantized,
        fallback_tensors: fallback,
        skipped_tensors: skipped,
        source_bytes,
        output_bytes,
        embedded_resolved_config: embedded_resolved_config.is_some(),
    })
}

fn should_quantize(name: &str, tensor: &Tensor, options: &QuantizeOptions) -> bool {
    if tensor.rank() != 2 {
        return false;
    }
    if !options.all_2d && !name.ends_with(".weight") {
        return false;
    }
    if !options.prefixes.is_empty()
        && !options.prefixes.iter().any(|prefix| name.starts_with(prefix))
    {
        return false;
    }
    match tensor.dim(1) {
        Ok(row) => row % options.dtype.block_size() == 0,
        Err(_) => false,
    }
}

/// `general.architecture` value: top-level `architecture`, then `family`,
/// then the caller-provided model name.
fn resolve_architecture(config: Option<&serde_json::Value>, fallback: &str) -> String {
    config
        .and_then(|config| {
            config
                .get("architecture")
                .and_then(serde_json::Value::as_str)
                .or_else(|| config.get("family").and_then(serde_json::Value::as_str))
        })
        .unwrap_or(fallback)
        .to_string()
}

/// GGUF dtype name used in metadata and CLI parsing.
pub const fn dtype_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
    }
}

/// Parse a GGUF dtype name; `None` for unknown names.
pub fn dtype_from_name(name: &str) -> Option<GgmlDType> {
    Some(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q4_0" => GgmlDType::Q4_0,
        "q4_1" => GgmlDType::Q4_1,
        "q5_0" => GgmlDType::Q5_0,
        "q5_1" => GgmlDType::Q5_1,
        "q8_0" => GgmlDType::Q8_0,
        "q8_1" => GgmlDType::Q8_1,
        "q2_k" => GgmlDType::Q2K,
        "q3_k" => GgmlDType::Q3K,
        "q4_k" => GgmlDType::Q4K,
        "q5_k" => GgmlDType::Q5K,
        "q6_k" => GgmlDType::Q6K,
        "q8_k" => GgmlDType::Q8K,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        QuantizeOptions, dtype_from_name, dtype_name, quantize_auxiliary_checkpoint,
        quantize_checkpoint, resolve_architecture, should_quantize,
    };
    use crate::contracts::ModelFamily;
    use candle_core::quantized::{GgmlDType, gguf_file};
    use candle_core::{Device, Tensor};
    use std::collections::HashMap;
    use std::path::PathBuf;

    #[test]
    fn architecture_prefers_config_architecture_then_family() {
        let explicit = serde_json::json!({"architecture": "BriaRMBG", "family": "rmbg"});
        assert_eq!(resolve_architecture(Some(&explicit), "fallback"), "BriaRMBG");
        let family = serde_json::json!({"family": "triposg"});
        assert_eq!(resolve_architecture(Some(&family), "fallback"), "triposg");
        let neither = serde_json::json!({"hidden_size": 64});
        assert_eq!(resolve_architecture(Some(&neither), "vggt"), "vggt");
        assert_eq!(resolve_architecture(None, "rmbg"), "rmbg");
    }

    #[test]
    fn dtype_names_round_trip() {
        for dtype in [
            GgmlDType::F32,
            GgmlDType::F16,
            GgmlDType::BF16,
            GgmlDType::Q4_0,
            GgmlDType::Q5_0,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
            GgmlDType::Q8K,
        ] {
            assert_eq!(dtype_from_name(dtype_name(dtype)), Some(dtype));
        }
        assert_eq!(dtype_from_name("nope"), None);
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "lux3d-quantize-{tag}-{}-{stamp}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_checkpoint(dir: &std::path::Path, with_config: bool) -> PathBuf {
        let cpu = Device::Cpu;
        let mut tensors = HashMap::new();
        tensors.insert(
            "transformer.blocks.0.attn.to_q.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        tensors.insert(
            "transformer.blocks.0.norm.weight".to_string(),
            Tensor::rand(0.5f32, 1.5f32, (64,), &cpu).unwrap(),
        );
        // Row length 10 is not a multiple of the Q8_0 block size.
        tensors.insert(
            "transformer.blocks.0.proj.weight".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 10), &cpu).unwrap(),
        );
        // Rank-2 but no `.weight` suffix: only quantized with `all_2d`.
        tensors.insert(
            "transformer.blocks.0.table".to_string(),
            Tensor::rand(-0.05f32, 0.05f32, (64, 64), &cpu).unwrap(),
        );
        // Rank-0 training-only buffer: cannot be represented in GGML.
        tensors.insert(
            "transformer.blocks.0.num_batches_tracked".to_string(),
            Tensor::zeros((), candle_core::DType::F32, &cpu).unwrap(),
        );
        if with_config {
            std::fs::write(
                dir.join("resolved_config.json"),
                r#"{"hidden_size":64,"num_layers":1}"#,
            )
            .unwrap();
        }
        let source = dir.join("model.safetensors");
        candle_core::safetensors::save(&tensors, &source).unwrap();
        source
    }

    fn read_gguf(path: &std::path::Path) -> gguf_file::Content {
        let mut file = std::fs::File::open(path).unwrap();
        gguf_file::Content::read(&mut file).unwrap()
    }

    fn gguf_architecture(content: &gguf_file::Content) -> &str {
        match content.metadata.get("general.architecture") {
            Some(gguf_file::Value::String(value)) => value,
            other => panic!("missing general.architecture: {other:?}"),
        }
    }

    #[test]
    fn quantize_selects_linear_weights_and_keeps_fallback() {
        let dir = scratch_dir("policy");
        let source = tiny_checkpoint(&dir, false);
        let output = dir.join("model.gguf");
        let report =
            quantize_checkpoint(&source, &output, ModelFamily::Vggt, QuantizeOptions::default())
                .unwrap();
        assert_eq!(report.tensors, 4);
        assert_eq!(report.quantized, 1);
        assert_eq!(report.fallback_tensors, 3);
        assert_eq!(report.skipped_tensors, 1);
        assert!(output.is_file());

        let content = read_gguf(&output);
        assert_eq!(gguf_architecture(&content), "vggt");
        assert_eq!(content.tensor_infos.len(), 4);
        assert!(
            !content
                .tensor_infos
                .contains_key("transformer.blocks.0.num_batches_tracked")
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.attn.to_q.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.norm.weight"].ggml_dtype,
            GgmlDType::F16
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.proj.weight"].ggml_dtype,
            GgmlDType::F16
        );
        assert_eq!(
            content.tensor_infos["transformer.blocks.0.table"].ggml_dtype,
            GgmlDType::F16
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_honours_all_2d_and_prefixes() {
        let dir = scratch_dir("options");
        let source = tiny_checkpoint(&dir, false);
        let all_2d = QuantizeOptions {
            all_2d: true,
            ..QuantizeOptions::default()
        };
        let report = quantize_checkpoint(
            &source,
            &dir.join("all_2d.gguf"),
            ModelFamily::Vggt,
            all_2d,
        )
        .unwrap();
        assert_eq!(report.quantized, 2);

        let prefixed = QuantizeOptions {
            prefixes: vec!["vae.".to_string()],
            ..QuantizeOptions::default()
        };
        let report = quantize_checkpoint(
            &source,
            &dir.join("prefixed.gguf"),
            ModelFamily::Vggt,
            prefixed,
        )
        .unwrap();
        assert_eq!(report.quantized, 0);
        assert_eq!(report.fallback_tensors, 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn quantize_rejects_activation_dtypes() {
        let dir = scratch_dir("dtype");
        let source = tiny_checkpoint(&dir, false);
        let options = QuantizeOptions {
            dtype: GgmlDType::Q8K,
            ..QuantizeOptions::default()
        };
        let err = quantize_checkpoint(
            &source,
            &dir.join("bad.gguf"),
            ModelFamily::Vggt,
            options,
        )
        .unwrap_err();
        assert!(err.to_string().contains("activation-side"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn should_quantize_checks_shape_and_suffix() {
        let cpu = Device::Cpu;
        let weight = Tensor::rand(0f32, 1f32, (64, 64), &cpu).unwrap();
        let options = QuantizeOptions::default();
        assert!(should_quantize("block.weight", &weight, &options));
        assert!(!should_quantize("block.table", &weight, &options));
        let narrow = Tensor::rand(0f32, 1f32, (64, 10), &cpu).unwrap();
        assert!(!should_quantize("block.weight", &narrow, &options));
        let bias = Tensor::rand(0f32, 1f32, (64,), &cpu).unwrap();
        assert!(!should_quantize("block.weight", &bias, &options));
    }

    #[test]
    fn standalone_gguf_loads_without_safetensors() {
        let dir = scratch_dir("standalone");
        let source = tiny_checkpoint(&dir, true);
        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        assert!(report.embedded_resolved_config);
        std::fs::remove_file(&source).unwrap();
        std::fs::remove_file(dir.join("resolved_config.json")).unwrap();

        let paths = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(dir.clone()),
                cache_dir: None,
            },
        )
        .unwrap();
        assert_eq!(paths.canonical_file, output);

        // The same GGUF can be passed directly (flat distribution layout).
        let file_paths = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(output.clone()),
                cache_dir: None,
            },
        )
        .unwrap();
        assert_eq!(
            file_paths.read_resolved_config_json().unwrap()["hidden_size"],
            64
        );
        let config = paths.read_resolved_config_json().unwrap();
        assert_eq!(config["hidden_size"].as_u64(), Some(64));
        let vb = unsafe {
            paths
                .var_builder(candle_core::DType::F32, &Device::Cpu)
                .unwrap()
        };
        let weight = vb
            .get((64, 64), "transformer.blocks.0.attn.to_q.weight")
            .unwrap();
        assert_eq!(weight.dims(), &[64, 64]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn standalone_gguf_without_embedded_config_is_rejected() {
        let dir = scratch_dir("standalone-no-config");
        let source = tiny_checkpoint(&dir, false);
        let output = dir.join("model.gguf");
        let report = quantize_checkpoint(
            &source,
            &output,
            ModelFamily::Vggt,
            QuantizeOptions::default(),
        )
        .unwrap();
        assert!(!report.embedded_resolved_config);
        std::fs::remove_file(&source).unwrap();

        let err = crate::weights::load_canonical_weights(
            ModelFamily::Vggt,
            crate::weights::ModelAssetOptions {
                canonical_dir: Some(dir.clone()),
                cache_dir: None,
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("no embedded"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn auxiliary_standalone_gguf_loads_in_f16() {
        let dir = scratch_dir("rmbg");
        let source = tiny_checkpoint(&dir, true);
        let output = dir.join("model.gguf");
        let options = QuantizeOptions {
            dtype: GgmlDType::F16,
            fallback: GgmlDType::F16,
            ..QuantizeOptions::default()
        };
        let report =
            quantize_auxiliary_checkpoint(&source, &output, "rmbg", options).unwrap();
        assert!(report.embedded_resolved_config);
        assert_eq!(report.skipped_tensors, 1);
        let content = read_gguf(&output);
        assert_eq!(gguf_architecture(&content), "rmbg");
        for info in content.tensor_infos.values() {
            assert_eq!(info.ggml_dtype, GgmlDType::F16);
        }
        std::fs::remove_file(&source).unwrap();
        std::fs::remove_file(dir.join("resolved_config.json")).unwrap();

        let paths = crate::weights::load_auxiliary_canonical_weights(&dir, "rmbg").unwrap();
        assert_eq!(paths.canonical_file, output);
        let file_paths =
            crate::weights::load_auxiliary_canonical_weights(&output, "rmbg").unwrap();
        assert_eq!(
            file_paths.read_resolved_config_json().unwrap()["hidden_size"],
            64
        );
        let config = paths.read_resolved_config_json().unwrap();
        assert_eq!(config["hidden_size"].as_u64(), Some(64));
        let vb = unsafe {
            paths
                .var_builder(candle_core::DType::F32, &Device::Cpu)
                .unwrap()
        };
        let weight = vb
            .get((64, 64), "transformer.blocks.0.attn.to_q.weight")
            .unwrap();
        assert_eq!(weight.dims(), &[64, 64]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
