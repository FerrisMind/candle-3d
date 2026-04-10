use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    Result,
    error::Lux3dError,
    weights::{CanonicalizationPlan, WeightLocator},
};

use super::{
    ContractSourceOfTruth, ContractStage, ModelFamily, NormalizationStats, Pi3InputSource,
    RgbRange, RuntimeArchitecture, SpatialSize, TensorDType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceDisposition {
    CleanRoomOk,
    ReferenceOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VendorSource {
    pub source_path: String,
    pub sha256: String,
    pub role: String,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicenseEntry {
    pub source_path: String,
    pub license: String,
    pub disposition: SourceDisposition,
    pub notes: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LicensePolicy {
    pub repo_license: String,
    pub weight_license: String,
    pub entries: Vec<LicenseEntry>,
}

impl LicensePolicy {
    pub fn covers_source(&self, source_path: &str) -> bool {
        self.entries
            .iter()
            .any(|entry| entry.source_path == source_path)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeTensorContract {
    pub name: String,
    pub stage: ContractStage,
    pub semantic: String,
    pub dims: Vec<String>,
    pub runtime_dtype: TensorDType,
    pub baseline_storage_dtype: Option<TensorDType>,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryBufferContract {
    pub name: String,
    pub semantic: String,
    pub dims: Vec<String>,
    pub runtime_dtype: TensorDType,
    pub baseline_storage_dtype: Option<TensorDType>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BaselineParityTap {
    pub name: String,
    pub semantic: String,
    pub symbolic_dims: Vec<String>,
    pub observed_shape: Vec<usize>,
    pub storage_dtype: TensorDType,
    pub source_path: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractExclusion {
    pub item: String,
    pub reason: String,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3PreprocessSpec {
    pub accepted_sources: Vec<Pi3InputSource>,
    pub input_layout: String,
    pub input_range: RgbRange,
    pub patch_multiple: u32,
    pub pixel_limit: u32,
    pub image_default_interval: usize,
    pub video_default_interval: usize,
    pub resize_filter: String,
    pub internal_normalization: NormalizationStats,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3xPreprocessSpec {
    pub accepted_sources: Vec<Pi3InputSource>,
    pub input_layout: String,
    pub input_range: RgbRange,
    pub patch_multiple: u32,
    pub pixel_limit: u32,
    pub image_default_interval: usize,
    pub video_default_interval: usize,
    pub resize_filter: String,
    pub internal_normalization: NormalizationStats,
    pub supports_conditions_npz: bool,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TripoPreprocessSpec {
    pub input_layout: String,
    pub tokenizer_input_layout: String,
    pub target_size: SpatialSize,
    pub optional_rgba_compositing: bool,
    pub background_value: f32,
    pub resize_mode: String,
    pub align_corners: bool,
    pub antialias: bool,
    pub normalization: NormalizationStats,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3NeuralSpec {
    pub encoder_backbone: String,
    pub patch_size: u32,
    pub positional_encoding: String,
    pub decoder_size: String,
    pub decoder_depth: usize,
    pub register_tokens: usize,
    pub point_decoder: String,
    pub confidence_decoder: String,
    pub camera_decoder: String,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3xNeuralSpec {
    pub encoder_backbone: String,
    pub patch_size: u32,
    pub positional_encoding: String,
    pub decoder_size: String,
    pub decoder_depth: usize,
    pub register_tokens: usize,
    pub pose_inject_blocks: usize,
    pub point_decoder: String,
    pub confidence_decoder: String,
    pub camera_decoder: String,
    pub metric_decoder: String,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TripoNeuralSpec {
    pub image_tokenizer: String,
    pub triplane_tokenizer: String,
    pub tokenizer_plane_size: u32,
    pub tokenizer_channels: usize,
    pub transformer_layers: usize,
    pub attention_heads: usize,
    pub attention_head_dim: usize,
    pub cross_attention_dim: usize,
    pub post_processor: String,
    pub decoder: String,
    pub feature_reduction: String,
    pub density_activation: String,
    pub density_bias: f32,
    pub samples_per_ray: usize,
    pub radius: f32,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3GeometrySpec {
    pub artifact_kind: String,
    pub assembly_formula: String,
    pub supports_marching_cubes: bool,
    pub supports_sim3_fusion_extension: bool,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3xGeometrySpec {
    pub artifact_kind: String,
    pub assembly_formula: String,
    pub supports_marching_cubes: bool,
    pub supports_sim3_fusion_extension: bool,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TripoGeometrySpec {
    pub artifact_kind: String,
    pub surface_extractor: String,
    pub default_resolution: u32,
    pub threshold: f32,
    pub axis_reorder: String,
    pub default_vertex_color_mode: String,
    pub supports_texture_baking_extension: bool,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3ExportSpec {
    pub primary_extension: String,
    pub alternate_extensions: Vec<String>,
    pub utility_notes: Vec<String>,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3xExportSpec {
    pub primary_extension: String,
    pub alternate_extensions: Vec<String>,
    pub utility_notes: Vec<String>,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TripoExportSpec {
    pub primary_extension: String,
    pub alternate_extensions: Vec<String>,
    pub utility_notes: Vec<String>,
    pub source_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "spec",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PreprocessSpec {
    Pi3(Pi3PreprocessSpec),
    Pi3x(Pi3xPreprocessSpec),
    #[serde(rename = "triposr")]
    TripoSr(TripoPreprocessSpec),
}

impl PreprocessSpec {
    pub fn source_paths(&self) -> &[String] {
        match self {
            Self::Pi3(spec) => &spec.source_paths,
            Self::Pi3x(spec) => &spec.source_paths,
            Self::TripoSr(spec) => &spec.source_paths,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "spec",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum NeuralSpec {
    Pi3(Pi3NeuralSpec),
    Pi3x(Pi3xNeuralSpec),
    #[serde(rename = "triposr")]
    TripoSr(TripoNeuralSpec),
}

impl NeuralSpec {
    pub fn source_paths(&self) -> &[String] {
        match self {
            Self::Pi3(spec) => &spec.source_paths,
            Self::Pi3x(spec) => &spec.source_paths,
            Self::TripoSr(spec) => &spec.source_paths,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "spec",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum GeometrySpec {
    Pi3(Pi3GeometrySpec),
    Pi3x(Pi3xGeometrySpec),
    #[serde(rename = "triposr")]
    TripoSr(TripoGeometrySpec),
}

impl GeometrySpec {
    pub fn source_paths(&self) -> &[String] {
        match self {
            Self::Pi3(spec) => &spec.source_paths,
            Self::Pi3x(spec) => &spec.source_paths,
            Self::TripoSr(spec) => &spec.source_paths,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "spec",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ExportSpec {
    Pi3(Pi3ExportSpec),
    Pi3x(Pi3xExportSpec),
    #[serde(rename = "triposr")]
    TripoSr(TripoExportSpec),
}

impl ExportSpec {
    pub fn source_paths(&self) -> &[String] {
        match self {
            Self::Pi3(spec) => &spec.source_paths,
            Self::Pi3x(spec) => &spec.source_paths,
            Self::TripoSr(spec) => &spec.source_paths,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3RuntimeGeometry {
    pub artifact_kind: String,
    pub assembly_formula: String,
    pub outputs: Vec<GeometryBufferContract>,
    pub supports_marching_cubes: bool,
    pub supports_sim3_fusion_extension: bool,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pi3xRuntimeGeometry {
    pub artifact_kind: String,
    pub assembly_formula: String,
    pub outputs: Vec<GeometryBufferContract>,
    pub supports_marching_cubes: bool,
    pub supports_sim3_fusion_extension: bool,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TripoRuntimeGeometry {
    pub artifact_kind: String,
    pub surface_extractor: String,
    pub default_resolution: u32,
    pub threshold: f32,
    pub axis_reorder: String,
    pub vertex_color_mode: String,
    pub outputs: Vec<GeometryBufferContract>,
    pub supports_texture_baking_extension: bool,
    pub source_paths: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "family",
    content = "spec",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RuntimeGeometry {
    Pi3(Pi3RuntimeGeometry),
    Pi3x(Pi3xRuntimeGeometry),
    #[serde(rename = "triposr")]
    TripoSr(TripoRuntimeGeometry),
}

impl RuntimeGeometry {
    pub fn source_paths(&self) -> &[String] {
        match self {
            Self::Pi3(spec) => &spec.source_paths,
            Self::Pi3x(spec) => &spec.source_paths,
            Self::TripoSr(spec) => &spec.source_paths,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub schema_version: u32,
    pub contract_source_of_truth: ContractSourceOfTruth,
    pub runtime_architecture: RuntimeArchitecture,
    pub family: ModelFamily,
    pub golden_baseline_sample: String,
    pub vendor_sources: Vec<VendorSource>,
    pub preprocess: PreprocessSpec,
    pub neural: NeuralSpec,
    pub geometry: GeometrySpec,
    pub export: ExportSpec,
    pub runtime_tensors: Vec<RuntimeTensorContract>,
    pub runtime_geometry: RuntimeGeometry,
    pub weight_plan: CanonicalizationPlan,
    pub baseline_parity_taps: Vec<BaselineParityTap>,
    pub license_policy: LicensePolicy,
    pub exclusions: Vec<ContractExclusion>,
    pub notes: Vec<String>,
}

impl ModelSpec {
    pub fn inspect(repo_root: PathBuf, family: ModelFamily) -> Result<Self> {
        let contract_path = Self::contract_path(&repo_root, family);
        let body =
            fs::read_to_string(&contract_path).map_err(|source| Lux3dError::ContractFileIo {
                path: contract_path.clone(),
                source,
            })?;
        let mut spec: Self =
            serde_json::from_str(&body).map_err(|source| Lux3dError::ContractFileJson {
                path: contract_path,
                source,
            })?;

        spec.resolve_paths(&repo_root);
        spec.validate(repo_root, family)?;
        Ok(spec)
    }

    pub fn contract_path(repo_root: &Path, family: ModelFamily) -> PathBuf {
        repo_root
            .join("3d")
            .join("contracts")
            .join(format!("{}.json", family.as_str()))
    }

    pub fn used_source_paths(&self) -> Vec<String> {
        let mut sources = BTreeSet::new();
        for source in self.preprocess.source_paths() {
            sources.insert(source.clone());
        }
        for source in self.neural.source_paths() {
            sources.insert(source.clone());
        }
        for source in self.geometry.source_paths() {
            sources.insert(source.clone());
        }
        for source in self.export.source_paths() {
            sources.insert(source.clone());
        }
        for source in self.runtime_geometry.source_paths() {
            sources.insert(source.clone());
        }
        for tensor in &self.runtime_tensors {
            for source in &tensor.source_paths {
                sources.insert(source.clone());
            }
        }
        for tap in &self.baseline_parity_taps {
            sources.insert(tap.source_path.clone());
        }
        for exclusion in &self.exclusions {
            for source in &exclusion.source_paths {
                sources.insert(source.clone());
            }
        }
        sources.into_iter().collect()
    }

    fn resolve_paths(&mut self, repo_root: &Path) {
        self.weight_plan.raw_files = self
            .weight_plan
            .raw_files
            .iter()
            .map(|path| resolve_repo_path(repo_root, path))
            .collect();
        self.weight_plan.canonical_root =
            resolve_repo_path(repo_root, &self.weight_plan.canonical_root);
    }

    fn validate(&self, repo_root: PathBuf, expected_family: ModelFamily) -> Result<()> {
        if self.family != expected_family {
            return Err(Lux3dError::ContractValidation {
                message: format!(
                    "contract family `{}` does not match requested family `{}`",
                    self.family, expected_family
                ),
            });
        }

        if self.contract_source_of_truth != ContractSourceOfTruth::Hybrid {
            return Err(Lux3dError::ContractValidation {
                message: "stage-2 contracts must use the hybrid source-of-truth strategy"
                    .to_string(),
            });
        }

        if self.runtime_architecture != RuntimeArchitecture::CandleFirst {
            return Err(Lux3dError::ContractValidation {
                message: "stage-2 contracts must use candle-first runtime architecture".to_string(),
            });
        }

        let located = WeightLocator::new(repo_root.clone()).locate(expected_family)?;
        if self.weight_plan != located {
            return Err(Lux3dError::ContractValidation {
                message: format!(
                    "contract weight plan {:?} does not match discovered plan {:?}",
                    self.weight_plan, located
                ),
            });
        }

        let vendor_sources = self
            .vendor_sources
            .iter()
            .map(|source| (source.source_path.as_str(), source))
            .collect::<BTreeMap<_, _>>();

        for source_path in self.used_source_paths() {
            if !vendor_sources.contains_key(source_path.as_str()) {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "referenced source path `{source_path}` is missing from vendor_sources"
                    ),
                });
            }
            if !self.license_policy.covers_source(&source_path) {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "referenced source path `{source_path}` is missing from license_policy"
                    ),
                });
            }
        }

        for source in &self.vendor_sources {
            let resolved = repo_root.join(&source.source_path);
            if !resolved.is_file() {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "vendor source `{}` does not exist at `{}`",
                        source.source_path,
                        resolved.display()
                    ),
                });
            }
            let actual = sha256_file(&resolved)?;
            if actual != source.sha256 {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "vendor source `{}` fingerprint mismatch: expected `{}`, found `{}`",
                        source.source_path, source.sha256, actual
                    ),
                });
            }
        }

        Ok(())
    }
}

fn resolve_repo_path(repo_root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| Lux3dError::ContractFileIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut digest = Sha256::new();
    digest.update(&bytes);
    let hash = digest.finalize();
    Ok(hash.iter().map(|byte| format!("{byte:02x}")).collect())
}
