//! TripoSG end-to-end pipeline assembly (Task 4): canonical weights ->
//! preprocess -> DINOv2 conditioning -> seeded rectified-flow latents ->
//! DiT denoise -> VAE SDF decode -> hierarchical marching cubes -> OBJ export.
//!
//! The pipeline mirrors [`TripoSrPipeline`](super::triposr::TripoSrPipeline):
//! a per-device model bundle is built once from the canonical F32 weight set
//! and cached in a [`DeviceLocalCache`], and every Candle failure is mapped
//! into `Lux3dError::CanonicalWeightsValidation` with a `TripoSG ... failed`
//! message.
//!
//! ## Seeded latent initialization
//!
//! The initial latents are sampled from a deterministic CPU RNG (SplitMix64 +
//! Box-Muller, f64 math rounded to f32) and only then moved to the target
//! device. Runs with the same `--seed` are therefore reproducible across
//! Lux3D invocations and backends, but they are **NOT bit-identical** to the
//! torch device RNG the reference pipeline uses (plan scope decision 3):
//! cross-framework parity is exercised by injecting the baseline
//! `latents_init` through `infer_with_latents` instead.
//!
//! ## Mesh extraction / export
//!
//! `extract_mesh` decodes the hierarchical SDF grid at the canonical
//! `+/-1.005` bounds (level 0, positive = inside) and assembles the CPU mesh
//! through the TripoSG geometry stage; device mesh buffers are materialized
//! on the latents' device. `export_obj` moves the buffers to CPU and writes
//! the reference-style ASCII OBJ through the export stage.

use std::{path::Path, sync::Arc};

use candle_core::{DType, Device, Tensor};

use super::{
    DeviceLocalCache, RmbgConfig, RmbgModel,
    triposg_dino::TripoSgDinoEncoder,
    triposg_dit::{TripoSgDiT, denoise},
    triposg_field::hierarchical_sdf_grid,
    triposg_flash::{compact_dmc_mesh, flash_dual_marching_cubes, flash_sdf_grid},
    triposg_vae::{
        TripoSgLatentDistribution, TripoSgVaeConfig, TripoSgVaeDecoder, TripoSgVaeEncoder,
    },
};
use crate::{
    AuxiliaryWeightSetPaths, CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result,
    error::Lux3dError,
    export::TripoSgExportStage,
    geometry::{TripoSgGeometryStage, TripoSgMeshCpu},
    load_auxiliary_canonical_weights, load_canonical_weights,
    neural::TripoSgNeuralStage,
    preprocess::TripoSgPreprocessStage,
};

use super::triposg_dit::TripoSgTransformerConfig;

/// TripoSG latent channel count (`transformer.in_channels` ==
/// `vae.latent_channels` == 64 in the shipped config).
pub(crate) const TRIPOSG_LATENT_CHANNELS: usize = 64;

/// Canonical SDF query bounds, `[-1.005, -1.005, -1.005, 1.005, 1.005, 1.005]`
/// (the reference `bbox_size = 2.01` cube; see `pipeline_defaults.bounds` in
/// the canonical `resolved_config.json`).
const CANONICAL_BOUNDS: [f32; 6] = [-1.005, -1.005, -1.005, 1.005, 1.005, 1.005];

/// Iso level the surface is extracted at (the decoder returns negated logits,
/// so positive = inside and the surface sits at 0).
const CANONICAL_SDF_LEVEL: f32 = 0.0;

/// Hard octree-depth cap for run options (grid side `2^depth` <= 1024, the
/// same cap `hierarchical_sdf_grid` enforces).
const MAX_OCTREE_DEPTH: u32 = 10;

/// Per-run options for the TripoSG pipeline. Defaults replicate the reference
/// `TripoSGPipeline` call in `pipeline_triposg.py` (`num_inference_steps=50`,
/// `guidance_scale=7.0`, `num_tokens=2048`, dense depth 8 / hierarchical
/// depth 9, `num_chunks`-equivalent query chunk 16384).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripoSgRunOptions {
    /// Rectified-flow Euler steps (reference `num_inference_steps`).
    pub steps: usize,
    /// Classifier-free guidance scale (batch order `[uncond, cond]`).
    pub guidance_scale: f32,
    /// Seed for the Lux3D-seeded latent initialization (see module docs).
    pub seed: u64,
    /// Number of latent tokens (`[1, num_tokens, 64]` init latents).
    pub num_tokens: usize,
    /// Dense octree depth of the hierarchical SDF grid (default 8 -> 256^3).
    pub dense_octree_depth: u32,
    /// Final hierarchical octree depth (default 9 -> 512^3 mesh grid).
    pub hierarchical_octree_depth: u32,
    /// Use the reference flash decoder + DiffDMC mesh path (default true,
    /// matching `pipeline_triposg.py`). When false, dense/hierarchical MC is
    /// used instead.
    pub use_flash_decoder: bool,
    /// Flash (DiffDMC) octree depth (default 9, reference
    /// `flash_octree_depth`).
    pub flash_octree_depth: u32,
    /// Maximum number of SDF query points per decoder pass.
    pub query_chunk_size: usize,
}

impl Default for TripoSgRunOptions {
    fn default() -> Self {
        Self {
            steps: 50,
            guidance_scale: 7.0,
            seed: 42,
            num_tokens: 2048,
            dense_octree_depth: 8,
            hierarchical_octree_depth: 9,
            use_flash_decoder: true,
            flash_octree_depth: 9,
            query_chunk_size: 16384,
        }
    }
}

impl TripoSgRunOptions {
    /// Validate the options before any weights are touched. The checks match
    /// the pipeline invariants: at least one denoise step and one latent
    /// token, octree depths within the 1024-side grid cap, and a finite
    /// guidance scale.
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG steps must be at least 1, got {}", self.steps),
            });
        }
        if self.num_tokens == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG num_tokens must be at least 1, got {}",
                    self.num_tokens
                ),
            });
        }
        if self.dense_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG dense_octree_depth {} exceeds the maximum {MAX_OCTREE_DEPTH}",
                    self.dense_octree_depth
                ),
            });
        }
        if self.hierarchical_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG hierarchical_octree_depth {} exceeds the maximum \
                     {MAX_OCTREE_DEPTH}",
                    self.hierarchical_octree_depth
                ),
            });
        }
        if self.flash_octree_depth == 0 || self.flash_octree_depth > MAX_OCTREE_DEPTH {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG flash_octree_depth {} must be in 1..={MAX_OCTREE_DEPTH}",
                    self.flash_octree_depth
                ),
            });
        }
        if !self.guidance_scale.is_finite() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG guidance_scale must be finite, got {}",
                    self.guidance_scale
                ),
            });
        }
        Ok(())
    }
}

/// Re-export of the preprocess stage output so callers can name the whole
/// TripoSG stage chain from `runtime`.
pub use crate::preprocess::TripoSgPreparedInputs;

/// TripoSG inference output: DINOv2 conditioning embeds, the seeded initial
/// latents and the denoised latents (all F32).
#[derive(Debug, Clone)]
pub struct TripoSgInferenceOutput {
    /// `[1, 257, 1024]` last hidden state of the DINOv2-large encoder.
    pub image_embeds: Tensor,
    /// `[1, num_tokens, 64]` seeded standard-normal init latents.
    pub latents_init: Tensor,
    /// `[1, num_tokens, 64]` latents after `steps` rectified-flow Euler steps.
    pub latents: Tensor,
}

/// Device mesh buffers produced by `extract_mesh` (positions in the canonical
/// `+/-1.005` world space, zero-based triangle indices).
#[derive(Debug, Clone)]
pub struct TripoSgMeshBuffers {
    /// `[N, 3]` F32 vertex positions on the latents' device.
    pub vertices: Tensor,
    /// `[F, 3]` I64 zero-based triangle indices on the latents' device.
    pub faces: Tensor,
}

/// Everything the pipeline needs per device: the DINOv2 image encoder, the
/// 21-block DiT and the VAE SDF decoder, all loaded from the canonical F32
/// weight set.
#[derive(Debug)]
struct TripoSgModelBundle {
    dino: TripoSgDinoEncoder,
    dit: TripoSgDiT,
    vae: TripoSgVaeDecoder,
    vae_encoder: TripoSgVaeEncoder,
}

impl TripoSgModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device) -> Result<Self> {
        // SAFETY: the canonical safetensors file is read-only and the mmap
        // stays alive through the VarBuilder's 'static borrow for the bundle's
        // lifetime (same pattern as TripoSR).
        let vb = unsafe {
            weights.var_builder(DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open TripoSG canonical weights: {source}"),
                }
            })?
        };
        let resolved = weights.read_resolved_config_json()?;
        let transformer_cfg =
            TripoSgTransformerConfig::from_resolved_config(&resolved).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to parse TripoSG transformer config: {source}"),
                }
            })?;
        let vae_cfg = TripoSgVaeConfig::from_resolved_config(&resolved).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to parse TripoSG VAE config: {source}"),
            }
        })?;
        Ok(Self {
            dino: TripoSgDinoEncoder::load(vb.pp("image_encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG image encoder: {source}"),
                }
            })?,
            dit: TripoSgDiT::load(vb.pp("transformer"), &transformer_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG DiT: {source}"),
                }
            })?,
            vae: TripoSgVaeDecoder::load(vb.pp("vae"), &vae_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG VAE decoder: {source}"),
                }
            })?,
            vae_encoder: TripoSgVaeEncoder::load(vb.pp("vae"), &vae_cfg).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSG VAE encoder: {source}"),
                }
            })?,
        })
    }
}

/// TripoSG image-to-3D pipeline: preprocess -> DINOv2 conditioning ->
/// seeded rectified-flow denoise -> VAE SDF decode -> OBJ export.
#[derive(Debug)]
pub struct TripoSgPipeline {
    pub weights: CanonicalWeightSetPaths,
    /// Optional separate RMBG-1.4 canonical package. Required only when the
    /// input image has no valid alpha channel (the reference fallback).
    pub rmbg_weights: Option<AuxiliaryWeightSetPaths>,
    pub preprocess: TripoSgPreprocessStage,
    pub neural: TripoSgNeuralStage,
    pub geometry: TripoSgGeometryStage,
    pub export: TripoSgExportStage,
    bundle_cache: Arc<DeviceLocalCache<TripoSgModelBundle>>,
    rmbg_cache: Arc<DeviceLocalCache<RmbgModel>>,
}

impl TripoSgPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Self::load_with_rmbg(model_assets, None)
    }

    /// Load TripoSG plus an optional separate RMBG-1.4 canonical package.
    ///
    /// RMBG is not part of the TripoSG weight file: its BRIA license differs,
    /// so it is distributed in its own canonical directory. Passing `None`
    /// keeps the alpha-only workflow; inputs without a valid alpha channel
    /// then fail with an explicit error, matching the reference's hard
    /// requirement for BriaRMBG.
    pub fn load_with_rmbg(
        model_assets: ModelAssetOptions,
        rmbg_model_dir: Option<std::path::PathBuf>,
    ) -> Result<Self> {
        let rmbg_weights = match rmbg_model_dir {
            Some(dir) => Some(load_auxiliary_canonical_weights(dir, "rmbg")?),
            None => None,
        };
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::TripoSg, model_assets)?,
            rmbg_weights,
            preprocess: TripoSgPreprocessStage::default(),
            neural: TripoSgNeuralStage,
            geometry: TripoSgGeometryStage,
            export: TripoSgExportStage,
            bundle_cache: Arc::default(),
            rmbg_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<TripoSgModelBundle>> {
        self.bundle_cache
            .get_or_try_init(device, || TripoSgModelBundle::load(&self.weights, device))
    }

    fn rmbg_for(&self, device: &Device) -> Result<Option<Arc<RmbgModel>>> {
        let Some(weights) = &self.rmbg_weights else {
            return Ok(None);
        };
        self.rmbg_cache
            .get_or_try_init(device, || {
                let config =
                    RmbgConfig::from_resolved_config(&weights.read_resolved_config_json()?)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("failed to parse RMBG config: {source}"),
                        })?;
                // SAFETY: canonical safetensors mmap, same lifetime contract
                // as the TripoSG bundle.
                let vb = unsafe { weights.var_builder(DType::F32, device) }.map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to open RMBG canonical weights: {source}"),
                    }
                })?;
                RmbgModel::load(vb, &config).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to construct RMBG model: {source}"),
                    }
                })
            })
            .map(Some)
    }

    /// Deterministic Lux3D-seeded latent initialization: `[1, num_tokens,
    /// channels]` F32 standard-normal values generated on CPU (SplitMix64 +
    /// Box-Muller, f64 math rounded to f32) and moved to `device`. Same seed,
    /// same tensor; different seeds differ; NOT torch-RNG-identical (see the
    /// module docs).
    pub fn sample_latents(
        seed: u64,
        num_tokens: usize,
        channels: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if num_tokens == 0 || channels == 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latent shape ({num_tokens}, {channels}) must be positive on \
                     both axes"
                ),
            });
        }
        let mut rng = SeededNormalRng::new(seed);
        let mut values = Vec::with_capacity(num_tokens * channels);
        for _ in 0..(num_tokens * channels) {
            values.push(rng.next_standard_normal() as f32);
        }
        Tensor::from_vec(values, (1, num_tokens, channels), &Device::Cpu)
            .and_then(|tensor| tensor.to_device(device))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize TripoSG seeded latents: {source}"),
            })
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        // Match reference `load_image`: BriaRMBG is only constructed when the
        // source has no valid alpha channel. Eager loading costs ~176 MiB of
        // VRAM and pushes native Vulkan/WGPU backends over the 12 GiB limit.
        match self
            .preprocess
            .prepare_inputs_from_path_with_rmbg(source, device, None)
        {
            Ok(prepared) => Ok(prepared),
            Err(Lux3dError::InvalidInput(message))
                if message.starts_with("TripoSG input has no valid alpha channel")
                    && self.rmbg_weights.is_some() =>
            {
                let rmbg = self.rmbg_for(device)?;
                self.preprocess
                    .prepare_inputs_from_path_with_rmbg(source, device, rmbg.as_deref())
            }
            Err(error) => Err(error),
        }
    }

    /// DINOv2-large conditioning embeds `[B, 257, 1024]` from prepared
    /// `[B, 3, 224, 224]` pixel values.
    pub fn image_embeds(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(pixel_values.device())?;
        bundle
            .dino
            .forward(pixel_values)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG image encoder forward failed: {source}"),
            })
    }

    /// Full inference from an image file: preprocess -> DINOv2 embeds ->
    /// seeded latents -> CFG denoise. The VAE/mesh stage is driven separately
    /// through [`Self::extract_mesh`] so callers can cache latents.
    pub fn infer_from_path(
        &self,
        source: &Path,
        device: &Device,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgInferenceOutput> {
        options.validate()?;
        let prepared = self.prepare_inputs_from_path(source, device)?;
        self.neural
            .infer_latents(self, &prepared.pixel_values, options)
    }

    /// Denoise caller-supplied initial latents with classifier-free guidance.
    /// The latents are NOT regenerated here: `latents_init` must already be
    /// `[1, num_tokens, 64]` (batch 1), e.g. from [`Self::sample_latents`] or
    /// a parity baseline. `image_embeds` is the conditional DINOv2 context;
    /// the uncond branch is `zeros_like` inside the denoise loop.
    pub fn infer_with_latents(
        &self,
        image_embeds: &Tensor,
        latents_init: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<Tensor> {
        options.validate()?;
        Self::validate_latents(latents_init, options.num_tokens)?;
        let bundle = self.bundle_for(latents_init.device())?;
        denoise(
            &bundle.dit,
            latents_init,
            image_embeds,
            options.steps,
            options.guidance_scale,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("TripoSG denoise failed: {source}"),
        })
    }

    /// Encode a point cloud into the TripoSG latent distribution.
    ///
    /// `points [B, N, C]` with the first three channels interpreted as
    /// positions and the remaining channels as per-point features (TripoSG's
    /// reference script feeds `[positions, normals]`, C = 6). The reference
    /// encoder samples `num_tokens * 4` points, runs deterministic FPS at
    /// ratio 1/4 (`random_start=False` in eval mode), and returns the
    /// `DiagonalGaussianDistribution` from `vae.quant`.
    ///
    /// `seed` controls the Lux3D deterministic index sampler; pass `None` for
    /// a process-time seed. The transformer/media pipeline does not call this
    /// path, but it is the full reference encode API.
    pub fn encode_point_cloud(
        &self,
        points: &Tensor,
        num_tokens: usize,
        seed: Option<u64>,
    ) -> Result<TripoSgLatentDistribution> {
        let dims = points.dims();
        if dims.len() != 3 || dims[0] == 0 || dims[1] == 0 || dims[2] < 3 {
            return Err(Lux3dError::InvalidInput(
                "TripoSG VAE encode expects point features [B, N, C] with B/N > 0 and C >= 3",
            ));
        }
        let bundle = self.bundle_for(points.device())?;
        bundle
            .vae_encoder
            .encode(points, num_tokens, seed)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Encode with an explicit sampled query subset (reference parity path).
    pub fn encode_point_cloud_sampled(
        &self,
        points: &Tensor,
        sampled_points: &Tensor,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(points.device())?;
        bundle
            .vae_encoder
            .encode_sampled(points, sampled_points)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Encode with an explicit initial NumPy-choice index vector; the FPS
    /// half still runs. Used by the reference parity tests.
    pub fn encode_point_cloud_with_indices(
        &self,
        points: &Tensor,
        indices: &[usize],
        num_tokens: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(points.device())?;
        let sampled = bundle
            .vae_encoder
            .sample_with_indices(points, indices, num_tokens)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE FPS sampling failed: {source}"),
            })?;
        bundle
            .vae_encoder
            .encode_sampled(points, &sampled)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG VAE encode failed: {source}"),
            })
    }

    /// Decode the SDF field for `latents [1, num_tokens, 64]` at the canonical
    /// `+/-1.005` bounds and assemble the level-0 marching-cubes mesh. The
    /// buffers are materialized on `latents.device()`.
    pub fn extract_mesh(
        &self,
        latents: &Tensor,
        options: &TripoSgRunOptions,
    ) -> Result<TripoSgMeshBuffers> {
        options.validate()?;
        Self::validate_latents(latents, options.num_tokens)?;
        let bundle = self.bundle_for(latents.device())?;
        let mesh_cpu = if options.use_flash_decoder {
            // Reference default (`use_flash_decoder=True`): hierarchical
            // flash SDF query schedule + clean-room DiffDMC.
            let grid = flash_sdf_grid(
                &bundle.vae,
                latents,
                CANONICAL_BOUNDS,
                options.flash_octree_depth,
                options.query_chunk_size,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG flash SDF grid decode failed: {source}"),
            })?;
            let mesh = flash_dual_marching_cubes(&grid).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("TripoSG flash DMC failed: {source}"),
                }
            })?;
            // Reference pipelines wrap the DMC output in
            // `trimesh.Trimesh(..., process=True)`, which drops vertices no
            // face references (NaN-padded boundary shells at depth >= 7).
            let mesh = compact_dmc_mesh(&mesh);
            TripoSgMeshCpu {
                resolution: grid.dims.saturating_sub(1) as u32,
                sdf_level: CANONICAL_SDF_LEVEL,
                bounds: CANONICAL_BOUNDS,
                vertices: mesh.vertices,
                faces: mesh.faces,
            }
        } else {
            let grid = hierarchical_sdf_grid(
                &bundle.vae,
                latents,
                CANONICAL_BOUNDS,
                options.dense_octree_depth,
                options.hierarchical_octree_depth,
                options.query_chunk_size,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSG SDF grid decode failed: {source}"),
            })?;
            self.geometry.assemble_cpu(&grid)?
        };

        let device = latents.device().clone();
        let vertex_count = mesh_cpu.vertices.len() / 3;
        let face_count = mesh_cpu.faces.len() / 3;
        let vertices =
            Tensor::from_vec(mesh_cpu.vertices, (vertex_count, 3), &device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to materialize TripoSG mesh vertices: {source}"),
                }
            })?;
        let faces =
            Tensor::from_vec(mesh_cpu.faces, (face_count, 3), &device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to materialize TripoSG mesh faces: {source}"),
                }
            })?;
        Ok(TripoSgMeshBuffers { vertices, faces })
    }

    /// Export mesh buffers as ASCII OBJ: gather both tensors to CPU, rebuild
    /// the CPU mesh and run the export-stage writer (`v x y z`, 1-based
    /// `f a b c`, no vertex colors).
    pub fn export_obj(&self, mesh: &TripoSgMeshBuffers, path: &Path) -> Result<()> {
        let vertices = mesh
            .vertices
            .to_device(&Device::Cpu)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to gather TripoSG mesh vertices on cpu: {source}"),
            })?;
        let faces = mesh
            .faces
            .to_device(&Device::Cpu)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1::<i64>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to gather TripoSG mesh faces on cpu: {source}"),
            })?;
        // The device buffers don't carry grid metadata (resolution lives in
        // the SDF grid, which is consumed inside extract_mesh); the OBJ writer
        // only reads vertices/faces.
        let mesh_cpu = TripoSgMeshCpu {
            resolution: 0,
            sdf_level: CANONICAL_SDF_LEVEL,
            bounds: CANONICAL_BOUNDS,
            vertices,
            faces,
        };
        self.export.write_obj(&mesh_cpu, path)
    }

    /// `latents` must be `[1, num_tokens, 64]` (batch 1 — the VAE decoder and
    /// the CFG denoise loop are batch-1 paths).
    fn validate_latents(latents: &Tensor, num_tokens: usize) -> Result<()> {
        let dims = latents.dims();
        if dims.len() != 3
            || dims[0] != 1
            || dims[1] != num_tokens
            || dims[2] != TRIPOSG_LATENT_CHANNELS
        {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "TripoSG latents must be [1, {num_tokens}, {TRIPOSG_LATENT_CHANNELS}] \
                     (batch 1), got {dims:?}"
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Deterministic CPU RNG (SplitMix64 + Box-Muller)
// ---------------------------------------------------------------------------

/// SplitMix64 state + Box-Muller transform for the latent initialization.
///
/// Chosen for reproducibility, not statistical sophistication: every run with
/// the same seed walks the exact same f64 sequence, and rounding to f32 at the
/// end keeps the values stable across platforms (all math happens in f64).
pub(crate) struct SeededNormalRng {
    state: u64,
}

impl SeededNormalRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// SplitMix64 next output.
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` from the top 53 bits.
    fn next_uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One standard normal per two uniforms (Box-Muller, cosine branch).
    /// The first uniform is clamped away from zero so `ln` stays finite.
    fn next_standard_normal(&mut self) -> f64 {
        let u1 = self.next_uniform().max(f64::EPSILON);
        let u2 = self.next_uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}
