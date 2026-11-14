use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{DType, Device, DeviceLocation, Tensor};
#[cfg(feature = "vision-preproc")]
use image::DynamicImage;
use image::imageops::FilterType;

mod attention_math;
mod nn_blocks;

/// Re-exported so the weights module (and its tests) can reshape GGUF-stored
/// rank>4 tensors without reaching into the private `nn_blocks` module.
#[cfg(test)]
pub(crate) use nn_blocks::get_weight_shaped;

/// Drain asynchronous weight-upload staging after one model layer/block.
///
/// Vulkan and WGPU upload each tensor through backend staging buffers. Those
/// copies are queued asynchronously, so loading a multi-GiB checkpoint without
/// an intermediate drain retains a second copy of the weights in staging
/// memory (measured: ~6.4 GiB host staging on Vulkan and ~7.3 GiB on WGPU for
/// the 7.4 GiB TripoSG checkpoint). CUDA reclaims upload staging per copy, so
/// its peak is just the live weight set. Synchronizing once per layer/block
/// keeps the staging window at roughly one block while leaving the load time
/// dominated by the weight upload itself.
pub(crate) fn drain_upload_staging(device: &Device) -> candle_core::Result<()> {
    if device.is_vulkan() || device.is_wgpu() {
        device.synchronize()?;
        log_wgpu_alloc(device, "wgpu-load");
    }
    Ok(())
}

/// Opt-in (`LUX3D_WGPU_MEM_STATS=1`) allocator telemetry for memory debugging.
pub(crate) fn log_wgpu_alloc(device: &Device, label: &str) {
    // Without the `wgpu` feature candle exposes a dummy backend whose
    // `WgpuDevice` has no allocator report. Keep the default build warning-free.
    #[cfg(not(feature = "wgpu"))]
    {
        let _ = (device, label);
    }
    #[cfg(feature = "wgpu")]
    {
        if std::env::var_os("LUX3D_WGPU_MEM_STATS").is_none() {
            return;
        }
        if let Ok(wgpu) = device.as_wgpu_device() {
            if let Some(report) = wgpu.allocator_report() {
                eprintln!(
                    "[{label}] allocated={:.1} MiB reserved={:.1} MiB blocks={}",
                    report.total_allocated_bytes as f64 / (1024.0 * 1024.0),
                    report.total_reserved_bytes as f64 / (1024.0 * 1024.0),
                    report.blocks.len()
                );
                if std::env::var("LUX3D_WGPU_MEM_STATS").as_deref() == Ok("2") {
                    let mut allocations = report.allocations;
                    allocations.sort_by_key(|allocation| std::cmp::Reverse(allocation.size));
                    for allocation in allocations.iter().take(12) {
                        eprintln!(
                            "[{label}]   {}={:.1} MiB",
                            allocation.name,
                            allocation.size as f64 / (1024.0 * 1024.0)
                        );
                    }
                }
            }
        }
    }
}

mod path_utils;
mod pi3_decoder;
mod pi3_encoder;
mod pi3_heads;
mod pi3x;
mod point_camera_ops;
mod resampling;
mod rmbg;
mod rmbg2;
mod trellis2;
mod trellis2_blocks;
mod trellis2_dino;
mod trellis2_dit;
mod trellis2_mesh;
mod trellis2_sampler;
mod trellis2_sparse;
mod trellis2_ss_dec;
mod trellis2_tex;
mod trellis2_vae;
mod triposg;
mod triposg_dino;
mod triposg_dit;
mod triposg_field;
mod triposg_flash;
mod triposg_vae;
mod triposr;
mod triposr_field;
mod vggt;
mod vggt_dpt;
mod vision_preproc;

pub(crate) use pi3x::prepare_pi3x_inputs_with_stage;
pub use pi3x::{
    Pi3xInferenceOutput, Pi3xInjectConditions, Pi3xPipeline, Pi3xPreparedInputs, Pi3xVoOutput,
    Pi3xVoPipeline,
};
pub use rmbg::{RmbgConfig, RmbgModel};
pub use rmbg2::{Rmbg2Config, Rmbg2Model, Rmbg2Outputs};
pub use trellis2::{
    TRELLIS2_CASCADE_HR_RESOLUTION, TRELLIS2_CASCADE_LR_RESOLUTION,
    TRELLIS2_CASCADE_UPSAMPLE_TIMES, TRELLIS2_SHAPE_SLAT_MEAN, TRELLIS2_SHAPE_SLAT_STD,
    TRELLIS2_TEX_NOISE_SALT, TRELLIS2_TEX_SLAT_MEAN, TRELLIS2_TEX_SLAT_STD, TRELLIS2_VOXEL_MARGIN,
    Trellis2CascadeShape, Trellis2Pipeline, Trellis2PipelineConfig, Trellis2Rng,
    Trellis2RunOptions, Trellis2Stages, quantize_hr_coords, quantize_hr_coords_with_stop,
};
pub use trellis2_blocks::{
    Trellis2AdaLn, Trellis2CrossAttention, Trellis2DenseBlock, Trellis2FeedForward,
    Trellis2QkRmsNorm, Trellis2Rope3d, Trellis2SelfAttention, Trellis2SparseBlock,
    Trellis2TimestepEmbedder, chunk_modulation, cubic_token_coords, timestep_embedding,
};
pub use trellis2_dino::{Trellis2Dino, Trellis2DinoConfig};
pub use trellis2_dit::{Trellis2SlatConfig, Trellis2SlatDit, Trellis2SsConfig, Trellis2SsDit};
pub use trellis2_mesh::{
    DualGridFields, DualGridMesh, flexible_dual_grid_to_mesh, softplus_stable,
};
pub use trellis2_sampler::{
    Trellis2SampleTrace, Trellis2SamplerParams, guided_velocity, pred_to_xstart, sample_dense,
    sample_sparse, ss_t_schedule, v_to_xstart_eps, xstart_to_pred,
};
pub use trellis2_sparse::{
    Channel2SpatialOutput, CoordMap, SparseConv3d, Spatial2ChannelOutput, channel_to_spatial,
    kernel_offset_delta, repeat_channels, sparse_cross_attention, sparse_self_attention,
    spatial_to_channel,
};
pub use trellis2_ss_dec::{Trellis2SsDecoder, Trellis2SsDecoderConfig, occupancy_coords};
pub use trellis2_tex::{
    TexDecodeOutput, Trellis2PbrLayout, Trellis2TexDecoder, Trellis2TexDit, concat_tex_input,
    pbr_attrs_from_raw, query_vertex_attrs, query_vertex_colors, split_pbr_attrs,
};
pub use trellis2_vae::{
    ShapeVaeBlockTrace, ShapeVaeDecodeOutput, Trellis2ShapeVaeConfig, Trellis2ShapeVaeDecoder,
    decode_shape_mesh, dual_grid_fields,
};
pub(crate) use triposg::TRIPOSG_LATENT_CHANNELS;
pub use triposg::{
    TripoSgInferenceOutput, TripoSgMeshBuffers, TripoSgPipeline, TripoSgPreparedInputs,
    TripoSgRunOptions,
};
pub use triposg_dino::TripoSgDinoEncoder;
pub use triposg_dit::{RectifiedFlowSchedule, TripoSgDiT, TripoSgTransformerConfig, denoise};
pub use triposg_field::{TripoSgSdfGrid, hierarchical_sdf_grid};
pub use triposg_flash::{
    TripoSgDmcMesh, TripoSgFlashGrid, compact_dmc_mesh, flash_dual_marching_cubes,
    flash_resolutions, flash_sdf_grid,
};
pub use triposg_vae::{
    TripoSgLatentDistribution, TripoSgVaeConfig, TripoSgVaeDecoder, TripoSgVaeEncoder,
    sample_point_features, sample_point_features_with_indices,
};
pub(crate) use triposr::prepare_triposr_inputs_with_stage;
pub use triposr::{TripoInferenceOutput, TripoMeshBuffers, TripoPreparedInputs, TripoSrPipeline};
pub use vggt::{
    VggtAggregatorOutput, VggtDebugTaps, VggtInferenceOutput, VggtPipeline, VggtPreparedInputs,
};

use crate::{
    CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result,
    contracts::SpatialSize,
    error::Lux3dError,
    export::{Pi3ExportStage, PointCloudFormat},
    geometry::Pi3GeometryStage,
    load_canonical_weights,
    neural::Pi3NeuralStage,
    preprocess::Pi3PreprocessStage,
};

#[derive(Debug)]
pub(super) struct DeviceLocalCache<T> {
    bundles: Mutex<HashMap<DeviceLocation, Arc<T>>>,
}

impl<T> Default for DeviceLocalCache<T> {
    fn default() -> Self {
        Self {
            bundles: Mutex::new(HashMap::new()),
        }
    }
}

impl<T> DeviceLocalCache<T> {
    pub(super) fn get_or_try_init<E, F>(
        &self,
        device: &Device,
        init: F,
    ) -> std::result::Result<Arc<T>, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        let location = device.location();
        {
            let guard = self
                .bundles
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(bundle) = guard.get(&location) {
                return Ok(bundle.clone());
            }
        }

        let bundle = Arc::new(init()?);
        let mut guard = self
            .bundles
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let entry = guard.entry(location).or_insert_with(|| bundle.clone());
        Ok(entry.clone())
    }
}

fn cast_tensor_to_f32(tensor: &Tensor) -> candle_core::Result<Tensor> {
    if tensor.dtype() == DType::F32 {
        Ok(tensor.clone())
    } else {
        tensor.to_dtype(DType::F32)
    }
}

#[derive(Debug)]
struct Pi3ModelBundle {
    encoder: pi3_encoder::Pi3DinoEncoder,
    decoder: pi3_decoder::Pi3Decoder,
    point_decoder: pi3_decoder::Pi3BranchDecoder,
    conf_decoder: pi3_decoder::Pi3BranchDecoder,
    camera_decoder: pi3_decoder::Pi3BranchDecoder,
    point_head: pi3_heads::LinearPts3dHead,
    conf_head: pi3_heads::LinearPts3dHead,
    camera_head: pi3_heads::CameraHead,
}

impl Pi3ModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device) -> Result<Self> {
        let vb = unsafe {
            weights.var_builder(DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open Pi3 canonical weights: {source}"),
                }
            })?
        };
        Ok(Self {
            encoder: pi3_encoder::load_pi3_encoder(vb.pp("encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 encoder: {source}"),
                }
            })?,
            decoder: pi3_decoder::load_pi3_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 decoder: {source}"),
                }
            })?,
            point_decoder: pi3_decoder::load_pi3_point_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 point decoder: {source}"),
                }
            })?,
            conf_decoder: pi3_decoder::load_pi3_conf_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 conf decoder: {source}"),
                }
            })?,
            camera_decoder: pi3_decoder::load_pi3_camera_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 camera decoder: {source}"),
                }
            })?,
            point_head: pi3_heads::load_point_head(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 point head: {source}"),
                }
            })?,
            conf_head: pi3_heads::load_conf_head(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 conf head: {source}"),
                }
            })?,
            camera_head: pi3_heads::load_camera_head(vb).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3 camera head: {source}"),
                }
            })?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Pi3PreparedInputs {
    pub interval: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub rgb_frames: Tensor,
    pub normalized_frames: Tensor,
}

#[derive(Debug, Clone)]
pub struct Pi3InferenceOutput {
    pub local_points: Tensor,
    pub confidence_logits: Tensor,
    pub camera_poses: Tensor,
    pub points: Tensor,
    pub export_mask: Tensor,
    pub rgb_frames: Tensor,
}

#[derive(Debug, Clone)]
pub struct Pi3Pipeline {
    pub weights: CanonicalWeightSetPaths,
    pub preprocess: Pi3PreprocessStage,
    pub neural: Pi3NeuralStage,
    pub geometry: Pi3GeometryStage,
    pub export: Pi3ExportStage,
    bundle_cache: Arc<DeviceLocalCache<Pi3ModelBundle>>,
}

impl Pi3Pipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::Pi3, model_assets)?,
            preprocess: Pi3PreprocessStage::default(),
            neural: Pi3NeuralStage,
            geometry: Pi3GeometryStage,
            export: Pi3ExportStage,
            bundle_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<Pi3ModelBundle>> {
        self.bundle_cache
            .get_or_try_init(device, || Pi3ModelBundle::load(&self.weights, device))
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<Pi3PreparedInputs> {
        self.preprocess
            .prepare_inputs_from_path(source, interval, device)
    }

    pub fn encode_patch_tokens(&self, normalized_frames: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(normalized_frames.device())?;

        let dims = normalized_frames.dims();
        let frames = dims[1];
        let channels = dims[2];
        let height = dims[3];
        let width = dims[4];
        let inputs = normalized_frames
            .reshape((frames, channels, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3 normalized frames: {source}"),
            })?;
        bundle
            .encoder
            .forward_patch_tokens(&inputs)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 encoder forward failed: {source}"),
            })
    }

    pub fn prepare_encoder_tokens(&self, normalized_frames: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(normalized_frames.device())?;

        let dims = normalized_frames.dims();
        let frames = dims[1];
        let channels = dims[2];
        let height = dims[3];
        let width = dims[4];
        let inputs = normalized_frames
            .reshape((frames, channels, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3 normalized frames: {source}"),
            })?;
        bundle
            .encoder
            .forward_prepared_tokens(&inputs)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 prepare_tokens failed: {source}"),
            })
    }

    pub fn decode_hidden(
        &self,
        patch_tokens: &Tensor,
        num_views: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(patch_tokens.device())?;
        let patch_tokens = cast_tensor_to_f32(patch_tokens).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 patch tokens to f32: {source}"),
            }
        })?;
        bundle
            .decoder
            .decode(&patch_tokens, num_views, height, width)
            .map(|(hidden, _)| hidden)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 decoder forward failed: {source}"),
            })
    }

    pub fn decode_positions_only(
        &self,
        patch_tokens: &Tensor,
        num_views: usize,
        height: usize,
        width: usize,
    ) -> Result<Tensor> {
        let patch_tokens = cast_tensor_to_f32(patch_tokens).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 patch tokens to f32: {source}"),
            }
        })?;
        let batch = patch_tokens.dims()[0] / num_views;
        let bundle = self.bundle_for(patch_tokens.device())?;
        bundle
            .decoder
            .decoder_positions(batch, num_views, height, width, patch_tokens.device())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 decoder forward failed: {source}"),
            })
    }

    pub fn point_decoder_hidden(&self, hidden: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 point hidden to f32: {source}"),
            }
        })?;
        bundle
            .point_decoder
            .forward(&hidden, positions)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 point decoder forward failed: {source}"),
            })
    }

    pub fn conf_decoder_hidden(&self, hidden: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 confidence hidden to f32: {source}"),
            }
        })?;
        bundle
            .conf_decoder
            .forward(&hidden, positions)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 conf decoder forward failed: {source}"),
            })
    }

    pub fn camera_decoder_hidden(&self, hidden: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 camera hidden to f32: {source}"),
            }
        })?;
        bundle
            .camera_decoder
            .forward(&hidden, positions)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 camera decoder forward failed: {source}"),
            })
    }

    pub fn point_head_output(
        &self,
        hidden: &Tensor,
        image_height: usize,
        image_width: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 point-head hidden to f32: {source}"),
            }
        })?;
        bundle
            .point_head
            .forward(&hidden, image_height, image_width, 5)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 point head forward failed: {source}"),
            })
    }

    pub fn conf_head_output(
        &self,
        hidden: &Tensor,
        image_height: usize,
        image_width: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 confidence-head hidden to f32: {source}"),
            }
        })?;
        bundle
            .conf_head
            .forward(&hidden, image_height, image_width, 5)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 conf head forward failed: {source}"),
            })
    }

    pub fn local_points_from_head_output(&self, head_output: &Tensor) -> Result<Tensor> {
        point_camera_ops::local_points_from_head_output(head_output)
            .and_then(|tensor| tensor.to_dtype(DType::F32))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 local_points reconstruction failed: {source}"),
            })
    }

    pub fn camera_poses_from_hidden(
        &self,
        hidden: &Tensor,
        patch_h: usize,
        patch_w: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(hidden.device())?;
        let hidden = cast_tensor_to_f32(hidden).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to cast Pi3 camera-head hidden to f32: {source}"),
            }
        })?;
        bundle
            .camera_head
            .forward(&hidden, patch_h, patch_w)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 camera head forward failed: {source}"),
            })
    }

    pub fn world_points(&self, local_points: &Tensor, camera_poses: &Tensor) -> Result<Tensor> {
        let local_points = if local_points.dtype() == camera_poses.dtype() {
            local_points.clone()
        } else {
            local_points
                .to_dtype(camera_poses.dtype())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to align Pi3 local point dtype with camera poses: {source}"
                    ),
                })?
        };
        point_camera_ops::world_points_from_local_and_pose(&local_points, camera_poses).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 world point assembly failed: {source}"),
            },
        )
    }

    pub fn export_mask(&self, local_points: &Tensor, confidence_logits: &Tensor) -> Result<Tensor> {
        let confidence_logits = if confidence_logits.dtype() == local_points.dtype() {
            confidence_logits.clone()
        } else {
            confidence_logits
                .to_dtype(local_points.dtype())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!(
                        "failed to align Pi3 confidence dtype with local points: {source}"
                    ),
                })?
        };
        point_camera_ops::export_mask_from_local_points_and_confidence(
            local_points,
            &confidence_logits,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3 export mask computation failed: {source}"),
        })
    }

    pub fn non_edge_mask(&self, local_points: &Tensor) -> Result<Tensor> {
        point_camera_ops::non_edge_mask_from_local_points(local_points).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 non-edge mask computation failed: {source}"),
            }
        })
    }

    pub fn infer_from_path(&self, source: &Path, device: &Device) -> Result<Pi3InferenceOutput> {
        self.infer_from_path_with_interval(source, None, device)
    }

    pub fn infer_from_path_with_interval(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<Pi3InferenceOutput> {
        let prepared = self.prepare_inputs_from_path(source, interval, device)?;
        self.neural.infer_tensors(self, &prepared)
    }

    /// Write the assembled point cloud as `.ply` (primary) or `.xyz`.
    pub fn export_point_cloud(
        &self,
        output: &Pi3InferenceOutput,
        path: &Path,
    ) -> Result<PointCloudFormat> {
        let point_cloud = self.geometry.assemble_cpu(output)?;
        self.export.write_point_cloud(&point_cloud, path)
    }

    /// Write the assembled point cloud as ASCII PLY (legacy entry point).
    pub fn export_ply(&self, output: &Pi3InferenceOutput, path: &Path) -> Result<()> {
        let point_cloud = self.geometry.assemble_cpu(output)?;
        self.export.write_ply(&point_cloud, path)
    }
}

pub(crate) fn prepare_pi3_inputs_with_stage(
    stage: &Pi3PreprocessStage,
    source: &Path,
    interval: Option<usize>,
    device: &Device,
) -> Result<Pi3PreparedInputs> {
    let interval = interval.unwrap_or_else(|| default_interval_for(source));
    if interval == 0 {
        return Err(Lux3dError::InvalidInput("interval must be at least 1"));
    }

    let sampled = load_sampled_pi3_frames(source, interval)?;
    let sampled_paths = &sampled.paths;

    let first = image::open(&sampled_paths[0])
        .map_err(|_| Lux3dError::InvalidInput("failed to open first Pi3 image"))?
        .to_rgb8();
    let original_size = SpatialSize::new(first.width(), first.height());
    let target_size = stage.target_size_for(original_size, 255_000)?;

    #[cfg(feature = "vision-preproc")]
    let (rgb_frames, normalized_frames) = {
        let mut rgb_frames = Vec::with_capacity(sampled_paths.len());
        let mut normalized_frames = Vec::with_capacity(sampled_paths.len());
        for path in sampled_paths {
            let image = image::open(path)
                .map_err(|_| Lux3dError::InvalidInput("failed to open Pi3 input image"))?
                .to_rgb8();
            let resized = image::imageops::resize(
                &image,
                target_size.width,
                target_size.height,
                FilterType::Lanczos3,
            );
            let dynamic = DynamicImage::ImageRgb8(resized);
            let rgb_chw =
                vision_preproc::tensorize_rgb_image(&dynamic, device).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3 tensorization failed: {source}"),
                    }
                })?;
            let rgb_chw = vision_preproc::resize_chw_image(
                &rgb_chw,
                target_size.height as usize,
                target_size.width as usize,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3 vision resize failed: {source}"),
            })?;
            let normalized_chw =
                vision_preproc::normalize_imagenet_chw(&rgb_chw).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3 vision normalization failed: {source}"),
                    }
                })?;
            rgb_frames.push(rgb_chw);
            normalized_frames.push(normalized_chw);
        }

        let rgb_refs = rgb_frames.iter().collect::<Vec<_>>();
        let normalized_refs = normalized_frames.iter().collect::<Vec<_>>();
        let rgb_frames = Tensor::stack(&rgb_refs, 0).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to stack Pi3 RGB tensors: {source}"),
            }
        })?;
        let normalized_frames = Tensor::stack(&normalized_refs, 0)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to stack Pi3 normalized tensors: {source}"),
            })?;
        (rgb_frames, normalized_frames)
    };

    #[cfg(not(feature = "vision-preproc"))]
    let pixels_per_frame = (3 * target_size.width * target_size.height) as usize;
    #[cfg(not(feature = "vision-preproc"))]
    let mut rgb_data = Vec::with_capacity(sampled_paths.len() * pixels_per_frame);
    #[cfg(not(feature = "vision-preproc"))]
    for path in sampled_paths {
        let image = image::open(path)
            .map_err(|_| Lux3dError::InvalidInput("failed to open Pi3 input image"))?
            .to_rgb8();
        let resized = image::imageops::resize(
            &image,
            target_size.width,
            target_size.height,
            FilterType::Lanczos3,
        );
        for channel in 0..3 {
            for y in 0..target_size.height {
                for x in 0..target_size.width {
                    rgb_data.push(f32::from(resized.get_pixel(x, y)[channel]) / 255.0);
                }
            }
        }
    }

    #[cfg(not(feature = "vision-preproc"))]
    let frame_count = sampled_paths.len();
    #[cfg(not(feature = "vision-preproc"))]
    let rgb_frames = Tensor::from_vec(
        rgb_data,
        (
            frame_count,
            3,
            target_size.height as usize,
            target_size.width as usize,
        ),
        device,
    )
    .map_err(|_| Lux3dError::InvalidInput("failed to materialize Pi3 RGB tensor"))?;

    #[cfg(not(feature = "vision-preproc"))]
    let mean = Tensor::from_vec(vec![0.485_f32, 0.456, 0.406], (1, 1, 3, 1, 1), device)
        .map_err(|_| Lux3dError::InvalidInput("failed to materialize Pi3 mean tensor"))?;
    #[cfg(not(feature = "vision-preproc"))]
    let std = Tensor::from_vec(vec![0.229_f32, 0.224, 0.225], (1, 1, 3, 1, 1), device)
        .map_err(|_| Lux3dError::InvalidInput("failed to materialize Pi3 std tensor"))?;
    #[cfg(not(feature = "vision-preproc"))]
    let rgb_batched = rgb_frames
        .unsqueeze(0)
        .map_err(|_| Lux3dError::InvalidInput("failed to batch Pi3 RGB tensor"))?;
    #[cfg(not(feature = "vision-preproc"))]
    let centered = rgb_batched
        .broadcast_sub(&mean)
        .map_err(|_| Lux3dError::InvalidInput("failed to subtract Pi3 mean"))?;
    #[cfg(not(feature = "vision-preproc"))]
    let normalized_frames = centered
        .broadcast_div(&std)
        .map_err(|_| Lux3dError::InvalidInput("failed to divide Pi3 std"))?;

    Ok(Pi3PreparedInputs {
        interval,
        original_size,
        target_size,
        rgb_frames,
        normalized_frames,
    })
}

fn default_interval_for(source: &Path) -> usize {
    match source.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if ext.eq_ignore_ascii_case("mp4") => 10,
        _ => 1,
    }
}

#[derive(Debug)]
struct TempFrameDir {
    path: PathBuf,
}

impl TempFrameDir {
    fn new() -> Result<Self> {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("lux3d-pi3-frames-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&path).map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.clone(),
            source,
        })?;
        Ok(Self { path })
    }
}

impl Drop for TempFrameDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[derive(Debug)]
struct SampledPi3Frames {
    paths: Vec<PathBuf>,
    _temp_dir: Option<TempFrameDir>,
}

fn load_sampled_pi3_frames(source: &Path, interval: usize) -> Result<SampledPi3Frames> {
    if source.is_dir() {
        let entries = collect_rgb_entries(source)?;
        if entries.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "input directory did not contain any RGB frames",
            ));
        }
        return Ok(SampledPi3Frames {
            paths: entries.into_iter().step_by(interval).collect(),
            _temp_dir: None,
        });
    }

    if source
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("mp4"))
    {
        return extract_video_frames(source, interval);
    }

    Err(Lux3dError::InvalidInput(
        "Pi3 input source must be a directory or .mp4 file",
    ))
}

fn collect_rgb_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir)
        .map_err(|source| Lux3dError::CanonicalManifestIo {
            path: source_pathbuf(dir),
            source,
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg"))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    // The vendor Pi3 loader uses Python's lexical sorted(...), so keep the same
    // ordering here rather than human/natural sorting.
    entries.sort_by(|lhs, rhs| {
        let lhs_name = lhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| lhs.to_string_lossy().to_ascii_lowercase());
        let rhs_name = rhs
            .file_name()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_else(|| rhs.to_string_lossy().to_ascii_lowercase());
        lhs_name.cmp(&rhs_name).then_with(|| lhs.cmp(rhs))
    });
    Ok(entries)
}

fn extract_video_frames(source: &Path, interval: usize) -> Result<SampledPi3Frames> {
    if !source.is_file() {
        return Err(Lux3dError::InvalidInput(
            "Pi3 video input file was not found",
        ));
    }

    let temp_dir = TempFrameDir::new()?;
    let output_pattern = temp_dir.path.join("%06d.png");
    let filter = format!("select=not(mod(n\\,{interval}))");
    let output = Command::new("ffmpeg")
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(source)
        .arg("-vf")
        .arg(filter)
        .arg("-vsync")
        .arg("vfr")
        .arg(output_pattern.as_os_str())
        .output()
        .map_err(|_| Lux3dError::InvalidInput("failed to start ffmpeg for Pi3 video input"))?;

    if !output.status.success() {
        return Err(Lux3dError::InvalidInput(
            "failed to extract Pi3 video frames with ffmpeg",
        ));
    }

    let paths = collect_rgb_entries(&temp_dir.path)?;
    if paths.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "Pi3 video input did not yield any RGB frames",
        ));
    }

    Ok(SampledPi3Frames {
        paths,
        _temp_dir: Some(temp_dir),
    })
}

fn source_pathbuf(path: &Path) -> PathBuf {
    path.to_path_buf()
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use std::path::PathBuf;

    #[cfg(feature = "cuda")]
    use candle_core::Device;

    #[cfg(feature = "cuda")]
    use super::Pi3Pipeline;
    #[cfg(feature = "cuda")]
    use crate::{
        ModelFamily,
        test_support::{GpuTestLock, model_asset_options, runtime_root},
    };

    #[cfg(feature = "cuda")]
    fn repo_root() -> PathBuf {
        runtime_root()
    }

    #[cfg(feature = "cuda")]
    fn accel_device() -> Device {
        Device::new_cuda(0).expect("CUDA required for runtime cache tests")
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn pi3_encoder_reuses_cached_interpolated_positional_embeddings() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let device = accel_device();
        let pipeline =
            Pi3Pipeline::load(model_asset_options(ModelFamily::Pi3)).expect("pi3 pipeline");
        let prepared = pipeline
            .prepare_inputs_from_path(
                &repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("house"),
                None,
                &device,
            )
            .expect("pi3 prepared inputs");
        let bundle = pipeline.bundle_for(&device).expect("pi3 bundle");

        assert_eq!(0, bundle.encoder.cached_pos_encoding_entries());
        pipeline
            .encode_patch_tokens(&prepared.normalized_frames)
            .expect("first encode");
        assert_eq!(1, bundle.encoder.cached_pos_encoding_entries());
        pipeline
            .encode_patch_tokens(&prepared.normalized_frames)
            .expect("second encode");
        assert_eq!(1, bundle.encoder.cached_pos_encoding_entries());
    }
}
