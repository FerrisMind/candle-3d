use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, GroupNorm, LayerNorm, Linear,
    Module, VarBuilder, conv_transpose2d, conv2d, group_norm, layer_norm,
};
use image::imageops::FilterType;
use ndarray::{Array3, Axis, Ix3};
use ndarray_npy::NpzReader;

use crate::{
    CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result, contracts::SpatialSize,
    error::Lux3dError, export::Pi3ExportStage, geometry::Pi3GeometryStage, load_canonical_weights,
    neural::Pi3xNeuralStage, preprocess::Pi3xPreprocessStage,
};

#[cfg(feature = "vision-preproc")]
use super::vision_preproc;
use super::{
    DeviceLocalCache,
    attention_math::{Rope2d, exact_query_chunked_sdpa, position_getter},
    nn_blocks::{LayerScale, Mlp, linear},
    path_utils::sort_paths_natural,
    pi3_decoder::{load_pi3_camera_decoder, load_pi3_conf_decoder, load_pi3_point_decoder},
    pi3_encoder::{load_pi3_depth_encoder, load_pi3_encoder},
    pi3_heads::load_camera_head,
    point_camera_ops::{
        export_mask_from_local_points_and_confidence, non_edge_mask_from_local_points,
        world_points_from_local_and_pose,
    },
};

#[derive(Debug)]
struct Pi3xModelBundle {
    encoder: super::pi3_encoder::Pi3DinoEncoder,
    depth_encoder: super::pi3_encoder::Pi3DinoEncoder,
    ray_embed: Pi3xPatchEmbed,
    depth_emb: Tensor,
    decoder: Pi3xCoreDecoder,
    point_decoder: super::pi3_decoder::Pi3BranchDecoder,
    conf_decoder: super::pi3_decoder::Pi3BranchDecoder,
    camera_decoder: super::pi3_decoder::Pi3BranchDecoder,
    camera_head: super::pi3_heads::CameraHead,
    point_head: Pi3xConvHead,
    conf_head: Pi3xConvHead,
    metric_token: Tensor,
    metric_decoder: Pi3xContextOnlyTransformerDecoder,
    metric_head: Linear,
}

impl Pi3xModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device) -> Result<Self> {
        let vb = unsafe {
            weights.var_builder(DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open Pi3X canonical weights: {source}"),
                }
            })?
        };
        Ok(Self {
            encoder: load_pi3_encoder(vb.pp("encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X encoder: {source}"),
                }
            })?,
            depth_encoder: load_pi3_depth_encoder(vb.pp("depth_encoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X depth encoder: {source}"),
                }
            })?,
            ray_embed: Pi3xPatchEmbed::load(vb.pp("ray_embed"), 14, 2, 1024).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X ray embed: {source}"),
                }
            })?,
            depth_emb: vb.get((1, 1, 1024), "depth_emb").map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load Pi3X depth embedding: {source}"),
                }
            })?,
            decoder: Pi3xCoreDecoder::load(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X core decoder: {source}"),
                }
            })?,
            point_decoder: load_pi3_point_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X point decoder: {source}"),
                }
            })?,
            conf_decoder: load_pi3_conf_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X confidence decoder: {source}"),
                }
            })?,
            camera_decoder: load_pi3_camera_decoder(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X camera decoder: {source}"),
                }
            })?,
            camera_head: load_camera_head(vb.clone()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X camera head: {source}"),
                }
            })?,
            point_head: Pi3xConvHead::load(vb.pp("point_head"), &[2, 1], true).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X point head: {source}"),
                },
            )?,
            conf_head: Pi3xConvHead::load(vb.pp("conf_head"), &[1], true).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X confidence head: {source}"),
                }
            })?,
            metric_token: vb.get((1, 1, 2048), "metric_token").map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load Pi3X metric token: {source}"),
                }
            })?,
            metric_decoder: Pi3xContextOnlyTransformerDecoder::load(
                vb.pp("metric_decoder"),
                2048,
                512,
                512,
                5,
                8,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to construct Pi3X metric decoder: {source}"),
            })?,
            metric_head: linear(vb.pp("metric_head"), 512, 1, true).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct Pi3X metric head: {source}"),
                }
            })?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Pi3xPreparedInputs {
    pub interval: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub rgb_frames: Tensor,
    pub normalized_frames: Tensor,
    pub depths: Option<Tensor>,
    pub intrinsics: Option<Tensor>,
    pub poses: Option<Tensor>,
    pub rays: Option<Tensor>,
    pub mask_add_depth: Tensor,
    pub mask_add_ray: Tensor,
    pub mask_add_pose: Tensor,
}

#[derive(Debug, Clone)]
pub struct Pi3xInferenceOutput {
    pub local_points: Tensor,
    pub confidence_logits: Tensor,
    pub camera_poses: Tensor,
    pub points: Tensor,
    pub rays: Tensor,
    pub metric: Tensor,
    pub export_mask: Tensor,
    pub non_edge_mask: Tensor,
    pub rgb_frames: Tensor,
}

#[derive(Debug, Clone)]
pub struct Pi3xVoOutput {
    pub points: Tensor,
    pub camera_poses: Tensor,
    pub confidence_logits: Tensor,
    pub export_mask: Tensor,
    pub sim3_transforms: Tensor,
    pub rgb_frames: Tensor,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Pi3xInjectConditions {
    pub pose: bool,
    pub depth: bool,
    pub ray: bool,
}

#[derive(Debug, Clone)]
pub struct Pi3xPipeline {
    pub weights: CanonicalWeightSetPaths,
    pub preprocess: Pi3xPreprocessStage,
    pub neural: Pi3xNeuralStage,
    pub geometry: Pi3GeometryStage,
    pub export: Pi3ExportStage,
    bundle_cache: Arc<DeviceLocalCache<Pi3xModelBundle>>,
}

#[derive(Debug, Clone)]
pub struct Pi3xVoPipeline {
    pub core: Pi3xPipeline,
}

impl Pi3xPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::Pi3x, model_assets)?,
            preprocess: Pi3xPreprocessStage::default(),
            neural: Pi3xNeuralStage,
            geometry: Pi3GeometryStage,
            export: Pi3ExportStage,
            bundle_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<Pi3xModelBundle>> {
        self.bundle_cache
            .get_or_try_init(device, || Pi3xModelBundle::load(&self.weights, device))
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        conditions_path: Option<&Path>,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<Pi3xPreparedInputs> {
        self.preprocess
            .prepare_inputs_from_path(source, conditions_path, interval, device)
    }

    pub fn export_ply(&self, output: &Pi3xInferenceOutput, path: &Path) -> Result<()> {
        let point_cloud = self
            .geometry
            .assemble_cpu(&crate::runtime::Pi3InferenceOutput {
                local_points: output.local_points.clone(),
                confidence_logits: output.confidence_logits.clone(),
                camera_poses: output.camera_poses.clone(),
                points: output.points.clone(),
                export_mask: output.export_mask.clone(),
                rgb_frames: output.rgb_frames.clone(),
            })?;
        self.export.write_ply(&point_cloud, path)
    }

    pub fn infer_from_path(
        &self,
        source: &Path,
        conditions_path: Option<&Path>,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<Pi3xInferenceOutput> {
        let prepared = self.prepare_inputs_from_path(source, conditions_path, interval, device)?;
        self.neural.infer_tensors(self, &prepared)
    }

    pub(crate) fn infer_prepared(
        &self,
        _prepared: &Pi3xPreparedInputs,
    ) -> Result<Pi3xInferenceOutput> {
        let prepared = _prepared;
        let device = prepared.normalized_frames.device();
        let bundle = self.bundle_for(device)?;

        let dims = prepared.normalized_frames.dims();
        let frames = dims[1];
        let channels = dims[2];
        let height = dims[3];
        let width = dims[4];
        let batch = dims[0];
        let patch_h = height / 14;
        let patch_w = width / 14;

        let image_inputs = prepared
            .normalized_frames
            .reshape((frames, channels, height, width))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3X normalized frames: {source}"),
            })?;
        let mut hidden = bundle
            .encoder
            .forward_patch_tokens(&image_inputs)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X encoder forward failed: {source}"),
            })?;

        let mut normalized_depths = None;
        let mut poses_rel = None;
        if let Some(depths) = prepared.depths.as_ref() {
            let (depths_norm, depth_scale) = normalize_depth_mean(depths, device)?;
            let depth_mask = depths_norm
                .gt(0.0)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth mask failed: {source}"),
                })?
                .to_dtype(DType::F32)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth mask cast failed: {source}"),
                })?;
            let depth_inputs = Tensor::cat(
                &[
                    &depths_norm
                        .reshape((frames, 1, height, width))
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X depth reshape failed: {source}"),
                        })?,
                    &depth_mask
                        .reshape((frames, 1, height, width))
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X depth mask reshape failed: {source}"),
                        })?,
                ],
                1,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X depth concat failed: {source}"),
            })?;
            let depth_hidden = bundle
                .depth_encoder
                .forward_patch_tokens(&depth_inputs)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth encoder forward failed: {source}"),
                })?;
            let depth_hidden = depth_hidden
                .broadcast_add(&bundle.depth_emb)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth embedding add failed: {source}"),
                })?;
            let depth_mask = prepared
                .mask_add_depth
                .to_dtype(DType::F32)
                .and_then(|tensor| tensor.reshape((frames, 1, 1)))
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth mask reshape failed: {source}"),
                })?;
            let depth_hidden = depth_hidden.broadcast_mul(&depth_mask).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X depth mask apply failed: {source}"),
                }
            })?;
            hidden = hidden.broadcast_add(&depth_hidden).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X hidden depth fusion failed: {source}"),
                }
            })?;
            if let Some(poses) = prepared.poses.as_ref() {
                let mut poses_norm = relative_poses(poses, device)?;
                poses_norm = divide_pose_translation(&poses_norm, &depth_scale)?;
                poses_rel = Some(poses_norm);
            }
            normalized_depths = Some(depths_norm);
        } else if let Some(poses) = prepared.poses.as_ref() {
            poses_rel = Some(relative_poses(poses, device)?);
        }

        if let Some(rays) = prepared.rays.as_ref() {
            let mut ray_inputs = rays
                .to_device(device)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to move Pi3X rays to device: {source}"),
                })?
                .clone();
            if ray_inputs.dim(D::Minus1).unwrap_or(0) == 3 {
                let xy = ray_inputs.i((.., .., .., .., ..2)).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X ray xy slice failed: {source}"),
                    }
                })?;
                let z = ray_inputs.i((.., .., .., .., 2..3)).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X ray z slice failed: {source}"),
                    }
                })?;
                ray_inputs = xy.broadcast_div(&z).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X ray xy/z failed: {source}"),
                    }
                })?;
            }
            let ray_inputs = ray_inputs
                .reshape((frames, height, width, 2))
                .and_then(|tensor| tensor.permute((0, 3, 1, 2)))
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X rays reshape failed: {source}"),
                })?;
            let ray_hidden = bundle.ray_embed.forward(&ray_inputs).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X ray embed forward failed: {source}"),
                }
            })?;
            let ray_mask = prepared
                .mask_add_ray
                .to_dtype(DType::F32)
                .and_then(|tensor| tensor.reshape((frames, 1, 1)))
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X ray mask reshape failed: {source}"),
                })?;
            let ray_hidden = ray_hidden.broadcast_mul(&ray_mask).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X ray mask apply failed: {source}"),
                }
            })?;
            hidden = hidden.broadcast_add(&ray_hidden).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X hidden ray fusion failed: {source}"),
                }
            })?;
        }

        let hidden = hidden
            .reshape((
                batch,
                frames,
                hidden.dim(1).unwrap_or_default(),
                hidden.dim(2).unwrap_or_default(),
            ))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X hidden reshape failed: {source}"),
            })?;
        let poses_ref = poses_rel.as_ref();
        let pose_mask = if poses_ref.is_some() {
            Some(&prepared.mask_add_pose)
        } else {
            None
        };
        let (hidden, pos) = bundle
            .decoder
            .decode(&hidden, frames, height, width, poses_ref, pose_mask)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X decoder failed: {source}"),
            })?;

        let ret_point = bundle
            .point_decoder
            .forward(&hidden, &pos)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X point decoder failed: {source}"),
            })?;
        let ret_camera = bundle
            .camera_decoder
            .forward(&hidden, &pos)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X camera decoder failed: {source}"),
            })?;
        let ret_conf = bundle
            .conf_decoder
            .forward(&hidden, &pos)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X confidence decoder failed: {source}"),
            })?;
        let pos_hw = pos
            .reshape((batch, frames * pos.dim(1).unwrap_or_default(), 2))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X metric position reshape failed: {source}"),
            })?;
        let metric_hidden = bundle
            .metric_decoder
            .forward(
                &bundle
                    .metric_token
                    .broadcast_as((batch, 1, 2048))
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X metric token broadcast failed: {source}"),
                    })?,
                &hidden
                    .reshape((batch, frames * hidden.dim(1).unwrap_or_default(), 2048))
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X metric hidden reshape failed: {source}"),
                    })?,
                &pos_hw.i((.., 0..1, ..)).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X metric qpos failed: {source}"),
                    }
                })?,
                &pos_hw,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X metric decoder failed: {source}"),
            })?;
        let metric = bundle
            .metric_head
            .forward(&metric_hidden)
            .and_then(|tensor| tensor.reshape((batch,)))
            .and_then(|tensor| tensor.exp())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X metric head failed: {source}"),
            })?;

        let point_feat = ret_point.i((.., 5.., ..)).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X point feature slice failed: {source}"),
            }
        })?;
        let point_outputs = run_pi3x_conv_head_chunked(
            &bundle.point_head,
            &point_feat,
            patch_h,
            patch_w,
            height,
            width,
            2,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3X point head failed: {source}"),
        })?;
        let xy = point_outputs[0]
            .permute((0, 2, 3, 1))
            .and_then(|tensor| tensor.reshape((batch, frames, height, width, 2)))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X xy reshape failed: {source}"),
            })?;
        let z = point_outputs[1]
            .permute((0, 2, 3, 1))
            .and_then(|tensor| tensor.reshape((batch, frames, height, width, 1)))
            .and_then(|tensor| tensor.clamp(f64::NEG_INFINITY, 15.0))
            .and_then(|tensor| tensor.exp())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X z reshape failed: {source}"),
            })?;
        let local_points = Tensor::cat(
            &[
                &xy.broadcast_mul(&z)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X local point xy*z failed: {source}"),
                    })?,
                &z,
            ],
            D::Minus1,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3X local point concat failed: {source}"),
        })?;
        let rays = normalize_last_dim(
            &Tensor::cat(
                &[
                    &xy,
                    &Tensor::ones_like(&z).map_err(|source| {
                        Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X ones_like failed: {source}"),
                        }
                    })?,
                ],
                D::Minus1,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X rays concat failed: {source}"),
            })?,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3X ray normalization failed: {source}"),
        })?;
        let mut camera_poses = bundle
            .camera_head
            .forward(&ret_camera, patch_h, patch_w)
            .and_then(|tensor| tensor.reshape((batch, frames, 4, 4)))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X camera head failed: {source}"),
            })?;

        let conf_feat =
            ret_conf
                .i((.., 5.., ..))
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X conf feature slice failed: {source}"),
                })?;
        let conf_outputs = run_pi3x_conv_head_chunked(
            &bundle.conf_head,
            &conf_feat,
            patch_h,
            patch_w,
            height,
            width,
            2,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Pi3X confidence head failed: {source}"),
        })?;
        let confidence_logits = conf_outputs[0]
            .permute((0, 2, 3, 1))
            .and_then(|tensor| tensor.reshape((batch, frames, height, width, 1)))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X confidence reshape failed: {source}"),
            })?;

        let metric_broadcast = metric.reshape((batch, 1, 1, 1, 1)).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X metric broadcast reshape failed: {source}"),
            }
        })?;
        let mut points =
            world_points_from_local_and_pose(&local_points, &camera_poses).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X world point assembly failed: {source}"),
                }
            })?;
        points = points.broadcast_mul(&metric_broadcast).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X point metric scaling failed: {source}"),
            }
        })?;
        camera_poses = scale_pose_translation(&camera_poses, &metric).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X pose metric scaling failed: {source}"),
            }
        })?;
        let local_points = local_points
            .broadcast_mul(&metric_broadcast)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X local point metric scaling failed: {source}"),
            })?;
        let export_mask =
            export_mask_from_local_points_and_confidence(&local_points, &confidence_logits)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X export mask failed: {source}"),
                })?;
        let non_edge_mask = non_edge_mask_from_local_points(&local_points).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3X non-edge mask failed: {source}"),
            }
        })?;

        let _ = normalized_depths;
        Ok(Pi3xInferenceOutput {
            local_points,
            confidence_logits,
            camera_poses,
            points,
            rays,
            metric,
            export_mask,
            non_edge_mask,
            rgb_frames: prepared.rgb_frames.clone(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn infer_vo_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        chunk_size: Option<usize>,
        overlap: Option<usize>,
        conf_threshold: Option<f32>,
        inject_conditions: Pi3xInjectConditions,
        device: &Device,
    ) -> Result<Pi3xVoOutput> {
        let chunk_size = chunk_size.unwrap_or(16);
        let overlap = overlap.unwrap_or(6);
        let conf_threshold = conf_threshold.unwrap_or(0.05);
        if chunk_size == 0 {
            return Err(Lux3dError::InvalidInput(
                "Pi3X VO chunk_size must be at least 1",
            ));
        }
        if overlap >= chunk_size {
            return Err(Lux3dError::InvalidInput(
                "Pi3X VO overlap must be smaller than chunk_size",
            ));
        }

        let prepared = self.prepare_inputs_from_path(source, None, interval, device)?;
        let dims = prepared.rgb_frames.dims();
        let frames = dims[0];
        let height = dims[2];
        let width = dims[3];
        let mut merged_points = Vec::new();
        let mut merged_poses = Vec::new();
        let mut merged_conf = Vec::new();
        let mut merged_export_mask = Vec::new();
        let mut sim3_transforms = Vec::new();

        let mut prev_global_pts_overlap: Option<Tensor> = None;
        let mut prev_global_mask_overlap: Option<Tensor> = None;
        let mut prev_aligned_poses_overlap: Option<Tensor> = None;
        let mut prev_local_depth_overlap: Option<Tensor> = None;
        let mut prev_local_conf_overlap: Option<Tensor> = None;
        let mut prev_rays_overlap: Option<Tensor> = None;

        let stride = chunk_size.saturating_sub(overlap).max(1);
        let mut start = 0usize;
        while start < frames {
            let end = (start + chunk_size).min(frames);
            let current_len = end - start;
            if current_len <= overlap && start > 0 {
                break;
            }

            let mut chunk = Pi3xPreparedInputs {
                interval: prepared.interval,
                original_size: prepared.original_size,
                target_size: prepared.target_size,
                rgb_frames: prepared.rgb_frames.narrow(0, start, current_len).map_err(
                    |source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO rgb chunk failed: {source}"),
                    },
                )?,
                normalized_frames: prepared
                    .normalized_frames
                    .narrow(1, start, current_len)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO normalized chunk failed: {source}"),
                    })?,
                depths: None,
                intrinsics: None,
                poses: None,
                rays: None,
                mask_add_depth: Tensor::zeros((1, current_len), DType::U8, device).map_err(
                    |source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO depth mask init failed: {source}"),
                    },
                )?,
                mask_add_ray: Tensor::zeros((1, current_len), DType::U8, device).map_err(
                    |source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO ray mask init failed: {source}"),
                    },
                )?,
                mask_add_pose: Tensor::zeros((1, current_len), DType::U8, device).map_err(
                    |source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO pose mask init failed: {source}"),
                    },
                )?,
            };

            if start > 0 {
                if inject_conditions.pose {
                    let overlap_take = overlap.min(current_len);
                    let identity_suffix =
                        repeated_identity_poses(current_len - overlap_take, device)?;
                    chunk.poses = Some(if let Some(prev) = prev_aligned_poses_overlap.as_ref() {
                        Tensor::cat(&[prev, &identity_suffix], 1).map_err(|source| {
                            Lux3dError::CanonicalWeightsValidation {
                                message: format!("Pi3X VO prior pose concat failed: {source}"),
                            }
                        })?
                    } else {
                        repeated_identity_poses(current_len, device)?
                    });
                    chunk.mask_add_pose = build_prefix_mask(current_len, overlap, device)?;
                }

                if inject_conditions.depth {
                    if let (Some(prev_depth), Some(prev_conf)) = (
                        prev_local_depth_overlap.as_ref(),
                        prev_local_conf_overlap.as_ref(),
                    ) {
                        let prev_depths = prev_depth.clone();
                        let valid_prev = prev_conf.ge(conf_threshold).map_err(|source| {
                            Lux3dError::CanonicalWeightsValidation {
                                message: format!(
                                    "Pi3X VO prior confidence threshold failed: {source}"
                                ),
                            }
                        })?;
                        let zero_depth = Tensor::zeros(prev_depths.shape(), DType::F32, device)
                            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                                message: format!("Pi3X VO zero prior depth init failed: {source}"),
                            })?;
                        let filtered_prev = valid_prev
                            .where_cond(&prev_depths, &zero_depth)
                            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                                message: format!("Pi3X VO prior depth filtering failed: {source}"),
                            })?;
                        let suffix = if current_len > overlap {
                            Tensor::zeros(
                                (1, current_len - overlap, height, width),
                                DType::F32,
                                device,
                            )
                            .map_err(|source| {
                                Lux3dError::CanonicalWeightsValidation {
                                    message: format!(
                                        "Pi3X VO prior depth suffix init failed: {source}"
                                    ),
                                }
                            })?
                        } else {
                            Tensor::zeros((1, 0, height, width), DType::F32, device).map_err(
                                |source| Lux3dError::CanonicalWeightsValidation {
                                    message: format!(
                                        "Pi3X VO prior depth suffix init failed: {source}"
                                    ),
                                },
                            )?
                        };
                        chunk.depths = Some(Tensor::cat(&[&filtered_prev, &suffix], 1).map_err(
                            |source| Lux3dError::CanonicalWeightsValidation {
                                message: format!("Pi3X VO prior depth concat failed: {source}"),
                            },
                        )?);
                        chunk.mask_add_depth = build_prefix_mask(current_len, overlap, device)?;
                    }
                }

                if inject_conditions.ray {
                    if let Some(prev_rays) = prev_rays_overlap.as_ref() {
                        let suffix = if current_len > overlap {
                            Tensor::zeros(
                                (1, current_len - overlap, height, width, 3),
                                DType::F32,
                                device,
                            )
                            .map_err(|source| {
                                Lux3dError::CanonicalWeightsValidation {
                                    message: format!(
                                        "Pi3X VO prior ray suffix init failed: {source}"
                                    ),
                                }
                            })?
                        } else {
                            Tensor::zeros((1, 0, height, width, 3), DType::F32, device).map_err(
                                |source| Lux3dError::CanonicalWeightsValidation {
                                    message: format!(
                                        "Pi3X VO prior ray suffix init failed: {source}"
                                    ),
                                },
                            )?
                        };
                        chunk.rays =
                            Some(Tensor::cat(&[prev_rays, &suffix], 1).map_err(|source| {
                                Lux3dError::CanonicalWeightsValidation {
                                    message: format!("Pi3X VO prior ray concat failed: {source}"),
                                }
                            })?);
                        chunk.mask_add_ray = build_prefix_mask(current_len, overlap, device)?;
                    }
                }
            }

            let pred = self.infer_prepared(&chunk)?;
            let curr_local_depth = pred.local_points.i((.., .., .., .., 2)).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO local depth slice failed: {source}"),
                }
            })?;
            let mut curr_conf =
                candle_nn::ops::sigmoid(&pred.confidence_logits.i((.., .., .., .., 0)).map_err(
                    |source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO confidence slice failed: {source}"),
                    },
                )?)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO confidence sigmoid failed: {source}"),
                })?;
            let non_edge =
                non_edge_mask_from_local_points(&pred.local_points).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO non-edge mask failed: {source}"),
                    }
                })?;
            let zero_conf = Tensor::zeros(curr_conf.shape(), curr_conf.dtype(), curr_conf.device())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO zero mask failed: {source}"),
                })?;
            curr_conf = non_edge
                .where_cond(&curr_conf, &zero_conf)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO edge suppression failed: {source}"),
                })?;
            let mut curr_mask = curr_conf.ge(conf_threshold).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO threshold mask failed: {source}"),
                }
            })?;
            let valid_count = curr_mask
                .to_dtype(DType::U32)
                .and_then(|tensor| tensor.sum_all())
                .and_then(|tensor| tensor.to_scalar::<u32>())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO valid-count reduction failed: {source}"),
                })?;
            if valid_count < 10 {
                curr_mask = adaptive_confidence_mask(&curr_conf)?;
            }

            let (aligned_points, aligned_poses) = if start == 0 {
                (pred.points.clone(), pred.camera_poses.clone())
            } else {
                let src_pts = pred.points.narrow(1, 0, overlap).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO source overlap failed: {source}"),
                    }
                })?;
                let src_mask = curr_mask.narrow(1, 0, overlap).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO source mask overlap failed: {source}"),
                    }
                })?;
                let tgt_pts = prev_global_pts_overlap.as_ref().ok_or_else(|| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: "Pi3X VO missing previous overlap points".to_string(),
                    }
                })?;
                let tgt_mask = prev_global_mask_overlap.as_ref().ok_or_else(|| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: "Pi3X VO missing previous overlap mask".to_string(),
                    }
                })?;
                let sim3 = compute_sim3_umeyama_masked(&src_pts, tgt_pts, &src_mask, tgt_mask)?;
                sim3_transforms.push(sim3.clone());
                (
                    apply_sim3_to_points(&pred.points, &sim3)?,
                    apply_sim3_to_poses(&pred.camera_poses, &sim3)?,
                )
            };

            if start == 0 {
                merged_points.push(aligned_points.clone());
                merged_poses.push(aligned_poses.clone());
                merged_conf.push(curr_conf.unsqueeze(D::Minus1).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO conf unsqueeze failed: {source}"),
                    }
                })?);
                merged_export_mask.push(curr_mask.clone());
            } else {
                merged_points.push(
                    aligned_points
                        .narrow(1, overlap, current_len - overlap)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X VO merged points narrow failed: {source}"),
                        })?,
                );
                merged_poses.push(
                    aligned_poses
                        .narrow(1, overlap, current_len - overlap)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X VO merged poses narrow failed: {source}"),
                        })?,
                );
                merged_conf.push(
                    curr_conf
                        .unsqueeze(D::Minus1)
                        .and_then(|tensor| tensor.narrow(1, overlap, current_len - overlap))
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X VO merged conf narrow failed: {source}"),
                        })?,
                );
                merged_export_mask.push(
                    curr_mask
                        .narrow(1, overlap, current_len - overlap)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Pi3X VO merged export mask narrow failed: {source}"),
                        })?,
                );
            }

            let overlap_take = overlap.min(current_len);
            prev_global_pts_overlap = Some(
                aligned_points
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO store overlap points failed: {source}"),
                    })?,
            );
            prev_global_mask_overlap = Some(
                curr_mask
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO store overlap mask failed: {source}"),
                    })?,
            );
            prev_aligned_poses_overlap = Some(
                aligned_poses
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO prior aligned pose slice failed: {source}"),
                    })?,
            );
            prev_local_depth_overlap = Some(
                curr_local_depth
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO prior local depth slice failed: {source}"),
                    })?,
            );
            prev_local_conf_overlap = Some(
                curr_conf
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO prior confidence slice failed: {source}"),
                    })?,
            );
            prev_rays_overlap = Some(
                pred.rays
                    .narrow(1, current_len - overlap_take, overlap_take)
                    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO prior rays slice failed: {source}"),
                    })?,
            );
            if end == frames {
                break;
            }
            start += stride;
        }

        let point_refs = merged_points.iter().collect::<Vec<_>>();
        let pose_refs = merged_poses.iter().collect::<Vec<_>>();
        let conf_refs = merged_conf.iter().collect::<Vec<_>>();
        let export_mask_refs = merged_export_mask.iter().collect::<Vec<_>>();

        Ok(Pi3xVoOutput {
            points: Tensor::cat(&point_refs, 1).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO final point concat failed: {source}"),
                }
            })?,
            camera_poses: Tensor::cat(&pose_refs, 1).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO final pose concat failed: {source}"),
                }
            })?,
            confidence_logits: Tensor::cat(&conf_refs, 1).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO final conf concat failed: {source}"),
                }
            })?,
            export_mask: Tensor::cat(&export_mask_refs, 1).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Pi3X VO final export mask concat failed: {source}"),
                }
            })?,
            sim3_transforms: if sim3_transforms.is_empty() {
                Tensor::zeros((0, 1, 4, 4), DType::F32, device).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO empty sim3 tensor init failed: {source}"),
                    }
                })?
            } else {
                let refs = sim3_transforms.iter().collect::<Vec<_>>();
                Tensor::stack(&refs, 0).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3X VO sim3 stack failed: {source}"),
                    }
                })?
            },
            rgb_frames: prepared.rgb_frames.clone(),
        })
    }
}

impl Pi3xVoPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Ok(Self {
            core: Pi3xPipeline::load(model_assets)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn infer_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        chunk_size: Option<usize>,
        overlap: Option<usize>,
        conf_threshold: Option<f32>,
        inject_conditions: Pi3xInjectConditions,
        device: &Device,
    ) -> Result<Pi3xVoOutput> {
        self.core.neural.infer_vo_from_path(
            &self.core,
            source,
            interval,
            chunk_size,
            overlap,
            conf_threshold,
            inject_conditions,
            device,
        )
    }

    pub fn export_ply(&self, output: &Pi3xVoOutput, path: &Path) -> Result<()> {
        let point_cloud = self
            .core
            .geometry
            .assemble_cpu(&crate::runtime::Pi3InferenceOutput {
                local_points: output.points.clone(),
                confidence_logits: output.confidence_logits.clone(),
                camera_poses: output.camera_poses.clone(),
                points: output.points.clone(),
                export_mask: output.export_mask.clone(),
                rgb_frames: output.rgb_frames.clone(),
            })?;
        self.core.export.write_ply(&point_cloud, path)
    }
}

#[derive(Debug)]
struct Pi3xPatchEmbed {
    proj: Conv2d,
    patch_size: usize,
}

impl Pi3xPatchEmbed {
    fn load(
        vb: VarBuilder,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
    ) -> CandleResult<Self> {
        let config = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        Ok(Self {
            proj: conv2d(in_channels, embed_dim, patch_size, config, vb.pp("proj"))?,
            patch_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (_b, _c, h, w) = xs.dims4()?;
        if h % self.patch_size != 0 || w % self.patch_size != 0 {
            candle_core::bail!("Pi3x patch embed expects image size divisible by patch size");
        }
        let xs = self.proj.forward(xs)?;
        let (b, c, h, w) = xs.dims4()?;
        xs.reshape((b, c, h * w))?.transpose(1, 2)
    }
}

#[derive(Debug)]
struct Pi3xCrossAttentionRope {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: Rope2d,
}

impl Pi3xCrossAttentionRope {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            q_proj: linear(vb.pp("q_proj"), dim, dim, true)?,
            k_proj: linear(vb.pp("k_proj"), dim, dim, true)?,
            v_proj: linear(vb.pp("v_proj"), dim, dim, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope,
        })
    }

    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        qpos: &Tensor,
        kpos: &Tensor,
    ) -> CandleResult<Tensor> {
        let (b, nq, c) = query.dims3()?;
        let nk = key.dim(1)?;
        let q = self
            .q_proj
            .forward(query)?
            .reshape((b, nq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(key)?
            .reshape((b, nk, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(value)?
            .reshape((b, nk, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let q_cache = self.rope.embeddings(qpos, self.head_dim)?;
        let k_cache = self.rope.embeddings(kpos, self.head_dim)?;
        let q = self
            .rope
            .apply_with_embeddings(&q, &q_cache)?
            .contiguous()?;
        let k = self
            .rope
            .apply_with_embeddings(&k, &k_cache)?
            .contiguous()?;
        let v = v.contiguous()?;
        let out = exact_query_chunked_sdpa(&q, &k, &v, self.scale, 128)?
            .transpose(1, 2)?
            .reshape((b, nq, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct Pi3xCrossOnlyBlock {
    norm2: LayerNorm,
    norm_y: LayerNorm,
    cross_attn: Pi3xCrossAttentionRope,
    norm3: LayerNorm,
    mlp: Mlp,
}

impl Pi3xCrossOnlyBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        Ok(Self {
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            norm_y: layer_norm(dim, 1e-6, vb.pp("norm_y"))?,
            cross_attn: Pi3xCrossAttentionRope::load(vb.pp("cross_attn"), dim, num_heads, rope)?,
            norm3: layer_norm(dim, 1e-6, vb.pp("norm3"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        y: &Tensor,
        xpos: &Tensor,
        ypos: &Tensor,
    ) -> CandleResult<Tensor> {
        let y_ = self.norm_y.forward(y)?;
        let x = (x + self
            .cross_attn
            .forward(&self.norm2.forward(x)?, &y_, &y_, xpos, ypos)?)?;
        let residual = &x;
        self.mlp.forward(&self.norm3.forward(&x)?)? + residual
    }
}

#[derive(Debug)]
struct Pi3xContextOnlyTransformerDecoder {
    projects_x: Linear,
    projects_y: Linear,
    blocks: Vec<Pi3xCrossOnlyBlock>,
    linear_out: Linear,
}

impl Pi3xContextOnlyTransformerDecoder {
    fn load(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
    ) -> CandleResult<Self> {
        let rope = Rope2d::new(100.0);
        let blocks = (0..depth)
            .map(|idx| {
                Pi3xCrossOnlyBlock::load(
                    vb.pp("blocks").pp(idx.to_string()),
                    embed_dim,
                    num_heads,
                    rope,
                )
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            projects_x: linear(vb.pp("projects_x"), in_dim, embed_dim, true)?,
            projects_y: linear(vb.pp("projects_y"), in_dim, embed_dim, true)?,
            blocks,
            linear_out: linear(vb.pp("linear_out"), embed_dim, out_dim, true)?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        context: &Tensor,
        xpos: &Tensor,
        ypos: &Tensor,
    ) -> CandleResult<Tensor> {
        let mut hidden = self.projects_x.forward(hidden)?;
        let context = self.projects_y.forward(context)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, &context, xpos, ypos)?;
        }
        self.linear_out.forward(&hidden)
    }
}

#[derive(Debug)]
struct Pi3xResidualConvBlock {
    norm1: GroupNorm,
    conv1: Conv2d,
    norm2: GroupNorm,
    conv2: Conv2d,
    skip: Option<Conv2d>,
}

impl Pi3xResidualConvBlock {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        hidden_channels: usize,
    ) -> CandleResult<Self> {
        let groups_hidden = (hidden_channels / 32).max(1);
        Ok(Self {
            norm1: group_norm(1, in_channels, 1e-5, vb.pp("layers").pp("0"))?,
            conv1: conv2d(
                in_channels,
                hidden_channels,
                3,
                Conv2dConfig::default(),
                vb.pp("layers").pp("2"),
            )?,
            norm2: group_norm(
                groups_hidden,
                hidden_channels,
                1e-5,
                vb.pp("layers").pp("3"),
            )?,
            conv2: conv2d(
                hidden_channels,
                out_channels,
                3,
                Conv2dConfig::default(),
                vb.pp("layers").pp("5"),
            )?,
            skip: if in_channels != out_channels {
                Some(conv2d(
                    in_channels,
                    out_channels,
                    1,
                    Conv2dConfig::default(),
                    vb.pp("skip_connection"),
                )?)
            } else {
                None
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let residual = if let Some(skip) = &self.skip {
            skip.forward(xs)?
        } else {
            xs.clone()
        };
        let x = self.norm1.forward(xs)?.relu()?;
        let x = candle_nn::ops::replication_pad2d(&x, 1)?;
        let x = self.conv1.forward(&x)?;
        let x = self.norm2.forward(&x)?.relu()?;
        let x = candle_nn::ops::replication_pad2d(&x, 1)?;
        let x = self.conv2.forward(&x)?;
        x + residual
    }
}

#[derive(Debug)]
struct Pi3xUpsampleBlock {
    upsample: ConvTranspose2d,
    conv: Conv2d,
    residuals: Vec<Pi3xResidualConvBlock>,
}

impl Pi3xUpsampleBlock {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        num_residuals: usize,
    ) -> CandleResult<Self> {
        Ok(Self {
            upsample: conv_transpose2d(
                in_channels,
                out_channels,
                2,
                ConvTranspose2dConfig {
                    stride: 2,
                    ..Default::default()
                },
                vb.pp("0").pp("0"),
            )?,
            conv: conv2d(
                out_channels,
                out_channels,
                3,
                Conv2dConfig::default(),
                vb.pp("0").pp("1"),
            )?,
            residuals: (1..=num_residuals)
                .map(|idx| {
                    Pi3xResidualConvBlock::load(
                        vb.pp(idx.to_string()),
                        out_channels,
                        out_channels,
                        out_channels * 2,
                    )
                })
                .collect::<CandleResult<Vec<_>>>()?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let x = self.upsample.forward(xs)?;
        let x = candle_nn::ops::replication_pad2d(&x, 1)?;
        let mut x = self.conv.forward(&x)?;
        for block in &self.residuals {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
struct Pi3xOutputBlock {
    conv_in: Conv2d,
    conv_out: Conv2d,
}

impl Pi3xOutputBlock {
    fn load(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> CandleResult<Self> {
        Ok(Self {
            conv_in: conv2d(in_channels, 32, 3, Conv2dConfig::default(), vb.pp("0"))?,
            conv_out: conv2d(
                32,
                out_channels,
                kernel_size,
                Conv2dConfig::default(),
                vb.pp("2"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, kernel_size: usize) -> CandleResult<Tensor> {
        let x = candle_nn::ops::replication_pad2d(xs, 1)?;
        let x = self.conv_in.forward(&x)?.relu()?;
        let x = if kernel_size > 1 {
            candle_nn::ops::replication_pad2d(&x, kernel_size / 2)?
        } else {
            x
        };
        self.conv_out.forward(&x)
    }
}

#[derive(Debug)]
struct Pi3xConvHead {
    project_identity: bool,
    upsample_blocks: Vec<Pi3xUpsampleBlock>,
    output_blocks: Vec<Pi3xOutputBlock>,
    output_kernel_size: usize,
    using_uv: bool,
}

impl Pi3xConvHead {
    fn load(vb: VarBuilder, output_dims: &[usize], using_uv: bool) -> CandleResult<Self> {
        let upsample_in = [1024usize, 256, 128];
        let upsample_out = [256usize, 128, 64];
        let mut upsample_blocks = Vec::with_capacity(3);
        for (idx, (&in_c, &out_c)) in upsample_in.iter().zip(upsample_out.iter()).enumerate() {
            upsample_blocks.push(Pi3xUpsampleBlock::load(
                vb.pp("upsample_blocks").pp(idx.to_string()),
                in_c + if using_uv { 2 } else { 0 },
                out_c,
                2,
            )?);
        }
        let mut output_blocks = Vec::with_capacity(output_dims.len());
        for (idx, &out_dim) in output_dims.iter().enumerate() {
            output_blocks.push(Pi3xOutputBlock::load(
                vb.pp("output_block").pp(idx.to_string()),
                64 + if using_uv { 2 } else { 0 },
                out_dim,
                1,
            )?);
        }
        Ok(Self {
            project_identity: true,
            upsample_blocks,
            output_blocks,
            output_kernel_size: 1,
            using_uv,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        patch_h: usize,
        patch_w: usize,
        img_h: usize,
        img_w: usize,
    ) -> CandleResult<Vec<Tensor>> {
        let (batch, seq, channels) = hidden_states.dims3()?;
        let mut x = if self.project_identity {
            hidden_states
                .transpose(1, 2)?
                .reshape((batch, channels, patch_h, patch_w))?
        } else {
            candle_core::bail!("Pi3xConvHead only supports identity project path");
        };

        for block in &self.upsample_blocks {
            if self.using_uv {
                x = Tensor::cat(
                    &[
                        &x,
                        &normalized_view_plane_uv(
                            x.dim(3)?,
                            x.dim(2)?,
                            img_w as f32 / img_h as f32,
                            x.device(),
                        )?
                        .broadcast_as((batch, 2, x.dim(2)?, x.dim(3)?))?,
                    ],
                    1,
                )?;
            }
            x = block.forward(&x)?;
        }
        x = x.upsample_bilinear2d(img_h, img_w, false)?;
        if self.using_uv {
            x = Tensor::cat(
                &[
                    &x,
                    &normalized_view_plane_uv(
                        img_w,
                        img_h,
                        img_w as f32 / img_h as f32,
                        x.device(),
                    )?
                    .broadcast_as((batch, 2, img_h, img_w))?,
                ],
                1,
            )?;
        }
        let mut outputs = Vec::with_capacity(self.output_blocks.len());
        for block in &self.output_blocks {
            outputs.push(block.forward(&x, self.output_kernel_size)?);
        }
        let _ = seq;
        Ok(outputs)
    }
}

#[derive(Debug)]
struct Pi3xCoreRopeAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    rope: Rope2d,
}

impl Pi3xCoreRopeAttention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            q_norm: layer_norm(head_dim, 1e-5, vb.pp("q_norm"))?,
            k_norm: layer_norm(head_dim, 1e-5, vb.pp("k_norm"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            rope,
        })
    }

    fn forward(&self, xs: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .transpose(1, 3)?;
        let q = self.q_norm.forward(&qkv.i((.., .., 0))?)?.contiguous()?;
        let k = self.k_norm.forward(&qkv.i((.., .., 1))?)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.contiguous()?;
        let cache = self.rope.embeddings(positions, self.head_dim)?;
        let q = self.rope.apply_with_embeddings(&q, &cache)?.contiguous()?;
        let k = self.rope.apply_with_embeddings(&k, &cache)?.contiguous()?;
        let out = exact_query_chunked_sdpa(&q, &k, &v, self.scale, 128)?
            .transpose(1, 2)?
            .reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct Pi3xCoreRopeBlock {
    norm1: LayerNorm,
    attn: Pi3xCoreRopeAttention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Pi3xCoreRopeBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize, rope: Rope2d) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: Pi3xCoreRopeAttention::load(vb.pp("attn"), dim, num_heads, rope)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }

    fn forward(&self, xs: &Tensor, positions: &Tensor) -> CandleResult<Tensor> {
        let xs = (xs
            + self
                .ls1
                .forward(&self.attn.forward(&self.norm1.forward(xs)?, positions)?)?)?;
        let residual = &xs;
        self.ls2
            .forward(&self.mlp.forward(&self.norm2.forward(&xs)?)?)?
            + residual
    }
}

#[derive(Debug)]
struct Pi3xProjectivePoseAttention {
    qkv: Linear,
    proj: Linear,
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
}

impl Pi3xProjectivePoseAttention {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> CandleResult<Self> {
        let head_dim = dim / num_heads;
        Ok(Self {
            qkv: linear(vb.pp("qkv"), dim, dim * 3, true)?,
            proj: linear(vb.pp("proj"), dim, dim, true)?,
            q_norm: layer_norm(head_dim, 1e-5, vb.pp("q_norm"))?,
            k_norm: layer_norm(head_dim, 1e-5, vb.pp("k_norm"))?,
            num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        extrinsics: &Tensor,
        patch_h: usize,
        patch_w: usize,
    ) -> CandleResult<Tensor> {
        let (b, n, c) = xs.dims3()?;
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, n, 3, self.num_heads, self.head_dim))?
            .transpose(1, 3)?;
        let q = self.q_norm.forward(&qkv.i((.., .., 0))?)?.contiguous()?;
        let k = self.k_norm.forward(&qkv.i((.., .., 1))?)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.contiguous()?;
        let q = apply_projective_q(&q, extrinsics, patch_h, patch_w)?;
        let k = apply_projective_kv(&k, extrinsics, patch_h, patch_w)?;
        let v = apply_projective_kv(&v, extrinsics, patch_h, patch_w)?;
        let out = exact_query_chunked_sdpa(&q, &k, &v, self.scale, 128)?;
        let out = apply_projective_o(&out, extrinsics, patch_h, patch_w)?;
        let out = out.transpose(1, 2)?.reshape((b, n, c))?;
        self.proj.forward(&out)
    }
}

#[derive(Debug)]
struct Pi3xPoseInjectBlock {
    norm1: LayerNorm,
    attn: Pi3xProjectivePoseAttention,
    ls1: LayerScale,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: LayerScale,
}

impl Pi3xPoseInjectBlock {
    fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> CandleResult<Self> {
        Ok(Self {
            norm1: layer_norm(dim, 1e-6, vb.pp("norm1"))?,
            attn: Pi3xProjectivePoseAttention::load(vb.pp("attn"), dim, num_heads)?,
            ls1: LayerScale::new(vb.pp("ls1"), dim)?,
            norm2: layer_norm(dim, 1e-6, vb.pp("norm2"))?,
            mlp: Mlp::new(vb.pp("mlp"), dim)?,
            ls2: LayerScale::new(vb.pp("ls2"), dim)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        poses: &Tensor,
        patch_h: usize,
        patch_w: usize,
    ) -> CandleResult<Tensor> {
        let extrinsics = invert_se3_tensor(poses)?;
        let attn = self.ls1.forward(&self.attn.forward(
            &self.norm1.forward(xs)?,
            &extrinsics,
            patch_h,
            patch_w,
        )?)?;
        let ffn = self
            .ls2
            .forward(&self.mlp.forward(&self.norm2.forward(xs)?)?)?;
        attn + ffn
    }
}

#[derive(Debug)]
struct Pi3xCoreDecoder {
    register_token: Tensor,
    blocks: Vec<Pi3xCoreRopeBlock>,
    pose_inject_blocks: Vec<Pi3xPoseInjectBlock>,
    patch_start_idx: usize,
    patch_size: usize,
}

impl Pi3xCoreDecoder {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let rope = Rope2d::new(100.0);
        let blocks = (0..36)
            .map(|idx| {
                Pi3xCoreRopeBlock::load(vb.pp("decoder").pp(idx.to_string()), 1024, 16, rope)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        let pose_inject_blocks = (0..5)
            .map(|idx| {
                Pi3xPoseInjectBlock::load(vb.pp("pose_inject_blk").pp(idx.to_string()), 1024, 16)
            })
            .collect::<CandleResult<Vec<_>>>()?;
        Ok(Self {
            register_token: vb.get((1, 1, 5, 1024), "register_token")?,
            blocks,
            pose_inject_blocks,
            patch_start_idx: 5,
            patch_size: 14,
        })
    }

    fn decode(
        &self,
        hidden: &Tensor,
        num_views: usize,
        height: usize,
        width: usize,
        poses: Option<&Tensor>,
        use_pose_mask: Option<&Tensor>,
    ) -> CandleResult<(Tensor, Tensor)> {
        let (b, n, hw, dim) = hidden.dims4()?;
        debug_assert_eq!(n, num_views);
        let hidden = hidden.reshape((b * n, hw, dim))?;
        let register = self
            .register_token
            .broadcast_as((b, n, self.patch_start_idx, dim))?
            .reshape((b * n, self.patch_start_idx, dim))?;
        let mut hidden = Tensor::cat(&[&register, &hidden], 1)?;
        let hw_with_special = hidden.dim(1)?;

        let pos_patch = position_getter(
            b * n,
            height / self.patch_size,
            width / self.patch_size,
            hidden.device(),
        )?;
        let pos_special = Tensor::zeros(
            (b * n, self.patch_start_idx, 2),
            candle_core::DType::I64,
            hidden.device(),
        )?;
        let pos_even = Tensor::cat(&[&pos_special, &pos_patch], 1)?;
        let pos_odd = pos_even.reshape((b, n * hw_with_special, 2))?;

        let mut final_features: Option<Tensor> = None;
        let mut pose_block_idx = 0usize;
        let pose_enabled = poses.is_some()
            && use_pose_mask
                .map(|mask| {
                    mask.sum_all()
                        .and_then(|tensor| tensor.to_scalar::<u8>())
                        .map(|value| value != 0)
                })
                .transpose()?
                .unwrap_or(false);

        for (idx, block) in self.blocks.iter().enumerate() {
            if idx % 2 == 0 {
                hidden = hidden.reshape((b * n, hw_with_special, dim))?;
                hidden = block.forward(&hidden, &pos_even)?;
            } else {
                hidden = hidden.reshape((b, n * hw_with_special, dim))?;
                hidden = block.forward(&hidden, &pos_odd)?;
            }

            if pose_enabled && matches!(idx, 1 | 9 | 17 | 25 | 33) {
                let (poses, use_pose_mask) = match (poses.as_ref(), use_pose_mask.as_ref()) {
                    (Some(poses), Some(mask)) => (poses, mask),
                    _ => {
                        candle_core::bail!(
                            "Pi3X pose injection requires both pose tensors and pose mask"
                        )
                    }
                };
                hidden = hidden.reshape((b, n, hw_with_special, dim))?;
                let patch_hidden = hidden.i((.., .., self.patch_start_idx.., ..))?.reshape((
                    b,
                    n * (hw_with_special - self.patch_start_idx),
                    dim,
                ))?;
                let pose_delta = self.pose_inject_blocks[pose_block_idx]
                    .forward(
                        &patch_hidden,
                        poses,
                        height / self.patch_size,
                        width / self.patch_size,
                    )?
                    .reshape((b, n, hw_with_special - self.patch_start_idx, dim))?;
                let mask = use_pose_mask.to_dtype(DType::F32)?.reshape((b, n, 1, 1))?;
                let updated = hidden
                    .i((.., .., self.patch_start_idx.., ..))?
                    .broadcast_add(&pose_delta.broadcast_mul(&mask)?)?;
                let hidden_prefix = hidden.i((.., .., ..self.patch_start_idx, ..))?;
                hidden = Tensor::cat(&[&hidden_prefix, &updated], 2)?;
                hidden = hidden.reshape((b, n * hw_with_special, dim))?;
                pose_block_idx += 1;
            }

            if idx + 1 == self.blocks.len() - 1 {
                final_features = Some(hidden.reshape((b * n, hw_with_special, dim))?);
            }
        }

        let final_features = final_features.ok_or_else(|| {
            candle_core::Error::msg("Pi3X decoder did not capture second-to-last features")
        })?;
        let hidden = hidden.reshape((b * n, hw_with_special, dim))?;
        Ok((
            Tensor::cat(&[&final_features, &hidden], D::Minus1)?,
            pos_even,
        ))
    }
}

pub(crate) fn prepare_pi3x_inputs_with_stage(
    stage: &Pi3xPreprocessStage,
    source: &Path,
    conditions_path: Option<&Path>,
    interval: Option<usize>,
    device: &Device,
) -> Result<Pi3xPreparedInputs> {
    let interval = interval.unwrap_or_else(|| default_interval_for(source));
    if interval == 0 {
        return Err(Lux3dError::InvalidInput("interval must be at least 1"));
    }

    let sampled = load_sampled_pi3x_frames(source, interval)?;
    let sampled_paths = &sampled.paths;

    let first = image::open(&sampled_paths[0])
        .map_err(|_| Lux3dError::InvalidInput("failed to open first Pi3x image"))?
        .to_rgb8();
    let original_size = SpatialSize::new(first.width(), first.height());
    let target_size = stage.target_size_for(original_size, 255_000)?;

    #[cfg(feature = "vision-preproc")]
    let (rgb_frames, normalized_frames) = {
        let mut rgb_frames = Vec::with_capacity(sampled_paths.len());
        let mut normalized_frames = Vec::with_capacity(sampled_paths.len());
        for path in sampled_paths {
            let image = image::open(path)
                .map_err(|_| Lux3dError::InvalidInput("failed to open Pi3x input image"))?
                .to_rgb8();
            let resized = image::imageops::resize(
                &image,
                target_size.width,
                target_size.height,
                FilterType::Lanczos3,
            );
            let dynamic = image::DynamicImage::ImageRgb8(resized);
            let rgb_chw =
                vision_preproc::tensorize_rgb_image(&dynamic, device).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3x tensorization failed: {source}"),
                    }
                })?;
            let rgb_chw = vision_preproc::resize_chw_image(
                &rgb_chw,
                target_size.height as usize,
                target_size.width as usize,
            )
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Pi3x image resize failed: {source}"),
            })?;
            let normalized =
                vision_preproc::normalize_imagenet_chw(&rgb_chw).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Pi3x image normalization failed: {source}"),
                    }
                })?;
            rgb_frames.push(rgb_chw);
            normalized_frames.push(normalized);
        }
        let rgb_refs = rgb_frames.iter().collect::<Vec<_>>();
        let normalized_refs = normalized_frames.iter().collect::<Vec<_>>();
        (
            Tensor::stack(&rgb_refs, 0).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to stack Pi3x RGB frames: {source}"),
                }
            })?,
            Tensor::stack(&normalized_refs, 0)
                .and_then(|tensor| tensor.unsqueeze(0))
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to stack Pi3x normalized frames: {source}"),
                })?,
        )
    };

    #[cfg(not(feature = "vision-preproc"))]
    let (rgb_frames, normalized_frames) = {
        let mut rgb_data = Vec::new();
        let mut normalized_data = Vec::new();
        let frame_count = sampled_paths.len();
        let target_h = target_size.height as usize;
        let target_w = target_size.width as usize;

        const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

        for path in sampled_paths {
            let image = image::open(path)
                .map_err(|_| Lux3dError::InvalidInput("failed to open Pi3x input image"))?
                .to_rgb8();
            let resized = image::imageops::resize(
                &image,
                target_size.width,
                target_size.height,
                FilterType::Lanczos3,
            );
            let mut frame_rgb = vec![0f32; 3 * target_h * target_w];
            for channel in 0..3 {
                for y in 0..target_h {
                    for x in 0..target_w {
                        let value =
                            f32::from(resized.get_pixel(x as u32, y as u32)[channel]) / 255.0;
                        frame_rgb[(channel * target_h + y) * target_w + x] = value;
                    }
                }
            }
            rgb_data.extend_from_slice(&frame_rgb);
            for channel in 0..3 {
                for y in 0..target_h {
                    for x in 0..target_w {
                        let value = frame_rgb[(channel * target_h + y) * target_w + x];
                        normalized_data
                            .push((value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel]);
                    }
                }
            }
        }

        let rgb_tensor = Tensor::from_vec(rgb_data, (frame_count, 3, target_h, target_w), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3x RGB tensor: {source}"),
        })?;
        let normalized_tensor = Tensor::from_vec(
            normalized_data,
            (1, frame_count, 3, target_h, target_w),
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3x normalized tensor: {source}"),
        })?;
        (rgb_tensor, normalized_tensor)
    };

    let conditions = if let Some(path) = conditions_path {
        Some(load_pi3x_conditions(
            path,
            original_size,
            target_size,
            interval,
            sampled_paths.len(),
        )?)
    } else {
        None
    };

    let mask_add_depth = build_condition_mask(
        conditions
            .as_ref()
            .and_then(|loaded| loaded.depths.as_ref())
            .is_some(),
        sampled_paths.len(),
        device,
    )?;
    let mask_add_ray = build_condition_mask(
        conditions
            .as_ref()
            .and_then(|loaded| loaded.rays.as_ref().or(loaded.intrinsics.as_ref()))
            .is_some(),
        sampled_paths.len(),
        device,
    )?;
    let mask_add_pose = build_condition_mask(
        conditions
            .as_ref()
            .and_then(|loaded| loaded.poses.as_ref())
            .is_some(),
        sampled_paths.len(),
        device,
    )?;

    Ok(Pi3xPreparedInputs {
        interval,
        original_size,
        target_size,
        rgb_frames,
        normalized_frames,
        depths: conditions.as_ref().and_then(|loaded| loaded.depths.clone()),
        intrinsics: conditions
            .as_ref()
            .and_then(|loaded| loaded.intrinsics.clone()),
        poses: conditions.as_ref().and_then(|loaded| loaded.poses.clone()),
        rays: conditions.as_ref().and_then(|loaded| loaded.rays.clone()),
        mask_add_depth,
        mask_add_ray,
        mask_add_pose,
    })
}

#[derive(Debug, Clone)]
struct LoadedPi3xConditions {
    poses: Option<Tensor>,
    depths: Option<Tensor>,
    intrinsics: Option<Tensor>,
    rays: Option<Tensor>,
}

fn load_pi3x_conditions(
    path: &Path,
    original_size: SpatialSize,
    target_size: SpatialSize,
    interval: usize,
    sampled_frames: usize,
) -> Result<LoadedPi3xConditions> {
    let file = fs::File::open(path).map_err(|source| Lux3dError::CanonicalManifestIo {
        path: path.to_path_buf(),
        source,
    })?;
    let mut npz = NpzReader::new(file)
        .map_err(|_| Lux3dError::InvalidInput("failed to open Pi3x conditions npz"))?;

    let poses = load_npz_tensor_f32_3(&mut npz, "poses", interval, sampled_frames)?;
    let depths = load_npz_depths(&mut npz, "depths", target_size, interval, sampled_frames)?;
    let intrinsics = load_npz_intrinsics(
        &mut npz,
        "intrinsics",
        original_size,
        target_size,
        interval,
        sampled_frames,
    )?;
    let rays = if let Some(intrinsics) = intrinsics.as_ref() {
        Some(build_rays_from_intrinsics(intrinsics, target_size)?)
    } else {
        None
    };

    Ok(LoadedPi3xConditions {
        poses,
        depths,
        intrinsics,
        rays,
    })
}

fn load_npz_tensor_f32_3(
    npz: &mut NpzReader<fs::File>,
    key: &str,
    interval: usize,
    sampled_frames: usize,
) -> Result<Option<Tensor>> {
    let names = [format!("{key}.npy"), key.to_string()];
    let mut loaded = None;
    for name in names {
        if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>(&name) {
            loaded = Some(array);
            break;
        }
    }
    let Some(array) = loaded else {
        return Ok(None);
    };
    let sliced = array
        .slice_axis(
            Axis(0),
            ndarray::Slice::new(0, None, interval_to_isize(interval)),
        )
        .to_owned();
    let frames = sliced.len_of(Axis(0)).min(sampled_frames);
    let sliced = sliced
        .slice_axis(Axis(0), ndarray::Slice::new(0, Some(frames as isize), 1))
        .to_owned();
    let dims = sliced.raw_dim();
    let (data, _offset) = sliced.into_raw_vec_and_offset();
    Tensor::from_vec(data, (1, dims[0], dims[1], dims[2]), &Device::Cpu)
        .map(Some)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize `{key}` condition tensor: {source}"),
        })
}

fn load_npz_depths(
    npz: &mut NpzReader<fs::File>,
    key: &str,
    target_size: SpatialSize,
    interval: usize,
    sampled_frames: usize,
) -> Result<Option<Tensor>> {
    let names = [format!("{key}.npy"), key.to_string()];
    let mut loaded: Option<Array3<f32>> = None;
    for name in names {
        if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f32>, Ix3>(&name) {
            loaded = Some(array);
            break;
        }
    }
    let Some(array) = loaded else {
        return Ok(None);
    };
    let indices = (0..array.len_of(Axis(0)))
        .step_by(interval)
        .take(sampled_frames)
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(
        indices.len() * target_size.height as usize * target_size.width as usize,
    );
    for &idx in &indices {
        let frame = array.index_axis(Axis(0), idx);
        let resized = resize_depth_nearest(
            frame
                .as_slice()
                .ok_or(Lux3dError::InvalidInput("depth array is not contiguous"))?,
            frame.shape()[0],
            frame.shape()[1],
            target_size.height as usize,
            target_size.width as usize,
        );
        data.extend(resized.into_iter().map(|value| {
            if value.is_finite() && value > 0.0 {
                value
            } else {
                0.0
            }
        }));
    }
    Tensor::from_vec(
        data,
        (
            1,
            indices.len(),
            target_size.height as usize,
            target_size.width as usize,
        ),
        &Device::Cpu,
    )
    .map(Some)
    .map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to materialize `{key}` depth tensor: {source}"),
    })
}

fn load_npz_intrinsics(
    npz: &mut NpzReader<fs::File>,
    key: &str,
    original_size: SpatialSize,
    target_size: SpatialSize,
    interval: usize,
    sampled_frames: usize,
) -> Result<Option<Tensor>> {
    let names = [format!("{key}.npy"), key.to_string()];
    let mut loaded = None;
    for name in names {
        if let Ok(array) = npz.by_name::<ndarray::OwnedRepr<f32>, ndarray::Ix3>(&name) {
            loaded = Some(array);
            break;
        }
    }
    let Some(array) = loaded else {
        return Ok(None);
    };
    let indices = (0..array.len_of(Axis(0)))
        .step_by(interval)
        .take(sampled_frames)
        .collect::<Vec<_>>();
    let scale_x = target_size.width as f32 / original_size.width as f32;
    let scale_y = target_size.height as f32 / original_size.height as f32;
    let mut data = Vec::with_capacity(indices.len() * 9);
    for &idx in &indices {
        let frame = array.index_axis(Axis(0), idx);
        let (mut values, _offset) = frame.to_owned().into_raw_vec_and_offset();
        values[0] *= scale_x;
        values[2] *= scale_x;
        values[4] *= scale_y;
        values[5] *= scale_y;
        data.extend(values);
    }
    Tensor::from_vec(data, (1, indices.len(), 3, 3), &Device::Cpu)
        .map(Some)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize `{key}` intrinsics tensor: {source}"),
        })
}

fn build_rays_from_intrinsics(intrinsics: &Tensor, target_size: SpatialSize) -> Result<Tensor> {
    let device = intrinsics.device();
    let dims = intrinsics.dims();
    let frames = dims[1];
    let h = target_size.height as usize;
    let w = target_size.width as usize;
    let x = Tensor::arange(0f32, w as f32, device)
        .and_then(|tensor| tensor.affine(1.0, 0.5))
        .and_then(|tensor| tensor.reshape((1, 1, 1, w)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X ray x-grid: {source}"),
        })?;
    let y = Tensor::arange(0f32, h as f32, device)
        .and_then(|tensor| tensor.affine(1.0, 0.5))
        .and_then(|tensor| tensor.reshape((1, 1, h, 1)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X ray y-grid: {source}"),
        })?;
    let fx = intrinsics
        .i((.., .., 0, 0))
        .and_then(|tensor| tensor.reshape((1, frames, 1, 1)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X fx intrinsics: {source}"),
        })?;
    let fy = intrinsics
        .i((.., .., 1, 1))
        .and_then(|tensor| tensor.reshape((1, frames, 1, 1)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X fy intrinsics: {source}"),
        })?;
    let cx = intrinsics
        .i((.., .., 0, 2))
        .and_then(|tensor| tensor.reshape((1, frames, 1, 1)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X cx intrinsics: {source}"),
        })?;
    let cy = intrinsics
        .i((.., .., 1, 2))
        .and_then(|tensor| tensor.reshape((1, frames, 1, 1)))
        .and_then(|tensor| tensor.broadcast_as((1, frames, h, w)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X cy intrinsics: {source}"),
        })?;
    let ray_x = x
        .broadcast_sub(&cx)
        .and_then(|tensor| tensor.broadcast_div(&fx))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X x-ray directions: {source}"),
        })?;
    let ray_y = y
        .broadcast_sub(&cy)
        .and_then(|tensor| tensor.broadcast_div(&fy))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X y-ray directions: {source}"),
        })?;
    let ray_x =
        ray_x
            .unsqueeze(D::Minus1)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to expand Pi3X x-ray directions: {source}"),
            })?;
    let ray_y =
        ray_y
            .unsqueeze(D::Minus1)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to expand Pi3X y-ray directions: {source}"),
            })?;
    Tensor::cat(&[&ray_x, &ray_y], D::Minus1).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize Pi3x rays: {source}"),
        }
    })
}

fn normalized_view_plane_uv(
    width: usize,
    height: usize,
    aspect_ratio: f32,
    device: &Device,
) -> CandleResult<Tensor> {
    let span_x = aspect_ratio / (1.0 + aspect_ratio * aspect_ratio).sqrt();
    let span_y = 1.0 / (1.0 + aspect_ratio * aspect_ratio).sqrt();
    let mut values = Vec::with_capacity(width * height * 2);
    for y in 0..height {
        let v = if height > 1 {
            let t = y as f32 / (height as f32);
            (-span_y * (height - 1) as f32 / height as f32)
                + (2.0 * span_y * (height - 1) as f32 / height as f32) * t
        } else {
            0.0
        };
        for x in 0..width {
            let u = if width > 1 {
                let t = x as f32 / (width as f32);
                (-span_x * (width - 1) as f32 / width as f32)
                    + (2.0 * span_x * (width - 1) as f32 / width as f32) * t
            } else {
                0.0
            };
            values.push(u);
            values.push(v);
        }
    }
    Tensor::from_vec(values, (1, height, width, 2), device)?.permute((0, 3, 1, 2))
}

fn normalize_depth_mean(depths: &Tensor, device: &Device) -> Result<(Tensor, Tensor)> {
    let depths =
        depths
            .to_device(device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to move Pi3X depths to device: {source}"),
            })?;
    let batch = depths.dim(0).unwrap_or(1);
    let positive_mask = depths
        .gt(0.0)
        .and_then(|tensor| tensor.to_dtype(DType::F32))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X positive-depth mask: {source}"),
        })?;
    let depth_sum = depths
        .broadcast_mul(&positive_mask)
        .and_then(|tensor| tensor.sum_keepdim((1, 2, 3)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to reduce Pi3X depth sum: {source}"),
        })?;
    let count = positive_mask.sum_keepdim((1, 2, 3)).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to reduce Pi3X depth count: {source}"),
        }
    })?;
    let factor = depth_sum
        .broadcast_div(&count.clamp(1.0, f64::INFINITY).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to clamp Pi3X depth count: {source}"),
            }
        })?)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X depth scale: {source}"),
        })?;
    let valid = count
        .gt(0.0)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X depth-valid mask: {source}"),
        })?;
    let fallback = Tensor::ones(factor.shape(), factor.dtype(), device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X depth-scale fallback: {source}"),
        }
    })?;
    let factor = valid.where_cond(&factor, &fallback).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to finalize Pi3X depth scale: {source}"),
        }
    })?;
    let normalized =
        depths
            .broadcast_div(&factor)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to normalize Pi3X depths: {source}"),
            })?;
    let factors =
        factor
            .reshape((batch,))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3X depth factors: {source}"),
            })?;
    Ok((normalized, factors))
}

fn run_pi3x_conv_head_chunked(
    head: &Pi3xConvHead,
    feat: &Tensor,
    patch_h: usize,
    patch_w: usize,
    img_h: usize,
    img_w: usize,
    chunk_size: usize,
) -> CandleResult<Vec<Tensor>> {
    let batch = feat.dim(0)?;
    if batch <= chunk_size {
        return head.forward(feat, patch_h, patch_w, img_h, img_w);
    }

    let mut merged: Option<Vec<Tensor>> = None;
    let mut start = 0usize;
    while start < batch {
        let len = (batch - start).min(chunk_size);
        let chunk = feat.narrow(0, start, len)?;
        let outputs = head.forward(&chunk, patch_h, patch_w, img_h, img_w)?;
        if let Some(existing) = merged.as_mut() {
            for (idx, output) in outputs.into_iter().enumerate() {
                existing[idx] = Tensor::cat(&[&existing[idx], &output], 0)?;
            }
        } else {
            merged = Some(outputs);
        }
        start += len;
    }
    merged.ok_or_else(|| candle_core::Error::msg("Pi3X conv head chunking received empty batch"))
}

fn invert_se3_tensor(poses: &Tensor) -> CandleResult<Tensor> {
    let dims = poses.dims();
    let batch = dims[0];
    let cameras = dims[1];
    let rotation = poses.i((.., .., ..3, ..3))?;
    let rotation_t = rotation.transpose(2, 3)?.contiguous()?;
    let translation = poses.i((.., .., ..3, 3..4))?.contiguous()?;
    let inv_translation = rotation_t.matmul(&translation)?.neg()?;
    let top = Tensor::cat(&[&rotation_t, &inv_translation], D::Minus1)?;
    let bottom = Tensor::from_vec(vec![0f32, 0.0, 0.0, 1.0], (1, 1, 1, 4), poses.device())?
        .broadcast_as((batch, cameras, 1, 4))?;
    Tensor::cat(&[&top, &bottom], 2)
}

fn relative_poses(poses: &Tensor, device: &Device) -> Result<Tensor> {
    let poses =
        poses
            .to_device(device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to move Pi3X poses to device: {source}"),
            })?;
    let pose0 =
        poses
            .i((.., 0..1, .., ..))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X first pose: {source}"),
            })?;
    let pose0_inv =
        invert_se3_tensor(&pose0).map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to invert Pi3X first pose: {source}"),
        })?;
    let batch = poses.dim(0).unwrap_or(1);
    let frames = poses.dim(1).unwrap_or(1);
    pose0_inv
        .broadcast_as((batch, frames, 4, 4))
        .and_then(|reference| reference.matmul(&poses))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X relative poses: {source}"),
        })
}

fn divide_pose_translation(poses: &Tensor, scale: &Tensor) -> Result<Tensor> {
    let rotation =
        poses
            .i((.., .., ..3, ..3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X pose rotation: {source}"),
            })?;
    let translation =
        poses
            .i((.., .., ..3, 3..4))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X pose translation: {source}"),
            })?;
    let scale = scale
        .reshape((scale.dim(0).unwrap_or(1), 1, 1, 1))
        .and_then(|tensor| tensor.broadcast_as(translation.shape().dims()))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X pose scale: {source}"),
        })?;
    let translation = translation.broadcast_div(&scale).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to divide Pi3X pose translation by scale: {source}"),
        }
    })?;
    let top = Tensor::cat(&[&rotation, &translation], D::Minus1).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to rebuild Pi3X pose top rows: {source}"),
        }
    })?;
    let bottom =
        poses
            .i((.., .., 3..4, ..))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X pose bottom row: {source}"),
            })?;
    Tensor::cat(&[&top, &bottom], 2).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to rebuild scaled Pi3X poses: {source}"),
    })
}

fn scale_pose_translation(poses: &Tensor, metric: &Tensor) -> Result<Tensor> {
    let rotation =
        poses
            .i((.., .., ..3, ..3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X camera rotation: {source}"),
            })?;
    let translation =
        poses
            .i((.., .., ..3, 3..4))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X camera translation: {source}"),
            })?;
    let metric = metric
        .reshape((metric.dim(0).unwrap_or(1), 1, 1, 1))
        .and_then(|tensor| tensor.broadcast_as(translation.shape().dims()))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X metric: {source}"),
        })?;
    let translation = translation.broadcast_mul(&metric).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to scale Pi3X camera translation: {source}"),
        }
    })?;
    let top = Tensor::cat(&[&rotation, &translation], D::Minus1).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to rebuild Pi3X camera top rows: {source}"),
        }
    })?;
    let bottom =
        poses
            .i((.., .., 3..4, ..))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X camera bottom row: {source}"),
            })?;
    Tensor::cat(&[&top, &bottom], 2).map_err(|source| Lux3dError::CanonicalWeightsValidation {
        message: format!("failed to rebuild metric-scaled Pi3X camera poses: {source}"),
    })
}

fn normalize_last_dim(xs: &Tensor) -> CandleResult<Tensor> {
    let squared = xs.sqr()?;
    let denom = squared.sum_keepdim(D::Minus1)?.sqrt()?;
    xs.broadcast_div(&denom)
}

fn rope_coeffs(
    positions: &[usize],
    feat_dim: usize,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let num_freqs = feat_dim / 2;
    let mut cos = Vec::with_capacity(positions.len() * num_freqs);
    let mut sin = Vec::with_capacity(positions.len() * num_freqs);
    for &pos in positions {
        for idx in 0..num_freqs {
            let freq = 100.0f32.powf(-(idx as f32) / num_freqs as f32);
            let angle = pos as f32 * freq;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    Ok((
        Tensor::from_vec(cos, (1, 1, positions.len(), num_freqs), device)?,
        Tensor::from_vec(sin, (1, 1, positions.len(), num_freqs), device)?,
    ))
}

fn rope_apply_coeffs(
    feats: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    inverse: bool,
) -> CandleResult<Tensor> {
    let half = feats.dim(D::Minus1)? / 2;
    let x_in = feats.i((.., .., .., ..half))?;
    let y_in = feats.i((.., .., .., half..))?;
    if inverse {
        Tensor::cat(
            &[
                &(x_in.broadcast_mul(cos)? - y_in.broadcast_mul(sin)?)?,
                &(x_in.broadcast_mul(sin)? + y_in.broadcast_mul(cos)?)?,
            ],
            D::Minus1,
        )
    } else {
        Tensor::cat(
            &[
                &(x_in.broadcast_mul(cos)? + y_in.broadcast_mul(sin)?)?,
                &(y_in.broadcast_mul(cos)? - x_in.broadcast_mul(sin)?)?,
            ],
            D::Minus1,
        )
    }
}

fn apply_projective_matrix(feats: &Tensor, matrix: &Tensor) -> CandleResult<Tensor> {
    let (batch, heads, seq_len, feat_dim) = feats.dims4()?;
    let cameras = matrix.dim(1)?;
    let tokens_per_view = seq_len / cameras;
    let groups = feat_dim / 4;
    let mut per_batch = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut per_camera = Vec::with_capacity(cameras);
        for camera in 0..cameras {
            let chunk = feats
                .i((
                    b,
                    ..,
                    camera * tokens_per_view..(camera + 1) * tokens_per_view,
                    ..,
                ))?
                .reshape((heads, tokens_per_view, groups, 4))?;
            let mat_t = matrix.i((b, camera))?.transpose(0, 1)?.contiguous()?;
            let out = chunk
                .reshape((heads * tokens_per_view * groups, 4))?
                .matmul(&mat_t)?
                .reshape((heads, tokens_per_view, feat_dim))?;
            per_camera.push(out);
        }
        let refs = per_camera.iter().collect::<Vec<_>>();
        per_batch.push(Tensor::cat(&refs, 1)?);
    }
    let refs = per_batch.iter().collect::<Vec<_>>();
    Tensor::stack(&refs, 0)
}

fn apply_projective_q(
    feats: &Tensor,
    extrinsics: &Tensor,
    patch_h: usize,
    patch_w: usize,
) -> CandleResult<Tensor> {
    let half = feats.dim(D::Minus1)? / 2;
    let quarter = feats.dim(D::Minus1)? / 4;
    let matrices = extrinsics.transpose(2, 3)?;
    let projective = apply_projective_matrix(&feats.i((.., .., .., ..half))?, &matrices)?;
    let cameras = extrinsics.dim(1)?;
    let seq_len = cameras * patch_h * patch_w;
    let x_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|_| 0..patch_w))
        .collect::<Vec<_>>();
    let y_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|y| std::iter::repeat_n(y, patch_w)))
        .collect::<Vec<_>>();
    debug_assert_eq!(x_positions.len(), seq_len);
    let (cos_x, sin_x) = rope_coeffs(&x_positions, quarter, feats.device())?;
    let (cos_y, sin_y) = rope_coeffs(&y_positions, quarter, feats.device())?;
    let rope_x = rope_apply_coeffs(
        &feats.i((.., .., .., half..half + quarter))?,
        &cos_x,
        &sin_x,
        false,
    )?;
    let rope_y = rope_apply_coeffs(
        &feats.i((.., .., .., half + quarter..))?,
        &cos_y,
        &sin_y,
        false,
    )?;
    Tensor::cat(&[&projective, &rope_x, &rope_y], D::Minus1)
}

fn apply_projective_kv(
    feats: &Tensor,
    extrinsics: &Tensor,
    patch_h: usize,
    patch_w: usize,
) -> CandleResult<Tensor> {
    let poses = invert_se3_tensor(extrinsics)?;
    let half = feats.dim(D::Minus1)? / 2;
    let quarter = feats.dim(D::Minus1)? / 4;
    let projective = apply_projective_matrix(&feats.i((.., .., .., ..half))?, &poses)?;
    let cameras = extrinsics.dim(1)?;
    let seq_len = cameras * patch_h * patch_w;
    let x_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|_| 0..patch_w))
        .collect::<Vec<_>>();
    let y_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|y| std::iter::repeat_n(y, patch_w)))
        .collect::<Vec<_>>();
    debug_assert_eq!(x_positions.len(), seq_len);
    let (cos_x, sin_x) = rope_coeffs(&x_positions, quarter, feats.device())?;
    let (cos_y, sin_y) = rope_coeffs(&y_positions, quarter, feats.device())?;
    let rope_x = rope_apply_coeffs(
        &feats.i((.., .., .., half..half + quarter))?,
        &cos_x,
        &sin_x,
        false,
    )?;
    let rope_y = rope_apply_coeffs(
        &feats.i((.., .., .., half + quarter..))?,
        &cos_y,
        &sin_y,
        false,
    )?;
    Tensor::cat(&[&projective, &rope_x, &rope_y], D::Minus1)
}

fn apply_projective_o(
    feats: &Tensor,
    extrinsics: &Tensor,
    patch_h: usize,
    patch_w: usize,
) -> CandleResult<Tensor> {
    let half = feats.dim(D::Minus1)? / 2;
    let quarter = feats.dim(D::Minus1)? / 4;
    let projective = apply_projective_matrix(&feats.i((.., .., .., ..half))?, extrinsics)?;
    let cameras = extrinsics.dim(1)?;
    let seq_len = cameras * patch_h * patch_w;
    let x_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|_| 0..patch_w))
        .collect::<Vec<_>>();
    let y_positions = (0..cameras)
        .flat_map(|_| (0..patch_h).flat_map(|y| std::iter::repeat_n(y, patch_w)))
        .collect::<Vec<_>>();
    debug_assert_eq!(x_positions.len(), seq_len);
    let (cos_x, sin_x) = rope_coeffs(&x_positions, quarter, feats.device())?;
    let (cos_y, sin_y) = rope_coeffs(&y_positions, quarter, feats.device())?;
    let rope_x = rope_apply_coeffs(
        &feats.i((.., .., .., half..half + quarter))?,
        &cos_x,
        &sin_x,
        true,
    )?;
    let rope_y = rope_apply_coeffs(
        &feats.i((.., .., .., half + quarter..))?,
        &cos_y,
        &sin_y,
        true,
    )?;
    Tensor::cat(&[&projective, &rope_x, &rope_y], D::Minus1)
}

fn adaptive_confidence_mask(curr_conf: &Tensor) -> Result<Tensor> {
    curr_conf
        .gt(0.0)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X VO adaptive confidence mask: {source}"),
        })
}

fn compute_sim3_umeyama_masked(
    src_points: &Tensor,
    tgt_points: &Tensor,
    src_mask: &Tensor,
    tgt_mask: &Tensor,
) -> Result<Tensor> {
    let result_device = src_points.device().clone();
    let identity = || {
        Tensor::eye(4, DType::F32, &result_device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize Pi3X VO identity sim3: {source}"),
            })
    };
    let point_count = src_points.elem_count() / 3;
    let src_flat = src_points.reshape((point_count, 3)).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to flatten Pi3X VO source points: {source}"),
        }
    })?;
    let tgt_flat = tgt_points.reshape((point_count, 3)).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to flatten Pi3X VO target points: {source}"),
        }
    })?;
    let valid_mask = src_mask
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.broadcast_mul(&tgt_mask.to_dtype(DType::F32)?))
        .and_then(|tensor| tensor.reshape((point_count, 1)))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3X VO joint mask: {source}"),
        })?;
    let valid_count = valid_mask
        .sum_all()
        .and_then(|tensor| tensor.to_scalar::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to reduce Pi3X VO joint mask: {source}"),
        })?;
    if valid_count < 10.0 {
        return identity();
    }

    let src_mean = src_flat
        .broadcast_mul(&valid_mask)
        .and_then(|tensor| tensor.sum(0))
        .and_then(|tensor| tensor.affine(1.0 / valid_count as f64, 0.0))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X VO source mean: {source}"),
        })?;
    let tgt_mean = tgt_flat
        .broadcast_mul(&valid_mask)
        .and_then(|tensor| tensor.sum(0))
        .and_then(|tensor| tensor.affine(1.0 / valid_count as f64, 0.0))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X VO target mean: {source}"),
        })?;
    let src_centered = src_flat
        .broadcast_sub(&src_mean.reshape((1, 3)).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3X VO source mean: {source}"),
            }
        })?)
        .and_then(|tensor| tensor.broadcast_mul(&valid_mask))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to center Pi3X VO source points: {source}"),
        })?;
    let tgt_centered = tgt_flat
        .broadcast_sub(&tgt_mean.reshape((1, 3)).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to reshape Pi3X VO target mean: {source}"),
            }
        })?)
        .and_then(|tensor| tensor.broadcast_mul(&valid_mask))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to center Pi3X VO target points: {source}"),
        })?;
    let cov = src_centered
        .transpose(0, 1)
        .and_then(|tensor| tensor.matmul(&tgt_centered))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X VO covariance: {source}"),
        })?;
    let src_var = src_centered
        .sqr()
        .and_then(|tensor| tensor.sum_all())
        .and_then(|tensor| tensor.to_scalar::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to compute Pi3X VO source variance: {source}"),
        })?
        / valid_count.max(1.0);
    if !src_var.is_finite() || src_var <= 1e-8 {
        return identity();
    }
    let cov = cov
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to read Pi3X VO covariance: {source}"),
        })?;
    let src_mean = src_mean
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to read Pi3X VO source mean: {source}"),
        })?;
    let tgt_mean = tgt_mean
        .to_device(&Device::Cpu)
        .and_then(|tensor| tensor.to_vec1::<f32>())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to read Pi3X VO target mean: {source}"),
        })?;

    let n = valid_count;
    let src_mean = nalgebra::Vector3::new(src_mean[0], src_mean[1], src_mean[2]);
    let tgt_mean = nalgebra::Vector3::new(tgt_mean[0], tgt_mean[1], tgt_mean[2]);
    let cov = nalgebra::Matrix3::<f32>::from_row_slice(&cov);
    let svd = nalgebra::SVD::new(cov, true, true);
    let Some(u) = svd.u else {
        return identity();
    };
    let Some(v_t) = svd.v_t else {
        return identity();
    };
    let mut v = v_t.transpose();
    let det = (v * u.transpose()).determinant();
    if !det.is_finite() {
        return identity();
    }
    let det = det.signum();
    v[(0, 2)] *= det;
    v[(1, 2)] *= det;
    v[(2, 2)] *= det;
    let r = v * u.transpose();
    let mut corrected_s = svd.singular_values;
    corrected_s[2] *= det;
    let scale = corrected_s.sum() / (src_var * n + 1e-6);
    if !scale.is_finite() {
        return identity();
    }
    let t = tgt_mean - scale * r * src_mean;
    if !t.iter().all(|value| value.is_finite()) || !r.iter().all(|value| value.is_finite()) {
        return identity();
    }
    let sim3 = vec![
        scale * r[(0, 0)],
        scale * r[(0, 1)],
        scale * r[(0, 2)],
        t[0],
        scale * r[(1, 0)],
        scale * r[(1, 1)],
        scale * r[(1, 2)],
        t[1],
        scale * r[(2, 0)],
        scale * r[(2, 1)],
        scale * r[(2, 2)],
        t[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    Tensor::from_vec(sim3, (1, 4, 4), &result_device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize Pi3X VO sim3: {source}"),
        }
    })
}

fn apply_sim3_to_points(points: &Tensor, sim3: &Tensor) -> Result<Tensor> {
    let linear =
        sim3.i((0, ..3, ..3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X VO sim3 linear: {source}"),
            })?;
    let translation =
        sim3.i((0, ..3, 3))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to slice Pi3X VO sim3 translation: {source}"),
            })?;
    let dims = points.dims();
    let batch = dims[0];
    let frames = dims[1];
    let height = dims[2];
    let width = dims[3];
    let flat = points
        .reshape((batch * frames * height * width, 3))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to flatten Pi3X VO points: {source}"),
        })?;
    let transformed = flat
        .matmul(
            &linear
                .transpose(0, 1)
                .and_then(|tensor| tensor.contiguous())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to transpose Pi3X VO sim3 linear: {source}"),
                })?,
        )
        .and_then(|tensor| tensor.broadcast_add(&translation))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to transform Pi3X VO points: {source}"),
        })?;
    transformed
        .reshape((batch, frames, height, width, 3))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to reshape Pi3X VO transformed points: {source}"),
        })
}

fn apply_sim3_to_poses(poses: &Tensor, sim3: &Tensor) -> Result<Tensor> {
    let batch = poses.dim(0).unwrap_or(1);
    let frames = poses.dim(1).unwrap_or(1);
    let sim3 = sim3
        .broadcast_as((batch, frames, 4, 4))
        .and_then(|tensor| tensor.contiguous())
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X VO sim3 to poses: {source}"),
        })?;
    let poses = poses
        .contiguous()
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize contiguous Pi3X VO poses: {source}"),
        })?;
    sim3.matmul(&poses)
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to transform Pi3X VO poses: {source}"),
        })
}

fn build_condition_mask(has_condition: bool, frames: usize, device: &Device) -> Result<Tensor> {
    let value = if has_condition { 1u8 } else { 0u8 };
    Tensor::full(value, (1, frames), device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3x condition mask: {source}"),
        }
    })
}

fn build_prefix_mask(frames: usize, prefix_len: usize, device: &Device) -> Result<Tensor> {
    let mut values = vec![0u8; frames];
    for value in values.iter_mut().take(prefix_len.min(frames)) {
        *value = 1;
    }
    Tensor::from_vec(values, (1, frames), device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to build Pi3x prefix mask: {source}"),
        }
    })
}

fn repeated_identity_poses(frames: usize, device: &Device) -> Result<Tensor> {
    let identity = vec![
        1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let base = Tensor::from_vec(identity, (1, 1, 4, 4), device).map_err(|source| {
        Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize Pi3X identity pose: {source}"),
        }
    })?;
    base.broadcast_as((1, frames, 4, 4))
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to broadcast Pi3X identity poses: {source}"),
        })
}

fn interval_to_isize(interval: usize) -> isize {
    interval as isize
}

fn resize_depth_nearest(
    data: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; dst_h * dst_w];
    for y in 0..dst_h {
        let src_y = y * src_h / dst_h;
        for x in 0..dst_w {
            let src_x = x * src_w / dst_w;
            out[y * dst_w + x] = data[src_y * src_w + src_x];
        }
    }
    out
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
    fn new(prefix: &str) -> Result<Self> {
        let mut path = std::env::temp_dir();
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!("{prefix}-{}-{nonce}", std::process::id()));
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
struct SampledPi3xFrames {
    paths: Vec<PathBuf>,
    _temp_dir: Option<TempFrameDir>,
}

fn load_sampled_pi3x_frames(source: &Path, interval: usize) -> Result<SampledPi3xFrames> {
    if source.is_dir() {
        let entries = collect_rgb_entries(source)?;
        if entries.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "input directory did not contain any RGB frames",
            ));
        }
        return Ok(SampledPi3xFrames {
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
        "Pi3x input source must be a directory or .mp4 file",
    ))
}

fn collect_rgb_entries(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut entries = fs::read_dir(dir)
        .map_err(|source| Lux3dError::CanonicalManifestIo {
            path: dir.to_path_buf(),
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
    sort_paths_natural(&mut entries);
    Ok(entries)
}

fn extract_video_frames(source: &Path, interval: usize) -> Result<SampledPi3xFrames> {
    if !source.is_file() {
        return Err(Lux3dError::InvalidInput(
            "Pi3x video input file was not found",
        ));
    }

    let temp_dir = TempFrameDir::new("lux3d-pi3x-frames")?;
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
        .map_err(|_| Lux3dError::InvalidInput("failed to start ffmpeg for Pi3x video input"))?;

    if !output.status.success() {
        return Err(Lux3dError::InvalidInput(
            "failed to extract Pi3x video frames with ffmpeg",
        ));
    }

    let paths = collect_rgb_entries(&temp_dir.path)?;
    if paths.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "Pi3x video input did not yield any RGB frames",
        ));
    }

    Ok(SampledPi3xFrames {
        paths,
        _temp_dir: Some(temp_dir),
    })
}
