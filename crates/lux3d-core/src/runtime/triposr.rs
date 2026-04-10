use std::{path::Path, sync::Arc};

use candle_core::{DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::{
    ConvTranspose2d, ConvTranspose2dConfig, GroupNorm, LayerNorm, Linear, Module, VarBuilder,
    conv_transpose2d, group_norm, layer_norm, linear, linear_no_bias,
};
use image::{DynamicImage, ImageBuffer, ImageReader, Rgba, RgbaImage};

use crate::{
    CanonicalWeightSetPaths, ModelAssetOptions, ModelFamily, Result, contracts::SpatialSize,
    error::Lux3dError, export::TripoExportStage, geometry::TripoGeometryStage,
    load_canonical_weights, neural::TripoNeuralStage, preprocess::TripoPreprocessStage,
};

#[cfg(feature = "vision-preproc")]
use super::vision_preproc;
use super::{
    DeviceLocalCache,
    attention_math::exact_sdpa_heads,
    nn_blocks::FeedForward,
    resampling::{clamp_isize, compute_aa_linear_weights, cubic_weight},
    triposr_field::{
        TriplaneDecoder, query_triplane_chunked, query_triplane_color_chunked,
        query_triplane_density_chunked,
    },
};

#[derive(Debug)]
struct TripoSrModelBundle {
    image_tokenizer: TripoImageTokenizer,
    triplane_tokenizer: Triplane1DTokenizer,
    backbone: Transformer1D,
    post_processor: TriplaneUpsampleNetwork,
    decoder: NerfMlp,
}

impl TripoSrModelBundle {
    fn load(weights: &CanonicalWeightSetPaths, device: &Device) -> Result<Self> {
        let vb = unsafe {
            weights.var_builder(DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open TripoSR canonical weights: {source}"),
                }
            })?
        };
        Ok(Self {
            image_tokenizer: TripoImageTokenizer::load(vb.pp("image_tokenizer")).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSR image tokenizer: {source}"),
                },
            )?,
            triplane_tokenizer: Triplane1DTokenizer::load(vb.pp("tokenizer")).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSR triplane tokenizer: {source}"),
                },
            )?,
            backbone: Transformer1D::load(vb.pp("backbone")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSR backbone: {source}"),
                }
            })?,
            post_processor: TriplaneUpsampleNetwork::load(vb.pp("post_processor")).map_err(
                |source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSR post processor: {source}"),
                },
            )?,
            decoder: NerfMlp::load(vb.pp("decoder")).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to construct TripoSR decoder: {source}"),
                }
            })?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct TripoPreparedInputs {
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    pub preprocessed_image: Tensor, // [1, 1, 512, 512, 3]
}

#[derive(Debug, Clone)]
pub struct TripoMeshBuffers {
    pub vertices: Tensor,
    pub faces: Tensor,
    pub vertex_colors: Tensor,
}

#[derive(Debug, Clone)]
pub struct TripoInferenceOutput {
    pub preprocessed_image: Tensor,
    pub image_tokens: Tensor,
    pub triplane_seed_tokens: Tensor,
    pub backbone_tokens: Tensor,
    pub detokenized_triplanes: Tensor,
    pub scene_codes: Tensor,
    pub mesh: Option<TripoMeshBuffers>,
}

#[derive(Debug, Clone)]
pub struct TripoSrPipeline {
    pub weights: CanonicalWeightSetPaths,
    pub preprocess: TripoPreprocessStage,
    pub neural: TripoNeuralStage,
    pub geometry: TripoGeometryStage,
    pub export: TripoExportStage,
    bundle_cache: Arc<DeviceLocalCache<TripoSrModelBundle>>,
}

impl TripoSrPipeline {
    pub fn load(model_assets: ModelAssetOptions) -> Result<Self> {
        Ok(Self {
            weights: load_canonical_weights(ModelFamily::TripoSr, model_assets)?,
            preprocess: TripoPreprocessStage::default(),
            neural: TripoNeuralStage,
            geometry: TripoGeometryStage,
            export: TripoExportStage,
            bundle_cache: Arc::default(),
        })
    }

    fn bundle_for(&self, device: &Device) -> Result<Arc<TripoSrModelBundle>> {
        self.bundle_cache
            .get_or_try_init(device, || TripoSrModelBundle::load(&self.weights, device))
    }

    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoPreparedInputs> {
        self.preprocess.prepare_inputs_from_path(source, device)
    }

    pub fn image_tokens(&self, prepared_image: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(prepared_image.device())?;
        bundle
            .image_tokenizer
            .forward(prepared_image)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR image tokenizer forward failed: {source}"),
            })
    }

    pub fn triplane_seed_tokens(&self, batch_size: usize, device: &Device) -> Result<Tensor> {
        let bundle = self.bundle_for(device)?;
        bundle
            .triplane_tokenizer
            .forward(batch_size)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR triplane tokenizer forward failed: {source}"),
            })
    }

    pub fn scene_codes(
        &self,
        prepared_image: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let device = prepared_image.device();
        let bundle = self.bundle_for(device)?;

        let image_tokens = bundle
            .image_tokenizer
            .forward(prepared_image)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR image tokenizer forward failed: {source}"),
            })?;
        let seed_tokens = bundle.triplane_tokenizer.forward(1).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR triplane tokenizer forward failed: {source}"),
            }
        })?;
        let backbone_tokens = bundle
            .backbone
            .forward(&seed_tokens, &image_tokens)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR backbone forward failed: {source}"),
            })?;
        let detokenized = bundle
            .triplane_tokenizer
            .detokenize(&backbone_tokens)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR detokenize failed: {source}"),
            })?;
        let scene_codes = bundle
            .post_processor
            .forward(&detokenized)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR post processor forward failed: {source}"),
            })?;
        Ok((
            image_tokens,
            seed_tokens,
            backbone_tokens,
            detokenized,
            scene_codes,
        ))
    }

    pub fn scene_codes_from_detokenized(&self, detokenized: &Tensor) -> Result<Tensor> {
        let bundle = self.bundle_for(detokenized.device())?;
        bundle
            .post_processor
            .forward(detokenized)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR post processor forward failed: {source}"),
            })
    }

    pub fn query_triplane(
        &self,
        scene_code: &Tensor,
        positions: &Tensor,
        chunk_size: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let bundle = self.bundle_for(scene_code.device())?;
        query_triplane_chunked(scene_code, positions, &bundle.decoder, chunk_size).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR triplane query failed: {source}"),
            },
        )
    }

    pub fn query_triplane_density(
        &self,
        scene_code: &Tensor,
        positions: &Tensor,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(scene_code.device())?;
        query_triplane_density_chunked(scene_code, positions, &bundle.decoder, chunk_size).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR density-only query failed: {source}"),
            },
        )
    }

    pub fn query_triplane_color(
        &self,
        scene_code: &Tensor,
        positions: &Tensor,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let bundle = self.bundle_for(scene_code.device())?;
        query_triplane_color_chunked(scene_code, positions, &bundle.decoder, chunk_size).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR color-only query failed: {source}"),
            },
        )
    }

    pub fn infer_from_path(&self, source: &Path, device: &Device) -> Result<TripoInferenceOutput> {
        let prepared = self.prepare_inputs_from_path(source, device)?;
        self.neural.infer_tensors(self, &prepared)
    }

    pub fn extract_mesh(
        &self,
        scene_codes: &Tensor,
        resolution: u32,
        threshold: f32,
        chunk_size: usize,
    ) -> Result<TripoMeshBuffers> {
        if resolution < 2 {
            return Err(Lux3dError::InvalidInput(
                "TripoSR marching-cubes resolution must be at least 2",
            ));
        }
        let density_grid = self.neural.build_density_grid_cpu(
            self,
            scene_codes,
            resolution,
            threshold,
            chunk_size,
        )?;
        let mesh_cpu = self.geometry.assemble_cpu(&density_grid)?;
        let vertex_colors = self.neural.sample_vertex_colors_cpu(
            self,
            scene_codes,
            &mesh_cpu.vertices,
            chunk_size,
        )?;
        let mesh_cpu = self.geometry.attach_vertex_colors(mesh_cpu, vertex_colors);
        self.geometry
            .materialize_buffers(&mesh_cpu, scene_codes.device())
    }

    pub fn export_obj(&self, mesh: &TripoMeshBuffers, path: &Path) -> Result<()> {
        let mesh_cpu = self.geometry.cpu_from_buffers(mesh)?;
        self.export.write_obj(&mesh_cpu, path)
    }
}

pub(crate) fn prepare_triposr_inputs_with_stage(
    stage: &TripoPreprocessStage,
    source: &Path,
    device: &Device,
) -> Result<TripoPreparedInputs> {
    let image = ImageReader::open(source)
        .map_err(|_| Lux3dError::InvalidInput("failed to open TripoSR input image"))?
        .decode()
        .map_err(|_| Lux3dError::InvalidInput("failed to decode TripoSR input image"))?;

    let original_size = SpatialSize::new(image.width(), image.height());
    let target_edge = stage.target_edge() as usize;
    let preprocessed =
        preprocess_image_to_bhwc_f32(&image, target_edge, stage.background_value(), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("TripoSR preprocess failed: {source}"),
            })?;

    Ok(TripoPreparedInputs {
        original_size,
        target_size: SpatialSize::new(stage.target_edge(), stage.target_edge()),
        preprocessed_image: preprocessed,
    })
}

fn preprocess_image_to_bhwc_f32(
    image: &DynamicImage,
    target_size: usize,
    background_value: f32,
    device: &Device,
) -> CandleResult<Tensor> {
    let rgba = resize_foreground_rgba(image.to_rgba8(), 0.85);
    let rgb = compose_rgba_over_gray_quantized(&rgba, background_value);

    #[cfg(feature = "vision-preproc")]
    {
        let chw = vision_preproc::tensorize_rgb_image(
            &DynamicImage::ImageRgb8(rgb.clone()),
            &Device::Cpu,
        )?;
        let chw =
            vision_preproc::resize_chw_image(&chw, rgb.height() as usize, rgb.width() as usize)?;
        let host = vision_preproc::chw_to_hwc_vec_f32(&chw)?;
        let resized = resize_rgb_bilinear_antialias(
            &host,
            rgb.height() as usize,
            rgb.width() as usize,
            target_size,
            target_size,
        );
        Tensor::from_vec(resized, (1, target_size, target_size, 3), &Device::Cpu)?
            .to_device(device)?
            .unsqueeze(1)
    }

    #[cfg(not(feature = "vision-preproc"))]
    {
        let (width, height) = rgb.dimensions();
        let mut host = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let px = rgb.get_pixel(x, y).0;
                host.push(f32::from(px[0]) / 255.0);
                host.push(f32::from(px[1]) / 255.0);
                host.push(f32::from(px[2]) / 255.0);
            }
        }
        let resized = resize_rgb_bilinear_antialias(
            &host,
            height as usize,
            width as usize,
            target_size,
            target_size,
        );
        let input = Tensor::from_vec(resized, (1, target_size, target_size, 3), &Device::Cpu)?
            .to_device(device)?
            .unsqueeze(1)?;
        Ok(input)
    }
}

fn compose_rgba_over_gray_quantized(image: &RgbaImage, background_value: f32) -> image::RgbImage {
    let (width, height) = image.dimensions();
    let mut out = image::RgbImage::new(width, height);
    for (x, y, pixel) in image.enumerate_pixels() {
        let px = pixel.0;
        let alpha = f32::from(px[3]) / 255.0;
        let r = ((f32::from(px[0]) / 255.0) * alpha + (1.0 - alpha) * background_value) * 255.0;
        let g = ((f32::from(px[1]) / 255.0) * alpha + (1.0 - alpha) * background_value) * 255.0;
        let b = ((f32::from(px[2]) / 255.0) * alpha + (1.0 - alpha) * background_value) * 255.0;
        out.put_pixel(
            x,
            y,
            image::Rgb([
                r.floor().clamp(0.0, 255.0) as u8,
                g.floor().clamp(0.0, 255.0) as u8,
                b.floor().clamp(0.0, 255.0) as u8,
            ]),
        );
    }
    out
}

fn resize_rgb_bilinear_antialias(
    src: &[f32],
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
) -> Vec<f32> {
    let horizontal = compute_aa_linear_weights(input_w, output_w);
    let vertical = compute_aa_linear_weights(input_h, output_h);
    let mut temp = vec![0f32; input_h * output_w * 3];
    for y in 0..input_h {
        for (out_x, spec) in horizontal.iter().enumerate().take(output_w) {
            for (offset, weight) in spec.weights.iter().enumerate() {
                let src_x = spec.start + offset;
                let src_idx = (y * input_w + src_x) * 3;
                let dst_idx = (y * output_w + out_x) * 3;
                temp[dst_idx] += src[src_idx] * *weight;
                temp[dst_idx + 1] += src[src_idx + 1] * *weight;
                temp[dst_idx + 2] += src[src_idx + 2] * *weight;
            }
        }
    }

    let mut out = vec![0f32; output_h * output_w * 3];
    for (out_y, spec) in vertical.iter().enumerate().take(output_h) {
        for out_x in 0..output_w {
            let dst_idx = (out_y * output_w + out_x) * 3;
            for (offset, weight) in spec.weights.iter().enumerate() {
                let src_y = spec.start + offset;
                let src_idx = (src_y * output_w + out_x) * 3;
                out[dst_idx] += temp[src_idx] * *weight;
                out[dst_idx + 1] += temp[src_idx + 1] * *weight;
                out[dst_idx + 2] += temp[src_idx + 2] * *weight;
            }
        }
    }
    out
}

fn resize_foreground_rgba(image: RgbaImage, ratio: f32) -> RgbaImage {
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    let mut found = false;

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > 0 {
            found = true;
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    if !found {
        return image;
    }

    let crop_w = (max_x - min_x).max(1);
    let crop_h = (max_y - min_y).max(1);
    let fg = image::imageops::crop_imm(&image, min_x, min_y, crop_w, crop_h).to_image();
    let square = crop_to_square_rgba(&fg);
    let new_size = ((square.width() as f32) / ratio).max(square.width() as f32) as u32;
    pad_rgba_to_size(&square, new_size)
}

fn crop_to_square_rgba(image: &RgbaImage) -> RgbaImage {
    let size = image.width().max(image.height());
    let x = (size - image.width()) / 2;
    let y = (size - image.height()) / 2;
    let mut out = ImageBuffer::from_pixel(size, size, Rgba([0, 0, 0, 0]));
    copy_rgba(image, &mut out, x, y);
    out
}

fn pad_rgba_to_size(image: &RgbaImage, size: u32) -> RgbaImage {
    let x = (size - image.width()) / 2;
    let y = (size - image.height()) / 2;
    let mut out = ImageBuffer::from_pixel(size, size, Rgba([0, 0, 0, 0]));
    copy_rgba(image, &mut out, x, y);
    out
}

fn copy_rgba(src: &RgbaImage, dst: &mut RgbaImage, offset_x: u32, offset_y: u32) {
    for sy in 0..src.height() {
        for sx in 0..src.width() {
            let px = *src.get_pixel(sx, sy);
            dst.put_pixel(offset_x + sx, offset_y + sy, px);
        }
    }
}

#[derive(Debug, Clone)]
struct TripoImageTokenizer {
    embeddings: VitEmbeddings,
    encoder: VitEncoder,
    layernorm: LayerNorm,
    mean: Tensor,
    std: Tensor,
}

impl TripoImageTokenizer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let model_vb = vb.pp("model");
        let dtype = vb.dtype();
        Ok(Self {
            embeddings: VitEmbeddings::load(model_vb.pp("embeddings"))?,
            encoder: VitEncoder::load(model_vb.pp("encoder"))?,
            layernorm: layer_norm(768, 1e-12, model_vb.pp("layernorm"))?,
            mean: Tensor::from_slice(&[0.485f32, 0.456, 0.406], (1, 1, 3, 1, 1), vb.device())?
                .to_dtype(dtype)?,
            std: Tensor::from_slice(&[0.229f32, 0.224, 0.225], (1, 1, 3, 1, 1), vb.device())?
                .to_dtype(dtype)?,
        })
    }

    fn forward(&self, image_bhwc: &Tensor) -> CandleResult<Tensor> {
        let dims = image_bhwc.dims();
        let batch = dims[0];
        let views = dims[1];
        let height = dims[2];
        let width = dims[3];

        let images = image_bhwc
            .permute((0, 1, 4, 2, 3))?
            .broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?
            .reshape((batch * views, 3, height, width))?;
        let embeddings = self.embeddings.forward(&images, true)?;
        let hidden = self
            .layernorm
            .forward(&self.encoder.forward(&embeddings)?)?;
        hidden
            .transpose(1, 2)?
            .reshape((batch, views, 768, hidden.dim(1)?))?
            .permute((0, 1, 3, 2))?
            .reshape((batch, views * hidden.dim(1)?, 768))
    }
}

#[derive(Debug, Clone)]
struct VitEmbeddings {
    cls_token: Tensor,
    patch_projection: candle_nn::Conv2d,
    position_embeddings: Tensor,
    resized_position_embeddings: Tensor,
}

impl VitEmbeddings {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let patch_projection = candle_nn::conv2d(
            3,
            768,
            16,
            candle_nn::Conv2dConfig {
                stride: 16,
                ..Default::default()
            },
            vb.pp("patch_embeddings").pp("projection"),
        )?;
        let position_embeddings = vb.get((1, 197, 768), "position_embeddings")?;
        let resized_position_embeddings = resize_vit_position_embeddings_32(&position_embeddings)?;
        Ok(Self {
            cls_token: vb.get((1, 1, 768), "cls_token")?,
            patch_projection,
            position_embeddings,
            resized_position_embeddings,
        })
    }

    fn forward(&self, images: &Tensor, interpolate_pos_encoding: bool) -> CandleResult<Tensor> {
        let (batch, _c, _h, _w) = images.dims4()?;
        let patch_tokens = self
            .patch_projection
            .forward(images)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let cls = self.cls_token.broadcast_as((batch, 1, 768))?;
        let embeddings = Tensor::cat(&[&cls, &patch_tokens], 1)?;
        if interpolate_pos_encoding {
            embeddings.broadcast_add(&self.resized_position_embeddings)
        } else {
            embeddings.broadcast_add(&self.position_embeddings)
        }
    }
}

fn resize_vit_position_embeddings_32(position_embeddings: &Tensor) -> CandleResult<Tensor> {
    let device = position_embeddings.device();
    let dtype = position_embeddings.dtype();
    let cls = position_embeddings.i((.., ..1, ..))?;
    let patch = position_embeddings
        .i((.., 1.., ..))?
        .reshape((14 * 14, 768))?
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let resized = resize_patch_pos_embed_bicubic(&patch, 14, 32, 32, 768);
    let resized = Tensor::from_vec(resized, (1, 32 * 32, 768), &Device::Cpu)?
        .to_dtype(dtype)?
        .to_device(device)?;
    Tensor::cat(&[&cls, &resized], 1)
}

fn resize_patch_pos_embed_bicubic(
    flat: &[f32],
    source_hw: usize,
    target_h: usize,
    target_w: usize,
    dim: usize,
) -> Vec<f32> {
    let mut resized = vec![0f32; target_h * target_w * dim];
    for out_y in 0..target_h {
        let src_y = ((out_y as f32 + 0.5) * source_hw as f32 / target_h as f32) - 0.5;
        let y_base = src_y.floor() as isize;
        let y_frac = src_y - y_base as f32;
        for out_x in 0..target_w {
            let src_x = ((out_x as f32 + 0.5) * source_hw as f32 / target_w as f32) - 0.5;
            let x_base = src_x.floor() as isize;
            let x_frac = src_x - x_base as f32;
            for channel in 0..dim {
                let mut value = 0f32;
                for ky in -1..=2 {
                    let iy = clamp_isize(y_base + ky, 0, source_hw as isize - 1) as usize;
                    let wy = cubic_weight(y_frac - ky as f32);
                    for kx in -1..=2 {
                        let ix = clamp_isize(x_base + kx, 0, source_hw as isize - 1) as usize;
                        let wx = cubic_weight(x_frac - kx as f32);
                        value += flat[(iy * source_hw + ix) * dim + channel] * wy * wx;
                    }
                }
                resized[(out_y * target_w + out_x) * dim + channel] = value;
            }
        }
    }
    resized
}

#[derive(Debug, Clone)]
struct VitSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl VitSelfAttention {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            query: linear(768, 768, vb.pp("query"))?,
            key: linear(768, 768, vb.pp("key"))?,
            value: linear(768, 768, vb.pp("value"))?,
            num_heads: 12,
            head_dim: 64,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let q = self.to_heads(&self.query.forward(xs)?)?;
        let k = self.to_heads(&self.key.forward(xs)?)?;
        let v = self.to_heads(&self.value.forward(xs)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let q = q.contiguous()?;
        let k = k.transpose(2, 3)?.contiguous()?;
        let v = v.contiguous()?;
        let scores = (q.matmul(&k)?.affine(scale, 0.0))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        attn.matmul(&v)?
            .transpose(1, 2)?
            .reshape((xs.dim(0)?, xs.dim(1)?, 768))
    }

    fn to_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        xs.reshape((batch, seq, self.num_heads, dim / self.num_heads))?
            .transpose(1, 2)
    }
}

#[derive(Debug, Clone)]
struct VitAttention {
    attention: VitSelfAttention,
    output: Linear,
}

impl VitAttention {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            attention: VitSelfAttention::load(vb.pp("attention"))?,
            output: linear(768, 768, vb.pp("output").pp("dense"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.output.forward(&self.attention.forward(xs)?)
    }
}

#[derive(Debug, Clone)]
struct VitLayer {
    attention: VitAttention,
    intermediate: Linear,
    output: Linear,
    layernorm_before: LayerNorm,
    layernorm_after: LayerNorm,
}

impl VitLayer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            attention: VitAttention::load(vb.pp("attention"))?,
            intermediate: linear(768, 3072, vb.pp("intermediate").pp("dense"))?,
            output: linear(3072, 768, vb.pp("output").pp("dense"))?,
            layernorm_before: layer_norm(768, 1e-12, vb.pp("layernorm_before"))?,
            layernorm_after: layer_norm(768, 1e-12, vb.pp("layernorm_after"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let attn = self
            .attention
            .forward(&self.layernorm_before.forward(xs)?)?;
        let xs = (xs + attn)?;
        let ff = self
            .intermediate
            .forward(&self.layernorm_after.forward(&xs)?)?
            .gelu_erf()?;
        xs + self.output.forward(&ff)?
    }
}

#[derive(Debug, Clone)]
struct VitEncoder {
    layers: Vec<VitLayer>,
}

impl VitEncoder {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let mut layers = Vec::with_capacity(12);
        for idx in 0..12 {
            layers.push(VitLayer::load(vb.pp("layer").pp(idx.to_string()))?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let mut hidden = xs.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }
        Ok(hidden)
    }
}

#[derive(Debug, Clone)]
struct Triplane1DTokenizer {
    embeddings: Tensor,
}

impl Triplane1DTokenizer {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            embeddings: vb.get((3, 1024, 32, 32), "embeddings")?,
        })
    }

    fn forward(&self, batch_size: usize) -> CandleResult<Tensor> {
        self.embeddings
            .broadcast_as((batch_size, 3, 1024, 32, 32))?
            .reshape((batch_size, 3, 1024, 32 * 32))?
            .transpose(1, 2)?
            .reshape((batch_size, 1024, 3 * 32 * 32))
    }

    fn detokenize(&self, tokens: &Tensor) -> CandleResult<Tensor> {
        tokens
            .reshape((tokens.dim(0)?, 1024, 3, 32, 32))?
            .permute((0, 2, 1, 3, 4))
    }
}

#[derive(Debug, Clone)]
struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    heads: usize,
    scale: f32,
}

impl CrossAttention {
    fn load(vb: VarBuilder, query_dim: usize, context_dim: usize) -> CandleResult<Self> {
        Ok(Self {
            to_q: linear_no_bias(query_dim, 1024, vb.pp("to_q"))?,
            to_k: linear_no_bias(context_dim, 1024, vb.pp("to_k"))?,
            to_v: linear_no_bias(context_dim, 1024, vb.pp("to_v"))?,
            to_out: linear(1024, query_dim, vb.pp("to_out").pp("0"))?,
            heads: 16,
            scale: 1.0 / 8.0,
        })
    }

    fn forward(&self, xs: &Tensor, context: Option<&Tensor>) -> CandleResult<Tensor> {
        let ctx = context.unwrap_or(xs);
        let q = self.to_heads(&self.to_q.forward(xs)?)?;
        let k = self.to_heads(&self.to_k.forward(ctx)?)?;
        let v = self.to_heads(&self.to_v.forward(ctx)?)?;
        let out = exact_sdpa_heads(&q, &k, &v, self.scale)?;
        let out = self.merge_heads(&out)?;
        self.to_out.forward(&out)
    }

    fn to_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, seq, dim) = xs.dims3()?;
        xs.reshape((batch, seq, self.heads, dim / self.heads))?
            .transpose(1, 2)
    }

    fn merge_heads(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let (batch, _heads, seq, head_dim) = xs.dims4()?;
        xs.reshape((batch, self.heads, seq, head_dim))?
            .transpose(1, 2)?
            .reshape((batch, seq, self.heads * head_dim))
    }
}

#[derive(Debug, Clone)]
struct BasicTransformerBlock1d {
    attn1: CrossAttention,
    attn2: CrossAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl BasicTransformerBlock1d {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        Ok(Self {
            attn1: CrossAttention::load(vb.pp("attn1"), 1024, 1024)?,
            attn2: CrossAttention::load(vb.pp("attn2"), 1024, 768)?,
            ff: FeedForward::load(vb.pp("ff"), 1024)?,
            norm1: layer_norm(1024, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(1024, 1e-5, vb.pp("norm2"))?,
            norm3: layer_norm(1024, 1e-5, vb.pp("norm3"))?,
        })
    }

    fn forward(&self, xs: &Tensor, encoder_hidden_states: &Tensor) -> CandleResult<Tensor> {
        let xs = (xs + self.attn1.forward(&self.norm1.forward(xs)?, None)?)?;
        let xs = (xs.clone()
            + self
                .attn2
                .forward(&self.norm2.forward(&xs)?, Some(encoder_hidden_states))?)?;
        xs.clone() + self.ff.forward(&self.norm3.forward(&xs)?)?
    }
}

#[derive(Debug, Clone)]
struct Transformer1D {
    norm: GroupNorm,
    proj_in: Linear,
    blocks: Vec<BasicTransformerBlock1d>,
    proj_out: Linear,
}

impl Transformer1D {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let mut blocks = Vec::with_capacity(16);
        for idx in 0..16 {
            blocks.push(BasicTransformerBlock1d::load(
                vb.pp("transformer_blocks").pp(idx.to_string()),
            )?);
        }
        Ok(Self {
            norm: group_norm(32, 1024, 1e-6, vb.pp("norm"))?,
            proj_in: linear(1024, 1024, vb.pp("proj_in"))?,
            blocks,
            proj_out: linear(1024, 1024, vb.pp("proj_out"))?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
    ) -> CandleResult<Tensor> {
        let (batch, channels, seq) = hidden_states.dims3()?;
        let residual = hidden_states.clone();
        let mut hidden = self.norm.forward(hidden_states)?;
        hidden = hidden.permute((0, 2, 1))?;
        hidden = self.proj_in.forward(&hidden)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden, encoder_hidden_states)?;
        }
        hidden = self.proj_out.forward(&hidden)?;
        hidden = hidden.reshape((batch, seq, channels))?.permute((0, 2, 1))?;
        hidden + residual
    }
}

#[derive(Debug, Clone)]
struct TriplaneUpsampleNetwork {
    upsample: ConvTranspose2d,
}

impl TriplaneUpsampleNetwork {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let cfg = ConvTranspose2dConfig {
            stride: 2,
            ..Default::default()
        };
        Ok(Self {
            upsample: conv_transpose2d(1024, 40, 2, cfg, vb.pp("upsample"))?,
        })
    }

    fn forward(&self, triplanes: &Tensor) -> CandleResult<Tensor> {
        let (batch, planes, channels, height, width) = triplanes.dims5()?;
        let upsampled = self.upsample.forward(&triplanes.reshape((
            batch * planes,
            channels,
            height,
            width,
        ))?)?;
        upsampled.reshape((
            batch,
            planes,
            upsampled.dim(1)?,
            upsampled.dim(2)?,
            upsampled.dim(3)?,
        ))
    }
}

#[derive(Debug, Clone)]
struct NerfMlp {
    layers: Vec<Linear>,
}

impl NerfMlp {
    fn load(vb: VarBuilder) -> CandleResult<Self> {
        let layer_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18];
        let mut layers = Vec::with_capacity(layer_indices.len());
        let mut in_dim = 120;
        for (i, idx) in layer_indices.into_iter().enumerate() {
            let out_dim = if i + 1 == layer_indices.len() { 4 } else { 64 };
            layers.push(linear(
                in_dim,
                out_dim,
                vb.pp("layers").pp(idx.to_string()),
            )?);
            in_dim = out_dim;
        }
        Ok(Self { layers })
    }

    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let mut hidden = xs.clone();
        for layer in &self.layers[..self.layers.len() - 1] {
            hidden = layer.forward(&hidden)?.silu()?;
        }
        let out = self.layers[self.layers.len() - 1].forward(&hidden)?;
        Ok((out.i((.., ..1))?, out.i((.., 1..4))?))
    }
}

impl TriplaneDecoder for NerfMlp {
    fn forward(&self, xs: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        self.forward(xs)
    }
}
