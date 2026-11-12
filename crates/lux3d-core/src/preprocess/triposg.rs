//! TripoSG image preprocessing (reference `scripts/image_process.py` alpha path
//! plus the `BitImageProcessor` DINOv2 conditioning stage).
//!
//! The upstream `load_image` has two branches. A valid alpha channel is used
//! directly; otherwise BriaRMBG (RMBG-1.4) predicts a mask. Both are
//! implemented here. The RMBG branch follows the reference exactly:
//! 1024x1024 torchvision bilinear-antialias resize, `(x - 0.5) / 1.0`
//! normalization, the IS-Net forward, mask resize back to the capped source
//! size with antialias, min-max scaling, uint8 quantization, OpenCV Otsu
//! threshold, 8-connected labelling, removal of components smaller than 200
//! pixels, and the largest contour bounding box.
//!
//! Pipeline (upstream alpha branch):
//! 1. decode with the `image` crate, convert to RGBA8; alpha is honored only
//!    when the source color type actually carries an alpha channel;
//! 2. cap the longest side at 2000 px with an area-average resample before any
//!    alpha analysis (cv2 `INTER_AREA` equivalent; alpha included);
//! 3. alpha validity: 20-bin histogram of alpha, bin `v * 20 / 256`, valid iff
//!    `hist[0]` and `hist[19]` both hold at least 1% of the pixels;
//! 4. foreground bbox over `alpha > 0` (all-zero alpha is an error), otherwise
//!    the full image;
//! 5. composite over white in f32 exactly like the reference: the reference
//!    runs on CUDA where scalar division is lowered to multiplication by the
//!    f32 reciprocal of 255, so the port uses `v * (1/255)` (not `v / 255`) to
//!    stay bit-identical; then `(out * 255) as u8` truncation reproduces
//!    `(img_np * 255).astype(np.uint8)` — identity for opaque inputs;
//! 6. square-pad with 10% of the long side using the reference truncating
//!    arithmetic;
//! 7. `BitImageProcessor`: resize the shortest edge to 256 with the
//!    **torch-native uint8 antialias bicubic** kernel and center-crop 224.
//!    The baseline was generated with transformers 5.17 whose `BitImageProcessor`
//!    routes PIL input through `torchvision.transforms.v2.functional.resize`
//!    on the uint8 tensor (aten `upsample_bicubic2d_aa` int16 fixed-point
//!    path). That kernel is PIL-structured (separable 8-bit passes, 22-bit
//!    capped int16 coefficients, half-LSB accumulator, truncating shift,
//!    saturating clamp) but quantizes coefficients to a measured fixed-point
//!    precision, so it differs from Pillow's own 8-bit BICUBIC by +-1 on
//!    ~0.04% of pixels. The port below reproduces the aten kernel exactly
//!    (`UpSampleKernel.cpp` v2.5.1), which was validated bit-exactly against
//!    `torchvision.transforms.v2.functional.resize` across 51 size pairs;
//! 8. produce CHW uint8 RGB (exact baseline parity) plus the normalized
//!    `[1, 3, 224, 224]` f32 `pixel_values`.

use std::path::Path;

use candle_core::{Device, Tensor};
use image::{DynamicImage, ImageReader, Rgb, RgbImage, RgbaImage};

use crate::{
    Result,
    contracts::{
        ModelFamily, NormalizationStats, PreprocessStage, SpatialSize, TensorContract,
        TripoSgPreparedImage, TripoSgRawInput,
    },
    error::Lux3dError,
};

/// Reference cap before any alpha analysis (`image_process.py:44-52`).
const MAX_INPUT_SIDE: u32 = 2000;
/// `feature_extractor_dinov2/preprocessor_config.json`: `size {"shortest_edge": 256}`.
const RESIZE_SHORTEST_EDGE: u32 = 256;
/// Center crop of the DINOv2 conditioning stage.
const CROP_SIZE: u32 = 224;
/// Reference `is_valid_alpha` histogram resolution.
const ALPHA_HISTOGRAM_BINS: usize = 20;
/// Reference `is_valid_alpha` minimum fraction per extreme bin.
const ALPHA_VALID_MIN_RATIO: f64 = 0.01;

/// CUDA `TensorIterator` lowers scalar division into multiplication by the f32
/// reciprocal; replicating that keeps the composite bit-identical to the
/// reference CUDA preprocessing (1/255 rounded to f32).
const COLOR_RECIPROCAL_255: f32 = 1.0 / 255.0;

/// Prepared TripoSG conditioning inputs.
#[derive(Debug, Clone)]
pub struct TripoSgPreparedInputs {
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    /// Final 224x224 RGB in CHW order (`R` plane, `G` plane, `B` plane,
    /// row-major). Must match `triposg.preprocessed_uint8` exactly.
    pub uint8_rgb: Vec<u8>,
    /// `[1, 3, 224, 224]` f32, `(v / 255 - mean) / std` with ImageNet stats.
    pub pixel_values: Tensor,
}

/// TripoSG image preprocess stage (reference alpha path + BitImageProcessor).
#[derive(Debug, Clone, PartialEq)]
pub struct TripoSgPreprocessStage {
    background_value: f32,
    padding_ratio: f32,
    resize_shortest_edge: u32,
    crop_size: u32,
    normalization: NormalizationStats,
}

impl Default for TripoSgPreprocessStage {
    fn default() -> Self {
        Self {
            background_value: 1.0,
            padding_ratio: 0.1,
            resize_shortest_edge: RESIZE_SHORTEST_EDGE,
            crop_size: CROP_SIZE,
            normalization: NormalizationStats::dino(),
        }
    }
}

impl TripoSgPreprocessStage {
    /// Decode `source` and run the full preprocessing pipeline. Inputs with a
    /// valid alpha channel do not need an RMBG model; inputs without one
    /// return an error (use [`Self::prepare_inputs_from_path_with_rmbg`]).
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        self.prepare_inputs_from_path_with_rmbg(source, device, None)
    }

    /// Decode `source` and run the full reference pipeline, using `rmbg` when
    /// the input has no valid alpha channel.
    pub fn prepare_inputs_from_path_with_rmbg(
        &self,
        source: &Path,
        device: &Device,
        rmbg: Option<&crate::runtime::RmbgModel>,
    ) -> Result<TripoSgPreparedInputs> {
        let image = ImageReader::open(source)
            .map_err(|_| Lux3dError::InvalidInput("failed to open TripoSG input image"))?
            .decode()
            .map_err(|_| Lux3dError::InvalidInput("failed to decode TripoSG input image"))?;
        self.prepare_dynamic_image_with_rmbg(&image, device, rmbg)
    }

    /// Run the full preprocessing pipeline on an already-decoded image.
    ///
    /// Alpha is honored only when the source color type actually carries an
    /// alpha channel (`Rgba8`/`La8`/`Rgba16`/... after conversion). Without a
    /// valid alpha channel the reference always invokes BriaRMBG, so `rmbg`
    /// must be supplied.
    pub fn prepare_dynamic_image(
        &self,
        image: &DynamicImage,
        device: &Device,
    ) -> Result<TripoSgPreparedInputs> {
        self.prepare_dynamic_image_with_rmbg(image, device, None)
    }

    /// Full reference preprocessing with an optional BriaRMBG model.
    pub fn prepare_dynamic_image_with_rmbg(
        &self,
        image: &DynamicImage,
        device: &Device,
        rmbg: Option<&crate::runtime::RmbgModel>,
    ) -> Result<TripoSgPreparedInputs> {
        let original_size = SpatialSize::new(image.width(), image.height());
        if image.width() == 0 || image.height() == 0 {
            return Err(Lux3dError::InvalidInput("TripoSG input image is empty"));
        }

        let has_alpha_channel = image.color().has_alpha();
        let rgba = image.to_rgba8();

        // Cap the longest side before any alpha analysis (alpha included).
        let capped = cap_to_max_side(&rgba, MAX_INPUT_SIDE);

        // Alpha validity per the reference `is_valid_alpha` histogram test.
        let alpha_bbox = if has_alpha_channel
            && is_valid_alpha(&capped, ALPHA_HISTOGRAM_BINS, ALPHA_VALID_MIN_RATIO)
        {
            Some(foreground_bbox(&capped).ok_or(Lux3dError::InvalidInput(
                "TripoSG input image alpha channel is entirely zero",
            ))?)
        } else {
            None
        };

        // Foreground crop (alpha branch) or BriaRMBG mask branch, then
        // composite over the background in f32 and quantize like the reference.
        let (_x, _y, w, h, composited) = match alpha_bbox {
            Some((x, y, w, h)) => (
                x,
                y,
                w,
                h,
                composite_over_background(&capped, x, y, w, h, true, self.background_value),
            ),
            None => {
                let rmbg = rmbg.ok_or(Lux3dError::InvalidInput(
                    "TripoSG input has no valid alpha channel; BriaRMBG weights are required (pass --rmbg-model-path or set the RMBG canonical package)",
                ))?;
                let mask = rmbg_alpha_mask(&capped, rmbg, device)?;
                let (x, y, w, h) = largest_foreground_bbox(&mask, capped.width(), capped.height())
                    .ok_or(Lux3dError::InvalidInput("TripoSG input image too small"))?;
                let composited =
                    composite_binary_mask(&capped, &mask, x, y, w, h, self.background_value);
                (x, y, w, h, composited)
            }
        };

        // Square-pad with 10% of the long side (reference truncating arithmetic).
        let (pad_lr, pad_tb) = self.square_pad_extents(w as usize, h as usize);
        let padded = square_pad(&composited, w, h, pad_lr, pad_tb, self.background_value);

        // BitImageProcessor: shortest-edge resize (torch-native uint8
        // antialias bicubic) + center crop.
        let (resized_w, resized_h) =
            shortest_edge_output_size(padded.width(), padded.height(), self.resize_shortest_edge);
        let resized = resize_rgb_bicubic_antialias_u8(&padded, resized_w, resized_h);
        let cropped = center_crop(&resized, self.crop_size);

        let target_size = SpatialSize::new(self.crop_size, self.crop_size);
        let uint8_rgb = chw_uint8(&cropped);
        let pixel_values = normalized_pixel_values(
            &uint8_rgb,
            cropped.width(),
            cropped.height(),
            &self.normalization,
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize TripoSG pixel values: {source}"),
        })?;

        Ok(TripoSgPreparedInputs {
            original_size,
            target_size,
            uint8_rgb,
            pixel_values,
        })
    }

    /// Reference square-pad arithmetic (`image_process.py:130-138`, Python
    /// `int()` truncation):
    /// - `w > h`: `pad_lr = int(w * ratio)`, `pad_tb = int(pad_lr + (w - h) / 2)`;
    /// - else:   `pad_tb = int(h * ratio)`, `pad_lr = int(pad_tb + (h - w) / 2)`.
    ///
    /// The result is square whenever `(w - h)` is even; for odd differences the
    /// reference arithmetic pads one row/col short — the shortest-edge resize +
    /// center crop stage handles rectangles, so the reference behavior is kept.
    pub fn square_pad_extents(&self, width: usize, height: usize) -> (usize, usize) {
        square_pad_extents(width, height, self.padding_ratio)
    }

    /// Dimensions after the reference >2000 px cap: `scale = 2000 / max(w, h)`
    /// with Python `int()` truncation, identity when already within the cap.
    pub fn capped_input_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        resized_dimensions(width, height, MAX_INPUT_SIDE)
    }

    /// Composite a single premultiplied-source RGBA pixel ([0, 1] floats) over
    /// the stage background color.
    pub fn composite_rgba_over_white(&self, rgba: [f32; 4]) -> [f32; 3] {
        let alpha = rgba[3];
        let inverse = 1.0 - alpha;
        [
            rgba[0] * alpha + self.background_value * inverse,
            rgba[1] * alpha + self.background_value * inverse,
            rgba[2] * alpha + self.background_value * inverse,
        ]
    }

    /// Normalize a composited RGB pixel ([0, 1] floats) with the stage stats.
    pub fn normalize_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        normalize_pixel(rgb, &self.normalization)
    }
}

impl PreprocessStage<TripoSgRawInput, TripoSgPreparedImage> for TripoSgPreprocessStage {
    /// Metadata/contract view: composite a single RGBA/RGB pixel over white,
    /// quantize it, and describe the `pixel_values` tensor. Normalization
    /// stats ride along in `normalization` and per-pixel normalized values are
    /// available via `normalize_rgb`. No file I/O.
    fn preprocess(&self, input: TripoSgRawInput) -> Result<TripoSgPreparedImage> {
        let composited = match (input.rgba, input.rgb) {
            (Some(rgba), _) => self.composite_rgba_over_white(rgba),
            (None, Some(rgb)) => rgb,
            (None, None) => {
                return Err(Lux3dError::InvalidInput(
                    "TripoSG raw input requires rgb or rgba values",
                ));
            }
        };
        let uint8_rgb = composited.map(|value| (value * 255.0) as u8).to_vec();

        Ok(TripoSgPreparedImage {
            family: ModelFamily::TripoSg,
            original_size: input.original_size,
            target_size: SpatialSize::new(self.crop_size, self.crop_size),
            background_value: self.background_value,
            normalization: self.normalization.clone(),
            uint8_rgb,
            pixel_values: TensorContract::new(
                "pixel_values",
                vec![1, 3, self.crop_size as usize, self.crop_size as usize],
            ),
        })
    }
}

fn normalize_pixel(rgb: [f32; 3], stats: &NormalizationStats) -> [f32; 3] {
    [
        (rgb[0] - stats.mean[0]) / stats.std[0],
        (rgb[1] - stats.mean[1]) / stats.std[1],
        (rgb[2] - stats.mean[2]) / stats.std[2],
    ]
}

fn square_pad_extents(width: usize, height: usize, padding_ratio: f32) -> (usize, usize) {
    if width > height {
        let pad_lr = (width as f32 * padding_ratio) as usize;
        let pad_tb = (pad_lr as f32 + (width - height) as f32 / 2.0) as usize;
        (pad_lr, pad_tb)
    } else {
        let pad_tb = (height as f32 * padding_ratio) as usize;
        let pad_lr = (pad_tb as f32 + (height - width) as f32 / 2.0) as usize;
        (pad_lr, pad_tb)
    }
}

/// Downscale when the longest side exceeds `max_side`, using an area average
/// (cv2 `INTER_AREA` equivalent, applied to alpha as well). Source dimensions
/// map with Python `int()` truncation.
fn cap_to_max_side(image: &RgbaImage, max_side: u32) -> RgbaImage {
    let (width, height) = image.dimensions();
    let longest = width.max(height);
    if longest <= max_side {
        return image.clone();
    }
    let scale = f64::from(max_side) / f64::from(longest);
    let new_width = ((f64::from(width) * scale) as u32).max(1);
    let new_height = ((f64::from(height) * scale) as u32).max(1);
    resize_rgba_area_average(image, new_width, new_height)
}

fn resized_dimensions(width: u32, height: u32, max_side: u32) -> (u32, u32) {
    let longest = width.max(height);
    if longest <= max_side {
        return (width, height);
    }
    let scale = f64::from(max_side) / f64::from(longest);
    (
        ((f64::from(width) * scale) as u32).max(1),
        ((f64::from(height) * scale) as u32).max(1),
    )
}

/// Area-average resample: every output pixel averages the source rectangle it
/// covers, with fractional edge weights and f64 accumulation (round half up,
/// saturating at u8). Approximates cv2 `INTER_AREA` for uint8; exact for
/// uniform regions and preserves the reference bbox/dimension behavior.
fn resize_rgba_area_average(image: &RgbaImage, out_width: u32, out_height: u32) -> RgbaImage {
    let (in_width, in_height) = image.dimensions();
    if in_width == out_width && in_height == out_height {
        return image.clone();
    }
    let source = image.as_raw();
    let out_width_usize = out_width as usize;
    let in_width_usize = in_width as usize;

    let horizontal = area_average_taps(in_width as usize, out_width as usize);
    let vertical = area_average_taps(in_height as usize, out_height as usize);

    // Horizontal pass into f32 rows: (in_height, out_width, 4).
    let mut intermediate = vec![0.0_f64; in_height as usize * out_width_usize * 4];
    for y in 0..in_height as usize {
        for (x_out, taps) in horizontal.iter().enumerate() {
            let mut accumulator = [0.0_f64; 4];
            for &(source_x, weight) in taps {
                let index = (y * in_width_usize + source_x) * 4;
                for channel in 0..4 {
                    accumulator[channel] += f64::from(source[index + channel]) * weight;
                }
            }
            let target = (y * out_width_usize + x_out) * 4;
            intermediate[target..target + 4].copy_from_slice(&accumulator);
        }
    }

    let mut output = vec![0_u8; out_height as usize * out_width_usize * 4];
    for (y_out, taps) in vertical.iter().enumerate() {
        for x_out in 0..out_width_usize {
            let mut accumulator = [0.0_f64; 4];
            for &(source_y, weight) in taps {
                let index = (source_y * out_width_usize + x_out) * 4;
                for channel in 0..4 {
                    accumulator[channel] += intermediate[index + channel] * weight;
                }
            }
            let target = (y_out * out_width_usize + x_out) * 4;
            for channel in 0..4 {
                output[target + channel] = (accumulator[channel] + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }

    RgbaImage::from_raw(out_width, out_height, output).expect("resized buffer dimensions")
}

/// Per-output-pixel `(source_index, weight)` taps covering `[out, out + 1)` in
/// source coordinates; weights sum to 1.
fn area_average_taps(input_size: usize, output_size: usize) -> Vec<Vec<(usize, f64)>> {
    let scale = input_size as f64 / output_size as f64;
    let mut taps = Vec::with_capacity(output_size);
    for out in 0..output_size {
        let start = out as f64 * scale;
        let end = start + scale;
        let first = start.floor() as usize;
        let last = (end.ceil() as usize).min(input_size);
        let mut row = Vec::with_capacity(last - first + 1);
        for index in first..last {
            let lo = index as f64;
            let overlap = end.min(lo + 1.0) - start.max(lo);
            if overlap > 0.0 {
                row.push((index, overlap / scale));
            }
        }
        taps.push(row);
    }
    taps
}

/// Reference `is_valid_alpha` (`image_process.py:25-32`): 20-bin histogram of
/// alpha with bin `v * 20 / 256`; valid iff the fully-transparent and
/// fully-opaque bins each hold at least 1% of the pixels.
fn is_valid_alpha(image: &RgbaImage, bins: usize, min_ratio: f64) -> bool {
    let (width, height) = image.dimensions();
    let mut histogram = vec![0_u64; bins];
    for pixel in image.pixels() {
        let alpha = pixel[3] as usize;
        histogram[alpha * bins / 256] += 1;
    }
    let min_hist = f64::from(width) * f64::from(height) * min_ratio;
    histogram[0] as f64 >= min_hist && histogram[bins - 1] as f64 >= min_hist
}

/// Bounding box of `alpha > 0` as `(x, y, w, h)`; `None` when all alpha is 0.
fn foreground_bbox(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut x_min = u32::MAX;
    let mut y_min = u32::MAX;
    let mut x_max = 0_u32;
    let mut y_max = 0_u32;
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] > 0 {
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
        }
    }
    if x_min == u32::MAX {
        None
    } else {
        Some((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))
    }
}

/// Composite the cropped region over the background in f32 with the exact
/// reference arithmetic. The reference runs on CUDA, where scalar division is
/// a multiplication by the f32 reciprocal of 255; `alpha_composited` selects
/// the real alpha plane (valid-alpha branch) or full opacity (no alpha /
/// invalid alpha / no RMBG model provided).
fn composite_over_background(
    image: &RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    alpha_composited: bool,
    background: f32,
) -> RgbImage {
    let mut out = RgbImage::new(width, height);
    for cy in 0..height {
        for cx in 0..width {
            let pixel = image.get_pixel(x + cx, y + cy);
            let alpha = if alpha_composited {
                f32::from(pixel[3]) * COLOR_RECIPROCAL_255
            } else {
                1.0
            };
            let inverse = 1.0 - alpha;
            let blend = |value: u8| -> u8 {
                let composited =
                    f32::from(value) * COLOR_RECIPROCAL_255 * alpha + background * inverse;
                (composited * 255.0) as u8
            };
            out.put_pixel(
                cx,
                cy,
                Rgb([blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]),
            );
        }
    }
    out
}

/// Square-pad an RGB image with the quantized background color.
fn square_pad(
    image: &RgbImage,
    width: u32,
    height: u32,
    pad_lr: usize,
    pad_tb: usize,
    background: f32,
) -> RgbImage {
    let fill = (background * 255.0) as u8;
    let out_width = width + 2 * pad_lr as u32;
    let out_height = height + 2 * pad_tb as u32;
    let mut out = RgbImage::from_pixel(out_width, out_height, Rgb([fill, fill, fill]));
    for cy in 0..height {
        for cx in 0..width {
            let pixel = *image.get_pixel(cx, cy);
            out.put_pixel(pad_lr as u32 + cx, pad_tb as u32 + cy, pixel);
        }
    }
    out
}

/// HF `get_resize_output_image_size` with `size={"shortest_edge": n}` and
/// `default_to_square=False`: aspect-preserving, short edge to `shortest_edge`
/// (identity when it already equals it), long edge truncated
/// `int(shortest_edge * long / short)`.
fn shortest_edge_output_size(width: u32, height: u32, shortest_edge: u32) -> (u32, u32) {
    let (short, long) = if width <= height {
        (width, height)
    } else {
        (height, width)
    };
    if short == shortest_edge {
        return (width, height);
    }
    let new_long = (f64::from(shortest_edge) * f64::from(long) / f64::from(short)) as u32;
    if width <= height {
        (shortest_edge, new_long)
    } else {
        (new_long, shortest_edge)
    }
}

/// HF `center_crop` (`TorchvisionBackend.center_crop`): `top/left =
/// int((size - crop) / 2)` integer floor. The crop is never larger than the
/// resized image (short edge >= crop after the shortest-edge resize).
fn center_crop(image: &RgbImage, crop: u32) -> RgbImage {
    let (width, height) = image.dimensions();
    if width == crop && height == crop {
        return image.clone();
    }
    let left = ((width - crop) / 2) as usize;
    let top = ((height - crop) / 2) as usize;
    let mut out = RgbImage::new(crop, crop);
    for y in 0..crop as usize {
        for x in 0..crop as usize {
            let pixel = *image.get_pixel((left + x) as u32, (top + y) as u32);
            out.put_pixel(x as u32, y as u32, pixel);
        }
    }
    out
}

fn chw_uint8(image: &RgbImage) -> Vec<u8> {
    let plane = (image.width() as usize) * (image.height() as usize);
    let mut out = Vec::with_capacity(3 * plane);
    for channel in 0..3 {
        for pixel in image.pixels() {
            out.push(pixel[channel]);
        }
    }
    out
}

fn normalized_pixel_values(
    uint8_rgb: &[u8],
    width: u32,
    height: u32,
    stats: &NormalizationStats,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let plane = (width as usize) * (height as usize);
    let mut values = Vec::with_capacity(uint8_rgb.len());
    for channel in 0..3 {
        for index in 0..plane {
            let v = f32::from(uint8_rgb[channel * plane + index]);
            values.push((v / 255.0 - stats.mean[channel]) / stats.std[channel]);
        }
    }
    Tensor::from_vec(values, (1, 3, height as usize, width as usize), device)
}

/// Reference `rmbg()` mask preprocessing plus the post-model cleanup.
///
/// `source` is the capped RGBA image (longest side <= 2000). The reference
/// discards alpha and resizes RGB to 1024x1024 with torchvision bilinear
/// antialias, normalizes with mean 0.5 / std 1, runs BriaRMBG, resizes the
/// side-1 sigmoid back to the source size with antialias, min-max normalizes,
/// quantizes to uint8 and applies Otsu. The returned mask is the cleaned
/// binary foreground (0/1, one byte per pixel).
fn rmbg_alpha_mask(
    source: &RgbaImage,
    rmbg: &crate::runtime::RmbgModel,
    device: &Device,
) -> Result<Vec<u8>> {
    const MODEL_SIZE: u32 = 1024;
    let (width, height) = source.dimensions();

    let input = resize_rgb_bilinear_antialias_f32(source, MODEL_SIZE, MODEL_SIZE);
    let input = Tensor::from_vec(
        input,
        (1, 3, MODEL_SIZE as usize, MODEL_SIZE as usize),
        device,
    )
    .map_err(candle_error)?;
    // Reference local wrapper: TF.normalize(image, [0.5; 3], [1.0; 3]).
    let input = input.affine(1.0, -0.5).map_err(candle_error)?;

    let output = rmbg.forward(&input).map_err(candle_error)?;
    let output = output
        .flatten_all()
        .map_err(candle_error)?
        .to_vec1::<f32>()
        .map_err(candle_error)?;
    let mut alpha = resize_plane_bilinear_antialias_f32(
        &output,
        MODEL_SIZE as usize,
        MODEL_SIZE as usize,
        width as usize,
        height as usize,
    );

    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in &alpha {
        min = min.min(*value);
        max = max.max(*value);
    }
    if !(max > min) {
        return Err(Lux3dError::InvalidInput(
            "TripoSG BriaRMBG produced a constant mask",
        ));
    }
    let scale = 1.0 / (max - min);
    for value in &mut alpha {
        *value = (*value - min) * scale;
    }

    let quantized: Vec<u8> = alpha.iter().map(|value| (value * 255.0) as u8).collect();
    let threshold = otsu_threshold(&quantized);
    let binary: Vec<u8> = quantized
        .iter()
        .map(|value| if *value > threshold { 255 } else { 0 })
        .collect();

    // skimage.measure.label default connectivity for 2-D is 2 (8-neighbour);
    // remove_small_objects(min_size=200) then drops whole labels smaller than
    // 200 pixels (strictly smaller, the pre-0.26 min_size semantics).
    let (labels, counts) = connected_components_8(&binary, width as usize, height as usize);
    let mut cleaned = vec![0_u8; binary.len()];
    for (index, label) in labels.iter().enumerate() {
        if *label != 0 && counts[*label] >= 200 {
            cleaned[index] = 1;
        }
    }
    Ok(cleaned)
}

/// Composite a cleaned binary mask over the background. The reference mask is
/// an f32 0/1 plane, so the foreground keeps its exact source value (no
/// reciprocal-255 rounding); background pixels become the background value.
fn composite_binary_mask(
    image: &RgbaImage,
    mask: &[u8],
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    background: f32,
) -> RgbImage {
    let mut out = RgbImage::new(width, height);
    let background_u8 = (background * 255.0) as u8;
    for cy in 0..height {
        for cx in 0..width {
            let source_index = ((y + cy) * image.width() + (x + cx)) as usize;
            if mask[source_index] != 0 {
                let pixel = image.get_pixel(x + cx, y + cy);
                out.put_pixel(cx, cy, Rgb([pixel[0], pixel[1], pixel[2]]));
            } else {
                out.put_pixel(cx, cy, Rgb([background_u8, background_u8, background_u8]));
            }
        }
    }
    out
}

/// Bounding box of the largest 8-connected foreground component after cleanup
/// (cv2 `findContours` + `boundingRect`). Components that touch only
/// diagonally are one OpenCV contour, matching the 8-connected labelling used
/// by `skimage.measure.label`.
fn largest_foreground_bbox(mask: &[u8], width: u32, height: u32) -> Option<(u32, u32, u32, u32)> {
    let (labels, counts) = connected_components_8(mask, width as usize, height as usize);
    let mut best_label = 0_usize;
    let mut best_count = 0_usize;
    for (label, count) in counts.iter().enumerate().skip(1) {
        if *count > best_count {
            best_count = *count;
            best_label = label;
        }
    }
    if best_label == 0 {
        return None;
    }
    let mut x_min = width;
    let mut y_min = height;
    let mut x_max = 0_u32;
    let mut y_max = 0_u32;
    for y in 0..height as usize {
        for x in 0..width as usize {
            if labels[y * width as usize + x] == best_label {
                x_min = x_min.min(x as u32);
                y_min = y_min.min(y as u32);
                x_max = x_max.max(x as u32);
                y_max = y_max.max(y as u32);
            }
        }
    }
    Some((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))
}

/// 8-connected connected components with a dense label image.
fn connected_components_8(mask: &[u8], width: usize, height: usize) -> (Vec<usize>, Vec<usize>) {
    let mut labels = vec![0_usize; width * height];
    let mut counts = vec![0_usize]; // label 0 is background
    let mut stack = Vec::new();
    let mut next_label = 1_usize;
    for start in 0..width * height {
        if mask[start] == 0 || labels[start] != 0 {
            continue;
        }
        labels[start] = next_label;
        stack.push(start);
        let mut count = 0_usize;
        while let Some(index) = stack.pop() {
            count += 1;
            let x = index % width;
            let y = index / width;
            for dy in -1_i32..=1 {
                for dx in -1_i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let neighbour = ny as usize * width + nx as usize;
                    if mask[neighbour] != 0 && labels[neighbour] == 0 {
                        labels[neighbour] = next_label;
                        stack.push(neighbour);
                    }
                }
            }
        }
        counts.push(count);
        next_label += 1;
    }
    (labels, counts)
}

/// OpenCV 8-bit Otsu threshold. `cv2.threshold` then binarizes with
/// `value > threshold`.
fn otsu_threshold(values: &[u8]) -> u8 {
    let mut histogram = [0_u64; 256];
    for value in values {
        histogram[*value as usize] += 1;
    }
    let total = values.len() as f64;
    let sum: f64 = histogram
        .iter()
        .enumerate()
        .map(|(value, count)| value as f64 * *count as f64)
        .sum();

    let mut sum_background = 0.0_f64;
    let mut weight_background = 0.0_f64;
    let mut best_variance = -1.0_f64;
    let mut best_threshold = 0_u8;
    for (threshold, count) in histogram.iter().enumerate() {
        weight_background += *count as f64;
        if weight_background == 0.0 {
            continue;
        }
        let weight_foreground = total - weight_background;
        if weight_foreground == 0.0 {
            break;
        }
        sum_background += threshold as f64 * *count as f64;
        let mean_background = sum_background / weight_background;
        let mean_foreground = (sum - sum_background) / weight_foreground;
        let difference = mean_background - mean_foreground;
        let variance = weight_background * weight_foreground * difference * difference;
        if variance > best_variance {
            best_variance = variance;
            best_threshold = threshold as u8;
        }
    }
    best_threshold
}

fn candle_error(source: candle_core::Error) -> Lux3dError {
    Lux3dError::CanonicalWeightsValidation {
        message: format!("TripoSG BriaRMBG forward failed: {source}"),
    }
}

/// Bilinear antialias resize of an RGB image into planar f32 CHW in `[0, 1]`.
fn resize_rgb_bilinear_antialias_f32(
    image: &RgbaImage,
    out_width: u32,
    out_height: u32,
) -> Vec<f32> {
    let (in_width, in_height) = image.dimensions();
    let plane_in = (in_width * in_height) as usize;
    let plane_out = (out_width * out_height) as usize;
    let mut output = vec![0.0_f32; 3 * plane_out];
    for channel in 0..3 {
        let mut plane = Vec::with_capacity(plane_in);
        for pixel in image.pixels() {
            plane.push(f32::from(pixel[channel]) * COLOR_RECIPROCAL_255);
        }
        let resized = resize_plane_bilinear_antialias_f32(
            &plane,
            in_width as usize,
            in_height as usize,
            out_width as usize,
            out_height as usize,
        );
        output[channel * plane_out..(channel + 1) * plane_out].copy_from_slice(&resized);
    }
    output
}

/// Bilinear antialias resize of one f32 plane (`torch.nn.functional.interpolate`
/// / torchvision `Resize(..., antialias=True)`, `align_corners=false`).
fn resize_plane_bilinear_antialias_f32(
    source: &[f32],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<f32> {
    if in_width == out_width && in_height == out_height {
        return source.to_vec();
    }
    debug_assert_eq!(source.len(), in_width * in_height);
    let horizontal = bilinear_axis_weights(in_width, out_width);
    let vertical = bilinear_axis_weights(in_height, out_height);

    let mut intermediate = vec![0.0_f32; out_width * in_height];
    for y in 0..in_height {
        let row = y * in_width;
        for (x_out, taps) in horizontal.iter().enumerate() {
            let mut value = 0.0_f32;
            for (index, weight) in taps {
                value += source[row + index] * weight;
            }
            intermediate[y * out_width + x_out] = value;
        }
    }

    let mut output = vec![0.0_f32; out_width * out_height];
    for (y_out, taps) in vertical.iter().enumerate() {
        for x in 0..out_width {
            let mut value = 0.0_f32;
            for (index, weight) in taps {
                value += intermediate[index * out_width + x] * weight;
            }
            output[y_out * out_width + x] = value;
        }
    }
    output
}

/// Per-output-pixel `(source_index, weight)` taps for a bilinear antialias
/// resample. Weights are normalized; the support widens by the downsampling
/// ratio so every source sample contributes to the output.
fn bilinear_axis_weights(input_size: usize, output_size: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input_size as f64 / output_size as f64;
    let (support, inverse_scale) = if scale >= 1.0 {
        (scale, 1.0 / scale)
    } else {
        (1.0, 1.0)
    };
    let mut taps = Vec::with_capacity(output_size);
    for index in 0..output_size {
        let center = scale * (index as f64 + 0.5);
        let min = ((center - support + 0.5) as i64).max(0) as usize;
        let max = ((center + support + 0.5) as i64).min(input_size as i64) as usize;
        let mut weights: Vec<(usize, f32)> = Vec::with_capacity(max.saturating_sub(min));
        let mut total = 0.0_f64;
        for source_index in min..max {
            let argument = ((source_index as f64 - center + 0.5) * inverse_scale).abs();
            let weight = (1.0 - argument).max(0.0);
            if weight > 0.0 {
                weights.push((source_index, weight as f32));
                total += weight;
            }
        }
        if total > 0.0 {
            let inv_total = 1.0 / total as f32;
            for (_, weight) in &mut weights {
                *weight *= inv_total;
            }
        }
        taps.push(weights);
    }
    taps
}

// ---------------------------------------------------------------------------
// Torch-native uint8 antialias bicubic resize (aten `upsample_bicubic2d_aa`,
// `UpSampleKernel.cpp` v2.5.1 int16 fixed-point path).
//
// Separable two-pass resample over 8-bit data (horizontal first, then
// vertical), exactly like Pillow's `ImagingResample`, but with coefficients
// quantized to a measured `weights_precision` (int16 range) instead of
// Pillow's fixed 22-bit int32. Accumulation per output pixel:
// `acc = 2^(p-1) + sum(w_i * v_i)`, output `clamp(acc >> p, 0, 255)` with an
// arithmetic (floor) shift. Integer math is order-independent, so no SIMD
// reassociation can change results.
// ---------------------------------------------------------------------------

struct AxisResampleWeights {
    /// First source index per output pixel.
    mins: Vec<u32>,
    /// Tap count per output pixel.
    sizes: Vec<usize>,
    /// Fixed-point coefficients (int16 range), row stride `stride`.
    weights: Vec<i32>,
    stride: usize,
    weights_precision: u32,
}

/// Pillow-derived bicubic kernel with `a = -0.5`
/// (`HelperInterpCubic::aa_filter<double, true>`).
fn torch_cubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// `HelperInterpBase::_compute_index_ranges_weights` (antialias) plus the
/// int16 quantization of `_compute_index_ranges_int16_weights`, computed in
/// f64 like the C++ (`scalar_t = double`).
fn compute_axis_resample_weights(input_size: usize, output_size: usize) -> AxisResampleWeights {
    // `area_pixel_compute_scale<double>` with align_corners=false and no
    // explicit scale.
    let scale = input_size as f64 / output_size as f64;
    // HelperInterpCubic: interp_size = 4; support scales with `scale` when
    // downsampling.
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let stride = (support.ceil() as usize) * 2 + 1;
    let inverse_scale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };

    let mut mins = vec![0_u32; output_size];
    let mut sizes = vec![0_usize; output_size];
    let mut weights_f64 = vec![0.0_f64; output_size * stride];
    let mut weight_max = 0.0_f64;

    for i in 0..output_size {
        let center = scale * (i as f64 + 0.5);
        // C `(int64_t)` casts truncate toward zero.
        let min = ((center - support + 0.5) as i64).max(0);
        let max = ((center + support + 0.5) as i64).min(input_size as i64);
        let size = (max - min).clamp(0, stride as i64) as usize;
        mins[i] = min as u32;
        sizes[i] = size;

        let row = &mut weights_f64[i * stride..i * stride + stride];
        let mut total = 0.0_f64;
        for (j, weight) in row.iter_mut().enumerate().take(size) {
            let argument = ((j + min as usize) as f64 - center + 0.5) * inverse_scale;
            *weight = torch_cubic_filter(argument);
            total += *weight;
        }
        if total != 0.0 {
            for weight in row.iter_mut().take(size) {
                *weight /= total;
                weight_max = weight_max.max(*weight);
            }
        }
    }

    // Largest fixed-point precision whose int16 weights stay in range: first
    // `p` where `round(weight_max * 2^(p+1)) >= 2^15` (22-bit cap like PIL).
    let mut weights_precision = 22_u32;
    for p in 0..22_u32 {
        let next_value = (0.5 + weight_max * f64::from(1_i32 << (p + 1))) as i32;
        if next_value >= (1 << 15) {
            weights_precision = p;
            break;
        }
    }

    let scale_factor = f64::from(1_i32 << weights_precision);
    let mut weights = vec![0_i32; output_size * stride];
    for (weight, value) in weights.iter_mut().zip(weights_f64.iter()) {
        let scaled = value * scale_factor;
        *weight = if scaled < 0.0 {
            (-0.5 + scaled) as i32
        } else {
            (0.5 + scaled) as i32
        };
        debug_assert!(*weight >= i16::MIN as i32 && *weight <= i16::MAX as i32);
    }

    AxisResampleWeights {
        mins,
        sizes,
        weights,
        stride,
        weights_precision,
    }
}

#[inline]
fn fixed_point_round(accumulator: i32, weights_precision: u32) -> u8 {
    (accumulator >> weights_precision).clamp(0, 255) as u8
}

/// Horizontal pass: `(in_width x in_height)` interleaved RGB -> `(out_width x
/// in_height)` 8-bit intermediate (Pillow-style quantized intermediate).
fn resample_horizontal_u8(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    weights: &AxisResampleWeights,
) -> Vec<u8> {
    let initial = 1_i32 << (weights.weights_precision - 1);
    let mut output = vec![0_u8; out_width * in_height * 3];
    for y in 0..in_height {
        let row = y * in_width * 3;
        for x_out in 0..out_width {
            let min = weights.mins[x_out] as usize;
            let size = weights.sizes[x_out];
            let mut accumulator = [initial; 3];
            for k in 0..size {
                let weight = weights.weights[x_out * weights.stride + k];
                let index = row + (min + k) * 3;
                for channel in 0..3 {
                    accumulator[channel] += i32::from(source[index + channel]) * weight;
                }
            }
            let target = (y * out_width + x_out) * 3;
            for channel in 0..3 {
                output[target + channel] =
                    fixed_point_round(accumulator[channel], weights.weights_precision);
            }
        }
    }
    output
}

/// Vertical pass over the 8-bit intermediate.
fn resample_vertical_u8(
    source: &[u8],
    width: usize,
    out_height: usize,
    weights: &AxisResampleWeights,
) -> Vec<u8> {
    let initial = 1_i32 << (weights.weights_precision - 1);
    let mut output = vec![0_u8; width * out_height * 3];
    for y_out in 0..out_height {
        let min = weights.mins[y_out] as usize;
        let size = weights.sizes[y_out];
        for x in 0..width {
            let mut accumulator = [initial; 3];
            for k in 0..size {
                let weight = weights.weights[y_out * weights.stride + k];
                let index = (min + k) * width * 3 + x * 3;
                for channel in 0..3 {
                    accumulator[channel] += i32::from(source[index + channel]) * weight;
                }
            }
            let target = (y_out * width + x) * 3;
            for channel in 0..3 {
                output[target + channel] =
                    fixed_point_round(accumulator[channel], weights.weights_precision);
            }
        }
    }
    output
}

/// Torch-native uint8 antialias bicubic resize for interleaved RGB, separable
/// horizontal-then-vertical passes over quantized 8-bit intermediates.
fn resize_rgb_bicubic_antialias_u8(image: &RgbImage, out_width: u32, out_height: u32) -> RgbImage {
    let (in_width, in_height) = image.dimensions();
    if in_width == out_width && in_height == out_height {
        return image.clone();
    }

    let horizontal = compute_axis_resample_weights(in_width as usize, out_width as usize);
    let intermediate = resample_horizontal_u8(
        image.as_raw(),
        in_width as usize,
        in_height as usize,
        out_width as usize,
        &horizontal,
    );
    let vertical = compute_axis_resample_weights(in_height as usize, out_height as usize);
    let output = resample_vertical_u8(
        &intermediate,
        out_width as usize,
        out_height as usize,
        &vertical,
    );

    RgbImage::from_raw(out_width, out_height, output).expect("resized buffer dimensions")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_pad_extents_matches_reference_arithmetic() {
        let stage = TripoSgPreprocessStage::default();
        assert_eq!(stage.square_pad_extents(454, 250), (45, 147));
        assert_eq!(stage.square_pad_extents(2, 4), (1, 0));
        assert_eq!(stage.square_pad_extents(4, 4), (0, 0));
        assert_eq!(stage.square_pad_extents(250, 454), (147, 45));
    }

    #[test]
    fn shortest_edge_output_size_matches_hf() {
        assert_eq!(shortest_edge_output_size(544, 544, 256), (256, 256));
        assert_eq!(shortest_edge_output_size(545, 544, 256), (256, 256));
        assert_eq!(shortest_edge_output_size(100, 60, 256), (426, 256));
        assert_eq!(shortest_edge_output_size(60, 100, 256), (256, 426));
        assert_eq!(shortest_edge_output_size(256, 300, 256), (256, 300));
    }

    #[test]
    fn capped_dimensions_match_reference_truncation() {
        assert_eq!(resized_dimensions(2001, 1000, 2000), (2000, 999));
        assert_eq!(resized_dimensions(614, 614, 2000), (614, 614));
    }

    #[test]
    fn axis_weights_match_torch_int16_quantization() {
        // Values extracted from the validated f64/int16 model
        // (bit-exact against `torchvision.transforms.v2.functional.resize`).
        let weights = compute_axis_resample_weights(544, 256);
        assert_eq!(weights.weights_precision, 16);
        assert_eq!(weights.stride, 11);
        assert_eq!(weights.mins[0], 0);
        assert_eq!(weights.sizes[0], 5);
        assert_eq!(weights.mins[2], 1);
        assert_eq!(weights.sizes[2], 9);
        assert_eq!(weights.weights[0..5], [28201, 30003, 10594, -1769, -1493]);

        let upscale = compute_axis_resample_weights(4, 256);
        assert_eq!(upscale.weights_precision, 14);
        assert_eq!(upscale.stride, 5);
        assert_eq!(upscale.sizes[0], 2);
        assert_eq!(upscale.weights[0..2], [18424, -2040]);
    }

    #[test]
    fn uniform_image_is_resize_invariant() {
        let image = RgbImage::from_pixel(64, 64, Rgb([255, 0, 0]));
        let resized = resize_rgb_bicubic_antialias_u8(&image, 256, 256);
        for pixel in resized.pixels() {
            assert_eq!(*pixel, Rgb([255, 0, 0]));
        }
    }
}
