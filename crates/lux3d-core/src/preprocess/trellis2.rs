//! TRELLIS.2 image preprocessing (reference
//! `trellis2/pipelines/trellis2_image_to_3d.py:127-162` alpha path plus the
//! `DinoV3FeatureExtractor.__call__` resize/normalize stage, `:95-118`).
//!
//! Pipeline:
//!
//! 1. alpha usage: RGBA is honored only when the alpha channel exists and is
//!    not all 255 (`mode == 'RGBA'` plus `np.all(alpha == 255)` check). Any
//!    other input is converted to RGB and passed through the reference
//!    RMBG-2.0/BiRefNet background remover; a missing remover is an error
//!    rather than a silent opaque fallback;
//! 2. downscale so `max(H, W) <= 1024` with dimensions
//!    `(int(w * scale), int(h * scale))`, `scale = min(1, 1024 / max_size)`,
//!    Pillow `LANCZOS` (alpha included);
//! 3. foreground bbox over `alpha > 0.8 * 255` (`np.argwhere` row/col order,
//!    so the bbox is `(min x, min y, max x, max y)`);
//! 4. square crop around the bbox: `center = ((x0 + x1) / 2, (y0 + y1) / 2)`,
//!    `size = max(x1 - x0, y1 - y0)` (extent, not inclusive width),
//!    `box = (center - size // 2, ..., center + size // 2)` as floats, then
//!    Pillow `Image.crop` rounds every coordinate with Python `round()`
//!    (round-half-to-even). The crop may extend outside the image; Pillow
//!    fills out-of-bounds pixels with transparent black;
//! 5. convert to float32 `/255`, premultiply RGB by alpha, quantize with
//!    `(value * 255).astype(np.uint8)` truncation;
//! 6. resize the square crop to `resolution x resolution` (512 or 1024) with
//!    Pillow `LANCZOS` for DINOv3, then `/255` and ImageNet
//!    `(x - mean) / std` normalization in CHW layout.
//!
//! The RMBG-2.0 wrapper is reproduced exactly: Pillow BILINEAR resize to
//! 1024x1024, ImageNet normalization, `sigmoid`, truncating uint8 conversion,
//! then Pillow BICUBIC resize of the mask back to the working RGB size.
//!
//! Every resample is a bit-exact port of Pillow 12.3.0 `ImagingResample` for
//! 8-bit images (`src/libImaging/Resample.c`): double-precision coefficients
//! quantized to INT32 with `PRECISION_BITS = 32 - 8 - 2`, half-LSB
//! accumulator (`1 << (PRECISION_BITS - 1)`), arithmetic-shift extraction and
//! the clip8 saturating table. This mirrors the TripoSG decision to port the
//! reference resampler instead of using `image::imageops` (whose filters
//! differ by +-1 LSB on edge pixels and would break golden pixel parity).

use std::path::Path;

use candle_core::{Device, Tensor};
use image::{ColorType, DynamicImage, ImageReader, RgbImage, RgbaImage};

use crate::{
    Result,
    contracts::{
        ModelFamily, SpatialSize, TensorContract, Trellis2PreparedImage, Trellis2RawInput,
    },
    error::Lux3dError,
};

/// Reference downscale cap (`1024 / max_size`).
pub const TRELLIS2_MAX_INPUT_SIDE: u32 = 1024;
/// `alpha > 0.8 * 255` on uint8 alpha: `0.8 * 255 = 204.000...3`, so the
/// integer test is `alpha >= 205`.
pub const TRELLIS2_ALPHA_THRESHOLD: u8 = 205;
/// DINOv3 conditioning resolutions accepted by `get_cond` in this slice.
pub const TRELLIS2_SUPPORTED_RESOLUTIONS: [u32; 2] = [512, 1024];
/// ImageNet mean used by both DINOv3 and BiRefNet (`transforms.Normalize`).
pub const TRELLIS2_IMAGE_NET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet std used by both DINOv3 and BiRefNet (`transforms.Normalize`).
pub const TRELLIS2_IMAGE_NET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Pillow `ImagingResample` fixed-point precision for 8-bit images
/// (`#define PRECISION_BITS (32 - 8 - 2)`).
const PIL_PRECISION_BITS: u32 = 22;
/// Pillow filter kernels used by the reference preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PillowFilter {
    /// `Image.Resampling.BILINEAR`: torchvision `transforms.Resize` default.
    Bilinear,
    /// `Image.Resampling.BICUBIC`: Pillow `Image.resize()` default.
    Bicubic,
    /// `Image.Resampling.LANCZOS`: TRELLIS.2 outer downscale and DINOv3 resize.
    Lanczos,
}

impl PillowFilter {
    fn support(self) -> f64 {
        match self {
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => 3.0,
        }
    }

    fn kernel(self, x: f64) -> f64 {
        match self {
            Self::Bilinear => {
                let x = x.abs();
                if x < 1.0 { 1.0 - x } else { 0.0 }
            }
            Self::Bicubic => {
                let x = x.abs();
                if x < 1.0 {
                    ((1.5 * x - 2.5) * x) * x + 1.0
                } else if x < 2.0 {
                    (((-0.5 * x + 2.5) * x) - 4.0) * x + 2.0
                } else {
                    0.0
                }
            }
            Self::Lanczos => pil_lanczos_filter(x),
        }
    }
}

/// Geometry actually used by the reference crop, recorded for tests and
/// golden metadata (all values are the post-`round()` Pillow crop box plus
/// the raw `np.argwhere` bbox).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trellis2CropInfo {
    /// `(min x, min y, max x, max y)` of `alpha > 0.8 * 255`, inclusive.
    pub bbox: (u32, u32, u32, u32),
    /// Pillow crop box after `map(int, map(round, box))`.
    pub crop_box: (i64, i64, i64, i64),
    /// Crop output side (`crop_box.2 - crop_box.0`).
    pub crop_size: u32,
    /// Whether the alpha path was used (false = opaque input).
    pub has_alpha: bool,
    /// Source size fed to the crop (after the <=1024 downscale).
    pub downscaled_size: (u32, u32),
}

/// Background remover used by the reference RGB branch. Implemented by the
/// candle RMBG-2.0/BiRefNet runtime; tests can supply a deterministic stub.
pub trait Trellis2BackgroundRemover: Send + Sync {
    fn remove_background(&self, rgb: &RgbImage) -> Result<RgbaImage>;
}

/// Resolution-independent output of the reference `preprocess_image` step.
#[derive(Debug, Clone)]
pub struct Trellis2PreparedCrop {
    rgb: RgbImage,
    info: Trellis2CropInfo,
    original_size: SpatialSize,
}

impl Trellis2PreparedCrop {
    pub fn rgb(&self) -> &RgbImage {
        &self.rgb
    }

    pub fn crop_info(&self) -> &Trellis2CropInfo {
        &self.info
    }

    pub fn original_size(&self) -> SpatialSize {
        self.original_size
    }
}

/// TRELLIS.2 image preprocess stage. `resolution` is the DINOv3 input side
/// (512 or 1024).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trellis2Preprocess {
    resolution: u32,
}

impl Trellis2Preprocess {
    /// Stage for the given DINOv3 input resolution (512 or 1024).
    pub fn new(resolution: u32) -> Result<Self> {
        if !TRELLIS2_SUPPORTED_RESOLUTIONS.contains(&resolution) {
            return Err(Lux3dError::ContractValidation {
                message: format!(
                    "TRELLIS.2 DINOv3 resolution must be 512 or 1024, got {resolution}"
                ),
            });
        }
        Ok(Self { resolution })
    }

    /// DINOv3 input side in pixels.
    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// Decode `source` and run the full reference preprocessing pipeline.
    pub fn prepare_inputs_from_path(&self, source: &Path) -> Result<Trellis2PreparedImage> {
        let image = ImageReader::open(source)
            .map_err(|_| Lux3dError::InvalidInput("failed to open TRELLIS.2 input image"))?
            .decode()
            .map_err(|_| Lux3dError::InvalidInput("failed to decode TRELLIS.2 input image"))?;
        self.prepare_dynamic_image(&image)
    }

    /// Run the full preprocessing pipeline on an already-decoded image.
    ///
    /// RGB inputs (and RGBA inputs whose alpha is all 255) require an
    /// RMBG-2.0 background remover; this overload fails instead of silently
    /// treating the image as opaque.
    pub fn prepare_dynamic_image(&self, image: &DynamicImage) -> Result<Trellis2PreparedImage> {
        self.prepare_dynamic_image_with_rembg(image, None)
    }

    /// [`Self::prepare_dynamic_image`] with an explicit RMBG-2.0 remover.
    pub fn prepare_dynamic_image_with_rembg(
        &self,
        image: &DynamicImage,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<Trellis2PreparedImage> {
        let crop = self.prepare_crop_from_dynamic_image(image, remover)?;
        self.resize_crop(&crop)
    }

    /// Decode the source and run the reference crop once, before any
    /// resolution-specific resize. The 1024_cascade pipeline uses this to
    /// avoid running RMBG-2.0 twice.
    pub fn prepare_crop_from_path(
        &self,
        source: &Path,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<Trellis2PreparedCrop> {
        let image = ImageReader::open(source)
            .map_err(|_| Lux3dError::InvalidInput("failed to open TRELLIS.2 input image"))?
            .decode()
            .map_err(|_| Lux3dError::InvalidInput("failed to decode TRELLIS.2 input image"))?;
        self.prepare_crop_from_dynamic_image(&image, remover)
    }

    /// Run the reference crop on an already-decoded image.
    pub fn prepare_crop_from_dynamic_image(
        &self,
        image: &DynamicImage,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<Trellis2PreparedCrop> {
        if image.width() == 0 || image.height() == 0 {
            return Err(Lux3dError::InvalidInput("TRELLIS.2 input image is empty"));
        }
        // Reference checks `input.mode == 'RGBA'`; greyscale+alpha modes do
        // not take the alpha branch even though they carry an alpha channel.
        let has_alpha_channel = matches!(
            image.color(),
            ColorType::Rgba8 | ColorType::Rgba16 | ColorType::Rgba32F
        );
        let rgba = image.to_rgba8();
        self.prepare_crop_from_rgba8(&rgba, has_alpha_channel, remover)
    }

    /// Same as [`Self::prepare_dynamic_image`] but also returns the crop
    /// geometry (used by tests and golden metadata).
    pub fn prepare_rgba8_with_crop(
        &self,
        rgba: &RgbaImage,
        has_alpha_channel: bool,
    ) -> Result<(Trellis2PreparedImage, Trellis2CropInfo)> {
        self.prepare_rgba8_with_crop_and_rembg(rgba, has_alpha_channel, None)
    }

    /// [`Self::prepare_rgba8_with_crop`] with an explicit RMBG-2.0 remover.
    pub fn prepare_rgba8_with_crop_and_rembg(
        &self,
        rgba: &RgbaImage,
        has_alpha_channel: bool,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<(Trellis2PreparedImage, Trellis2CropInfo)> {
        let crop = self.prepare_crop_from_rgba8(rgba, has_alpha_channel, remover)?;
        let info = crop.info.clone();
        Ok((self.resize_crop(&crop)?, info))
    }

    /// Reference `preprocess_image` core: max-side downscale, optional
    /// RMBG-2.0, alpha bbox, square crop, premultiply. The returned RGB crop
    /// is resolution-independent; both 512 and 1024 DINOv3 inputs are
    /// produced from it with [`Self::resize_crop`].
    pub fn prepare_crop_from_rgba8(
        &self,
        rgba: &RgbaImage,
        has_alpha_channel: bool,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<Trellis2PreparedCrop> {
        if rgba.width() == 0 || rgba.height() == 0 {
            return Err(Lux3dError::InvalidInput("TRELLIS.2 input image is empty"));
        }
        let (width, height) = rgba.dimensions();
        let has_alpha = has_alpha_channel && !is_alpha_fully_opaque(rgba);

        // 2. Downscale so max(H, W) <= 1024 (LANCZOS, alpha included).
        let downscaled = downscale_to_max_side(rgba, TRELLIS2_MAX_INPUT_SIDE);
        let downscaled_size = downscaled.dimensions();

        // 3. No usable alpha: run RMBG-2.0 on the downscaled RGB image.
        let output = if has_alpha {
            downscaled
        } else {
            let rgb = rgb_from_rgba(&downscaled);
            let remover = remover.ok_or_else(|| {
                Lux3dError::InvalidInput(
                    "TRELLIS.2 input has no usable alpha channel and no RMBG-2.0 model was configured",
                )
            })?;
            let rgba = remover.remove_background(&rgb)?;
            if rgba.dimensions() != (rgb.width(), rgb.height()) {
                return Err(Lux3dError::ContractValidation {
                    message: format!(
                        "RMBG-2.0 returned {}x{} for a {}x{} input",
                        rgba.width(),
                        rgba.height(),
                        rgb.width(),
                        rgb.height()
                    ),
                });
            }
            rgba
        };

        // 4. Foreground bbox over alpha > 0.8 * 255.
        let bbox = alpha_foreground_bbox(&output).ok_or(Lux3dError::InvalidInput(
            "TRELLIS.2 alpha channel has no pixel above the 0.8 foreground threshold",
        ))?;

        // 5. Square crop box (Pillow `map(int, map(round, box))` rounding).
        let crop_box = reference_crop_box(bbox);
        let crop_w = (crop_box.2 - crop_box.0).max(0) as u32;
        let crop_h = (crop_box.3 - crop_box.1).max(0) as u32;
        if crop_w == 0 || crop_h == 0 {
            return Err(Lux3dError::ContractValidation {
                message: format!("TRELLIS.2 reference crop box {crop_box:?} is degenerate"),
            });
        }

        // 6. Crop (zero-filled outside the source) + premultiply.
        let cropped = crop_premultiplied_rgb(&output, crop_box);

        Ok(Trellis2PreparedCrop {
            original_size: SpatialSize::new(width, height),
            info: Trellis2CropInfo {
                bbox,
                crop_box,
                crop_size: crop_w,
                has_alpha,
                downscaled_size,
            },
            rgb: cropped,
        })
    }

    /// Resize a previously prepared crop to this stage's `resolution`.
    pub fn resize_crop(&self, crop: &Trellis2PreparedCrop) -> Result<Trellis2PreparedImage> {
        let cropped = &crop.rgb;
        let resized = resize_lanczos_u8::<3>(
            cropped.as_raw(),
            cropped.width() as usize,
            cropped.height() as usize,
            self.resolution as usize,
            self.resolution as usize,
        );
        let resized = RgbImage::from_raw(self.resolution, self.resolution, resized)
            .expect("resized buffer dimensions");
        Ok(Trellis2PreparedImage {
            family: ModelFamily::Trellis2,
            original_size: crop.original_size,
            target_size: SpatialSize::new(self.resolution, self.resolution),
            uint8_rgb: resized.into_raw(),
            point: None,
            pixel_values: TensorContract::new(
                "pixel_values",
                vec![1, 3, self.resolution as usize, self.resolution as usize],
            ),
        })
    }

    /// Contract-style preprocess from pixel buffers. Values are f32 in
    /// `[0, 1]`; `rgba` takes precedence over `rgb`. Buffers are quantized
    /// to RGBA8 with `(value * 255) as u8` before the reference geometry
    /// runs, which keeps a single exact core path (the reference itself is
    /// uint8-native; this mirrors the TripoSG contract-stage quantization).
    pub fn preprocess(&self, input: Trellis2RawInput) -> Result<Trellis2PreparedImage> {
        self.preprocess_with_rembg(input, None)
    }

    /// Contract-style preprocess with an explicit RMBG-2.0 remover for
    /// buffers without a usable alpha channel.
    pub fn preprocess_with_rembg(
        &self,
        input: Trellis2RawInput,
        remover: Option<&dyn Trellis2BackgroundRemover>,
    ) -> Result<Trellis2PreparedImage> {
        let plane = (input.original_size.width as usize)
            .checked_mul(input.original_size.height as usize)
            .ok_or_else(|| Lux3dError::InvalidInput("TRELLIS.2 input size overflows usize"))?;
        let (rgba, has_alpha_channel) = match (input.rgba, input.rgb) {
            (Some(values), _) => {
                if values.len() != plane {
                    return Err(Lux3dError::ContractValidation {
                        message: format!(
                            "TRELLIS.2 raw input rgba length {} does not match {}x{}",
                            values.len(),
                            input.original_size.width,
                            input.original_size.height
                        ),
                    });
                }
                let mut buf = vec![0_u8; plane * 4];
                for (index, pixel) in values.iter().enumerate() {
                    for channel in 0..4 {
                        buf[index * 4 + channel] = (pixel[channel].clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
                (buf, true)
            }
            (None, Some(values)) => {
                if values.len() != plane {
                    return Err(Lux3dError::ContractValidation {
                        message: format!(
                            "TRELLIS.2 raw input rgb length {} does not match {}x{}",
                            values.len(),
                            input.original_size.width,
                            input.original_size.height
                        ),
                    });
                }
                let mut buf = vec![0_u8; plane * 4];
                for (index, pixel) in values.iter().enumerate() {
                    for channel in 0..3 {
                        buf[index * 4 + channel] = (pixel[channel].clamp(0.0, 1.0) * 255.0) as u8;
                    }
                    buf[index * 4 + 3] = 255;
                }
                (buf, false)
            }
            (None, None) => {
                return Err(Lux3dError::InvalidInput(
                    "TRELLIS.2 raw input requires rgb or rgba values",
                ));
            }
        };
        let image =
            RgbaImage::from_raw(input.original_size.width, input.original_size.height, rgba)
                .ok_or_else(|| {
                    Lux3dError::InvalidInput("TRELLIS.2 raw input buffer size mismatch")
                })?;
        let crop = self.prepare_crop_from_rgba8(&image, has_alpha_channel, remover)?;
        self.resize_crop(&crop)
    }

    /// `[1, 3, res, res]` f32 tensor with `(v / 255 - mean) / std` ImageNet
    /// normalization of the prepared uint8 RGB pixels (the exact
    /// `DinoV3FeatureExtractor.__call__` input).
    pub fn normalized_pixel_values(
        &self,
        prepared: &Trellis2PreparedImage,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        normalized_pixel_values_from_u8(&prepared.uint8_rgb, self.resolution, device)
    }
}

impl crate::contracts::PreprocessStage<Trellis2RawInput, Trellis2PreparedImage>
    for Trellis2Preprocess
{
    fn preprocess(&self, input: Trellis2RawInput) -> Result<Trellis2PreparedImage> {
        Trellis2Preprocess::preprocess(self, input)
    }
}

/// Normalize `uint8_rgb` (HWC, `res x res x 3`) into the `[1, 3, res, res]`
/// f32 DINOv3 input tensor. Exposed for tests that start from the golden
/// uint8 image.
pub fn normalized_pixel_values_from_u8(
    uint8_rgb: &[u8],
    resolution: u32,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let plane = (resolution as usize) * (resolution as usize);
    if uint8_rgb.len() != plane * 3 {
        candle_core::bail!(
            "expected {} uint8 RGB values for {}x{}, got {}",
            plane * 3,
            resolution,
            resolution,
            uint8_rgb.len()
        );
    }
    let mut values = Vec::with_capacity(plane * 3);
    for channel in 0..3 {
        for index in 0..plane {
            let v = f32::from(uint8_rgb[index * 3 + channel]);
            values.push(
                (v / 255.0 - TRELLIS2_IMAGE_NET_MEAN[channel]) / TRELLIS2_IMAGE_NET_STD[channel],
            );
        }
    }
    Tensor::from_vec(
        values,
        (1, 3, resolution as usize, resolution as usize),
        device,
    )
}

/// True when every alpha byte is 255 (reference `np.all(alpha == 255)`).
fn is_alpha_fully_opaque(image: &RgbaImage) -> bool {
    image.pixels().all(|pixel| pixel[3] == 255)
}

/// `Image.convert("RGB")` on the downscaled RGBA buffer.
fn rgb_from_rgba(image: &RgbaImage) -> RgbImage {
    let mut out = vec![0_u8; image.width() as usize * image.height() as usize * 3];
    for (index, pixel) in image.pixels().enumerate() {
        out[index * 3] = pixel[0];
        out[index * 3 + 1] = pixel[1];
        out[index * 3 + 2] = pixel[2];
    }
    RgbImage::from_raw(image.width(), image.height(), out).expect("RGB buffer dimensions")
}

/// Bounding box of `alpha >= TRELLIS2_ALPHA_THRESHOLD` as inclusive
/// `(min x, min y, max x, max y)` (`np.argwhere` extents); `None` when no
/// pixel passes the threshold.
pub fn alpha_foreground_bbox(image: &RgbaImage) -> Option<(u32, u32, u32, u32)> {
    let mut x_min = u32::MAX;
    let mut y_min = u32::MAX;
    let mut x_max = 0_u32;
    let mut y_max = 0_u32;
    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel[3] >= TRELLIS2_ALPHA_THRESHOLD {
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
        }
    }
    if x_min == u32::MAX {
        None
    } else {
        Some((x_min, y_min, x_max, y_max))
    }
}

/// Reference square-crop box arithmetic
/// (`trellis2_image_to_3d.py:152-158`) followed by the Pillow
/// `map(int, map(round, box))` rounding. `bbox` is the inclusive
/// `(min x, min y, max x, max y)` extents.
pub fn reference_crop_box(bbox: (u32, u32, u32, u32)) -> (i64, i64, i64, i64) {
    let (x_min, y_min, x_max, y_max) = bbox;
    // center = ((x0 + x1) / 2, (y0 + y1) / 2) as f64
    let center_x = (f64::from(x_min) + f64::from(x_max)) / 2.0;
    let center_y = (f64::from(y_min) + f64::from(y_max)) / 2.0;
    // size = int(max(x1 - x0, y1 - y0) * 1) — extents, not inclusive widths.
    let size = (x_max - x_min).max(y_max - y_min) as i64;
    let half = size / 2; // Python `size // 2`, non-negative
    (
        python_round(center_x - half as f64),
        python_round(center_y - half as f64),
        python_round(center_x + half as f64),
        python_round(center_y + half as f64),
    )
}

/// Python 3 `round()`: round-half-to-even on exact .5 ties.
pub fn python_round(value: f64) -> i64 {
    let floor = value.floor();
    let diff = value - floor;
    if diff > 0.5 {
        floor as i64 + 1
    } else if diff < 0.5 {
        floor as i64
    } else if (floor as i64) % 2 == 0 {
        floor as i64
    } else {
        floor as i64 + 1
    }
}

/// Reference downscale: identity while `max(w, h) <= max_side`, otherwise
/// `(int(w * scale), int(h * scale))` with `scale = max_side / max(w, h)`,
/// resampled with the Pillow LANCZOS port (alpha included).
fn downscale_to_max_side(image: &RgbaImage, max_side: u32) -> RgbaImage {
    let (width, height) = image.dimensions();
    let longest = width.max(height);
    if longest <= max_side {
        return image.clone();
    }
    let scale = f64::from(max_side) / f64::from(longest);
    // Python `int(input.width * scale)` truncation.
    let new_width = ((f64::from(width) * scale) as u32).max(1);
    let new_height = ((f64::from(height) * scale) as u32).max(1);
    let resized = resize_lanczos_u8::<4>(
        image.as_raw(),
        width as usize,
        height as usize,
        new_width as usize,
        new_height as usize,
    );
    RgbaImage::from_raw(new_width, new_height, resized).expect("resized buffer dimensions")
}

/// Crop the (possibly out-of-bounds) Pillow box and premultiply RGB by alpha
/// exactly like the reference numpy path. Public because this is exactly the
/// intermediate artifact the reference `preprocess_image` returns (a square
/// premultiplied RGB image); golden tests compare it directly. See
/// [`crop_premultiplied_rgb`] for the arithmetic notes.
pub fn square_crop_premultiplied(image: &RgbaImage, box_xyxy: (i64, i64, i64, i64)) -> RgbImage {
    crop_premultiplied_rgb(image, box_xyxy)
}

/// Crop the (possibly out-of-bounds) Pillow box and premultiply RGB by alpha
/// exactly like the reference numpy path:
/// `((rgb / 255) * (alpha / 255) * 255).astype(np.uint8)` with f32 division
/// and multiplication.
fn crop_premultiplied_rgb(image: &RgbaImage, box_xyxy: (i64, i64, i64, i64)) -> RgbImage {
    let (x0, y0, x1, y1) = box_xyxy;
    let out_w = (x1 - x0).max(0) as u32;
    let out_h = (y1 - y0).max(0) as u32;
    let (in_w, in_h) = image.dimensions();
    let mut out = vec![0_u8; (out_w as usize) * (out_h as usize) * 3];
    for cy in 0..out_h as i64 {
        let source_y = y0 + cy;
        for cx in 0..out_w as i64 {
            let source_x = x0 + cx;
            let target = (cy as u32 as usize * out_w as usize + cx as u32 as usize) * 3;
            let in_bounds =
                source_x >= 0 && source_y >= 0 && source_x < in_w as i64 && source_y < in_h as i64;
            let pixel = if in_bounds {
                image.get_pixel(source_x as u32, source_y as u32)
            } else {
                // Pillow crop fills out-of-bounds with transparent black.
                continue;
            };
            for channel in 0..3 {
                let rgb = f32::from(pixel[channel]) / 255.0;
                let alpha = f32::from(pixel[3]) / 255.0;
                out[target + channel] = (rgb * alpha * 255.0) as u8;
            }
        }
    }
    RgbImage::from_raw(out_w, out_h, out).expect("crop buffer dimensions")
}

// ---------------------------------------------------------------------------
// Pillow 12.3.0 `ImagingResample` 8-bit port (LANCZOS).
// Source: Pillow `src/libImaging/Resample.c` (sinc/lanczos filters,
// precompute_coeffs, normalize_coeffs_8bpc, ImagingResampleHorizontal_8bpc,
// ImagingResampleVertical_8bpc, clip8).
// ---------------------------------------------------------------------------

/// Pillow `sinc_filter`: `sin(pi * x) / (pi * x)`, 1.0 at x = 0.
fn pil_sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let scaled = x * std::f64::consts::PI;
    scaled.sin() / scaled
}

/// Pillow `lanczos_filter` (truncated sinc), support 3.0, domain `[-3, 3)`.
fn pil_lanczos_filter(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        pil_sinc(x) * pil_sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Pillow `clip8` lookup semantics for the reachable range
/// (`clip8_lookups[in >> 22]` covers -640..639 and saturates at 0/255).
#[inline]
fn pil_clip8(accumulator: i64) -> u8 {
    (accumulator >> PIL_PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` for a full-image
/// resize (`in0 = 0`, `in1 = in_size`).
///
/// Returns the per-output-pixel `(xmin, count)` bounds and the INT32
/// quantized coefficients (row stride `ksize`).
fn pil_resample_coeffs(
    in_size: usize,
    out_size: usize,
    filter: PillowFilter,
) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    // filterscale = scale = (in1 - in0) / outSize, floored at 1.0
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let inv_filterscale = 1.0 / filterscale;

    let mut bounds = Vec::with_capacity(out_size);
    let mut coefficients = vec![0.0_f64; out_size * ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        // C `(int)` casts truncate toward zero.
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let count = (xmax - xmin).max(0) as usize;
        let row = &mut coefficients[xx * ksize..xx * ksize + ksize];
        let mut weight_sum = 0.0_f64;
        for (x, weight) in row.iter_mut().enumerate().take(count) {
            *weight = filter.kernel((x as f64 + xmin as f64 - center + 0.5) * inv_filterscale);
            weight_sum += *weight;
        }
        if weight_sum != 0.0 {
            for weight in row.iter_mut().take(count) {
                *weight /= weight_sum;
            }
        }
        bounds.push((xmin as usize, count));
    }

    // normalize_coeffs_8bpc: INT32 quantization with PRECISION_BITS.
    let scale_factor = (1_u64 << PIL_PRECISION_BITS) as f64;
    let quantized = coefficients
        .iter()
        .map(|&weight| {
            if weight < 0.0 {
                (-0.5 + weight * scale_factor) as i32
            } else {
                (0.5 + weight * scale_factor) as i32
            }
        })
        .collect();
    (ksize, bounds, quantized)
}

/// Pillow `ImagingResampleHorizontal_8bpc` for `C` interleaved channels.
fn pil_resample_horizontal<const C: usize>(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    ksize: usize,
    bounds: &[(usize, usize)],
    coefficients: &[i32],
) -> Vec<u8> {
    let mut output = vec![0_u8; out_width * in_height * C];
    for y in 0..in_height {
        for xx in 0..out_width {
            let (xmin, count) = bounds[xx];
            let weights = &coefficients[xx * ksize..xx * ksize + count];
            let mut accumulators = [1_i64 << (PIL_PRECISION_BITS - 1); C];
            for (x, &weight) in weights.iter().enumerate() {
                let base = (y * in_width + xmin + x) * C;
                for (channel, accumulator) in accumulators.iter_mut().enumerate() {
                    *accumulator += i64::from(source[base + channel]) * i64::from(weight);
                }
            }
            let target = (y * out_width + xx) * C;
            for (channel, accumulator) in accumulators.iter().enumerate() {
                output[target + channel] = pil_clip8(*accumulator);
            }
        }
    }
    output
}

/// Pillow `ImagingResampleVertical_8bpc` for `C` interleaved channels.
/// Row indices come from the per-row bounds, so only the source width and
/// the output height are needed.
fn pil_resample_vertical<const C: usize>(
    source: &[u8],
    width: usize,
    out_height: usize,
    ksize: usize,
    bounds: &[(usize, usize)],
    coefficients: &[i32],
) -> Vec<u8> {
    let mut output = vec![0_u8; width * out_height * C];
    for yy in 0..out_height {
        let (ymin, count) = bounds[yy];
        let weights = &coefficients[yy * ksize..yy * ksize + count];
        for x in 0..width {
            let mut accumulators = [1_i64 << (PIL_PRECISION_BITS - 1); C];
            for (y, &weight) in weights.iter().enumerate() {
                let base = ((ymin + y) * width + x) * C;
                for (channel, accumulator) in accumulators.iter_mut().enumerate() {
                    *accumulator += i64::from(source[base + channel]) * i64::from(weight);
                }
            }
            let target = (yy * width + x) * C;
            for (channel, accumulator) in accumulators.iter().enumerate() {
                output[target + channel] = pil_clip8(*accumulator);
            }
        }
    }
    output
}

/// Pillow `ImagingResample` for 8-bit `C`-channel images and the identity
/// box: horizontal pass first (skipped when the width is unchanged, matching
/// `need_horizontal`), then the vertical pass (skipped when the height is
/// unchanged).
fn resize_pil_u8<const C: usize>(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
    filter: PillowFilter,
) -> Vec<u8> {
    if in_width == out_width && in_height == out_height {
        return source.to_vec();
    }
    let mut current = source.to_vec();
    if out_width != in_width {
        let (ksize, bounds, coefficients) = pil_resample_coeffs(in_width, out_width, filter);
        current = pil_resample_horizontal::<C>(
            &current,
            in_width,
            in_height,
            out_width,
            ksize,
            &bounds,
            &coefficients,
        );
    }
    if out_height != in_height {
        let (ksize, bounds, coefficients) = pil_resample_coeffs(in_height, out_height, filter);
        current = pil_resample_vertical::<C>(
            &current,
            out_width,
            out_height,
            ksize,
            &bounds,
            &coefficients,
        );
    }
    current
}

/// Pillow LANCZOS resize (reference downscale and DINOv3 input).
pub(crate) fn resize_lanczos_u8<const C: usize>(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    resize_pil_u8::<C>(
        source,
        in_width,
        in_height,
        out_width,
        out_height,
        PillowFilter::Lanczos,
    )
}

/// Pillow BILINEAR resize (torchvision `transforms.Resize((1024, 1024))`).
pub(crate) fn resize_bilinear_u8<const C: usize>(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    resize_pil_u8::<C>(
        source,
        in_width,
        in_height,
        out_width,
        out_height,
        PillowFilter::Bilinear,
    )
}

/// Pillow BICUBIC resize (Pillow `Image.resize()` default for the mask).
pub(crate) fn resize_bicubic_u8<const C: usize>(
    source: &[u8],
    in_width: usize,
    in_height: usize,
    out_width: usize,
    out_height: usize,
) -> Vec<u8> {
    resize_pil_u8::<C>(
        source,
        in_width,
        in_height,
        out_width,
        out_height,
        PillowFilter::Bicubic,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, Rgba};

    fn rgba_image(width: u32, height: u32, fill: impl Fn(u32, u32) -> [u8; 4]) -> RgbaImage {
        let mut image = RgbaImage::new(width, height);
        for y in 0..height {
            for x in 0..width {
                image.put_pixel(x, y, Rgba(fill(x, y)));
            }
        }
        image
    }

    fn rgb(x: u8, y: u8, z: u8) -> Rgb<u8> {
        Rgb([x, y, z])
    }

    #[test]
    fn python_round_is_round_half_to_even() {
        assert_eq!(python_round(0.5), 0);
        assert_eq!(python_round(1.5), 2);
        assert_eq!(python_round(2.5), 2);
        assert_eq!(python_round(98.5), 98);
        assert_eq!(python_round(99.5), 100);
        assert_eq!(python_round(-0.5), 0);
        assert_eq!(python_round(-1.5), -2);
        assert_eq!(python_round(7.4), 7);
        assert_eq!(python_round(7.6), 8);
        assert_eq!(python_round(-7.5), -8);
    }

    #[test]
    fn crop_box_matches_reference_arithmetic() {
        // Fully opaque 100x100: bbox spans extents (0,0,99,99), size 99,
        // center (49.5, 49.5), half 49 -> box (0.5, 0.5, 98.5, 98.5) which
        // banker's-rounds to (0, 0, 98, 98) — a 98x98 crop, exactly like
        // Pillow's `map(round, box)`.
        assert_eq!(reference_crop_box((0, 0, 99, 99)), (0, 0, 98, 98));
        // Integer center keeps the full size: box (0, 0, 98, 98).
        assert_eq!(reference_crop_box((0, 0, 98, 98)), (0, 0, 98, 98));
        // Mixed: x center integer, y center .5 — only y endpoints shift.
        // (2,5,40,50): center (21.0, 27.5), size 45, half 22 ->
        // x (-1.0, 43.0), y (5.5, 49.5) -> (-1, 6, 43, 50).
        assert_eq!(reference_crop_box((2, 5, 40, 50)), (-1, 6, 43, 50));
        // .5 center on both axes, odd size: (1,1,6,6) -> center (3.5, 3.5),
        // size 5, half 2 -> endpoints (1.5, 5.5) round to (2, 6): the crop
        // shrinks to the even 2*half = 4 px side.
        assert_eq!(reference_crop_box((1, 1, 6, 6)), (2, 2, 6, 6));
        // Single pixel: size 0 -> degenerate zero-size box (rejected later).
        assert_eq!(reference_crop_box((4, 7, 4, 7)), (4, 7, 4, 7));
    }

    #[test]
    fn alpha_bbox_extents_match_argwhere() {
        let image = rgba_image(8, 8, |x, y| {
            if (3..6).contains(&x) && (2..5).contains(&y) && !(x == 5 && y == 2) {
                [255, 0, 0, 255]
            } else {
                [0, 0, 0, 0]
            }
        });
        // Foreground covers x 3..5, y 2..4 except (5, 2).
        assert_eq!(alpha_foreground_bbox(&image), Some((3, 2, 5, 4)));
        // A pixel at 204 is below the 0.8*255 threshold.
        let image_204 = rgba_image(4, 4, |_, _| [255, 255, 255, 204]);
        assert_eq!(alpha_foreground_bbox(&image_204), None);
        let image_205 = rgba_image(4, 4, |_, _| [255, 255, 255, 205]);
        assert_eq!(alpha_foreground_bbox(&image_205), Some((0, 0, 3, 3)));
    }

    #[test]
    fn premultiply_follows_reference_f32_arithmetic() {
        // alpha 128 / rgb 255: (255/255) * (128/255) * 255 with the exact
        // f32 rounding sequence replicated from the numpy path.
        let image = rgba_image(1, 1, |_, _| [255, 128, 0, 128]);
        let cropped = crop_premultiplied_rgb(&image, (0, 0, 1, 1));
        let expected_r = (1.0_f32 * (128.0_f32 / 255.0) * 255.0) as u8;
        let expected_g = (f32::from(128_u8) / 255.0 * (f32::from(128_u8) / 255.0) * 255.0) as u8;
        assert_eq!(*cropped.get_pixel(0, 0), rgb(expected_r, expected_g, 0));
    }

    #[test]
    fn crop_extends_outside_image_with_transparent_black_fill() {
        // 2x2 image, crop box (-1, -1, 3, 3): out-of-bounds pixels are 0.
        let image = rgba_image(2, 2, |x, y| [10 * x as u8 + 1, 20 * y as u8 + 2, 3, 255]);
        let cropped = crop_premultiplied_rgb(&image, (-1, -1, 3, 3));
        assert_eq!(cropped.width(), 4);
        assert_eq!(cropped.height(), 4);
        assert_eq!(*cropped.get_pixel(0, 0), rgb(0, 0, 0));
        let premul = |v: u8| (f32::from(v) / 255.0 * (255.0_f32 / 255.0) * 255.0) as u8;
        assert_eq!(
            *cropped.get_pixel(1, 1),
            rgb(premul(1), premul(2), premul(3))
        );
        assert_eq!(
            *cropped.get_pixel(2, 2),
            rgb(premul(11), premul(22), premul(3))
        );
        assert_eq!(*cropped.get_pixel(3, 3), rgb(0, 0, 0));
    }

    #[test]
    fn lanczos_resize_dimensions_and_uniform_invariance() {
        // Uniform images are resize-invariant (weights sum to 1 after
        // normalization; half-LSB accumulator + truncation reproduces the
        // source value).
        let uniform = vec![173_u8; 37 * 23 * 3];
        let resized = resize_lanczos_u8::<3>(&uniform, 37, 23, 61, 29);
        assert_eq!(resized.len(), 61 * 29 * 3);
        assert!(resized.iter().all(|&value| value == 173));

        let rgba = vec![200_u8; 40 * 40 * 4];
        let downscaled = resize_lanczos_u8::<4>(&rgba, 40, 40, 17, 17);
        assert_eq!(downscaled.len(), 17 * 17 * 4);
        assert!(downscaled.iter().all(|&value| value == 200));
    }

    #[test]
    fn downscale_keeps_identity_below_cap_and_truncates_dimensions() {
        let image = rgba_image(100, 60, |_, _| [1, 2, 3, 255]);
        let unchanged = downscale_to_max_side(&image, 1024);
        assert_eq!(unchanged.dimensions(), (100, 60));
        let downscaled = downscale_to_max_side(&rgba_image(614, 614, |_, _| [9, 9, 9, 255]), 1024);
        assert_eq!(downscaled.dimensions(), (614, 614));
        // 2000x1000 -> scale 1024/2000 = 0.512 -> (1024, 512) exactly.
        let downscaled_wide =
            downscale_to_max_side(&rgba_image(2000, 1000, |_, _| [0, 0, 0, 255]), 1024);
        assert_eq!(downscaled_wide.dimensions(), (1024, 512));
    }

    #[test]
    fn opaque_input_full_pipeline_covers_reference_bankers_quirk() {
        // An opaque 100x100 input must produce a 98x98 crop then a square
        // resize, exercising the reference banker's-rounding behavior end to
        // end through `prepare_rgba8`.
        let stage = Trellis2Preprocess::new(512).expect("stage");
        let image = rgba_image(100, 100, |x, y| [(x % 255) as u8, (y % 255) as u8, 7, 255]);
        let (prepared, crop) = stage
            .prepare_rgba8_with_crop_and_rembg(&image, false, Some(&OpaqueRemover))
            .expect("prepared");
        assert!(!crop.has_alpha);
        assert_eq!(crop.bbox, (0, 0, 99, 99));
        assert_eq!(crop.crop_box, (0, 0, 98, 98));
        assert_eq!(crop.crop_size, 98);
        assert_eq!(prepared.target_size, SpatialSize::new(512, 512));
        assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);
        assert_eq!(prepared.family, ModelFamily::Trellis2);
    }

    #[test]
    fn alpha_input_crops_to_foreground_square() {
        // 40x30 image; opaque rect x 10..25, y 5..24 (alpha 255 inside).
        let stage = Trellis2Preprocess::new(512).expect("stage");
        let image = rgba_image(40, 30, |x, y| {
            if (10..25).contains(&x) && (5..24).contains(&y) {
                [200, 100, 50, 255]
            } else {
                [0, 0, 0, 0]
            }
        });
        let (prepared, crop) = stage
            .prepare_rgba8_with_crop(&image, true)
            .expect("prepared");
        assert!(crop.has_alpha);
        assert_eq!(crop.bbox, (10, 5, 24, 23));
        // size = max(14, 18) = 18; center (17.0, 14.0); box (8, 5, 26, 23).
        assert_eq!(crop.crop_box, (8, 5, 26, 23));
        assert_eq!(crop.crop_size, 18);
        assert_eq!(prepared.target_size, SpatialSize::new(512, 512));
    }

    struct OpaqueRemover;

    impl Trellis2BackgroundRemover for OpaqueRemover {
        fn remove_background(&self, rgb: &RgbImage) -> Result<RgbaImage> {
            let mut out = RgbaImage::new(rgb.width(), rgb.height());
            for (x, y, pixel) in rgb.enumerate_pixels() {
                out.put_pixel(x, y, image::Rgba([pixel[0], pixel[1], pixel[2], 255]));
            }
            Ok(out)
        }
    }

    #[test]
    fn contract_preprocess_quantizes_and_rejects_bad_inputs() {
        let stage = Trellis2Preprocess::new(512).expect("stage");
        // 3x3 opaque: bbox (0,0,2,2), size 2, center (1,1) -> box (0,0,2,2),
        // a valid 2x2 crop (a 2x2 input would degenerate to a 0-size crop,
        // exactly like the reference arithmetic).
        let raw = Trellis2RawInput {
            original_size: SpatialSize::new(3, 3),
            rgba: Some(vec![[1.0, 0.5, 0.0, 1.0]; 9]),
            rgb: None,
        };
        let prepared = stage
            .preprocess_with_rembg(raw, Some(&OpaqueRemover))
            .expect("prepared");
        assert_eq!(prepared.uint8_rgb.len(), 512 * 512 * 3);
        assert_eq!(prepared.pixel_values.dims, vec![1, 3, 512, 512]);
        let missing = Trellis2RawInput {
            original_size: SpatialSize::new(3, 3),
            rgba: None,
            rgb: None,
        };
        assert!(stage.preprocess(missing).is_err());
        assert!(Trellis2Preprocess::new(256).is_err());
    }

    #[test]
    fn normalized_pixel_values_match_imagenet_formula() {
        let resolution: usize = 2;
        let mut pixels = vec![0_u8; resolution * resolution * 3];
        pixels[0] = 255; // R of pixel (0, 0)
        pixels[1] = 128; // G
        pixels[2] = 0; // B
        let tensor = normalized_pixel_values_from_u8(&pixels, resolution as u32, &Device::Cpu)
            .expect("tensor");
        assert_eq!(tensor.dims(), &[1, 3, 2, 2]);
        let values = tensor
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("values");
        let expected_r = (1.0 - TRELLIS2_IMAGE_NET_MEAN[0]) / TRELLIS2_IMAGE_NET_STD[0];
        assert!((values[0] - expected_r).abs() < 1e-6);
        // CHW planes: G and B of pixel (0, 0) live at index 4 and 8 (2x2 plane).
        let expected_g =
            (f32::from(128_u8) / 255.0 - TRELLIS2_IMAGE_NET_MEAN[1]) / TRELLIS2_IMAGE_NET_STD[1];
        assert!((values[4] - expected_g).abs() < 1e-6);
        let expected_b = (0.0 - TRELLIS2_IMAGE_NET_MEAN[2]) / TRELLIS2_IMAGE_NET_STD[2];
        assert!((values[8] - expected_b).abs() < 1e-6);
    }

    #[test]
    fn pillow_bilinear_and_bicubic_match_reference_bytes() {
        let golden = Path::new("../../.swarm/results/trellis2-rmbg2-golden/preprocess_golden.npz");
        if !golden.exists() {
            eprintln!("skipping Pillow resampler parity: golden fixture missing");
            return;
        }
        let mut npz = ndarray_npy::NpzReader::new(std::fs::File::open(golden).unwrap()).unwrap();
        let source = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("downscaled_rgb")
            .expect("downscaled_rgb");
        let expected = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix3>("rembg_input_rgb")
            .expect("rembg_input_rgb");
        let got = resize_bilinear_u8::<3>(
            source.as_slice().unwrap(),
            source.shape()[1],
            source.shape()[0],
            expected.shape()[1],
            expected.shape()[0],
        );
        assert_eq!(got, expected.as_slice().unwrap(), "Pillow BILINEAR drift");

        let pred = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix2>("rembg_pred_u8_1024")
            .expect("rembg_pred_u8_1024");
        let expected_mask = npz
            .by_name::<ndarray::OwnedRepr<u8>, ndarray::Ix2>("rembg_mask_u8")
            .expect("rembg_mask_u8");
        let got = resize_bicubic_u8::<1>(
            pred.as_slice().unwrap(),
            pred.shape()[1],
            pred.shape()[0],
            expected_mask.shape()[1],
            expected_mask.shape()[0],
        );
        assert_eq!(
            got,
            expected_mask.as_slice().unwrap(),
            "Pillow BICUBIC drift"
        );
    }
}
