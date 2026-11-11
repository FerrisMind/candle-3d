use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_core::{Device, Tensor};
use image::{Rgb, RgbImage};

use crate::{
    Result,
    contracts::{
        ModelFamily, PreprocessStage, RgbRange, SpatialSize, VggtPreparedBatch, VggtRawInput,
    },
    error::Lux3dError,
};

const TARGET_WIDTH: u32 = 518;
const PATCH_MULTIPLE: u32 = 14;

#[derive(Debug, Clone)]
pub struct VggtPreparedInputs {
    pub interval: usize,
    pub original_size: SpatialSize,
    pub target_size: SpatialSize,
    /// `[S, 3, H, W]` f32, range `[0, 1]`.
    pub rgb_frames: Tensor,
    /// `[1, S, 3, H, W]` f32, ImageNet-normalized.
    pub normalized_frames: Tensor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VggtPreprocessStage {
    target_width: u32,
    patch_multiple: u32,
}

impl Default for VggtPreprocessStage {
    fn default() -> Self {
        Self {
            target_width: TARGET_WIDTH,
            patch_multiple: PATCH_MULTIPLE,
        }
    }
}

impl VggtPreprocessStage {
    /// Load a directory of RGB frames or a video file (`.mp4`/`.mov`/`.avi`/
    /// `.mkv`), then apply the VGGT `mode="crop"` preprocessing recipe from
    /// `vggt.utils.load_fn`.
    pub fn prepare_inputs_from_path(
        &self,
        source: &Path,
        interval: Option<usize>,
        device: &Device,
    ) -> Result<VggtPreparedInputs> {
        let interval = interval.unwrap_or_else(|| default_interval_for(source));
        if interval == 0 {
            return Err(Lux3dError::InvalidInput("interval must be at least 1"));
        }

        let sampled = load_sampled_frames(source, interval)?;
        let sampled_paths = &sampled.paths;

        let first = load_rgb_frame(&sampled_paths[0])?;
        let original_size = SpatialSize::new(first.width(), first.height());

        let mut shapes = Vec::with_capacity(sampled_paths.len());
        let mut frames = Vec::with_capacity(sampled_paths.len());
        for path in sampled_paths {
            let image = load_rgb_frame(path)?;
            let (shape, chw) = self.resize_and_crop_frame(&image)?;
            shapes.push(shape);
            frames.push(chw);
        }

        let max_height = shapes.iter().map(|shape| shape.height).max().unwrap_or(0);
        let max_width = shapes.iter().map(|shape| shape.width).max().unwrap_or(0);
        if max_height == 0 || max_width == 0 {
            return Err(Lux3dError::InvalidInput(
                "VGGT preprocessing produced an empty frame",
            ));
        }

        let plane = (max_height as usize) * (max_width as usize);
        let mut flattened = Vec::with_capacity(frames.len() * 3 * plane);
        for (frame, shape) in frames.into_iter().zip(shapes) {
            flattened.extend_from_slice(&pad_chw_to_max(&frame, shape, max_height, max_width));
        }

        let frame_count = sampled_paths.len();
        let rgb_frames = Tensor::from_vec(
            flattened,
            (frame_count, 3, max_height as usize, max_width as usize),
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("failed to materialize VGGT RGB tensor: {source}"),
        })?;

        let mean = Tensor::from_vec(vec![0.485_f32, 0.456, 0.406], (1, 1, 3, 1, 1), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize VGGT mean tensor: {source}"),
            })?;
        let std = Tensor::from_vec(vec![0.229_f32, 0.224, 0.225], (1, 1, 3, 1, 1), device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to materialize VGGT std tensor: {source}"),
            })?;
        let normalized_frames = rgb_frames
            .unsqueeze(0)
            .and_then(|tensor| tensor.broadcast_sub(&mean))
            .and_then(|tensor| tensor.broadcast_div(&std))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to normalise VGGT RGB frames: {source}"),
            })?;

        Ok(VggtPreparedInputs {
            interval,
            original_size,
            target_size: SpatialSize::new(max_width, max_height),
            rgb_frames,
            normalized_frames,
        })
    }

    /// Reference crop-mode sizing *before* the center crop: width is fixed to
    /// `target_width`, height keeps the aspect ratio and is rounded to
    /// `patch_multiple` with Python's ties-to-even rounding. The prepared frame
    /// is then center-cropped to `target_width` rows, so its final size never
    /// exceeds `target_width` x `target_width`.
    pub fn target_size_for(&self, original: SpatialSize, target_width: u32) -> Result<SpatialSize> {
        scaled_crop_size(
            original.width,
            original.height,
            target_width,
            self.patch_multiple,
        )
    }

    fn resize_and_crop_frame(&self, image: &RgbImage) -> Result<(SpatialSize, Vec<f32>)> {
        let (width, height) = image.dimensions();
        let target = scaled_crop_size(width, height, self.target_width, self.patch_multiple)?;
        let resized = resize_bicubic_pillow(image, target.width, target.height);

        // Crop mode keeps the middle `target_width` rows when the resized
        // height exceeds the target width; width already equals target_width.
        let out_height = resized.height().min(self.target_width);
        let start_y = (resized.height() - out_height) / 2;
        let plane = (out_height as usize) * (self.target_width as usize);
        let mut chw = vec![0.0_f32; 3 * plane];

        for y in 0..out_height {
            let source_y = start_y + y;
            for x in 0..self.target_width {
                let pixel = resized.get_pixel(x, source_y);
                for channel in 0..3 {
                    chw[channel * plane
                        + (y as usize) * (self.target_width as usize)
                        + x as usize] = f32::from(pixel[channel]) / 255.0;
                }
            }
        }

        Ok((SpatialSize::new(self.target_width, out_height), chw))
    }
}

impl PreprocessStage<VggtRawInput, VggtPreparedBatch> for VggtPreprocessStage {
    fn preprocess(&self, input: VggtRawInput) -> Result<VggtPreparedBatch> {
        if input.interval == 0 {
            return Err(Lux3dError::InvalidInput("interval must be at least 1"));
        }
        let target_size = self.target_size_for(input.original_size, self.target_width)?;

        Ok(VggtPreparedBatch {
            family: ModelFamily::Vggt,
            source: input.source,
            sampled_frames: input.frame_count.div_ceil(input.interval),
            original_size: input.original_size,
            target_size,
            interval: input.interval,
            pixel_range: RgbRange::ZeroToOne,
        })
    }
}

fn scaled_crop_size(
    width: u32,
    height: u32,
    target_width: u32,
    patch_multiple: u32,
) -> Result<SpatialSize> {
    if width == 0 || height == 0 {
        return Err(Lux3dError::InvalidInput("original size must be non-zero"));
    }
    if target_width == 0 {
        return Err(Lux3dError::InvalidInput("target width must be non-zero"));
    }
    if patch_multiple == 0 {
        return Err(Lux3dError::InvalidInput("patch multiple must be non-zero"));
    }

    let scale = f64::from(target_width) / f64::from(width);
    let multiple = f64::from(patch_multiple);
    let steps = (f64::from(height) * scale / multiple).round_ties_even();
    if steps < 1.0 {
        return Err(Lux3dError::InvalidInput(
            "scaled target height must be non-zero",
        ));
    }

    Ok(SpatialSize::new(
        target_width,
        (steps as u32) * patch_multiple,
    ))
}

/// Pillow's 8-bit resampler uses 22 fixed-point bits:
/// `PRECISION_BITS = 32 - 8 - 2` (`src/libImaging/Resample.c`).
const PILLOW_PRECISION_BITS: i64 = 22;
const PILLOW_PRECISION_SCALE: f64 = (1u64 << PILLOW_PRECISION_BITS) as f64;

/// Pillow's BICUBIC kernel (`a = -0.5` Catmull-Rom). Algebraically identical
/// to `image::imageops::FilterType::CatmullRom`, but Pillow evaluates it in
/// f64 and quantizes the coefficients to 22-bit fixed point.
fn pillow_bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct PillowCoefficients {
    starts: Vec<usize>,
    counts: Vec<usize>,
    weights: Vec<i32>,
    stride: usize,
}

/// `precompute_coeffs` + `normalize_coeffs_8bpc` from Pillow's `Resample.c`.
fn pillow_coefficients(input_size: usize, output_size: usize) -> PillowCoefficients {
    let scale = input_size as f64 / output_size as f64;
    let filter_scale = scale.max(1.0);
    let support = 2.0 * filter_scale;
    let stride = (support.ceil() as usize) * 2 + 1;
    let inverse_filter_scale = 1.0 / filter_scale;

    let mut starts = Vec::with_capacity(output_size);
    let mut counts = Vec::with_capacity(output_size);
    let mut weights = vec![0_i32; output_size * stride];
    let mut scratch = vec![0.0_f64; stride];

    for out in 0..output_size {
        let center = (out as f64 + 0.5) * scale;
        let mut start = (center - support + 0.5) as i64;
        if start < 0 {
            start = 0;
        }
        let mut end = (center + support + 0.5) as i64;
        if end > input_size as i64 {
            end = input_size as i64;
        }
        let count = (end - start) as usize;
        debug_assert!(count <= stride);

        let mut sum = 0.0_f64;
        for x in 0..count {
            let weight =
                pillow_bicubic((x as f64 + start as f64 - center + 0.5) * inverse_filter_scale);
            scratch[x] = weight;
            sum += weight;
        }
        if sum != 0.0 {
            for value in scratch.iter_mut().take(count) {
                *value /= sum;
            }
        }
        for x in 0..count {
            let scaled = scratch[x] * PILLOW_PRECISION_SCALE;
            weights[out * stride + x] = if scratch[x] < 0.0 {
                (scaled - 0.5) as i32
            } else {
                (scaled + 0.5) as i32
            };
        }

        starts.push(start as usize);
        counts.push(count);
    }

    PillowCoefficients {
        starts,
        counts,
        weights,
        stride,
    }
}

#[inline]
fn pillow_clip8(accumulator: i64) -> u8 {
    (accumulator >> PILLOW_PRECISION_BITS).clamp(0, 255) as u8
}

/// Pillow-compatible 8-bit BICUBIC resize: horizontal pass into an 8-bit
/// intermediate, then a vertical pass, exactly like `ImagingResampleInner`.
///
/// The `image` crate's `CatmullRom` filter uses the same kernel but a
/// single-rounding f32 pipeline, which differs from Pillow by up to one 8-bit
/// level (3.9e-3 in `[0, 1]`). That exceeds the VGGT preprocess parity budget
/// (1e-5), so this port reproduces Pillow's integer arithmetic instead.
fn resize_bicubic_pillow(image: &RgbImage, out_width: u32, out_height: u32) -> RgbImage {
    let (in_width, in_height) = image.dimensions();
    if in_width == out_width && in_height == out_height {
        return image.clone();
    }

    let source = image.as_raw();
    let out_width_usize = out_width as usize;
    let out_height_usize = out_height as usize;

    // Horizontal pass: every input row, output width.
    let horizontal = pillow_coefficients(in_width as usize, out_width_usize);
    let mut intermediate = vec![0_u8; (in_height as usize) * out_width_usize * 3];
    for y in 0..in_height as usize {
        let row = y * in_width as usize * 3;
        for x_out in 0..out_width_usize {
            let start = horizontal.starts[x_out];
            let count = horizontal.counts[x_out];
            let mut accumulator = [1_i64 << (PILLOW_PRECISION_BITS - 1); 3];
            for x in 0..count {
                let weight = i64::from(horizontal.weights[x_out * horizontal.stride + x]);
                let index = row + (start + x) * 3;
                for channel in 0..3 {
                    accumulator[channel] += i64::from(source[index + channel]) * weight;
                }
            }
            let target = (y * out_width_usize + x_out) * 3;
            for channel in 0..3 {
                intermediate[target + channel] = pillow_clip8(accumulator[channel]);
            }
        }
    }

    // Vertical pass: output width from the intermediate image.
    let vertical = pillow_coefficients(in_height as usize, out_height_usize);
    let mut output = vec![0_u8; out_height_usize * out_width_usize * 3];
    for y_out in 0..out_height_usize {
        let start = vertical.starts[y_out];
        let count = vertical.counts[y_out];
        for x in 0..out_width_usize {
            let mut accumulator = [1_i64 << (PILLOW_PRECISION_BITS - 1); 3];
            for y in 0..count {
                let weight = i64::from(vertical.weights[y_out * vertical.stride + y]);
                let index = ((start + y) * out_width_usize + x) * 3;
                for channel in 0..3 {
                    accumulator[channel] += i64::from(intermediate[index + channel]) * weight;
                }
            }
            let target = (y_out * out_width_usize + x) * 3;
            for channel in 0..3 {
                output[target + channel] = pillow_clip8(accumulator[channel]);
            }
        }
    }

    RgbImage::from_raw(out_width, out_height, output).expect("resized buffer dimensions")
}

/// PIL parity: an RGBA image is alpha-composited over opaque white before the
/// RGB conversion. Frames without alpha are converted directly.
fn load_rgb_frame(path: &Path) -> Result<RgbImage> {
    let dynamic = image::open(path)
        .map_err(|_| Lux3dError::InvalidInput("failed to open VGGT input image"))?;

    if !dynamic.color().has_alpha() {
        return Ok(dynamic.to_rgb8());
    }

    let rgba = dynamic.to_rgba8();
    let (width, height) = rgba.dimensions();
    let mut composited = RgbImage::new(width, height);
    for (x, y, pixel) in rgba.enumerate_pixels() {
        let alpha = f32::from(pixel[3]) / 255.0;
        let inverse = 1.0 - alpha;
        let blend =
            |value: u8| -> u8 { (f32::from(value) * alpha + 255.0 * inverse).round() as u8 };
        composited.put_pixel(
            x,
            y,
            Rgb([blend(pixel[0]), blend(pixel[1]), blend(pixel[2])]),
        );
    }
    Ok(composited)
}

/// Center-pad a CHW frame to `max_height` x `max_width` with white (1.0),
/// matching the reference `pad_top = pad_h / 2` split.
fn pad_chw_to_max(frame: &[f32], shape: SpatialSize, max_height: u32, max_width: u32) -> Vec<f32> {
    let height = shape.height;
    let width = shape.width;
    if height == max_height && width == max_width {
        return frame.to_vec();
    }

    let pad_top = (max_height - height) / 2;
    let pad_left = (max_width - width) / 2;
    let max_plane = (max_height as usize) * (max_width as usize);
    let plane = (height as usize) * (width as usize);
    let mut padded = vec![1.0_f32; 3 * max_plane];

    for channel in 0..3 {
        for y in 0..height as usize {
            let source_start = channel * plane + y * (width as usize);
            let destination_start = channel * max_plane
                + (pad_top as usize + y) * (max_width as usize)
                + pad_left as usize;
            padded[destination_start..destination_start + width as usize]
                .copy_from_slice(&frame[source_start..source_start + width as usize]);
        }
    }

    padded
}

fn is_video_extension(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "mp4" | "mov" | "avi" | "mkv"
    )
}

fn default_interval_for(source: &Path) -> usize {
    match source.extension().and_then(|ext| ext.to_str()) {
        Some(ext) if is_video_extension(ext) => 10,
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
        path.push(format!("lux3d-vggt-frames-{}-{nonce}", std::process::id()));
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
struct SampledVggtFrames {
    paths: Vec<PathBuf>,
    _temp_dir: Option<TempFrameDir>,
}

fn load_sampled_frames(source: &Path, interval: usize) -> Result<SampledVggtFrames> {
    if source.is_dir() {
        let entries = collect_rgb_entries(source)?;
        if entries.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "VGGT input directory did not contain any RGB frames",
            ));
        }
        return Ok(SampledVggtFrames {
            paths: entries.into_iter().step_by(interval).collect(),
            _temp_dir: None,
        });
    }

    if source
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(is_video_extension)
    {
        return extract_video_frames(source, interval);
    }

    Err(Lux3dError::InvalidInput(
        "VGGT input source must be a directory or a video file (.mp4/.mov/.avi/.mkv)",
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
    // Mirror the reference's Python `sorted(...)`: lexicographic by lowercased
    // file name, then by full path.
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

fn extract_video_frames(source: &Path, interval: usize) -> Result<SampledVggtFrames> {
    if !source.is_file() {
        return Err(Lux3dError::InvalidInput(
            "VGGT video input file was not found",
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
        .map_err(|_| Lux3dError::InvalidInput("failed to start ffmpeg for VGGT video input"))?;

    if !output.status.success() {
        return Err(Lux3dError::InvalidInput(
            "failed to extract VGGT video frames with ffmpeg",
        ));
    }

    let paths = collect_rgb_entries(&temp_dir.path)?;
    if paths.is_empty() {
        return Err(Lux3dError::InvalidInput(
            "VGGT video input did not yield any RGB frames",
        ));
    }

    Ok(SampledVggtFrames {
        paths,
        _temp_dir: Some(temp_dir),
    })
}

#[cfg(test)]
mod tests {
    use image::{Rgba, RgbaImage};

    use super::*;

    #[test]
    fn composites_rgba_over_white() {
        let mut rgba = RgbaImage::new(2, 1);
        rgba.put_pixel(0, 0, Rgba([0, 0, 0, 128]));
        rgba.put_pixel(1, 0, Rgba([10, 20, 30, 255]));

        let mut path = std::env::temp_dir();
        path.push(format!(
            "lux3d-vggt-rgba-{}-{}.png",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        rgba.save(&path).expect("save rgba fixture");

        let rgb = load_rgb_frame(&path).expect("composite rgba");
        let _ = fs::remove_file(&path);

        // 0 * (128/255) + 255 * (1 - 128/255) == 127.
        assert_eq!(*rgb.get_pixel(0, 0), Rgb([127, 127, 127]));
        assert_eq!(*rgb.get_pixel(1, 0), Rgb([10, 20, 30]));
    }

    #[test]
    fn recognizes_all_server_video_extensions() {
        for extension in ["mp4", "MOV", "avi", "MkV"] {
            assert!(is_video_extension(extension), "expected video: {extension}");
            let path = PathBuf::from(format!("clip.{extension}"));
            assert_eq!(10, default_interval_for(&path));
        }
        assert!(!is_video_extension("png"));
        assert!(!is_video_extension("zip"));
        assert_eq!(1, default_interval_for(Path::new("frames")));
    }
}
