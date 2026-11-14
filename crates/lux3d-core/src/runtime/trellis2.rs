//! TRELLIS.2 image-to-3D pipeline orchestration (T5 geometry slice + T6
//! 1024 cascade).
//!
//! 512 path (`pipeline_type = "512"`):
//!
//! 1. preprocess: RGBA foreground bbox -> square crop -> premultiplied RGB
//!    at 512 (`preprocess::Trellis2Preprocess`).
//! 2. conditioning: DINOv3 ViT-L/16 -> `cond [1, 1029, 1024]`,
//!    `neg_cond = zeros_like(cond)`.
//! 3. sparse structure: seeded noise `[1, 8, 16, 16, 16]` -> 12-step Euler
//!    CFG sampler (strength 7.5, rescale 0.7, interval [0.6, 1.0],
//!    rescale_t 5.0) -> `ss_dec` decoder -> occupancy at 32^3.
//! 4. shape SLAT: seeded noise `[N, 32]` on the active coords -> 12-step
//!    Euler CFG sampler (strength 7.5, rescale 0.5, interval [0.6, 1.0],
//!    rescale_t 3.0) -> de-normalize `slat * std + mean`.
//! 5. shape VAE decode at resolution 512 -> dual-grid head split ->
//!    O-Voxel mesh extraction -> export.
//!
//! 1024 cascade (`pipeline_type = "1024_cascade"`, reference
//! `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:277-364`): the SS
//! stage is unchanged (32^3 occupancy, `cond_512`); the shape stages become
//!
//! 4a. LR shape SLAT on the 32^3 coords with `shape_dit_512` + `cond_512`
//!     -> de-normalize.
//! 4b. shape VAE `upsample(slat, 4)` (`Trellis2ShapeVaeDecoder::
//!     upsample_coords`): `from_latent` + stages 0..3 exactly like the full
//!     decode, returning the children of the stage-3 C2S — the active coords
//!     on the 512 grid.
//! 4c. quantize onto the `hr_resolution // 16` grid:
//!     `((coord + 0.5) / 512 * (hr_resolution // 16)).int()`, deduplicate
//!     (lexicographically sorted rows, `torch.unique(dim=0)` semantics) and
//!     enforce the token budget: the loop breaks when
//!     `rows < max_num_tokens || hr_resolution == 1024`, else
//!     `hr_resolution -= 128`. At resolution 1024 the loop always breaks on
//!     its first pass, so the 1024 cascade never reduces its resolution.
//! 4d. HR shape SLAT on the quantized coords with `shape_dit_1024` +
//!     `cond_1024` (DINOv3 at 1024, `[1, 4101, 1024]`) -> de-normalize.
//! 5. shape VAE decode at `hr_resolution` -> dual-grid head split ->
//!    O-Voxel mesh extraction.
//!
//! Texture (both paths, reference `sample_tex_slat` /
//! `decode_tex_slat`, enabled with `Trellis2RunOptions::with_texture`):
//!
//! 6. texture SLAT on the FINAL shape latent coords (the HR quantized
//!    coords in the cascade): noise `[N, 32]` re-normalized shape SLAT
//!    `(shape_slat - mean) / std` concatenated along channels -> `[N, 64]`
//!    -> `tex_dit_512` (512 path, `cond_512`) or `tex_dit_1024` (cascade,
//!    `cond_1024`) -> 12-step Euler sampler (guidance 1.0, rescale 0.0,
//!    interval [0.6, 0.9], R=3.0) -> de-normalize with the texture vectors.
//! 7. guided texture VAE decode (`tex_dec`, `pred_subdiv = false`): the
//!    shape decoder's raw C2S subdivision logits steer the tex decode, so
//!    the tex output coords equal the shape decode's final coords; the
//!    decoded 6 channels are `raw * 0.5 + 0.5` clamped to [0, 1]
//!    (base_color 0:3, metallic 3, roughness 4, alpha 5).
//! 8. per-vertex base colors: flex_gemm sparse trilinear sampling of the
//!    voxel attributes at the mesh vertices (`MeshWithVoxel.
//!    query_vertex_attrs`); carried on the mesh buffers for PLY/GLB export
//!    alongside the `<stem>.pbrvox.bin` sidecar payload.
//!
//! Stage weights load sequentially from the mmap-ed canonical package and
//! drop after their stage; no more than one DiT is resident (the cascade
//! also reloads the shape VAE between its upsample and decode passes). The
//! RNG is an internal CPU-seeded normal generator (locked decision 4):
//! reproducible across Lux3D runs, NOT bit-identical to `torch.randn` —
//! parity tests inject oracle latents instead of comparing RNG streams.
//!
//! Denormalization vectors and sampler defaults are hard-coded from
//! `pipeline.json` (mirrored in the canonical package's
//! `resolved_config.json`) and can be overridden via
//! [`Trellis2PipelineConfig::from_resolved_config`].

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use super::trellis2_dino::Trellis2Dino;
use super::trellis2_dit::{Trellis2SlatConfig, Trellis2SlatDit, Trellis2SsConfig, Trellis2SsDit};
use super::trellis2_mesh::DualGridMesh;
use super::trellis2_sampler::{Trellis2SamplerParams, sample_dense, sample_sparse};
use super::trellis2_ss_dec::{Trellis2SsDecoder, Trellis2SsDecoderConfig, occupancy_coords};
use super::trellis2_tex::{
    Trellis2TexDecoder, Trellis2TexDit, pbr_attrs_from_raw, query_vertex_colors,
};
use super::trellis2_vae::{Trellis2ShapeVaeConfig, Trellis2ShapeVaeDecoder, decode_shape_mesh};
use crate::ModelFamily;
use crate::Result;
use crate::contracts::Trellis2PipelineType;
use crate::error::Lux3dError;
use crate::export::Trellis2ExportStage;
use crate::geometry::Trellis2GeometryStage;
use crate::preprocess::Trellis2Preprocess;
use crate::weights::{
    CanonicalWeightSetPaths, ModelAssetOptions, VarBuilder, load_canonical_weights,
};

/// The voxel margin of the FlexiDualGrid head (`fdg_vae.py` default).
pub const TRELLIS2_VOXEL_MARGIN: f32 = 0.5;

/// Cascade LR grid resolution (`sample_shape_slat_cascade(lr_resolution=512)`
/// in `run(...)`, `trellis2_image_to_3d.py:566-573`).
pub const TRELLIS2_CASCADE_LR_RESOLUTION: u32 = 512;

/// Cascade HR target resolution (`resolution=1024` for `1024_cascade`). The
/// token-budget loop always breaks once `hr_resolution` reaches 1024, so the
/// 1024 cascade decodes at 1024 regardless of the token count.
pub const TRELLIS2_CASCADE_HR_RESOLUTION: u32 = 1024;

/// Cascade C2S upsample stages (`upsample(slat, upsample_times=4)`): 32^3 LR
/// coords double four times to the 512 grid.
pub const TRELLIS2_CASCADE_UPSAMPLE_TIMES: usize = 4;

/// RNG salt for the texture SLAT noise stream (`Trellis2Rng::new(seed ^
/// SALT)`); the shape stages use their own salts so the streams differ.
pub const TRELLIS2_TEX_NOISE_SALT: u64 = 0x534C_4154_5F54_4558;

/// `shape_slat_normalization.mean` (`pipeline.json`, 32 channels).
pub const TRELLIS2_SHAPE_SLAT_MEAN: [f32; 32] = [
    0.781296, 0.018091, -0.495192, -0.558457, 1.06053, 0.093252, 1.518149, -0.933218, -0.732996,
    2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944, 0.261421, -2.217463,
    1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667, -0.059621, 2.22078, 1.621212,
    0.87723, 0.567247, -3.175944, -3.186688, 1.578665,
];

/// `shape_slat_normalization.std` (`pipeline.json`, 32 channels).
pub const TRELLIS2_SHAPE_SLAT_STD: [f32; 32] = [
    5.972266, 4.706852, 5.44501, 5.209927, 5.32022, 4.547237, 5.020802, 5.444004, 5.226681,
    5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578, 4.805265, 5.124013,
    5.530808, 5.619001, 5.10393, 5.41767, 5.269677, 5.547194, 5.634698, 5.235274, 6.110351,
    5.511298, 6.237273, 4.879207, 5.347008, 5.405691,
];

/// `tex_slat_normalization.mean` (`pipeline.json`, 32 channels).
pub const TRELLIS2_TEX_SLAT_MEAN: [f32; 32] = [
    3.501659, 2.212398, 2.226094, 0.251093, -0.026248, -0.687364, 0.439898, -0.928075, 0.029398,
    -0.339596, -0.869527, 1.038479, -0.972385, 0.126042, -1.129303, 0.455149, -1.209521, 2.069067,
    0.544735, 2.569128, -0.323407, 2.293, -1.925608, -1.217717, 1.213905, 0.971588, -0.023631,
    0.10675, 2.021786, 0.250524, -0.662387, -0.768862,
];

/// `tex_slat_normalization.std` (`pipeline.json`, 32 channels).
pub const TRELLIS2_TEX_SLAT_STD: [f32; 32] = [
    2.665652, 2.743913, 2.765121, 2.595319, 3.037293, 2.291316, 2.144656, 2.911822, 2.969419,
    2.501689, 2.154811, 3.163343, 2.621215, 2.381943, 3.186697, 3.021588, 2.295916, 3.234985,
    3.233086, 2.26014, 2.874801, 2.810596, 3.29272, 2.674999, 2.680878, 2.372054, 2.451546,
    2.353556, 2.995195, 2.379849, 2.786195, 2.77519,
];

/// Deterministic CPU normal RNG: splitmix64-scrambled xorshift64* state with
/// Box-Muller normals (one-value carry). Reproducible across runs; streams
/// intentionally differ from `torch.randn` (locked decision 4).
pub struct Trellis2Rng {
    state: u64,
    spare: Option<f32>,
}

impl Trellis2Rng {
    /// Seed the generator (any u64, including 0 — the state is scrambled
    /// through a splitmix64 round first).
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let state = z ^ (z >> 31);
        Self {
            state: if state == 0 { 1 } else { state },
            spare: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// One standard-normal sample (Box-Muller with a cached spare).
    pub fn normal_f32(&mut self) -> f32 {
        if let Some(value) = self.spare.take() {
            return value;
        }
        let scale = (1u64 << 53) as f64;
        let u1 = ((self.next_u64() >> 11) as f64 / scale).max(1e-300);
        let u2 = (self.next_u64() >> 11) as f64 / scale;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = core::f64::consts::TAU * u2;
        self.spare = Some((radius * angle.sin()) as f32);
        (radius * angle.cos()) as f32
    }

    /// `len` standard-normal samples.
    pub fn randn(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| self.normal_f32()).collect()
    }
}

/// Stage-resolved pipeline parameters. Defaults mirror `pipeline.json`
/// (`resolved_config.json` in the canonical package); `from_resolved_config`
/// applies package overrides.
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2PipelineConfig {
    /// Shape SLAT de-normalization mean (32).
    pub shape_slat_mean: Vec<f32>,
    /// Shape SLAT de-normalization std (32).
    pub shape_slat_std: Vec<f32>,
    /// Texture SLAT de-normalization mean (32).
    pub tex_slat_mean: Vec<f32>,
    /// Texture SLAT de-normalization std (32).
    pub tex_slat_std: Vec<f32>,
    /// Sparse-structure sampler defaults (steps overridden per run).
    pub ss_sampler: Trellis2SamplerParams,
    /// Shape SLAT sampler defaults (steps overridden per run).
    pub shape_sampler: Trellis2SamplerParams,
    /// Texture SLAT sampler defaults (steps overridden per run).
    pub tex_sampler: Trellis2SamplerParams,
    /// Sparse-structure occupancy resolution for the 512 path (32).
    pub ss_resolution: u32,
    /// Mesh extraction resolution for the 512 path (512).
    pub shape_resolution: u32,
    /// FlexiDualGrid voxel margin (0.5).
    pub voxel_margin: f32,
}

impl Default for Trellis2PipelineConfig {
    fn default() -> Self {
        Self {
            shape_slat_mean: TRELLIS2_SHAPE_SLAT_MEAN.to_vec(),
            shape_slat_std: TRELLIS2_SHAPE_SLAT_STD.to_vec(),
            tex_slat_mean: TRELLIS2_TEX_SLAT_MEAN.to_vec(),
            tex_slat_std: TRELLIS2_TEX_SLAT_STD.to_vec(),
            ss_sampler: Trellis2SamplerParams::default(),
            shape_sampler: Trellis2SamplerParams {
                steps: 12,
                sigma_min: 1e-5,
                guidance_strength: 7.5,
                guidance_rescale: 0.5,
                guidance_interval: [0.6, 1.0],
                rescale_t: 3.0,
            },
            // `tex_slat_sampler` (pipeline.json): guidance is cond-only
            // (1.0), no rescale, interval [0.6, 0.9].
            tex_sampler: Trellis2SamplerParams {
                steps: 12,
                sigma_min: 1e-5,
                guidance_strength: 1.0,
                guidance_rescale: 0.0,
                guidance_interval: [0.6, 0.9],
                rescale_t: 3.0,
            },
            ss_resolution: 32,
            shape_resolution: 512,
            voxel_margin: TRELLIS2_VOXEL_MARGIN,
        }
    }
}

fn read_f32_vec(value: Option<&serde_json::Value>) -> Option<Vec<f32>> {
    let array = value?.as_array()?;
    Some(
        array
            .iter()
            .filter_map(|v| v.as_f64().map(|v| v as f32))
            .collect(),
    )
}

fn read_sampler(value: Option<&serde_json::Value>) -> Option<Trellis2SamplerParams> {
    let value = value?;
    let params = value.get("params")?;
    let mut out = Trellis2SamplerParams {
        steps: params.get("steps")?.as_u64()? as usize,
        ..Trellis2SamplerParams::default()
    };
    if let Some(sigma) = value
        .get("args")
        .and_then(|args| args.get("sigma_min"))
        .and_then(|v| v.as_f64())
    {
        out.sigma_min = sigma;
    }
    if let Some(strength) = params.get("guidance_strength").and_then(|v| v.as_f64()) {
        out.guidance_strength = strength;
    }
    if let Some(rescale) = params.get("guidance_rescale").and_then(|v| v.as_f64()) {
        out.guidance_rescale = rescale;
    }
    if let Some(interval) = params.get("guidance_interval").and_then(|v| v.as_array()) {
        if interval.len() == 2 {
            if let (Some(low), Some(high)) = (interval[0].as_f64(), interval[1].as_f64()) {
                out.guidance_interval = [low, high];
            }
        }
    }
    if let Some(rescale_t) = params.get("rescale_t").and_then(|v| v.as_f64()) {
        out.rescale_t = rescale_t;
    }
    Some(out)
}

impl Trellis2PipelineConfig {
    /// Apply overrides from a `resolved_config.json` / raw `pipeline.json`
    /// document (`pipeline_args` nesting tolerated). Missing sections keep
    /// the hard-coded defaults.
    pub fn from_resolved_config(value: &serde_json::Value) -> Self {
        let mut config = Self::default();
        // `resolved_config.json` nests under `pipeline_args`; a raw
        // `pipeline.json` carries the same keys at the root.
        let root = value.get("pipeline_args").unwrap_or(value);
        if let Some(mean) = read_f32_vec(
            root.get("shape_slat_normalization")
                .and_then(|n| n.get("mean")),
        ) {
            if mean.len() == 32 {
                config.shape_slat_mean = mean;
            }
        }
        if let Some(std) = read_f32_vec(
            root.get("shape_slat_normalization")
                .and_then(|n| n.get("std")),
        ) {
            if std.len() == 32 {
                config.shape_slat_std = std;
            }
        }
        if let Some(mean) = read_f32_vec(
            root.get("tex_slat_normalization")
                .and_then(|n| n.get("mean")),
        ) {
            if mean.len() == 32 {
                config.tex_slat_mean = mean;
            }
        }
        if let Some(std) = read_f32_vec(
            root.get("tex_slat_normalization")
                .and_then(|n| n.get("std")),
        ) {
            if std.len() == 32 {
                config.tex_slat_std = std;
            }
        }
        if let Some(sampler) = read_sampler(root.get("sparse_structure_sampler")) {
            config.ss_sampler = sampler;
        }
        if let Some(sampler) = read_sampler(root.get("shape_slat_sampler")) {
            config.shape_sampler = sampler;
        }
        if let Some(sampler) = read_sampler(root.get("tex_slat_sampler")) {
            config.tex_sampler = sampler;
        }
        config
    }
}

/// Cascade HR coordinate quantization + token-budget loop (reference
/// `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:327-339`).
///
/// Each pass quantizes the upsampled 512-grid coords onto the
/// `hr_resolution // 16` grid with the exact reference expression
/// `((coord + 0.5) / lr_resolution * (hr_resolution // 16)).int()`
/// (F32 arithmetic, truncate toward zero — the upsampled coords are
/// non-negative, so this is a floor), deduplicates the rows and breaks when
/// `unique_rows < max_num_tokens || hr_resolution == 1024`, otherwise
/// `hr_resolution -= 128`. `torch.unique(dim=0)` returns lexicographically
/// sorted rows; the returned coords are sorted the same way.
///
/// Returns `(unique_sorted_coords, hr_resolution)`.
pub fn quantize_hr_coords(
    hr_coords: &[[i32; 3]],
    lr_resolution: u32,
    resolution: u32,
    max_num_tokens: usize,
) -> (Vec<[i32; 3]>, u32) {
    quantize_hr_coords_with_stop(
        hr_coords,
        lr_resolution,
        resolution,
        max_num_tokens,
        TRELLIS2_CASCADE_HR_RESOLUTION,
    )
}

/// [`quantize_hr_coords`] with the loop's stop resolution explicit. The
/// reference hardcodes 1024 (the 1024-DiT latent grid ceiling:
/// `hr_resolution // 16 <= 64`), so production runs always stop at 1024 and
/// the 1024 cascade can never reduce its resolution regardless of the token
/// count. The parameter is exposed so unit tests can drive the decrement
/// path (1024 -> 896 -> 768 ...). `resolution - stop_resolution` must be a
/// multiple of 128 for the loop to terminate (true for the reference's 1024
/// and 1536 cascade targets).
pub fn quantize_hr_coords_with_stop(
    hr_coords: &[[i32; 3]],
    lr_resolution: u32,
    resolution: u32,
    max_num_tokens: usize,
    stop_resolution: u32,
) -> (Vec<[i32; 3]>, u32) {
    let lr = lr_resolution as f32;
    let mut hr_resolution = resolution;
    let coords = loop {
        let scale = (hr_resolution / 16) as f32;
        // ((coord + 0.5) / lr_resolution * (hr_resolution // 16)).int():
        // torch promotes to float32, divides, multiplies, then truncates
        // toward zero (`x as i32` saturates like torch only for in-range
        // values, which the non-negative <= 511 coords always are).
        let mut quant: Vec<[i32; 3]> = hr_coords
            .iter()
            .map(|c| {
                [
                    ((c[0] as f32 + 0.5) / lr * scale) as i32,
                    ((c[1] as f32 + 0.5) / lr * scale) as i32,
                    ((c[2] as f32 + 0.5) / lr * scale) as i32,
                ]
            })
            .collect();
        quant.sort_unstable();
        quant.dedup();
        if quant.len() < max_num_tokens || hr_resolution == stop_resolution {
            break quant;
        }
        hr_resolution -= 128;
    };
    (coords, hr_resolution)
}

/// Run options (`plan.md` frozen surface; defaults match the reference
/// `run(...)` signature).
#[derive(Debug, Clone, PartialEq)]
pub struct Trellis2RunOptions {
    /// RNG seed (42).
    pub seed: u64,
    /// Pipeline variant (Cascade1024 by contract; T5 implements the `512`
    /// path only and rejects the others until T6 lands).
    pub pipeline_type: Trellis2PipelineType,
    /// Sparse-structure Euler steps (12).
    pub ss_steps: usize,
    /// Shape SLAT Euler steps (12).
    pub shape_steps: usize,
    /// Texture SLAT Euler steps (12).
    pub tex_steps: usize,
    /// Run the texture stages (texture SLAT + guided texture VAE + PBR
    /// vertex colors, true — the reference pipeline always textures). With
    /// `false` the pipeline stays geometry-only (the mesh buffers carry the
    /// 7-channel shape attributes and no PBR payload).
    pub with_texture: bool,
    /// Cascade token budget (49152). The reference loop breaks when the
    /// quantized HR coord count drops below this value — but also whenever
    /// `hr_resolution == 1024`, so the 1024 cascade decodes at 1024 even
    /// when the budget would be exceeded.
    pub max_num_tokens: usize,
    /// Run image preprocessing inside `run_from_path` (true).
    pub preprocess: bool,
}

impl Default for Trellis2RunOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            pipeline_type: Trellis2PipelineType::Cascade1024,
            ss_steps: 12,
            shape_steps: 12,
            tex_steps: 12,
            with_texture: true,
            max_num_tokens: 49152,
            preprocess: true,
        }
    }
}

/// Per-stage outputs exposed for tests and parity taps.
///
/// `shape_coords`/`shape_latent`/`decoded_feats` always describe the FINAL
/// shape stage: in the 512 path the latent lives on `ss_coords`, in the
/// cascade it is the HR latent on `quant_coords`. The cascade inputs are
/// kept in the additive `lr_*`/`hr_*` fields.
#[derive(Debug, Clone)]
pub struct Trellis2Stages {
    /// DINOv3 conditioning `[1, 1029, 1024]` (or the injected cond).
    pub cond: Tensor,
    /// Negative conditioning (zeros).
    pub neg_cond: Tensor,
    /// Sparse-structure flow latent `[1, 8, R, R, R]` (`None` when the
    /// coordinates were injected directly).
    pub z_s: Option<Tensor>,
    /// Active voxel coordinates at the SS resolution (batch dim dropped).
    pub ss_coords: Vec<[i32; 3]>,
    /// Shape SLAT coordinates (identical to `ss_coords` in the 512 path; the
    /// quantized HR coords in the cascade).
    pub shape_coords: Vec<[i32; 3]>,
    /// De-normalized shape SLAT `[N, 32]` (the HR slat in the cascade).
    pub shape_latent: Tensor,
    /// Decoded 7-channel shape attributes `[N, 7]`.
    pub decoded_feats: Tensor,
    /// Extracted mesh buffers.
    pub mesh: Option<crate::geometry::Trellis2MeshBuffers>,
    /// Cascade only: de-normalized LR (32^3) shape SLAT before the upsample
    /// (`None` in the 512 path).
    pub lr_shape_latent: Option<Tensor>,
    /// Cascade only: shape VAE `upsample(slat, 4)` coords on the 512 grid,
    /// before quantization (`None` in the 512 path).
    pub hr_upsampled_coords: Option<Vec<[i32; 3]>>,
    /// Cascade only: quantized + deduplicated HR coords the HR stage ran on
    /// (identical to `shape_coords`; kept for stage-tap symmetry).
    pub quant_coords: Option<Vec<[i32; 3]>>,
    /// Cascade only: the token-budget resolution the HR stage/decode ran at
    /// (1024 unless the reference loop reduced it; `None` in the 512 path).
    pub hr_resolution: Option<u32>,
    /// Shape decoder raw C2S subdivision logits (one `[N_parents, 8]` F32
    /// tensor per C2S stage, stage order) — the texture decoder's
    /// `guide_subs`. Empty only for degenerate inputs.
    pub subdiv_logits: Vec<Tensor>,
    /// De-normalized texture SLAT `[N, 32]` on the shape latent coords
    /// (`None` for geometry-only runs).
    pub tex_latent: Option<Tensor>,
    /// Per-voxel PBR attributes `[N, 6]` in [0, 1] (base_color 0:3,
    /// metallic 3, roughness 4, alpha 5) on the shape decode's final coords
    /// (`None` for geometry-only runs).
    pub tex_attrs: Option<Vec<f32>>,
}

/// Shape-stage products of [`Trellis2Pipeline::sample_shape_slat_cascade`]
/// (all tensors de-normalized, F32).
#[derive(Debug, Clone)]
pub struct Trellis2CascadeShape {
    /// De-normalized LR (32^3) shape SLAT `[N_lr, 32]`.
    pub lr_slat: Tensor,
    /// Shape VAE `upsample(slat, 4)` coords on the 512 grid
    /// (pre-quantization; `hr_resolution`-grid parents would be
    /// `hr_resolution / 16` if the budget loop ever reduced the target).
    pub hr_upsampled_coords: Vec<[i32; 3]>,
    /// Quantized + deduplicated coords the HR stage ran on (lexicographically
    /// sorted, `torch.unique(dim=0)` semantics).
    pub quant_coords: Vec<[i32; 3]>,
    /// The token-budget resolution (`1024` for the 1024 cascade — the loop
    /// always breaks at 1024).
    pub hr_resolution: u32,
    /// De-normalized HR shape SLAT `[N_hr, 32]` on `quant_coords`.
    pub hr_slat: Tensor,
}

/// Raw TRELLIS.2 checkpoint layout (`models/TRELLIS.2-4B`), stage prefix ->
/// file relative path. The 1024 shape/texture DiTs serve the cascade; the
/// texture decoder produces the per-voxel PBR attributes.
const RAW_STAGE_FILES: [(&str, &str); 9] = [
    ("image_cond", "external/dinov3/model.safetensors"),
    ("ss_flow", "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors"),
    (
        "ss_dec",
        "external/ss_dec/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    (
        "shape_dit_512",
        "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "shape_dit_1024",
        "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
    ),
    (
        "shape_dec",
        "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
    ),
    (
        "tex_dit_512",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
    ),
    (
        "tex_dit_1024",
        "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
    ),
    ("tex_dec", "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors"),
];

/// TRELLIS.2 image-to-3D pipeline (geometry slice).
#[derive(Debug)]
pub struct Trellis2Pipeline {
    /// Canonical package paths (`Some` when loaded through
    /// [`Self::load`]).
    pub weights: Option<CanonicalWeightSetPaths>,
    pub preprocess: Trellis2Preprocess,
    pub geometry: Trellis2GeometryStage,
    pub export: Trellis2ExportStage,
    pub config: Trellis2PipelineConfig,
    /// Raw checkpoint dir + device when loaded through
    /// [`Self::load_from_raw_dir`].
    raw_dir: Option<(PathBuf, Device)>,
}

impl Trellis2Pipeline {
    /// Load from the canonical weights package (no HF access).
    pub fn load(options: ModelAssetOptions) -> Result<Self> {
        let weights = load_canonical_weights(ModelFamily::Trellis2, options)?;
        let config = weights
            .read_resolved_config_json()
            .map(|value| Trellis2PipelineConfig::from_resolved_config(&value))
            .unwrap_or_default();
        Ok(Self {
            weights: Some(weights),
            preprocess: Trellis2Preprocess::new(config.shape_resolution)?,
            geometry: Trellis2GeometryStage,
            export: Trellis2ExportStage,
            config,
            raw_dir: None,
        })
    }

    /// Local-development loader over the raw TRELLIS.2 checkpoint directory
    /// (`models/TRELLIS.2-4B` layout). Each stage opens its own checkpoint
    /// file with the canonical stage prefix applied.
    pub fn load_from_raw_dir(path: &Path, device: &Device) -> Result<Self> {
        let mut config = Trellis2PipelineConfig::default();
        let pipeline_json = path.join("pipeline.json");
        if pipeline_json.is_file() {
            let body = std::fs::read_to_string(&pipeline_json).map_err(|source| {
                Lux3dError::ModelAssetIo {
                    path: pipeline_json.clone(),
                    source,
                }
            })?;
            let value: serde_json::Value = serde_json::from_str(&body).map_err(|source| {
                Lux3dError::CanonicalManifestJson {
                    path: pipeline_json.clone(),
                    source,
                }
            })?;
            config = Trellis2PipelineConfig::from_resolved_config(&value);
        }
        for (prefix, relative) in RAW_STAGE_FILES {
            let file = path.join(relative);
            if !file.is_file() {
                return Err(Lux3dError::MissingWeightFile { path: file });
            }
            let _ = prefix;
        }
        Ok(Self {
            weights: None,
            preprocess: Trellis2Preprocess::new(config.shape_resolution)?,
            geometry: Trellis2GeometryStage,
            export: Trellis2ExportStage,
            config,
            raw_dir: Some((path.to_path_buf(), device.clone())),
        })
    }

    /// VarBuilder rooted at one stage's namespace: the returned builder
    /// resolves the stage's bare tensor names (e.g. `from_latent.weight`),
    /// matching the stage loaders' convention. For the canonical package the
    /// merged file is namespaced with `.pp(prefix)`; for a raw dir the stage
    /// file already carries bare keys, so it is mmap-ed directly.
    fn stage_var_builder(&self, prefix: &str, device: &Device) -> Result<VarBuilder<'static>> {
        match (&self.raw_dir, &self.weights) {
            (Some((dir, raw_device)), _) => {
                if raw_device.location() != device.location() {
                    return Err(Lux3dError::InvalidInput(
                        "Trellis2 raw-dir pipeline is bound to a different device",
                    ));
                }
                let (_, relative) = RAW_STAGE_FILES
                    .iter()
                    .find(|(name, _)| *name == prefix)
                    .ok_or_else(|| Lux3dError::ModelAssetResolution {
                        message: format!(
                            "no raw TRELLIS.2 checkpoint mapped for stage prefix `{prefix}`"
                        ),
                    })?;
                let file = dir.join(relative);
                // SAFETY: the mmap-backed builder keeps the mapping alive for
                // every tensor created from it; the file is not mutated.
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        std::slice::from_ref(&file),
                        DType::F32,
                        device,
                    )
                }
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to open raw TRELLIS.2 stage `{prefix}`: {source}"),
                })?;
                Ok(vb)
            }
            (None, Some(weights)) => {
                // SAFETY: mmap-backed; the canonical file outlives the tensor
                // set produced from it and is not mutated.
                let vb = unsafe { weights.var_builder(DType::F32, device) }.map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("failed to open canonical TRELLIS.2 weights: {source}"),
                    }
                })?;
                Ok(vb.pp(prefix))
            }
            (None, None) => Err(Lux3dError::InvalidInput(
                "Trellis2Pipeline has no weights loaded",
            )),
        }
    }

    fn resolutions(&self, options: &Trellis2RunOptions) -> Result<(u32, u32)> {
        match options.pipeline_type {
            Trellis2PipelineType::R512 => {
                Ok((self.config.ss_resolution, self.config.shape_resolution))
            }
            Trellis2PipelineType::Cascade1024 => {
                Ok((self.config.ss_resolution, TRELLIS2_CASCADE_HR_RESOLUTION))
            }
            other => Err(Lux3dError::ModelAssetResolution {
                message: format!(
                    "Trellis2 pipeline type `{}` is not implemented in this slice (the 1024 \
                     cascade is implemented; the single-resolution `1024` path lands with the \
                     texture stage); use `512` or `1024_cascade`",
                    other.as_str()
                ),
            }),
        }
    }

    /// DINOv3 conditioning: `cond [1, 1029, 1024]` + zero negative cond.
    pub fn image_conditioning(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let pixel_values = self
            .preprocess
            .normalized_pixel_values(prepared, device)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
            })?;
        self.dino_conditioning_pixels(&pixel_values, device)
    }

    /// Cascade conditioning at the HR resolution (1024): DINOv3 over
    /// pre-built pixel values (`[1, 3, 1024, 1024]`), producing
    /// `[1, 4101, 1024]` + zero negative cond.
    fn dino_conditioning_pixels(
        &self,
        pixel_values: &Tensor,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let vb = self.stage_var_builder("image_cond", device)?;
        let dino = Trellis2Dino::load(vb, &super::trellis2_dino::Trellis2DinoConfig::vitl16())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load TRELLIS.2 DINOv3: {source}"),
            })?;
        let cond = dino.forward(pixel_values).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 DINOv3 forward failed: {source}"),
            }
        })?;
        let neg_cond =
            Tensor::zeros_like(&cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        Ok((cond, neg_cond))
    }

    /// The preprocess stage at the cascade HR resolution: the reference
    /// crops once and lets `get_cond` resize the same crop to each DINOv3
    /// input size, so re-running the (deterministic, geometry-only)
    /// preprocess at 1024 reproduces the reference `cond_1024` input exactly.
    fn cascade_preprocess_hr(&self) -> Result<Trellis2Preprocess> {
        Trellis2Preprocess::new(TRELLIS2_CASCADE_HR_RESOLUTION)
    }

    /// Cascade `cond_1024` for an already-prepared 512 image
    /// ([`Self::run_prepared`], which no longer holds the source pixels):
    /// the prepared pixels are the square crop resized to 512, so resizing
    /// them to 1024 (LANCZOS3, the preprocess resampler) approximates the
    /// reference's direct crop -> 1024 resize (a double resample).
    /// [`Self::run_from_path_with_stages`] re-prepares from the source and
    /// is exact.
    fn cascade_cond_1024_from_prepared(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let src = prepared.target_size.width as usize;
        let dst = TRELLIS2_CASCADE_HR_RESOLUTION as usize;
        if prepared.target_size.width != prepared.target_size.height
            || prepared.uint8_rgb.len() != src * src * 3
        {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade 1024 conditioning needs a square 512 prepared image",
            ));
        }
        let resized =
            crate::preprocess::resize_lanczos_u8::<3>(&prepared.uint8_rgb, src, src, dst, dst);
        let pixel_values = crate::preprocess::normalized_pixel_values_from_u8(
            &resized,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            device,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
        })?;
        self.dino_conditioning_pixels(&pixel_values, device)
    }

    /// Sparse-structure stage: sample `z_s` from `noise` (or the seeded RNG
    /// when `noise` is `None`) and decode the occupancy coords at
    /// `ss_resolution`. The SS flow DiT and decoder are dropped afterwards.
    pub fn sample_sparse_structure(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        device: &Device,
        ss_resolution: u32,
    ) -> Result<(Tensor, Vec<[i32; 3]>)> {
        let noise = match noise {
            Some(noise) => noise.clone(),
            None => {
                let mut rng = Trellis2Rng::new(seed ^ 0x5353_5F4E_4F49_5345);
                let ss_config = Trellis2SsConfig::trellis_2_4b();
                // The SS flow latent lives on the DiT's dense grid
                // (`ss_flow_img_dit_1_3B_64` -> resolution 16); the
                // occupancy decode then upsamples to `ss_resolution`.
                let latent_res = ss_config.resolution;
                let values = rng.randn(ss_config.token_count() * ss_config.in_channels);
                Tensor::from_vec(
                    values,
                    (1, ss_config.in_channels, latent_res, latent_res, latent_res),
                    device,
                )
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 SS noise allocation failed: {source}"),
                })?
            }
        };
        let vb = self.stage_var_builder("ss_flow", device)?;
        let ss_dit =
            Trellis2SsDit::load(vb, &Trellis2SsConfig::trellis_2_4b()).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 ss_flow: {source}"),
                }
            })?;
        let params = Trellis2SamplerParams {
            steps,
            ..self.config.ss_sampler.clone()
        };
        let (z_s, _trace) = sample_dense(
            &|x: &Tensor, t_model: f32, c: &Tensor| ss_dit.forward(x, t_model, c),
            &noise,
            cond,
            neg_cond,
            &params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 SS sampling failed: {source}"),
        })?;
        drop(ss_dit);

        let vb = self.stage_var_builder("ss_dec", device)?;
        let decoder = Trellis2SsDecoder::load(vb, &Trellis2SsDecoderConfig::ss_dec_16l8())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load TRELLIS.2 ss_dec: {source}"),
            })?;
        let logits =
            decoder
                .forward(&z_s)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 SS decode failed: {source}"),
                })?;
        drop(decoder);
        let coords4 = occupancy_coords(&logits, ss_resolution).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 occupancy extraction failed: {source}"),
            }
        })?;
        let coords: Vec<[i32; 3]> = coords4
            .iter()
            .filter(|c| c[0] == 0)
            .map(|c| [c[1], c[2], c[3]])
            .collect();
        Ok((z_s, coords))
    }

    /// Shape SLAT stage (512): sample the (normalized) latent on `coords`
    /// and de-normalize. The flow DiT is dropped afterwards.
    pub fn sample_shape_slat(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_shape_slat_stage(
            "shape_dit_512",
            Trellis2SlatConfig::shape_512(),
            0x534C_4154_5F4E_4F49,
            cond,
            neg_cond,
            coords,
            noise,
            seed,
            steps,
            device,
        )
    }

    /// Cascade HR shape SLAT stage (1024): same sampler params, the
    /// `shape_dit_1024` checkpoint and a distinct RNG salt so the HR noise
    /// stream differs from the LR one. Negative cond is `zeros_like(cond)`
    /// exactly like the reference.
    fn sample_shape_slat_hr(
        &self,
        cond: &Tensor,
        coords: &[[i32; 3]],
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let neg_cond =
            Tensor::zeros_like(cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        self.sample_shape_slat_stage(
            "shape_dit_1024",
            Trellis2SlatConfig::shape_1024(),
            0x534C_4154_5F48_5231,
            cond,
            &neg_cond,
            coords,
            None,
            seed,
            steps,
            device,
        )
    }

    /// One shape SLAT sampling pass over a stage DiT: seeded (or injected)
    /// noise `[N, 32]` on `coords` -> Euler/CFG -> de-normalize. The DiT is
    /// dropped before the de-normalization so only one flow model is ever
    /// resident (12 GiB budget).
    #[allow(clippy::too_many_arguments)]
    fn sample_shape_slat_stage(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        noise_salt: u64,
        cond: &Tensor,
        neg_cond: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let noise = match noise {
            Some(noise) => noise.clone(),
            None => {
                let mut rng = Trellis2Rng::new(seed ^ noise_salt);
                let values = rng.randn(coords.len() * 32);
                Tensor::from_vec(values, (coords.len(), 32), device).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Trellis2 SLAT noise allocation failed: {source}"),
                    }
                })?
            }
        };
        let vb = self.stage_var_builder(stage, device)?;
        let dit = Trellis2SlatDit::load(vb, &config).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load TRELLIS.2 {stage}: {source}"),
            }
        })?;
        let params = Trellis2SamplerParams {
            steps,
            ..self.config.shape_sampler.clone()
        };
        let (slat_norm, _trace) = sample_sparse(
            &|x: &Tensor, t_model: f32, c: &Tensor, cs: &[[i32; 3]]| dit.forward(x, cs, t_model, c),
            coords,
            &noise,
            cond,
            neg_cond,
            &params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 shape SLAT sampling failed: {source}"),
        })?;
        drop(dit);
        self.denormalize_shape_slat(&slat_norm)
    }

    /// `slat * std + mean` with the package shape-normalization vectors.
    pub fn denormalize_shape_slat(&self, slat_norm: &Tensor) -> Result<Tensor> {
        let device = slat_norm.device();
        let mean = Tensor::from_vec(self.config.shape_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.shape_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        slat_norm
            .broadcast_mul(&std)
            .and_then(|t| t.broadcast_add(&mean))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape SLAT de-normalization failed: {source}"),
            })
    }

    /// Cascade upsample + quantize steps (the middle of the reference
    /// `sample_shape_slat_cascade`, `trellis2_image_to_3d.py:319-339`):
    ///
    /// 1. load the shape VAE and run `upsample(slat, 4)` — `from_latent` +
    ///    stages 0..3 exactly like the full decode — to obtain the HR coords
    ///    on the 512 grid;
    /// 2. drop the VAE, then quantize onto the `hr_resolution // 16` grid,
    ///    deduplicate and run the token-budget loop
    ///    ([`quantize_hr_coords`]).
    ///
    /// Returns `(hr_upsampled_coords, quant_coords, hr_resolution)`. The
    /// 1024 cascade's loop always breaks at `hr_resolution == 1024`, so
    /// `hr_resolution` is 1024 here regardless of the token count.
    pub fn upsample_and_quantize_shape_slat(
        &self,
        lr_coords: &[[i32; 3]],
        lr_slat_denorm: &Tensor,
        max_num_tokens: usize,
        device: &Device,
    ) -> Result<(Vec<[i32; 3]>, Vec<[i32; 3]>, u32)> {
        let vb = self.stage_var_builder("shape_dec", device)?;
        let decoder =
            Trellis2ShapeVaeDecoder::load(vb, &Trellis2ShapeVaeConfig::flexi_dual_shape())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 shape_dec: {source}"),
                })?;
        let hr_coords = decoder
            .upsample_coords(lr_coords, lr_slat_denorm, TRELLIS2_CASCADE_UPSAMPLE_TIMES)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape VAE upsample failed: {source}"),
            })?;
        drop(decoder);
        let (quant_coords, hr_resolution) = quantize_hr_coords(
            &hr_coords,
            TRELLIS2_CASCADE_LR_RESOLUTION,
            TRELLIS2_CASCADE_HR_RESOLUTION,
            max_num_tokens,
        );
        Ok((hr_coords, quant_coords, hr_resolution))
    }

    /// Full cascade shape stage (reference `sample_shape_slat_cascade`):
    /// LR SLAT on `coords` (32^3) with `shape_dit_512` + `cond_lr`, shape
    /// VAE `upsample(slat, 4)`, HR quantization + token budget, HR SLAT on
    /// the quantized coords with `shape_dit_1024` + `cond_hr`. Both stages
    /// use the `shape_slat_sampler` params (12 steps, guidance 7.5,
    /// rescale 0.5, interval [0.6, 1.0], R=3.0); the DiTs and the VAE are
    /// loaded and dropped sequentially.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_shape_slat_cascade(
        &self,
        cond_lr: &Tensor,
        cond_hr: &Tensor,
        coords: &[[i32; 3]],
        lr_noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        max_num_tokens: usize,
        device: &Device,
    ) -> Result<Trellis2CascadeShape> {
        let neg_lr = Tensor::zeros_like(cond_lr).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            }
        })?;
        let lr_slat =
            self.sample_shape_slat(cond_lr, &neg_lr, coords, lr_noise, seed, steps, device)?;
        let (hr_upsampled_coords, quant_coords, hr_resolution) =
            self.upsample_and_quantize_shape_slat(coords, &lr_slat, max_num_tokens, device)?;
        if quant_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade quantized the upsampled coords to zero HR tokens",
            ));
        }
        let hr_slat = self.sample_shape_slat_hr(cond_hr, &quant_coords, seed, steps, device)?;
        Ok(Trellis2CascadeShape {
            lr_slat,
            hr_upsampled_coords,
            quant_coords,
            hr_resolution,
            hr_slat,
        })
    }

    /// Shape VAE decode + dual-grid mesh extraction from a de-normalized
    /// shape latent on `coords` (no flow DiT involved). `cond`/`z_s` are
    /// placeholders; the stage struct carries the decode products.
    /// `resolution` is the mesh extraction resolution (512 production;
    /// oracle fixtures replay at their own resolution).
    pub fn decode_shape_slat(
        &self,
        coords: &[[i32; 3]],
        slat_denorm: &Tensor,
        resolution: u32,
        device: &Device,
    ) -> Result<Trellis2Stages> {
        let shape_resolution = resolution;
        let (decoded, mesh) =
            self.decode_shape_vae(coords, slat_denorm, device, shape_resolution)?;
        let mesh = self.assemble_mesh(shape_resolution, mesh, &decoded.coords, &decoded.feats)?;
        Ok(Trellis2Stages {
            cond: Tensor::zeros((1,), DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("placeholder cond allocation failed: {source}"),
                }
            })?,
            neg_cond: Tensor::zeros((1,), DType::F32, device).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("placeholder neg cond allocation failed: {source}"),
                }
            })?,
            z_s: None,
            ss_coords: coords.to_vec(),
            shape_coords: coords.to_vec(),
            shape_latent: slat_denorm.clone(),
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: None,
            hr_upsampled_coords: None,
            quant_coords: None,
            hr_resolution: None,
            subdiv_logits: decoded.subdiv_logits,
            tex_latent: None,
            tex_attrs: None,
        })
    }

    fn decode_shape_vae(
        &self,
        coords: &[[i32; 3]],
        slat_denorm: &Tensor,
        device: &Device,
        shape_resolution: u32,
    ) -> Result<(super::trellis2_vae::ShapeVaeDecodeOutput, DualGridMesh)> {
        let vb = self.stage_var_builder("shape_dec", device)?;
        let decoder =
            Trellis2ShapeVaeDecoder::load(vb, &Trellis2ShapeVaeConfig::flexi_dual_shape())
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("failed to load TRELLIS.2 shape_dec: {source}"),
                })?;
        decode_shape_mesh(
            &decoder,
            coords,
            slat_denorm,
            shape_resolution as usize,
            self.config.voxel_margin,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 shape VAE decode failed: {source}"),
        })
    }

    /// Re-normalize the de-normalized shape SLAT for the texture stage:
    /// `shape_slat_norm = (shape_slat - mean) / std` with the shape
    /// normalization vectors (`sample_tex_slat`, pipeline 407-409).
    pub fn normalize_shape_slat(&self, shape_slat_denorm: &Tensor) -> Result<Tensor> {
        let device = shape_slat_denorm.device();
        let mean = Tensor::from_vec(self.config.shape_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.shape_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        shape_slat_denorm
            .broadcast_sub(&mean)
            .and_then(|t| t.broadcast_div(&std))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape SLAT normalization failed: {source}"),
            })
    }

    /// Texture SLAT stage (512): re-normalize the shape SLAT, sample the
    /// (normalized) texture latent with `tex_dit_512` on the shape latent
    /// coords, de-normalize with the texture vectors. The flow DiT is
    /// dropped before the de-normalization.
    pub fn sample_tex_slat(
        &self,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_tex_slat_stage(
            "tex_dit_512",
            Trellis2SlatConfig::tex_512(),
            cond,
            shape_slat_denorm,
            coords,
            noise,
            seed,
            steps,
            device,
        )
    }

    /// Texture SLAT stage (cascade/1024): same sampler params with the
    /// `tex_dit_1024` checkpoint (reference conditions the texture stage on
    /// `cond_1024`).
    pub fn sample_tex_slat_hr(
        &self,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        self.sample_tex_slat_stage(
            "tex_dit_1024",
            Trellis2SlatConfig::tex_1024(),
            cond,
            shape_slat_denorm,
            coords,
            None,
            seed,
            steps,
            device,
        )
    }

    /// One texture SLAT sampling pass: seeded (or injected) noise `[N, 32]`
    /// on `coords`, concatenated with the re-normalized shape SLAT
    /// (`noise || shape_norm`, reference concat order) -> Euler sampler ->
    /// `slat * tex_std + tex_mean`.
    #[allow(clippy::too_many_arguments)]
    fn sample_tex_slat_stage(
        &self,
        stage: &str,
        config: Trellis2SlatConfig,
        cond: &Tensor,
        shape_slat_denorm: &Tensor,
        coords: &[[i32; 3]],
        noise: Option<&Tensor>,
        seed: u64,
        steps: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if shape_slat_denorm
            .dim(0)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 shape latent has no rows: {source}"),
            })?
            != coords.len()
        {
            return Err(Lux3dError::ModelAssetResolution {
                message: format!(
                    "shape latent row count does not match the {} texture coords",
                    coords.len()
                ),
            });
        }
        let noise = match noise {
            Some(noise) => noise.clone(),
            None => {
                let mut rng = Trellis2Rng::new(seed ^ TRELLIS2_TEX_NOISE_SALT);
                let values = rng.randn(coords.len() * 32);
                Tensor::from_vec(values, (coords.len(), 32), device).map_err(|source| {
                    Lux3dError::CanonicalWeightsValidation {
                        message: format!("Trellis2 texture SLAT noise allocation failed: {source}"),
                    }
                })?
            }
        };
        let shape_slat_norm = self.normalize_shape_slat(shape_slat_denorm)?;
        let vb = self.stage_var_builder(stage, device)?;
        let dit = Trellis2TexDit::load(vb, &config).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load TRELLIS.2 {stage}: {source}"),
            }
        })?;
        let params = Trellis2SamplerParams {
            steps,
            ..self.config.tex_sampler.clone()
        };
        let (slat_norm, _trace) = sample_sparse(
            &|x: &Tensor, t_model: f32, c: &Tensor, cs: &[[i32; 3]]| {
                dit.forward(x, &shape_slat_norm, cs, t_model, c)
            },
            coords,
            &noise,
            cond,
            cond,
            &params,
        )
        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
            message: format!("Trellis2 texture SLAT sampling failed: {source}"),
        })?;
        drop(dit);
        self.denormalize_tex_slat(&slat_norm)
    }

    /// `slat * std + mean` with the package texture-normalization vectors.
    pub fn denormalize_tex_slat(&self, slat_norm: &Tensor) -> Result<Tensor> {
        let device = slat_norm.device();
        let mean = Tensor::from_vec(self.config.tex_slat_mean.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 mean vector allocation failed: {source}"),
            },
        )?;
        let std = Tensor::from_vec(self.config.tex_slat_std.clone(), (1, 32), device).map_err(
            |source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 std vector allocation failed: {source}"),
            },
        )?;
        slat_norm
            .broadcast_mul(&std)
            .and_then(|t| t.broadcast_add(&mean))
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 texture SLAT de-normalization failed: {source}"),
            })
    }

    /// Guided texture VAE decode (`decode_tex_slat`): `tex_dec` on the
    /// de-normalized texture latent with the shape decoder's raw C2S
    /// subdivision logits as guides; the decoded 6 channels are
    /// `raw * 0.5 + 0.5` clamped to [0, 1]. Returns the final voxel coords
    /// (equal to the shape decode's final coords) and the flat `[N, 6]`
    /// PBR attributes.
    pub fn decode_tex_attrs(
        &self,
        coords: &[[i32; 3]],
        tex_latent_denorm: &Tensor,
        guide_subs: &[Tensor],
        device: &Device,
    ) -> Result<(Vec<[i32; 3]>, Vec<f32>)> {
        let vb = self.stage_var_builder("tex_dec", device)?;
        let decoder = Trellis2TexDecoder::load(vb, &Trellis2ShapeVaeConfig::tex_decoder())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("failed to load TRELLIS.2 tex_dec: {source}"),
            })?;
        let decoded = decoder
            .decode(coords, tex_latent_denorm, guide_subs)
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 texture VAE decode failed: {source}"),
            })?;
        drop(decoder);
        let attrs = pbr_attrs_from_raw(&decoded.feats).map_err(|source| {
            Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 PBR postprocess failed: {source}"),
            }
        })?;
        Ok((decoded.coords, attrs))
    }

    /// Attach the texture stage products to assembled mesh buffers: the
    /// voxel payload switches to the 6-channel PBR attributes on the shape
    /// decode's final coords and the per-vertex base colors are sampled
    /// with the reference sparse trilinear (`query_vertex_attrs`).
    fn attach_texture(
        &self,
        mesh: &mut crate::geometry::Trellis2MeshBuffers,
        coords: &[[i32; 3]],
        attrs: &[f32],
        resolution: u32,
    ) -> Result<()> {
        let vertices: Vec<[f32; 3]> = mesh
            .vertices
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let colors =
            query_vertex_colors(&vertices, coords, attrs, resolution).map_err(|source| {
                Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 vertex color sampling failed: {source}"),
                }
            })?;
        mesh.voxel_attrs = attrs.to_vec();
        mesh.voxel_attr_channels = super::trellis2_tex::Trellis2PbrLayout::CHANNELS;
        mesh.vertex_colors = Some(colors);
        mesh.validate()
    }

    fn assemble_mesh(
        &self,
        resolution: u32,
        mesh: DualGridMesh,
        coords: &[[i32; 3]],
        feats: &Tensor,
    ) -> Result<crate::geometry::Trellis2MeshBuffers> {
        let mut voxel_coords = Vec::with_capacity(coords.len() * 3);
        for coord in coords {
            voxel_coords.extend_from_slice(&[coord[0], coord[1], coord[2]]);
        }
        let attrs = feats
            .to_device(&Device::Cpu)
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 decoded feats download failed: {source}"),
            })?;
        let attr_channels = if coords.is_empty() {
            7
        } else {
            attrs.len() / coords.len()
        };
        self.geometry.assemble(
            resolution,
            mesh.vertices,
            mesh.faces,
            voxel_coords,
            attrs,
            attr_channels,
        )
    }

    /// Core execution dispatch: the 512 path or the 1024 cascade. `cond_hr`
    /// is the cascade's 1024 conditioning (`None` -> reuse `cond`, the
    /// documented `run_with_latents` behavior). `tex_latent_override` is an
    /// injected de-normalized texture latent (the `run_with_latents`
    /// `tex_init` surface; skipped when `with_texture` is false).
    #[allow(clippy::too_many_arguments)]
    fn run_core(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        cond_hr: Option<&Tensor>,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        shape_noise: Option<&Tensor>,
        shape_latent_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        match options.pipeline_type {
            Trellis2PipelineType::R512 => self.run_core_512(
                cond,
                neg_cond,
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
            ),
            Trellis2PipelineType::Cascade1024 => self.run_core_cascade_1024(
                cond,
                neg_cond,
                cond_hr.unwrap_or(cond),
                ss_noise,
                coords_override,
                shape_noise,
                shape_latent_override,
                tex_latent_override,
                device,
                options,
            ),
            other => Err(self.resolutions_error(other)),
        }
    }

    /// 512 core: SS stage (unless `coords_override`), shape SLAT stage
    /// (unless `shape_latent_override`), texture SLAT stage (unless
    /// `tex_latent_override` or `with_texture = false`), VAE decode + mesh.
    #[allow(clippy::too_many_arguments)]
    fn run_core_512(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        shape_noise: Option<&Tensor>,
        shape_latent_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        let (ss_resolution, shape_resolution) = self.resolutions(options)?;
        let (z_s, ss_coords) = match coords_override {
            Some(coords) => (None, coords),
            None => {
                let (z_s, coords) = self.sample_sparse_structure(
                    cond,
                    neg_cond,
                    ss_noise,
                    options.seed,
                    options.ss_steps,
                    device,
                    ss_resolution,
                )?;
                (Some(z_s), coords)
            }
        };
        if ss_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 sparse structure decoded to zero active voxels",
            ));
        }
        let shape_latent = match shape_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 shape latent has no rows: {source}"),
                        })?;
                if rows != ss_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "shape latent row count {rows} does not match the active voxel count {}",
                            ss_coords.len()
                        ),
                    });
                }
                latent.clone()
            }
            None => self.sample_shape_slat(
                cond,
                neg_cond,
                &ss_coords,
                shape_noise,
                options.seed,
                options.shape_steps,
                device,
            )?,
        };
        // Texture SLAT on the shape latent coords (reference order: after
        // the shape stage, before the decodes).
        let tex_latent = self.texture_latent_512(
            cond,
            &shape_latent,
            &ss_coords,
            tex_latent_override,
            device,
            options,
        )?;
        let (decoded, mesh) =
            self.decode_shape_vae(&ss_coords, &shape_latent, device, shape_resolution)?;
        let mut mesh =
            self.assemble_mesh(shape_resolution, mesh, &decoded.coords, &decoded.feats)?;
        let tex_attrs = match &tex_latent {
            Some(tex_latent) => {
                let (tex_coords, attrs) =
                    self.decode_tex_attrs(&ss_coords, tex_latent, &decoded.subdiv_logits, device)?;
                assert_eq!(
                    tex_coords, decoded.coords,
                    "guided texture decode coords must equal the shape decode coords"
                );
                self.attach_texture(&mut mesh, &decoded.coords, &attrs, shape_resolution)?;
                Some(attrs)
            }
            None => None,
        };
        Ok(Trellis2Stages {
            cond: cond.clone(),
            neg_cond: neg_cond.clone(),
            z_s,
            ss_coords: ss_coords.clone(),
            shape_coords: ss_coords,
            shape_latent,
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: None,
            hr_upsampled_coords: None,
            quant_coords: None,
            hr_resolution: None,
            subdiv_logits: decoded.subdiv_logits,
            tex_latent,
            tex_attrs,
        })
    }

    /// Shared texture-latent step for the 512 core: `None` when
    /// `with_texture` is off; the injected de-normalized latent when
    /// provided (validated against the coord count); sampled otherwise.
    #[allow(clippy::too_many_arguments)]
    fn texture_latent_512(
        &self,
        cond: &Tensor,
        shape_latent: &Tensor,
        coords: &[[i32; 3]],
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Option<Tensor>> {
        if !options.with_texture {
            return Ok(None);
        }
        match tex_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 texture latent has no rows: {source}"),
                        })?;
                if rows != coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "texture latent row count {rows} does not match the shape latent \
                             coord count {}",
                            coords.len()
                        ),
                    });
                }
                Ok(Some(latent.clone()))
            }
            None => Ok(Some(self.sample_tex_slat(
                cond,
                shape_latent,
                coords,
                None,
                options.seed,
                options.tex_steps,
                device,
            )?)),
        }
    }

    /// 1024 cascade core (reference `run(...)`, `pipeline_type =
    /// '1024_cascade'`): SS stage at 32^3 on `cond` (the reference
    /// conditions the SS stage on `cond_512`), LR shape SLAT (unless
    /// `shape_latent_override` — the injected de-normalized LR slat), shape
    /// VAE `upsample(slat, 4)` -> HR quantization + token budget, HR shape
    /// SLAT on `cond_hr`, texture SLAT on `cond_hr` with `tex_dit_1024` on
    /// the HR quantized coords, decode at `hr_resolution` + mesh. Model
    /// residency stays sequential: LR DiT -> drop, shape VAE -> drop, HR
    /// DiT -> drop, tex DiT -> drop, shape VAE (decode) -> drop, tex VAE.
    #[allow(clippy::too_many_arguments)]
    fn run_core_cascade_1024(
        &self,
        cond: &Tensor,
        neg_cond: &Tensor,
        cond_hr: &Tensor,
        ss_noise: Option<&Tensor>,
        coords_override: Option<Vec<[i32; 3]>>,
        lr_noise: Option<&Tensor>,
        lr_slat_override: Option<&Tensor>,
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        let (ss_resolution, _) = self.resolutions(options)?;
        let (z_s, ss_coords) = match coords_override {
            Some(coords) => (None, coords),
            None => {
                let (z_s, coords) = self.sample_sparse_structure(
                    cond,
                    neg_cond,
                    ss_noise,
                    options.seed,
                    options.ss_steps,
                    device,
                    ss_resolution,
                )?;
                (Some(z_s), coords)
            }
        };
        if ss_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 sparse structure decoded to zero active voxels",
            ));
        }
        // LR shape SLAT: injected de-normalized latent or sampled at 32^3.
        let lr_slat = match lr_slat_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 shape latent has no rows: {source}"),
                        })?;
                if rows != ss_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "LR shape latent row count {rows} does not match the active voxel count {}",
                            ss_coords.len()
                        ),
                    });
                }
                latent.clone()
            }
            None => self.sample_shape_slat(
                cond,
                neg_cond,
                &ss_coords,
                lr_noise,
                options.seed,
                options.shape_steps,
                device,
            )?,
        };
        let (hr_upsampled_coords, quant_coords, hr_resolution) = self
            .upsample_and_quantize_shape_slat(
                &ss_coords,
                &lr_slat,
                options.max_num_tokens,
                device,
            )?;
        if quant_coords.is_empty() {
            return Err(Lux3dError::InvalidInput(
                "Trellis2 cascade quantized the upsampled coords to zero HR tokens",
            ));
        }
        let hr_slat = self.sample_shape_slat_hr(
            cond_hr,
            &quant_coords,
            options.seed,
            options.shape_steps,
            device,
        )?;
        // Texture SLAT on the HR quantized coords, conditioned on `cond_hr`
        // (reference texture stage uses the 1024 conditioning + DiT).
        let tex_latent = self.texture_latent_cascade(
            cond_hr,
            &hr_slat,
            &quant_coords,
            tex_latent_override,
            device,
            options,
        )?;
        let (decoded, mesh) =
            self.decode_shape_vae(&quant_coords, &hr_slat, device, hr_resolution)?;
        let mut mesh = self.assemble_mesh(hr_resolution, mesh, &decoded.coords, &decoded.feats)?;
        let tex_attrs = match &tex_latent {
            Some(tex_latent) => {
                let (tex_coords, attrs) = self.decode_tex_attrs(
                    &quant_coords,
                    tex_latent,
                    &decoded.subdiv_logits,
                    device,
                )?;
                assert_eq!(
                    tex_coords, decoded.coords,
                    "guided texture decode coords must equal the shape decode coords"
                );
                self.attach_texture(&mut mesh, &decoded.coords, &attrs, hr_resolution)?;
                Some(attrs)
            }
            None => None,
        };
        Ok(Trellis2Stages {
            cond: cond.clone(),
            neg_cond: neg_cond.clone(),
            z_s,
            ss_coords,
            shape_coords: quant_coords.clone(),
            shape_latent: hr_slat,
            decoded_feats: decoded.feats,
            mesh: Some(mesh),
            lr_shape_latent: Some(lr_slat),
            hr_upsampled_coords: Some(hr_upsampled_coords),
            quant_coords: Some(quant_coords),
            hr_resolution: Some(hr_resolution),
            subdiv_logits: decoded.subdiv_logits,
            tex_latent,
            tex_attrs,
        })
    }

    /// Shared texture-latent step for the cascade core: identical guards to
    /// [`Self::texture_latent_512`] with the 1024 texture stage.
    #[allow(clippy::too_many_arguments)]
    fn texture_latent_cascade(
        &self,
        cond_hr: &Tensor,
        hr_slat: &Tensor,
        quant_coords: &[[i32; 3]],
        tex_latent_override: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Option<Tensor>> {
        if !options.with_texture {
            return Ok(None);
        }
        match tex_latent_override {
            Some(latent) => {
                let rows =
                    latent
                        .dim(0)
                        .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                            message: format!("Trellis2 texture latent has no rows: {source}"),
                        })?;
                if rows != quant_coords.len() {
                    return Err(Lux3dError::ModelAssetResolution {
                        message: format!(
                            "texture latent row count {rows} does not match the HR coord \
                             count {}",
                            quant_coords.len()
                        ),
                    });
                }
                Ok(Some(latent.clone()))
            }
            None => Ok(Some(self.sample_tex_slat_hr(
                cond_hr,
                hr_slat,
                quant_coords,
                options.seed,
                options.tex_steps,
                device,
            )?)),
        }
    }

    /// The shared "pipeline type not implemented" error.
    fn resolutions_error(&self, pipeline_type: Trellis2PipelineType) -> Lux3dError {
        Lux3dError::ModelAssetResolution {
            message: format!(
                "Trellis2 pipeline type `{}` is not implemented in this slice (the 1024 \
                 cascade is implemented); use `512` or `1024_cascade`",
                pipeline_type.as_str()
            ),
        }
    }

    /// Run the pipeline on an image path: preprocess -> DINOv3 -> SS -> SLAT
    /// -> VAE decode -> mesh.
    pub fn run_from_path(
        &self,
        path: &Path,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<crate::contracts::Trellis2Mesh> {
        let stages = self.run_from_path_with_stages(path, device, options)?;
        let mesh = stages.mesh.expect("run_from_path always produces a mesh");
        mesh.to_contract()
    }

    /// [`Self::run_from_path`] returning the full per-stage outputs. The
    /// cascade branch prepares the source twice (512 and 1024 DINOv3 inputs)
    /// — the crop geometry is deterministic, so this reproduces the
    /// reference's single-crop/two-resize `cond_512` + `cond_1024` exactly.
    pub fn run_from_path_with_stages(
        &self,
        path: &Path,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        if !options.preprocess {
            return Err(Lux3dError::InvalidInput(
                "run_from_path requires options.preprocess = true; use run_prepared for \
                 pre-prepared images",
            ));
        }
        if options.pipeline_type == Trellis2PipelineType::Cascade1024 {
            let preprocess_hr = self.cascade_preprocess_hr()?;
            let prepared = self.preprocess.prepare_inputs_from_path(path)?;
            let prepared_hr = preprocess_hr.prepare_inputs_from_path(path)?;
            let (cond, neg_cond) = self.image_conditioning(&prepared, device)?;
            let pixel_values_hr = preprocess_hr
                .normalized_pixel_values(&prepared_hr, device)
                .map_err(|source| Lux3dError::CanonicalWeightsValidation {
                    message: format!("Trellis2 DINOv3 input normalization failed: {source}"),
                })?;
            let (cond_hr, _neg_hr) = self.dino_conditioning_pixels(&pixel_values_hr, device)?;
            return self.run_core(
                &cond,
                &neg_cond,
                Some(&cond_hr),
                None,
                None,
                None,
                None,
                None,
                device,
                options,
            );
        }
        let prepared = self.preprocess.prepare_inputs_from_path(path)?;
        self.run_prepared(&prepared, device, options)
    }

    /// Run the pipeline on an already-preprocessed image.
    ///
    /// Cascade1024 note: the prepared image only carries the 512 crop, so
    /// `cond_1024` is built by LANCZOS-resizing the prepared pixels to 1024
    /// (a documented double-resample approximation —
    /// [`Self::run_from_path_with_stages`] re-prepares from the source and
    /// is exact).
    pub fn run_prepared(
        &self,
        prepared: &crate::contracts::Trellis2PreparedImage,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<Trellis2Stages> {
        let (cond, neg_cond) = self.image_conditioning(prepared, device)?;
        let cond_hr = match options.pipeline_type {
            Trellis2PipelineType::Cascade1024 => {
                let (cond_hr, _neg) = self.cascade_cond_1024_from_prepared(prepared, device)?;
                Some(cond_hr)
            }
            _ => None,
        };
        self.run_core(
            &cond,
            &neg_cond,
            cond_hr.as_ref(),
            None,
            None,
            None,
            None,
            None,
            device,
            options,
        )
    }

    /// Run with injected stage tensors (frozen test surface):
    ///
    /// * `ss_init`: sparse-structure flow noise `[1, 8, R, R, R]`.
    /// * `shape_init`: de-normalized shape latent `[N, 32]` on the SS coords
    ///   — when provided the LR shape SLAT sampler is skipped. `512` decodes
    ///   it directly; `1024_cascade` treats it as the LR slat (upsample ->
    ///   quantize -> HR sample -> decode; the row count must match the
    ///   decoded coords).
    /// * `tex_init`: de-normalized texture latent `[N, 32]` on the shape
    ///   latent coords — when provided the texture SLAT sampler is skipped
    ///   (`512` decodes it on the SS coords; `1024_cascade` on the HR
    ///   quantized coords). Requires `with_texture`; must be `None` when
    ///   texture is off.
    ///
    /// Cascade1024 note: the frozen surface carries a single `cond`, so the
    /// injected cond feeds both the LR (`cond_512`) and HR (`cond_1024`)
    /// stages — cross-attention accepts any context length and the injected
    /// parity runs supply the cond they compare against.
    pub fn run_with_latents(
        &self,
        cond: &Tensor,
        ss_init: &Tensor,
        shape_init: Option<&Tensor>,
        tex_init: Option<&Tensor>,
        device: &Device,
        options: &Trellis2RunOptions,
    ) -> Result<crate::contracts::Trellis2Mesh> {
        if tex_init.is_some() && !options.with_texture {
            return Err(Lux3dError::InvalidInput(
                "tex_init requires options.with_texture = true (texture is disabled)",
            ));
        }
        if !matches!(options.pipeline_type, Trellis2PipelineType::R512) {
            self.resolutions(options)?;
        }
        let neg_cond =
            Tensor::zeros_like(cond).map_err(|source| Lux3dError::CanonicalWeightsValidation {
                message: format!("Trellis2 negative cond allocation failed: {source}"),
            })?;
        let stages = self.run_core(
            cond,
            &neg_cond,
            None,
            Some(ss_init),
            None,
            None,
            shape_init,
            tex_init,
            device,
            options,
        )?;
        let mesh = stages
            .mesh
            .expect("run_with_latents always produces a mesh");
        mesh.to_contract()
    }

    /// Write the mesh buffers with the TRELLIS.2 export stage.
    pub fn export_mesh(
        &self,
        mesh: &crate::geometry::Trellis2MeshBuffers,
        path: &Path,
    ) -> Result<crate::export::MeshFormat> {
        self.export.write_mesh(mesh, path)
    }

    /// Write the mesh plus the PBR voxel sidecar pair
    /// (`<stem>.pbrvox.bin` + `<stem>.pbr.json`); returns the binary
    /// sidecar path. Fails for geometry-only meshes (7-channel shape
    /// attrs) — run with `with_texture` first.
    pub fn export_mesh_with_pbr(
        &self,
        mesh: &crate::geometry::Trellis2MeshBuffers,
        path: &Path,
    ) -> Result<std::path::PathBuf> {
        self.export.write_mesh_with_pbr(mesh, path)
    }
}
