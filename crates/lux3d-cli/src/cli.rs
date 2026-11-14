use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use lux3d_core::runtime::{
    Pi3Pipeline, Pi3xInjectConditions, Pi3xPipeline, Pi3xVoPipeline, Trellis2Pipeline,
    Trellis2RunOptions, TripoSgPipeline, TripoSgRunOptions, TripoSrPipeline, VggtPipeline,
};
use lux3d_core::{
    ModelAssetOptions, ModelFamily, QuantizeOptions, Trellis2PipelineType, WeightLocator,
    dtype_from_name, ensure_canonical_weights, quantize_auxiliary_checkpoint, quantize_checkpoint,
};

#[derive(Debug, Parser, PartialEq)]
#[command(
    name = "lux3d",
    about = "CUDA-first runtime and weight tools for Pi3, Pi3X, TripoSR, VGGT, TripoSG, and TRELLIS.2."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand, PartialEq)]
pub enum Command {
    Inspect(InspectArgs),
    Weights(WeightsArgs),
    Run(RunArgs),
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct InspectArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(value_enum)]
    pub family: Family,
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct WeightsArgs {
    #[command(subcommand)]
    pub command: WeightsCommand,
}

#[derive(Debug, Clone, Subcommand, PartialEq, Eq)]
pub enum WeightsCommand {
    Normalize(NormalizeArgs),
    Quantize(QuantizeArgs),
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct QuantizeArgs {
    #[arg(
        value_enum,
        required_unless_present = "aux_family",
        help = "Output model family (triposg, vggt, ...); omit when using `--aux-family`."
    )]
    pub family: Option<Family>,

    #[arg(
        long,
        conflicts_with = "family",
        help = "Auxiliary package family label, e.g. `rmbg`; makes FAMILY optional."
    )]
    pub aux_family: Option<String>,

    #[arg(long, help = "Input canonical safetensors (`model.safetensors`).")]
    pub source: PathBuf,

    #[arg(
        long,
        help = "Output GGUF path; use the package's `model.gguf` name for auto-detection."
    )]
    pub output: PathBuf,

    #[arg(
        long,
        default_value = "q8_0",
        help = "Quantization dtype for 2D linear weights (f16, q8_0, q4_0, q4_k, q5_k, q6_k, ...)."
    )]
    pub dtype: String,

    #[arg(
        long,
        default_value = "f16",
        help = "Dtype for norms, biases, embeddings, convolutions, and non-divisible weights."
    )]
    pub fallback_dtype: String,

    #[arg(
        long,
        help = "Quantize every rank-2 tensor instead of only names ending in `.weight`."
    )]
    pub all_2d: bool,

    #[arg(
        long,
        help = "Only quantize tensors whose name starts with this prefix (repeatable)."
    )]
    pub prefix: Vec<String>,
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct NormalizeArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(
        long,
        help = "Path to the upstream raw model directory used only for maintainer-side canonicalization."
    )]
    pub raw_model_dir: PathBuf,

    #[arg(long)]
    pub output_dir: PathBuf,

    #[arg(value_enum)]
    pub family: Family,
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct RunArgs {
    #[arg(value_enum)]
    pub family: Family,

    #[arg(long, value_enum, default_value = "cuda")]
    pub device: DeviceBackend,

    #[arg(long)]
    pub source: PathBuf,

    #[arg(long)]
    pub interval: Option<usize>,

    #[arg(long)]
    pub conditions: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub vo: bool,

    #[arg(long)]
    pub chunk_size: Option<usize>,

    #[arg(long)]
    pub overlap: Option<usize>,

    #[arg(long)]
    pub conf_threshold: Option<f32>,

    #[arg(long, value_delimiter = ',')]
    pub inject_condition: Vec<InjectCondition>,

    #[arg(long)]
    pub mc_resolution: Option<u32>,

    #[arg(long)]
    pub mc_threshold: Option<f32>,

    #[arg(
        long,
        help = "Denoise steps (TripoSG rectified-flow default 50; TRELLIS.2 sparse-structure/shape/texture Euler default 12 each)."
    )]
    pub steps: Option<usize>,

    #[arg(long, help = "TripoSG classifier-free guidance scale (default 7.0).")]
    pub guidance_scale: Option<f32>,

    #[arg(
        long,
        help = "TripoSG/TRELLIS.2 latent-initialization seed (default 42)."
    )]
    pub seed: Option<u64>,

    #[arg(long, help = "TripoSG latent token count (default 2048).")]
    pub num_tokens: Option<usize>,

    #[arg(
        long,
        help = "TripoSG dense octree depth of the hierarchical SDF grid (default 8)."
    )]
    pub dense_octree_depth: Option<u32>,

    #[arg(long, help = "TripoSG final hierarchical octree depth (default 9).")]
    pub hierarchical_octree_depth: Option<u32>,

    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "true",
        help = "TripoSG flash decoder + DiffDMC path (reference default: true). Pass --flash-decoder false to use dense/hierarchical marching cubes."
    )]
    pub flash_decoder: Option<bool>,

    #[arg(
        long,
        help = "TripoSG flash octree depth used by the reference default flash path (default 9)."
    )]
    pub flash_octree_depth: Option<u32>,

    #[arg(
        long,
        value_enum,
        help = "TRELLIS.2 pipeline type: `512`, `1024`, or `1024_cascade` (reference default). The single-resolution `1024` path is not implemented yet."
    )]
    pub pipeline_type: Option<Trellis2PipelineKind>,

    #[arg(
        long,
        help = "TRELLIS.2 cascade token budget; the reference loop always decodes at 1024 (default 49152)."
    )]
    pub max_num_tokens: Option<usize>,

    #[arg(
        long,
        num_args = 0..=1,
        default_missing_value = "true",
        help = "TRELLIS.2 texture stages: texture SLAT + guided VAE + PBR vertex colors (reference default: true). Pass --texture false for geometry-only output."
    )]
    pub texture: Option<bool>,

    #[arg(
        long,
        help = "Path to a canonical package directory, a directory containing `model.gguf`, or a standalone `.gguf` file."
    )]
    pub model_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Path to the separate RMBG-1.4 package directory or its `.gguf` file (required for inputs without a valid alpha channel)."
    )]
    pub rmbg_model_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Override the Hugging Face cache root used when --model-path is not supplied."
    )]
    pub cache_dir: Option<PathBuf>,

    #[arg(
        long,
        help = "Output artifact path. Mesh families dispatch on extension: `.glb` (default), `.obj`, `.ply`, `.stl`. Point clouds use `.ply` (default) or `.xyz`."
    )]
    pub output: PathBuf,

    #[arg(
        long,
        help = "Dump raw inference tensors (.npz) for quantization-fidelity metrics (vggt only)."
    )]
    pub dump_npz: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Family {
    Pi3,
    Pi3x,
    Triposr,
    Triposg,
    Vggt,
    /// TRELLIS.2-4B image-to-3D: preprocess -> DINOv3 -> SS flow -> shape SLAT
    /// (512 or 1024 cascade) -> texture SLAT -> dual-grid mesh + PBR sidecar.
    Trellis2,
}

/// TRELLIS.2 pipeline variant (`--pipeline-type`); names mirror the reference
/// config strings (`512`, `1024`, `1024_cascade`).
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Trellis2PipelineKind {
    #[value(name = "512")]
    R512,
    #[value(name = "1024")]
    R1024,
    #[value(name = "1024_cascade")]
    Cascade1024,
}

impl From<Trellis2PipelineKind> for Trellis2PipelineType {
    fn from(kind: Trellis2PipelineKind) -> Self {
        match kind {
            Trellis2PipelineKind::R512 => Self::R512,
            Trellis2PipelineKind::R1024 => Self::R1024,
            Trellis2PipelineKind::Cascade1024 => Self::Cascade1024,
        }
    }
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum InjectCondition {
    Pose,
    Depth,
    Ray,
    Intrinsic,
}

/// Backend device used to run inference.
#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum DeviceBackend {
    Cpu,
    Cuda,
    Wgpu,
    Vulkan,
}

impl DeviceBackend {
    /// Resolve the backend into a candle-core device.
    pub fn to_device(self) -> anyhow::Result<candle_core::Device> {
        match self {
            DeviceBackend::Cpu => Ok(candle_core::Device::Cpu),
            DeviceBackend::Cuda => Ok(candle_core::Device::new_cuda(0)?),
            DeviceBackend::Wgpu => Ok(candle_core::Device::new_wgpu(0)?),
            DeviceBackend::Vulkan => Ok(candle_core::Device::new_vulkan(0)?),
        }
    }
}

/// Human-readable device identity for demos/logs (GPU name, not just backend).
fn describe_device(backend: DeviceBackend, device: &candle_core::Device) -> String {
    let loc = device.location();
    match backend {
        DeviceBackend::Cpu => "[device] backend=cpu".to_string(),
        DeviceBackend::Cuda => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Cuda { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "cuda")]
            {
                match device.as_cuda_device() {
                    Ok(cuda) => {
                        let stream = cuda.cuda_stream();
                        let ctx = stream.context();
                        let name = ctx.name().unwrap_or_else(|_| format!("cuda:{gpu_id}"));
                        let cc = ctx
                            .compute_capability()
                            .map(|(maj, min)| format!("{maj}.{min}"))
                            .unwrap_or_else(|_| "?".to_string());
                        let mem = ctx
                            .total_mem()
                            .map(|bytes| {
                                format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                            })
                            .unwrap_or_else(|_| "?".to_string());
                        return format!(
                            "[device] backend=cuda gpu=\"{name}\" compute={cc} mem={mem} id={gpu_id}"
                        );
                    }
                    Err(_) => {}
                }
            }
            format!("[device] backend=cuda id={gpu_id}")
        }
        DeviceBackend::Wgpu => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Wgpu { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "wgpu")]
            {
                if let Ok(wgpu) = device.as_wgpu_device() {
                    let driver = wgpu.adapter_driver();
                    let driver_info = wgpu.adapter_driver_info();
                    let driver_bit = if driver.is_empty() && driver_info.is_empty() {
                        String::new()
                    } else if driver_info.is_empty() {
                        format!(" driver=\"{driver}\"")
                    } else if driver.is_empty() {
                        format!(" driver_info=\"{driver_info}\"")
                    } else {
                        format!(" driver=\"{driver}\" driver_info=\"{driver_info}\"")
                    };
                    return format!(
                        "[device] backend=wgpu gpu=\"{}\" api={}{driver_bit} id={gpu_id}",
                        wgpu.adapter_name(),
                        wgpu.adapter_backend(),
                    );
                }
            }
            format!("[device] backend=wgpu id={gpu_id}")
        }
        DeviceBackend::Vulkan => {
            let gpu_id = match loc {
                candle_core::DeviceLocation::Vulkan { gpu_id } => gpu_id,
                _ => 0,
            };
            #[cfg(feature = "vulkan")]
            {
                if let Ok(vk) = device.as_vulkan_device() {
                    return format!(
                        "[device] backend=vulkan gpu=\"{}\" type={:?} id={gpu_id}",
                        vk.physical_device_name(),
                        vk.physical_device_type(),
                    );
                }
            }
            format!("[device] backend=vulkan id={gpu_id}")
        }
    }
}

impl Family {
    pub const fn model_family(self) -> ModelFamily {
        match self {
            Self::Pi3 => ModelFamily::Pi3,
            Self::Pi3x => ModelFamily::Pi3x,
            Self::Triposr => ModelFamily::TripoSr,
            Self::Triposg => ModelFamily::TripoSg,
            Self::Vggt => ModelFamily::Vggt,
            Self::Trellis2 => ModelFamily::Trellis2,
        }
    }
}

impl Command {
    pub fn family_name(&self) -> &str {
        match self {
            Self::Inspect(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
                Family::Triposg => "triposg",
                Family::Vggt => "vggt",
                Family::Trellis2 => "trellis2",
            },
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => match family.family {
                    Family::Pi3 => "pi3",
                    Family::Pi3x => "pi3x",
                    Family::Triposr => "triposr",
                    Family::Triposg => "triposg",
                    Family::Vggt => "vggt",
                    Family::Trellis2 => "trellis2",
                },
                WeightsCommand::Quantize(ref args) => {
                    if let Some(aux) = args.aux_family.as_deref() {
                        return aux;
                    }
                    match args.family {
                        Some(Family::Pi3) => "pi3",
                        Some(Family::Pi3x) => "pi3x",
                        Some(Family::Triposr) => "triposr",
                        Some(Family::Triposg) => "triposg",
                        Some(Family::Vggt) => "vggt",
                        Some(Family::Trellis2) => "trellis2",
                        None => "aux",
                    }
                }
            },
            Self::Run(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
                Family::Triposg => "triposg",
                Family::Vggt => "vggt",
                Family::Trellis2 => "trellis2",
            },
        }
    }

    pub fn family(&self) -> Option<ModelFamily> {
        match self {
            Self::Inspect(args) => Some(args.family.model_family()),
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => Some(family.family.model_family()),
                WeightsCommand::Quantize(ref args) => args.family.map(Family::model_family),
            },
            Self::Run(args) => Some(args.family.model_family()),
        }
    }

    pub fn repo_root(&self) -> Option<&PathBuf> {
        match self {
            Self::Inspect(args) => Some(&args.repo_root),
            Self::Weights(args) => match &args.command {
                WeightsCommand::Normalize(family) => Some(&family.repo_root),
                WeightsCommand::Quantize(_) => None,
            },
            Self::Run(_) => None,
        }
    }
}

pub fn inspect_model(repo_root: PathBuf, family: ModelFamily) -> anyhow::Result<String> {
    let spec = lux3d_core::ModelSpec::inspect(repo_root, family)?;
    Ok(serde_json::to_string_pretty(&spec)?)
}

pub fn quantize_weights(args: QuantizeArgs) -> anyhow::Result<String> {
    let dtype = dtype_from_name(&args.dtype)
        .ok_or_else(|| anyhow::anyhow!("unknown quantization dtype `{}`", args.dtype))?;
    let fallback = dtype_from_name(&args.fallback_dtype).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown fallback quantization dtype `{}`",
            args.fallback_dtype
        )
    })?;
    let report = if let Some(aux_family) = args.aux_family.as_deref() {
        quantize_auxiliary_checkpoint(
            &args.source,
            &args.output,
            aux_family,
            QuantizeOptions {
                dtype,
                fallback,
                all_2d: args.all_2d,
                prefixes: args.prefix.clone(),
            },
        )?
    } else {
        let family = args
            .family
            .ok_or_else(|| anyhow::anyhow!("FAMILY is required unless `--aux-family` is used"))?;
        quantize_checkpoint(
            &args.source,
            &args.output,
            family.model_family(),
            QuantizeOptions {
                dtype,
                fallback,
                all_2d: args.all_2d,
                prefixes: args.prefix.clone(),
            },
        )?
    };
    Ok(report.render())
}

pub fn normalize_weights(
    repo_root: PathBuf,
    family: ModelFamily,
    raw_model_dir: PathBuf,
    output_dir: PathBuf,
) -> anyhow::Result<String> {
    let python = python_executable(&repo_root, family);
    let script = normalizer_script(&repo_root)?;

    let family_name = family.as_str();
    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg("--family")
        .arg(family_name)
        .arg("--raw-model-dir")
        .arg(&raw_model_dir)
        .arg("--output-dir")
        .arg(&output_dir)
        .output()
        .map_err(|source| {
            anyhow::anyhow!(
                "failed to start Python interpreter `{}` for `{family_name}`: {source}; \
                 install Python or set LUX3D_PYTHON",
                python.display()
            )
        })?;

    if !output.status.success() {
        anyhow::bail!(
            "python normalizer failed for `{family_name}`:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let locator = WeightLocator::new(raw_model_dir, output_dir);
    let plan = locator.locate(family)?;
    let canonical = ensure_canonical_weights(&plan)?;
    Ok(serde_json::to_string_pretty(&canonical.manifest)?)
}

pub fn run_model(args: RunArgs) -> anyhow::Result<PathBuf> {
    let model_assets = ModelAssetOptions {
        canonical_dir: args.model_path.clone(),
        cache_dir: args.cache_dir.clone(),
    };
    let device = args.device.to_device()?;
    eprintln!("{}", describe_device(args.device, &device));
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    if args.family.model_family() != ModelFamily::Vggt && args.dump_npz.is_some() {
        anyhow::bail!("`--dump-npz` is only supported for `vggt`");
    }

    match args.family.model_family() {
        ModelFamily::Pi3 => {
            reject_triposg_only_options(&args, args.family)?;
            run_pi3(
                model_assets,
                &args.source,
                args.interval,
                &args.output,
                &device,
            )?
        }
        ModelFamily::Pi3x => {
            reject_triposg_only_options(&args, args.family)?;
            run_pi3x(
                model_assets,
                &args.source,
                args.conditions.as_deref(),
                args.interval,
                args.vo,
                args.chunk_size,
                args.overlap,
                args.conf_threshold,
                &args.inject_condition,
                &args.output,
                &device,
            )?
        }
        ModelFamily::TripoSr => {
            reject_triposg_only_options(&args, args.family)?;
            run_triposr(
                model_assets,
                &args.source,
                args.mc_resolution.unwrap_or(256),
                args.mc_threshold.unwrap_or(25.0),
                &args.output,
                &device,
            )?
        }
        ModelFamily::TripoSg => {
            reject_unsupported_triposg_options(&args)?;
            let options = triposg_run_options_from_args(&args)?;
            let rmbg_model_path = resolve_rmbg_model_path(&args);
            run_triposg(
                model_assets,
                rmbg_model_path,
                &args.source,
                options,
                &args.output,
                &device,
            )?
        }
        ModelFamily::Vggt => {
            reject_unsupported_vggt_options(&args)?;
            reject_triposg_only_options(&args, args.family)?;
            run_vggt(
                model_assets,
                &args.source,
                args.interval,
                args.conf_threshold,
                &args.output,
                args.dump_npz.as_deref(),
                &device,
            )?
        }
        ModelFamily::Trellis2 => {
            reject_unsupported_trellis2_options(&args)?;
            let options = trellis2_run_options_from_args(&args)?;
            run_trellis2(model_assets, &args.source, options, &args.output, &device)?
        }
    }

    // Backend CPU-fallback observability: nonzero values = hidden host/CPU compute.
    match args.device {
        DeviceBackend::Wgpu => {
            eprintln!(
                "[candle-obs] wgpu cpu_fallback={} host_compute={}",
                candle_core::wgpu_cpu_fallback_count(),
                candle_core::wgpu_host_compute_count(),
            );
            #[cfg(feature = "wgpu")]
            if let Some(rows) = candle_core::wgpu_gpu_profile_report() {
                let _ = device.synchronize();
                eprintln!("[gpu-profile] wgpu kernels={}", rows.len());
                for (name, count, total_ms) in rows.iter().take(40) {
                    eprintln!(
                        "[gpu-profile] {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                        total_ms * 1000.0 / (*count as f64)
                    );
                }
            }
        }
        #[cfg(feature = "vulkan")]
        DeviceBackend::Vulkan => {
            eprintln!(
                "[candle-obs] vulkan cpu_fallback={} host_compute={}",
                candle_core::vulkan_cpu_fallback_count(),
                candle_core::vulkan_host_compute_count(),
            );
            if std::env::var_os("LUX3D_GPU_PROFILE").is_some() {
                device.synchronize()?;
                if let Some((wall, rows)) = candle_core::vulkan_gpu_profile_report() {
                    eprintln!(
                        "[gpu-profile] wall={wall:.2}s (aggregation start -> report) kernels={}",
                        rows.len()
                    );
                    for (name, count, total_ms) in rows.iter().take(50) {
                        eprintln!(
                            "[gpu-profile] {name:44} {count:>7}x {total_ms:>10.1}ms {:>8.1}us/ea",
                            total_ms * 1000.0 / (*count as f64)
                        );
                    }
                }
            }
            if let Some(rows) = candle_core::vulkan_flush_reason_report() {
                eprintln!("[cpu-profile] vulkan flush reasons:");
                for (name, count) in rows {
                    eprintln!("[cpu-profile] flush:{name:28} {count:>8}x");
                }
            }
            if let Some(rows) = candle_core::vulkan_cpu_profile_report() {
                eprintln!("[cpu-profile] vulkan dispatch hot path:");
                for (name, count, total_ms) in rows {
                    eprintln!(
                        "[cpu-profile] {name:24} {count:>8}x {total_ms:>10.1}ms {:>8.1}us/ea",
                        total_ms * 1000.0 / (count as f64)
                    );
                }
            }
        }
        _ => {}
    }

    Ok(args.output)
}

fn run_pi3(
    model_assets: ModelAssetOptions,
    source: &Path,
    interval: Option<usize>,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = Pi3Pipeline::load(model_assets)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path_with_interval(source, interval, device)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    pipeline.export_point_cloud(&inference, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

/// VGGT exposes point/depth/camera inference only. Reject Pi3X VO and
/// TripoSR meshing options instead of silently ignoring them.
fn reject_unsupported_vggt_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `vggt`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `vggt`");
    }
    if args.chunk_size.is_some() {
        anyhow::bail!("`--chunk-size` is not supported for `vggt`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `vggt`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `vggt`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `vggt`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `vggt`");
    }
    Ok(())
}

/// TripoSG exposes image-to-mesh diffusion only. Reject the Pi3/Pi3X/VGGT
/// conditioning and point-cloud options plus the TripoSR marching-cubes knobs
/// instead of silently ignoring them. `--chunk-size` stays allowed: it feeds
/// the TripoSG SDF query chunk.
fn reject_unsupported_triposg_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.interval.is_some() {
        anyhow::bail!("`--interval` is not supported for `triposg`");
    }
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `triposg`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `triposg`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `triposg`");
    }
    if args.conf_threshold.is_some() {
        anyhow::bail!("`--conf-threshold` is not supported for `triposg`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `triposg`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `triposg`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `triposg`");
    }
    Ok(())
}

/// The `--steps`/`--guidance-scale`/`--seed`/`--num-tokens`/`--dense-octree-depth`/
/// `--hierarchical-octree-depth` flags are TripoSG-only; every other family must
/// reject them instead of silently ignoring them.
fn reject_triposg_only_options(args: &RunArgs, family: Family) -> anyhow::Result<()> {
    if args.steps.is_some() {
        anyhow::bail!("`--steps` is not supported for `{}`", family.model_family());
    }
    if args.guidance_scale.is_some() {
        anyhow::bail!(
            "`--guidance-scale` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.seed.is_some() {
        anyhow::bail!("`--seed` is not supported for `{}`", family.model_family());
    }
    if args.num_tokens.is_some() {
        anyhow::bail!(
            "`--num-tokens` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.dense_octree_depth.is_some() {
        anyhow::bail!(
            "`--dense-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.hierarchical_octree_depth.is_some() {
        anyhow::bail!(
            "`--hierarchical-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.flash_decoder.is_some() {
        anyhow::bail!(
            "`--flash-decoder` is not supported for `{}`",
            family.model_family()
        );
    }
    if args.flash_octree_depth.is_some() {
        anyhow::bail!(
            "`--flash-octree-depth` is not supported for `{}`",
            family.model_family()
        );
    }
    Ok(())
}

/// Resolve the RMBG package: explicit `--rmbg-model-path` wins, otherwise a
/// sibling `rmbg-1.4` canonical directory next to an explicit TripoSG
/// `--model-path` is used when present. (The canonical directories are
/// distributed separately because the BRIA license differs from TripoSG's
/// MIT license.)
fn resolve_rmbg_model_path(args: &RunArgs) -> Option<PathBuf> {
    if args.rmbg_model_path.is_some() {
        return args.rmbg_model_path.clone();
    }
    args.model_path
        .as_ref()
        .and_then(|path| path.parent())
        .map(|parent| parent.join("rmbg-1.4"))
        .filter(|candidate| candidate.is_dir())
}

/// TRELLIS.2 exposes image-to-mesh diffusion + texture only. Reject the
/// Pi3/Pi3X/VGGT conditioning and point-cloud options, the TripoSR
/// marching-cubes knobs, the TripoSG octree/flash/latent knobs it does not
/// share, and the RMBG package (a documented TRELLIS.2 slice exclusion:
/// RGB input is treated as fully opaque) instead of silently ignoring them.
/// `--steps`/`--seed`/`--pipeline-type`/`--max-num-tokens`/`--texture` stay
/// allowed; `--guidance-scale` has no runtime override (sampler guidance comes
/// from `pipeline.json`/defaults).
fn reject_unsupported_trellis2_options(args: &RunArgs) -> anyhow::Result<()> {
    if args.interval.is_some() {
        anyhow::bail!("`--interval` is not supported for `trellis2`");
    }
    if args.conditions.is_some() {
        anyhow::bail!("`--conditions` is not supported for `trellis2`");
    }
    if args.vo {
        anyhow::bail!("`--vo` is not supported for `trellis2`");
    }
    if args.chunk_size.is_some() {
        anyhow::bail!("`--chunk-size` is not supported for `trellis2`");
    }
    if args.overlap.is_some() {
        anyhow::bail!("`--overlap` is not supported for `trellis2`");
    }
    if args.conf_threshold.is_some() {
        anyhow::bail!("`--conf-threshold` is not supported for `trellis2`");
    }
    if !args.inject_condition.is_empty() {
        anyhow::bail!("`--inject-condition` is not supported for `trellis2`");
    }
    if args.mc_resolution.is_some() {
        anyhow::bail!("`--mc-resolution` is not supported for `trellis2`");
    }
    if args.mc_threshold.is_some() {
        anyhow::bail!("`--mc-threshold` is not supported for `trellis2`");
    }
    if args.guidance_scale.is_some() {
        anyhow::bail!("`--guidance-scale` is not supported for `trellis2`");
    }
    if args.num_tokens.is_some() {
        anyhow::bail!("`--num-tokens` is not supported for `trellis2`; use `--max-num-tokens`");
    }
    if args.dense_octree_depth.is_some() {
        anyhow::bail!("`--dense-octree-depth` is not supported for `trellis2`");
    }
    if args.hierarchical_octree_depth.is_some() {
        anyhow::bail!("`--hierarchical-octree-depth` is not supported for `trellis2`");
    }
    if args.flash_decoder.is_some() {
        anyhow::bail!("`--flash-decoder` is not supported for `trellis2`");
    }
    if args.flash_octree_depth.is_some() {
        anyhow::bail!("`--flash-octree-depth` is not supported for `trellis2`");
    }
    if args.rmbg_model_path.is_some() {
        anyhow::bail!(
            "`--rmbg-model-path` is not supported for `trellis2`; RMBG background removal is a \
             documented exclusion and RGB input is treated as fully opaque"
        );
    }
    Ok(())
}

/// Build the TRELLIS.2 run options from the reference defaults plus explicit
/// CLI overrides. `--steps` feeds all three Euler samplers (sparse-structure,
/// shape, texture); `--guidance-scale` has no runtime override.
fn trellis2_run_options_from_args(args: &RunArgs) -> anyhow::Result<Trellis2RunOptions> {
    let defaults = Trellis2RunOptions::default();
    let options = Trellis2RunOptions {
        seed: args.seed.unwrap_or(defaults.seed),
        pipeline_type: args
            .pipeline_type
            .map(Trellis2PipelineType::from)
            .unwrap_or(defaults.pipeline_type),
        ss_steps: args.steps.unwrap_or(defaults.ss_steps),
        shape_steps: args.steps.unwrap_or(defaults.shape_steps),
        tex_steps: args.steps.unwrap_or(defaults.tex_steps),
        with_texture: args.texture.unwrap_or(defaults.with_texture),
        max_num_tokens: args.max_num_tokens.unwrap_or(defaults.max_num_tokens),
        preprocess: defaults.preprocess,
    };
    Ok(options)
}

/// Build the TripoSG run options from the reference defaults plus explicit
/// CLI overrides; `--chunk-size` doubles as the SDF query chunk.
fn triposg_run_options_from_args(args: &RunArgs) -> anyhow::Result<TripoSgRunOptions> {
    let defaults = TripoSgRunOptions::default();
    let options = TripoSgRunOptions {
        steps: args.steps.unwrap_or(defaults.steps),
        guidance_scale: args.guidance_scale.unwrap_or(defaults.guidance_scale),
        seed: args.seed.unwrap_or(defaults.seed),
        num_tokens: args.num_tokens.unwrap_or(defaults.num_tokens),
        dense_octree_depth: args
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: args
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        use_flash_decoder: args.flash_decoder.unwrap_or(defaults.use_flash_decoder),
        flash_octree_depth: args
            .flash_octree_depth
            .unwrap_or(defaults.flash_octree_depth),
        query_chunk_size: args.chunk_size.unwrap_or(defaults.query_chunk_size),
    };
    options.validate().map_err(anyhow::Error::from)?;
    Ok(options)
}

fn run_vggt(
    model_assets: ModelAssetOptions,
    source: &Path,
    interval: Option<usize>,
    conf_threshold: Option<f32>,
    output: &Path,
    dump_npz: Option<&Path>,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = VggtPipeline::load(model_assets)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path(source, interval, device)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    if let Some(npz_path) = dump_npz {
        pipeline
            .export_inference_npz(&inference, npz_path)
            .map_err(anyhow::Error::from)?;
    }
    pipeline.export_point_cloud_with_threshold(
        &inference,
        output,
        conf_threshold.unwrap_or(1.0),
    )?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pi3x(
    model_assets: ModelAssetOptions,
    source: &Path,
    conditions: Option<&Path>,
    interval: Option<usize>,
    vo: bool,
    chunk_size: Option<usize>,
    overlap: Option<usize>,
    conf_threshold: Option<f32>,
    inject_condition: &[InjectCondition],
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    if vo {
        let inject_conditions = Pi3xInjectConditions {
            pose: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Pose)),
            depth: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Depth)),
            ray: inject_condition
                .iter()
                .any(|item| matches!(item, InjectCondition::Ray | InjectCondition::Intrinsic)),
        };
        let t0 = std::time::Instant::now();
        let pipeline = Pi3xVoPipeline::load(model_assets)?;
        if stage_time {
            eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
        }
        let t1 = std::time::Instant::now();
        let inference = pipeline.infer_from_path(
            source,
            interval,
            chunk_size,
            overlap,
            conf_threshold,
            inject_conditions,
            device,
        )?;
        if stage_time {
            eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();
        pipeline.export_point_cloud(&inference, output)?;
        if stage_time {
            eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
        }
    } else {
        let t0 = std::time::Instant::now();
        let pipeline = Pi3xPipeline::load(model_assets)?;
        if stage_time {
            eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
        }
        let t1 = std::time::Instant::now();
        let inference = pipeline.infer_from_path(source, conditions, interval, device)?;
        if stage_time {
            eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
        }
        let t2 = std::time::Instant::now();
        pipeline.export_point_cloud(&inference, output)?;
        if stage_time {
            eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

fn run_triposr(
    model_assets: ModelAssetOptions,
    source: &Path,
    mc_resolution: u32,
    mc_threshold: f32,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    anyhow::ensure!(mc_resolution >= 2, "--mc-resolution must be at least 2");
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = TripoSrPipeline::load(model_assets)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path(source, device)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mesh = pipeline.extract_mesh(&inference.scene_codes, mc_resolution, mc_threshold, 8192)?;
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    let t3 = std::time::Instant::now();
    pipeline.export_mesh(&mesh, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t3.elapsed().as_secs_f64());
    }
    Ok(())
}

fn run_triposg(
    model_assets: ModelAssetOptions,
    rmbg_model_path: Option<PathBuf>,
    source: &Path,
    options: TripoSgRunOptions,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    options.validate().map_err(anyhow::Error::from)?;
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = TripoSgPipeline::load_with_rmbg(model_assets, rmbg_model_path)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let inference = pipeline.infer_from_path(source, device, &options)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mesh = pipeline.extract_mesh(&inference.latents, &options)?;
    if stage_time {
        eprintln!("[stage] mesh: {:.2}s", t2.elapsed().as_secs_f64());
    }
    let t3 = std::time::Instant::now();
    pipeline.export_mesh(&mesh, output)?;
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t3.elapsed().as_secs_f64());
    }
    Ok(())
}

/// Load the TRELLIS.2 pipeline from `--model-path`. A raw TRELLIS.2 checkpoint
/// directory (`pipeline.json` + `ckpts/`, i.e. the upstream
/// `models/TRELLIS.2-4B` layout) uses the per-stage raw loader; anything else
/// (canonical package dir, GGUF, or HF cache auto-download) goes through the
/// canonical weights path.
fn load_trellis2_pipeline(
    model_assets: &ModelAssetOptions,
    device: &candle_core::Device,
) -> anyhow::Result<Trellis2Pipeline> {
    if let Some(dir) = &model_assets.canonical_dir {
        let looks_raw =
            dir.is_dir() && (dir.join("pipeline.json").is_file() || dir.join("ckpts").is_dir());
        if looks_raw {
            return Trellis2Pipeline::load_from_raw_dir(dir, device).map_err(anyhow::Error::from);
        }
    }
    Trellis2Pipeline::load(model_assets.clone()).map_err(anyhow::Error::from)
}

fn run_trellis2(
    model_assets: ModelAssetOptions,
    source: &Path,
    options: Trellis2RunOptions,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let stage_time = std::env::var_os("LUX3D_STAGE_TIME").is_some();
    let t0 = std::time::Instant::now();
    let pipeline = load_trellis2_pipeline(&model_assets, device)?;
    if stage_time {
        eprintln!("[stage] load: {:.2}s", t0.elapsed().as_secs_f64());
    }
    let t1 = std::time::Instant::now();
    let stages = pipeline
        .run_from_path_with_stages(source, device, &options)
        .map_err(anyhow::Error::from)?;
    if stage_time {
        eprintln!("[stage] infer: {:.2}s", t1.elapsed().as_secs_f64());
    }
    let t2 = std::time::Instant::now();
    let mesh = stages.mesh.expect("trellis2 run always produces a mesh");
    if options.with_texture && mesh.voxel_attr_channels == lux3d_core::export::pbrvox::CHANNELS {
        let sidecar = pipeline
            .export_mesh_with_pbr(&mesh, output)
            .map_err(anyhow::Error::from)?;
        eprintln!("[export] pbr sidecar: {}", sidecar.display());
    } else {
        pipeline
            .export_mesh(&mesh, output)
            .map_err(anyhow::Error::from)?;
    }
    if stage_time {
        eprintln!("[stage] export: {:.2}s", t2.elapsed().as_secs_f64());
    }
    Ok(())
}

fn normalizer_script(repo_root: &std::path::Path) -> anyhow::Result<PathBuf> {
    let candidates = [
        repo_root
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        repo_root
            .join("candle-3d")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
        repo_root
            .join("3d-rs")
            .join("tools")
            .join("python_baseline")
            .join("normalize_weights.py"),
    ];
    for candidate in candidates {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    anyhow::bail!(
        "missing normalizer script `tools/python_baseline/normalize_weights.py` under `{}`",
        repo_root.display()
    )
}

fn python_executable(repo_root: &std::path::Path, family: ModelFamily) -> PathBuf {
    if let Some(override_path) = std::env::var_os("LUX3D_PYTHON") {
        return PathBuf::from(override_path);
    }
    let env_root = repo_root
        .join("3d")
        .join("_generated")
        .join("python-envs")
        .join(family.as_str());
    for candidate in [
        env_root.join("Scripts").join("python.exe"),
        env_root.join("bin").join("python"),
    ] {
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::{fs, path::Path};

    use clap::Parser;

    use super::{Cli, Command, RunArgs, inspect_model, normalize_weights, run_model};
    use lux3d_core::{
        ModelFamily,
        test_support::{
            GpuTestLock, canonical_package_dir, resolve_raw_model_dir_for_tests, runtime_root,
        },
    };

    fn repo_root() -> PathBuf {
        runtime_root()
    }

    fn temp_output_dir(family: ModelFamily) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("current time")
            .as_nanos();
        std::env::temp_dir().join(format!("lux3d-cli-{}-{unique}", family.as_str()))
    }

    fn cleanup_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn routes_inspect_pi3_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "inspect",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "pi3",
        ]);
        assert_eq!("pi3", cli.command.family_name());
    }

    #[test]
    fn routes_inspect_vggt_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "inspect",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "vggt",
        ]);
        assert_eq!("vggt", cli.command.family_name());
    }

    #[test]
    fn routes_run_vggt_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "vggt",
            "--source",
            repo_root()
                .join("test-assets")
                .join("pi3-frames")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Vggt)
                .to_string_lossy()
                .as_ref(),
            "--conf-threshold",
            "1.0",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("vggt.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("vggt", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(1.0), args.conf_threshold);
        super::reject_unsupported_vggt_options(&args).expect("vggt supported options");
    }

    #[test]
    fn run_vggt_rejects_pi3x_only_options() {
        let cli = Cli::parse_from([
            "lux3d", "run", "vggt", "--source", "frames", "--vo", "--output", "out.ply",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let error = super::reject_unsupported_vggt_options(&args)
            .expect_err("--vo must be rejected for vggt");
        assert!(error.to_string().contains("--vo"));
    }

    #[test]
    fn routes_run_triposg_family_with_reference_defaults() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.obj",
        ]);
        assert_eq!("triposg", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(ModelFamily::TripoSg, args.family.model_family());
        super::reject_unsupported_triposg_options(&args).expect("default triposg options");
        let options = super::triposg_run_options_from_args(&args).expect("default run options");
        assert_eq!(super::TripoSgRunOptions::default(), options);
        let defaults = super::TripoSgRunOptions::default();
        assert_eq!(
            (
                50usize, 7.0f32, 42u64, 2048usize, 8u32, 9u32, true, 9u32, 16384usize
            ),
            (
                defaults.steps,
                defaults.guidance_scale,
                defaults.seed,
                defaults.num_tokens,
                defaults.dense_octree_depth,
                defaults.hierarchical_octree_depth,
                defaults.use_flash_decoder,
                defaults.flash_octree_depth,
                defaults.query_chunk_size
            )
        );
    }

    #[test]
    fn triposg_run_options_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "triposg",
            "--source",
            "image.png",
            "--output",
            "out.obj",
            "--steps",
            "12",
            "--guidance-scale",
            "3.5",
            "--seed",
            "7",
            "--num-tokens",
            "1024",
            "--dense-octree-depth",
            "5",
            "--hierarchical-octree-depth",
            "6",
            "--flash-decoder",
            "false",
            "--flash-octree-depth",
            "7",
            "--chunk-size",
            "4096",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        let options = super::triposg_run_options_from_args(&args).expect("run options");
        assert_eq!(12, options.steps);
        assert_eq!(3.5, options.guidance_scale);
        assert_eq!(7, options.seed);
        assert_eq!(1024, options.num_tokens);
        assert_eq!(5, options.dense_octree_depth);
        assert_eq!(6, options.hierarchical_octree_depth);
        assert!(!options.use_flash_decoder);
        assert_eq!(7, options.flash_octree_depth);
        assert_eq!(4096, options.query_chunk_size);
    }

    #[test]
    fn routes_run_trellis2_family_with_reference_defaults() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
        ]);
        assert_eq!("trellis2", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(ModelFamily::Trellis2, args.family.model_family());
        super::reject_unsupported_trellis2_options(&args).expect("default trellis2 options");
        let options =
            super::trellis2_run_options_from_args(&args).expect("default trellis2 run options");
        assert_eq!(super::Trellis2RunOptions::default(), options);
        let defaults = super::Trellis2RunOptions::default();
        assert_eq!(
            (
                42u64,
                "1024_cascade",
                12usize,
                12usize,
                12usize,
                true,
                49152usize
            ),
            (
                defaults.seed,
                defaults.pipeline_type.as_str(),
                defaults.ss_steps,
                defaults.shape_steps,
                defaults.tex_steps,
                defaults.with_texture,
                defaults.max_num_tokens
            )
        );
    }

    #[test]
    fn trellis2_run_options_map_cli_overrides() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--output",
            "out.glb",
            "--pipeline-type",
            "512",
            "--steps",
            "4",
            "--seed",
            "7",
            "--max-num-tokens",
            "8192",
            "--texture",
            "false",
            "--model-path",
            "models/TRELLIS.2-4B",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(super::Family::Trellis2), Some(args.family));
        assert_eq!(Some(4), args.steps);
        assert_eq!(Some(7), args.seed);
        assert_eq!(Some(8192), args.max_num_tokens);
        assert_eq!(Some(false), args.texture);
        let options = super::trellis2_run_options_from_args(&args).expect("trellis2 run options");
        assert_eq!(7, options.seed);
        assert_eq!(super::Trellis2PipelineType::R512, options.pipeline_type);
        assert_eq!(4, options.ss_steps);
        assert_eq!(4, options.shape_steps);
        assert_eq!(4, options.tex_steps);
        assert!(!options.with_texture);
        assert_eq!(8192, options.max_num_tokens);
    }

    #[test]
    fn trellis2_pipeline_type_accepts_reference_names() {
        for (name, expected) in [
            ("512", super::Trellis2PipelineType::R512),
            ("1024", super::Trellis2PipelineType::R1024),
            ("1024_cascade", super::Trellis2PipelineType::Cascade1024),
        ] {
            let cli = Cli::parse_from([
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--pipeline-type",
                name,
                "--output",
                "out.glb",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let options = super::trellis2_run_options_from_args(&args).expect("run options");
            assert_eq!(expected, options.pipeline_type, "pipeline type `{name}`");
        }
    }

    #[test]
    fn trellis2_texture_flag_defaults_to_true_without_value() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "trellis2",
            "--source",
            "image.png",
            "--texture",
            "--output",
            "out.glb",
        ]);
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(true), args.texture);
        let options = super::trellis2_run_options_from_args(&args).expect("run options");
        assert!(options.with_texture);
    }

    #[test]
    fn run_trellis2_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 15] = [
            ("--interval", &["--interval", "2"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--vo", &["--vo"]),
            ("--chunk-size", &["--chunk-size", "8"]),
            ("--overlap", &["--overlap", "4"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--mc-resolution", &["--mc-resolution", "256"]),
            ("--mc-threshold", &["--mc-threshold", "25.0"]),
            ("--guidance-scale", &["--guidance-scale", "3.5"]),
            ("--num-tokens", &["--num-tokens", "1024"]),
            ("--dense-octree-depth", &["--dense-octree-depth", "8"]),
            (
                "--hierarchical-octree-depth",
                &["--hierarchical-octree-depth", "9"],
            ),
            ("--flash-decoder", &["--flash-decoder", "false"]),
            ("--rmbg-model-path", &["--rmbg-model-path", "rmbg-1.4"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "trellis2",
                "--source",
                "image.png",
                "--output",
                "out.glb",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_trellis2_options(&args)
                .expect_err(&format!("{flag} must be rejected for trellis2"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `trellis2`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn run_triposg_rejects_foreign_family_options() {
        let flag_options: [(&str, &[&str]); 8] = [
            ("--interval", &["--interval", "2"]),
            ("--conditions", &["--conditions", "conditions.npz"]),
            ("--vo", &["--vo"]),
            ("--overlap", &["--overlap", "4"]),
            ("--conf-threshold", &["--conf-threshold", "0.5"]),
            ("--inject-condition", &["--inject-condition", "pose"]),
            ("--mc-resolution", &["--mc-resolution", "256"]),
            ("--mc-threshold", &["--mc-threshold", "25.0"]),
        ];
        for (flag, extra) in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposg",
                "--source",
                "image.png",
                "--output",
                "out.obj",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_unsupported_triposg_options(&args)
                .expect_err(&format!("{flag} must be rejected for triposg"));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `triposg`"),
                "unexpected message for {flag}: `{message}`"
            );
            assert!(message.contains(flag), "`{flag}` missing in `{message}`");
        }
    }

    #[test]
    fn triposg_only_steps_rejected_for_other_families() {
        for family in ["pi3", "pi3x", "triposr", "vggt"] {
            let cli = Cli::parse_from([
                "lux3d", "run", family, "--source", "input", "--steps", "10", "--output", "out",
            ]);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("--steps must be rejected for {family}"));
            let message = error.to_string();
            assert!(
                message.contains("--steps"),
                "`--steps` missing in `{message}`"
            );
            assert!(
                message.contains(&format!("is not supported for `{family}`")),
                "unexpected message: `{message}`"
            );
        }
    }

    #[test]
    fn all_triposg_only_flags_rejected_for_triposr() {
        let flag_options: [&[&str]; 6] = [
            &["--steps", "10"],
            &["--guidance-scale", "7.0"],
            &["--seed", "1"],
            &["--num-tokens", "1024"],
            &["--dense-octree-depth", "5"],
            &["--hierarchical-octree-depth", "6"],
        ];
        for extra in flag_options {
            let mut argv = vec![
                "lux3d",
                "run",
                "triposr",
                "--source",
                "horse.png",
                "--output",
                "out.obj",
            ];
            argv.extend_from_slice(extra);
            let cli = Cli::parse_from(argv);
            let Command::Run(args) = cli.command else {
                panic!("expected run command");
            };
            let error = super::reject_triposg_only_options(&args, args.family)
                .expect_err(&format!("{} must be rejected for triposr", extra[0]));
            let message = error.to_string();
            assert!(
                message.contains("is not supported for `triposr`"),
                "unexpected message: `{message}`"
            );
            assert!(
                message.contains(extra[0]),
                "`{}` missing in `{message}`",
                extra[0]
            );
        }
    }

    #[test]
    fn routes_weights_normalize_triposr_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "weights",
            "normalize",
            "--repo-root",
            repo_root().to_string_lossy().as_ref(),
            "--raw-model-dir",
            r"C:\raw-models\triposr",
            "--output-dir",
            canonical_package_dir(ModelFamily::TripoSr)
                .to_string_lossy()
                .as_ref(),
            "triposr",
        ]);
        assert_eq!("triposr", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3)
                .to_string_lossy()
                .as_ref(),
            "--interval",
            "10",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(10), args.interval);
        assert_eq!(
            Some(canonical_package_dir(ModelFamily::Pi3)),
            args.model_path
        );
    }

    #[test]
    fn routes_run_pi3x_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb")
                .to_string_lossy()
                .as_ref(),
            "--conditions",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("condition.npz")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3x", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3x_vo_inject_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--vo",
            "--chunk-size",
            "8",
            "--overlap",
            "4",
            "--conf-threshold",
            "0.05",
            "--inject-condition",
            "pose,depth,ray",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x-vo.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert_eq!("pi3x", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert!(args.vo);
        assert_eq!(Some(8), args.chunk_size);
        assert_eq!(Some(4), args.overlap);
        assert_eq!(Some(0.05), args.conf_threshold);
        assert_eq!(
            vec![
                super::InjectCondition::Pose,
                super::InjectCondition::Depth,
                super::InjectCondition::Ray
            ],
            args.inject_condition
        );
    }

    #[test]
    fn routes_run_pi3x_vo_inject_rejects_unknown_token() {
        let result = Cli::try_parse_from([
            "lux3d",
            "run",
            "pi3x",
            "--source",
            repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4")
                .to_string_lossy()
                .as_ref(),
            "--model-path",
            canonical_package_dir(ModelFamily::Pi3x)
                .to_string_lossy()
                .as_ref(),
            "--vo",
            "--inject-condition",
            "pose,unknown",
            "--output",
            repo_root()
                .join("3d")
                .join("_generated")
                .join("pi3x-vo.ply")
                .to_string_lossy()
                .as_ref(),
        ]);
        assert!(
            result.is_err(),
            "unknown inject-condition token must be rejected"
        );
    }

    #[test]
    fn renders_pi3_model_spec_as_json() {
        let json = inspect_model(repo_root(), ModelFamily::Pi3).expect("pi3 inspection");
        assert!(json.contains("\"family\": \"pi3\""));
        assert!(json.contains("\"canonical_filename\": \"model.safetensors\""));
        for forbidden in [
            "3d/models",
            "3d/canonical-weights",
            "yyfz233-Pi3",
            "yyfz233-Pi3X",
            "stabilityai-TripoSR",
        ] {
            assert!(
                !json.contains(forbidden),
                "inspect output must stay path-free, found `{forbidden}`"
            );
        }
    }

    #[test]
    fn resolves_normalizer_script_in_this_checkout() {
        let script = super::normalizer_script(&repo_root()).expect("normalizer script");
        assert!(script.is_file(), "resolved script must exist");
        assert!(
            script.ends_with("tools/python_baseline/normalize_weights.py"),
            "unexpected script path: {}",
            script.display()
        );
    }

    #[test]
    fn normalize_weights_runs_python_canonicalizer_for_pi3() {
        let raw_model_dir =
            resolve_raw_model_dir_for_tests(ModelFamily::Pi3).expect("resolve pi3 raw model dir");
        let output_dir = temp_output_dir(ModelFamily::Pi3);
        let result = normalize_weights(
            repo_root(),
            ModelFamily::Pi3,
            raw_model_dir,
            output_dir.clone(),
        )
        .expect("pi3 canonical weights");
        cleanup_dir(&output_dir);
        assert!(result.contains("\"family\": \"pi3\""));
        assert!(result.contains("\"tensor_count\": 1210"));
        assert!(result.contains("\"raw_files\": [\n    \"model.safetensors\"\n  ]"));
        assert!(result.contains("\"canonical_file\": \"model.safetensors\""));
        assert!(!result.contains("3d/models"));
    }

    #[test]
    fn run_pi3_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output = std::env::temp_dir().join(format!("lux3d-cli-pi3-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3,
            device: super::DeviceBackend::Cuda,
            source: repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("house"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
        };

        let emitted = run_model(args).expect("pi3 cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3 output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }

    #[test]
    fn run_triposr_smoke_emits_obj_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-triposr-{}.obj", std::process::id()));
        let args = RunArgs {
            family: super::Family::Triposr,
            device: super::DeviceBackend::Cuda,
            source: repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(256),
            mc_threshold: Some(25.0),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
        };

        let emitted = run_model(args).expect("triposr cli run");
        let body = fs::read_to_string(&emitted).expect("read triposr output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("# https://github.com/mikedh/trimesh\n"));
        assert!(body.lines().any(|line| line.starts_with("v ")));
        assert!(body.lines().any(|line| line.starts_with("f ")));
    }

    #[test]
    fn run_triposr_rejects_invalid_marching_cubes_resolution() {
        let output = std::env::temp_dir().join(format!(
            "lux3d-cli-triposr-invalid-{}.obj",
            std::process::id()
        ));
        let args = RunArgs {
            family: super::Family::Triposr,
            device: super::DeviceBackend::Cuda,
            source: repo_root()
                .join("tp")
                .join("3d")
                .join("TripoSR")
                .join("examples")
                .join("horse.png"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(1),
            mc_threshold: Some(25.0),
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            rmbg_model_path: None,
            cache_dir: None,
            output,
            dump_npz: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
        };

        let err = run_model(args).expect_err("invalid marching-cubes resolution must fail");
        assert!(
            err.to_string()
                .contains("--mc-resolution must be at least 2")
        );
    }

    #[test]
    fn run_pi3x_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-pi3x-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3x,
            device: super::DeviceBackend::Cuda,
            source: repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("room")
                .join("rgb"),
            interval: Some(2),
            conditions: Some(
                repo_root()
                    .join("tp")
                    .join("3d")
                    .join("Pi3")
                    .join("examples")
                    .join("room")
                    .join("condition.npz"),
            ),
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
        };

        let emitted = run_model(args).expect("pi3x cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }

    #[test]
    fn run_pi3x_vo_inject_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output =
            std::env::temp_dir().join(format!("lux3d-cli-pi3x-vo-{}.ply", std::process::id()));
        let args = RunArgs {
            family: super::Family::Pi3x,
            device: super::DeviceBackend::Cuda,
            source: repo_root()
                .join("tp")
                .join("3d")
                .join("Pi3")
                .join("examples")
                .join("skating.mp4"),
            interval: None,
            conditions: None,
            vo: true,
            chunk_size: Some(8),
            overlap: Some(4),
            conf_threshold: Some(0.05),
            inject_condition: vec![
                super::InjectCondition::Pose,
                super::InjectCondition::Depth,
                super::InjectCondition::Ray,
            ],
            mc_resolution: None,
            mc_threshold: None,
            steps: None,
            guidance_scale: None,
            seed: None,
            num_tokens: None,
            dense_octree_depth: None,
            hierarchical_octree_depth: None,
            flash_decoder: None,
            flash_octree_depth: None,
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            rmbg_model_path: None,
            cache_dir: None,
            output: output.clone(),
            dump_npz: None,
            pipeline_type: None,
            max_num_tokens: None,
            texture: None,
        };

        let emitted = run_model(args).expect("pi3x vo cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x vo output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }
}
