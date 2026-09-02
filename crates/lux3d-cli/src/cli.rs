use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use lux3d_core::runtime::{
    Pi3Pipeline, Pi3xInjectConditions, Pi3xPipeline, Pi3xVoPipeline, TripoSrPipeline,
};
use lux3d_core::{ModelAssetOptions, ModelFamily, WeightLocator, ensure_canonical_weights};

#[derive(Debug, Parser, PartialEq)]
#[command(
    name = "lux3d",
    about = "CUDA-first runtime and weight tools for Pi3, Pi3X, and TripoSR."
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
        help = "Path to a canonical package directory laid out like 3d/canonical-weights/<family>."
    )]
    pub model_path: Option<PathBuf>,

    #[arg(
        long,
        help = "Override the Hugging Face cache root used when --model-path is not supplied."
    )]
    pub cache_dir: Option<PathBuf>,

    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub enum Family {
    Pi3,
    Pi3x,
    Triposr,
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

impl Family {
    pub const fn model_family(self) -> ModelFamily {
        match self {
            Self::Pi3 => ModelFamily::Pi3,
            Self::Pi3x => ModelFamily::Pi3x,
            Self::Triposr => ModelFamily::TripoSr,
        }
    }
}

impl Command {
    pub const fn family_name(&self) -> &'static str {
        match self {
            Self::Inspect(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
            },
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => match family.family {
                    Family::Pi3 => "pi3",
                    Family::Pi3x => "pi3x",
                    Family::Triposr => "triposr",
                },
            },
            Self::Run(args) => match args.family {
                Family::Pi3 => "pi3",
                Family::Pi3x => "pi3x",
                Family::Triposr => "triposr",
            },
        }
    }

    pub const fn family(&self) -> ModelFamily {
        match self {
            Self::Inspect(args) => args.family.model_family(),
            Self::Weights(args) => match args.command {
                WeightsCommand::Normalize(ref family) => family.family.model_family(),
            },
            Self::Run(args) => args.family.model_family(),
        }
    }

    pub fn repo_root(&self) -> Option<&PathBuf> {
        match self {
            Self::Inspect(args) => Some(&args.repo_root),
            Self::Weights(args) => match &args.command {
                WeightsCommand::Normalize(family) => Some(&family.repo_root),
            },
            Self::Run(_) => None,
        }
    }
}

pub fn inspect_model(repo_root: PathBuf, family: ModelFamily) -> anyhow::Result<String> {
    let spec = lux3d_core::ModelSpec::inspect(repo_root, family)?;
    Ok(serde_json::to_string_pretty(&spec)?)
}

pub fn normalize_weights(
    repo_root: PathBuf,
    family: ModelFamily,
    raw_model_dir: PathBuf,
    output_dir: PathBuf,
) -> anyhow::Result<String> {
    let python = python_executable(&repo_root, family);
    let script = repo_root
        .join("3d-rs")
        .join("tools")
        .join("python_baseline")
        .join("normalize_weights.py");

    if !python.is_file() {
        anyhow::bail!("missing python executable at `{}`", python.display());
    }
    if !script.is_file() {
        anyhow::bail!("missing normalizer script at `{}`", script.display());
    }

    let family_name = family.as_str();
    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg("--family")
        .arg(family_name)
        .arg("--raw-model-dir")
        .arg(&raw_model_dir)
        .arg("--output-dir")
        .arg(&output_dir)
        .output()?;

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
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    match args.family.model_family() {
        ModelFamily::Pi3 => run_pi3(
            model_assets,
            &args.source,
            args.interval,
            &args.output,
            &device,
        )?,
        ModelFamily::Pi3x => run_pi3x(
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
        )?,
        ModelFamily::TripoSr => run_triposr(
            model_assets,
            &args.source,
            args.mc_resolution.unwrap_or(256),
            args.mc_threshold.unwrap_or(25.0),
            &args.output,
            &device,
        )?,
    }

    // Backend CPU-fallback observability: nonzero values = hidden host/CPU compute.
    match args.device {
        DeviceBackend::Wgpu => eprintln!(
            "[candle-obs] wgpu cpu_fallback={} host_compute={}",
            candle_core::wgpu_cpu_fallback_count(),
            candle_core::wgpu_host_compute_count(),
        ),
        DeviceBackend::Vulkan => eprintln!(
            "[candle-obs] vulkan cpu_fallback={} host_compute={}",
            candle_core::vulkan_cpu_fallback_count(),
            candle_core::vulkan_host_compute_count(),
        ),
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
    let pipeline = Pi3Pipeline::load(model_assets)?;
    let inference = pipeline.infer_from_path_with_interval(source, interval, device)?;
    pipeline.export_ply(&inference, output)?;
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
        let pipeline = Pi3xVoPipeline::load(model_assets)?;
        let inference = pipeline.infer_from_path(
            source,
            interval,
            chunk_size,
            overlap,
            conf_threshold,
            inject_conditions,
            device,
        )?;
        pipeline.export_ply(&inference, output)?;
    } else {
        let pipeline = Pi3xPipeline::load(model_assets)?;
        let inference = pipeline.infer_from_path(source, conditions, interval, device)?;
        pipeline.export_ply(&inference, output)?;
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
    let pipeline = TripoSrPipeline::load(model_assets)?;
    let inference = pipeline.infer_from_path(source, device)?;
    let mesh = pipeline.extract_mesh(&inference.scene_codes, mc_resolution, mc_threshold, 8192)?;
    pipeline.export_obj(&mesh, output)?;
    Ok(())
}

fn python_executable(repo_root: &std::path::Path, family: ModelFamily) -> PathBuf {
    repo_root
        .join("3d")
        .join("_generated")
        .join("python-envs")
        .join(family.as_str())
        .join("Scripts")
        .join("python.exe")
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
            model_path: Some(canonical_package_dir(ModelFamily::Pi3)),
            cache_dir: None,
            output: output.clone(),
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
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            cache_dir: None,
            output: output.clone(),
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
            model_path: Some(canonical_package_dir(ModelFamily::TripoSr)),
            cache_dir: None,
            output,
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
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            cache_dir: None,
            output: output.clone(),
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
            model_path: Some(canonical_package_dir(ModelFamily::Pi3x)),
            cache_dir: None,
            output: output.clone(),
        };

        let emitted = run_model(args).expect("pi3x vo cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x vo output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }
}
