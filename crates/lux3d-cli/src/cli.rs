use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use lux3d_core::runtime::{
    Pi3Pipeline, Pi3xInjectConditions, Pi3xPipeline, Pi3xVoPipeline, TripoSrPipeline,
};
use lux3d_core::{ModelFamily, WeightLocator, ensure_canonical_weights};

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
    Normalize(FamilyArgs),
}

#[derive(Debug, Clone, Args, PartialEq, Eq)]
pub struct FamilyArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(value_enum)]
    pub family: Family,
}

#[derive(Debug, Clone, Args, PartialEq)]
pub struct RunArgs {
    #[arg(long)]
    pub repo_root: PathBuf,

    #[arg(value_enum)]
    pub family: Family,

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

    pub fn repo_root(&self) -> &PathBuf {
        match self {
            Self::Inspect(args) => &args.repo_root,
            Self::Weights(args) => match &args.command {
                WeightsCommand::Normalize(family) => &family.repo_root,
            },
            Self::Run(args) => &args.repo_root,
        }
    }
}

pub fn inspect_model(repo_root: PathBuf, family: ModelFamily) -> anyhow::Result<String> {
    let spec = lux3d_core::ModelSpec::inspect(repo_root, family)?;
    Ok(serde_json::to_string_pretty(&spec)?)
}

pub fn normalize_weights(repo_root: PathBuf, family: ModelFamily) -> anyhow::Result<String> {
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
        .arg("--repo-root")
        .arg(&repo_root)
        .output()?;

    if !output.status.success() {
        anyhow::bail!(
            "python normalizer failed for `{family_name}`:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let locator = WeightLocator::new(repo_root.clone());
    let plan = locator.locate(family)?;
    let canonical = ensure_canonical_weights(&plan)?;
    Ok(serde_json::to_string_pretty(&canonical.manifest)?)
}

pub fn run_model(args: RunArgs) -> anyhow::Result<PathBuf> {
    let device = candle_core::Device::new_cuda(0)?;
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }

    match args.family.model_family() {
        ModelFamily::Pi3 => run_pi3(
            args.repo_root,
            &args.source,
            args.interval,
            &args.output,
            &device,
        )?,
        ModelFamily::Pi3x => run_pi3x(
            args.repo_root,
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
            args.repo_root,
            &args.source,
            args.mc_resolution.unwrap_or(256),
            args.mc_threshold.unwrap_or(25.0),
            &args.output,
            &device,
        )?,
    }

    Ok(args.output)
}

fn run_pi3(
    repo_root: PathBuf,
    source: &Path,
    interval: Option<usize>,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    let pipeline = Pi3Pipeline::load(repo_root)?;
    let inference = pipeline.infer_from_path_with_interval(source, interval, device)?;
    pipeline.export_ply(&inference, output)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_pi3x(
    repo_root: PathBuf,
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
        let pipeline = Pi3xVoPipeline::load(repo_root)?;
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
        let pipeline = Pi3xPipeline::load(repo_root)?;
        let inference = pipeline.infer_from_path(source, conditions, interval, device)?;
        pipeline.export_ply(&inference, output)?;
    }
    Ok(())
}

fn run_triposr(
    repo_root: PathBuf,
    source: &Path,
    mc_resolution: u32,
    mc_threshold: f32,
    output: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<()> {
    anyhow::ensure!(mc_resolution >= 2, "--mc-resolution must be at least 2");
    let pipeline = TripoSrPipeline::load(repo_root)?;
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
    use std::fs;
    use std::path::PathBuf;

    use clap::Parser;

    use super::{Cli, Command, RunArgs, inspect_model, normalize_weights, run_model};
    use lux3d_core::{ModelFamily, test_support::GpuTestLock};

    #[test]
    fn routes_inspect_pi3_family() {
        let cli = Cli::parse_from(["lux3d", "inspect", "--repo-root", r"H:\GitHub\LuxRT", "pi3"]);
        assert_eq!("pi3", cli.command.family_name());
    }

    #[test]
    fn routes_weights_normalize_triposr_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "weights",
            "normalize",
            "--repo-root",
            r"H:\GitHub\LuxRT",
            "triposr",
        ]);
        assert_eq!("triposr", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "--repo-root",
            r"H:\GitHub\LuxRT",
            "pi3",
            "--source",
            r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\house",
            "--interval",
            "10",
            "--output",
            r"H:\GitHub\LuxRT\3d\_generated\pi3.ply",
        ]);
        assert_eq!("pi3", cli.command.family_name());
        let Command::Run(args) = cli.command else {
            panic!("expected run command");
        };
        assert_eq!(Some(10), args.interval);
    }

    #[test]
    fn routes_run_pi3x_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "--repo-root",
            r"H:\GitHub\LuxRT",
            "pi3x",
            "--source",
            r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\rgb",
            "--conditions",
            r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\condition.npz",
            "--output",
            r"H:\GitHub\LuxRT\3d\_generated\pi3x.ply",
        ]);
        assert_eq!("pi3x", cli.command.family_name());
    }

    #[test]
    fn routes_run_pi3x_vo_inject_family() {
        let cli = Cli::parse_from([
            "lux3d",
            "run",
            "--repo-root",
            r"H:\GitHub\LuxRT",
            "pi3x",
            "--source",
            r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4",
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
            r"H:\GitHub\LuxRT\3d\_generated\pi3x-vo.ply",
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
            "--repo-root",
            r"H:\GitHub\LuxRT",
            "pi3x",
            "--source",
            r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4",
            "--vo",
            "--inject-condition",
            "pose,unknown",
            "--output",
            r"H:\GitHub\LuxRT\3d\_generated\pi3x-vo.ply",
        ]);
        assert!(
            result.is_err(),
            "unknown inject-condition token must be rejected"
        );
    }

    #[test]
    fn renders_pi3_model_spec_as_json() {
        let json = inspect_model(PathBuf::from(r"H:\GitHub\LuxRT"), ModelFamily::Pi3)
            .expect("pi3 inspection");
        assert!(json.contains("\"family\": \"pi3\""));
        assert!(json.contains("\"canonical_filename\": \"model.safetensors\""));
    }

    #[test]
    fn normalize_weights_runs_python_canonicalizer_for_pi3() {
        let result = normalize_weights(PathBuf::from(r"H:\GitHub\LuxRT"), ModelFamily::Pi3)
            .expect("pi3 canonical weights");
        assert!(result.contains("\"family\": \"pi3\""));
        assert!(result.contains("\"tensor_count\": 1210"));
    }

    #[test]
    fn run_pi3_smoke_emits_ply_output() {
        let _guard = GpuTestLock::acquire().expect("gpu test lock");
        let output = std::env::temp_dir().join(format!("lux3d-cli-pi3-{}.ply", std::process::id()));
        let args = RunArgs {
            repo_root: PathBuf::from(r"H:\GitHub\LuxRT"),
            family: super::Family::Pi3,
            source: PathBuf::from(r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\house"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
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
            repo_root: PathBuf::from(r"H:\GitHub\LuxRT"),
            family: super::Family::Triposr,
            source: PathBuf::from(r"H:\GitHub\LuxRT\tp\3d\TripoSR\examples\horse.png"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(256),
            mc_threshold: Some(25.0),
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
            repo_root: PathBuf::from(r"H:\GitHub\LuxRT"),
            family: super::Family::Triposr,
            source: PathBuf::from(r"H:\GitHub\LuxRT\tp\3d\TripoSR\examples\horse.png"),
            interval: None,
            conditions: None,
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: Some(1),
            mc_threshold: Some(25.0),
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
            repo_root: PathBuf::from(r"H:\GitHub\LuxRT"),
            family: super::Family::Pi3x,
            source: PathBuf::from(r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\rgb"),
            interval: Some(2),
            conditions: Some(PathBuf::from(
                r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\room\condition.npz",
            )),
            vo: false,
            chunk_size: None,
            overlap: None,
            conf_threshold: None,
            inject_condition: Vec::new(),
            mc_resolution: None,
            mc_threshold: None,
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
            repo_root: PathBuf::from(r"H:\GitHub\LuxRT"),
            family: super::Family::Pi3x,
            source: PathBuf::from(r"H:\GitHub\LuxRT\tp\3d\Pi3\examples\skating.mp4"),
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
            output: output.clone(),
        };

        let emitted = run_model(args).expect("pi3x vo cli run");
        let body = fs::read_to_string(&emitted).expect("read pi3x vo output");
        let _ = fs::remove_file(&emitted);

        assert!(body.starts_with("ply\nformat ascii 1.0\n"));
        assert!(body.contains("property float nx"));
    }
}
