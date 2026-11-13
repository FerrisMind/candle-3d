//! Cross-backend numerical-parity dump tool.
//!
//! Runs one model on an explicitly selected candle backend and writes the
//! numerical outputs to an `.npz` file. The companion script
//! `scripts/compare_backend_parity.py` compares dumps produced on CUDA,
//! Vulkan and WGPU.
//!
//! ```text
//! cargo run --release --example backend_parity_dump --features cuda,vulkan,wgpu -- \
//!   triposg --device cuda --model-path <pkg> --source <image> --out triposg-cuda.npz
//!
//! cargo run --release --example backend_parity_dump --features cuda,vulkan,wgpu -- \
//!   vggt --device vulkan --model-path <pkg> --source <frames> --out vggt-vulkan.npz
//!
//! cargo run --release --example backend_parity_dump --features cuda,vulkan,wgpu -- \
//!   rmbg --device wgpu --model-path <pkg> --input-safetensors <artifacts.safetensors> \
//!   --input-key rmbg.model_input --out rmbg-wgpu.npz
//! ```
//!
//! `--dtype f16` selects F16 weights/activations for TripoSG, VGGT and RMBG.
//! Public outputs are converted back to F32 before the dump; `--dtype f32` is
//! the strict-F32 reference mode. RMBG true F16 is cross-backend-consistent
//! (~1e-3) but differs from F32 by ~1e-1 rel_l2 on the mask, so use it for
//! backend-parity checks rather than as an F32-equivalent result.
//!
//! `--reduced-precision` mirrors the CLI/server default: Vulkan cooperative
//! matmul runs with TF32-class inputs and WGPU cooperative matmul runs with
//! f16 tensor-core MMA. Without it, `DType::F32` keeps exact fp32
//! accumulation. The companion script calls this mode "FP16-class" because
//! the WGPU path is genuine f16 MMA; it is not storage-only F16 weights.

use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use lux3d_core::{
    ModelAssetOptions, RmbgConfig, RmbgModel, load_auxiliary_canonical_weights,
    runtime::{TripoSgPipeline, TripoSgRunOptions, VggtPipeline},
};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        bail!("usage: backend_parity_dump <triposg|vggt|rmbg> [options]");
    };
    let options = Options::parse(&args[1..])?;
    // Mirrors `lux3d-cli`/`lux3d-server` defaults: WGPU f16-MMA cooperative
    // matmul and Vulkan TF32-class cooperative matmul. Those reduced-precision
    // paths are what this harness labels "FP16-class"; without the flag the
    // comparisons exercise exact FP32 accumulation.
    if options.flag("reduced-precision") {
        lux3d_core::enable_reduced_precision_gemm_opt_ins();
    }
    match command.as_str() {
        "triposg" => run_triposg(&options),
        "vggt" => run_vggt(&options),
        "rmbg" => run_rmbg(&options),
        other => bail!("unknown model `{other}`; expected triposg, vggt or rmbg"),
    }
}

struct Options {
    values: HashMap<String, String>,
    flags: Vec<String>,
}

impl Options {
    fn parse(args: &[String]) -> Result<Self> {
        let mut values = HashMap::new();
        let mut flags = Vec::new();
        let mut index = 0;
        while index < args.len() {
            let key = &args[index];
            if !key.starts_with("--") {
                bail!("unexpected positional argument `{key}`");
            }
            let name = key.trim_start_matches("--").to_string();
            let next_is_value = args
                .get(index + 1)
                .map(|value| !value.starts_with("--"))
                .unwrap_or(false);
            if next_is_value {
                values.insert(name, args[index + 1].clone());
                index += 2;
            } else {
                flags.push(name);
                index += 1;
            }
        }
        Ok(Self { values, flags })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    fn required(&self, name: &str) -> Result<&str> {
        self.get(name)
            .with_context(|| format!("missing required `--{name}`"))
    }

    fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }

    fn parse_or<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T>
    where
        T::Err: std::fmt::Display,
    {
        match self.get(name) {
            Some(value) => value
                .parse()
                .map_err(|error| anyhow::anyhow!("invalid `--{name}`: {error}")),
            None => Ok(default),
        }
    }

    fn backend(&self) -> Result<Device> {
        let backend = self.get("device").unwrap_or("cuda");
        make_device(backend, self.parse_or("device-id", 0usize)?)
    }

    fn dtype(&self) -> Result<DType> {
        match self.get("dtype").unwrap_or("f32") {
            "f32" => Ok(DType::F32),
            "f16" => Ok(DType::F16),
            "bf16" => Ok(DType::BF16),
            other => bail!("unsupported `--dtype {other}`; expected f32, f16 or bf16"),
        }
    }

    fn output(&self) -> Result<PathBuf> {
        Ok(PathBuf::from(self.required("out")?))
    }
}

fn make_device(backend: &str, index: usize) -> Result<Device> {
    match backend {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Device::new_cuda(index).context("create CUDA device")?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = index;
                bail!("binary was built without the `cuda` feature")
            }
        }
        "vulkan" => {
            #[cfg(feature = "vulkan")]
            {
                Ok(Device::new_vulkan(index).context("create Vulkan device")?)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = index;
                bail!("binary was built without the `vulkan` feature")
            }
        }
        "wgpu" => {
            #[cfg(feature = "wgpu")]
            {
                Ok(Device::new_wgpu(index).context("create WGPU device")?)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let _ = index;
                bail!("binary was built without the `wgpu` feature")
            }
        }
        other => bail!("unknown device `{other}`; expected cpu, cuda, vulkan or wgpu"),
    }
}

fn model_assets(path: &str) -> ModelAssetOptions {
    ModelAssetOptions {
        canonical_dir: Some(PathBuf::from(path)),
        cache_dir: None,
    }
}

fn run_triposg(options: &Options) -> Result<()> {
    let device = options.backend()?;
    let pipeline = TripoSgPipeline::load_with_rmbg_and_dtype(
        model_assets(options.required("model-path")?),
        options.get("rmbg-model-path").map(PathBuf::from),
        options.dtype()?,
    )?;
    let mut run = TripoSgRunOptions::default();
    run.steps = options.parse_or("steps", run.steps)?;
    run.guidance_scale = options.parse_or("guidance", run.guidance_scale)?;
    run.seed = options.parse_or("seed", run.seed)?;
    run.num_tokens = options.parse_or("tokens", run.num_tokens)?;
    run.dense_octree_depth = options.parse_or("dense-depth", run.dense_octree_depth)?;
    run.hierarchical_octree_depth =
        options.parse_or("hier-depth", run.hierarchical_octree_depth)?;
    run.flash_octree_depth = options.parse_or("flash-depth", run.flash_octree_depth)?;
    run.query_chunk_size = options.parse_or("chunk", run.query_chunk_size)?;
    if options.flag("no-flash") {
        run.use_flash_decoder = false;
    }
    run.validate()?;

    let inference = pipeline
        .infer_from_path(Path::new(options.required("source")?), &device, &run)
        .context("TripoSG inference")?;
    // `init` isolates the VAE/flash decoder on bit-identical latents; `final`
    // measures the complete pipeline including the chaotic CFG ODE.
    let mesh_latents = match options.get("mesh-latents").unwrap_or("final") {
        "init" => &inference.latents_init,
        "final" => &inference.latents,
        other => bail!("`--mesh-latents` must be `init` or `final`, got `{other}`"),
    };
    let mesh = pipeline
        .extract_mesh(mesh_latents, &run)
        .context("TripoSG mesh extraction")?;

    let mut npz = Npz::create(&options.output()?)?;
    npz.add("image_embeds", &inference.image_embeds)?;
    npz.add("latents_init", &inference.latents_init)?;
    npz.add("latents", &inference.latents)?;
    npz.add("mesh_vertices", &mesh.vertices)?;
    npz.add("mesh_faces", &mesh.faces)?;
    npz.finish()?;

    if let Some(obj) = options.get("obj") {
        pipeline.export_obj(&mesh, Path::new(obj))?;
    }
    Ok(())
}

fn run_vggt(options: &Options) -> Result<()> {
    let device = options.backend()?;
    let dtype = options.dtype()?;
    let pipeline =
        VggtPipeline::load_with_dtype(model_assets(options.required("model-path")?), dtype)?;
    let interval = options
        .get("interval")
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| anyhow::anyhow!("invalid `--interval`: {error}"))?;
    let inference = pipeline
        .infer_from_path(Path::new(options.required("source")?), interval, &device)
        .context("VGGT inference")?;
    pipeline.export_inference_npz(&inference, &options.output()?)?;
    if let Some(ply) = options.get("ply") {
        let threshold = options.parse_or("conf-threshold", 1.0f32)?;
        pipeline.export_ply_with_threshold(&inference, Path::new(ply), threshold)?;
    }
    Ok(())
}

fn run_rmbg(options: &Options) -> Result<()> {
    let device = options.backend()?;
    let dtype = options.dtype()?;
    let package =
        load_auxiliary_canonical_weights(PathBuf::from(options.required("model-path")?), "rmbg")?;
    let config = RmbgConfig::from_resolved_config(&package.read_resolved_config_json()?)?;
    let vb = unsafe { package.var_builder(dtype, &device) }?;
    let model = RmbgModel::load(vb, &config)?;

    let input_path = PathBuf::from(options.required("input-safetensors")?);
    let key = options.get("input-key").unwrap_or("rmbg.model_input");
    let input = unsafe { candle_core::safetensors::MmapedSafetensors::new(&input_path) }
        .with_context(|| format!("open {}", input_path.display()))?
        .load(key, &device)
        .with_context(|| format!("load `{key}` from {}", input_path.display()))?
        .to_dtype(dtype)?;

    let sides = model.forward_sides(&input)?;
    let mut npz = Npz::create(&options.output()?)?;
    for (index, side) in sides.iter().enumerate() {
        npz.add(&format!("side{}", index + 1), side)?;
    }
    // `forward` returns the first reference side (side1 in the Python code).
    npz.add("mask", &sides[0])?;
    npz.finish()
}

struct Npz {
    writer: ndarray_npy::NpzWriter<File>,
}

impl Npz {
    fn create(path: &Path) -> Result<Self> {
        let file =
            File::create(path).with_context(|| format!("create npz `{}`", path.display()))?;
        Ok(Self {
            writer: ndarray_npy::NpzWriter::new(file),
        })
    }

    fn add(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        use ndarray::{ArrayD, IxDyn};
        let tensor = tensor.to_device(&Device::Cpu)?.contiguous()?;
        let shape = tensor.dims().to_vec();
        match tensor.dtype() {
            DType::F32 => {
                let data = tensor.flatten_all()?.to_vec1::<f32>()?;
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
                self.writer.add_array(name, &array)?;
            }
            DType::I64 => {
                let data = tensor.flatten_all()?.to_vec1::<i64>()?;
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
                self.writer.add_array(name, &array)?;
            }
            DType::U32 => {
                let data = tensor.flatten_all()?.to_vec1::<u32>()?;
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
                self.writer.add_array(name, &array)?;
            }
            _ => {
                let data = tensor
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let array = ArrayD::from_shape_vec(IxDyn(&shape), data)?;
                self.writer.add_array(name, &array)?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        self.writer.finish()?;
        Ok(())
    }
}
