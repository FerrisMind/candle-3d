use super::BenchDevice;
use criterion::{criterion_group, Criterion, Throughput};
use lux3d_core::runtime::{Pi3Pipeline, Pi3xPipeline, TripoSrPipeline};
use lux3d_core::ModelAssetOptions;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const MODELS_DIR_ENV: &str = "LUX3D_BENCH_MODELS_DIR";
const ASSETS_DIR_ENV: &str = "LUX3D_BENCH_ASSETS_DIR";

fn workspace_dir(env_key: &str, from_manifest: &str) -> PathBuf {
    std::env::var_os(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(from_manifest))
}

type InferFn = Box<dyn Fn() -> anyhow::Result<Duration> + 'static>;

/// Load weights once, return a closure timing one full `infer_from_path`
/// (preprocess + neural pass) per call — the same scope as the CLI's
/// `[stage] infer`.
fn make_infer(
    model: &str,
    source: &Path,
    device: &candle_core::Device,
) -> anyhow::Result<InferFn> {
    let models_dir = workspace_dir(MODELS_DIR_ENV, "../../models");
    let assets = ModelAssetOptions {
        canonical_dir: Some(models_dir.join(model)),
        cache_dir: None,
    };
    let device = device.clone();
    let source = source.to_path_buf();
    Ok(match model {
        "pi3" => {
            let pipeline = Pi3Pipeline::load(assets)?;
            Box::new(move || {
                let start = Instant::now();
                let _ = pipeline.infer_from_path_with_interval(&source, None, &device)?;
                Ok(start.elapsed())
            })
        }
        "pi3x" => {
            let pipeline = Pi3xPipeline::load(assets)?;
            Box::new(move || {
                let start = Instant::now();
                let _ = pipeline.infer_from_path(&source, None, None, &device)?;
                Ok(start.elapsed())
            })
        }
        "triposr" => {
            let pipeline = TripoSrPipeline::load(assets)?;
            Box::new(move || {
                let start = Instant::now();
                let _ = pipeline.infer_from_path(&source, &device)?;
                Ok(start.elapsed())
            })
        }
        other => anyhow::bail!("unknown model `{other}`"),
    })
}

/// One bench per (model, backend). Group naming and device sync follow
/// candle-core's bench pattern (`vulkan_pi3/iter`, `cuda_pi3/iter`, ...).
fn bench_model(c: &mut Criterion, device: &candle_core::Device, model: &str, source: &Path) {
    let models_dir = workspace_dir(MODELS_DIR_ENV, "../../models");
    if !models_dir.join(model).is_dir() {
        eprintln!(
            "[bench] skipping {model}: no canonical weights at {}",
            models_dir.join(model).display()
        );
        return;
    }

    let name = device.bench_name(model);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let infer = make_infer(model, source, device)?;
        let mut group = c.benchmark_group(&name);
        // Whole-model inference is seconds per iteration; keep criterion's
        // 10-sample floor and let it adapt the per-sample iteration count.
        group.sample_size(10);
        group.throughput(Throughput::Elements(1));
        group.bench_function("iter", |b| {
            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    black_box(&infer)().expect("model inference failed");
                }
                device.synchronize().expect("device sync failed");
                start.elapsed()
            })
        });
        group.finish();
        Ok::<(), anyhow::Error>(())
    }));

    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(err)) => eprintln!("[bench] {name} failed: {err:#}"),
        Err(_) => eprintln!("[bench] {name} panicked; skipped"),
    }
    let _ = device.synchronize();
}

/// Whole-model pipelines are gigabytes of device memory; unlike candle's
/// kernel micro-benches they cannot coexist. One process benches exactly one
/// (device, model) pair selected by LUX3D_BENCH_DEVICE / LUX3D_BENCH_MODEL —
/// the caller runs the 9 combinations sequentially.
fn criterion_benchmark(c: &mut Criterion) {
    let device_kind = std::env::var("LUX3D_BENCH_DEVICE").unwrap_or_else(|_| "cuda".to_string());
    let model = std::env::var("LUX3D_BENCH_MODEL").unwrap_or_else(|_| "pi3".to_string());
    let assets_dir = workspace_dir(ASSETS_DIR_ENV, "../../test-assets");

    let device = match device_kind.as_str() {
        "vulkan" => candle_core::Device::new_vulkan(0),
        "wgpu" => candle_core::Device::new_wgpu(0),
        "cuda" => candle_core::Device::new_cuda(0),
        "cpu" => Ok(candle_core::Device::Cpu),
        other => Err(candle_core::Error::Msg(format!(
            "unknown LUX3D_BENCH_DEVICE `{other}`"
        ))
        .bt()),
    }
    .expect("device init failed");

    let source = match model.as_str() {
        "pi3" | "pi3x" => assets_dir.join("pi3-frames"),
        "triposr" => assets_dir.join("mesh-input.png"),
        other => panic!("unknown LUX3D_BENCH_MODEL `{other}`"),
    };

    bench_model(c, &device, &model, &source);
}

criterion_group!(benches, criterion_benchmark);
