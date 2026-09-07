//! Chrome-trace probe: run the pi3x pipeline on vulkan with per-dispatch
//! spans (shader names included) so model-level regressions can be
//! attributed to specific kernels. Output: vk_pi3x_trace.json.
//!
//! Run: cargo run --release -p lux3d-core --features vulkan \
//!      --example vk_pi3x_trace
use std::path::Path;
use tracing_chrome::ChromeLayerBuilder;
use tracing_subscriber::prelude::*;

fn main() -> anyhow::Result<()> {
    let (chrome_layer, guard) = ChromeLayerBuilder::new().file("vk_pi3x_trace.json").build();
    let _subscriber = tracing_subscriber::registry().with(chrome_layer).set_default();

    let models_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models");
    let assets_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-assets");
    let assets = lux3d_core::ModelAssetOptions {
        canonical_dir: Some(models_dir.join("pi3x")),
        cache_dir: None,
    };
    let device = candle_core::Device::new_vulkan(0)?;
    let pipeline = lux3d_core::runtime::Pi3xPipeline::load(assets)?;
    let source = assets_dir.join("pi3-frames");

    // Warm iteration populates pipeline caches/compiles; the traced window
    // is the second iteration so the trace reflects steady state.
    let _ = pipeline.infer_from_path(&source, None, None, &device)?;
    println!("warm done, tracing steady-state iteration");
    let _ = pipeline.infer_from_path(&source, None, None, &device)?;
    drop(guard);
    println!("trace written");
    Ok(())
}
