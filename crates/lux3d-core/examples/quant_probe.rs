//! Diagnostic probe: GGUF dequantization vs quantized matmul per backend.
//!
//! ```text
//! cargo run --release --example quant_probe -p lux3d-core --features cuda,vulkan,wgpu -- \
//!   --device cuda
//! ```
//!
//! Quantizes the same deterministic `[256, 1024]` weight tensor to every GGUF
//! type, then compares:
//! * dequantized weights against the F32 source (storage error);
//! * `QMatMul` output against F32 matmul with the dequantized weights (kernel
//!   error, including activation quantization for integer types).
//!
//! This isolates a quantized-matmul kernel defect from chaotic amplification in
//! a multi-step diffusion sampler.

use candle_core::{
    DType, Device, Module, Result, Tensor,
    quantized::{GgmlDType, QMatMul, QTensor},
};

fn main() -> Result<()> {
    let mut device_name = "cpu".to_string();
    let mut device_id = 0usize;
    let mut rows = 4usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--device" => device_name = args.next().unwrap_or_else(|| "cpu".into()),
            "--device-id" => {
                device_id = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(0)
            }
            "--rows" => {
                rows = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(4)
            }
            other => candle_core::bail!("unexpected argument `{other}`"),
        }
    }
    let dev = device(&device_name, device_id)?;

    // Deterministic pseudo-random weights in a realistic range.
    let n = 256 * 1024;
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((state >> 33) as f32) / ((1u64 << 31) as f32);
        values.push((u * 2.0 - 1.0) * 0.05);
    }
    let x = Tensor::from_vec(values, (256, 1024), &dev)?.to_dtype(DType::F32)?;
    let xv = x.flatten_all()?.to_vec1::<f32>()?;
    let norm = xv
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt();

    let act = Tensor::from_vec(
        (0..rows * 1024)
            .map(|i| ((i as f32) * 0.001).sin())
            .collect::<Vec<_>>(),
        (rows, 1024),
        &dev,
    )?;

    println!("device: {device_name}, rows: {rows}");
    for dtype in [
        GgmlDType::F16,
        GgmlDType::Q8_0,
        GgmlDType::Q5_0,
        GgmlDType::Q4_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let q = QTensor::quantize(&x, dtype)?;
        let y = q
            .dequantize(&dev)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let qm = QMatMul::from_qtensor(QTensor::quantize(&x, dtype)?)?;

        let qout = qm
            .forward(&act)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        let w = q.dequantize(&dev)?;
        let fout = act
            .matmul(&w.t()?)?
            .flatten_all()?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;

        let mut merr = 0.0f64;
        let mut mnorm = 0.0f64;
        for (a, b) in fout.iter().zip(qout.iter()) {
            let d = (*a - *b) as f64;
            merr += d * d;
            mnorm += (*a as f64) * (*a as f64);
        }
        let mut err = 0.0f64;
        let mut max = 0.0f32;
        for (a, b) in xv.iter().zip(y.iter()) {
            let d = (*a - *b) as f64;
            err += d * d;
            max = max.max(d.abs() as f32);
        }
        println!(
            "{dtype:?}: weight_rel_l2 {:.4e} max {:.4e}; matmul_rel_l2 {:.4e}",
            err.sqrt() / norm,
            max,
            merr.sqrt() / mnorm.sqrt()
        );
    }
    Ok(())
}

fn device(name: &str, index: usize) -> Result<Device> {
    match name {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(index)
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = index;
                candle_core::bail!("binary was built without the `cuda` feature")
            }
        }
        "vulkan" => {
            #[cfg(feature = "vulkan")]
            {
                Device::new_vulkan(index)
            }
            #[cfg(not(feature = "vulkan"))]
            {
                let _ = index;
                candle_core::bail!("binary was built without the `vulkan` feature")
            }
        }
        "wgpu" => {
            #[cfg(feature = "wgpu")]
            {
                Device::new_wgpu(index)
            }
            #[cfg(not(feature = "wgpu"))]
            {
                let _ = index;
                candle_core::bail!("binary was built without the `wgpu` feature")
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(index)
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = index;
                candle_core::bail!("binary was built without the `metal` feature")
            }
        }
        other => candle_core::bail!("unknown device `{other}`"),
    }
}
