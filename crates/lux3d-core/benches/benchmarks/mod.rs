pub(crate) mod models;

use candle_core::Device;

/// Mirrors candle-core's bench device trait: prefix group names with the
/// backend (`vulkan_pi3`, `cuda_pi3`, ...).
pub(crate) trait BenchDevice {
    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu => format!("cpu_{}", name.into()),
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
            Device::Wgpu(_) => format!("wgpu_{}", name.into()),
            Device::Vulkan(_) => format!("vulkan_{}", name.into()),
        }
    }
}
