//! Host, device, and diagnostics endpoints.

use std::path::PathBuf;

use axum::{Json, extract::State};
use candle_core::Device;
use dirs::home_dir;
use serde::Serialize;
use utoipa::ToSchema;

use crate::types::{DeviceBackend, ExtractedLux3dState};

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct SystemInfoResponse {
    pub object: &'static str,
    pub server_version: String,
    pub device: DeviceInfo,
    pub upload_dir: String,
    pub active_generations: usize,
    pub loaded_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DeviceInfo {
    pub backend: String,
    pub device_index: usize,
    pub available: bool,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DoctorReport {
    pub object: &'static str,
    pub ok: bool,
    pub checks: Vec<DoctorCheck>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct DoctorCheck {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/system/info",
    responses((status = 200, description = "Host and runtime information", body = SystemInfoResponse))
)]
pub async fn system_info(State(state): ExtractedLux3dState) -> Json<SystemInfoResponse> {
    let device = probe_device(state.backend, state.cuda_device);
    Json(SystemInfoResponse {
        object: "system.info",
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        device,
        upload_dir: state.upload_dir.display().to_string(),
        active_generations: state.jobs.active_count(),
        loaded_models: state.models.loaded_model_ids(),
    })
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/system/doctor",
    responses((status = 200, description = "Environment diagnostics", body = DoctorReport))
)]
pub async fn system_doctor(State(state): ExtractedLux3dState) -> Json<DoctorReport> {
    let mut checks = Vec::new();

    let device = probe_device(state.backend, state.cuda_device);
    checks.push(DoctorCheck {
        name: "device".to_string(),
        ok: device.available,
        detail: format!("backend={} index={}", device.backend, device.device_index),
    });

    let upload_ok = state.upload_dir.is_dir()
        && std::fs::create_dir_all(&state.upload_dir)
            .map(|_| true)
            .unwrap_or(false);
    checks.push(DoctorCheck {
        name: "upload_dir".to_string(),
        ok: upload_ok,
        detail: state.upload_dir.display().to_string(),
    });

    let ffmpeg_ok = which_ffmpeg();
    checks.push(DoctorCheck {
        name: "ffmpeg".to_string(),
        ok: ffmpeg_ok,
        detail: if ffmpeg_ok {
            "ffmpeg found on PATH".to_string()
        } else {
            "ffmpeg not found; video frame extraction may fail".to_string()
        },
    });

    let model_assets_ok = state.model_assets.canonical_dir.is_some()
        || state.model_assets.cache_dir.is_some()
        || default_hf_cache_exists();
    checks.push(DoctorCheck {
        name: "model_assets".to_string(),
        ok: model_assets_ok,
        detail: describe_model_assets(&state.model_assets.canonical_dir, &state.model_assets.cache_dir),
    });

    let ok = checks.iter().all(|check| check.ok || check.name == "ffmpeg");
    Json(DoctorReport {
        object: "system.doctor",
        ok,
        checks,
    })
}

fn probe_device(backend: DeviceBackend, device_index: usize) -> DeviceInfo {
    let backend_name = match backend {
        DeviceBackend::Cpu => "cpu",
        DeviceBackend::Cuda => "cuda",
        DeviceBackend::Wgpu => "wgpu",
        DeviceBackend::Vulkan => "vulkan",
        DeviceBackend::Metal => "metal",
    };

    let available = match backend {
        DeviceBackend::Cpu => true,
        DeviceBackend::Cuda => Device::new_cuda(device_index).is_ok(),
        DeviceBackend::Wgpu => Device::new_wgpu(device_index).is_ok(),
        DeviceBackend::Vulkan => Device::new_vulkan(device_index).is_ok(),
        DeviceBackend::Metal => Device::new_metal(device_index).is_ok(),
    };

    DeviceInfo {
        backend: backend_name.to_string(),
        device_index,
        available,
    }
}

fn which_ffmpeg() -> bool {
    std::process::Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn default_hf_cache_exists() -> bool {
    home_dir()
        .map(|home| home.join(".cache").join("huggingface").is_dir())
        .unwrap_or(false)
}

fn describe_model_assets(canonical_dir: &Option<PathBuf>, cache_dir: &Option<PathBuf>) -> String {
    match (canonical_dir, cache_dir) {
        (Some(path), Some(cache)) => {
            format!(
                "canonical_dir=`{}`, cache_dir=`{}`",
                path.display(),
                cache.display()
            )
        }
        (Some(path), None) => format!("canonical_dir=`{}`", path.display()),
        (None, Some(cache)) => format!("cache_dir=`{}`", cache.display()),
        (None, None) => "using default Hugging Face cache when available".to_string(),
    }
}

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
