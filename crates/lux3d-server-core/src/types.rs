//! Shared server state and Axum extractors.

use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::extract::State;
use candle_core::Device;
use lux3d_core::ModelAssetOptions;
use reqwest::Client;

use crate::{
    config::{GenerationDefaults, ObservabilityConfig},
    files::FileStore,
    jobs::JobStore,
    models::ModelRegistry,
};

/// Compute backend used to drive inference.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DeviceBackend {
    #[default]
    Cuda,
    Cpu,
    Wgpu,
    Vulkan,
    Metal,
}

impl DeviceBackend {
    /// Backend used when the caller does not select one explicitly.
    pub fn platform_default() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::Metal
        }
        #[cfg(not(target_os = "macos"))]
        {
            Self::Cuda
        }
    }
}

/// Shared Lux3D server state used by Axum handlers.
#[derive(Debug, Clone)]
pub struct Lux3dState {
    pub model_assets: ModelAssetOptions,
    pub cuda_device: usize,
    pub backend: DeviceBackend,
    pub upload_dir: PathBuf,
    pub creation_time: u64,
    pub generation_defaults: GenerationDefaults,
    pub observability: ObservabilityConfig,
    pub jobs: JobStore,
    pub files: FileStore,
    pub models: ModelRegistry,
    pub http_client: Client,
    inference_lock: Arc<Mutex<()>>,
}

pub type SharedLux3dState = Arc<Lux3dState>;
pub type ExtractedLux3dState = State<SharedLux3dState>;

impl Lux3dState {
    pub(crate) fn new(
        model_assets: ModelAssetOptions,
        cuda_device: usize,
        backend: DeviceBackend,
        upload_dir: PathBuf,
        generation_defaults: GenerationDefaults,
        observability: ObservabilityConfig,
    ) -> Self {
        let creation_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);

        Self {
            model_assets,
            cuda_device,
            backend,
            upload_dir,
            creation_time,
            generation_defaults,
            observability,
            jobs: JobStore::default(),
            files: FileStore::default(),
            models: ModelRegistry::default(),
            http_client: Client::builder()
                .user_agent(format!("lux3d-server/{}", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("reqwest client should build"),
            inference_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn device(&self) -> anyhow::Result<Device> {
        let ordinal = self.cuda_device;
        match self.backend {
            DeviceBackend::Cpu => Ok(Device::Cpu),
            DeviceBackend::Cuda => Ok(Device::new_cuda(ordinal)?),
            DeviceBackend::Wgpu => Ok(Device::new_wgpu(ordinal)?),
            DeviceBackend::Vulkan => Ok(Device::new_vulkan(ordinal)?),
            DeviceBackend::Metal => Ok(Device::new_metal(ordinal)?),
        }
    }

    /// Runs blocking GPU inference on a worker thread while holding the server inference lock.
    pub async fn run_blocking<F, T>(&self, f: F) -> anyhow::Result<T>
    where
        F: FnOnce(&Self) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let state = self.clone();
        tokio::task::spawn_blocking(move || {
            let _guard = state
                .inference_lock
                .lock()
                .map_err(|_| anyhow::anyhow!("inference lock poisoned"))?;
            f(&state)
        })
        .await
        .map_err(|error| anyhow::anyhow!("inference task failed: {error}"))?
    }
}
