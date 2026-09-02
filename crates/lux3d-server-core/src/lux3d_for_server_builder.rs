//! Builder for the shared Lux3D server runtime state.

use std::{path::PathBuf, time::Duration};

use anyhow::{Context, Result};
use lux3d_core::ModelAssetOptions;

use crate::{
    config::{GenerationDefaults, ObservabilityConfig},
    types::{DeviceBackend, Lux3dState, SharedLux3dState},
};

pub mod defaults {
    use std::path::PathBuf;

    pub const CUDA_DEVICE: usize = 0;
    pub const UPLOAD_DIR: Option<PathBuf> = None;
    pub const MODEL_PATH: Option<PathBuf> = None;
    pub const CACHE_DIR: Option<PathBuf> = None;
}

/// Builder for creating a shared Lux3D server state.
pub struct Lux3dForServerBuilder {
    model_path: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    cuda_device: usize,
    backend: Option<DeviceBackend>,
    upload_dir: Option<PathBuf>,
    generation_defaults: GenerationDefaults,
    observability: ObservabilityConfig,
}

impl Default for Lux3dForServerBuilder {
    fn default() -> Self {
        Self {
            model_path: defaults::MODEL_PATH,
            cache_dir: defaults::CACHE_DIR,
            cuda_device: defaults::CUDA_DEVICE,
            backend: None,
            upload_dir: defaults::UPLOAD_DIR,
            generation_defaults: GenerationDefaults::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

impl Lux3dForServerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    pub fn with_model_path_optional(mut self, model_path: Option<PathBuf>) -> Self {
        if let Some(model_path) = model_path {
            self = self.with_model_path(model_path);
        }
        self
    }

    pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(cache_dir.into());
        self
    }

    pub fn with_cache_dir_optional(mut self, cache_dir: Option<PathBuf>) -> Self {
        if let Some(cache_dir) = cache_dir {
            self = self.with_cache_dir(cache_dir);
        }
        self
    }

    pub fn with_cuda_device(mut self, cuda_device: usize) -> Self {
        self.cuda_device = cuda_device;
        self
    }

    /// Select the compute backend. When unset, the platform default is used
    /// (CUDA on non-macOS, Metal on macOS), preserving prior behavior.
    pub fn with_device_backend(mut self, backend: DeviceBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn with_upload_dir(mut self, upload_dir: impl Into<PathBuf>) -> Self {
        self.upload_dir = Some(upload_dir.into());
        self
    }

    pub fn with_upload_dir_optional(mut self, upload_dir: Option<PathBuf>) -> Self {
        if let Some(upload_dir) = upload_dir {
            self = self.with_upload_dir(upload_dir);
        }
        self
    }

    pub fn with_job_ttl_secs(mut self, job_ttl_secs: u64) -> Self {
        self.generation_defaults.job_ttl_secs = job_ttl_secs;
        self
    }

    pub fn with_cleanup_interval_secs(mut self, cleanup_interval_secs: u64) -> Self {
        self.generation_defaults.cleanup_interval = Duration::from_secs(cleanup_interval_secs);
        self
    }

    pub fn with_observability(mut self, observability: ObservabilityConfig) -> Self {
        self.observability = observability;
        self
    }

    pub fn build(self) -> Result<SharedLux3dState> {
        let upload_dir = match self.upload_dir {
            Some(path) => {
                std::fs::create_dir_all(&path).with_context(|| {
                    format!("failed to create upload directory at `{}`", path.display())
                })?;
                path
            }
            None => {
                let dir = std::env::temp_dir().join("lux3d-server-uploads");
                std::fs::create_dir_all(&dir).with_context(|| {
                    format!("failed to create upload directory at `{}`", dir.display())
                })?;
                dir
            }
        };

        let model_assets = ModelAssetOptions {
            canonical_dir: self.model_path,
            cache_dir: self.cache_dir,
        };

        let backend = self.backend.unwrap_or_else(DeviceBackend::platform_default);

        Ok(std::sync::Arc::new(Lux3dState::new(
            model_assets,
            self.cuda_device,
            backend,
            upload_dir,
            self.generation_defaults,
            self.observability,
        )))
    }
}
