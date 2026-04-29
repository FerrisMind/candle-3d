use std::{
    fs,
    fs::{File, OpenOptions},
    io,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard, OnceLock},
};

use anyhow::{Context, ensure};
use fs2::FileExt;
use hf_hub::api::sync::ApiBuilder;

use crate::{ModelAssetOptions, ModelFamily};

fn process_lock() -> &'static Mutex<()> {
    static PROCESS_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    PROCESS_LOCK.get_or_init(|| Mutex::new(()))
}

fn lock_path() -> PathBuf {
    std::env::temp_dir().join("ferrismind-lux3d-gpu-tests.lock")
}

fn raw_model_lock_path(family: ModelFamily) -> PathBuf {
    std::env::temp_dir().join(format!(
        "ferrismind-lux3d-raw-model-{}.lock",
        family.as_str()
    ))
}

#[derive(Debug)]
pub struct GpuTestLock {
    _process_guard: MutexGuard<'static, ()>,
    file: File,
}

impl GpuTestLock {
    pub fn acquire() -> io::Result<Self> {
        let process_guard = process_lock()
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(false)
            .open(lock_path())?;
        file.lock_exclusive()?;
        Ok(Self {
            _process_guard: process_guard,
            file,
        })
    }
}

impl Drop for GpuTestLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

pub fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

pub fn runtime_root() -> PathBuf {
    workspace_root()
        .parent()
        .expect("runtime root")
        .to_path_buf()
}

pub fn canonical_package_dir(family: ModelFamily) -> PathBuf {
    runtime_root()
        .join("3d")
        .join("canonical-weights")
        .join(family.as_str())
}

pub fn model_asset_options(family: ModelFamily) -> ModelAssetOptions {
    ModelAssetOptions {
        canonical_dir: Some(canonical_package_dir(family)),
        cache_dir: None,
    }
}

pub fn raw_model_dir_env_var(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Pi3 => "LUX3D_PI3_RAW_MODEL_DIR",
        ModelFamily::Pi3x => "LUX3D_PI3X_RAW_MODEL_DIR",
        ModelFamily::TripoSr => "LUX3D_TRIPOSR_RAW_MODEL_DIR",
    }
}

pub fn resolve_raw_model_dir_for_tests(family: ModelFamily) -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os(raw_model_dir_env_var(family)).map(PathBuf::from) {
        validate_raw_model_dir(family, &path).with_context(|| {
            format!(
                "raw model dir from {} is invalid",
                raw_model_dir_env_var(family)
            )
        })?;
        return Ok(path);
    }

    download_raw_model_dir_for_tests(family)
}

fn validate_raw_model_dir(family: ModelFamily, path: &Path) -> anyhow::Result<()> {
    ensure!(
        path.is_dir(),
        "raw model dir for {} does not exist: {}",
        family,
        path.display()
    );
    for filename in raw_model_required_files(family) {
        let file = path.join(filename);
        ensure!(
            file.is_file(),
            "raw model dir for {} is missing required file {}",
            family,
            file.display()
        );
    }
    Ok(())
}

fn download_raw_model_dir_for_tests(family: ModelFamily) -> anyhow::Result<PathBuf> {
    let cache_base = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("LuxRT")
        .join("raw-test-models");
    let raw_dir = cache_base.join(family.as_str());
    fs::create_dir_all(&raw_dir)
        .with_context(|| format!("failed to create {}", raw_dir.display()))?;

    let lock_file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(raw_model_lock_path(family))
        .with_context(|| format!("failed to open raw-model lock for {}", family))?;
    lock_file
        .lock_exclusive()
        .with_context(|| format!("failed to lock raw-model download for {}", family))?;

    if validate_raw_model_dir(family, &raw_dir).is_ok() {
        return Ok(raw_dir);
    }

    let hub_cache = cache_base.join("hf-hub");
    fs::create_dir_all(&hub_cache)
        .with_context(|| format!("failed to create {}", hub_cache.display()))?;

    let token = std::env::var("HF_TOKEN")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let api = ApiBuilder::from_env()
        .with_cache_dir(hub_cache)
        .with_token(token)
        .with_user_agent("ferrismind", "LuxRT-tests")
        .with_progress(false)
        .build()
        .with_context(|| {
            format!(
                "failed to initialize Hugging Face API client for {}",
                official_raw_model_repo_id(family)
            )
        })?;
    let repo = api.model(official_raw_model_repo_id(family).to_string());

    for filename in raw_model_required_files(family) {
        let fetched = repo.get(filename).with_context(|| {
            format!(
                "failed to download {} from {}",
                filename,
                official_raw_model_repo_id(family)
            )
        })?;
        let target = raw_dir.join(filename);
        fs::copy(&fetched, &target).with_context(|| {
            format!(
                "failed to copy downloaded raw model asset {} to {}",
                fetched.display(),
                target.display()
            )
        })?;
    }

    validate_raw_model_dir(family, &raw_dir)?;
    Ok(raw_dir)
}

fn official_raw_model_repo_id(family: ModelFamily) -> &'static str {
    match family {
        ModelFamily::Pi3 => "yyfz233/Pi3",
        ModelFamily::Pi3x => "yyfz233/Pi3X",
        ModelFamily::TripoSr => "stabilityai/TripoSR",
    }
}

fn raw_model_required_files(family: ModelFamily) -> &'static [&'static str] {
    match family {
        ModelFamily::Pi3 => &["model.safetensors", "config.json"],
        ModelFamily::Pi3x => &["model.safetensors", "config.json"],
        ModelFamily::TripoSr => &["model.ckpt", "config.yaml"],
    }
}
