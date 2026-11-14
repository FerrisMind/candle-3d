//! In-memory model pipeline cache for faster repeated inference.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result, bail};
use lux3d_core::runtime::{
    Pi3Pipeline, Pi3xPipeline, Pi3xVoPipeline, Trellis2Pipeline, TripoSgPipeline, TripoSrPipeline,
    VggtPipeline,
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::types::Lux3dState;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStatus {
    Loaded,
    Unloaded,
    NotFound,
    Reloading,
    InternalError,
}

#[derive(Debug)]
enum CachedPipeline {
    Pi3(Pi3Pipeline),
    Pi3x(Pi3xPipeline),
    Pi3xVo(Pi3xVoPipeline),
    TripoSr(TripoSrPipeline),
    TripoSg(TripoSgPipeline),
    Trellis2(Trellis2Pipeline),
    Vggt(VggtPipeline),
}

#[derive(Debug, Default, Clone)]
pub struct ModelRegistry {
    inner: Arc<RwLock<HashMap<String, CachedPipeline>>>,
}

impl ModelRegistry {
    pub fn loaded_model_ids(&self) -> Vec<String> {
        self.inner
            .read()
            .ok()
            .map(|models| models.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn status(&self, model_id: &str) -> ModelLoadStatus {
        let Ok(models) = self.inner.read() else {
            return ModelLoadStatus::InternalError;
        };
        if models.contains_key(model_id) {
            ModelLoadStatus::Loaded
        } else if is_supported_model(model_id) {
            ModelLoadStatus::Unloaded
        } else {
            ModelLoadStatus::NotFound
        }
    }

    pub fn unload(&self, model_id: &str) -> Result<ModelLoadStatus> {
        let mut models = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        if models.remove(model_id).is_some() || is_supported_model(model_id) {
            Ok(ModelLoadStatus::Unloaded)
        } else {
            Ok(ModelLoadStatus::NotFound)
        }
    }

    pub fn reload(&self, state: &Lux3dState, model_id: &str) -> Result<ModelLoadStatus> {
        self.unload(model_id)?;
        self.ensure_loaded(state, model_id)?;
        Ok(ModelLoadStatus::Loaded)
    }

    pub fn ensure_loaded(&self, state: &Lux3dState, model_id: &str) -> Result<()> {
        {
            let models = self
                .inner
                .read()
                .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
            if models.contains_key(model_id) {
                return Ok(());
            }
        }

        let assets = state.model_assets.clone();
        let pipeline = match model_id {
            "pi3" => CachedPipeline::Pi3(
                Pi3Pipeline::load(assets).context("failed to load pi3 pipeline")?,
            ),
            "pi3x" => CachedPipeline::Pi3x(
                Pi3xPipeline::load(assets).context("failed to load pi3x pipeline")?,
            ),
            "pi3x_vo" => CachedPipeline::Pi3xVo(
                Pi3xVoPipeline::load(assets).context("failed to load pi3x vo pipeline")?,
            ),
            "triposr" => CachedPipeline::TripoSr(
                TripoSrPipeline::load(assets).context("failed to load triposr pipeline")?,
            ),
            "triposg" => CachedPipeline::TripoSg(
                TripoSgPipeline::load_with_rmbg(assets, state.rmbg_model_path.clone())
                    .context("failed to load triposg pipeline")?,
            ),
            // TRELLIS.2 loads from the canonical package only (the raw
            // `models/TRELLIS.2-4B` checkpoint layout is a CLI-only loader).
            // Stage weights are mmap-backed and touched lazily per stage.
            "trellis2" => CachedPipeline::Trellis2(
                Trellis2Pipeline::load(assets).context("failed to load trellis2 pipeline")?,
            ),
            "vggt" => CachedPipeline::Vggt(
                VggtPipeline::load(assets).context("failed to load vggt pipeline")?,
            ),
            other => bail!("unsupported model `{other}`"),
        };

        let mut models = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        models.insert(model_id.to_string(), pipeline);
        Ok(())
    }

    pub fn with_pi3<R>(&self, f: impl FnOnce(&Pi3Pipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("pi3") {
            Some(CachedPipeline::Pi3(pipeline)) => f(pipeline),
            _ => bail!("pi3 pipeline is not loaded"),
        }
    }

    pub fn with_pi3x<R>(&self, f: impl FnOnce(&Pi3xPipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("pi3x") {
            Some(CachedPipeline::Pi3x(pipeline)) => f(pipeline),
            _ => bail!("pi3x pipeline is not loaded"),
        }
    }

    pub fn with_pi3x_vo<R>(&self, f: impl FnOnce(&Pi3xVoPipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("pi3x_vo") {
            Some(CachedPipeline::Pi3xVo(pipeline)) => f(pipeline),
            _ => bail!("pi3x vo pipeline is not loaded"),
        }
    }

    pub fn with_triposr<R>(&self, f: impl FnOnce(&TripoSrPipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("triposr") {
            Some(CachedPipeline::TripoSr(pipeline)) => f(pipeline),
            _ => bail!("triposr pipeline is not loaded"),
        }
    }

    pub fn with_triposg<R>(&self, f: impl FnOnce(&TripoSgPipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("triposg") {
            Some(CachedPipeline::TripoSg(pipeline)) => f(pipeline),
            _ => bail!("triposg pipeline is not loaded"),
        }
    }

    pub fn with_vggt<R>(&self, f: impl FnOnce(&VggtPipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("vggt") {
            Some(CachedPipeline::Vggt(pipeline)) => f(pipeline),
            _ => bail!("vggt pipeline is not loaded"),
        }
    }

    pub fn with_trellis2<R>(&self, f: impl FnOnce(&Trellis2Pipeline) -> Result<R>) -> Result<R> {
        let models = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("model registry lock poisoned"))?;
        match models.get("trellis2") {
            Some(CachedPipeline::Trellis2(pipeline)) => f(pipeline),
            _ => bail!("trellis2 pipeline is not loaded"),
        }
    }
}

fn is_supported_model(model_id: &str) -> bool {
    matches!(
        model_id,
        "pi3" | "pi3x" | "pi3x_vo" | "triposr" | "triposg" | "trellis2" | "vggt"
    )
}

#[cfg(test)]
mod tests {
    use super::{ModelLoadStatus, ModelRegistry};

    #[test]
    fn trellis2_is_a_supported_model_id() {
        let registry = ModelRegistry::default();
        assert_eq!(
            ModelLoadStatus::Unloaded,
            registry.status("trellis2"),
            "`trellis2` must be a known (unloadable) model id without weights on disk"
        );
        assert_eq!(
            ModelLoadStatus::NotFound,
            registry.status("trellis-2"),
            "unknown model ids must stay NotFound"
        );
        assert_eq!(
            ModelLoadStatus::Unloaded,
            registry
                .unload("trellis2")
                .expect("unload of a supported id succeeds")
        );
        assert_eq!(
            ModelLoadStatus::NotFound,
            registry.unload("trellis-2").expect("unload unknown id")
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelOperationRequest {
    pub model_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelStatusResponse {
    pub model_id: String,
    pub status: ModelLoadStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
