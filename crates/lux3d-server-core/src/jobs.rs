//! Async generation job store and inference execution.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use lux3d_core::runtime::Pi3xInjectConditions;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{
    api_types::{
        GenerationError, GenerationObject, GenerationStatus, MeshGenerationOptions,
        PointCloudGenerationOptions,
    },
    metrics::{record_generation_finished, record_generation_started},
    types::Lux3dState,
    upload::{normalize_source_for_frames, normalize_source_for_image},
    webhooks::deliver_generation_webhook,
};

#[derive(Debug, Clone)]
pub struct JobRecord {
    pub object: GenerationObject,
    pub output_path: Option<PathBuf>,
    pub content_type: String,
    pub download_filename: String,
    pub workspace: PathBuf,
    pub webhook_url: Option<String>,
    cancel_token: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct JobStore {
    inner: Arc<RwLock<HashMap<String, JobRecord>>>,
}

impl Default for JobStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl JobStore {
    pub fn insert(&self, record: JobRecord) {
        if let Ok(mut jobs) = self.inner.write() {
            jobs.insert(record.object.id.clone(), record);
        }
    }

    pub fn get(&self, id: &str) -> Option<JobRecord> {
        self.inner
            .read()
            .ok()
            .and_then(|jobs| jobs.get(id).cloned())
    }

    pub fn list(&self, limit: usize) -> Vec<GenerationObject> {
        let Ok(jobs) = self.inner.read() else {
            return Vec::new();
        };
        let mut entries: Vec<_> = jobs.values().map(|record| record.object.clone()).collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.created_at));
        entries.truncate(limit);
        entries
    }

    pub fn active_count(&self) -> usize {
        self.inner
            .read()
            .ok()
            .map(|jobs| {
                jobs.values()
                    .filter(|record| {
                        matches!(
                            record.object.status.as_str(),
                            "queued" | "in_progress"
                        )
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    fn update<F>(&self, id: &str, update: F) -> Result<()>
    where
        F: FnOnce(&mut JobRecord),
    {
        let mut jobs = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("job store lock poisoned"))?;
        let record = jobs
            .get_mut(id)
            .with_context(|| format!("generation `{id}` not found"))?;
        update(record);
        Ok(())
    }

    pub fn mark_in_progress(&self, id: &str) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::InProgress.as_str().to_string();
            record.object.progress = 10;
        })
    }

    pub fn mark_completed(&self, id: &str, output_path: PathBuf) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Completed.as_str().to_string();
            record.object.progress = 100;
            record.object.completed_at = Some(now_unix());
            record.object.error = None;
            record.output_path = Some(output_path);
        })
    }

    pub fn mark_failed(&self, id: &str, message: String) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Failed.as_str().to_string();
            record.object.completed_at = Some(now_unix());
            record.object.error = Some(GenerationError {
                code: "generation_failed".to_string(),
                message,
            });
        })
    }

    pub fn mark_cancelled(&self, id: &str) -> Result<()> {
        self.update(id, |record| {
            record.object.status = GenerationStatus::Cancelled.as_str().to_string();
            record.object.completed_at = Some(now_unix());
            record.object.error = Some(GenerationError {
                code: "cancelled".to_string(),
                message: "generation was cancelled".to_string(),
            });
        })
    }

    pub fn request_cancel(&self, id: &str) -> Result<bool> {
        let jobs = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("job store lock poisoned"))?;
        let Some(record) = jobs.get(id) else {
            return Ok(false);
        };
        if matches!(
            record.object.status.as_str(),
            "completed" | "failed" | "cancelled"
        ) {
            return Ok(false);
        }
        record.cancel_token.cancel();
        Ok(true)
    }

    pub fn is_cancelled(&self, id: &str) -> bool {
        self.inner
            .read()
            .ok()
            .and_then(|jobs| jobs.get(id).map(|record| record.cancel_token.is_cancelled()))
            .unwrap_or(false)
    }

    pub fn remove(&self, id: &str) -> Option<JobRecord> {
        let mut jobs = self.inner.write().ok()?;
        jobs.remove(id)
    }

    pub fn cleanup_expired(&self, now: u64) -> usize {
        let expired: Vec<String> = self
            .inner
            .read()
            .ok()
            .map(|jobs| {
                jobs
                    .iter()
                    .filter_map(|(id, record)| {
                        record
                            .object
                            .expires_at
                            .filter(|expires_at| *expires_at <= now)
                            .map(|_| id.clone())
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut removed = 0;
        for id in expired {
            if let Some(record) = self.remove(&id) {
                let _ = fs::remove_dir_all(&record.workspace);
                removed += 1;
            }
        }
        removed
    }
}

pub fn new_job_id() -> String {
    format!("gen_{}", Uuid::new_v4().simple())
}

pub fn job_workspace(upload_dir: &Path, job_id: &str) -> PathBuf {
    upload_dir.join("jobs").join(job_id)
}

pub struct NewJobRecord {
    pub job_id: String,
    pub model: String,
    pub output_format: String,
    pub workspace: PathBuf,
    pub content_type: String,
    pub download_filename: String,
    pub ttl_secs: u64,
    pub webhook_url: Option<String>,
}

pub fn create_job_record(params: NewJobRecord) -> JobRecord {
    let created_at = now_unix();
    JobRecord {
        object: GenerationObject {
            id: params.job_id,
            object: "generation",
            status: GenerationStatus::Queued.as_str().to_string(),
            model: params.model,
            created_at,
            completed_at: None,
            expires_at: Some(created_at.saturating_add(params.ttl_secs)),
            progress: 0,
            output_format: params.output_format,
            error: None,
        },
        output_path: None,
        content_type: params.content_type,
        download_filename: params.download_filename,
        workspace: params.workspace,
        webhook_url: params.webhook_url,
        cancel_token: CancellationToken::new(),
    }
}

pub async fn spawn_point_cloud_generation(
    state: Arc<Lux3dState>,
    record: JobRecord,
    source: PathBuf,
    conditions: Option<PathBuf>,
    options: PointCloudGenerationOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    ensure!(
        matches!(model.as_str(), "pi3" | "pi3x"),
        "unsupported point-cloud model `{model}`; expected `pi3` or `pi3x`"
    );

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state.jobs.get(&job_id).expect("job just inserted").object.clone();
    record_generation_started(&model);

    let state_for_task = state.clone();
    tokio::spawn(async move {
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", 0.0);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        if let Err(error) = state_for_task.jobs.mark_in_progress(&job_id) {
            tracing::error!("failed to mark job in progress: {error}");
            return;
        }

        let started = Instant::now();
        let cache_model = if model == "pi3" {
            "pi3"
        } else if options.vo.unwrap_or(false) {
            "pi3x_vo"
        } else {
            "pi3x"
        };
        let job_id_for_blocking = job_id.clone();

        let result = state_for_task
            .run_blocking({
                let workspace = state_for_task
                    .jobs
                    .get(&job_id)
                    .map(|record| record.workspace.clone());
                let source = source.clone();
                let options = options.clone();
                let model = model.clone();
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, cache_model)
                        .with_context(|| format!("failed to load `{cache_model}` pipeline"))?;
                    let workspace =
                        workspace.ok_or_else(|| anyhow::anyhow!("job workspace missing"))?;
                    let source = normalize_source_for_frames(&source, &workspace)?;
                    let device = runtime.device()?;
                    let output_path = workspace.join(format!("{model}.ply"));
                    match model.as_str() {
                        "pi3" => runtime.models.with_pi3(|pipeline| {
                            let inference = pipeline
                                .infer_from_path_with_interval(&source, options.interval, &device)
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_ply(&inference, &output_path)
                                .map_err(anyhow::Error::from)
                        })?,
                        "pi3x" if options.vo.unwrap_or(false) => {
                            let inject_conditions = parse_inject_conditions(&options.inject_condition);
                            runtime.models.with_pi3x_vo(|pipeline| {
                                let inference = pipeline
                                    .infer_from_path(
                                        &source,
                                        options.interval,
                                        options.chunk_size,
                                        options.overlap,
                                        options.conf_threshold,
                                        inject_conditions,
                                        &device,
                                    )
                                    .map_err(anyhow::Error::from)?;
                                pipeline
                                    .export_ply(&inference, &output_path)
                                    .map_err(anyhow::Error::from)
                            })?
                        }
                        "pi3x" => runtime.models.with_pi3x(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(
                                    &source,
                                    conditions.as_deref(),
                                    options.interval,
                                    &device,
                                )
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_ply(&inference, &output_path)
                                .map_err(anyhow::Error::from)
                        })?,
                        _ => bail!("unsupported point-cloud model `{model}`"),
                    }
                    Ok(output_path)
                }
            })
            .await;

        let elapsed = started.elapsed().as_secs_f64();
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", elapsed);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        match result {
            Ok(output_path) => {
                if let Err(error) = state_for_task
                    .jobs
                    .mark_completed(&job_id, output_path)
                {
                    tracing::error!("failed to mark job completed: {error}");
                } else {
                    record_generation_finished(&model, "completed", elapsed);
                    maybe_deliver_webhook(&state_for_task, &job_id, "generation.completed").await;
                }
            }
            Err(error) => {
                let status = if error.to_string().contains("cancelled") {
                    let _ = state_for_task.jobs.mark_cancelled(&job_id);
                    "cancelled"
                } else if let Err(update_error) = state_for_task
                    .jobs
                    .mark_failed(&job_id, error.to_string())
                {
                    tracing::error!("failed to mark job failed: {update_error}");
                    "failed"
                } else {
                    "failed"
                };
                record_generation_finished(&model, status, elapsed);
                maybe_deliver_webhook(&state_for_task, &job_id, "generation.failed").await;
            }
        }
    });

    Ok(response)
}

pub async fn spawn_mesh_generation(
    state: Arc<Lux3dState>,
    record: JobRecord,
    source: PathBuf,
    options: MeshGenerationOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    ensure!(
        model == "triposr",
        "unsupported mesh model `{model}`; expected `triposr`"
    );
    ensure!(options.mc_resolution >= 2, "mc_resolution must be at least 2");

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state.jobs.get(&job_id).expect("job just inserted").object.clone();
    record_generation_started(&model);

    let state_for_task = state.clone();
    tokio::spawn(async move {
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", 0.0);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        if let Err(error) = state_for_task.jobs.mark_in_progress(&job_id) {
            tracing::error!("failed to mark job in progress: {error}");
            return;
        }

        let started = Instant::now();
        let job_id_for_blocking = job_id.clone();
        let result = state_for_task
            .run_blocking({
                let workspace = state_for_task
                    .jobs
                    .get(&job_id)
                    .map(|record| record.workspace.clone());
                let source = source.clone();
                let mc_resolution = options.mc_resolution;
                let mc_threshold = options.mc_threshold;
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, "triposr")
                        .context("failed to load triposr pipeline")?;
                    let workspace =
                        workspace.ok_or_else(|| anyhow::anyhow!("job workspace missing"))?;
                    fs::create_dir_all(&workspace).with_context(|| {
                        format!(
                            "failed to create generation workspace at `{}`",
                            workspace.display()
                        )
                    })?;
                    let source = normalize_source_for_image(&source)?;
                    let device = runtime.device()?;
                    let output_path = workspace.join("triposr.obj");
                    runtime.models.with_triposr(|pipeline| {
                        let inference = pipeline
                            .infer_from_path(&source, &device)
                            .map_err(anyhow::Error::from)?;
                        let mesh = pipeline.extract_mesh(
                            &inference.scene_codes,
                            mc_resolution,
                            mc_threshold,
                            8192,
                        )
                        .map_err(anyhow::Error::from)?;
                        pipeline
                            .export_obj(&mesh, &output_path)
                            .map_err(anyhow::Error::from)
                    })?;
                    Ok(output_path)
                }
            })
            .await;

        let elapsed = started.elapsed().as_secs_f64();
        if state_for_task.jobs.is_cancelled(&job_id) {
            let _ = state_for_task.jobs.mark_cancelled(&job_id);
            record_generation_finished(&model, "cancelled", elapsed);
            maybe_deliver_webhook(&state_for_task, &job_id, "generation.cancelled").await;
            return;
        }

        match result {
            Ok(output_path) => {
                if let Err(error) = state_for_task
                    .jobs
                    .mark_completed(&job_id, output_path)
                {
                    tracing::error!("failed to mark job completed: {error}");
                } else {
                    record_generation_finished(&model, "completed", elapsed);
                    maybe_deliver_webhook(&state_for_task, &job_id, "generation.completed").await;
                }
            }
            Err(error) => {
                let status = if error.to_string().contains("cancelled") {
                    let _ = state_for_task.jobs.mark_cancelled(&job_id);
                    "cancelled"
                } else if let Err(update_error) = state_for_task
                    .jobs
                    .mark_failed(&job_id, error.to_string())
                {
                    tracing::error!("failed to mark job failed: {update_error}");
                    "failed"
                } else {
                    "failed"
                };
                record_generation_finished(&model, status, elapsed);
                maybe_deliver_webhook(&state_for_task, &job_id, "generation.failed").await;
            }
        }
    });

    Ok(response)
}

async fn maybe_deliver_webhook(state: &Lux3dState, job_id: &str, event: &'static str) {
    let Some(record) = state.jobs.get(job_id) else {
        return;
    };
    let Some(webhook_url) = record.webhook_url.clone() else {
        return;
    };
    deliver_generation_webhook(&state.http_client, &webhook_url, event, &record.object).await;
}

fn parse_inject_conditions(values: &[String]) -> Pi3xInjectConditions {
    Pi3xInjectConditions {
        pose: values.iter().any(|value| value.eq_ignore_ascii_case("pose")),
        depth: values.iter().any(|value| value.eq_ignore_ascii_case("depth")),
        ray: values.iter().any(|value| {
            value.eq_ignore_ascii_case("ray") || value.eq_ignore_ascii_case("intrinsic")
        }),
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
