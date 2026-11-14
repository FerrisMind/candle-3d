//! Async generation job store and inference execution.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use lux3d_core::export::{MeshFormat, PointCloudFormat};
use lux3d_core::runtime::{Pi3xInjectConditions, Trellis2RunOptions, TripoSgRunOptions};
use lux3d_core::{Trellis2PipelineType, export::pbrvox};
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
                        matches!(record.object.status.as_str(), "queued" | "in_progress")
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
            .and_then(|jobs| {
                jobs.get(id)
                    .map(|record| record.cancel_token.is_cancelled())
            })
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
                jobs.iter()
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
    let format = PointCloudFormat::from_extension(&record.object.output_format)
        .map_err(anyhow::Error::from)?;
    ensure!(
        matches!(model.as_str(), "pi3" | "pi3x" | "vggt"),
        "unsupported point-cloud model `{model}`; expected \"pi3\", \"pi3x\" or \"vggt\""
    );
    if model == "vggt" {
        ensure!(
            conditions.is_none(),
            "`conditions` is not supported for `vggt`"
        );
        ensure!(
            !options.vo.unwrap_or(false),
            "`vo` is not supported for `vggt`"
        );
        ensure!(
            options.chunk_size.is_none(),
            "`chunk_size` is not supported for `vggt`"
        );
        ensure!(
            options.overlap.is_none(),
            "`overlap` is not supported for `vggt`"
        );
        ensure!(
            options.inject_condition.is_empty(),
            "`inject_condition` is not supported for `vggt`"
        );
    }

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state
        .jobs
        .get(&job_id)
        .expect("job just inserted")
        .object
        .clone();
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
        } else if model == "vggt" {
            "vggt"
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
                    let output_path = workspace.join(format!("{model}.{}", format.extension()));
                    match model.as_str() {
                        "pi3" => runtime.models.with_pi3(|pipeline| {
                            let inference = pipeline
                                .infer_from_path_with_interval(&source, options.interval, &device)
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_point_cloud(&inference, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "pi3x" if options.vo.unwrap_or(false) => {
                            let inject_conditions =
                                parse_inject_conditions(&options.inject_condition);
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
                                    .export_point_cloud(&inference, &output_path)
                                    .map_err(anyhow::Error::from)?;
                                Ok(())
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
                                .export_point_cloud(&inference, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "vggt" => runtime.models.with_vggt(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(&source, options.interval, &device)
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_point_cloud_with_threshold(
                                    &inference,
                                    &output_path,
                                    options.conf_threshold.unwrap_or(1.0),
                                )
                                .map_err(anyhow::Error::from)?;
                            Ok(())
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
                if let Err(error) = state_for_task.jobs.mark_completed(&job_id, output_path) {
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
                } else if let Err(update_error) =
                    state_for_task.jobs.mark_failed(&job_id, error.to_string())
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

/// Build the TripoSG run options from the reference defaults plus explicit
/// mesh-request overrides (`query_chunk_size` defaults to 16384). Pure helper
/// so the mapping and validation can be unit-tested without a server state.
fn triposg_options_from_mesh_options(options: &MeshGenerationOptions) -> Result<TripoSgRunOptions> {
    let defaults = TripoSgRunOptions::default();
    let run_options = TripoSgRunOptions {
        steps: options.steps.unwrap_or(defaults.steps),
        guidance_scale: options.guidance_scale.unwrap_or(defaults.guidance_scale),
        seed: options.seed.unwrap_or(defaults.seed),
        num_tokens: options.num_tokens.unwrap_or(defaults.num_tokens),
        dense_octree_depth: options
            .dense_octree_depth
            .unwrap_or(defaults.dense_octree_depth),
        hierarchical_octree_depth: options
            .hierarchical_octree_depth
            .unwrap_or(defaults.hierarchical_octree_depth),
        use_flash_decoder: options
            .use_flash_decoder
            .unwrap_or(defaults.use_flash_decoder),
        flash_octree_depth: options
            .flash_octree_depth
            .unwrap_or(defaults.flash_octree_depth),
        query_chunk_size: options
            .query_chunk_size
            .unwrap_or(defaults.query_chunk_size),
    };
    run_options.validate().map_err(anyhow::Error::from)?;
    Ok(run_options)
}

/// Build the TRELLIS.2 run options from the reference defaults plus explicit
/// mesh-request overrides. Pure helper so the mapping and validation can be
/// unit-tested without a server state.
fn trellis2_options_from_mesh_options(
    options: &MeshGenerationOptions,
) -> Result<Trellis2RunOptions> {
    let defaults = Trellis2RunOptions::default();
    let pipeline_type = match options.pipeline_type.as_deref() {
        None => defaults.pipeline_type,
        Some("512") => Trellis2PipelineType::R512,
        // Accepted as a known reference value; the runtime rejects the
        // not-yet-implemented single-resolution path with a clear job error.
        Some("1024") => Trellis2PipelineType::R1024,
        Some("1024_cascade") => Trellis2PipelineType::Cascade1024,
        Some(other) => bail!(
            "unknown trellis2 `pipeline_type` `{other}`; expected `512`, `1024` or `1024_cascade`"
        ),
    };
    let steps = options.steps.unwrap_or(defaults.ss_steps);
    Ok(Trellis2RunOptions {
        seed: options.seed.unwrap_or(defaults.seed),
        pipeline_type,
        ss_steps: steps,
        shape_steps: options.steps.unwrap_or(defaults.shape_steps),
        tex_steps: options.steps.unwrap_or(defaults.tex_steps),
        with_texture: options.texture.unwrap_or(defaults.with_texture),
        max_num_tokens: options.max_num_tokens.unwrap_or(defaults.max_num_tokens),
        preprocess: defaults.preprocess,
    })
}

/// Pure mesh-request guard shared by [`spawn_mesh_generation`]: the model
/// allow-list plus per-model option validation, unit-tested without a server
/// state.
fn validate_mesh_generation_request(model: &str, options: &MeshGenerationOptions) -> Result<()> {
    ensure!(
        matches!(model, "triposr" | "triposg" | "trellis2"),
        "unsupported mesh model `{model}`; expected `triposr`, `triposg` or `trellis2`"
    );
    if model == "triposr" {
        ensure!(
            options.mc_resolution >= 2,
            "mc_resolution must be at least 2"
        );
        for (name, present) in [
            ("steps", options.steps.is_some()),
            ("guidance_scale", options.guidance_scale.is_some()),
            ("seed", options.seed.is_some()),
            ("num_tokens", options.num_tokens.is_some()),
            ("dense_octree_depth", options.dense_octree_depth.is_some()),
            (
                "hierarchical_octree_depth",
                options.hierarchical_octree_depth.is_some(),
            ),
            ("use_flash_decoder", options.use_flash_decoder.is_some()),
            ("flash_octree_depth", options.flash_octree_depth.is_some()),
            ("query_chunk_size", options.query_chunk_size.is_some()),
            ("pipeline_type", options.pipeline_type.is_some()),
            ("max_num_tokens", options.max_num_tokens.is_some()),
            ("texture", options.texture.is_some()),
        ] {
            ensure!(!present, "`{name}` is not supported for `triposr`");
        }
        return Ok(());
    }
    if model == "trellis2" {
        // TRELLIS.2 owns the voxel grid (512 or 1024_cascade pipeline types),
        // so the TripoSR marching-cubes knobs and the TripoSG octree/flash
        // knobs must stay at their defaults here.
        ensure!(
            options.mc_resolution == 256,
            "`mc_resolution` is not supported for `trellis2`; only the default 256 is supported"
        );
        ensure!(
            options.mc_threshold == 25.0,
            "`mc_threshold` is not supported for `trellis2`; only the default 25.0 is supported"
        );
        for (name, present) in [
            ("guidance_scale", options.guidance_scale.is_some()),
            ("num_tokens", options.num_tokens.is_some()),
            ("dense_octree_depth", options.dense_octree_depth.is_some()),
            (
                "hierarchical_octree_depth",
                options.hierarchical_octree_depth.is_some(),
            ),
            ("use_flash_decoder", options.use_flash_decoder.is_some()),
            ("flash_octree_depth", options.flash_octree_depth.is_some()),
            ("query_chunk_size", options.query_chunk_size.is_some()),
        ] {
            ensure!(!present, "`{name}` is not supported for `trellis2`");
        }
        return trellis2_options_from_mesh_options(options).map(|_| ());
    }
    // TripoSG owns the marching-cubes grid through its own octree-depth
    // options, so the TripoSR knobs must stay at their defaults here.
    ensure!(
        options.mc_resolution == 256,
        "`mc_resolution` is not supported for `triposg`; only the default 256 is supported"
    );
    ensure!(
        options.mc_threshold == 25.0,
        "`mc_threshold` is not supported for `triposg`; only the default 25.0 is supported"
    );
    for (name, present) in [
        ("pipeline_type", options.pipeline_type.is_some()),
        ("max_num_tokens", options.max_num_tokens.is_some()),
        ("texture", options.texture.is_some()),
    ] {
        ensure!(!present, "`{name}` is not supported for `triposg`");
    }
    triposg_options_from_mesh_options(options).map(|_| ())
}

pub async fn spawn_mesh_generation(
    state: Arc<Lux3dState>,
    record: JobRecord,
    source: PathBuf,
    options: MeshGenerationOptions,
) -> Result<GenerationObject> {
    let model = record.object.model.clone();
    validate_mesh_generation_request(&model, &options)?;
    let output_format = match options.output_format.as_deref() {
        Some(format) => MeshFormat::from_extension(format).map_err(anyhow::Error::from)?,
        None => MeshFormat::Glb,
    };
    let triposg_run_options = if model == "triposg" {
        Some(triposg_options_from_mesh_options(&options)?)
    } else {
        None
    };
    let trellis2_run_options = if model == "trellis2" {
        Some(trellis2_options_from_mesh_options(&options)?)
    } else {
        None
    };

    let job_id = record.object.id.clone();
    state.jobs.insert(record);
    let response = state
        .jobs
        .get(&job_id)
        .expect("job just inserted")
        .object
        .clone();
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
                let model = model.clone();
                let mc_resolution = options.mc_resolution;
                let mc_threshold = options.mc_threshold;
                move |runtime| {
                    if runtime.jobs.is_cancelled(&job_id_for_blocking) {
                        bail!("generation was cancelled");
                    }
                    runtime
                        .models
                        .ensure_loaded(runtime, &model)
                        .with_context(|| format!("failed to load `{model}` pipeline"))?;
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
                    let output_path =
                        workspace.join(format!("{model}.{}", output_format.extension()));
                    match model.as_str() {
                        "triposr" => runtime.models.with_triposr(|pipeline| {
                            let inference = pipeline
                                .infer_from_path(&source, &device)
                                .map_err(anyhow::Error::from)?;
                            let mesh = pipeline
                                .extract_mesh(
                                    &inference.scene_codes,
                                    mc_resolution,
                                    mc_threshold,
                                    8192,
                                )
                                .map_err(anyhow::Error::from)?;
                            pipeline
                                .export_mesh(&mesh, &output_path)
                                .map_err(anyhow::Error::from)?;
                            Ok(())
                        })?,
                        "triposg" => {
                            let run_options = triposg_run_options
                                .expect("triposg options validated before the job was queued");
                            runtime.models.with_triposg(|pipeline| {
                                let inference = pipeline
                                    .infer_from_path(&source, &device, &run_options)
                                    .map_err(anyhow::Error::from)?;
                                let mesh = pipeline
                                    .extract_mesh(&inference.latents, &run_options)
                                    .map_err(anyhow::Error::from)?;
                                pipeline
                                    .export_mesh(&mesh, &output_path)
                                    .map_err(anyhow::Error::from)?;
                                Ok(())
                            })?
                        }
                        "trellis2" => {
                            let run_options = trellis2_run_options
                                .expect("trellis2 options validated before the job was queued");
                            runtime.models.with_trellis2(|pipeline| {
                                let stages = pipeline
                                    .run_from_path_with_stages(&source, &device, &run_options)
                                    .map_err(anyhow::Error::from)?;
                                let mesh =
                                    stages.mesh.expect("trellis2 run always produces a mesh");
                                if run_options.with_texture
                                    && mesh.voxel_attr_channels == pbrvox::CHANNELS
                                {
                                    // The mesh plus the <stem>.pbrvox.bin /
                                    // <stem>.pbr.json sidecar pair; the sidecars
                                    // stay in the job workspace (only the mesh is
                                    // served through /content).
                                    pipeline
                                        .export_mesh_with_pbr(&mesh, &output_path)
                                        .map_err(anyhow::Error::from)?;
                                } else {
                                    pipeline
                                        .export_mesh(&mesh, &output_path)
                                        .map_err(anyhow::Error::from)?;
                                }
                                Ok(())
                            })?
                        }
                        other => bail!("unsupported mesh model `{other}`"),
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
                if let Err(error) = state_for_task.jobs.mark_completed(&job_id, output_path) {
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
                } else if let Err(update_error) =
                    state_for_task.jobs.mark_failed(&job_id, error.to_string())
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
        pose: values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("pose")),
        depth: values
            .iter()
            .any(|value| value.eq_ignore_ascii_case("depth")),
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

#[cfg(test)]
mod tests {
    use super::{
        MeshGenerationOptions, Trellis2PipelineType, Trellis2RunOptions, TripoSgRunOptions,
        trellis2_options_from_mesh_options, triposg_options_from_mesh_options,
        validate_mesh_generation_request,
    };

    fn default_mesh_options() -> MeshGenerationOptions {
        MeshGenerationOptions::default()
    }

    #[test]
    fn triposg_options_default_to_reference_values() {
        let run = triposg_options_from_mesh_options(&default_mesh_options())
            .expect("default triposg options");
        assert_eq!(TripoSgRunOptions::default(), run);
        assert_eq!(
            (
                50usize, 7.0f32, 42u64, 2048usize, 8u32, 9u32, true, 9u32, 16384usize
            ),
            (
                run.steps,
                run.guidance_scale,
                run.seed,
                run.num_tokens,
                run.dense_octree_depth,
                run.hierarchical_octree_depth,
                run.use_flash_decoder,
                run.flash_octree_depth,
                run.query_chunk_size
            )
        );
    }

    #[test]
    fn triposg_options_map_every_override() {
        let options = MeshGenerationOptions {
            steps: Some(12),
            guidance_scale: Some(3.5),
            seed: Some(7),
            num_tokens: Some(1024),
            dense_octree_depth: Some(5),
            hierarchical_octree_depth: Some(6),
            use_flash_decoder: Some(false),
            flash_octree_depth: Some(7),
            query_chunk_size: Some(4096),
            ..default_mesh_options()
        };
        let run = triposg_options_from_mesh_options(&options).expect("overridden triposg options");
        assert_eq!(12, run.steps);
        assert_eq!(3.5, run.guidance_scale);
        assert_eq!(7, run.seed);
        assert_eq!(1024, run.num_tokens);
        assert_eq!(5, run.dense_octree_depth);
        assert_eq!(6, run.hierarchical_octree_depth);
        assert!(!run.use_flash_decoder);
        assert_eq!(7, run.flash_octree_depth);
        assert_eq!(4096, run.query_chunk_size);
    }

    #[test]
    fn triposg_accepts_default_mesh_request() {
        validate_mesh_generation_request("triposg", &default_mesh_options())
            .expect("default triposg request");
    }

    #[test]
    fn triposg_rejects_non_default_marching_cubes_resolution() {
        let options = MeshGenerationOptions {
            mc_resolution: 128,
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposg", &options)
            .expect_err("mc_resolution=128 must be rejected for triposg");
        let message = error.to_string();
        assert!(
            message.contains("`mc_resolution`"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("is not supported for `triposg`"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn triposg_rejects_non_default_marching_cubes_threshold() {
        let options = MeshGenerationOptions {
            mc_threshold: 10.0,
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposg", &options)
            .expect_err("mc_threshold=10.0 must be rejected for triposg");
        let message = error.to_string();
        assert!(
            message.contains("`mc_threshold`"),
            "unexpected message: `{message}`"
        );
        assert!(
            message.contains("is not supported for `triposg`"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn triposr_accepts_default_mesh_request() {
        validate_mesh_generation_request("triposr", &default_mesh_options())
            .expect("default triposr request");
    }

    #[test]
    fn triposr_rejects_triposg_only_steps() {
        let options = MeshGenerationOptions {
            steps: Some(2),
            ..default_mesh_options()
        };
        let error = validate_mesh_generation_request("triposr", &options)
            .expect_err("steps=Some(2) must be rejected for triposr");
        assert!(
            error
                .to_string()
                .contains("`steps` is not supported for `triposr`"),
            "unexpected message: `{}`",
            error
        );
    }

    #[test]
    fn unsupported_mesh_models_are_rejected() {
        for model in ["pi3", "pi3x", "pi3x_vo", "vggt", "", "triposrg"] {
            let error = validate_mesh_generation_request(model, &default_mesh_options())
                .expect_err("unsupported mesh model must be rejected");
            let expected = format!(
                "unsupported mesh model `{model}`; expected `triposr`, `triposg` or `trellis2`"
            );
            assert!(
                error.to_string().contains(&expected),
                "unexpected message for `{model}`: `{}`",
                error
            );
        }
    }

    #[test]
    fn trellis2_accepts_default_mesh_request() {
        validate_mesh_generation_request("trellis2", &default_mesh_options())
            .expect("default trellis2 request");
    }

    #[test]
    fn trellis2_options_default_to_reference_values() {
        let run = trellis2_options_from_mesh_options(&default_mesh_options())
            .expect("default trellis2 options");
        assert_eq!(Trellis2RunOptions::default(), run);
        assert_eq!(
            (42u64, "1024_cascade", 12usize, true, 49152usize),
            (
                run.seed,
                run.pipeline_type.as_str(),
                run.ss_steps,
                run.with_texture,
                run.max_num_tokens
            )
        );
    }

    #[test]
    fn trellis2_options_map_every_override() {
        let options = MeshGenerationOptions {
            steps: Some(4),
            seed: Some(7),
            pipeline_type: Some("512".to_string()),
            max_num_tokens: Some(8192),
            texture: Some(false),
            ..default_mesh_options()
        };
        let run =
            trellis2_options_from_mesh_options(&options).expect("overridden trellis2 options");
        assert_eq!(7, run.seed);
        assert_eq!(Trellis2PipelineType::R512, run.pipeline_type);
        assert_eq!(4, run.ss_steps);
        assert_eq!(4, run.shape_steps);
        assert_eq!(4, run.tex_steps);
        assert!(!run.with_texture);
        assert_eq!(8192, run.max_num_tokens);

        let cascade = trellis2_options_from_mesh_options(&MeshGenerationOptions {
            pipeline_type: Some("1024_cascade".to_string()),
            ..default_mesh_options()
        })
        .expect("cascade options");
        assert_eq!(Trellis2PipelineType::Cascade1024, cascade.pipeline_type);

        let single_1024 = trellis2_options_from_mesh_options(&MeshGenerationOptions {
            pipeline_type: Some("1024".to_string()),
            ..default_mesh_options()
        })
        .expect("single-resolution options parse (runtime rejects it at run time)");
        assert_eq!(Trellis2PipelineType::R1024, single_1024.pipeline_type);
    }

    #[test]
    fn trellis2_rejects_unknown_pipeline_type() {
        let options = MeshGenerationOptions {
            pipeline_type: Some("1536_cascade".to_string()),
            ..default_mesh_options()
        };
        let error = trellis2_options_from_mesh_options(&options)
            .expect_err("unknown pipeline_type must be rejected");
        let message = error.to_string();
        assert!(
            message.contains("unknown trellis2 `pipeline_type` `1536_cascade`"),
            "unexpected message: `{message}`"
        );
    }

    #[test]
    fn trellis2_rejects_foreign_mesh_options() {
        let cases: [(&str, MeshGenerationOptions); 7] = [
            (
                "mc_resolution",
                MeshGenerationOptions {
                    mc_resolution: 128,
                    ..default_mesh_options()
                },
            ),
            (
                "mc_threshold",
                MeshGenerationOptions {
                    mc_threshold: 10.0,
                    ..default_mesh_options()
                },
            ),
            (
                "guidance_scale",
                MeshGenerationOptions {
                    guidance_scale: Some(3.5),
                    ..default_mesh_options()
                },
            ),
            (
                "num_tokens",
                MeshGenerationOptions {
                    num_tokens: Some(1024),
                    ..default_mesh_options()
                },
            ),
            (
                "use_flash_decoder",
                MeshGenerationOptions {
                    use_flash_decoder: Some(true),
                    ..default_mesh_options()
                },
            ),
            (
                "flash_octree_depth",
                MeshGenerationOptions {
                    flash_octree_depth: Some(9),
                    ..default_mesh_options()
                },
            ),
            (
                "query_chunk_size",
                MeshGenerationOptions {
                    query_chunk_size: Some(4096),
                    ..default_mesh_options()
                },
            ),
        ];
        for (name, options) in cases {
            let error = validate_mesh_generation_request("trellis2", &options)
                .expect_err("foreign option must be rejected for trellis2");
            let message = error.to_string();
            assert!(
                message.contains(&format!("`{name}` is not supported for `trellis2`")),
                "unexpected message for {name}: `{message}`"
            );
        }
    }

    #[test]
    fn triposg_and_triposr_reject_trellis2_only_options() {
        for model in ["triposg", "triposr"] {
            for (name, options) in [
                (
                    "pipeline_type",
                    MeshGenerationOptions {
                        pipeline_type: Some("512".to_string()),
                        ..default_mesh_options()
                    },
                ),
                (
                    "max_num_tokens",
                    MeshGenerationOptions {
                        max_num_tokens: Some(8192),
                        ..default_mesh_options()
                    },
                ),
                (
                    "texture",
                    MeshGenerationOptions {
                        texture: Some(false),
                        ..default_mesh_options()
                    },
                ),
            ] {
                let error = validate_mesh_generation_request(model, &options)
                    .expect_err("trellis2-only option must be rejected");
                let message = error.to_string();
                assert!(
                    message.contains(&format!("`{name}` is not supported for `{model}`")),
                    "unexpected message for {model}/{name}: `{message}`"
                );
            }
        }
    }
}
