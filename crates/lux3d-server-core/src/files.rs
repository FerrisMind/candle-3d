//! OpenAI-style Files API backed by disk storage under the upload directory.

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, ensure};
use axum::{
    Json,
    extract::{Multipart, Path as AxumPath, Query, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;

use crate::{
    config::DEFAULT_MAX_FILE_BYTES,
    errors::{internal_error, invalid_request, not_found},
    types::ExtractedLux3dState,
};

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FileObject {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    pub mime_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct FileListResponse {
    pub object: &'static str,
    pub data: Vec<FileObject>,
}

#[derive(Debug, Deserialize)]
pub struct FileListQuery {
    #[serde(default = "default_file_list_limit")]
    pub limit: usize,
}

fn default_file_list_limit() -> usize {
    20
}

#[derive(Debug, Clone)]
pub(crate) struct FileRecord {
    metadata: FileObject,
    path: PathBuf,
}

#[derive(Debug, Default, Clone)]
pub struct FileStore {
    inner: Arc<RwLock<HashMap<String, FileRecord>>>,
}

impl FileStore {
    pub(crate) fn insert(&self, record: FileRecord) {
        if let Ok(mut files) = self.inner.write() {
            files.insert(record.metadata.id.clone(), record);
        }
    }

    pub fn get(&self, id: &str) -> Option<FileObject> {
        self.inner
            .read()
            .ok()
            .and_then(|files| files.get(id).map(|record| record.metadata.clone()))
    }

    pub fn get_path(&self, id: &str) -> Option<PathBuf> {
        self.inner
            .read()
            .ok()
            .and_then(|files| files.get(id).map(|record| record.path.clone()))
    }

    pub fn list(&self, limit: usize) -> Vec<FileObject> {
        let Ok(files) = self.inner.read() else {
            return Vec::new();
        };
        let mut entries: Vec<_> = files.values().map(|record| record.metadata.clone()).collect();
        entries.sort_by_key(|entry| std::cmp::Reverse(entry.created_at));
        entries.truncate(limit);
        entries
    }

    pub(crate) fn remove(&self, id: &str) -> Option<FileRecord> {
        let mut files = self.inner.write().ok()?;
        files.remove(id)
    }

    pub fn cleanup_expired(&self, now: u64) -> usize {
        let expired: Vec<String> = self
            .inner
            .read()
            .ok()
            .map(|files| {
                files
                    .iter()
                    .filter_map(|(id, record)| {
                        record
                            .metadata
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
                let _ = fs::remove_file(&record.path);
                removed += 1;
            }
        }
        removed
    }
}

pub fn new_file_id() -> String {
    format!("file_{}", Uuid::new_v4().simple())
}

pub fn files_dir(upload_dir: &Path) -> PathBuf {
    upload_dir.join("files")
}

pub async fn store_upload(
    store: &FileStore,
    upload_dir: &Path,
    purpose: &str,
    filename: &str,
    mime_type: &str,
    bytes: &[u8],
    ttl_secs: u64,
) -> Result<FileObject> {
    ensure!(
        bytes.len() <= DEFAULT_MAX_FILE_BYTES,
        "file exceeds maximum size of {DEFAULT_MAX_FILE_BYTES} bytes"
    );

    let file_id = new_file_id();
    let directory = files_dir(upload_dir);
    fs::create_dir_all(&directory).with_context(|| {
        format!(
            "failed to create files directory at `{}`",
            directory.display()
        )
    })?;

    let sanitized = sanitize_filename(filename);
    let path = directory.join(format!("{file_id}-{sanitized}"));
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write uploaded file to `{}`", path.display()))?;

    let created_at = now_unix();
    let metadata = FileObject {
        id: file_id.clone(),
        object: "file",
        bytes: bytes.len() as u64,
        created_at,
        filename: sanitized,
        purpose: purpose.to_string(),
        mime_type: mime_type.to_string(),
        expires_at: Some(created_at.saturating_add(ttl_secs)),
    };

    store.insert(FileRecord {
        metadata: metadata.clone(),
        path,
    });
    Ok(metadata)
}

#[utoipa::path(
    post,
    tag = "Lux3D",
    path = "/v1/files",
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Uploaded file metadata", body = FileObject),
        (status = 400, description = "Invalid upload")
    )
)]
pub async fn upload_file(
    State(state): ExtractedLux3dState,
    multipart: Multipart,
) -> Response {
    match parse_file_upload(multipart).await {
        Ok(upload) => match store_upload(
            &state.files,
            &state.upload_dir,
            &upload.purpose,
            &upload.filename,
            upload.mime_type.as_deref().unwrap_or("application/octet-stream"),
            &upload.bytes,
            state.generation_defaults.job_ttl_secs,
        )
        .await
        {
            Ok(file) => (StatusCode::OK, Json(file)).into_response(),
            Err(error) => invalid_request(error.to_string()),
        },
        Err(error) => invalid_request(error.to_string()),
    }
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/files",
    responses((status = 200, description = "Recent uploaded files", body = FileListResponse))
)]
pub async fn list_files(
    State(state): ExtractedLux3dState,
    Query(query): Query<FileListQuery>,
) -> Json<FileListResponse> {
    Json(FileListResponse {
        object: "list",
        data: state.files.list(query.limit),
    })
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/files/{id}",
    responses(
        (status = 200, description = "File metadata", body = FileObject),
        (status = 404, description = "File not found")
    )
)]
pub async fn get_file(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    match state.files.get(&id) {
        Some(file) => (StatusCode::OK, Json(file)).into_response(),
        None => not_found(format!("file `{id}` not found")),
    }
}

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/files/{id}/content",
    responses(
        (status = 200, description = "File bytes"),
        (status = 404, description = "File not found")
    )
)]
pub async fn get_file_content(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(path) = state.files.get_path(&id) else {
        return not_found(format!("file `{id}` not found"));
    };
    let Some(metadata) = state.files.get(&id) else {
        return not_found(format!("file `{id}` not found"));
    };

    match fs::read(&path) {
        Ok(bytes) => (
            StatusCode::OK,
            [
                (header::CONTENT_TYPE, metadata.mime_type.as_str()),
                (
                    header::CONTENT_DISPOSITION,
                    &format!("attachment; filename=\"{}\"", metadata.filename),
                ),
            ],
            bytes,
        )
            .into_response(),
        Err(error) => internal_error(format!(
            "failed to read file `{}`: {error}",
            path.display()
        )),
    }
}

#[utoipa::path(
    delete,
    tag = "Lux3D",
    path = "/v1/files/{id}",
    responses(
        (status = 200, description = "Deleted file metadata", body = FileObject),
        (status = 404, description = "File not found")
    )
)]
pub async fn delete_file(
    State(state): ExtractedLux3dState,
    AxumPath(id): AxumPath<String>,
) -> Response {
    let Some(record) = state.files.remove(&id) else {
        return not_found(format!("file `{id}` not found"));
    };
    if let Err(error) = fs::remove_file(&record.path) {
        tracing::warn!(
            path = %record.path.display(),
            %error,
            "failed to delete file from disk"
        );
    }
    (StatusCode::OK, Json(record.metadata)).into_response()
}

struct ParsedFileUpload {
    purpose: String,
    filename: String,
    mime_type: Option<String>,
    bytes: Vec<u8>,
}

async fn parse_file_upload(mut multipart: Multipart) -> Result<ParsedFileUpload> {
    let mut purpose = None;
    let mut file = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .context("failed to read multipart field")?
    {
        match field.name().unwrap_or_default() {
            "purpose" => {
                purpose = Some(field.text().await.context("failed to read purpose field")?);
            }
            "file" => {
                let filename = field
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("uploaded file is missing a filename"))?
                    .to_string();
                let mime_type = field.content_type().map(ToString::to_string);
                let bytes = field
                    .bytes()
                    .await
                    .context("failed to read uploaded file bytes")?
                    .to_vec();
                file = Some((filename, mime_type, bytes));
            }
            _ => {}
        }
    }

    let purpose = purpose
        .filter(|value| !value.trim().is_empty())
        .context("missing required multipart field `purpose`")?;
    let (filename, mime_type, bytes) =
        file.context("missing required multipart field `file`")?;

    Ok(ParsedFileUpload {
        purpose,
        filename,
        mime_type,
        bytes,
    })
}

fn sanitize_filename(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        format!("upload-{}", Uuid::new_v4())
    } else {
        sanitized
    }
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub fn resolve_file_path(state: &crate::types::Lux3dState, file_id: &str) -> Result<PathBuf> {
    state
        .files
        .get_path(file_id)
        .with_context(|| format!("file `{file_id}` not found"))
}
