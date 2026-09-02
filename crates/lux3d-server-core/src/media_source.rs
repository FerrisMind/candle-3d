//! Resolve generation sources from URLs, file IDs, base64 payloads, or local paths.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine, engine::general_purpose::STANDARD};
use url::Url;

use crate::{
    files::resolve_file_path,
    types::Lux3dState,
};

#[derive(Debug, Clone, Default)]
pub struct MediaSourceInput {
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
    pub conditions_file_id: Option<String>,
}

pub async fn resolve_source_path(
    state: &Lux3dState,
    workspace: &Path,
    input: &MediaSourceInput,
) -> Result<PathBuf> {
    if let Some(url) = input.source_url.as_deref() {
        return fetch_url_to_workspace(state, workspace, url, input.source_filename.as_deref())
            .await;
    }
    if let Some(file_id) = input.source_file_id.as_deref() {
        let source_path = resolve_file_path(state, file_id)?;
        return copy_file_to_workspace(workspace, &source_path);
    }
    if let Some(encoded) = input.source_base64.as_deref() {
        return write_base64_to_workspace(workspace, encoded, input.source_filename.as_deref());
    }
    bail!("one of `source`, `source_url`, `source_file_id`, or `source_base64` is required")
}

pub fn resolve_conditions_path(
    state: &Lux3dState,
    file_id: Option<&str>,
) -> Result<Option<PathBuf>> {
    match file_id {
        Some(id) => resolve_file_path(state, id).map(Some),
        None => Ok(None),
    }
}

async fn fetch_url_to_workspace(
    state: &Lux3dState,
    workspace: &Path,
    url: &str,
    filename: Option<&str>,
) -> Result<PathBuf> {
    ensure!(is_allowed_url(url)?, "source URL is not allowed");

    let response = state
        .http_client
        .get(url)
        .send()
        .await
        .with_context(|| format!("failed to fetch source URL `{url}`"))?
        .error_for_status()
        .with_context(|| format!("source URL `{url}` returned an error status"))?;

    let bytes = response
        .bytes()
        .await
        .with_context(|| format!("failed to read response body from `{url}`"))?;

    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;

    let filename = filename
        .map(ToString::to_string)
        .or_else(|| infer_filename_from_url(url))
        .unwrap_or_else(|| "source.bin".to_string());
    let path = workspace.join(sanitize_filename(&filename));
    fs::write(&path, &bytes)
        .with_context(|| format!("failed to write downloaded source to `{}`", path.display()))?;
    Ok(path)
}

fn copy_file_to_workspace(workspace: &Path, source: &Path) -> Result<PathBuf> {
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;
    let filename = source
        .file_name()
        .and_then(|name| name.to_str())
        .map(sanitize_filename)
        .unwrap_or_else(|| "source.bin".to_string());
    let path = workspace.join(filename);
    fs::copy(source, &path).with_context(|| {
        format!(
            "failed to copy source `{}` into workspace at `{}`",
            source.display(),
            path.display()
        )
    })?;
    Ok(path)
}

fn write_base64_to_workspace(
    workspace: &Path,
    encoded: &str,
    filename: Option<&str>,
) -> Result<PathBuf> {
    let bytes = STANDARD
        .decode(encoded.trim())
        .context("failed to decode source_base64 payload")?;
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;
    let filename = filename.unwrap_or("source.bin");
    let path = workspace.join(sanitize_filename(filename));
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write base64 source to `{}`", path.display()))?;
    Ok(path)
}

fn infer_filename_from_url(url: &str) -> Option<String> {
    Url::parse(url)
        .ok()
        .and_then(|parsed| {
            parsed
                .path_segments()
                .and_then(|mut segments| segments.next_back().map(str::to_string))
        })
        .filter(|name| !name.is_empty())
}

fn is_allowed_url(url: &str) -> Result<bool> {
    let parsed = Url::parse(url).context("invalid source URL")?;
    match parsed.scheme() {
        "http" | "https" => Ok(true),
        _ => Ok(false),
    }
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
        "source.bin".to_string()
    } else {
        sanitized
    }
}
