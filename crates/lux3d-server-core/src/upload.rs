//! Upload helpers and multipart parsing utilities.

use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use axum::extract::multipart::{Field, Multipart};
use serde::de::DeserializeOwned;
use uuid::Uuid;

pub struct PreparedUpload {
    pub source: PathBuf,
    pub conditions: Option<PathBuf>,
    pub model: Option<String>,
}

pub async fn parse_generation_upload<T>(
    mut multipart: Multipart,
    workspace: &Path,
) -> Result<(PreparedUpload, T)>
where
    T: DeserializeOwned + Default,
{
    fs::create_dir_all(workspace).with_context(|| {
        format!(
            "failed to create generation workspace at `{}`",
            workspace.display()
        )
    })?;

    let mut source_path = None;
    let mut conditions_path = None;
    let mut model = None;
    let mut options_json = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .context("failed to read multipart field")?
    {
        match field.name().unwrap_or_default() {
            "source" => {
                source_path =
                    Some(save_upload_field(field, workspace, "source").await?);
            }
            "conditions" => {
                conditions_path =
                    Some(save_upload_field(field, workspace, "conditions").await?);
            }
            "model" => {
                model = Some(field.text().await.context("failed to read model field")?);
            }
            "options" => {
                options_json = Some(field.text().await.context("failed to read options field")?);
            }
            _ => {}
        }
    }

    let source_path = source_path.context("missing required multipart field `source`")?;
    let options = match options_json {
        Some(json) => serde_json::from_str(&json).context("failed to parse `options` JSON field")?,
        None => T::default(),
    };

    Ok((
        PreparedUpload {
            source: source_path,
            conditions: conditions_path,
            model,
        },
        options,
    ))
}

pub fn normalize_source_for_frames(source: &Path, workspace: &Path) -> Result<PathBuf> {
    let extension = source
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("zip") => extract_zip(source, workspace.join("frames")),
        Some("png" | "jpg" | "jpeg" | "webp") => {
            let frames_dir = workspace.join("frames");
            fs::create_dir_all(&frames_dir)?;
            let target = frames_dir.join(source.file_name().unwrap_or(source.as_os_str()));
            fs::copy(source, &target).with_context(|| {
                format!(
                    "failed to copy uploaded image into frame directory `{}`",
                    frames_dir.display()
                )
            })?;
            Ok(frames_dir)
        }
        Some("mp4" | "mov" | "avi" | "mkv") => Ok(source.to_path_buf()),
        _ if source.is_dir() => Ok(source.to_path_buf()),
        _ => bail!(
            "unsupported source upload `{}`; expected frame directory, .zip archive, image, or video file",
            source.display()
        ),
    }
}

pub fn normalize_source_for_image(source: &Path) -> Result<PathBuf> {
    let extension = source
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase);

    match extension.as_deref() {
        Some("png" | "jpg" | "jpeg" | "webp") => Ok(source.to_path_buf()),
        _ => bail!(
            "unsupported mesh source `{}`; expected a single image file",
            source.display()
        ),
    }
}

async fn save_upload_field(
    mut field: Field<'_>,
    workspace: &Path,
    prefix: &str,
) -> Result<PathBuf> {
    let original_name = field.file_name().unwrap_or(prefix).to_string();
    let sanitized = sanitize_filename(&original_name);
    let path = workspace.join(format!("{prefix}-{sanitized}"));
    let mut file = fs::File::create(&path).with_context(|| {
        format!("failed to create upload file at `{}`", path.display())
    })?;
    while let Some(chunk) = field.chunk().await.context("failed to read upload chunk")? {
        file.write_all(&chunk)
            .with_context(|| format!("failed to write upload chunk to `{}`", path.display()))?;
    }
    Ok(path)
}

fn extract_zip(source: &Path, destination: PathBuf) -> Result<PathBuf> {
    fs::create_dir_all(&destination).with_context(|| {
        format!(
            "failed to create extraction directory at `{}`",
            destination.display()
        )
    })?;

    let file = fs::File::open(source)
        .with_context(|| format!("failed to open zip archive at `{}`", source.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read zip archive at `{}`", source.display()))?;

    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .with_context(|| format!("failed to read zip entry {index}"))?;
        if entry.is_dir() {
            continue;
        }

        let Some(relative) = entry.enclosed_name() else {
            continue;
        };
        let out_path = destination.join(relative);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut out_file = fs::File::create(&out_path).with_context(|| {
            format!("failed to create extracted file at `{}`", out_path.display())
        })?;
        let mut buffer = Vec::new();
        entry
            .read_to_end(&mut buffer)
            .with_context(|| format!("failed to read zip entry at `{}`", out_path.display()))?;
        out_file
            .write_all(&buffer)
            .with_context(|| format!("failed to write extracted file at `{}`", out_path.display()))?;
    }

    Ok(destination)
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
