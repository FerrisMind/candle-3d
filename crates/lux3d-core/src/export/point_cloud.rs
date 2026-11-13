//! Shared point-cloud export writers.
//!
//! Pi3/Pi3X/VGGT predict colored point clouds, not triangle meshes. PLY stays
//! the primary format because reconstruction tools consume it directly; XYZ is
//! a small plain-text interchange alternative.

use std::{fmt::Write as _, path::Path};

use crate::{Result, error::Lux3dError};

/// Supported point-cloud artifact formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointCloudFormat {
    /// ASCII PLY with positions, zero normals and RGB colors.
    Ply,
    /// Plain-text `x y z r g b` rows.
    Xyz,
}

impl PointCloudFormat {
    /// Resolve a point-cloud format from a case-insensitive file extension.
    pub fn from_path(path: &Path) -> Result<Self> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .unwrap_or_default();
        Self::from_extension(extension)
    }

    /// Resolve a point-cloud format from an extension or format name.
    pub fn from_extension(extension: &str) -> Result<Self> {
        match extension
            .trim_start_matches('.')
            .to_ascii_lowercase()
            .as_str()
        {
            "ply" => Ok(Self::Ply),
            "xyz" => Ok(Self::Xyz),
            other => Err(Lux3dError::CanonicalWeightsValidation {
                message: format!(
                    "unsupported point-cloud output format `{other}`; use `ply` or `xyz`"
                ),
            }),
        }
    }

    /// Canonical lowercase extension.
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Ply => "ply",
            Self::Xyz => "xyz",
        }
    }

    /// Content type used for HTTP downloads.
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Ply => "model/ply",
            Self::Xyz => "text/plain; charset=utf-8",
        }
    }
}

/// Borrowed CPU point-cloud buffers.
#[derive(Debug, Clone, Copy)]
pub struct PointCloudView<'a> {
    /// Flat `[x, y, z, ...]` positions.
    pub points: &'a [f32],
    /// Flat `[r, g, b, ...]` colors in `[0, 255]`.
    pub colors: &'a [u8],
}

impl PointCloudView<'_> {
    fn vertex_count(&self) -> usize {
        self.points.len() / 3
    }

    fn validate(&self) -> Result<()> {
        if self.points.len() % 3 != 0 {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "point-cloud position buffer length must be a multiple of 3".to_string(),
            });
        }
        if self.colors.len() != self.points.len() {
            return Err(Lux3dError::CanonicalWeightsValidation {
                message: "point-cloud color buffer must match the position buffer length"
                    .to_string(),
            });
        }
        for value in self.points {
            if !value.is_finite() {
                return Err(Lux3dError::CanonicalWeightsValidation {
                    message: "point-cloud export rejected a non-finite point".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Write a point cloud, selecting the format from `path`'s extension.
pub fn write_point_cloud(
    point_cloud: &PointCloudView<'_>,
    path: &Path,
) -> Result<PointCloudFormat> {
    let format = PointCloudFormat::from_path(path)?;
    match format {
        PointCloudFormat::Ply => write_ply(point_cloud, path)?,
        PointCloudFormat::Xyz => write_xyz(point_cloud, path)?,
    }
    Ok(format)
}

/// Write an ASCII PLY point cloud with positions, zero normals and RGB.
pub fn write_ply(point_cloud: &PointCloudView<'_>, path: &Path) -> Result<()> {
    point_cloud.validate()?;
    let vertex_count = point_cloud.vertex_count();
    let mut body = String::new();
    writeln!(body, "ply").expect("writing to String cannot fail");
    writeln!(body, "format ascii 1.0").expect("writing to String cannot fail");
    writeln!(body, "element vertex {vertex_count}").expect("writing to String cannot fail");
    writeln!(body, "property float x").expect("writing to String cannot fail");
    writeln!(body, "property float y").expect("writing to String cannot fail");
    writeln!(body, "property float z").expect("writing to String cannot fail");
    writeln!(body, "property float nx").expect("writing to String cannot fail");
    writeln!(body, "property float ny").expect("writing to String cannot fail");
    writeln!(body, "property float nz").expect("writing to String cannot fail");
    writeln!(body, "property uchar red").expect("writing to String cannot fail");
    writeln!(body, "property uchar green").expect("writing to String cannot fail");
    writeln!(body, "property uchar blue").expect("writing to String cannot fail");
    writeln!(body, "end_header").expect("writing to String cannot fail");
    for index in 0..vertex_count {
        let offset = index * 3;
        writeln!(
            body,
            "{} {} {} 0 0 0 {} {} {}",
            point_cloud.points[offset],
            point_cloud.points[offset + 1],
            point_cloud.points[offset + 2],
            point_cloud.colors[offset],
            point_cloud.colors[offset + 1],
            point_cloud.colors[offset + 2],
        )
        .expect("writing to String cannot fail");
    }
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

/// Write plain-text `x y z r g b` rows.
pub fn write_xyz(point_cloud: &PointCloudView<'_>, path: &Path) -> Result<()> {
    point_cloud.validate()?;
    let vertex_count = point_cloud.vertex_count();
    let mut body = String::new();
    for index in 0..vertex_count {
        let offset = index * 3;
        writeln!(
            body,
            "{} {} {} {} {} {}",
            point_cloud.points[offset],
            point_cloud.points[offset + 1],
            point_cloud.points[offset + 2],
            point_cloud.colors[offset],
            point_cloud.colors[offset + 1],
            point_cloud.colors[offset + 2],
        )
        .expect("writing to String cannot fail");
    }
    std::fs::write(path, body).map_err(|source| Lux3dError::ModelAssetIo {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(extension: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "lux3d-point-cloud-export-{}-{nonce}.{extension}",
            std::process::id()
        ))
    }

    #[test]
    fn ply_and_xyz_writers_preserve_positions_and_colors() {
        let points = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let colors = [255, 0, 128, 0, 255, 64];
        let cloud = PointCloudView {
            points: &points,
            colors: &colors,
        };

        let ply_path = temp_path("ply");
        write_point_cloud(&cloud, &ply_path).expect("write ply");
        let ply = std::fs::read_to_string(&ply_path).expect("read ply");
        let _ = std::fs::remove_file(&ply_path);
        assert!(ply.starts_with("ply\nformat ascii 1.0\n"));
        assert!(ply.contains("element vertex 2"));
        assert!(ply.contains("1 2 3 0 0 0 255 0 128\n"));

        let xyz_path = temp_path("xyz");
        write_point_cloud(&cloud, &xyz_path).expect("write xyz");
        let xyz = std::fs::read_to_string(&xyz_path).expect("read xyz");
        let _ = std::fs::remove_file(&xyz_path);
        assert_eq!(xyz, "1 2 3 255 0 128\n4 5 6 0 255 64\n");
    }
}
