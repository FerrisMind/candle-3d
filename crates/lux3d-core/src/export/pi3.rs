use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, Pi3PointCloud},
    error::Lux3dError,
    geometry::Pi3PointCloudCpu,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3ExportStage;

impl ExportStage<Pi3PointCloud> for Pi3ExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &Pi3PointCloud) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "ply".to_string(),
            alternate_extensions: Vec::new(),
            utility_notes: vec![
                "sigmoid(confidence_logits) > 0.1".to_string(),
                "depth_edge(local_points[..., 2])".to_string(),
                "write_ply(points, rgb)".to_string(),
            ],
        })
    }
}

impl Pi3ExportStage {
    pub fn write_ply(&self, point_cloud: &Pi3PointCloudCpu, path: &Path) -> Result<()> {
        let file = File::create(path).map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "ply").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(writer, "format ascii 1.0").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(writer, "element vertex {}", point_cloud.vertex_count).map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property float x").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(writer, "property float y").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(writer, "property float z").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;
        writeln!(writer, "property float nx").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property float ny").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property float nz").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property uchar red").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property uchar green").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "property uchar blue").map_err(|source| {
            Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            }
        })?;
        writeln!(writer, "end_header").map_err(|source| Lux3dError::CanonicalManifestIo {
            path: path.to_path_buf(),
            source,
        })?;

        for index in 0..point_cloud.vertex_count {
            let offset = index * 3;
            writeln!(
                writer,
                "{} {} {} {} {} {} {} {} {}",
                point_cloud.points[offset],
                point_cloud.points[offset + 1],
                point_cloud.points[offset + 2],
                0.0,
                0.0,
                0.0,
                point_cloud.colors[offset],
                point_cloud.colors[offset + 1],
                point_cloud.colors[offset + 2]
            )
            .map_err(|source| Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            })?;
        }
        writer
            .flush()
            .map_err(|source| Lux3dError::CanonicalManifestIo {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(())
    }
}
