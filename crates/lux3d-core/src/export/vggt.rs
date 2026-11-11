use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, VggtPointCloud},
    error::Lux3dError,
    geometry::VggtPointCloudCpu,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VggtExportStage;

impl ExportStage<VggtPointCloud> for VggtExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &VggtPointCloud) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "ply".to_string(),
            alternate_extensions: Vec::new(),
            utility_notes: vec![
                "world_points_conf > 1.0".to_string(),
                "colors from preprocessed RGB frames".to_string(),
                "write_ply(points, rgb)".to_string(),
            ],
        })
    }
}

impl VggtExportStage {
    pub fn write_ply(&self, point_cloud: &VggtPointCloudCpu, path: &Path) -> Result<()> {
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
