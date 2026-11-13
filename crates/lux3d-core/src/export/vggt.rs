use std::path::Path;

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, VggtPointCloud},
    geometry::VggtPointCloudCpu,
};

use super::point_cloud::{PointCloudFormat, PointCloudView, write_ply, write_point_cloud};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VggtExportStage;

impl ExportStage<VggtPointCloud> for VggtExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &VggtPointCloud) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "ply".to_string(),
            alternate_extensions: vec!["xyz".to_string()],
            utility_notes: vec![
                "world_points_conf > 1.0".to_string(),
                "colors from preprocessed RGB frames".to_string(),
                "write_ply(points, rgb); .xyz is a plain-text alternative".to_string(),
            ],
        })
    }
}

impl VggtExportStage {
    /// Write in the format selected by `path`'s extension (`.ply` or `.xyz`).
    pub fn write_point_cloud(
        &self,
        point_cloud: &VggtPointCloudCpu,
        path: &Path,
    ) -> Result<PointCloudFormat> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_point_cloud(&view, path)
    }

    /// Write an ASCII PLY point cloud, preserving the historical output.
    pub fn write_ply(&self, point_cloud: &VggtPointCloudCpu, path: &Path) -> Result<()> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_ply(&view, path)
    }
}
