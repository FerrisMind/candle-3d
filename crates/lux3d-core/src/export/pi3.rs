use std::path::Path;

use crate::{
    Result,
    contracts::{ExportPlan, ExportStage, Pi3PointCloud},
    geometry::Pi3PointCloudCpu,
};

use super::point_cloud::{PointCloudFormat, PointCloudView, write_ply, write_point_cloud};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pi3ExportStage;

impl ExportStage<Pi3PointCloud> for Pi3ExportStage {
    type Output = ExportPlan;

    fn export(&self, _artifact: &Pi3PointCloud) -> Result<Self::Output> {
        Ok(ExportPlan {
            primary_extension: "ply".to_string(),
            alternate_extensions: vec!["xyz".to_string()],
            utility_notes: vec![
                "sigmoid(confidence_logits) > 0.1".to_string(),
                "depth_edge(local_points[..., 2])".to_string(),
                "write_ply(points, rgb); .xyz is a plain-text alternative".to_string(),
            ],
        })
    }
}

impl Pi3ExportStage {
    /// Write in the format selected by `path`'s extension (`.ply` or `.xyz`).
    pub fn write_point_cloud(
        &self,
        point_cloud: &Pi3PointCloudCpu,
        path: &Path,
    ) -> Result<PointCloudFormat> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_point_cloud(&view, path)
    }

    /// Write an ASCII PLY point cloud, preserving the historical output.
    pub fn write_ply(&self, point_cloud: &Pi3PointCloudCpu, path: &Path) -> Result<()> {
        let view = PointCloudView {
            points: &point_cloud.points,
            colors: &point_cloud.colors,
        };
        write_ply(&view, path)
    }
}
