//! OpenAI-style HTTP API types for Lux3D generation endpoints.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    Queued,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl GenerationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationError {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationObject {
    pub id: String,
    pub object: &'static str,
    pub status: String,
    pub model: String,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<u64>,
    pub progress: u8,
    pub output_format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<GenerationError>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct GenerationListResponse {
    pub object: &'static str,
    pub data: Vec<GenerationObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelObject {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    pub kind: String,
    pub output_format: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelListResponse {
    pub object: &'static str,
    pub data: Vec<ModelObject>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ServerInfoResponse {
    pub object: &'static str,
    pub name: String,
    pub version: String,
    pub routes: Vec<RouteSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct RouteSummary {
    pub path: String,
    pub methods: String,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, ToSchema)]
pub struct PointCloudGenerationOptions {
    pub interval: Option<usize>,
    pub vo: Option<bool>,
    pub chunk_size: Option<usize>,
    pub overlap: Option<usize>,
    pub conf_threshold: Option<f32>,
    #[serde(default)]
    pub inject_condition: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MeshGenerationOptions {
    #[serde(default = "default_mc_resolution")]
    pub mc_resolution: u32,
    #[serde(default = "default_mc_threshold")]
    pub mc_threshold: f32,
}

fn default_mc_resolution() -> u32 {
    256
}

fn default_mc_threshold() -> f32 {
    25.0
}

impl Default for MeshGenerationOptions {
    fn default() -> Self {
        Self {
            mc_resolution: default_mc_resolution(),
            mc_threshold: default_mc_threshold(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct PointCloudGenerationRequest {
    pub model: String,
    #[serde(default)]
    pub options: PointCloudGenerationOptions,
    #[serde(default)]
    pub webhook_url: Option<String>,
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
    pub conditions_file_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, ToSchema)]
pub struct MeshGenerationRequest {
    #[serde(default = "default_mesh_model")]
    pub model: String,
    #[serde(default)]
    pub options: MeshGenerationOptions,
    #[serde(default)]
    pub webhook_url: Option<String>,
    pub source_url: Option<String>,
    pub source_file_id: Option<String>,
    pub source_base64: Option<String>,
    pub source_filename: Option<String>,
}

fn default_mesh_model() -> String {
    "triposr".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ErrorBody {
    pub message: String,
}
