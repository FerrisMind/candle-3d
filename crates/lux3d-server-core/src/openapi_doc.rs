//! OpenAPI document generation for the Lux3D server.

use utoipa::OpenApi;

use crate::{
    api_types::{
        ErrorBody, ErrorResponse, GenerationError, GenerationListResponse, GenerationObject,
        MeshGenerationOptions, MeshGenerationRequest, ModelListResponse, ModelObject,
        PointCloudGenerationOptions, PointCloudGenerationRequest, RouteSummary,
        ServerInfoResponse,
    },
    errors::{ApiErrorBody, ApiErrorResponse},
    files::{FileListResponse, FileObject},
    handlers,
    models::{ModelOperationRequest, ModelStatusResponse},
    system::{DoctorCheck, DoctorReport, SystemInfoResponse},
};

#[derive(OpenApi)]
#[openapi(
    paths(
        handlers::root,
        handlers::health,
        handlers::models,
        handlers::reload_model,
        handlers::unload_model,
        handlers::model_status,
        handlers::list_generations,
        handlers::get_generation,
        handlers::cancel_generation,
        handlers::delete_generation,
        handlers::get_generation_content,
        handlers::create_point_cloud_generation,
        handlers::create_mesh_generation,
        crate::files::upload_file,
        crate::files::list_files,
        crate::files::get_file,
        crate::files::get_file_content,
        crate::files::delete_file,
        crate::stream::stream_generation,
        crate::system::system_info,
        crate::system::system_doctor,
    ),
    components(schemas(
        ServerInfoResponse,
        RouteSummary,
        ModelListResponse,
        ModelObject,
        ModelOperationRequest,
        ModelStatusResponse,
        GenerationListResponse,
        GenerationObject,
        GenerationError,
        PointCloudGenerationOptions,
        PointCloudGenerationRequest,
        MeshGenerationOptions,
        MeshGenerationRequest,
        FileListResponse,
        FileObject,
        SystemInfoResponse,
        DoctorReport,
        DoctorCheck,
        ErrorResponse,
        ErrorBody,
        ApiErrorResponse,
        ApiErrorBody,
    )),
    tags(
        (name = "Lux3D", description = "OpenAI-style 3D generation server endpoints")
    ),
    info(
        title = "Lux3D Server API",
        version = "0.1.0",
        description = "Async HTTP API for Pi3/Pi3X point clouds and TripoSR meshes."
    )
)]
struct ApiDoc;

pub fn get_openapi_doc(base_path: Option<&str>) -> utoipa::openapi::OpenApi {
    let mut doc = ApiDoc::openapi();
    if let Some(prefix) = base_path.filter(|path| !path.is_empty()) {
        doc.servers = Some(vec![utoipa::openapi::Server::new(prefix.to_string())]);
    }
    doc
}
