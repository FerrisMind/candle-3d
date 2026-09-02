#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RouteInfo {
    pub path: &'static str,
    pub methods: &'static str,
}

impl RouteInfo {
    pub const fn new(path: &'static str, methods: &'static str) -> Self {
        Self { path, methods }
    }
}

pub const ROOT_ROUTE: RouteInfo = RouteInfo::new("/", "GET");
pub const HEALTH_ROUTE: RouteInfo = RouteInfo::new("/health", "GET");
pub const METRICS_ROUTE: RouteInfo = RouteInfo::new("/metrics", "GET");
pub const MODELS_ROUTE: RouteInfo = RouteInfo::new("/v1/models", "GET");
pub const MODEL_RELOAD_ROUTE: RouteInfo = RouteInfo::new("/v1/models/reload", "POST");
pub const MODEL_UNLOAD_ROUTE: RouteInfo = RouteInfo::new("/v1/models/unload", "POST");
pub const MODEL_STATUS_ROUTE: RouteInfo = RouteInfo::new("/v1/models/status", "POST");
pub const FILES_ROUTE: RouteInfo = RouteInfo::new("/v1/files", "GET,POST");
pub const FILE_ROUTE: RouteInfo = RouteInfo::new("/v1/files/{id}", "GET,DELETE");
pub const FILE_CONTENT_ROUTE: RouteInfo = RouteInfo::new("/v1/files/{id}/content", "GET");
pub const GENERATIONS_ROUTE: RouteInfo = RouteInfo::new("/v1/generations", "GET");
pub const GENERATION_ROUTE: RouteInfo = RouteInfo::new("/v1/generations/{id}", "GET,DELETE");
pub const GENERATION_CANCEL_ROUTE: RouteInfo =
    RouteInfo::new("/v1/generations/{id}/cancel", "POST");
pub const GENERATION_CONTENT_ROUTE: RouteInfo =
    RouteInfo::new("/v1/generations/{id}/content", "GET");
pub const GENERATION_STREAM_ROUTE: RouteInfo =
    RouteInfo::new("/v1/generations/{id}/stream", "GET");
pub const POINT_CLOUD_GENERATIONS_ROUTE: RouteInfo =
    RouteInfo::new("/v1/point-clouds/generations", "POST");
pub const MESH_GENERATIONS_ROUTE: RouteInfo = RouteInfo::new("/v1/meshes/generations", "POST");
pub const SYSTEM_INFO_ROUTE: RouteInfo = RouteInfo::new("/v1/system/info", "GET");
pub const SYSTEM_DOCTOR_ROUTE: RouteInfo = RouteInfo::new("/v1/system/doctor", "POST");

pub const LUX3D_API_ROUTES: &[RouteInfo] = &[
    ROOT_ROUTE,
    HEALTH_ROUTE,
    METRICS_ROUTE,
    MODELS_ROUTE,
    MODEL_RELOAD_ROUTE,
    MODEL_UNLOAD_ROUTE,
    MODEL_STATUS_ROUTE,
    FILES_ROUTE,
    FILE_ROUTE,
    FILE_CONTENT_ROUTE,
    GENERATIONS_ROUTE,
    GENERATION_ROUTE,
    GENERATION_CANCEL_ROUTE,
    GENERATION_CONTENT_ROUTE,
    GENERATION_STREAM_ROUTE,
    POINT_CLOUD_GENERATIONS_ROUTE,
    MESH_GENERATIONS_ROUTE,
    SYSTEM_INFO_ROUTE,
    SYSTEM_DOCTOR_ROUTE,
];
