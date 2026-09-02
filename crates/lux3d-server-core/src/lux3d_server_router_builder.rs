//! Axum router builder for embedding the Lux3D server.

use anyhow::Result;
use axum::{
    Router,
    extract::DefaultBodyLimit,
    http::{Method, header::HeaderName},
    middleware::from_fn_with_state,
    routing::{get, post},
};
use tower_http::cors::{AllowOrigin, CorsLayer};
#[cfg(feature = "swagger-ui")]
use utoipa_swagger_ui::SwaggerUi;

#[cfg(feature = "swagger-ui")]
use crate::openapi_doc::get_openapi_doc;
use crate::{
    background_tasks::spawn_cleanup_task,
    files::{delete_file, get_file, get_file_content, list_files, upload_file},
    handlers::{
        cancel_generation, create_mesh_generation, create_point_cloud_generation,
        delete_generation, get_generation, get_generation_content, health, list_generations,
        model_status, models, reload_model, root, unload_model,
    },
    metrics::{self, ObservabilityState, install_prometheus_recorder, metrics_handler},
    stream::stream_generation,
    system::{system_doctor, system_info},
    types::SharedLux3dState,
};

const N_INPUT_SIZE: usize = 200;
const MB_TO_B: usize = 1024 * 1024;

/// Default request body limit: 200 MiB.
pub const DEFAULT_MAX_BODY_LIMIT: usize = N_INPUT_SIZE * MB_TO_B;

/// Builder for creating a nestable Lux3D Axum router.
pub struct Lux3dServerRouterBuilder {
    lux3d: Option<SharedLux3dState>,
    #[cfg(feature = "swagger-ui")]
    include_swagger_routes: bool,
    #[cfg(feature = "swagger-ui")]
    base_path: Option<String>,
    allowed_origins: Option<Vec<String>>,
    max_body_limit: Option<usize>,
    spawn_background_tasks: bool,
}

impl Default for Lux3dServerRouterBuilder {
    fn default() -> Self {
        Self {
            lux3d: None,
            #[cfg(feature = "swagger-ui")]
            include_swagger_routes: true,
            #[cfg(feature = "swagger-ui")]
            base_path: None,
            allowed_origins: None,
            max_body_limit: None,
            spawn_background_tasks: true,
        }
    }
}

impl Lux3dServerRouterBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_lux3d(mut self, lux3d: SharedLux3dState) -> Self {
        self.lux3d = Some(lux3d);
        self
    }

    #[cfg(feature = "swagger-ui")]
    pub fn with_include_swagger_routes(mut self, include_swagger_routes: bool) -> Self {
        self.include_swagger_routes = include_swagger_routes;
        self
    }

    #[cfg(feature = "swagger-ui")]
    pub fn with_base_path(mut self, base_path: &str) -> Self {
        self.base_path = Some(base_path.to_owned());
        self
    }

    pub fn with_allowed_origins(mut self, origins: Vec<String>) -> Self {
        self.allowed_origins = Some(origins);
        self
    }

    pub fn with_max_body_limit(mut self, max_body_limit: usize) -> Self {
        self.max_body_limit = Some(max_body_limit);
        self
    }

    pub fn with_spawn_background_tasks(mut self, spawn_background_tasks: bool) -> Self {
        self.spawn_background_tasks = spawn_background_tasks;
        self
    }

    pub async fn build(self) -> Result<Router> {
        let lux3d = self
            .lux3d
            .ok_or_else(|| anyhow::anyhow!("`lux3d` state must be set. Use `with_lux3d`."))?;
        let router_max_body_limit = self.max_body_limit.unwrap_or(DEFAULT_MAX_BODY_LIMIT);

        if self.spawn_background_tasks {
            spawn_cleanup_task(lux3d.clone());
        }

        if lux3d.observability.metrics {
            install_prometheus_recorder();
        }

        let allow_origin = if let Some(origins) = self.allowed_origins {
            let parsed_origins: Result<Vec<_>, _> =
                origins.into_iter().map(|origin| origin.parse()).collect();
            match parsed_origins {
                Ok(origins) => AllowOrigin::list(origins),
                Err(_) => anyhow::bail!("Invalid origin format"),
            }
        } else {
            AllowOrigin::any()
        };

        let cors = CorsLayer::new()
            .allow_origin(allow_origin)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([
                header::CONTENT_TYPE,
                header::AUTHORIZATION,
                HeaderName::from_static("x-requested-with"),
                HeaderName::from_static("x-request-id"),
            ]);

        let observability = ObservabilityState::new(lux3d.observability.clone(), lux3d.clone());

        let mut router = Router::new()
            .route("/", get(root))
            .route("/health", get(health))
            .route("/metrics", get(metrics_handler))
            .route("/v1/models", get(models))
            .route("/v1/models/reload", post(reload_model))
            .route("/v1/models/unload", post(unload_model))
            .route("/v1/models/status", post(model_status))
            .route("/v1/files", get(list_files).post(upload_file))
            .route("/v1/files/{id}", get(get_file).delete(delete_file))
            .route("/v1/files/{id}/content", get(get_file_content))
            .route("/v1/generations", get(list_generations))
            .route(
                "/v1/generations/{id}",
                get(get_generation).delete(delete_generation),
            )
            .route("/v1/generations/{id}/cancel", post(cancel_generation))
            .route("/v1/generations/{id}/content", get(get_generation_content))
            .route("/v1/generations/{id}/stream", get(stream_generation))
            .route(
                "/v1/point-clouds/generations",
                post(create_point_cloud_generation),
            )
            .route("/v1/meshes/generations", post(create_mesh_generation))
            .route("/v1/system/info", get(system_info))
            .route("/v1/system/doctor", post(system_doctor))
            .layer(from_fn_with_state(
                observability,
                metrics::observe_http,
            ))
            .layer(DefaultBodyLimit::max(router_max_body_limit))
            .layer(cors)
            .with_state(lux3d);

        #[cfg(feature = "swagger-ui")]
        if self.include_swagger_routes {
            let prefix = self.base_path.as_deref().unwrap_or("");
            let doc = get_openapi_doc(None);
            router = router.merge(
                SwaggerUi::new(format!("{prefix}/docs"))
                    .url(format!("{prefix}/api-doc/openapi.json"), doc),
            );
        }

        Ok(router)
    }
}

use axum::http::header;
