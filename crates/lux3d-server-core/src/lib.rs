//! > **Lux3D server core**
//!
//! ## About
//!
//! This crate powers the Lux3D HTTP server. It exposes an OpenAI-style async generation API
//! so other applications can embed 3D point-cloud and mesh generation endpoints.
//!
//! ### Features
//! 1. Incorporate the Lux3D server into another Axum project.
//! 2. Poll async generation jobs and download completed assets.
//! 3. Files API, SSE streams, model cache management, and webhooks.

pub mod api_types;
pub mod background_tasks;
pub mod config;
pub mod errors;
pub mod files;
pub mod handlers;
pub mod jobs;
pub mod lux3d_for_server_builder;
pub mod lux3d_server_router_builder;
pub mod media_source;
pub mod metrics;
pub mod models;
pub mod openapi_doc;
pub mod route_registry;
pub mod stream;
pub mod system;
pub mod types;
pub mod webhooks;
mod upload;

pub use config::{GenerationDefaults, ObservabilityConfig};
pub use types::{DeviceBackend, ExtractedLux3dState, Lux3dState, SharedLux3dState};
