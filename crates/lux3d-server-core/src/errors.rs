//! Structured HTTP error responses.

use axum::{Json, http::StatusCode, response::{IntoResponse, Response}};
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct ApiErrorResponse {
    pub error: ApiErrorBody,
}

pub fn api_error(status: StatusCode, error_type: &str, code: &str, message: impl Into<String>) -> Response {
    let body = ApiErrorResponse {
        error: ApiErrorBody {
            message: message.into(),
            error_type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

pub fn invalid_request(message: impl Into<String>) -> Response {
    api_error(
        StatusCode::BAD_REQUEST,
        "invalid_request",
        "invalid_request",
        message,
    )
}

pub fn not_found(message: impl Into<String>) -> Response {
    api_error(StatusCode::NOT_FOUND, "not_found", "not_found", message)
}

pub fn conflict(message: impl Into<String>) -> Response {
    api_error(StatusCode::CONFLICT, "conflict", "conflict", message)
}

pub fn internal_error(message: impl Into<String>) -> Response {
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        "internal_error",
        message,
    )
}
