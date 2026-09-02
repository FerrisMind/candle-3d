//! Prometheus metrics, access logs, and request-id middleware.

use std::{
    sync::OnceLock,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{MatchedPath, Request, State},
    http::{HeaderName, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use tracing::{info, warn};
use uuid::Uuid;

use crate::{
    config::{AccessLogFormat, ObservabilityConfig},
    types::SharedLux3dState,
};

static PROMETHEUS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

const REQUEST_ID_HEADER: &str = "x-request-id";
const UNMATCHED_ROUTE: &str = "<unmatched>";
const OPTIONS_METHOD: &str = "OPTIONS";

const HTTP_REQUEST_DURATION_BUCKETS: [f64; 14] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0, 600.0,
];

#[derive(Clone)]
pub struct ObservabilityState {
    pub config: ObservabilityConfig,
    pub state: SharedLux3dState,
}

impl ObservabilityState {
    pub fn new(config: ObservabilityConfig, state: SharedLux3dState) -> Self {
        Self { config, state }
    }
}

pub fn install_prometheus_recorder() {
    if PROMETHEUS_HANDLE.get().is_some() {
        return;
    }
    let handle = PrometheusBuilder::new()
        .set_buckets_for_metric(
            Matcher::Full("http_request_duration_seconds".to_string()),
            &HTTP_REQUEST_DURATION_BUCKETS,
        )
        .expect("valid HTTP request duration buckets")
        .install_recorder()
        .expect("failed to install Prometheus recorder");
    let _ = PROMETHEUS_HANDLE.set(handle);
}

pub async fn metrics_handler() -> impl IntoResponse {
    match PROMETHEUS_HANDLE.get() {
        Some(handle) => (StatusCode::OK, handle.render()).into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            "metrics recorder not initialized",
        )
            .into_response(),
    }
}

pub async fn observe_http(
    State(observability): State<ObservabilityState>,
    mut request: Request,
    next: Next,
) -> Response {
    let config = observability.config.clone();
    let start = Instant::now();
    let method = request.method().to_string();
    let route = request
        .extensions()
        .get::<MatchedPath>()
        .map(MatchedPath::as_str)
        .unwrap_or(UNMATCHED_ROUTE)
        .to_string();
    let uri_path = request.uri().path().to_string();
    let request_id = if config.request_id_header {
        Some(assign_request_id(&mut request))
    } else {
        None
    };

    let in_flight_labels = [
        ("method", method.clone()),
        ("route", route.clone()),
    ];
    metrics::gauge!("http_requests_in_flight", &in_flight_labels).increment(1.0);

    let response = next.run(request).await;
    metrics::gauge!("http_requests_in_flight", &in_flight_labels).decrement(1.0);

    let status = response.status().as_u16().to_string();
    let elapsed = start.elapsed().as_secs_f64();
    let duration_labels = [
        ("method", method.clone()),
        ("route", route.clone()),
        ("status", status.clone()),
    ];
    metrics::counter!("http_requests_total", &duration_labels).increment(1);
    metrics::histogram!("http_request_duration_seconds", &duration_labels).record(elapsed);

    let housekeeping = is_housekeeping(&method, &route, &uri_path);
    if config.access_log && (config.access_log_health || !housekeeping) {
        log_access(
            &config.access_log_format,
            &method,
            &route,
            &status,
            elapsed,
            request_id.as_deref(),
        );
    }

    let mut response = response;
    if let Some(request_id) = request_id {
        if let Ok(value) = request_id.parse() {
            response
                .headers_mut()
                .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
        }
    }
    response
}

fn assign_request_id(request: &mut Request) -> String {
    if let Some(existing) = request
        .headers()
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty())
    {
        return existing.to_string();
    }
    let request_id = format!("req_{}", Uuid::new_v4().simple());
    if let Ok(value) = request_id.parse() {
        request
            .headers_mut()
            .insert(HeaderName::from_static(REQUEST_ID_HEADER), value);
    }
    request_id
}

fn is_housekeeping(method: &str, route: &str, uri_path: &str) -> bool {
    method == OPTIONS_METHOD
        || route == "/health"
        || uri_path == "/health"
        || route == "/metrics"
        || uri_path == "/metrics"
}

fn log_access(
    format: &AccessLogFormat,
    method: &str,
    route: &str,
    status: &str,
    elapsed_secs: f64,
    request_id: Option<&str>,
) {
    let elapsed_ms = (elapsed_secs * 1_000.0).round() as u64;
    match format {
        AccessLogFormat::Text => {
            info!(
                method,
                route,
                status,
                elapsed_ms,
                request_id,
                "http access"
            );
        }
        AccessLogFormat::Json => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0);
            info!(
                target: "lux3d.access",
                timestamp,
                method,
                route,
                status,
                elapsed_ms,
                request_id,
                "http access"
            );
        }
    }
}

pub fn record_generation_started(model: &str) {
    metrics::counter!("lux3d_generations_started_total", "model" => model.to_string()).increment(1);
    metrics::gauge!("lux3d_generations_active").increment(1.0);
}

pub fn record_generation_finished(model: &str, status: &str, elapsed_secs: f64) {
    metrics::counter!(
        "lux3d_generations_finished_total",
        "model" => model.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
    metrics::histogram!(
        "lux3d_generation_duration_seconds",
        "model" => model.to_string(),
        "status" => status.to_string()
    )
    .record(elapsed_secs);
    metrics::gauge!("lux3d_generations_active").decrement(1.0);
}

pub fn record_webhook_delivery(success: bool) {
    let status = if success { "success" } else { "failure" };
    metrics::counter!("lux3d_webhook_deliveries_total", "status" => status).increment(1);
}

pub fn warn_metrics_unavailable(error: &str) {
    warn!(error, "metrics unavailable");
}
