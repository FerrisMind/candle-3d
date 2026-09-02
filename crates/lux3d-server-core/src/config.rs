//! Server-level configuration and defaults.

use std::time::Duration;

/// Default TTL for generation jobs and uploaded files (24 hours).
pub const DEFAULT_JOB_TTL_SECS: u64 = 86_400;

/// Default interval for background cleanup tasks.
pub const DEFAULT_CLEANUP_INTERVAL_SECS: u64 = 300;

/// Default maximum concurrent GPU inference jobs.
pub const DEFAULT_MAX_CONCURRENT_JOBS: usize = 1;

/// Default maximum entries returned by list endpoints.
pub const DEFAULT_LIST_LIMIT: usize = 20;

/// Default maximum upload size for the Files API (64 MiB).
pub const DEFAULT_MAX_FILE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    pub job_ttl_secs: u64,
    pub max_concurrent_jobs: usize,
    pub cleanup_interval: Duration,
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        Self {
            job_ttl_secs: DEFAULT_JOB_TTL_SECS,
            max_concurrent_jobs: DEFAULT_MAX_CONCURRENT_JOBS,
            cleanup_interval: Duration::from_secs(DEFAULT_CLEANUP_INTERVAL_SECS),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AccessLogFormat {
    #[default]
    Text,
    Json,
}

#[derive(Debug, Clone)]
pub struct ObservabilityConfig {
    pub access_log: bool,
    pub access_log_health: bool,
    pub access_log_format: AccessLogFormat,
    pub request_id_header: bool,
    pub metrics: bool,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            access_log: true,
            access_log_health: false,
            access_log_format: AccessLogFormat::Text,
            request_id_header: true,
            metrics: true,
        }
    }
}
