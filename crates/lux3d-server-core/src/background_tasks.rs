//! Background cleanup for expired jobs and uploaded files.

use std::sync::Arc;

use tracing::{debug, info};

use crate::types::Lux3dState;

pub fn spawn_cleanup_task(state: Arc<Lux3dState>) {
    let interval = state.generation_defaults.cleanup_interval;
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        loop {
            ticker.tick().await;
            let now = crate::system::now_unix();
            let removed_jobs = state.jobs.cleanup_expired(now);
            let removed_files = state.files.cleanup_expired(now);
            if removed_jobs > 0 || removed_files > 0 {
                info!(
                    removed_jobs,
                    removed_files,
                    "completed background cleanup pass"
                );
            } else {
                debug!("background cleanup pass found nothing to remove");
            }
        }
    });
}
