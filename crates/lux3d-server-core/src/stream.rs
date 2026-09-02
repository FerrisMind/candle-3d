//! Server-Sent Events stream for generation job status updates.

use std::{
    convert::Infallible,
    time::Duration,
};

use async_stream::stream;
use axum::{
    extract::{Path, State},
    response::sse::{Event, KeepAlive, Sse},
};
use futures::Stream;
use tokio::time::sleep;

use crate::{
    api_types::GenerationObject,
    errors::not_found,
    types::ExtractedLux3dState,
};

const STREAM_POLL_INTERVAL: Duration = Duration::from_millis(500);
const STREAM_KEEPALIVE: Duration = Duration::from_secs(15);

#[utoipa::path(
    get,
    tag = "Lux3D",
    path = "/v1/generations/{id}/stream",
    responses(
        (status = 200, description = "SSE stream of generation status updates"),
        (status = 404, description = "Generation not found")
    )
)]
pub async fn stream_generation(
    State(state): ExtractedLux3dState,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, axum::response::Response> {
    if state.jobs.get(&id).is_none() {
        return Err(not_found(format!("generation `{id}` not found")));
    }

    let jobs = state.jobs.clone();
    let stream = stream! {
        let mut last_payload = String::new();
        loop {
            let Some(record) = jobs.get(&id) else {
                break;
            };
            let payload = match serde_json::to_string(&record.object) {
                Ok(json) => json,
                Err(error) => {
                    yield Ok(Event::default().event("error").data(error.to_string()));
                    break;
                }
            };

            if payload != last_payload {
                last_payload = payload.clone();
                yield Ok(Event::default().event("generation.update").data(payload));
            }

            if is_terminal(&record.object) {
                break;
            }
            sleep(STREAM_POLL_INTERVAL).await;
        }
    };

    Ok(Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(STREAM_KEEPALIVE)
            .text("keep-alive"),
    ))
}

fn is_terminal(generation: &GenerationObject) -> bool {
    matches!(
        generation.status.as_str(),
        "completed" | "failed" | "cancelled"
    )
}
