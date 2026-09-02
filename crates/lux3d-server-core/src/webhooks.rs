//! Webhook delivery for completed or failed generation jobs.

use reqwest::Client;
use serde::Serialize;
use tracing::{info, warn};
use url::Url;

use crate::{
    api_types::GenerationObject,
    metrics::record_webhook_delivery,
};

#[derive(Debug, Serialize)]
struct WebhookPayload<'a> {
    object: &'static str,
    event: &'static str,
    data: &'a GenerationObject,
}

pub async fn deliver_generation_webhook(
    client: &Client,
    webhook_url: &str,
    event: &'static str,
    generation: &GenerationObject,
) {
    if !is_allowed_webhook_url(webhook_url) {
        warn!(webhook_url, "skipping webhook delivery for disallowed URL");
        record_webhook_delivery(false);
        return;
    }

    let payload = WebhookPayload {
        object: "event",
        event,
        data: generation,
    };

    match client
        .post(webhook_url)
        .json(&payload)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
    {
        Ok(response) if response.status().is_success() => {
            info!(webhook_url, event, generation_id = %generation.id, "webhook delivered");
            record_webhook_delivery(true);
        }
        Ok(response) => {
            warn!(
                webhook_url,
                event,
                generation_id = %generation.id,
                status = response.status().as_u16(),
                "webhook returned non-success status"
            );
            record_webhook_delivery(false);
        }
        Err(error) => {
            warn!(
                webhook_url,
                event,
                generation_id = %generation.id,
                %error,
                "webhook delivery failed"
            );
            record_webhook_delivery(false);
        }
    }
}

fn is_allowed_webhook_url(url: &str) -> bool {
    Url::parse(url)
        .ok()
        .is_some_and(|parsed| matches!(parsed.scheme(), "http" | "https"))
}
