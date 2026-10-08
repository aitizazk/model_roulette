//! Client-facing API surfaces. Each harness speaks one of these protocols.
//!
//! * [`anthropic`] – Anthropic Messages API (Claude Code)
//! * [`responses`] – OpenAI Responses API (Codex)
//! * [`chat`]      – OpenAI Chat Completions (generic; most other harnesses)

pub mod anthropic;
pub mod chat;
pub mod responses;

use std::io::Read;
use std::sync::Arc;

use axum::http::{HeaderMap, StatusCode};
use bytes::Bytes;
use serde_json::Value;

use crate::config::UnknownModelPolicy;
use crate::roulette::{Lane, RouteError, Roulette};

#[derive(Clone)]
pub struct AppState {
    pub roulette: Arc<Roulette>,
}

/// Which lane a requested model name maps to.
pub enum ModelRoute {
    Lane(Lane),
    Passthrough,
    Reject,
}

impl AppState {
    pub fn route_model(&self, model: &str) -> ModelRoute {
        let s = &self.roulette.cfg.server;
        if model == s.model_name || model.starts_with(&format!("{}[", s.model_name)) {
            return ModelRoute::Lane(Lane::Main);
        }
        if model == s.fast_model_name {
            return ModelRoute::Lane(Lane::Fast);
        }
        match s.unknown_models {
            UnknownModelPolicy::Roulette => ModelRoute::Lane(Lane::Fast),
            UnknownModelPolicy::Passthrough => ModelRoute::Passthrough,
            UnknownModelPolicy::Reject => ModelRoute::Reject,
        }
    }

    /// Check the client's key if the proxy requires one.
    pub fn authorized(&self, headers: &HeaderMap) -> bool {
        let Some(expected) = &self.roulette.cfg.server.api_key else { return true };
        let presented = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .or_else(|| {
                headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer ").map(str::to_string))
            });
        presented.as_deref() == Some(expected.as_str())
    }
}

/// Parse a JSON body, transparently handling gzip/deflate encodings.
pub fn parse_body(headers: &HeaderMap, body: &Bytes) -> Result<Value, String> {
    let enc = headers.get("content-encoding").and_then(|v| v.to_str().ok()).unwrap_or("");
    let decoded: Vec<u8> = match enc {
        "gzip" => {
            let mut d = flate2::read::GzDecoder::new(&body[..]);
            let mut out = Vec::new();
            d.read_to_end(&mut out).map_err(|e| format!("bad gzip body: {e}"))?;
            out
        }
        "deflate" => {
            let mut d = flate2::read::ZlibDecoder::new(&body[..]);
            let mut out = Vec::new();
            d.read_to_end(&mut out).map_err(|e| format!("bad deflate body: {e}"))?;
            out
        }
        _ => body.to_vec(),
    };
    serde_json::from_slice(&decoded).map_err(|e| format!("invalid JSON body: {e}"))
}

/// Status code and retry-after for a routing error.
pub fn error_status(err: &RouteError) -> (StatusCode, Option<u64>) {
    match err {
        RouteError::Exhausted { retry_after, .. } => {
            (StatusCode::TOO_MANY_REQUESTS, Some(retry_after.map(|d| d.as_secs().max(1)).unwrap_or(30)))
        }
        RouteError::Upstream(f) => (
            f.status.and_then(|s| StatusCode::from_u16(s).ok()).unwrap_or(StatusCode::BAD_REQUEST),
            None,
        ),
        RouteError::NoAccounts => (StatusCode::SERVICE_UNAVAILABLE, None),
    }
}

/// Extract a session id from typical harness headers / body fields.
pub fn session_from_headers(headers: &HeaderMap, names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| {
        headers.get(*n).and_then(|v| v.to_str().ok()).filter(|s| !s.is_empty()).map(str::to_string)
    })
}

pub fn sse_frame(event: Option<&str>, data: &Value) -> Bytes {
    let mut s = String::new();
    if let Some(e) = event {
        s.push_str("event: ");
        s.push_str(e);
        s.push('\n');
    }
    s.push_str("data: ");
    s.push_str(&data.to_string());
    s.push_str("\n\n");
    Bytes::from(s)
}
