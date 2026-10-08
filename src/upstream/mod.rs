//! Upstream provider adapters.
//!
//! Each adapter takes a canonical (Anthropic-shaped) request and returns a
//! stream of canonical (Anthropic SSE) events. Errors that happen before the
//! stream starts are returned as [`UpstreamFailure`] so the router can fail
//! over; errors inside the stream are yielded as `Err` items.

pub mod anthropic;
pub mod openai_chat;

use std::pin::Pin;
use std::sync::Arc;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

use crate::config::AccountConfig;
use crate::providers::Wire;
use crate::ratelimit::{UpstreamFailure, classify};
use crate::state::StateStore;

pub type EventStream = Pin<Box<dyn Stream<Item = Result<Value, UpstreamFailure>> + Send>>;

/// A configured account with its credentials resolved.
#[derive(Debug, Clone)]
pub struct Account {
    pub cfg: AccountConfig,
    pub api_key: Option<String>,
    pub base_url: String,
}

impl Account {
    pub fn from_config(cfg: AccountConfig) -> Self {
        let api_key = cfg.resolve_api_key();
        let base_url = cfg.base_url().unwrap_or_default();
        Self { cfg, api_key, base_url }
    }

    pub fn id(&self) -> &str {
        &self.cfg.id
    }
}

/// One upstream call.
pub struct Call<'a> {
    pub account: &'a Account,
    pub model: String,
    /// Canonical request (Anthropic Messages shape).
    pub request: Value,
    /// Headers from the client (used for `anthropic-beta`/`anthropic-version`).
    pub client_headers: &'a HeaderMap,
}

pub struct UpstreamResponse {
    pub events: EventStream,
    pub headers: HeaderMap,
}

pub async fn send(
    http: &reqwest::Client,
    store: &Arc<StateStore>,
    call: Call<'_>,
) -> Result<UpstreamResponse, UpstreamFailure> {
    match call.account.cfg.wire() {
        Wire::Anthropic => anthropic::send(http, store, call).await,
        Wire::OpenAiChat => openai_chat::send(http, store, call).await,
    }
}

/// POST JSON and turn non-2xx responses into classified failures.
pub(crate) async fn post_json(
    req: reqwest::RequestBuilder,
    body: &Value,
) -> Result<reqwest::Response, UpstreamFailure> {
    let resp = req
        .header("content-type", "application/json")
        .body(serde_json::to_vec(body).unwrap_or_default())
        .send()
        .await
        .map_err(UpstreamFailure::network)?;
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status().as_u16();
    let headers = resp.headers().clone();
    let text = resp.text().await.unwrap_or_default();
    Err(classify(status, &headers, &text))
}

pub fn is_event_stream(resp: &reqwest::Response) -> bool {
    resp.headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.contains("text/event-stream"))
        .unwrap_or(false)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Parse a byte stream as Server-Sent Events.
pub fn sse_events<S, E>(body: S) -> impl Stream<Item = Result<SseEvent, UpstreamFailure>> + Send
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    async_stream::stream! {
        let mut body = Box::pin(body);
        let mut buf: Vec<u8> = Vec::new();
        let mut event: Option<String> = None;
        let mut data: Vec<String> = Vec::new();
        loop {
            let chunk = match body.next().await {
                Some(Ok(c)) => c,
                Some(Err(e)) => { yield Err(UpstreamFailure::network(e)); return; }
                None => break,
            };
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|b| *b == b'\n') {
                let mut line: Vec<u8> = buf.drain(..=pos).collect();
                line.pop();
                if line.last() == Some(&b'\r') { line.pop(); }
                let line = String::from_utf8_lossy(&line).into_owned();
                if line.is_empty() {
                    if !data.is_empty() || event.is_some() {
                        yield Ok(SseEvent { event: event.take(), data: data.join("\n") });
                        data.clear();
                    }
                    continue;
                }
                if line.starts_with(':') { continue; }
                let (field, value) = match line.split_once(':') {
                    Some((f, v)) => (f.to_string(), v.strip_prefix(' ').unwrap_or(v).to_string()),
                    None => (line.clone(), String::new()),
                };
                match field.as_str() {
                    "event" => event = Some(value),
                    "data" => data.push(value),
                    _ => {}
                }
            }
        }
        if !data.is_empty() {
            yield Ok(SseEvent { event, data: data.join("\n") });
        }
    }
}

/// Turn a complete (non-streamed) Anthropic message into canonical events.
pub fn message_to_events(msg: &Value) -> Vec<Value> {
    let mut start = msg.clone();
    if let Some(obj) = start.as_object_mut() {
        obj.insert("content".into(), json!([]));
        obj.insert("stop_reason".into(), Value::Null);
    }
    let mut out = vec![json!({"type": "message_start", "message": start})];
    for (i, b) in msg.get("content").and_then(Value::as_array).cloned().unwrap_or_default().into_iter().enumerate() {
        let ty = b.get("type").and_then(Value::as_str).unwrap_or("").to_string();
        match ty.as_str() {
            "text" => {
                out.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"text","text":""}}));
                out.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"text_delta","text":b.get("text").cloned().unwrap_or(json!(""))}}));
            }
            "thinking" => {
                out.push(json!({"type":"content_block_start","index":i,"content_block":{"type":"thinking","thinking":"","signature":""}}));
                out.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"thinking_delta","thinking":b.get("thinking").cloned().unwrap_or(json!(""))}}));
                if let Some(sig) = b.get("signature") {
                    out.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"signature_delta","signature":sig}}));
                }
            }
            "tool_use" => {
                let mut start = b.clone();
                start["input"] = json!({});
                out.push(json!({"type":"content_block_start","index":i,"content_block":start}));
                let input = b.get("input").cloned().unwrap_or(json!({}));
                out.push(json!({"type":"content_block_delta","index":i,"delta":{"type":"input_json_delta","partial_json":input.to_string()}}));
            }
            _ => out.push(json!({"type":"content_block_start","index":i,"content_block":b})),
        }
        out.push(json!({"type":"content_block_stop","index":i}));
    }
    out.push(json!({
        "type": "message_delta",
        "delta": {"stop_reason": msg.get("stop_reason").cloned().unwrap_or(json!("end_turn")), "stop_sequence": msg.get("stop_sequence").cloned().unwrap_or(Value::Null)},
        "usage": msg.get("usage").cloned().unwrap_or(json!({"output_tokens": 0}))
    }));
    out.push(json!({"type": "message_stop"}));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn sse_parsing_handles_split_chunks_and_crlf() {
        let chunks: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from("event: a\r\ndata: {\"x\"")),
            Ok(Bytes::from(":1}\r\n\r\n: comment\n\ndata: second\n")),
            Ok(Bytes::from("data: line2\n\ndata: [DONE]\n\n")),
        ];
        let evs: Vec<_> = sse_events(futures::stream::iter(chunks)).collect().await;
        let evs: Vec<SseEvent> = evs.into_iter().map(|e| e.unwrap()).collect();
        assert_eq!(evs[0], SseEvent { event: Some("a".into()), data: "{\"x\":1}".into() });
        assert_eq!(evs[1].data, "second\nline2");
        assert_eq!(evs[2].data, "[DONE]");
        assert_eq!(evs.len(), 3);
    }
}
