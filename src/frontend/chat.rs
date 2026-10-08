//! OpenAI Chat Completions front end. Not used by Claude Code or Codex, but
//! most other coding harnesses (Aider, OpenCode, Cline, Continue, ...) speak
//! it, so new harness integrations usually need no new protocol code.

use std::collections::HashMap;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::responses::{route_error_response, safe_tool_name};
use super::{AppState, ModelRoute, parse_body, session_from_headers, sse_frame};
use crate::roulette::{Lane, RouteRequest};

fn text_of(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn user_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                Some("text") => Some(
                    json!({"type": "text", "text": p.get("text").cloned().unwrap_or(json!(""))}),
                ),
                Some("image_url") => {
                    let url = p.pointer("/image_url/url").and_then(Value::as_str)?;
                    Some(
                        match url.strip_prefix("data:").and_then(|r| r.split_once(',')) {
                            Some((meta, data)) => {
                                json!({"type": "image", "source": {"type": "base64",
                            "media_type": meta.trim_end_matches(";base64"), "data": data}})
                            }
                            None => json!({"type": "image", "source": {"type": "url", "url": url}}),
                        },
                    )
                }
                _ => None,
            })
            .collect(),
        other => vec![json!({"type": "text", "text": text_of(other)})],
    }
}

/// Chat request → canonical request, plus canonical→client tool name map.
pub fn convert_request(body: &Value) -> (Value, HashMap<String, String>) {
    let mut system = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for m in body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let content = m.get("content").cloned().unwrap_or(Value::Null);
        match m.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => system.push(text_of(&content)),
            "assistant" => {
                let mut blocks = Vec::new();
                let t = text_of(&content);
                if !t.is_empty() {
                    blocks.push(json!({"type": "text", "text": t}));
                }
                for tc in m
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default()
                {
                    let args = tc
                        .pointer("/function/arguments")
                        .and_then(Value::as_str)
                        .unwrap_or("{}");
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.get("id").cloned().unwrap_or(json!("call")),
                        "name": safe_tool_name(tc.pointer("/function/name").and_then(Value::as_str).unwrap_or("")),
                        "input": serde_json::from_str::<Value>(args).ok().filter(Value::is_object).unwrap_or(json!({}))
                    }));
                }
                messages.push(json!({"role": "assistant", "content": blocks}));
            }
            "tool" => messages.push(json!({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id").cloned().unwrap_or(json!("")),
                "content": text_of(&content)
            }]})),
            _ => messages.push(json!({"role": "user", "content": user_blocks(&content)})),
        }
    }
    let mut names = HashMap::new();
    let tools: Vec<Value> = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|t| {
            let f = t.get("function")?;
            let name = f.get("name")?.as_str()?;
            let canon = safe_tool_name(name);
            names.insert(canon.clone(), name.to_string());
            Some(json!({
                "name": canon,
                "description": f.get("description").cloned().unwrap_or(json!("")),
                "input_schema": f.get("parameters").cloned().unwrap_or(json!({"type": "object", "properties": {}}))
            }))
        })
        .collect();
    let max = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(32_000);
    let mut req = json!({"model": body.get("model").cloned().unwrap_or(json!("")), "max_tokens": max, "messages": messages});
    if !system.is_empty() {
        req["system"] = json!(system.join("\n\n"));
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = body.get(k).filter(|v| !v.is_null()) {
            req[k] = v.clone();
        }
    }
    if let Some(stop) = body.get("stop") {
        req["stop_sequences"] = match stop {
            Value::String(s) => json!([s]),
            other => other.clone(),
        };
    }
    if !tools.is_empty() {
        req["tools"] = Value::Array(tools);
        req["tool_choice"] = match body.get("tool_choice") {
            Some(Value::String(s)) if s == "required" => json!({"type": "any"}),
            Some(Value::String(s)) if s == "none" => json!({"type": "none"}),
            Some(Value::Object(o)) => match o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
            {
                Some(n) => json!({"type": "tool", "name": safe_tool_name(n)}),
                None => json!({"type": "auto"}),
            },
            _ => json!({"type": "auto"}),
        };
    }
    (req, names)
}

/// Canonical events → chat completion chunks.
pub struct ChatEncoder {
    id: String,
    model: String,
    created: i64,
    names: HashMap<String, String>,
    tool_index: HashMap<u64, usize>,
    next_tool: usize,
    usage: Value,
    finish: Option<String>,
    sent_role: bool,
}

impl ChatEncoder {
    pub fn new(model: String, names: HashMap<String, String>) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            model,
            created: chrono::Utc::now().timestamp(),
            names,
            tool_index: HashMap::new(),
            next_tool: 0,
            usage: json!({}),
            finish: None,
            sent_role: false,
        }
    }

    fn chunk(&mut self, mut delta: Value, finish: Option<&str>) -> Value {
        if !self.sent_role {
            delta["role"] = json!("assistant");
            self.sent_role = true;
        }
        json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
    }

    pub fn push(&mut self, e: &Value) -> Vec<Value> {
        let mut out = Vec::new();
        match e.get("type").and_then(Value::as_str).unwrap_or("") {
            "content_block_start" => {
                let cb = e.get("content_block").cloned().unwrap_or(json!({}));
                if cb["type"] == "tool_use" {
                    let i = self.next_tool;
                    self.next_tool += 1;
                    self.tool_index.insert(e["index"].as_u64().unwrap_or(0), i);
                    let canon = cb["name"].as_str().unwrap_or("");
                    let name = self
                        .names
                        .get(canon)
                        .cloned()
                        .unwrap_or_else(|| canon.to_string());
                    let c = self.chunk(
                        json!({"tool_calls": [{"index": i, "id": cb["id"], "type": "function",
                        "function": {"name": name, "arguments": ""}}]}),
                        None,
                    );
                    out.push(c);
                }
            }
            "content_block_delta" => {
                let d = &e["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => out.push(self.chunk(json!({"content": d["text"]}), None)),
                    "thinking_delta" => {
                        out.push(self.chunk(json!({"reasoning_content": d["thinking"]}), None))
                    }
                    "input_json_delta" => {
                        if let Some(i) = self
                            .tool_index
                            .get(&e["index"].as_u64().unwrap_or(0))
                            .copied()
                        {
                            out.push(self.chunk(json!({"tool_calls": [{"index": i, "function": {"arguments": d["partial_json"]}}]}), None));
                        }
                    }
                    _ => {}
                }
            }
            "message_start" => {
                if let Some(u) = e.pointer("/message/usage") {
                    self.usage = u.clone();
                }
            }
            "message_delta" => {
                if let Some(s) = e.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.finish = Some(
                        match s {
                            "tool_use" => "tool_calls",
                            "max_tokens" => "length",
                            _ => "stop",
                        }
                        .to_string(),
                    );
                }
                if let Some(Value::Object(u)) = e.get("usage") {
                    for (k, v) in u {
                        if !v.is_null() {
                            self.usage[k] = v.clone();
                        }
                    }
                }
            }
            "message_stop" => {
                let f = self.finish.clone().unwrap_or_else(|| "stop".into());
                out.push(self.chunk(json!({}), Some(&f)));
            }
            _ => {}
        }
        out
    }

    pub fn usage(&self) -> Value {
        let g = |k: &str| self.usage.get(k).and_then(Value::as_u64).unwrap_or(0);
        let cached = g("cache_read_input_tokens");
        let prompt = g("input_tokens") + cached + g("cache_creation_input_tokens");
        json!({"prompt_tokens": prompt, "completion_tokens": g("output_tokens"),
            "total_tokens": prompt + g("output_tokens"), "prompt_tokens_details": {"cached_tokens": cached}})
    }

    pub fn usage_chunk(&self) -> Value {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
            "choices": [], "usage": self.usage()})
    }
}

fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        axum::Json(json!({"error": {"type": "invalid_request_error", "message": message}})),
    )
        .into_response()
}

pub async fn chat_completions(
    State(app): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !app.authorized(&headers) {
        return error_response(StatusCode::UNAUTHORIZED, "invalid model-roulette api key");
    }
    let body = match parse_body(&headers, &body) {
        Ok(v) => v,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e),
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let lane = match app.route_model(&model) {
        ModelRoute::Lane(l) => l,
        ModelRoute::Passthrough => Lane::Fast,
        ModelRoute::Reject => {
            return error_response(StatusCode::NOT_FOUND, &format!("unknown model '{model}'"));
        }
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let session = session_from_headers(&headers, &["x-session-id", "session-id", "session_id"])
        .or_else(|| body.get("user").and_then(Value::as_str).map(str::to_string));
    let (request, names) = convert_request(&body);
    let rr = RouteRequest {
        request,
        session,
        lane,
        headers: headers.clone(),
        converted: true,
    };
    let routed = match app.roulette.dispatch(rr).await {
        Ok(r) => r,
        Err(e) => return route_error_response(&e),
    };
    let mut enc = ChatEncoder::new(model.clone(), names);

    if stream {
        let events = routed.events;
        let body = async_stream::stream! {
            futures::pin_mut!(events);
            while let Some(item) = events.next().await {
                match item {
                    Ok(ev) => for c in enc.push(&ev) {
                        yield Ok::<Bytes, std::convert::Infallible>(sse_frame(None, &c));
                    },
                    Err(f) => {
                        yield Ok(sse_frame(None, &json!({"error": {"type": "server_error", "message": f.message}})));
                        return;
                    }
                }
            }
            if include_usage {
                yield Ok(sse_frame(None, &enc.usage_chunk()));
            }
            yield Ok(Bytes::from_static(b"data: [DONE]\n\n"));
        };
        let mut resp = Response::new(Body::from_stream(body));
        resp.headers_mut().insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        return resp;
    }

    let mut events = routed.events;
    let mut acc = crate::canonical::Accumulator::default();
    while let Some(item) = events.next().await {
        match item {
            Ok(ev) => {
                enc.push(&ev);
                acc.push(&ev);
            }
            Err(f) => return error_response(StatusCode::BAD_GATEWAY, &f.message),
        }
    }
    let msg = acc.finish();
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for b in msg["content"].as_array().cloned().unwrap_or_default() {
        match b["type"].as_str() {
            Some("text") => text.push_str(b["text"].as_str().unwrap_or("")),
            Some("tool_use") => {
                let canon = b["name"].as_str().unwrap_or("");
                let name = enc
                    .names
                    .get(canon)
                    .cloned()
                    .unwrap_or_else(|| canon.to_string());
                tool_calls.push(json!({"id": b["id"], "type": "function",
                    "function": {"name": name, "arguments": b["input"].to_string()}}));
            }
            _ => {}
        }
    }
    let mut message = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }
    axum::Json(json!({
        "id": enc.id, "object": "chat.completion", "created": enc.created, "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": enc.finish.clone().unwrap_or_else(|| "stop".into())}],
        "usage": enc.usage()
    }))
    .into_response()
}
