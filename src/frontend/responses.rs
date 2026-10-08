//! OpenAI Responses API front end (Codex).

use std::collections::HashMap;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::{AppState, ModelRoute, error_status, parse_body, session_from_headers, sse_frame};
use crate::ratelimit::{FailureKind, UpstreamFailure};
use crate::roulette::{Lane, RouteError, RouteRequest};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolKind {
    Function,
    Custom,
    LocalShell,
}

#[derive(Debug, Clone)]
pub struct ToolInfo {
    pub kind: ToolKind,
    pub namespace: Option<String>,
    pub name: String,
}

pub type ToolMap = HashMap<String, ToolInfo>;

/// Canonical tool name: ≤64 chars of `[A-Za-z0-9_-]`, deterministic.
pub fn safe_tool_name(name: &str) -> String {
    let clean: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if !clean.is_empty() && clean.len() <= 64 && clean == name {
        return clean;
    }
    let hash = &crate::canonical::hash_str(name)[..8];
    let keep = clean.len().min(55);
    format!("{}_{}", &clean[..keep], hash)
}

/// Flatten a namespaced tool into one canonical name.
pub fn flatten_name(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(ns) if !ns.is_empty() && !name.starts_with(ns) => {
            safe_tool_name(&format!("{ns}__{name}"))
        }
        _ => safe_tool_name(name),
    }
}

fn convert_tools(tools: &[Value], map: &mut ToolMap, ns: Option<&str>, out: &mut Vec<Value>) {
    for t in tools {
        match t.get("type").and_then(Value::as_str).unwrap_or("") {
            "function" => {
                let Some(name) = t.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let canon = flatten_name(ns, name);
                let mut schema = t
                    .get("parameters")
                    .cloned()
                    .unwrap_or(json!({"type": "object", "properties": {}}));
                if schema.is_null() {
                    schema = json!({"type": "object", "properties": {}});
                }
                out.push(json!({
                    "name": canon,
                    "description": t.get("description").cloned().unwrap_or(json!("")),
                    "input_schema": schema
                }));
                map.insert(
                    canon,
                    ToolInfo {
                        kind: ToolKind::Function,
                        namespace: ns.map(str::to_string),
                        name: name.to_string(),
                    },
                );
            }
            "namespace" => {
                let inner_ns = t.get("name").and_then(Value::as_str);
                if let Some(Value::Array(inner)) = t.get("tools") {
                    convert_tools(inner, map, inner_ns, out);
                }
            }
            "custom" => {
                let Some(name) = t.get("name").and_then(Value::as_str) else {
                    continue;
                };
                let canon = flatten_name(ns, name);
                let mut desc = t
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if let Some(def) = t.pointer("/format/definition").and_then(Value::as_str) {
                    desc.push_str("\n\nPass the raw tool input as the `input` string. It must follow this grammar:\n");
                    desc.push_str(def);
                } else {
                    desc.push_str("\n\nPass the raw tool input as the `input` string.");
                }
                out.push(json!({
                    "name": canon,
                    "description": desc,
                    "input_schema": {
                        "type": "object",
                        "properties": {"input": {"type": "string", "description": "Raw input for the tool"}},
                        "required": ["input"]
                    }
                }));
                map.insert(
                    canon,
                    ToolInfo {
                        kind: ToolKind::Custom,
                        namespace: ns.map(str::to_string),
                        name: name.to_string(),
                    },
                );
            }
            "local_shell" => {
                out.push(json!({
                    "name": "local_shell",
                    "description": "Run a shell command locally.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "array", "items": {"type": "string"}},
                            "workdir": {"type": "string"},
                            "timeout_ms": {"type": "number"}
                        },
                        "required": ["command"]
                    }
                }));
                map.insert(
                    "local_shell".into(),
                    ToolInfo {
                        kind: ToolKind::LocalShell,
                        namespace: None,
                        name: "local_shell".into(),
                    },
                );
            }
            // Hosted tools (web_search, file_search, image_generation, ...)
            // only exist on OpenAI's servers.
            _ => {}
        }
    }
}

fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media = meta.strip_suffix(";base64")?;
    Some((media.to_string(), data.to_string()))
}

fn image_block(url: &str) -> Value {
    match parse_data_url(url) {
        Some((media, data)) => {
            json!({"type": "image", "source": {"type": "base64", "media_type": media, "data": data}})
        }
        None => json!({"type": "image", "source": {"type": "url", "url": url}}),
    }
}

fn content_blocks(content: &Value) -> Vec<Value> {
    match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| match p.get("type").and_then(Value::as_str).unwrap_or("") {
                "input_text" | "output_text" | "text" => {
                    Some(json!({"type": "text", "text": p.get("text").cloned().unwrap_or(json!(""))}))
                }
                "refusal" => Some(json!({"type": "text", "text": p.get("refusal").cloned().unwrap_or(json!(""))})),
                "input_image" => {
                    let url = p.get("image_url").and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Value::as_str)))?;
                    Some(image_block(url))
                }
                "input_file" => Some(json!({"type": "text", "text": format!(
                    "[attached file: {}]",
                    p.get("filename").and_then(Value::as_str).unwrap_or("file")
                )})),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

fn output_to_content(output: &Value) -> Value {
    match output {
        Value::String(s) => json!(s),
        Value::Array(_) => Value::Array(content_blocks(output)),
        other => json!(other.to_string()),
    }
}

/// Convert a Responses request into a canonical request.
pub fn convert_request(body: &Value) -> (Value, ToolMap) {
    let mut system_parts: Vec<String> = Vec::new();
    if let Some(s) = body.get("instructions").and_then(Value::as_str)
        && !s.trim().is_empty()
    {
        system_parts.push(s.to_string());
    }
    let mut messages: Vec<Value> = Vec::new();
    let mut push = |role: &str, blocks: Vec<Value>| {
        if blocks.is_empty() {
            return;
        }
        if let Some(last) = messages.last_mut()
            && last["role"] == role
        {
            last["content"].as_array_mut().unwrap().extend(blocks);
            return;
        }
        messages.push(json!({"role": role, "content": blocks}));
    };

    let items: Vec<Value> = match body.get("input") {
        Some(Value::String(s)) => vec![json!({"type": "message", "role": "user", "content": s})],
        Some(Value::Array(a)) => a.clone(),
        _ => vec![],
    };
    for item in &items {
        let ty = item
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("message");
        match ty {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let blocks = content_blocks(item.get("content").unwrap_or(&Value::Null));
                match role {
                    "system" | "developer" => {
                        let text: Vec<String> = blocks
                            .iter()
                            .filter_map(|b| {
                                b.get("text").and_then(Value::as_str).map(str::to_string)
                            })
                            .collect();
                        system_parts.push(text.join("\n"));
                    }
                    "assistant" => push("assistant", blocks),
                    _ => push("user", blocks),
                }
            }
            "function_call" => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let ns = item.get("namespace").and_then(Value::as_str);
                let args = item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let input = match serde_json::from_str::<Value>(args) {
                    Ok(v @ Value::Object(_)) => v,
                    Ok(v) => json!({"value": v}),
                    Err(_) => json!({"_raw_arguments": args}),
                };
                push(
                    "assistant",
                    vec![json!({
                        "type": "tool_use",
                        "id": item.get("call_id").cloned().unwrap_or(json!("call")),
                        "name": flatten_name(ns, name),
                        "input": input
                    })],
                );
            }
            "custom_tool_call" => {
                let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                let ns = item.get("namespace").and_then(Value::as_str);
                push(
                    "assistant",
                    vec![json!({
                        "type": "tool_use",
                        "id": item.get("call_id").cloned().unwrap_or(json!("call")),
                        "name": flatten_name(ns, name),
                        "input": {"input": item.get("input").cloned().unwrap_or(json!(""))}
                    })],
                );
            }
            "local_shell_call" => {
                let id = item
                    .get("call_id")
                    .or_else(|| item.get("id"))
                    .cloned()
                    .unwrap_or(json!("call"));
                let action = item.get("action").cloned().unwrap_or(json!({}));
                push(
                    "assistant",
                    vec![json!({
                        "type": "tool_use",
                        "id": id,
                        "name": "local_shell",
                        "input": {
                            "command": action.get("command").cloned().unwrap_or(json!([])),
                            "workdir": action.get("working_directory").cloned().unwrap_or(Value::Null),
                            "timeout_ms": action.get("timeout_ms").cloned().unwrap_or(Value::Null)
                        }
                    })],
                );
            }
            "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" => {
                let output = item.get("output").cloned().unwrap_or(json!(""));
                match item.get("call_id").and_then(Value::as_str) {
                    Some(call_id) => push(
                        "user",
                        vec![json!({
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": output_to_content(&output)
                        })],
                    ),
                    None => {
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("tool");
                        let text = match &output {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        push(
                            "user",
                            vec![
                                json!({"type": "text", "text": format!("[{name} output]\n{text}")}),
                            ],
                        );
                    }
                }
            }
            // reasoning, web_search_call, compaction, tool_search_* ... are
            // provider-internal and can't be replayed to other models.
            _ => {}
        }
    }

    let mut tool_map = ToolMap::new();
    let mut tools = Vec::new();
    if let Some(Value::Array(ts)) = body.get("tools") {
        convert_tools(ts, &mut tool_map, None, &mut tools);
    }

    let mut req = json!({
        "model": body.get("model").cloned().unwrap_or(json!("model-roulette")),
        "max_tokens": body.get("max_output_tokens").and_then(Value::as_u64).unwrap_or(32_000),
        "messages": messages,
        "stream": true,
    });
    let system = system_parts.join("\n\n");
    if !system.trim().is_empty() {
        req["system"] = json!(system);
    }
    for k in ["temperature", "top_p"] {
        if let Some(v) = body.get(k).filter(|v| !v.is_null()) {
            req[k] = v.clone();
        }
    }
    if !tools.is_empty() {
        req["tools"] = Value::Array(tools);
        let mut tc = match body.get("tool_choice") {
            Some(Value::String(s)) if s == "required" => json!({"type": "any"}),
            Some(Value::String(s)) if s == "none" => json!({"type": "none"}),
            Some(Value::Object(o)) => match o.get("name").and_then(Value::as_str) {
                Some(n) => json!({"type": "tool", "name": flatten_name(None, n)}),
                None => json!({"type": "auto"}),
            },
            _ => json!({"type": "auto"}),
        };
        if body.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
            && tc["type"] != "none"
        {
            tc["disable_parallel_tool_use"] = json!(true);
        }
        req["tool_choice"] = tc;
    }
    (req, tool_map)
}

enum Block {
    Text {
        item_id: String,
        output_index: usize,
        text: String,
    },
    Reasoning {
        item_id: String,
        output_index: usize,
        text: String,
    },
    Tool {
        item_id: String,
        output_index: usize,
        call_id: String,
        name: String,
        json: String,
    },
    Ignored,
}

/// Converts canonical events into Responses API stream events.
pub struct ResponsesEncoder {
    pub id: String,
    model: String,
    created_at: i64,
    tools: ToolMap,
    seq: u64,
    next_output: usize,
    blocks: HashMap<u64, Block>,
    output: Vec<(usize, Value)>,
    stop_reason: Option<String>,
    usage: Value,
}

impl ResponsesEncoder {
    pub fn new(model: String, tools: ToolMap) -> Self {
        Self {
            id: format!("resp_{}", uuid::Uuid::new_v4().simple()),
            model,
            created_at: chrono::Utc::now().timestamp(),
            tools,
            seq: 0,
            next_output: 0,
            blocks: HashMap::new(),
            output: Vec::new(),
            stop_reason: None,
            usage: json!({}),
        }
    }

    fn ev(&mut self, mut v: Value) -> Value {
        v["sequence_number"] = json!(self.seq);
        self.seq += 1;
        v
    }

    fn response_obj(&self, status: &str) -> Value {
        let mut out = self.output.clone();
        out.sort_by_key(|(i, _)| *i);
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": out.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
            "parallel_tool_calls": true,
            "tool_choice": "auto",
            "tools": [],
        })
    }

    fn usage_obj(&self) -> Value {
        let g = |k: &str| self.usage.get(k).and_then(Value::as_u64).unwrap_or(0);
        let cached = g("cache_read_input_tokens");
        let input = g("input_tokens") + cached + g("cache_creation_input_tokens");
        let output = g("output_tokens");
        json!({
            "input_tokens": input,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens": output,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input + output
        })
    }

    pub fn push(&mut self, e: &Value) -> Vec<Value> {
        let mut out = Vec::new();
        match e.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(u) = e.pointer("/message/usage") {
                    self.usage = u.clone();
                }
                let r = self.response_obj("in_progress");
                out.push(self.ev(json!({"type": "response.created", "response": r})));
                let r = self.response_obj("in_progress");
                out.push(self.ev(json!({"type": "response.in_progress", "response": r})));
            }
            "content_block_start" => {
                let idx = e.get("index").and_then(Value::as_u64).unwrap_or(0);
                let cb = e.get("content_block").cloned().unwrap_or(json!({}));
                let oi = self.next_output;
                let block = match cb.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => {
                        self.next_output += 1;
                        let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                        out.push(self.ev(json!({"type":"response.output_item.added","output_index":oi,
                            "item":{"type":"message","id":item_id,"status":"in_progress","role":"assistant","content":[]}})));
                        out.push(self.ev(json!({"type":"response.content_part.added","item_id":item_id,"output_index":oi,"content_index":0,
                            "part":{"type":"output_text","text":"","annotations":[]}})));
                        Block::Text {
                            item_id,
                            output_index: oi,
                            text: String::new(),
                        }
                    }
                    "thinking" => {
                        self.next_output += 1;
                        let item_id = format!("rs_{}", uuid::Uuid::new_v4().simple());
                        out.push(self.ev(
                            json!({"type":"response.output_item.added","output_index":oi,
                            "item":{"type":"reasoning","id":item_id,"summary":[]}}),
                        ));
                        out.push(self.ev(json!({"type":"response.reasoning_summary_part.added","item_id":item_id,"output_index":oi,"summary_index":0,
                            "part":{"type":"summary_text","text":""}})));
                        Block::Reasoning {
                            item_id,
                            output_index: oi,
                            text: String::new(),
                        }
                    }
                    "tool_use" => {
                        self.next_output += 1;
                        let item_id = format!("fc_{}", uuid::Uuid::new_v4().simple());
                        let call_id = cb
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or("call")
                            .to_string();
                        let name = cb
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let item = self.tool_item(&item_id, &call_id, &name, None, "in_progress");
                        out.push(self.ev(json!({"type":"response.output_item.added","output_index":oi,"item":item})));
                        Block::Tool {
                            item_id,
                            output_index: oi,
                            call_id,
                            name,
                            json: String::new(),
                        }
                    }
                    _ => Block::Ignored,
                };
                self.blocks.insert(idx, block);
            }
            "content_block_delta" => {
                let idx = e.get("index").and_then(Value::as_u64).unwrap_or(0);
                let d = e.get("delta").cloned().unwrap_or(json!({}));
                let mut pending = None;
                match self.blocks.get_mut(&idx) {
                    Some(Block::Text {
                        item_id,
                        output_index,
                        text,
                    }) => {
                        if let Some(t) = d.get("text").and_then(Value::as_str) {
                            text.push_str(t);
                            pending = Some(
                                json!({"type":"response.output_text.delta","item_id":item_id,"output_index":output_index,"content_index":0,"delta":t}),
                            );
                        }
                    }
                    Some(Block::Reasoning {
                        item_id,
                        output_index,
                        text,
                    }) => {
                        if let Some(t) = d.get("thinking").and_then(Value::as_str) {
                            text.push_str(t);
                            pending = Some(
                                json!({"type":"response.reasoning_summary_text.delta","item_id":item_id,"output_index":output_index,"summary_index":0,"delta":t}),
                            );
                        }
                    }
                    Some(Block::Tool {
                        item_id,
                        output_index,
                        json: buf,
                        ..
                    }) => {
                        if let Some(t) = d.get("partial_json").and_then(Value::as_str) {
                            buf.push_str(t);
                            pending = Some(
                                json!({"type":"response.function_call_arguments.delta","item_id":item_id,"output_index":output_index,"delta":t}),
                            );
                        }
                    }
                    _ => {}
                }
                if let Some(p) = pending {
                    out.push(self.ev(p));
                }
            }
            "content_block_stop" => {
                let idx = e.get("index").and_then(Value::as_u64).unwrap_or(0);
                match self.blocks.remove(&idx) {
                    Some(Block::Text {
                        item_id,
                        output_index,
                        text,
                    }) => {
                        out.push(self.ev(json!({"type":"response.output_text.done","item_id":item_id,"output_index":output_index,"content_index":0,"text":text})));
                        out.push(self.ev(json!({"type":"response.content_part.done","item_id":item_id,"output_index":output_index,"content_index":0,
                            "part":{"type":"output_text","text":text,"annotations":[]}})));
                        let item = json!({"type":"message","id":item_id,"status":"completed","role":"assistant",
                            "content":[{"type":"output_text","text":text,"annotations":[]}]});
                        out.push(self.ev(json!({"type":"response.output_item.done","output_index":output_index,"item":item})));
                        self.output.push((output_index, item));
                    }
                    Some(Block::Reasoning {
                        item_id,
                        output_index,
                        text,
                    }) => {
                        out.push(self.ev(json!({"type":"response.reasoning_summary_text.done","item_id":item_id,"output_index":output_index,"summary_index":0,"text":text})));
                        out.push(self.ev(json!({"type":"response.reasoning_summary_part.done","item_id":item_id,"output_index":output_index,"summary_index":0,
                            "part":{"type":"summary_text","text":text}})));
                        let item = json!({"type":"reasoning","id":item_id,"summary":[{"type":"summary_text","text":text}],"encrypted_content":null});
                        out.push(self.ev(json!({"type":"response.output_item.done","output_index":output_index,"item":item})));
                        self.output.push((output_index, item));
                    }
                    Some(Block::Tool {
                        item_id,
                        output_index,
                        call_id,
                        name,
                        json: buf,
                    }) => {
                        let args = if buf.trim().is_empty() {
                            "{}".to_string()
                        } else {
                            buf
                        };
                        let item =
                            self.tool_item(&item_id, &call_id, &name, Some(&args), "completed");
                        if item["type"] == "function_call" {
                            out.push(self.ev(json!({"type":"response.function_call_arguments.done","item_id":item_id,"output_index":output_index,"arguments":item["arguments"]})));
                        }
                        out.push(self.ev(json!({"type":"response.output_item.done","output_index":output_index,"item":item})));
                        self.output.push((output_index, item));
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                if let Some(s) = e.pointer("/delta/stop_reason").and_then(Value::as_str) {
                    self.stop_reason = Some(s.to_string());
                }
                if let Some(Value::Object(u)) = e.get("usage") {
                    for (k, v) in u {
                        if !v.is_null() {
                            self.usage[k] = v.clone();
                        }
                    }
                }
            }
            "message_stop" => out.push(self.finish()),
            _ => {}
        }
        out
    }

    fn tool_item(
        &self,
        item_id: &str,
        call_id: &str,
        canon: &str,
        args: Option<&str>,
        status: &str,
    ) -> Value {
        let info = self.tools.get(canon).cloned().unwrap_or(ToolInfo {
            kind: ToolKind::Function,
            namespace: None,
            name: canon.to_string(),
        });
        match info.kind {
            ToolKind::Function => {
                let mut item = json!({"type":"function_call","id":item_id,"call_id":call_id,"name":info.name,
                    "arguments": args.unwrap_or(""),"status":status});
                if let Some(ns) = info.namespace {
                    item["namespace"] = json!(ns);
                }
                item
            }
            ToolKind::Custom => {
                let input = args
                    .map(|a| match serde_json::from_str::<Value>(a) {
                        Ok(v) => match v.get("input") {
                            Some(Value::String(s)) => s.clone(),
                            Some(other) => other.to_string(),
                            None => a.to_string(),
                        },
                        Err(_) => a.to_string(),
                    })
                    .unwrap_or_default();
                let mut item = json!({"type":"custom_tool_call","id":item_id,"call_id":call_id,"name":info.name,"input":input,"status":status});
                if let Some(ns) = info.namespace {
                    item["namespace"] = json!(ns);
                }
                item
            }
            ToolKind::LocalShell => {
                let a: Value = args
                    .and_then(|a| serde_json::from_str(a).ok())
                    .unwrap_or(json!({}));
                json!({"type":"local_shell_call","id":item_id,"call_id":call_id,"status":status,
                    "action":{"type":"exec","command":a.get("command").cloned().unwrap_or(json!([])),
                    "working_directory":a.get("workdir").cloned().unwrap_or(Value::Null),
                    "timeout_ms":a.get("timeout_ms").cloned().unwrap_or(Value::Null),"env":{}}})
            }
        }
    }

    fn finish(&mut self) -> Value {
        let incomplete = self.stop_reason.as_deref() == Some("max_tokens");
        let mut r = self.response_obj(if incomplete {
            "incomplete"
        } else {
            "completed"
        });
        r["usage"] = self.usage_obj();
        if incomplete {
            r["incomplete_details"] = json!({"reason": "max_output_tokens"});
            self.ev(json!({"type": "response.incomplete", "response": r}))
        } else {
            self.ev(json!({"type": "response.completed", "response": r}))
        }
    }

    pub fn failed(&mut self, f: &UpstreamFailure) -> Value {
        let mut r = self.response_obj("failed");
        let code = match f.kind {
            FailureKind::RateLimited | FailureKind::QuotaExhausted => "rate_limit_exceeded",
            _ => "server_error",
        };
        r["error"] = json!({"code": code, "message": format!("model-roulette upstream error: {}", f.message)});
        self.ev(json!({"type": "response.failed", "response": r}))
    }
}

fn error_body(kind: &str, code: &str, message: &str) -> Value {
    json!({"error": {"type": kind, "code": code, "message": message, "param": null}})
}

fn error_response(
    status: StatusCode,
    kind: &str,
    code: &str,
    message: &str,
    retry: Option<u64>,
) -> Response {
    let mut resp = (status, axum::Json(error_body(kind, code, message))).into_response();
    if let Some(s) = retry {
        resp.headers_mut()
            .insert("retry-after", HeaderValue::from(s));
    }
    resp
}

pub fn route_error_response(err: &RouteError) -> Response {
    let (status, retry) = error_status(err);
    let (kind, code) = match err {
        RouteError::Exhausted { .. } => ("rate_limit_error", "rate_limit_exceeded"),
        RouteError::Upstream(_) => ("invalid_request_error", "invalid_request"),
        RouteError::NoAccounts => ("server_error", "no_accounts"),
    };
    error_response(status, kind, code, &err.to_string(), retry)
}

pub fn session_id(headers: &HeaderMap, body: &Value) -> Option<String> {
    session_from_headers(
        headers,
        &[
            "session-id",
            "session_id",
            "thread-id",
            "conversation_id",
            "x-session-id",
        ],
    )
    .or_else(|| {
        body.get("prompt_cache_key")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

pub async fn responses(State(app): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    if !app.authorized(&headers) {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "invalid_request_error",
            "invalid_api_key",
            "invalid model-roulette api key",
            None,
        );
    }
    let body = match parse_body(&headers, &body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "invalid_body",
                &e,
                None,
            );
        }
    };
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let lane = match app.route_model(&model) {
        ModelRoute::Lane(l) => l,
        // The Responses surface has no passthrough target; serve via roulette.
        ModelRoute::Passthrough => Lane::Fast,
        ModelRoute::Reject => {
            return error_response(
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
                &format!("unknown model '{model}'"),
                None,
            );
        }
    };
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let session = session_id(&headers, &body);
    let (request, tools) = convert_request(&body);
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
    let mut enc = ResponsesEncoder::new(model, tools);
    let account = routed.account.clone();
    let upstream_model = routed.model.clone();

    if stream {
        let events = routed.events;
        let body = async_stream::stream! {
            futures::pin_mut!(events);
            while let Some(item) = events.next().await {
                match item {
                    Ok(ev) => for out in enc.push(&ev) {
                        let ty = out["type"].as_str().unwrap_or("").to_string();
                        yield Ok::<Bytes, std::convert::Infallible>(sse_frame(Some(&ty), &out));
                    },
                    Err(f) => {
                        let out = enc.failed(&f);
                        yield Ok(sse_frame(Some("response.failed"), &out));
                        return;
                    }
                }
            }
        };
        let mut resp = Response::new(Body::from_stream(body));
        let h = resp.headers_mut();
        h.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        h.insert("cache-control", HeaderValue::from_static("no-cache"));
        h.insert(
            "x-model-roulette-account",
            HeaderValue::from_str(&account).unwrap_or(HeaderValue::from_static("?")),
        );
        h.insert(
            "x-model-roulette-model",
            HeaderValue::from_str(&upstream_model).unwrap_or(HeaderValue::from_static("?")),
        );
        return resp;
    }

    let mut events = routed.events;
    let mut final_resp = Value::Null;
    while let Some(item) = events.next().await {
        match item {
            Ok(ev) => {
                for out in enc.push(&ev) {
                    if matches!(
                        out["type"].as_str(),
                        Some("response.completed") | Some("response.incomplete")
                    ) {
                        final_resp = out["response"].clone();
                    }
                }
            }
            Err(f) => {
                return error_response(
                    StatusCode::BAD_GATEWAY,
                    "server_error",
                    "upstream_error",
                    &f.message,
                    None,
                );
            }
        }
    }
    axum::Json(final_resp).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_codex_request() {
        let body = json!({
            "model": "model-roulette",
            "instructions": "base",
            "input": [
                {"type":"message","role":"developer","content":[{"type":"input_text","text":"dev rules"}]},
                {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"},{"type":"input_image","image_url":"data:image/png;base64,QUJD"}]},
                {"type":"reasoning","summary":[],"encrypted_content":"xyz"},
                {"type":"message","role":"assistant","content":[{"type":"output_text","text":"let me look"}]},
                {"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"ls\"}","call_id":"c1"},
                {"type":"function_call","name":"spawn_agent","namespace":"collab","arguments":"{}","call_id":"c2"},
                {"type":"function_call_output","call_id":"c1","output":"a.txt"},
                {"type":"function_call_output","call_id":"c2","output":[{"type":"input_text","text":"ok"}]},
                {"type":"custom_tool_call","name":"apply_patch","input":"*** Begin Patch","call_id":"c3"},
                {"type":"custom_tool_call_output","call_id":"c3","output":"Done"}
            ],
            "tools": [
                {"type":"function","name":"exec_command","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}}},
                {"type":"namespace","name":"collab","tools":[{"type":"function","name":"spawn_agent","parameters":{"type":"object","properties":{}}}]},
                {"type":"custom","name":"apply_patch","description":"patch","format":{"type":"grammar","definition":"start: x"}},
                {"type":"web_search"}
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true
        });
        let (req, map) = convert_request(&body);
        assert_eq!(req["system"], "base\n\ndev rules");
        let msgs = req["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"][1]["source"]["media_type"], "image/png");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"][1]["name"], "exec_command");
        assert_eq!(msgs[1]["content"][2]["name"], "collab__spawn_agent");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[3]["content"][0]["input"]["input"], "*** Begin Patch");
        assert_eq!(req["tools"].as_array().unwrap().len(), 3);
        assert_eq!(
            map["collab__spawn_agent"].namespace.as_deref(),
            Some("collab")
        );
        assert_eq!(map["apply_patch"].kind, ToolKind::Custom);
        assert_eq!(req["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn encodes_events() {
        let mut map = ToolMap::new();
        map.insert(
            "collab__spawn_agent".into(),
            ToolInfo {
                kind: ToolKind::Function,
                namespace: Some("collab".into()),
                name: "spawn_agent".into(),
            },
        );
        map.insert(
            "apply_patch".into(),
            ToolInfo {
                kind: ToolKind::Custom,
                namespace: None,
                name: "apply_patch".into(),
            },
        );
        let mut enc = ResponsesEncoder::new("model-roulette".into(), map);
        let evs = vec![
            json!({"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"t1","name":"collab__spawn_agent","input":{}}}),
            json!({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}),
            json!({"type":"content_block_stop","index":2}),
            json!({"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"t2","name":"apply_patch","input":{}}}),
            json!({"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"input\":\"*** Begin\"}"}}),
            json!({"type":"content_block_stop","index":3}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}),
            json!({"type":"message_stop"}),
        ];
        let out: Vec<Value> = evs.iter().flat_map(|e| enc.push(e)).collect();
        let done: Vec<&Value> = out
            .iter()
            .filter(|e| e["type"] == "response.output_item.done")
            .collect();
        assert_eq!(done[0]["item"]["type"], "reasoning");
        assert_eq!(done[1]["item"]["content"][0]["text"], "Hi");
        assert_eq!(done[2]["item"]["name"], "spawn_agent");
        assert_eq!(done[2]["item"]["namespace"], "collab");
        assert_eq!(done[2]["item"]["arguments"], "{\"a\":1}");
        assert_eq!(done[3]["item"]["type"], "custom_tool_call");
        assert_eq!(done[3]["item"]["input"], "*** Begin");
        let last = out.last().unwrap();
        assert_eq!(last["type"], "response.completed");
        assert_eq!(last["response"]["usage"]["total_tokens"], 15);
        assert_eq!(last["response"]["output"].as_array().unwrap().len(), 4);
    }
}
