//! OpenAI Chat Completions wire (`POST {base}/chat/completions`).
//!
//! Used for OpenAI, Gemini (OpenAI-compatible endpoint), Meta, DeepSeek, xAI
//! and any other compatible endpoint. Converts canonical requests to chat
//! format and chat completion chunks back into canonical Anthropic events.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Map, Value, json};

use super::{Call, UpstreamResponse, is_event_stream, post_json, sse_events};
use crate::canonical::{blocks, role, system_text, tool_result_text};
use crate::providers::Quirks;
use crate::ratelimit::{FailureKind, UpstreamFailure, classify_stream_error};
use crate::state::StateStore;

/// Dummy signature Gemini documents for function calls whose real thought
/// signature is unknown (e.g. history produced by another model).
const GEMINI_SKIP_SIGNATURE: &str = "skip_thought_signature_validator";

/// Maps canonical tool names to names valid for OpenAI-style APIs
/// (`^[a-zA-Z0-9_-]{1,64}$`) and back.
#[derive(Default, Clone)]
pub struct NameMap {
    to_upstream: HashMap<String, String>,
    to_canonical: HashMap<String, String>,
}

impl NameMap {
    pub fn upstream(&mut self, name: &str) -> String {
        if let Some(n) = self.to_upstream.get(name) {
            return n.clone();
        }
        let valid = !name.is_empty()
            && name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
        let mapped = if valid {
            name.to_string()
        } else {
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
            let hash = &crate::canonical::hash_str(name)[..8];
            let keep = clean.len().min(55);
            format!("{}_{}", &clean[..keep], hash)
        };
        self.to_upstream.insert(name.to_string(), mapped.clone());
        self.to_canonical.insert(mapped.clone(), name.to_string());
        mapped
    }

    pub fn canonical(&self, upstream: &str) -> String {
        self.to_canonical
            .get(upstream)
            .cloned()
            .unwrap_or_else(|| upstream.to_string())
    }
}

pub fn build_request(call: &Call<'_>, store: &StateStore) -> (Value, NameMap) {
    let acct = call.account;
    let quirks = acct.cfg.preset().quirks;
    let req = &call.request;
    let mut names = NameMap::default();
    let mut messages: Vec<Value> = Vec::new();

    // Claude Code embeds an Anthropic billing marker line in its system
    // prompt; it means nothing to other providers.
    let system: String = system_text(req)
        .lines()
        .filter(|l| !l.starts_with("x-anthropic-billing-header:"))
        .collect::<Vec<_>>()
        .join("\n");
    if !system.trim().is_empty() {
        messages.push(json!({"role": "system", "content": system}));
    }

    for m in req
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        match role(&m) {
            "assistant" => messages.push(convert_assistant(&m, &mut names, store, quirks)),
            "system" => {
                let text: Vec<String> = blocks(&m)
                    .iter()
                    .filter_map(|b| b.get("text").and_then(Value::as_str).map(str::to_string))
                    .collect();
                messages.push(json!({"role": "system", "content": text.join("\n")}));
            }
            _ => convert_user(&m, &mut messages),
        }
    }

    let mut body = Map::new();
    body.insert("model".into(), json!(call.model));
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), json!(true));
    body.insert("stream_options".into(), json!({"include_usage": true}));

    let mut max_tokens = req.get("max_tokens").and_then(Value::as_u64);
    if let Some(cap) = acct.cfg.max_output_tokens() {
        max_tokens = Some(max_tokens.unwrap_or(cap).min(cap));
    }
    if let Some(mt) = max_tokens {
        let key = if quirks.max_completion_tokens {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        body.insert(key.into(), json!(mt));
    }
    if !quirks.drop_sampling {
        for k in ["temperature", "top_p"] {
            if let Some(v) = req.get(k) {
                body.insert(k.into(), v.clone());
            }
        }
    }
    if let Some(stops) = req.get("stop_sequences").and_then(Value::as_array)
        && !stops.is_empty()
    {
        body.insert("stop".into(), Value::Array(stops.clone()));
    }
    if let Some(effort) = &acct.cfg.reasoning_effort {
        body.insert("reasoning_effort".into(), json!(effort));
    }

    let tools: Vec<Value> = req
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|t| {
            // Anthropic server tools (web_search_..., code_execution_...) have
            // no input_schema and can't be offered to other providers.
            let schema = t.get("input_schema")?;
            let name = t.get("name")?.as_str()?;
            let mut params = schema.clone();
            clean_schema(&mut params, quirks.gemini_schema);
            let mut f = json!({"name": names.upstream(name), "parameters": params});
            if let Some(d) = t.get("description").and_then(Value::as_str) {
                f["description"] = json!(d);
            }
            Some(json!({"type": "function", "function": f}))
        })
        .collect();
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
        if let Some(tc) = req.get("tool_choice") {
            match tc.get("type").and_then(Value::as_str) {
                Some("any") => {
                    body.insert("tool_choice".into(), json!("required"));
                }
                Some("tool") => {
                    let n = tc.get("name").and_then(Value::as_str).unwrap_or("");
                    body.insert(
                        "tool_choice".into(),
                        json!({"type": "function", "function": {"name": names.upstream(n)}}),
                    );
                }
                Some("none") => {
                    body.insert("tool_choice".into(), json!("none"));
                }
                _ => {}
            }
            if tc.get("disable_parallel_tool_use").and_then(Value::as_bool) == Some(true) {
                body.insert("parallel_tool_calls".into(), json!(false));
            }
        }
    }
    for f in &acct.cfg.drop_fields {
        body.remove(f);
    }
    (Value::Object(body), names)
}

fn convert_user(m: &Value, out: &mut Vec<Value>) {
    let mut parts: Vec<Value> = Vec::new();
    let mut tool_msgs: Vec<Value> = Vec::new();
    let mut tool_images: Vec<Value> = Vec::new();
    for b in blocks(m) {
        match b.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => parts
                .push(json!({"type": "text", "text": b.get("text").cloned().unwrap_or(json!(""))})),
            "image" => {
                if let Some(p) = image_part(&b) {
                    parts.push(p);
                }
            }
            "tool_result" => {
                let mut text = tool_result_text(&b);
                if b.get("is_error").and_then(Value::as_bool) == Some(true) {
                    text = format!("Error: {text}");
                }
                if text.is_empty() {
                    text = "(empty)".into();
                }
                tool_msgs.push(json!({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id").cloned().unwrap_or(json!("")),
                    "content": text
                }));
                if let Some(Value::Array(content)) = b.get("content") {
                    tool_images.extend(content.iter().filter_map(image_part));
                }
            }
            "document" => parts.push(json!({"type": "text", "text": "[document omitted]"})),
            _ => {}
        }
    }
    out.extend(tool_msgs);
    if !tool_images.is_empty() {
        let mut p =
            vec![json!({"type": "text", "text": "Images returned by the tool call(s) above:"})];
        p.extend(tool_images);
        out.push(json!({"role": "user", "content": p}));
    }
    if !parts.is_empty() {
        // Plain string when it's just text (widest compatibility).
        if parts.iter().all(|p| p["type"] == "text") {
            let text: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
            out.push(json!({"role": "user", "content": text.join("\n\n")}));
        } else {
            out.push(json!({"role": "user", "content": parts}));
        }
    }
}

fn image_part(b: &Value) -> Option<Value> {
    if b.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    let src = b.get("source")?;
    let url = match src.get("type").and_then(Value::as_str) {
        Some("base64") => format!(
            "data:{};base64,{}",
            src.get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png"),
            src.get("data").and_then(Value::as_str).unwrap_or("")
        ),
        Some("url") => src.get("url")?.as_str()?.to_string(),
        _ => return None,
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

fn convert_assistant(m: &Value, names: &mut NameMap, store: &StateStore, quirks: Quirks) -> Value {
    let mut text = String::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    for b in blocks(m) {
        match b.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(b.get("text").and_then(Value::as_str).unwrap_or(""));
            }
            "tool_use" => {
                let id = b
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                let args = b.get("input").cloned().unwrap_or(json!({}));
                let mut tc = json!({
                    "id": id,
                    "type": "function",
                    "function": {"name": names.upstream(name), "arguments": args.to_string()}
                });
                if quirks.gemini_thought_signatures {
                    let sig = store
                        .tool_extra(&id)
                        .and_then(|e| e.gemini_thought_signature)
                        .or_else(|| {
                            tool_calls
                                .is_empty()
                                .then(|| GEMINI_SKIP_SIGNATURE.to_string())
                        });
                    if let Some(sig) = sig {
                        tc["extra_content"] = json!({"google": {"thought_signature": sig}});
                    }
                }
                tool_calls.push(tc);
            }
            _ => {}
        }
    }
    let mut msg = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
    if !tool_calls.is_empty() {
        if quirks.echo_reasoning_content {
            let first = tool_calls[0]["id"].as_str().unwrap_or("").to_string();
            let rc = store.tool_extra(&first).and_then(|e| e.reasoning_content);
            msg["reasoning_content"] = json!(rc.unwrap_or_default());
        }
        msg["tool_calls"] = Value::Array(tool_calls);
    } else if text.is_empty() {
        msg["content"] = json!("");
    }
    msg
}

const GEMINI_UNSUPPORTED: &[&str] = &[
    "$schema",
    "$id",
    "additionalProperties",
    "propertyNames",
    "patternProperties",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "unevaluatedProperties",
    "dependentSchemas",
    "dependentRequired",
    "if",
    "then",
    "else",
    "examples",
    "default",
];

fn clean_schema(v: &mut Value, gemini: bool) {
    match v {
        Value::Object(map) => {
            map.remove("$schema");
            if gemini {
                for k in GEMINI_UNSUPPORTED {
                    map.remove(*k);
                }
            }
            if map.get("type").and_then(Value::as_str) == Some("object")
                && !map.contains_key("properties")
            {
                map.insert("properties".into(), json!({}));
            }
            for (k, child) in map.iter_mut() {
                if k == "properties" {
                    if let Value::Object(props) = child {
                        props.values_mut().for_each(|p| clean_schema(p, gemini));
                    }
                } else {
                    clean_schema(child, gemini);
                }
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| clean_schema(x, gemini)),
        _ => {}
    }
}

pub async fn send(
    http: &reqwest::Client,
    store: &Arc<StateStore>,
    call: Call<'_>,
) -> Result<UpstreamResponse, UpstreamFailure> {
    let (body, names) = build_request(&call, store);
    let acct = call.account;
    let url = format!("{}/chat/completions", acct.base_url);
    let mut req = http.post(&url);
    if let Some(key) = &acct.api_key {
        req = req.header("authorization", format!("Bearer {key}"));
    }
    for (k, v) in &acct.cfg.headers {
        req = req.header(k, v);
    }
    let resp = post_json(req, &body).await?;
    let headers: HeaderMap = resp.headers().clone();
    let model = call.model.clone();
    let quirks = acct.cfg.preset().quirks;
    let store = Arc::clone(store);

    if !is_event_stream(&resp) {
        let full: Value = resp.json().await.map_err(|e| {
            UpstreamFailure::new(
                FailureKind::Transient,
                format!("bad JSON from upstream: {e}"),
            )
        })?;
        let chunks = completion_to_chunks(&full);
        let mut conv = ChunkConverter::new(model, names, quirks, store);
        let mut out = Vec::new();
        for c in chunks {
            match conv.push(&c) {
                Ok(evs) => out.extend(evs.into_iter().map(Ok)),
                Err(e) => {
                    out.push(Err(e));
                    break;
                }
            }
        }
        out.extend(conv.finish().into_iter().map(Ok));
        return Ok(UpstreamResponse {
            events: Box::pin(futures::stream::iter(out)),
            headers,
        });
    }

    let sse = sse_events(resp.bytes_stream());
    let events = async_stream::stream! {
        futures::pin_mut!(sse);
        let mut conv = ChunkConverter::new(model, names, quirks, store);
        while let Some(item) = sse.next().await {
            let ev = match item {
                Ok(ev) => ev,
                Err(e) => { yield Err(e); return; }
            };
            if ev.data.trim() == "[DONE]" { break; }
            let Ok(chunk) = serde_json::from_str::<Value>(&ev.data) else { continue };
            match conv.push(&chunk) {
                Ok(evs) => for e in evs { yield Ok(e); },
                Err(e) => { yield Err(e); return; }
            }
        }
        for e in conv.finish() { yield Ok(e); }
    };
    Ok(UpstreamResponse {
        events: Box::pin(events),
        headers,
    })
}

/// Turn a non-streamed chat completion into equivalent stream chunks.
fn completion_to_chunks(full: &Value) -> Vec<Value> {
    if full.get("error").is_some() {
        return vec![full.clone()];
    }
    let choice = full.pointer("/choices/0").cloned().unwrap_or(json!({}));
    let msg = choice.get("message").cloned().unwrap_or(json!({}));
    let mut delta = json!({});
    for k in ["content", "reasoning_content", "reasoning"] {
        if let Some(v) = msg.get(k) {
            delta[k] = v.clone();
        }
    }
    if let Some(Value::Array(tcs)) = msg.get("tool_calls") {
        let tcs: Vec<Value> = tcs
            .iter()
            .enumerate()
            .map(|(i, tc)| {
                let mut tc = tc.clone();
                tc["index"] = json!(i);
                tc
            })
            .collect();
        delta["tool_calls"] = Value::Array(tcs);
    }
    vec![
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": choice.get("finish_reason").cloned().unwrap_or(Value::Null)}]}),
        json!({"choices": [], "usage": full.get("usage").cloned().unwrap_or(Value::Null)}),
    ]
}

#[derive(PartialEq)]
enum Open {
    None,
    Thinking,
    Text,
}

struct ToolBuf {
    id: String,
    name: String,
    args: String,
}

/// Stateful chat-chunk → canonical-event converter.
///
/// Text and reasoning are streamed as they arrive. Tool calls are buffered
/// (their argument deltas may interleave across indices) and emitted as whole
/// blocks when the message ends or other content follows them.
pub struct ChunkConverter {
    model: String,
    names: NameMap,
    quirks: Quirks,
    store: Arc<StateStore>,
    started: bool,
    index: usize,
    open: Open,
    tools: indexmap::IndexMap<u64, ToolBuf>,
    first_tool_id: Option<String>,
    reasoning: String,
    finish: Option<String>,
    usage: Value,
    saw_tool: bool,
}

impl ChunkConverter {
    pub fn new(model: String, names: NameMap, quirks: Quirks, store: Arc<StateStore>) -> Self {
        Self {
            model,
            names,
            quirks,
            store,
            started: false,
            index: 0,
            open: Open::None,
            tools: indexmap::IndexMap::new(),
            first_tool_id: None,
            reasoning: String::new(),
            finish: None,
            usage: json!({"input_tokens": 0, "output_tokens": 0}),
            saw_tool: false,
        }
    }

    fn start(&mut self, out: &mut Vec<Value>) {
        if self.started {
            return;
        }
        self.started = true;
        out.push(json!({
            "type": "message_start",
            "message": {
                "id": format!("msg_{}", uuid::Uuid::new_v4().simple()),
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}
            }
        }));
    }

    fn close(&mut self, out: &mut Vec<Value>) {
        if self.open != Open::None {
            out.push(json!({"type": "content_block_stop", "index": self.index}));
            self.index += 1;
            self.open = Open::None;
        }
    }

    fn flush_tools(&mut self, out: &mut Vec<Value>) {
        if self.tools.is_empty() {
            return;
        }
        self.close(out);
        for (_, t) in std::mem::take(&mut self.tools) {
            let name = self.names.canonical(&t.name);
            out.push(json!({"type":"content_block_start","index":self.index,"content_block":{"type":"tool_use","id":t.id,"name":name,"input":{}}}));
            let args = if t.args.trim().is_empty() {
                "{}".to_string()
            } else {
                t.args
            };
            let args = match serde_json::from_str::<Value>(&args) {
                Ok(v) if v.is_object() => args,
                Ok(v) => json!({"value": v}).to_string(),
                Err(_) => json!({"_raw_arguments": args}).to_string(),
            };
            out.push(json!({"type":"content_block_delta","index":self.index,"delta":{"type":"input_json_delta","partial_json":args}}));
            out.push(json!({"type": "content_block_stop", "index": self.index}));
            self.index += 1;
        }
    }

    pub fn push(&mut self, chunk: &Value) -> Result<Vec<Value>, UpstreamFailure> {
        if let Some(err) = chunk.get("error") {
            return Err(classify_stream_error(err));
        }
        let mut out = Vec::new();
        self.start(&mut out);
        if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
            let input = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
            let output = u
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let cached = u
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(Value::as_u64)
                .or_else(|| u.get("prompt_cache_hit_tokens").and_then(Value::as_u64))
                .unwrap_or(0);
            self.usage = json!({
                "input_tokens": input.saturating_sub(cached),
                "output_tokens": output,
                "cache_read_input_tokens": cached
            });
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return Ok(out);
        };
        let delta = choice.get("delta").cloned().unwrap_or(json!({}));

        let reasoning = delta
            .get("reasoning_content")
            .or_else(|| delta.get("reasoning"))
            .and_then(Value::as_str)
            .unwrap_or("");
        if !reasoning.is_empty() {
            self.flush_tools(&mut out);
            self.reasoning.push_str(reasoning);
            if self.open != Open::Thinking {
                self.close(&mut out);
                out.push(json!({"type":"content_block_start","index":self.index,"content_block":{"type":"thinking","thinking":"","signature":""}}));
                self.open = Open::Thinking;
            }
            out.push(json!({"type":"content_block_delta","index":self.index,"delta":{"type":"thinking_delta","thinking":reasoning}}));
        }

        if let Some(text) = delta.get("content").and_then(Value::as_str)
            && !text.is_empty()
        {
            self.flush_tools(&mut out);
            if self.open != Open::Text {
                self.close(&mut out);
                out.push(json!({"type":"content_block_start","index":self.index,"content_block":{"type":"text","text":""}}));
                self.open = Open::Text;
            }
            out.push(json!({"type":"content_block_delta","index":self.index,"delta":{"type":"text_delta","text":text}}));
        }

        if let Some(Value::Array(tcs)) = delta.get("tool_calls") {
            for tc in tcs {
                let up_idx = tc
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(self.tools.len() as u64);
                if !self.tools.contains_key(&up_idx) {
                    let id = tc
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("call_{}", uuid::Uuid::new_v4().simple()));
                    if self.first_tool_id.is_none() {
                        self.first_tool_id = Some(id.clone());
                    }
                    self.saw_tool = true;
                    self.tools.insert(
                        up_idx,
                        ToolBuf {
                            id,
                            name: String::new(),
                            args: String::new(),
                        },
                    );
                }
                let t = self.tools.get_mut(&up_idx).unwrap();
                if let Some(n) = tc.pointer("/function/name").and_then(Value::as_str)
                    && t.name.is_empty()
                {
                    t.name = n.to_string();
                }
                if let Some(a) = tc.pointer("/function/arguments").and_then(Value::as_str) {
                    t.args.push_str(a);
                }
                if let Some(sig) = tc
                    .pointer("/extra_content/google/thought_signature")
                    .and_then(Value::as_str)
                    && self.quirks.gemini_thought_signatures
                {
                    let sig = sig.to_string();
                    let id = t.id.clone();
                    self.store
                        .put_tool_extra(&id, |e| e.gemini_thought_signature = Some(sig));
                }
            }
        }

        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish = Some(fr.to_string());
        }
        Ok(out)
    }

    pub fn finish(&mut self) -> Vec<Value> {
        let mut out = Vec::new();
        self.start(&mut out);
        self.flush_tools(&mut out);
        self.close(&mut out);
        if self.quirks.echo_reasoning_content
            && !self.reasoning.is_empty()
            && let Some(id) = &self.first_tool_id
        {
            let rc = self.reasoning.clone();
            self.store
                .put_tool_extra(id, |e| e.reasoning_content = Some(rc));
        }
        let stop = match self.finish.as_deref() {
            _ if self.saw_tool => "tool_use",
            Some("length") => "max_tokens",
            _ => "end_turn",
        };
        out.push(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop, "stop_sequence": null},
            "usage": self.usage
        }));
        out.push(json!({"type": "message_stop"}));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Accumulator;
    use crate::config::AccountConfig;
    use crate::providers::ProviderKind;
    use crate::upstream::Account;

    fn account(kind: ProviderKind) -> Account {
        Account::from_config(AccountConfig {
            id: "a".into(),
            provider: kind,
            model: Some("m".into()),
            compact_model: None,
            fast_model: None,
            api_key: Some("k".into()),
            api_key_env: None,
            base_url: Some("http://x/v1".into()),
            auth: None,
            context_window: None,
            max_output_tokens: None,
            reasoning_effort: Some("low".into()),
            headers: Default::default(),
            drop_fields: vec![],
            enabled: true,
        })
    }

    #[test]
    fn converts_request() {
        let store = StateStore::open(None);
        let acct = account(ProviderKind::Gemini);
        let h = HeaderMap::new();
        let long_name = "mcp__some_server__".to_string() + &"x".repeat(80);
        let call = Call {
            account: &acct,
            model: "gem".into(),
            client_headers: &h,
            request: json!({
                "system": [{"type":"text","text":"be nice"}],
                "max_tokens": 32000,
                "messages": [
                    {"role":"user","content":[{"type":"text","text":"hi"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAA"}}]},
                    {"role":"assistant","content":[{"type":"thinking","thinking":"hmm","signature":""},{"type":"text","text":"calling"},{"type":"tool_use","id":"t1","name":long_name,"input":{"a":1}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"res"}],"is_error":true},{"type":"text","text":"next"}]}
                ],
                "tools": [
                    {"name": long_name, "description":"d", "input_schema":{"$schema":"x","type":"object","additionalProperties":false,"properties":{"a":{"type":"number","exclusiveMinimum":0}}}},
                    {"type":"web_search_20250305","name":"web_search"}
                ],
                "tool_choice": {"type":"any"}
            }),
        };
        let (body, names) = build_request(&call, &store);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(
            msgs[1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAA"
        );
        assert_eq!(msgs[2]["content"], "calling");
        let up_name = msgs[2]["tool_calls"][0]["function"]["name"]
            .as_str()
            .unwrap();
        assert!(up_name.len() <= 64);
        assert_eq!(names.canonical(up_name), long_name);
        assert_eq!(
            msgs[2]["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            GEMINI_SKIP_SIGNATURE
        );
        assert_eq!(msgs[3]["role"], "tool");
        assert_eq!(msgs[3]["content"], "Error: res");
        assert_eq!(msgs[4]["content"], "next");
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert!(
            tools[0]["function"]["parameters"]
                .get("additionalProperties")
                .is_none()
        );
        assert!(
            tools[0]["function"]["parameters"]["properties"]["a"]
                .get("exclusiveMinimum")
                .is_none()
        );
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["max_tokens"], 32000);
    }

    #[test]
    fn converts_stream_with_reasoning_and_tools() {
        let store = StateStore::open(None);
        let quirks = ProviderKind::Deepseek.preset().quirks;
        let mut conv =
            ChunkConverter::new("m".into(), NameMap::default(), quirks, Arc::clone(&store));
        let chunks = [
            json!({"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think "}}]}),
            json!({"choices":[{"index":0,"delta":{"reasoning_content":"more"}}]}),
            json!({"choices":[{"index":0,"delta":{"content":"Hello"}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"f","arguments":"{\"x\""}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":2}"}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"c2","type":"function","function":{"name":"g","arguments":"{}"}}]}}]}),
            json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":""}}]}}]}),
            json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,"prompt_cache_hit_tokens":40}}),
        ];
        let mut acc = Accumulator::default();
        for c in &chunks {
            for e in conv.push(c).unwrap() {
                acc.push(&e);
            }
        }
        for e in conv.finish() {
            acc.push(&e);
        }
        let m = acc.finish();
        assert_eq!(m["content"][0]["type"], "thinking");
        assert_eq!(m["content"][0]["thinking"], "think more");
        assert_eq!(m["content"][1]["text"], "Hello");
        assert_eq!(m["content"][2]["input"]["x"], 2);
        assert_eq!(m["content"][3]["name"], "g");
        assert_eq!(m["stop_reason"], "tool_use");
        assert_eq!(m["usage"]["input_tokens"], 60);
        assert_eq!(m["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(
            store.tool_extra("c1").unwrap().reasoning_content.as_deref(),
            Some("think more")
        );
    }

    #[test]
    fn stream_error_chunk() {
        let store = StateStore::open(None);
        let mut conv =
            ChunkConverter::new("m".into(), NameMap::default(), Quirks::default(), store);
        let err = conv
            .push(&json!({"error":{"message":"Rate limit reached","type":"rate_limit_error"}}))
            .unwrap_err();
        assert_eq!(err.kind, FailureKind::RateLimited);
    }
}
