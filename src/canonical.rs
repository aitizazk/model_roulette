//! The canonical request/response format.
//!
//! Internally everything is an Anthropic Messages API request (a JSON object)
//! and responses are streams of Anthropic SSE events (`message_start`,
//! `content_block_start`, ...). Anthropic's format is the richest of the
//! supported wires (typed content blocks, thinking, tool use/results), so it
//! loses the least when translating. Front ends convert into it; upstream
//! adapters convert out of it.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

/// Stable 64-bit FNV-1a hash, hex encoded. Stable across runs and builds,
/// which matters because hashes are persisted.
pub fn hash_str(s: &str) -> String {
    format!("{:016x}", fnv1a(s.as_bytes(), 0xcbf29ce484222325))
}

fn fnv1a(bytes: &[u8], mut h: u64) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Rough token estimate (≈4 chars/token on serialized JSON, images at a flat
/// cost). Good enough for thresholds; never used for billing.
pub fn estimate_tokens(v: &Value) -> u64 {
    fn walk(v: &Value, acc: &mut u64) {
        match v {
            Value::String(s) => *acc += s.len() as u64,
            Value::Array(a) => a.iter().for_each(|x| walk(x, acc)),
            Value::Object(m) => {
                if m.get("type").and_then(Value::as_str) == Some("image") {
                    *acc += 1600 * 4;
                    return;
                }
                for (k, x) in m {
                    *acc += k.len() as u64 + 2;
                    walk(x, acc);
                }
            }
            other => *acc += other.to_string().len() as u64,
        }
    }
    let mut chars = 0;
    walk(v, &mut chars);
    chars / 4 + 1
}

pub fn estimate_request_tokens(req: &Value) -> u64 {
    let mut t = 0;
    for k in ["system", "messages", "tools"] {
        if let Some(v) = req.get(k) {
            t += estimate_tokens(v);
        }
    }
    t
}

/// Content of a message as a list of blocks (strings become a text block).
pub fn blocks(msg: &Value) -> Vec<Value> {
    match msg.get("content") {
        Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
        Some(Value::Array(a)) => a.clone(),
        _ => vec![],
    }
}

pub fn role(msg: &Value) -> &str {
    msg.get("role").and_then(Value::as_str).unwrap_or("")
}

/// System prompt as plain text.
pub fn system_text(req: &Value) -> String {
    match req.get("system") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// Text of a tool_result's content.
pub fn tool_result_text(block: &Value) -> String {
    match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) => a
            .iter()
            .map(|b| match b.get("type").and_then(Value::as_str) {
                Some("text") => b
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                Some("image") => "[image]".to_string(),
                Some(other) => format!("[{other}]"),
                None => String::new(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// A normalized, provider-independent digest of one message, used to
/// recognise a conversation prefix across requests. Ignores thinking blocks,
/// cache_control markers, tool result bodies (harnesses rewrite those, e.g.
/// Claude Code's micro-compaction) and whitespace differences.
pub fn message_digest(msg: &Value) -> String {
    let mut out = String::new();
    out.push_str(role(msg));
    out.push('|');
    for b in blocks(msg) {
        match b.get("type").and_then(Value::as_str).unwrap_or("") {
            "text" => {
                let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                let t: String = t.split_whitespace().collect::<Vec<_>>().join(" ");
                if t.is_empty() {
                    continue;
                }
                out.push_str("t:");
                out.push_str(&hash_str(&t));
            }
            "tool_use" => {
                out.push_str("u:");
                out.push_str(b.get("id").and_then(Value::as_str).unwrap_or(""));
                out.push(':');
                out.push_str(b.get("name").and_then(Value::as_str).unwrap_or(""));
            }
            "tool_result" => {
                out.push_str("r:");
                out.push_str(b.get("tool_use_id").and_then(Value::as_str).unwrap_or(""));
            }
            "image" => out.push('i'),
            "thinking" | "redacted_thinking" => continue,
            other => {
                out.push_str("o:");
                out.push_str(other);
            }
        }
        out.push(';');
    }
    out
}

/// Hash of the first `n` conversation turns (user/assistant messages;
/// mid-conversation system messages are ignored because harnesses inject and
/// regenerate them freely).
pub fn prefix_hash(messages: &[Value], n: usize) -> String {
    let mut h = 0xcbf29ce484222325u64;
    for m in messages.iter().filter(|m| role(m) != "system").take(n) {
        h = fnv1a(message_digest(m).as_bytes(), h);
        h = fnv1a(b"\x1e", h);
    }
    format!("{h:016x}-{n}")
}

/// Raw index just after the `n`-th user/assistant message.
pub fn turn_cut_index(messages: &[Value], n: usize) -> Option<usize> {
    if n == 0 {
        return Some(0);
    }
    let mut count = 0;
    for (i, m) in messages.iter().enumerate() {
        if role(m) != "system" {
            count += 1;
            if count == n {
                return Some(i + 1);
            }
        }
    }
    None
}

/// Number of user/assistant messages in `messages`.
pub fn turn_count(messages: &[Value]) -> usize {
    messages.iter().filter(|m| role(m) != "system").count()
}

/// Fingerprint identifying a conversation: its first message.
pub fn conversation_fingerprint(messages: &[Value]) -> String {
    match messages.first() {
        Some(m) => hash_str(&message_digest(m)),
        None => "empty".to_string(),
    }
}

/// Make a message list structurally valid for strict providers:
/// - consecutive messages with the same role are merged
/// - every tool_use is answered by a tool_result in the next user message
///   (a placeholder is inserted for interrupted calls)
/// - tool_results without a matching tool_use become plain text
/// - tool_result blocks come first in a user message
/// - the conversation starts with a user message and empty messages are dropped
pub fn sanitize_messages(messages: Vec<Value>) -> Vec<Value> {
    // 1. merge same-role neighbours, normalise content to block arrays.
    let mut merged: Vec<Value> = Vec::new();
    for m in inline_system_messages(messages) {
        let r = role(&m).to_string();
        if r != "user" && r != "assistant" {
            continue;
        }
        let bl: Vec<Value> = blocks(&m)
            .into_iter()
            .filter(|b| {
                !(b.get("type").and_then(Value::as_str) == Some("text")
                    && b.get("text")
                        .and_then(Value::as_str)
                        .map(|t| t.trim().is_empty())
                        .unwrap_or(true))
            })
            .collect();
        if bl.is_empty() {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && role(last) == r
            && let Some(arr) = last.get_mut("content").and_then(Value::as_array_mut)
        {
            arr.extend(bl);
            continue;
        }
        merged.push(json!({"role": r, "content": bl}));
    }

    // 2. pair tool_use / tool_result.
    let mut out: Vec<Value> = Vec::with_capacity(merged.len() + 2);
    let mut i = 0;
    while i < merged.len() {
        let msg = merged[i].clone();
        if role(&msg) == "assistant" {
            let ids: Vec<String> = blocks(&msg)
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
                .filter_map(|b| b.get("id").and_then(Value::as_str).map(str::to_string))
                .collect();
            out.push(msg);
            if !ids.is_empty() {
                let next_is_user = merged
                    .get(i + 1)
                    .map(|m| role(m) == "user")
                    .unwrap_or(false);
                let mut next = if next_is_user {
                    i += 1;
                    merged[i].clone()
                } else {
                    json!({"role": "user", "content": []})
                };
                let present: HashSet<String> = blocks(&next)
                    .iter()
                    .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
                    .filter_map(|b| {
                        b.get("tool_use_id")
                            .and_then(Value::as_str)
                            .map(str::to_string)
                    })
                    .collect();
                let mut results: Vec<Value> = Vec::new();
                let mut rest: Vec<Value> = Vec::new();
                for b in blocks(&next) {
                    if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                        let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                        if ids.iter().any(|x| x == id) {
                            results.push(b);
                        } else {
                            rest.push(orphan_result_as_text(&b));
                        }
                    } else {
                        rest.push(b);
                    }
                }
                for id in &ids {
                    if !present.contains(id) {
                        results.push(json!({
                            "type": "tool_result",
                            "tool_use_id": id,
                            "content": "[no result: the tool call was interrupted]",
                            "is_error": true
                        }));
                    }
                }
                // Order results like the calls.
                results.sort_by_key(|b| {
                    let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                    ids.iter().position(|x| x == id).unwrap_or(usize::MAX)
                });
                results.extend(rest);
                next["content"] = Value::Array(results);
                out.push(next);
            }
        } else {
            // user message not following a tool-calling assistant turn
            let bl: Vec<Value> = blocks(&msg)
                .into_iter()
                .map(|b| {
                    if b.get("type").and_then(Value::as_str) == Some("tool_result") {
                        orphan_result_as_text(&b)
                    } else {
                        b
                    }
                })
                .collect();
            out.push(json!({"role": "user", "content": bl}));
        }
        i += 1;
    }

    // 3. merge again (step 2 may have created neighbours) and fix the start.
    let mut final_msgs: Vec<Value> = Vec::with_capacity(out.len());
    for m in out {
        if let Some(last) = final_msgs.last_mut()
            && role(last) == role(&m)
        {
            let extra = blocks(&m);
            if let Some(arr) = last.get_mut("content").and_then(Value::as_array_mut) {
                arr.extend(extra);
            }
            continue;
        }
        final_msgs.push(m);
    }
    if final_msgs
        .first()
        .map(|m| role(m) != "user")
        .unwrap_or(false)
    {
        final_msgs.insert(0, json!({"role": "user", "content": [{"type": "text", "text": "(continuing an earlier conversation)"}]}));
    }
    final_msgs
}

/// Claude Code sends mid-conversation `role: "system"` messages (an Anthropic
/// beta). Providers that don't support them get the text as a
/// `<system-reminder>` block in a user message instead.
pub fn inline_system_messages(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|m| {
            if role(&m) != "system" {
                return m;
            }
            let text: Vec<String> = blocks(&m)
                .iter()
                .filter_map(|b| b.get("text").and_then(Value::as_str).map(str::to_string))
                .collect();
            json!({"role": "user", "content": [{"type": "text", "text": format!("<system-reminder>\n{}\n</system-reminder>", text.join("\n"))}]})
        })
        .collect()
}

/// The last message that isn't a mid-conversation system message.
pub fn last_turn(messages: &[Value]) -> Option<&Value> {
    messages.iter().rev().find(|m| role(m) != "system")
}

fn orphan_result_as_text(b: &Value) -> Value {
    json!({"type": "text", "text": format!("[tool result]\n{}", tool_result_text(b))})
}

/// Accumulates canonical (Anthropic) stream events into a complete message.
#[derive(Default)]
pub struct Accumulator {
    pub message: Map<String, Value>,
    blocks: Vec<Value>,
    partial_json: Vec<String>,
}

impl Accumulator {
    pub fn push(&mut self, ev: &Value) {
        match ev.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                if let Some(m) = ev.get("message").and_then(Value::as_object) {
                    self.message = m.clone();
                }
            }
            "content_block_start" => {
                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let block = ev.get("content_block").cloned().unwrap_or(json!({}));
                while self.blocks.len() <= idx {
                    self.blocks.push(Value::Null);
                    self.partial_json.push(String::new());
                }
                self.blocks[idx] = block;
            }
            "content_block_delta" => {
                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                let Some(block) = self.blocks.get_mut(idx) else {
                    return;
                };
                let d = ev.get("delta").cloned().unwrap_or(json!({}));
                match d.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text_delta" => append(block, "text", d.get("text")),
                    "thinking_delta" => append(block, "thinking", d.get("thinking")),
                    "signature_delta" => append(block, "signature", d.get("signature")),
                    "input_json_delta" => {
                        if let Some(s) = d.get("partial_json").and_then(Value::as_str) {
                            self.partial_json[idx].push_str(s);
                        }
                    }
                    "citations_delta" => {
                        if let Some(c) = d.get("citation")
                            && let Some(obj) = block.as_object_mut()
                            && let Some(a) = obj
                                .entry("citations")
                                .or_insert_with(|| json!([]))
                                .as_array_mut()
                        {
                            a.push(c.clone())
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                if let (Some(block), Some(pj)) =
                    (self.blocks.get_mut(idx), self.partial_json.get(idx))
                    && !pj.is_empty()
                {
                    block["input"] = serde_json::from_str(pj).unwrap_or_else(|_| json!({}));
                }
            }
            "message_delta" => {
                if let Some(d) = ev.get("delta").and_then(Value::as_object) {
                    for (k, v) in d {
                        self.message.insert(k.clone(), v.clone());
                    }
                }
                if let Some(u) = ev.get("usage").and_then(Value::as_object) {
                    let usage = self.message.entry("usage").or_insert_with(|| json!({}));
                    if let Some(obj) = usage.as_object_mut() {
                        for (k, v) in u {
                            if !v.is_null() {
                                obj.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn finish(mut self) -> Value {
        let content: Vec<Value> = self.blocks.into_iter().filter(|b| !b.is_null()).collect();
        self.message.insert("content".into(), Value::Array(content));
        self.message.entry("type").or_insert(json!("message"));
        self.message.entry("role").or_insert(json!("assistant"));
        Value::Object(self.message)
    }

    /// Concatenated text of the accumulated message so far.
    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect()
    }
}

fn append(block: &mut Value, key: &str, s: Option<&Value>) {
    let Some(s) = s.and_then(Value::as_str) else {
        return;
    };
    let cur = block
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    block[key] = Value::String(cur + s);
}

/// Is this event "real output" (as opposed to preamble)? Once real output has
/// been forwarded to the client we can no longer transparently fail over.
pub fn is_content_event(ev: &Value) -> bool {
    matches!(
        ev.get("type").and_then(Value::as_str),
        Some("content_block_start")
            | Some("content_block_delta")
            | Some("message_delta")
            | Some("message_stop")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_ignores_thinking_and_cache_control() {
        let a = json!({"role":"assistant","content":[{"type":"thinking","thinking":"x","signature":"s"},{"type":"text","text":"hello  world","cache_control":{"type":"ephemeral"}}]});
        let b = json!({"role":"assistant","content":[{"type":"text","text":"hello world"}]});
        assert_eq!(message_digest(&a), message_digest(&b));
        let c = json!({"role":"assistant","content":"hello world"});
        assert_eq!(message_digest(&b), message_digest(&c));
    }

    #[test]
    fn sanitize_pairs_tools() {
        let msgs = vec![
            json!({"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"x","input":{}}]}),
            json!({"role":"user","content":[{"type":"text","text":"hi"}]}),
            json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"zz","content":"orphan"}]}),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(role(&out[0]), "user");
        assert_eq!(role(&out[1]), "assistant");
        let last = blocks(&out[2]);
        assert_eq!(last[0]["type"], "tool_result");
        assert_eq!(last[0]["tool_use_id"], "t1");
        assert_eq!(last[1]["type"], "text");
        assert!(last[2]["text"].as_str().unwrap().contains("orphan"));
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn sanitize_inlines_system_messages() {
        let msgs = vec![
            json!({"role":"user","content":"hi"}),
            json!({"role":"system","content":[{"type":"text","text":"env info"}]}),
        ];
        let out = sanitize_messages(msgs);
        assert_eq!(out.len(), 1);
        assert!(
            blocks(&out[0])[1]["text"]
                .as_str()
                .unwrap()
                .contains("<system-reminder>\nenv info")
        );
        let msgs = vec![
            json!({"role":"user","content":"a"}),
            json!({"role":"system","content":"b"}),
        ];
        assert_eq!(role(last_turn(&msgs).unwrap()), "user");
    }

    #[test]
    fn accumulator_builds_message() {
        let evs = [
            json!({"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"x","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}),
            json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}),
            json!({"type":"content_block_stop","index":0}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"t","name":"f","input":{}}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}),
            json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"1}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":7}}),
        ];
        let mut acc = Accumulator::default();
        evs.iter().for_each(|e| acc.push(e));
        let m = acc.finish();
        assert_eq!(m["content"][0]["text"], "Hello");
        assert_eq!(m["content"][1]["input"]["a"], 1);
        assert_eq!(m["stop_reason"], "tool_use");
        assert_eq!(m["usage"]["output_tokens"], 7);
        assert_eq!(m["usage"]["input_tokens"], 5);
    }
}
