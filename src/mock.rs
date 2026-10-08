//! A scriptable fake upstream that speaks both the Anthropic Messages and the
//! OpenAI Chat Completions wire. Used by the integration tests and handy for
//! trying model-roulette (and real harnesses) without spending tokens.
//!
//! Behaviour is selected by the API key the proxy sends, e.g.
//! `mock-a?rl_after=2&retry_after=30`:
//!
//! * `rl_after=N`      – after N requests, answer 429 (rate limited)
//! * `retry_after=S`   – send `retry-after: S` with 429s
//! * `quota`           – always fail with an "out of credits" error
//! * `fail=STATUS`     – always fail with that HTTP status
//! * `reasoning`       – (chat) stream `reasoning_content` before the answer
//! * `thought_sig`     – (chat) attach Gemini thought signatures to tool calls
//!   and reject histories whose tool calls lack one
//! * `echo_reasoning`  – (chat) reject assistant tool-call messages without
//!   `reasoning_content` (DeepSeek thinking mode)
//! * `midstream_error` – (anthropic) send an overloaded error after the first delta
//! * `ctx_limit=N`     – reject request bodies longer than N bytes as "prompt is too long"
//!
//! Replies: a summarizer prompt gets `MOCK-SUMMARY(...)`; a message containing
//! `CALL_TOOL <name> <json>` makes the model call that tool; a tool result gets
//! an acknowledgement; anything else gets an echo-style reply.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use serde_json::{Value, json};

#[derive(Default)]
pub struct MockState {
    pub counts: Mutex<HashMap<String, u64>>,
    pub requests: Mutex<Vec<Value>>,
}

#[derive(Debug, Default, Clone)]
struct Behaviour {
    name: String,
    rl_after: Option<u64>,
    retry_after: Option<u64>,
    quota: bool,
    fail: Option<u16>,
    reasoning: bool,
    thought_sig: bool,
    echo_reasoning: bool,
    midstream_error: bool,
    ctx_limit: Option<usize>,
}

fn behaviour(headers: &HeaderMap) -> Behaviour {
    let key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer ").map(str::to_string))
        })
        .unwrap_or_else(|| "anonymous".into());
    let (name, opts) = key.split_once('?').unwrap_or((&key, ""));
    let mut b = Behaviour {
        name: name.to_string(),
        ..Default::default()
    };
    for kv in opts.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
        match k {
            "rl_after" => b.rl_after = v.parse().ok(),
            "retry_after" => b.retry_after = v.parse().ok(),
            "quota" => b.quota = true,
            "fail" => b.fail = v.parse().ok(),
            "reasoning" => b.reasoning = true,
            "thought_sig" => b.thought_sig = true,
            "echo_reasoning" => b.echo_reasoning = true,
            "midstream_error" => b.midstream_error = true,
            "ctx_limit" => b.ctx_limit = v.parse().ok(),
            _ => {}
        }
    }
    b
}

pub fn router(state: Arc<MockState>) -> Router {
    Router::new()
        .route("/v1/messages", post(anthropic))
        .route("/v1/chat/completions", post(chat))
        .route("/chat/completions", post(chat))
        .route("/_mock/requests", get(list_requests).delete(clear_requests))
        .with_state(state)
}

pub async fn spawn(addr: &str) -> anyhow::Result<(SocketAddr, Arc<MockState>)> {
    let state = Arc::new(MockState::default());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    let app = router(Arc::clone(&state));
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    Ok((local, state))
}

async fn list_requests(State(st): State<Arc<MockState>>) -> Response {
    axum::Json(Value::Array(st.requests.lock().unwrap().clone())).into_response()
}

async fn clear_requests(State(st): State<Arc<MockState>>) -> Response {
    st.requests.lock().unwrap().clear();
    st.counts.lock().unwrap().clear();
    StatusCode::NO_CONTENT.into_response()
}

/// Shared gatekeeping. Returns the 1-based request number for this account or
/// an error response.
#[allow(clippy::result_large_err)]
fn admit(st: &MockState, b: &Behaviour, wire: &str, body: &Value) -> Result<u64, Response> {
    let n = {
        let mut c = st.counts.lock().unwrap();
        let e = c.entry(b.name.clone()).or_insert(0);
        *e += 1;
        *e
    };
    st.requests
        .lock()
        .unwrap()
        .push(json!({"account": b.name, "wire": wire, "n": n, "body": body}));
    if let Some(status) = b.fail {
        let code = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        return Err((code, axum::Json(json!({"error": {"type": "api_error", "message": format!("mock failure {status}")}}))).into_response());
    }
    if b.quota {
        return Err(match wire {
            "anthropic" => (StatusCode::BAD_REQUEST, axum::Json(json!({"type":"error","error":{"type":"invalid_request_error",
                "message":"Your credit balance is too low to access the Anthropic API."}}))).into_response(),
            _ => (StatusCode::TOO_MANY_REQUESTS, axum::Json(json!({"error":{"type":"insufficient_quota","code":"insufficient_quota",
                "message":"You exceeded your current quota, please check your plan and billing details."}}))).into_response(),
        });
    }
    if let Some(limit) = b.ctx_limit {
        let len = body.to_string().len();
        if len > limit {
            return Err((
                StatusCode::BAD_REQUEST,
                axum::Json(
                    json!({"type":"error","error":{"type":"invalid_request_error",
                "message":format!("prompt is too long: {len} > {limit} maximum")}}),
                ),
            )
                .into_response());
        }
    }
    if let Some(limit) = b.rl_after
        && n > limit
    {
        let mut r = (
            StatusCode::TOO_MANY_REQUESTS,
            axum::Json(json!({"type":"error","error":{"type":"rate_limit_error",
                "message":format!("mock rate limit for {}", b.name)}})),
        )
            .into_response();
        if let Some(s) = b.retry_after {
            r.headers_mut().insert("retry-after", HeaderValue::from(s));
        }
        return Err(r);
    }
    Ok(n)
}

enum Reply {
    Text(String),
    Tool { name: String, input: Value },
}

/// Decide the reply from a normalized view of the conversation.
fn decide(
    b: &Behaviour,
    model: &str,
    system: &str,
    last_user_text: Option<String>,
    last_is_tool_result: Option<String>,
    tools: &[String],
    n_messages: usize,
) -> Reply {
    if system.contains("context compactor") {
        return Reply::Text(format!(
            "MOCK-SUMMARY({}): the user is working on a task; {} messages summarized.",
            b.name, n_messages
        ));
    }
    if let Some(res) = last_is_tool_result {
        let short: String = res.chars().take(120).collect();
        return Reply::Text(format!(
            "[{}] tool result received: {}",
            b.name,
            short.trim()
        ));
    }
    let text = last_user_text.unwrap_or_default();
    if let Some(pos) = text.find("CALL_TOOL ") {
        let rest = &text[pos + 10..];
        let (name, json_part) = rest.split_once(' ').unwrap_or((rest, "{}"));
        let name = name.trim().to_string();
        if tools.contains(&name) {
            // Parse the first JSON value; anything after it is ignored.
            let input = serde_json::Deserializer::from_str(json_part.trim_start())
                .into_iter::<Value>()
                .next()
                .and_then(Result::ok)
                .unwrap_or(json!({}));
            return Reply::Tool { name, input };
        }
    }
    let short: String = text.chars().take(60).collect();
    Reply::Text(format!(
        "[{}/{}] reply to: {} (messages={})",
        b.name,
        model,
        short.trim(),
        n_messages
    ))
}

fn sse(events: Vec<(Option<&str>, Value)>) -> Response {
    let mut s = String::new();
    for (ev, data) in events {
        if let Some(e) = ev {
            s.push_str(&format!("event: {e}\n"));
        }
        s.push_str(&format!("data: {data}\n\n"));
    }
    let mut r = Response::new(Body::from(s));
    r.headers_mut().insert(
        "content-type",
        HeaderValue::from_static("text/event-stream"),
    );
    r
}

async fn anthropic(State(st): State<Arc<MockState>>, headers: HeaderMap, body: Bytes) -> Response {
    let b = behaviour(&headers);
    let req: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
    let n = match admit(&st, &b, "anthropic", &req) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    // Strictness checks a real API would do.
    if msgs.first().map(|m| m["role"] != "user").unwrap_or(true) {
        return (StatusCode::BAD_REQUEST, axum::Json(json!({"type":"error","error":{"type":"invalid_request_error","message":"first message must use the user role"}}))).into_response();
    }
    let system = crate::canonical::system_text(&req);
    let last = msgs
        .iter()
        .rev()
        .find(|m| m["role"] != "system")
        .cloned()
        .unwrap_or(json!({}));
    let blocks = crate::canonical::blocks(&last);
    let tool_res = blocks
        .iter()
        .find(|x| x["type"] == "tool_result")
        .map(crate::canonical::tool_result_text);
    let text = blocks
        .iter()
        .filter_map(|x| x["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let tools: Vec<String> = req["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|t| t["name"].as_str().map(str::to_string))
        .collect();
    let model = req["model"].as_str().unwrap_or("").to_string();
    let reply = decide(
        &b,
        &model,
        &system,
        Some(text),
        tool_res,
        &tools,
        msgs.len(),
    );
    let input_tokens = crate::canonical::estimate_request_tokens(&req);

    let mut evs: Vec<(Option<&str>, Value)> = vec![(
        Some("message_start"),
        json!({"type":"message_start","message":{
        "id": format!("msg_mock_{}_{n}", b.name), "type":"message","role":"assistant","model":model,"content":[],
        "stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":input_tokens,"output_tokens":1}}}),
    )];
    let stop = match &reply {
        Reply::Text(t) => {
            evs.push((Some("content_block_start"), json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}})));
            evs.push((Some("content_block_delta"), json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"mock thinking"}})));
            evs.push((Some("content_block_delta"), json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":format!("sig-{}-{n}", b.name)}})));
            evs.push((
                Some("content_block_stop"),
                json!({"type":"content_block_stop","index":0}),
            ));
            evs.push((Some("content_block_start"), json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}})));
            let (a, c) = t.split_at(t.len() / 2);
            evs.push((Some("content_block_delta"), json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":a}})));
            if b.midstream_error {
                evs.push((Some("error"), json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}})));
                return sse(evs);
            }
            evs.push((Some("content_block_delta"), json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":c}})));
            evs.push((
                Some("content_block_stop"),
                json!({"type":"content_block_stop","index":1}),
            ));
            "end_turn"
        }
        Reply::Tool { name, input } => {
            evs.push((Some("content_block_start"), json!({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":format!("toolu_mock_{n}"),"name":name,"input":{}}})));
            evs.push((Some("content_block_delta"), json!({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":input.to_string()}})));
            evs.push((
                Some("content_block_stop"),
                json!({"type":"content_block_stop","index":0}),
            ));
            "tool_use"
        }
    };
    evs.push((Some("message_delta"), json!({"type":"message_delta","delta":{"stop_reason":stop,"stop_sequence":null},"usage":{"output_tokens":12}})));
    evs.push((Some("message_stop"), json!({"type":"message_stop"})));
    if req["stream"].as_bool() == Some(true) {
        sse(evs)
    } else {
        let mut acc = crate::canonical::Accumulator::default();
        evs.iter().for_each(|(_, e)| acc.push(e));
        axum::Json(acc.finish()).into_response()
    }
}

async fn chat(State(st): State<Arc<MockState>>, headers: HeaderMap, body: Bytes) -> Response {
    let b = behaviour(&headers);
    let req: Value = serde_json::from_slice(&body).unwrap_or(json!({}));
    let n = match admit(&st, &b, "openai", &req) {
        Ok(n) => n,
        Err(r) => return r,
    };
    let msgs = req["messages"].as_array().cloned().unwrap_or_default();
    let bad = |m: &str| {
        (
            StatusCode::BAD_REQUEST,
            axum::Json(json!({"error":{"type":"invalid_request_error","message":m}})),
        )
            .into_response()
    };
    for m in &msgs {
        if m["role"] == "assistant" && m.get("tool_calls").is_some() {
            if b.thought_sig {
                let first = &m["tool_calls"][0];
                if first
                    .pointer("/extra_content/google/thought_signature")
                    .is_none()
                {
                    return bad("Function call is missing a thought_signature");
                }
            }
            if b.echo_reasoning && m.get("reasoning_content").is_none() {
                return bad("Missing reasoning_content field in the assistant message");
            }
        }
    }
    let system = msgs
        .iter()
        .filter(|m| m["role"] == "system")
        .filter_map(|m| m["content"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let last = msgs
        .iter()
        .rev()
        .find(|m| m["role"] != "system")
        .cloned()
        .unwrap_or(json!({}));
    let tool_res =
        (last["role"] == "tool").then(|| last["content"].as_str().unwrap_or("").to_string());
    let text = match &last["content"] {
        Value::String(s) => s.clone(),
        Value::Array(p) => p
            .iter()
            .filter_map(|x| x["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    };
    let tools: Vec<String> = req["tools"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .filter_map(|t| {
            t.pointer("/function/name")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .collect();
    let model = req["model"].as_str().unwrap_or("").to_string();
    let reply = decide(
        &b,
        &model,
        &system,
        Some(text),
        tool_res,
        &tools,
        msgs.len(),
    );
    let id = format!("chatcmpl-mock-{}-{n}", b.name);
    let chunk = |delta: Value, finish: Value| json!({"id":id,"object":"chat.completion.chunk","model":model,"choices":[{"index":0,"delta":delta,"finish_reason":finish}]});
    let mut evs: Vec<(Option<&str>, Value)> = Vec::new();
    if b.reasoning {
        evs.push((
            None,
            chunk(
                json!({"role":"assistant","reasoning_content":"mock "}),
                Value::Null,
            ),
        ));
        evs.push((
            None,
            chunk(json!({"reasoning_content":"reasoning"}), Value::Null),
        ));
    }
    let finish = match &reply {
        Reply::Text(t) => {
            let (a, c) = t.split_at(t.len() / 2);
            evs.push((
                None,
                chunk(json!({"role":"assistant","content":a}), Value::Null),
            ));
            evs.push((None, chunk(json!({"content":c}), Value::Null)));
            "stop"
        }
        Reply::Tool { name, input } => {
            let mut tc = json!({"index":0,"id":format!("call_mock_{n}"),"type":"function","function":{"name":name,"arguments":""}});
            if b.thought_sig {
                tc["extra_content"] = json!({"google":{"thought_signature":format!("gsig-{n}")}});
            }
            evs.push((
                None,
                chunk(json!({"role":"assistant","tool_calls":[tc]}), Value::Null),
            ));
            let args = input.to_string();
            let (a, c) = args.split_at(args.len() / 2);
            evs.push((
                None,
                chunk(
                    json!({"tool_calls":[{"index":0,"function":{"arguments":a}}]}),
                    Value::Null,
                ),
            ));
            evs.push((
                None,
                chunk(
                    json!({"tool_calls":[{"index":0,"function":{"arguments":c}}]}),
                    Value::Null,
                ),
            ));
            "tool_calls"
        }
    };
    evs.push((None, chunk(json!({}), json!(finish))));
    let prompt_tokens = (body.len() / 4) as u64;
    evs.push((None, json!({"id":id,"object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":prompt_tokens,"completion_tokens":12,"total_tokens":prompt_tokens+12}})));
    let mut r = sse(evs);
    if req["stream"].as_bool() == Some(true) {
        // Append the terminator.
        let body = std::mem::replace(r.body_mut(), Body::empty());
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap_or_default();
        let mut v = bytes.to_vec();
        v.extend_from_slice(b"data: [DONE]\n\n");
        *r.body_mut() = Body::from(v);
        r
    } else {
        let content = match &reply {
            Reply::Text(t) => json!({"role":"assistant","content":t}),
            Reply::Tool { name, input } => {
                json!({"role":"assistant","content":null,"tool_calls":[{"id":format!("call_mock_{n}"),"type":"function","function":{"name":name,"arguments":input.to_string()}}]})
            }
        };
        axum::Json(json!({"id":id,"object":"chat.completion","model":model,"choices":[{"index":0,"message":content,"finish_reason":finish}],
            "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":12,"total_tokens":prompt_tokens+12}})).into_response()
    }
}
