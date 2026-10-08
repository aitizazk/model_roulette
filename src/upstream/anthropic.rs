//! Anthropic Messages wire (`POST {base}/v1/messages`).

use std::sync::Arc;

use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

use super::{Call, UpstreamResponse, is_event_stream, message_to_events, post_json, sse_events};
use crate::config::AuthStyle;
use crate::ratelimit::{FailureKind, UpstreamFailure, classify_stream_error};
use crate::state::StateStore;

/// Fields Claude Code sends that only first-party Anthropic understands.
const ANTHROPIC_ONLY_FIELDS: &[&str] = &[
    "context_management",
    "output_config",
    "container",
    "mcp_servers",
];

pub fn build_request(call: &Call<'_>, store: &StateStore) -> (Value, Vec<(String, String)>) {
    let acct = call.account;
    let quirks = acct.cfg.preset().quirks;
    let mut body = call.request.clone();
    let obj = body
        .as_object_mut()
        .expect("canonical request is an object");
    obj.insert("model".into(), json!(call.model));
    obj.insert("stream".into(), json!(true));
    if !quirks.forward_anthropic_extras {
        for f in ANTHROPIC_ONLY_FIELDS {
            obj.remove(*f);
        }
    }
    for f in &acct.cfg.drop_fields {
        obj.remove(f);
    }
    if let Some(max) = acct.cfg.max_output_tokens() {
        let cur = obj.get("max_tokens").and_then(Value::as_u64).unwrap_or(max);
        obj.insert("max_tokens".into(), json!(cur.min(max)));
    }
    if !obj.contains_key("max_tokens") {
        obj.insert("max_tokens".into(), json!(16_384));
    }

    if !quirks.forward_anthropic_extras
        && let Some(Value::Array(msgs)) = obj.get_mut("messages")
        && msgs.iter().any(|m| crate::canonical::role(m) == "system")
    {
        *msgs = crate::canonical::sanitize_messages(std::mem::take(msgs));
    }
    if let Some(Value::Array(msgs)) = obj.get_mut("messages") {
        for m in msgs.iter_mut() {
            clean_message(m, acct.id(), store);
        }
    }
    fix_thinking_constraint(obj);

    let mut headers = vec![(
        "anthropic-version".to_string(),
        call.client_headers
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("2023-06-01")
            .to_string(),
    )];
    if quirks.forward_anthropic_extras
        && let Some(beta) = call
            .client_headers
            .get("anthropic-beta")
            .and_then(|v| v.to_str().ok())
    {
        let filtered: Vec<&str> = beta
            .split(',')
            .map(str::trim)
            .filter(|b| !b.is_empty() && !b.starts_with("oauth-"))
            .collect();
        if !filtered.is_empty() {
            headers.push(("anthropic-beta".into(), filtered.join(",")));
        }
    }
    if let Some(key) = &acct.api_key {
        match acct.cfg.auth.unwrap_or(AuthStyle::Default) {
            AuthStyle::Default => headers.push(("x-api-key".into(), key.clone())),
            AuthStyle::Bearer => headers.push(("authorization".into(), format!("Bearer {key}"))),
        }
    }
    for (k, v) in &acct.cfg.headers {
        headers.push((k.clone(), v.clone()));
    }
    (body, headers)
}

/// Drop thinking blocks this account can't verify (produced by another
/// account, or synthesized from a non-Anthropic model) and make tool ids
/// conform to Anthropic's charset.
fn clean_message(m: &mut Value, account: &str, store: &StateStore) {
    let Some(Value::Array(blocks)) = m.get_mut("content") else {
        return;
    };
    blocks.retain(|b| match b.get("type").and_then(Value::as_str) {
        Some("thinking") => {
            let sig = b.get("signature").and_then(Value::as_str).unwrap_or("");
            !sig.is_empty()
                && store
                    .signature_owner(sig)
                    .map(|o| o == account)
                    .unwrap_or(true)
        }
        Some("redacted_thinking") => {
            let data = b.get("data").and_then(Value::as_str).unwrap_or("");
            store
                .signature_owner(data)
                .map(|o| o == account)
                .unwrap_or(true)
        }
        _ => true,
    });
    for b in blocks.iter_mut() {
        for key in ["id", "tool_use_id"] {
            if let Some(id) = b.get(key).and_then(Value::as_str) {
                let clean = sanitize_id(id);
                if clean != id {
                    b[key] = json!(clean);
                }
            }
        }
    }
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": "(no content)"}));
    }
}

pub fn sanitize_id(id: &str) -> String {
    let s: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "tool".into() } else { s }
}

/// With extended thinking on, Anthropic requires the assistant message that
/// opened an unfinished tool loop to begin with a thinking block. After a
/// model switch that block may be missing (we can't forge one), so thinking
/// is turned off for that request instead.
fn fix_thinking_constraint(obj: &mut serde_json::Map<String, Value>) {
    let thinking_on = obj
        .get("thinking")
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .map(|t| t != "disabled")
        .unwrap_or(false);
    if !thinking_on {
        return;
    }
    let Some(Value::Array(msgs)) = obj.get("messages") else {
        return;
    };
    let ends_in_tool_result = crate::canonical::last_turn(msgs)
        .map(|m| {
            crate::canonical::blocks(m)
                .iter()
                .any(|b| b.get("type").and_then(Value::as_str) == Some("tool_result"))
        })
        .unwrap_or(false);
    if !ends_in_tool_result {
        return;
    }
    let last_assistant = msgs
        .iter()
        .rev()
        .find(|m| crate::canonical::role(m) == "assistant");
    let starts_with_thinking = last_assistant
        .map(|m| {
            crate::canonical::blocks(m)
                .first()
                .and_then(|b| b.get("type").and_then(Value::as_str))
                .map(|t| t == "thinking" || t == "redacted_thinking")
                .unwrap_or(false)
        })
        .unwrap_or(true);
    if !starts_with_thinking {
        obj.remove("thinking");
        obj.remove("context_management");
    }
}

pub async fn send(
    http: &reqwest::Client,
    store: &Arc<StateStore>,
    call: Call<'_>,
) -> Result<UpstreamResponse, UpstreamFailure> {
    let (body, headers) = build_request(&call, store);
    let url = format!("{}/v1/messages", call.account.base_url);
    let mut req = http.post(&url);
    for (k, v) in &headers {
        req = req.header(k, v);
    }
    let resp = post_json(req, &body).await?;
    let resp_headers: HeaderMap = resp.headers().clone();

    if !is_event_stream(&resp) {
        // Some compatible servers ignore stream=true.
        let msg: Value = resp.json().await.map_err(|e| {
            UpstreamFailure::new(
                FailureKind::Transient,
                format!("bad JSON from upstream: {e}"),
            )
        })?;
        let events = message_to_events(&msg).into_iter().map(Ok);
        return Ok(UpstreamResponse {
            events: Box::pin(futures::stream::iter(events)),
            headers: resp_headers,
        });
    }

    let sse = sse_events(resp.bytes_stream());
    let events = async_stream::stream! {
        futures::pin_mut!(sse);
        while let Some(item) = sse.next().await {
            let ev = match item {
                Ok(ev) => ev,
                Err(e) => { yield Err(e); return; }
            };
            let Ok(data) = serde_json::from_str::<Value>(&ev.data) else { continue };
            match data.get("type").and_then(Value::as_str) {
                Some("ping") => continue,
                Some("error") => {
                    yield Err(classify_stream_error(data.get("error").unwrap_or(&data)));
                    return;
                }
                Some(_) => yield Ok(data),
                None => continue,
            }
        }
    };
    Ok(UpstreamResponse {
        events: Box::pin(events),
        headers: resp_headers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
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
            base_url: Some("http://x".into()),
            auth: None,
            context_window: None,
            max_output_tokens: Some(1000),
            reasoning_effort: None,
            headers: Default::default(),
            drop_fields: vec![],
            enabled: true,
        })
    }

    #[test]
    fn strips_foreign_thinking_and_fixes_constraint() {
        let store = StateStore::open(None);
        store.record_signature("mine", "a");
        store.record_signature("theirs", "b");
        let acct = account(ProviderKind::Anthropic);
        let mut h = HeaderMap::new();
        h.insert(
            "anthropic-beta",
            "oauth-2025-04-20,interleaved-thinking-2025-05-14"
                .parse()
                .unwrap(),
        );
        let call = Call {
            account: &acct,
            model: "m".into(),
            client_headers: &h,
            request: json!({
                "model": "model-roulette",
                "max_tokens": 32000,
                "thinking": {"type": "adaptive"},
                "context_management": {"edits": []},
                "messages": [
                    {"role":"user","content":"hi"},
                    {"role":"assistant","content":[{"type":"thinking","thinking":"a","signature":"mine"},{"type":"text","text":"x"}]},
                    {"role":"user","content":"again"},
                    {"role":"assistant","content":[{"type":"thinking","thinking":"b","signature":"theirs"},{"type":"tool_use","id":"call:1","name":"f","input":{}}]},
                    {"role":"user","content":[{"type":"tool_result","tool_use_id":"call:1","content":"ok"}]}
                ]
            }),
        };
        let (body, headers) = build_request(&call, &store);
        assert_eq!(body["max_tokens"], 1000);
        assert_eq!(body["messages"][1]["content"][0]["type"], "thinking");
        assert_eq!(body["messages"][3]["content"][0]["type"], "tool_use");
        assert_eq!(body["messages"][3]["content"][0]["id"], "call_1");
        assert_eq!(body["messages"][4]["content"][0]["tool_use_id"], "call_1");
        assert!(body.get("thinking").is_none());
        assert!(body.get("context_management").is_none());
        let beta = headers.iter().find(|(k, _)| k == "anthropic-beta").unwrap();
        assert_eq!(beta.1, "interleaved-thinking-2025-05-14");
        assert!(headers.iter().any(|(k, v)| k == "x-api-key" && v == "k"));
    }

    #[test]
    fn compatible_drops_first_party_fields() {
        let store = StateStore::open(None);
        let acct = account(ProviderKind::AnthropicCompatible);
        let h = HeaderMap::new();
        let call = Call {
            account: &acct,
            model: "m".into(),
            client_headers: &h,
            request: json!({"messages":[{"role":"user","content":"hi"}],"context_management":{},"output_config":{}}),
        };
        let (body, _) = build_request(&call, &store);
        assert!(body.get("context_management").is_none());
        assert!(body.get("output_config").is_none());
    }
}
