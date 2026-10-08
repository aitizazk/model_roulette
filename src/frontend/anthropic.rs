//! Anthropic Messages API front end (Claude Code).

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

use super::{AppState, ModelRoute, error_status, parse_body, session_from_headers, sse_frame};
use crate::canonical::{Accumulator, estimate_request_tokens};
use crate::ratelimit::FailureKind;
use crate::roulette::{RouteError, RouteRequest};

pub fn error_json(kind: &str, message: &str) -> Value {
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

fn error_response(
    status: StatusCode,
    kind: &str,
    message: &str,
    retry_after: Option<u64>,
) -> Response {
    let mut resp = (status, axum::Json(error_json(kind, message))).into_response();
    if let Some(s) = retry_after {
        resp.headers_mut()
            .insert("retry-after", HeaderValue::from(s));
    }
    resp
}

fn route_error_response(err: &RouteError) -> Response {
    let (status, retry) = error_status(err);
    let kind = match err {
        RouteError::Exhausted { .. } => "rate_limit_error",
        RouteError::Upstream(_) => "invalid_request_error",
        RouteError::NoAccounts => "api_error",
    };
    error_response(status, kind, &err.to_string(), retry)
}

/// Claude Code sends `X-Claude-Code-Session-Id`; older versions only put the
/// session in `metadata.user_id` (either JSON or `..._session_<uuid>`).
pub fn session_id(headers: &HeaderMap, body: &Value) -> Option<String> {
    if let Some(s) = session_from_headers(headers, &["x-claude-code-session-id", "x-session-id"]) {
        return Some(s);
    }
    let uid = body.pointer("/metadata/user_id")?.as_str()?;
    if let Ok(v) = serde_json::from_str::<Value>(uid)
        && let Some(s) = v.get("session_id").and_then(Value::as_str)
    {
        return Some(s.to_string());
    }
    if let Some(pos) = uid.find("_session_") {
        return Some(uid[pos + 9..].to_string());
    }
    Some(uid.to_string())
}

pub async fn messages(
    State(app): State<AppState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if !app.authorized(&headers) {
        return error_response(
            StatusCode::UNAUTHORIZED,
            "authentication_error",
            "invalid model-roulette api key",
            None,
        );
    }
    let req = match parse_body(&headers, &body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_request_error", &e, None);
        }
    };
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let lane = match app.route_model(&model) {
        ModelRoute::Lane(l) => l,
        ModelRoute::Passthrough => return passthrough(&app, &uri, headers, body).await,
        ModelRoute::Reject => {
            return error_response(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("unknown model '{model}'"),
                None,
            );
        }
    };
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let session = session_id(&headers, &req);
    let rr = RouteRequest {
        request: req,
        session,
        lane,
        headers: headers.clone(),
        converted: false,
    };
    let routed = match app.roulette.dispatch(rr).await {
        Ok(r) => r,
        Err(e) => return route_error_response(&e),
    };
    let mut extra = HeaderMap::new();
    extra.insert(
        "x-model-roulette-account",
        HeaderValue::from_str(&routed.account).unwrap_or(HeaderValue::from_static("?")),
    );
    extra.insert(
        "x-model-roulette-model",
        HeaderValue::from_str(&routed.model).unwrap_or(HeaderValue::from_static("?")),
    );

    if stream {
        let events = routed.events;
        let body = async_stream::stream! {
            futures::pin_mut!(events);
            while let Some(item) = events.next().await {
                match item {
                    Ok(ev) => {
                        let ty = ev.get("type").and_then(Value::as_str).unwrap_or("message").to_string();
                        yield Ok::<Bytes, std::convert::Infallible>(sse_frame(Some(&ty), &ev));
                    }
                    Err(f) => {
                        let kind = match f.kind {
                            FailureKind::RateLimited => "rate_limit_error",
                            FailureKind::Transient => "overloaded_error",
                            _ => "api_error",
                        };
                        yield Ok(sse_frame(Some("error"), &error_json(kind, &format!("model-roulette upstream error: {}", f.message))));
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
        h.extend(extra);
        return resp;
    }

    let mut acc = Accumulator::default();
    let mut events = routed.events;
    while let Some(item) = events.next().await {
        match item {
            Ok(ev) => acc.push(&ev),
            Err(f) => {
                return error_response(StatusCode::BAD_GATEWAY, "api_error", &f.message, None);
            }
        }
    }
    let mut resp = axum::Json(acc.finish()).into_response();
    resp.headers_mut().extend(extra);
    resp
}

pub async fn count_tokens(
    State(app): State<AppState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let req = match parse_body(&headers, &body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, "invalid_request_error", &e, None);
        }
    };
    let model = req.get("model").and_then(Value::as_str).unwrap_or("");
    if matches!(app.route_model(model), ModelRoute::Passthrough) {
        return passthrough(&app, &uri, headers, body).await;
    }
    axum::Json(json!({"input_tokens": estimate_request_tokens(&req)})).into_response()
}

/// Forward a request untouched to the passthrough upstream (normally
/// api.anthropic.com) using the client's own credentials. Lets Claude Code be
/// pointed at the proxy permanently while regular models keep working.
pub async fn passthrough(app: &AppState, uri: &Uri, headers: HeaderMap, body: Bytes) -> Response {
    let base = app
        .roulette
        .cfg
        .server
        .passthrough_base_url
        .trim_end_matches('/');
    let path = uri
        .path_and_query()
        .map(|p| p.as_str())
        .unwrap_or(uri.path());
    let url = format!("{base}{path}");
    let mut req = app.roulette.http.post(&url).body(body);
    for (k, v) in headers.iter() {
        let name = k.as_str();
        if matches!(
            name,
            "host" | "content-length" | "connection" | "accept-encoding" | "transfer-encoding"
        ) {
            continue;
        }
        req = req.header(k, v);
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                "api_error",
                &format!("passthrough failed: {e}"),
                None,
            );
        }
    };
    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out_headers = HeaderMap::new();
    for (k, v) in resp.headers().iter() {
        if matches!(
            k.as_str(),
            "content-length" | "transfer-encoding" | "connection" | "content-encoding"
        ) {
            continue;
        }
        out_headers.insert(k.clone(), v.clone());
    }
    let mut r = Response::new(Body::from_stream(resp.bytes_stream()));
    *r.status_mut() = status;
    *r.headers_mut() = out_headers;
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_extraction() {
        let mut h = HeaderMap::new();
        let body = json!({"metadata":{"user_id":"{\"device_id\":\"d\",\"session_id\":\"abc\"}"}});
        assert_eq!(session_id(&h, &body).as_deref(), Some("abc"));
        let body2 = json!({"metadata":{"user_id":"user_x_account_y_session_1234"}});
        assert_eq!(session_id(&h, &body2).as_deref(), Some("1234"));
        h.insert("x-claude-code-session-id", "hdr".parse().unwrap());
        assert_eq!(session_id(&h, &body).as_deref(), Some("hdr"));
    }
}
