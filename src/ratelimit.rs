//! Classifying upstream failures and extracting "when can I retry" hints.

use std::time::Duration;

use reqwest::header::HeaderMap;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureKind {
    /// Too many requests/tokens per unit of time. Recovers on its own.
    RateLimited,
    /// Out of credits / quota / balance.
    QuotaExhausted,
    /// Bad key, no permission, unknown model - the account is unusable.
    AccountError,
    /// Overloaded, 5xx, network trouble. Short cooldown.
    Transient,
    /// The prompt is too long for this model. Compact and retry.
    ContextOverflow,
    /// The request itself is invalid. Do not fail over; report to client.
    BadRequest,
}

impl FailureKind {
    /// Whether this failure should move the session to another account.
    pub fn fails_over(self) -> bool {
        !matches!(self, FailureKind::BadRequest | FailureKind::ContextOverflow)
    }
}

#[derive(Debug, Clone)]
pub struct UpstreamFailure {
    pub kind: FailureKind,
    pub status: Option<u16>,
    pub message: String,
    pub retry_after: Option<Duration>,
}

impl std::fmt::Display for UpstreamFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.status {
            Some(s) => write!(f, "{:?} (HTTP {}): {}", self.kind, s, self.message),
            None => write!(f, "{:?}: {}", self.kind, self.message),
        }
    }
}

impl UpstreamFailure {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            status: None,
            message: message.into(),
            retry_after: None,
        }
    }

    pub fn network(err: impl std::fmt::Display) -> Self {
        Self::new(FailureKind::Transient, format!("network error: {err}"))
    }
}

const QUOTA_MARKERS: &[&str] = &[
    "insufficient_quota",
    "insufficient balance",
    "insufficient_balance",
    "credit balance is too low",
    "credit balance",
    "out of credits",
    "no credits",
    "billing",
    "payment required",
    "spending limit",
    "usage limit",
    "monthly limit",
];

const CONTEXT_MARKERS: &[&str] = &[
    "prompt is too long",
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "input is too long",
    "exceeds the maximum number of tokens",
    "reduce the length",
    "request too large",
];

fn error_message(body: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let candidates = [
            v.pointer("/error/message"),
            v.pointer("/message"),
            v.pointer("/0/error/message"),
            v.pointer("/error"),
        ];
        for c in candidates.into_iter().flatten() {
            if let Some(s) = c.as_str() {
                return s.to_string();
            }
        }
    }
    let mut s = body.trim().to_string();
    if s.len() > 500 {
        s.truncate(s.floor_char_boundary(500));
    }
    if s.is_empty() {
        "(empty body)".into()
    } else {
        s
    }
}

/// Classify a non-2xx upstream response.
pub fn classify(status: u16, headers: &HeaderMap, body: &str) -> UpstreamFailure {
    let lower = body.to_ascii_lowercase();
    let hint = retry_hint(headers, body);
    let has = |markers: &[&str]| markers.iter().any(|m| lower.contains(m));
    let kind = match status {
        402 => FailureKind::QuotaExhausted,
        429 => {
            // Gemini says "exceeded your current quota" for per-minute limits
            // too, but includes a retry delay; trust the hint when present.
            if hint.is_none()
                && (has(QUOTA_MARKERS) || lower.contains("exceeded your current quota"))
            {
                FailureKind::QuotaExhausted
            } else {
                FailureKind::RateLimited
            }
        }
        401 => FailureKind::AccountError,
        403 => {
            if has(QUOTA_MARKERS) {
                FailureKind::QuotaExhausted
            } else {
                FailureKind::AccountError
            }
        }
        404 => FailureKind::AccountError,
        413 => FailureKind::ContextOverflow,
        408 | 409 | 425 => FailureKind::Transient,
        400 | 422 => {
            if has(QUOTA_MARKERS) {
                FailureKind::QuotaExhausted
            } else if has(CONTEXT_MARKERS) {
                FailureKind::ContextOverflow
            } else if lower.contains("rate limit") || lower.contains("rate_limit") {
                FailureKind::RateLimited
            } else {
                FailureKind::BadRequest
            }
        }
        500..=599 => FailureKind::Transient,
        _ => FailureKind::BadRequest,
    };
    UpstreamFailure {
        kind,
        status: Some(status),
        message: error_message(body),
        retry_after: hint,
    }
}

/// Classify an error delivered inside a stream (Anthropic `error` events,
/// OpenAI `{"error": ...}` chunks).
pub fn classify_stream_error(err: &Value) -> UpstreamFailure {
    let ty = err
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| err.get("code").and_then(Value::as_str))
        .unwrap_or("")
        .to_ascii_lowercase();
    let message = err
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("stream error")
        .to_string();
    let lower = format!("{ty} {}", message.to_ascii_lowercase());
    let kind = if lower.contains("rate_limit") || lower.contains("rate limit") {
        FailureKind::RateLimited
    } else if QUOTA_MARKERS.iter().any(|m| lower.contains(m)) {
        FailureKind::QuotaExhausted
    } else if CONTEXT_MARKERS.iter().any(|m| lower.contains(m)) {
        FailureKind::ContextOverflow
    } else if lower.contains("authentication") || lower.contains("permission") {
        FailureKind::AccountError
    } else if lower.contains("invalid_request") {
        FailureKind::BadRequest
    } else {
        FailureKind::Transient
    };
    UpstreamFailure {
        kind,
        status: None,
        message,
        retry_after: None,
    }
}

/// Look for a reset hint in headers and body.
pub fn retry_hint(headers: &HeaderMap, body: &str) -> Option<Duration> {
    let h = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
    };

    // Standard Retry-After: seconds or HTTP date.
    if let Some(v) = h("retry-after") {
        if let Ok(secs) = v.parse::<f64>()
            && secs >= 0.0
        {
            return Some(Duration::from_secs_f64(secs));
        }
        if let Ok(t) = httpdate::parse_http_date(v) {
            return Some(
                t.duration_since(std::time::SystemTime::now())
                    .unwrap_or_default(),
            );
        }
    }
    if let Some(v) = h("retry-after-ms").and_then(|v| v.parse::<f64>().ok()) {
        return Some(Duration::from_secs_f64(v / 1000.0));
    }

    // Anthropic: RFC 3339 reset timestamps. Use the latest exhausted one or
    // the earliest of all when we can't tell which limit tripped.
    let anthropic: Vec<(Option<&str>, &str)> = [
        (
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
        ),
        (
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset",
        ),
        (
            "anthropic-ratelimit-input-tokens-remaining",
            "anthropic-ratelimit-input-tokens-reset",
        ),
        (
            "anthropic-ratelimit-output-tokens-remaining",
            "anthropic-ratelimit-output-tokens-reset",
        ),
        (
            "anthropic-ratelimit-unified-remaining",
            "anthropic-ratelimit-unified-reset",
        ),
    ]
    .into_iter()
    .filter_map(|(rem, reset)| h(reset).map(|r| (h(rem), r)))
    .collect();
    if let Some(d) = pick_reset(
        anthropic
            .iter()
            .map(|(rem, r)| (*rem, parse_reset_value(r))),
    ) {
        return Some(d);
    }

    // OpenAI style: x-ratelimit-reset-requests: "1s", "6m0s".
    let openai: Vec<(Option<&str>, &str)> = [
        (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        ),
        ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
        ("x-ratelimit-remaining", "x-ratelimit-reset"),
    ]
    .into_iter()
    .filter_map(|(rem, reset)| h(reset).map(|r| (h(rem), r)))
    .collect();
    if let Some(d) = pick_reset(openai.iter().map(|(rem, r)| (*rem, parse_reset_value(r)))) {
        return Some(d);
    }

    // Gemini: google.rpc.RetryInfo { retryDelay: "37s" } in the body.
    if let Ok(v) = serde_json::from_str::<Value>(body)
        && let Some(d) = find_retry_delay(&v)
    {
        return Some(d);
    }
    let lower = body.to_ascii_lowercase();
    // "Please try again in 20s" / "retry in 1m30s" (OpenAI, Gemini text).
    for marker in ["try again in ", "retry in ", "retry after "] {
        if let Some(pos) = lower.find(marker) {
            let rest = &lower[pos + marker.len()..];
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '.')
                .collect();
            if let Some(d) = parse_go_duration(&token) {
                return Some(d);
            }
        }
    }
    None
}

fn pick_reset<'a>(
    items: impl Iterator<Item = (Option<&'a str>, Option<Duration>)>,
) -> Option<Duration> {
    let items: Vec<_> = items.collect();
    let exhausted = items
        .iter()
        .filter(|(rem, d)| d.is_some() && rem.map(|r| r.trim() == "0").unwrap_or(false))
        .filter_map(|(_, d)| *d)
        .max();
    exhausted.or_else(|| items.iter().filter_map(|(_, d)| *d).min())
}

fn find_retry_delay(v: &Value) -> Option<Duration> {
    match v {
        Value::Object(map) => {
            if let Some(s) = map.get("retryDelay").and_then(Value::as_str) {
                return parse_go_duration(s);
            }
            map.values().find_map(find_retry_delay)
        }
        Value::Array(a) => a.iter().find_map(find_retry_delay),
        _ => None,
    }
}

/// Parse a reset value: RFC 3339 timestamp, Go-style duration, or seconds.
pub fn parse_reset_value(s: &str) -> Option<Duration> {
    let s = s.trim();
    if let Ok(t) = chrono::DateTime::parse_from_rfc3339(s) {
        let now = chrono::Utc::now();
        let d = t.with_timezone(&chrono::Utc) - now;
        return Some(d.to_std().unwrap_or_default());
    }
    if let Ok(secs) = s.parse::<f64>() {
        // Large numbers are epoch timestamps.
        if secs > 1_000_000_000.0 {
            let now = chrono::Utc::now().timestamp() as f64;
            return Some(Duration::from_secs_f64((secs - now).max(0.0)));
        }
        return Some(Duration::from_secs_f64(secs.max(0.0)));
    }
    parse_go_duration(s)
}

/// Parse "1h2m3.5s", "6m0s", "20ms", "37s", "1.5s".
pub fn parse_go_duration(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut total = 0f64;
    let mut num = String::new();
    let mut chars = s.chars().peekable();
    let mut any = false;
    while let Some(c) = chars.next() {
        if c.is_ascii_digit() || c == '.' {
            num.push(c);
            continue;
        }
        let mut unit = c.to_string();
        if c == 'm' && chars.peek() == Some(&'s') {
            unit.push(chars.next().unwrap());
        }
        let n: f64 = num.parse().ok()?;
        num.clear();
        total += match unit.as_str() {
            "h" => n * 3600.0,
            "m" => n * 60.0,
            "s" => n,
            "ms" => n / 1000.0,
            _ => return None,
        };
        any = true;
    }
    if !num.is_empty() {
        // Trailing bare number: seconds.
        total += num.parse::<f64>().ok()?;
        any = true;
    }
    any.then(|| Duration::from_secs_f64(total))
}

/// On a successful response: if headers say the budget is already exhausted,
/// return how long until it resets so the account can be benched early.
pub fn preemptive_cooldown(headers: &HeaderMap) -> Option<Duration> {
    let h = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .map(str::trim)
    };
    let pairs = [
        (
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
        ),
        (
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset",
        ),
        (
            "anthropic-ratelimit-input-tokens-remaining",
            "anthropic-ratelimit-input-tokens-reset",
        ),
        (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        ),
        ("x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens"),
    ];
    pairs
        .iter()
        .filter(|(rem, _)| h(rem) == Some("0"))
        .filter_map(|(_, reset)| h(reset).and_then(parse_reset_value))
        .filter(|d| *d > Duration::from_secs(1))
        .max()
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::{HeaderName, HeaderValue};

    fn headers(pairs: &[(&str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(
                HeaderName::from_bytes(k.as_bytes()).unwrap(),
                HeaderValue::from_str(v).unwrap(),
            );
        }
        h
    }

    #[test]
    fn go_durations() {
        assert_eq!(parse_go_duration("6m0s"), Some(Duration::from_secs(360)));
        assert_eq!(parse_go_duration("20ms"), Some(Duration::from_millis(20)));
        assert_eq!(parse_go_duration("1h2m3s"), Some(Duration::from_secs(3723)));
        assert_eq!(parse_go_duration("1.5s"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_go_duration("37s"), Some(Duration::from_secs(37)));
        assert_eq!(parse_go_duration("abc"), None);
    }

    #[test]
    fn retry_after_header() {
        let f = classify(429, &headers(&[("retry-after", "42")]), "{}");
        assert_eq!(f.kind, FailureKind::RateLimited);
        assert_eq!(f.retry_after, Some(Duration::from_secs(42)));
    }

    #[test]
    fn openai_insufficient_quota() {
        let body = r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","code":"insufficient_quota"}}"#;
        let f = classify(429, &HeaderMap::new(), body);
        assert_eq!(f.kind, FailureKind::QuotaExhausted);
        assert!(f.message.contains("exceeded"));
    }

    #[test]
    fn gemini_retry_delay() {
        let body = r#"[{"error":{"code":429,"message":"You exceeded your current quota","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"37s"}]}}]"#;
        let f = classify(429, &HeaderMap::new(), body);
        assert_eq!(f.kind, FailureKind::RateLimited);
        assert_eq!(f.retry_after, Some(Duration::from_secs(37)));
    }

    #[test]
    fn anthropic_credit_and_context() {
        let f = classify(
            400,
            &HeaderMap::new(),
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}"#,
        );
        assert_eq!(f.kind, FailureKind::QuotaExhausted);
        let f = classify(
            400,
            &HeaderMap::new(),
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}"#,
        );
        assert_eq!(f.kind, FailureKind::ContextOverflow);
        let f = classify(
            400,
            &HeaderMap::new(),
            r#"{"error":{"message":"tools.0: bad schema"}}"#,
        );
        assert_eq!(f.kind, FailureKind::BadRequest);
    }

    #[test]
    fn anthropic_reset_headers() {
        let reset = (chrono::Utc::now() + chrono::Duration::seconds(90)).to_rfc3339();
        let h = headers(&[
            ("anthropic-ratelimit-tokens-remaining", "0"),
            ("anthropic-ratelimit-tokens-reset", &reset),
        ]);
        let d = retry_hint(&h, "").unwrap();
        assert!(d.as_secs() >= 85 && d.as_secs() <= 90, "{d:?}");
        assert!(preemptive_cooldown(&h).is_some());
    }

    #[test]
    fn openai_reset_headers() {
        let h = headers(&[
            ("x-ratelimit-remaining-requests", "0"),
            ("x-ratelimit-reset-requests", "6m0s"),
        ]);
        assert_eq!(retry_hint(&h, ""), Some(Duration::from_secs(360)));
    }

    #[test]
    fn deepseek_402_and_overload() {
        assert_eq!(
            classify(402, &HeaderMap::new(), "Insufficient Balance").kind,
            FailureKind::QuotaExhausted
        );
        assert_eq!(
            classify(529, &HeaderMap::new(), "overloaded").kind,
            FailureKind::Transient
        );
        assert_eq!(
            classify(401, &HeaderMap::new(), "bad key").kind,
            FailureKind::AccountError
        );
    }

    #[test]
    fn stream_errors() {
        let f = classify_stream_error(
            &serde_json::json!({"type":"overloaded_error","message":"Overloaded"}),
        );
        assert_eq!(f.kind, FailureKind::Transient);
        let f = classify_stream_error(
            &serde_json::json!({"type":"rate_limit_error","message":"slow down"}),
        );
        assert_eq!(f.kind, FailureKind::RateLimited);
    }
}
