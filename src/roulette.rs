//! The router: picks an account, applies/creates compaction checkpoints,
//! fails over on rate limits and keeps per-session stickiness.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use reqwest::header::HeaderMap;
use serde_json::{Value, json};

use crate::canonical::{
    Accumulator, conversation_fingerprint, estimate_request_tokens, estimate_tokens,
    is_content_event, sanitize_messages,
};
use crate::compaction;
use crate::config::Config;
use crate::ratelimit::{FailureKind, UpstreamFailure, preemptive_cooldown};
use crate::state::{Checkpoint, StateStore, now_ts};
use crate::upstream::{self, Account, Call, EventStream};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lane {
    /// The roulette model: sticky sessions, compaction on switch.
    Main,
    /// Cheap background calls: first available account, `fast_model`.
    Fast,
}

pub struct RouteRequest {
    /// Canonical (Anthropic-shaped) request.
    pub request: Value,
    /// Session id from the harness, if any.
    pub session: Option<String>,
    pub lane: Lane,
    pub headers: HeaderMap,
    /// Whether the messages came from a format conversion and need
    /// structural sanitizing (Anthropic-native requests are sent as-is).
    pub converted: bool,
}

pub struct Routed {
    pub account: String,
    pub model: String,
    pub events: EventStream,
    pub compacted: bool,
}

#[derive(Debug)]
pub enum RouteError {
    /// Every account is cooling down (or failed for this request).
    Exhausted {
        retry_after: Option<Duration>,
        last: Option<UpstreamFailure>,
    },
    /// A non-retryable upstream error (bad request) to report to the client.
    Upstream(UpstreamFailure),
    NoAccounts,
}

impl std::fmt::Display for RouteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RouteError::Exhausted { retry_after, last } => {
                write!(
                    f,
                    "all model-roulette accounts are rate limited or unavailable"
                )?;
                if let Some(d) = retry_after {
                    write!(f, "; next one frees up in {}s", d.as_secs())?;
                }
                if let Some(l) = last {
                    write!(f, " (last error: {l})")?;
                }
                Ok(())
            }
            RouteError::Upstream(u) => write!(f, "{}", u.message),
            RouteError::NoAccounts => write!(f, "no usable accounts configured (check API keys)"),
        }
    }
}

pub struct Roulette {
    pub cfg: Arc<Config>,
    pub accounts: Vec<Account>,
    pub store: Arc<StateStore>,
    pub http: reqwest::Client,
}

impl Roulette {
    pub fn new(cfg: Config, store: Arc<StateStore>) -> anyhow::Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .read_timeout(Duration::from_secs(cfg.rotation.request_timeout_secs))
            .build()?;
        let accounts = cfg
            .accounts
            .iter()
            .cloned()
            .map(Account::from_config)
            .collect();
        Ok(Self {
            cfg: Arc::new(cfg),
            accounts,
            store,
            http,
        })
    }

    pub fn usable(&self, a: &Account) -> bool {
        a.cfg.enabled
            && (a.api_key.is_some()
                || matches!(
                    a.cfg.provider,
                    crate::providers::ProviderKind::OpenaiCompatible
                        | crate::providers::ProviderKind::AnthropicCompatible
                ))
    }

    fn account(&self, id: &str) -> Option<&Account> {
        self.accounts.iter().find(|a| a.id() == id)
    }

    /// Next available account. With `current` set (sticky session) the scan
    /// starts at that account and wraps around, so a session keeps moving
    /// forward through the list.
    fn select(&self, current: Option<&str>, tried: &HashSet<String>) -> Option<&Account> {
        let n = self.accounts.len();
        if n == 0 {
            return None;
        }
        let start = current
            .filter(|_| self.cfg.rotation.sticky)
            .and_then(|c| self.accounts.iter().position(|a| a.id() == c))
            .unwrap_or(0);
        let now = now_ts();
        (0..n).map(|i| &self.accounts[(start + i) % n]).find(|a| {
            self.usable(a) && !tried.contains(a.id()) && self.store.is_available(a.id(), now)
        })
    }

    /// Seconds until the first usable account comes off cooldown.
    fn earliest_recovery(&self) -> Option<Duration> {
        let now = now_ts();
        let st = self.store.lock();
        self.accounts
            .iter()
            .filter(|a| self.usable(a))
            .map(|a| {
                st.accounts
                    .get(a.id())
                    .and_then(|s| s.cooldown_until)
                    .unwrap_or(now)
            })
            .min()
            .map(|t| Duration::from_secs((t - now).max(0) as u64))
    }

    fn cooldown_for(&self, account: &str, f: &UpstreamFailure) -> Option<Duration> {
        let r = &self.cfg.rotation;
        let consecutive = self.store.account(account).consecutive_failures;
        let d = match f.kind {
            FailureKind::RateLimited => f.retry_after.unwrap_or_else(|| {
                let factor = 2u64.saturating_pow(consecutive.min(16));
                Duration::from_secs(
                    r.rate_limit_cooldown_secs
                        .saturating_mul(factor)
                        .min(r.max_backoff_secs),
                )
            }),
            FailureKind::QuotaExhausted => f
                .retry_after
                .unwrap_or(Duration::from_secs(r.quota_cooldown_secs)),
            FailureKind::AccountError => Duration::from_secs(r.auth_cooldown_secs),
            FailureKind::Transient => f
                .retry_after
                .unwrap_or(Duration::from_secs(r.transient_cooldown_secs))
                .min(Duration::from_secs(r.transient_cooldown_secs.max(1) * 10)),
            FailureKind::ContextOverflow | FailureKind::BadRequest => return None,
        };
        Some(d.max(Duration::from_secs(1)))
    }

    pub fn bench(&self, account: &str, f: &UpstreamFailure) {
        if let Some(d) = self.cooldown_for(account, f) {
            let until = now_ts() + d.as_secs().max(1) as i64;
            tracing::warn!(account, kind = ?f.kind, cooldown_secs = d.as_secs(), "benching account: {}", f.message);
            self.store.bench(account, f.kind, until, &f.message);
        }
    }

    // ---- session bookkeeping --------------------------------------------

    fn session_account(&self, key: &str) -> Option<String> {
        self.store
            .lock()
            .sessions
            .get(key)
            .and_then(|s| s.account.clone())
    }

    fn checkpoint(&self, key: &str, fp: &str) -> Option<Checkpoint> {
        self.store
            .lock()
            .sessions
            .get(key)
            .and_then(|s| s.conversations.get(fp))
            .and_then(|c| c.checkpoint.clone())
    }

    fn set_checkpoint(&self, key: &str, fp: &str, ck: Option<Checkpoint>) {
        self.store.update(|st| {
            let s = st.sessions.entry(key.to_string()).or_default();
            let c = s.conversations.entry(fp.to_string()).or_default();
            if ck.is_some() {
                c.compactions += 1;
            }
            c.checkpoint = ck;
            c.last_used = now_ts();
        });
    }

    fn touch_session(&self, key: &str, fp: &str, account: &str) {
        self.store.update(|st| {
            let s = st.sessions.entry(key.to_string()).or_default();
            if s.created_at == 0 {
                s.created_at = now_ts();
            }
            if s.account.as_deref() != Some(account) {
                if s.account.is_some() {
                    s.switches += 1;
                }
                s.account = Some(account.to_string());
            }
            s.last_used = now_ts();
            s.conversations.entry(fp.to_string()).or_default().last_used = now_ts();
        });
    }

    // ---- dispatch --------------------------------------------------------

    pub async fn dispatch(&self, rr: RouteRequest) -> Result<Routed, RouteError> {
        if !self.accounts.iter().any(|a| self.usable(a)) {
            return Err(RouteError::NoAccounts);
        }
        let original: Vec<Value> = rr
            .request
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let fp = conversation_fingerprint(&original);
        let skey = rr.session.clone().unwrap_or_else(|| format!("conv:{fp}"));
        let current = match rr.lane {
            Lane::Main => self.session_account(&skey),
            Lane::Fast => None,
        };
        let overhead = {
            let mut r = rr.request.clone();
            r.as_object_mut().map(|o| o.remove("messages"));
            estimate_request_tokens(&r)
        };

        let mut tried: HashSet<String> = HashSet::new();
        let mut force_compact = false;
        let mut last: Option<UpstreamFailure> = None;
        let deadline = now_ts() + self.cfg.rotation.max_wait_secs as i64;

        loop {
            let Some(acct) = self.select(current.as_deref(), &tried) else {
                let wait = self.earliest_recovery();
                if let Some(w) = wait
                    && now_ts() + w.as_secs() as i64 <= deadline
                    && self.cfg.rotation.max_wait_secs > 0
                {
                    tracing::info!("all accounts cooling down; waiting {}s", w.as_secs());
                    tokio::time::sleep(w + Duration::from_millis(200)).await;
                    tried.clear();
                    continue;
                }
                return Err(RouteError::Exhausted {
                    retry_after: wait,
                    last,
                });
            };
            let model = match rr.lane {
                Lane::Main => acct.cfg.model().to_string(),
                Lane::Fast => acct.cfg.fast_model().to_string(),
            };

            let mut msgs = original.clone();
            let mut rewritten = false;
            let mut compacted = false;
            if rr.lane == Lane::Main {
                let mut ck = self.checkpoint(&skey, &fp);
                if let Some(c) = &ck {
                    match compaction::apply_checkpoint(&original, c) {
                        Some(rw) => {
                            msgs = rw;
                            rewritten = true;
                        }
                        None => {
                            tracing::info!(session = %skey, "checkpoint no longer matches history; dropping it");
                            self.set_checkpoint(&skey, &fp, None);
                            ck = None;
                        }
                    }
                }
                let switching = current.as_deref().map(|c| c != acct.id()).unwrap_or(false);
                let est = estimate_tokens(&Value::Array(msgs.clone())) + overhead;
                let window_limit =
                    (acct.cfg.context_window() as f64 * self.cfg.compaction.proactive_ratio) as u64;
                let need = self.cfg.compaction.enabled
                    && ((switching && est >= self.cfg.compaction.trigger_tokens)
                        || est > window_limit
                        || force_compact);
                if need
                    && let Some(new_ck) = self.compact(&original, &msgs, ck.as_ref(), acct).await
                    && let Some(rw) = compaction::apply_checkpoint(&original, &new_ck)
                {
                    let after = estimate_tokens(&Value::Array(rw.clone())) + overhead;
                    tracing::info!(
                        session = %skey, to = acct.id(), covered = new_ck.covered, method = %new_ck.method,
                        before = est, after, "compacted conversation"
                    );
                    msgs = rw;
                    rewritten = true;
                    compacted = true;
                    self.set_checkpoint(&skey, &fp, Some(new_ck));
                }
            }
            if rewritten || rr.converted {
                msgs = sanitize_messages(msgs);
            }
            let mut request = rr.request.clone();
            request["messages"] = Value::Array(msgs);

            tracing::debug!(account = acct.id(), model = %model, "sending upstream");
            let call = Call {
                account: acct,
                model: model.clone(),
                request,
                client_headers: &rr.headers,
            };
            let result = match upstream::send(&self.http, &self.store, call).await {
                Ok(resp) => peek(resp.events).await.map(|ev| (ev, resp.headers)),
                Err(f) => Err(f),
            };
            match result {
                Ok((events, headers)) => {
                    self.store.record_success(acct.id());
                    if self.cfg.rotation.preemptive
                        && let Some(d) = preemptive_cooldown(&headers)
                    {
                        tracing::info!(
                            account = acct.id(),
                            secs = d.as_secs(),
                            "budget exhausted per headers; benching early"
                        );
                        self.store.bench(
                            acct.id(),
                            FailureKind::RateLimited,
                            now_ts() + d.as_secs() as i64,
                            "remaining budget is 0",
                        );
                    }
                    if rr.lane == Lane::Main {
                        if current.as_deref().is_some_and(|c| c != acct.id()) {
                            tracing::info!(session = %skey, from = ?current, to = acct.id(), "session switched account");
                        }
                        self.touch_session(&skey, &fp, acct.id());
                    }
                    return Ok(Routed {
                        account: acct.id().to_string(),
                        model,
                        events: self.observe(events, acct.id().to_string()),
                        compacted,
                    });
                }
                Err(f) => {
                    tracing::warn!(account = acct.id(), "upstream failure: {f}");
                    match f.kind {
                        FailureKind::BadRequest => return Err(RouteError::Upstream(f)),
                        FailureKind::ContextOverflow if !force_compact && rr.lane == Lane::Main => {
                            force_compact = true;
                        }
                        FailureKind::ContextOverflow => {
                            tried.insert(acct.id().to_string());
                        }
                        _ => {
                            self.bench(acct.id(), &f);
                            tried.insert(acct.id().to_string());
                        }
                    }
                    last = Some(f);
                }
            }
        }
    }

    /// Watch a live stream: remember which account produced thinking
    /// signatures, and bench the account if it errors mid-stream.
    fn observe(&self, events: EventStream, account: String) -> EventStream {
        let store = Arc::clone(&self.store);
        let cfg = Arc::clone(&self.cfg);
        let me_cooldown = move |f: &UpstreamFailure| -> Option<i64> {
            let r = &cfg.rotation;
            let secs = match f.kind {
                FailureKind::RateLimited => f
                    .retry_after
                    .map(|d| d.as_secs())
                    .unwrap_or(r.rate_limit_cooldown_secs),
                FailureKind::QuotaExhausted => r.quota_cooldown_secs,
                FailureKind::AccountError => r.auth_cooldown_secs,
                FailureKind::Transient => r.transient_cooldown_secs,
                _ => return None,
            };
            Some(now_ts() + secs.max(1) as i64)
        };
        Box::pin(events.map(move |item| {
            match &item {
                Ok(ev) => match ev.get("type").and_then(Value::as_str) {
                    Some("content_block_delta") => {
                        if let Some(sig) = ev.pointer("/delta/signature").and_then(Value::as_str) {
                            store.record_signature(sig, &account);
                        }
                    }
                    Some("content_block_start") => {
                        if let Some(data) =
                            ev.pointer("/content_block/data").and_then(Value::as_str)
                        {
                            store.record_signature(data, &account);
                        }
                        if let Some(sig) = ev
                            .pointer("/content_block/signature")
                            .and_then(Value::as_str)
                        {
                            store.record_signature(sig, &account);
                        }
                    }
                    _ => {}
                },
                Err(f) => {
                    if let Some(until) = me_cooldown(f) {
                        tracing::warn!(account = %account, "mid-stream failure, benching: {f}");
                        store.bench(&account, f.kind, until, &f.message);
                    }
                }
            }
            item
        }))
    }

    /// Summarize the older part of `working` (the list that would be sent,
    /// possibly already beginning with a summary) and return a checkpoint
    /// over `original`.
    async fn compact(
        &self,
        original: &[Value],
        working: &[Value],
        existing: Option<&Checkpoint>,
        target: &Account,
    ) -> Option<Checkpoint> {
        let b = compaction::choose_boundary(working, &self.cfg.compaction)?;
        let head = &working[..b];
        let (previous, entries) = compaction::render_transcript(head, &self.cfg.compaction);
        let chunks = compaction::chunk_entries(&entries, self.cfg.compaction.chunk_tokens);
        let mut summary = previous.clone();
        let mut method = String::new();
        let mut failed = false;
        for (i, chunk) in chunks.iter().enumerate() {
            let prompt = compaction::chunk_prompt(summary.as_deref(), chunk, i + 1, chunks.len());
            match self
                .complete_text(
                    compaction::SYSTEM_PROMPT,
                    &prompt,
                    self.cfg.compaction.summary_max_tokens,
                    Some(target.id()),
                )
                .await
            {
                Ok((text, who)) if !text.trim().is_empty() => {
                    summary = Some(text.trim().to_string());
                    method = format!("llm:{who}");
                }
                Ok(_) | Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        let summary = match summary {
            Some(s) if !failed && !method.is_empty() => s,
            _ => {
                tracing::warn!("summarizer unavailable; using extractive fallback summary");
                method = "fallback".into();
                compaction::fallback_summary(previous.as_deref(), head)
            }
        };
        let summary = compaction::with_latest_request(summary, head, &working[b..]);
        Some(compaction::make_checkpoint(
            original,
            b,
            existing.and_then(|c| compaction::checkpoint_cut(original, c)),
            summary,
            target.id(),
            method,
        ))
    }

    /// One-shot text completion with failover, using each account's
    /// `compact_model`. Returns the text and "account/model".
    pub async fn complete_text(
        &self,
        system: &str,
        prompt: &str,
        max_tokens: u64,
        prefer: Option<&str>,
    ) -> Result<(String, String), UpstreamFailure> {
        let mut order: Vec<&Account> = Vec::new();
        let configured = &self.cfg.compaction.compactor_accounts;
        if !configured.is_empty() {
            order.extend(configured.iter().filter_map(|id| self.account(id)));
        } else {
            if let Some(p) = prefer.and_then(|id| self.account(id)) {
                order.push(p);
            }
            order.extend(self.accounts.iter().filter(|a| Some(a.id()) != prefer));
        }
        let empty = HeaderMap::new();
        let mut last = UpstreamFailure::new(
            FailureKind::Transient,
            "no account available for summarization",
        );
        for acct in order {
            if !self.usable(acct) || !self.store.is_available(acct.id(), now_ts()) {
                continue;
            }
            let model = acct.cfg.compact_model().to_string();
            let request = json!({
                "model": model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            });
            let call = Call {
                account: acct,
                model: model.clone(),
                request,
                client_headers: &empty,
            };
            let res = match upstream::send(&self.http, &self.store, call).await {
                Ok(resp) => collect(resp.events).await,
                Err(f) => Err(f),
            };
            match res {
                Ok(msg) => {
                    self.store.record_success(acct.id());
                    let text: String = msg
                        .get("content")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter()
                                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                                .filter_map(|b| b.get("text").and_then(Value::as_str))
                                .collect()
                        })
                        .unwrap_or_default();
                    return Ok((text, format!("{}/{}", acct.id(), model)));
                }
                Err(f) => {
                    tracing::warn!(account = acct.id(), "summarizer call failed: {f}");
                    if f.kind.fails_over() {
                        self.bench(acct.id(), &f);
                    }
                    last = f;
                }
            }
        }
        Err(last)
    }

    pub fn status(&self) -> Value {
        let now = now_ts();
        let st = self.store.snapshot();
        let accounts: Vec<Value> = self
            .accounts
            .iter()
            .map(|a| {
                let s = st.accounts.get(a.id()).cloned().unwrap_or_default();
                let remaining = s.cooldown_until.map(|t| (t - now).max(0)).unwrap_or(0);
                json!({
                    "id": a.id(),
                    "provider": a.cfg.provider.as_str(),
                    "model": a.cfg.model(),
                    "compact_model": a.cfg.compact_model(),
                    "enabled": a.cfg.enabled,
                    "has_key": a.api_key.is_some(),
                    "available": self.usable(a) && remaining == 0,
                    "cooldown_remaining_secs": remaining,
                    "last_failure": s.last_failure,
                    "last_error": s.last_error,
                    "requests": s.requests,
                    "failures": s.failures,
                })
            })
            .collect();
        let mut sessions: Vec<(&String, &crate::state::SessionState)> =
            st.sessions.iter().collect();
        sessions.sort_by_key(|(_, s)| -s.last_used);
        let sessions: Vec<Value> = sessions
            .into_iter()
            .take(20)
            .map(|(id, s)| {
                json!({
                    "id": id,
                    "account": s.account,
                    "switches": s.switches,
                    "last_used": s.last_used,
                    "conversations": s.conversations.len(),
                    "compactions": s.conversations.values().map(|c| c.compactions).sum::<u32>(),
                })
            })
            .collect();
        json!({"model": self.cfg.server.model_name, "accounts": accounts, "recent_sessions": sessions, "session_count": st.sessions.len()})
    }
}

/// Read events until real output appears. Errors before that point are
/// returned so the router can fail over transparently.
async fn peek(mut events: EventStream) -> Result<EventStream, UpstreamFailure> {
    let mut buf = Vec::new();
    loop {
        match events.next().await {
            Some(Ok(ev)) => {
                let content = is_content_event(&ev);
                buf.push(ev);
                if content {
                    break;
                }
            }
            Some(Err(f)) => return Err(f),
            None => {
                if buf.is_empty() {
                    return Err(UpstreamFailure::new(
                        FailureKind::Transient,
                        "upstream returned an empty stream",
                    ));
                }
                break;
            }
        }
    }
    Ok(Box::pin(
        futures::stream::iter(buf.into_iter().map(Ok)).chain(events),
    ))
}

/// Collect a whole stream into a message.
pub async fn collect(mut events: EventStream) -> Result<Value, UpstreamFailure> {
    let mut acc = Accumulator::default();
    while let Some(item) = events.next().await {
        acc.push(&item?);
    }
    Ok(acc.finish())
}
