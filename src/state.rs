//! Persistent roulette state: per-account cooldowns, per-session stickiness,
//! compaction checkpoints and small caches needed to round-trip provider
//! specific data (thinking signatures, tool-call extras).
//!
//! State is kept in memory behind a mutex and flushed to a JSON file by a
//! debounced background task, so new processes (and new sessions) know which
//! accounts are still cooling down.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use crate::ratelimit::FailureKind;

pub fn now_ts() -> i64 {
    chrono::Utc::now().timestamp()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct AccountState {
    /// Unix seconds until which the account is benched.
    pub cooldown_until: Option<i64>,
    pub last_failure: Option<FailureKind>,
    pub last_error: Option<String>,
    /// Consecutive rate-limit failures (drives exponential backoff).
    pub consecutive_failures: u32,
    pub requests: u64,
    pub failures: u64,
    pub last_used: Option<i64>,
}

impl AccountState {
    pub fn available_at(&self, now: i64) -> bool {
        self.cooldown_until.map(|t| t <= now).unwrap_or(true)
    }
}

/// A compaction checkpoint: the first `covered` messages of a conversation
/// (identified by `prefix_hash`) are replaced by `summary`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub covered: usize,
    pub prefix_hash: String,
    pub summary: String,
    pub created_at: i64,
    pub by_account: String,
    /// "llm:<account>/<model>" or "fallback".
    pub method: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ConversationState {
    pub checkpoint: Option<Checkpoint>,
    pub compactions: u32,
    pub last_used: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionState {
    pub account: Option<String>,
    pub switches: u32,
    pub created_at: i64,
    pub last_used: i64,
    /// Keyed by a fingerprint of the conversation's first message, so that
    /// sub-agents sharing a session id don't clobber each other.
    pub conversations: BTreeMap<String, ConversationState>,
}

/// Extra provider data attached to a tool call that the harness won't round
/// trip for us.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolExtra {
    pub gemini_thought_signature: Option<String>,
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PersistedState {
    pub accounts: BTreeMap<String, AccountState>,
    pub sessions: BTreeMap<String, SessionState>,
    /// hash(thinking signature) -> account id that produced it.
    pub thinking_signatures: IndexMap<String, String>,
    /// tool call id -> extras.
    pub tool_extras: IndexMap<String, ToolExtra>,
}

const MAX_SESSIONS: usize = 2000;
const MAX_CONVERSATIONS_PER_SESSION: usize = 64;
const MAX_SIGNATURES: usize = 8192;
const MAX_TOOL_EXTRAS: usize = 8192;

pub struct StateStore {
    inner: Mutex<PersistedState>,
    path: Option<PathBuf>,
    dirty: Notify,
}

impl StateStore {
    /// Load from `path` (missing/corrupt files start fresh). `None` = memory only.
    pub fn open(path: Option<PathBuf>) -> Arc<Self> {
        let state = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| match serde_json::from_str::<PersistedState>(&s) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!("ignoring unreadable state file: {e}");
                    None
                }
            })
            .unwrap_or_default();
        Arc::new(Self {
            inner: Mutex::new(state),
            path,
            dirty: Notify::new(),
        })
    }

    pub fn lock(&self) -> MutexGuard<'_, PersistedState> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Mutate state and schedule a flush.
    pub fn update<R>(&self, f: impl FnOnce(&mut PersistedState) -> R) -> R {
        let r = f(&mut self.lock());
        self.dirty.notify_one();
        r
    }

    pub fn snapshot(&self) -> PersistedState {
        self.lock().clone()
    }

    pub fn flush(&self) -> std::io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        let json = {
            let mut st = self.lock();
            prune(&mut st);
            serde_json::to_vec_pretty(&*st).map_err(std::io::Error::other)?
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, json)?;
        std::fs::rename(&tmp, path)
    }

    /// Background task: flush shortly after changes.
    pub fn spawn_flusher(self: &Arc<Self>) {
        let me = Arc::clone(self);
        tokio::spawn(async move {
            loop {
                me.dirty.notified().await;
                tokio::time::sleep(Duration::from_millis(250)).await;
                let me2 = Arc::clone(&me);
                let res = tokio::task::spawn_blocking(move || me2.flush()).await;
                if let Ok(Err(e)) = res {
                    tracing::warn!("failed to write state file: {e}");
                }
            }
        });
    }

    // ---- account helpers -------------------------------------------------

    pub fn account(&self, id: &str) -> AccountState {
        self.lock().accounts.get(id).cloned().unwrap_or_default()
    }

    pub fn is_available(&self, id: &str, now: i64) -> bool {
        self.lock()
            .accounts
            .get(id)
            .map(|a| a.available_at(now))
            .unwrap_or(true)
    }

    pub fn record_success(&self, id: &str) {
        self.update(|st| {
            let a = st.accounts.entry(id.to_string()).or_default();
            a.requests += 1;
            a.consecutive_failures = 0;
            a.last_used = Some(now_ts());
        });
    }

    pub fn bench(&self, id: &str, kind: FailureKind, until: i64, message: &str) {
        self.update(|st| {
            let a = st.accounts.entry(id.to_string()).or_default();
            a.cooldown_until = Some(a.cooldown_until.unwrap_or(0).max(until));
            a.last_failure = Some(kind);
            a.last_error = Some(truncate(message, 300));
            a.failures += 1;
            if kind == FailureKind::RateLimited {
                a.consecutive_failures += 1;
            }
        });
    }

    pub fn reset_account(&self, id: Option<&str>) {
        self.update(|st| {
            for (aid, a) in st.accounts.iter_mut() {
                if id.is_none() || id == Some(aid.as_str()) {
                    a.cooldown_until = None;
                    a.consecutive_failures = 0;
                    a.last_failure = None;
                    a.last_error = None;
                }
            }
        });
    }

    // ---- caches ----------------------------------------------------------

    pub fn record_signature(&self, sig: &str, account: &str) {
        if sig.is_empty() {
            return;
        }
        self.update(|st| {
            st.thinking_signatures
                .insert(crate::canonical::hash_str(sig), account.to_string());
            while st.thinking_signatures.len() > MAX_SIGNATURES {
                st.thinking_signatures.shift_remove_index(0);
            }
        });
    }

    pub fn signature_owner(&self, sig: &str) -> Option<String> {
        self.lock()
            .thinking_signatures
            .get(&crate::canonical::hash_str(sig))
            .cloned()
    }

    pub fn put_tool_extra(&self, call_id: &str, f: impl FnOnce(&mut ToolExtra)) {
        self.update(|st| {
            f(st.tool_extras.entry(call_id.to_string()).or_default());
            while st.tool_extras.len() > MAX_TOOL_EXTRAS {
                st.tool_extras.shift_remove_index(0);
            }
        });
    }

    pub fn tool_extra(&self, call_id: &str) -> Option<ToolExtra> {
        self.lock().tool_extras.get(call_id).cloned()
    }
}

fn prune(st: &mut PersistedState) {
    if st.sessions.len() > MAX_SESSIONS {
        let mut by_age: Vec<(i64, String)> = st
            .sessions
            .iter()
            .map(|(k, s)| (s.last_used, k.clone()))
            .collect();
        by_age.sort();
        let excess = st.sessions.len() - MAX_SESSIONS;
        for (_, k) in by_age.into_iter().take(excess) {
            st.sessions.remove(&k);
        }
    }
    for s in st.sessions.values_mut() {
        if s.conversations.len() > MAX_CONVERSATIONS_PER_SESSION {
            let mut by_age: Vec<(i64, String)> = s
                .conversations
                .iter()
                .map(|(k, c)| (c.last_used, k.clone()))
                .collect();
            by_age.sort();
            let excess = s.conversations.len() - MAX_CONVERSATIONS_PER_SESSION;
            for (_, k) in by_age.into_iter().take(excess) {
                s.conversations.remove(&k);
            }
        }
    }
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..s.floor_char_boundary(max)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persist_roundtrip() {
        let dir = std::env::temp_dir().join(format!("mr-state-{}", uuid::Uuid::new_v4()));
        let path = dir.join("state.json");
        let s = StateStore::open(Some(path.clone()));
        s.bench("a", FailureKind::RateLimited, now_ts() + 100, "429");
        s.record_signature("sig123", "a");
        s.flush().unwrap();
        let s2 = StateStore::open(Some(path));
        assert!(!s2.is_available("a", now_ts()));
        assert!(s2.is_available("b", now_ts()));
        assert_eq!(s2.signature_owner("sig123").as_deref(), Some("a"));
        s2.reset_account(Some("a"));
        assert!(s2.is_available("a", now_ts()));
        let _ = std::fs::remove_dir_all(dir);
    }
}
