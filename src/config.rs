//! Configuration file (`~/.model-roulette/config.toml`).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::providers::{Preset, ProviderKind, Wire};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: ServerConfig,
    pub rotation: RotationConfig,
    pub compaction: CompactionConfig,
    pub accounts: Vec<AccountConfig>,
}

/// What to do with requests whose `model` is neither the roulette model nor
/// the fast roulette model (for example when Claude Code is pointed at the
/// proxy globally and the user picks a regular Claude model).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnknownModelPolicy {
    /// Forward untouched to `passthrough_base_url` with the client's own
    /// credentials, so other models keep working exactly as before.
    Passthrough,
    /// Serve them through the roulette (fast lane).
    Roulette,
    /// Reject with 404.
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// The model name harnesses select.
    pub model_name: String,
    /// Model name for cheap background calls (titles, summaries...). Uses each
    /// account's `fast_model`.
    pub fast_model_name: String,
    /// Optional key clients must present (x-api-key or Bearer). Unset = any.
    pub api_key: Option<String>,
    /// Where cooldowns and session checkpoints are persisted.
    pub state_file: Option<PathBuf>,
    pub unknown_models: UnknownModelPolicy,
    pub passthrough_base_url: String,
    /// Context window reported to harnesses that need one.
    pub harness_context_tokens: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8787,
            model_name: "model-roulette".into(),
            fast_model_name: "model-roulette-fast".into(),
            api_key: None,
            state_file: None,
            unknown_models: UnknownModelPolicy::Passthrough,
            passthrough_base_url: "https://api.anthropic.com".into(),
            harness_context_tokens: 200_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RotationConfig {
    /// Cooldown after a 429 when the provider gives no reset hint. Doubles on
    /// consecutive rate limits, up to `max_backoff_secs`.
    pub rate_limit_cooldown_secs: u64,
    pub max_backoff_secs: u64,
    /// Cooldown when credits/quota are exhausted and no reset hint is given.
    pub quota_cooldown_secs: u64,
    /// Cooldown after auth/permission/model-not-found errors.
    pub auth_cooldown_secs: u64,
    /// Cooldown after overloads, 5xx and network errors.
    pub transient_cooldown_secs: u64,
    /// When every account is cooling down, wait up to this long for the first
    /// one to recover instead of returning 429 immediately.
    pub max_wait_secs: u64,
    /// Keep a session on its current account until that account fails.
    /// When false, every request starts from the top of the list.
    pub sticky: bool,
    /// Bench an account as soon as response headers say its remaining
    /// request/token budget is zero, before a 429 happens.
    pub preemptive: bool,
    /// Abort an upstream request if no bytes arrive for this long.
    pub request_timeout_secs: u64,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            rate_limit_cooldown_secs: 60,
            max_backoff_secs: 3600,
            quota_cooldown_secs: 6 * 3600,
            auth_cooldown_secs: 24 * 3600,
            transient_cooldown_secs: 20,
            max_wait_secs: 0,
            sticky: true,
            preemptive: true,
            request_timeout_secs: 600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CompactionConfig {
    pub enabled: bool,
    /// Only compact on an account switch if the conversation is at least this
    /// many (estimated) tokens. Small conversations are just handed over.
    pub trigger_tokens: u64,
    /// Recent context kept verbatim after compaction.
    pub keep_recent_tokens: u64,
    /// Max transcript tokens per summarizer call. Larger histories are
    /// summarized in a rolling fashion, chunk by chunk.
    pub chunk_tokens: u64,
    pub summary_max_tokens: u64,
    /// Compact proactively when a request would exceed this fraction of the
    /// target account's context window.
    pub proactive_ratio: f64,
    /// Tool results longer than this are truncated (head+tail) in the
    /// summarizer transcript.
    pub tool_result_max_chars: usize,
    pub tool_input_max_chars: usize,
    /// Accounts to use for summarization, in order. Empty = the account the
    /// session is switching to first, then every other available account.
    pub compactor_accounts: Vec<String>,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_tokens: 12_000,
            keep_recent_tokens: 8_000,
            chunk_tokens: 60_000,
            summary_max_tokens: 4_096,
            proactive_ratio: 0.85,
            tool_result_max_chars: 2_000,
            tool_input_max_chars: 1_200,
            compactor_accounts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthStyle {
    /// `x-api-key` for the Anthropic wire, `Authorization: Bearer` otherwise.
    Default,
    /// Always `Authorization: Bearer`.
    Bearer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccountConfig {
    /// Unique name, e.g. "claude-work".
    pub id: String,
    pub provider: ProviderKind,
    /// Model to use. Defaults to the provider preset's model.
    #[serde(default)]
    pub model: Option<String>,
    /// Cheaper model used to summarize history when compacting.
    #[serde(default)]
    pub compact_model: Option<String>,
    /// Model used for the fast lane (background calls).
    #[serde(default)]
    pub fast_model: Option<String>,
    /// Literal API key. Prefer `api_key_env`.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Environment variable that holds the API key.
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub auth: Option<AuthStyle>,
    #[serde(default)]
    pub context_window: Option<u64>,
    /// Clamp `max_tokens` to this.
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    /// `reasoning_effort` for OpenAI-wire providers that support it.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// Extra headers sent upstream.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Top-level request fields to remove before sending upstream.
    #[serde(default)]
    pub drop_fields: Vec<String>,
    /// Set to false to keep an account in the file but out of rotation.
    #[serde(default = "yes")]
    pub enabled: bool,
}

fn yes() -> bool {
    true
}

impl AccountConfig {
    pub fn preset(&self) -> Preset {
        self.provider.preset()
    }

    pub fn wire(&self) -> Wire {
        self.preset().wire
    }

    pub fn model(&self) -> &str {
        self.model
            .as_deref()
            .or(self.preset().default_model)
            .unwrap_or("default")
    }

    pub fn compact_model(&self) -> &str {
        self.compact_model
            .as_deref()
            .or(self.preset().default_compact_model)
            .or(self.fast_model.as_deref())
            .unwrap_or_else(|| self.model())
    }

    pub fn fast_model(&self) -> &str {
        self.fast_model
            .as_deref()
            .or(self.preset().default_compact_model)
            .unwrap_or_else(|| self.model())
    }

    pub fn base_url(&self) -> Option<String> {
        self.base_url
            .clone()
            .or(self.preset().base_url.map(str::to_string))
            .map(|u| u.trim_end_matches('/').to_string())
    }

    pub fn context_window(&self) -> u64 {
        self.context_window.unwrap_or(self.preset().context_window)
    }

    pub fn max_output_tokens(&self) -> Option<u64> {
        self.max_output_tokens.or(self.preset().max_output_tokens)
    }

    pub fn api_key_env_name(&self) -> Option<String> {
        self.api_key_env
            .clone()
            .or(self.preset().api_key_env.map(str::to_string))
    }

    /// Resolve the API key: literal, then the configured/preset env var.
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(k) = &self.api_key
            && !k.is_empty()
        {
            return Some(k.clone());
        }
        let var = self.api_key_env_name()?;
        std::env::var(var).ok().filter(|v| !v.is_empty())
    }
}

impl Config {
    pub fn default_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("MODEL_ROULETTE_HOME") {
            return PathBuf::from(dir);
        }
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".model-roulette")
    }

    pub fn default_path() -> PathBuf {
        Self::default_dir().join("config.toml")
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: Config =
            toml::from_str(&text).with_context(|| format!("parsing config {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for a in &self.accounts {
            if a.id.trim().is_empty() {
                bail!("an account has an empty id");
            }
            if !seen.insert(a.id.as_str()) {
                bail!("duplicate account id '{}'", a.id);
            }
            if a.base_url().is_none() {
                bail!(
                    "account '{}' (provider {}) needs a base_url",
                    a.id,
                    a.provider.as_str()
                );
            }
            if a.model.is_none() && a.preset().default_model.is_none() {
                bail!("account '{}' needs a model", a.id);
            }
        }
        if self.server.model_name == self.server.fast_model_name {
            bail!("server.model_name and server.fast_model_name must differ");
        }
        Ok(())
    }

    pub fn state_file(&self) -> PathBuf {
        self.server
            .state_file
            .clone()
            .unwrap_or_else(|| Self::default_dir().join("state.json"))
    }

    pub fn base_url(&self) -> String {
        let host = if self.server.host == "0.0.0.0" {
            "127.0.0.1"
        } else {
            &self.server.host
        };
        format!("http://{}:{}", host, self.server.port)
    }

    /// Key harnesses should present to the proxy.
    pub fn client_key(&self) -> String {
        self.server
            .api_key
            .clone()
            .unwrap_or_else(|| "model-roulette".to_string())
    }
}

pub const EXAMPLE_CONFIG: &str = r#"# model-roulette configuration
#
# Accounts are tried in the order listed. A session sticks to its current
# account until that account is rate limited / out of credits, then the
# proxy compacts the conversation and moves to the next available account.

[server]
host = "127.0.0.1"
port = 8787
model_name = "model-roulette"          # pick this model in Claude Code / Codex
fast_model_name = "model-roulette-fast"  # background/cheap calls
# api_key = "choose-a-secret"          # require this key from clients
# unknown_models = "passthrough"       # other model names go straight to Anthropic

[rotation]
rate_limit_cooldown_secs = 60    # used when the provider gives no reset time
quota_cooldown_secs = 21600      # out of credits / quota
max_wait_secs = 0                # wait for a recovering account instead of 429

[compaction]
enabled = true
trigger_tokens = 12000           # compact on switch only above this size
keep_recent_tokens = 8000        # recent context kept verbatim

# Model IDs below are examples - replace with the models you have access to.

[[accounts]]
id = "claude"
provider = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-5-5"
compact_model = "claude-haiku-5-5"

[[accounts]]
id = "openai"
provider = "openai"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.6"
# reasoning_effort = "medium"

[[accounts]]
id = "gemini"
provider = "gemini"
api_key_env = "GEMINI_API_KEY"
model = "gemini-3.1-pro-preview"
compact_model = "gemini-3.5-flash"

[[accounts]]
id = "meta"
provider = "meta"
api_key_env = "META_API_KEY"
model = "muse-spark-1.3"

[[accounts]]
id = "deepseek"
provider = "deepseek"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"

# More of the same provider (a second account) works too:
# [[accounts]]
# id = "claude-2"
# provider = "anthropic"
# api_key_env = "ANTHROPIC_API_KEY_2"
#
# Any OpenAI-compatible endpoint:
# [[accounts]]
# id = "openrouter"
# provider = "openai_compatible"
# base_url = "https://openrouter.ai/api/v1"
# api_key_env = "OPENROUTER_API_KEY"
# model = "qwen/qwen3-coder"
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_config_parses() {
        let cfg: Config = toml::from_str(EXAMPLE_CONFIG).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.accounts.len(), 5);
        assert_eq!(cfg.accounts[2].compact_model(), "gemini-3.5-flash");
        assert_eq!(
            cfg.accounts[1].base_url().unwrap(),
            "https://api.openai.com/v1"
        );
    }

    #[test]
    fn compatible_needs_base_url() {
        let cfg: Config =
            toml::from_str("[[accounts]]\nid='x'\nprovider='openai_compatible'\nmodel='m'\n")
                .unwrap();
        assert!(cfg.validate().is_err());
    }
}
