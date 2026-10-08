//! Claude Code integration (Anthropic Messages protocol).
//!
//! Claude Code honours `ANTHROPIC_BASE_URL`, and `ANTHROPIC_CUSTOM_MODEL_OPTION`
//! adds an entry to its `/model` picker. Requests for other models are passed
//! through to Anthropic untouched (server.unknown_models = "passthrough"), so
//! the proxy can stay configured permanently.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde_json::{Value, json};

use super::{Harness, LaunchSpec, Protocol};
use crate::config::Config;

pub struct ClaudeCode;

impl ClaudeCode {
    fn env(&self, cfg: &Config, launch: bool) -> Vec<(String, String)> {
        let s = &cfg.server;
        let mut env = vec![
            ("ANTHROPIC_BASE_URL".to_string(), cfg.base_url()),
            ("ANTHROPIC_CUSTOM_MODEL_OPTION".to_string(), s.model_name.clone()),
            (
                "ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION".to_string(),
                "Model Roulette - rotates across your accounts on rate limits".to_string(),
            ),
        ];
        if launch {
            env.extend([
                ("ANTHROPIC_MODEL".to_string(), s.model_name.clone()),
                ("ANTHROPIC_AUTH_TOKEN".to_string(), cfg.client_key()),
                // Background calls (titles, summaries) go through the roulette too.
                ("ANTHROPIC_DEFAULT_HAIKU_MODEL".to_string(), s.fast_model_name.clone()),
                ("ANTHROPIC_SMALL_FAST_MODEL".to_string(), s.fast_model_name.clone()),
                ("CLAUDE_CODE_MAX_CONTEXT_TOKENS".to_string(), s.harness_context_tokens.to_string()),
            ]);
        }
        env
    }

    pub fn settings_path() -> PathBuf {
        if let Ok(dir) = std::env::var("CLAUDE_CONFIG_DIR") {
            return PathBuf::from(dir).join("settings.json");
        }
        dirs::home_dir().unwrap_or_default().join(".claude").join("settings.json")
    }
}

impl Harness for ClaudeCode {
    fn id(&self) -> &'static str {
        "claude-code"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["claude", "claude_code", "cc"]
    }

    fn display_name(&self) -> &'static str {
        "Claude Code"
    }

    fn protocol(&self) -> Protocol {
        Protocol::AnthropicMessages
    }

    fn launch(&self, cfg: &Config, extra_args: &[String]) -> LaunchSpec {
        LaunchSpec {
            program: "claude".into(),
            args: extra_args.to_vec(),
            env: self.env(cfg, true),
            // An API key in the environment would take precedence over the
            // proxy token and trigger an auth-conflict warning.
            env_remove: vec!["ANTHROPIC_API_KEY".into()],
        }
    }

    fn setup_instructions(&self, cfg: &Config) -> String {
        let s = &cfg.server;
        format!(
            "Claude Code\n\
             ===========\n\
             One-off (no login needed, everything goes through the roulette):\n\
             \n    model-roulette launch claude-code [-- <claude args>]\n\n\
             or set these yourself:\n\n    export ANTHROPIC_BASE_URL={base}\n    export ANTHROPIC_AUTH_TOKEN={key}\n    export ANTHROPIC_MODEL={model}\n    export ANTHROPIC_DEFAULT_HAIKU_MODEL={fast}\n    claude\n\n\
             Permanent, keeping your normal Claude login for other models\n\
             (`model-roulette install claude-code` writes this to {path}):\n\n{snippet}\n\n\
             Then pick \"{model}\" in /model. Other models are passed through to\n\
             Anthropic with your own credentials (server.unknown_models = \"passthrough\").",
            base = cfg.base_url(),
            key = cfg.client_key(),
            model = s.model_name,
            fast = s.fast_model_name,
            path = Self::settings_path().display(),
            snippet = serde_json::to_string_pretty(&json!({"env": self.env(cfg, false).into_iter()
                .map(|(k, v)| (k, Value::String(v))).collect::<serde_json::Map<_, _>>()})).unwrap()
                .lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n"),
        )
    }

    fn install(&self, cfg: &Config) -> Result<String> {
        let path = Self::settings_path();
        let mut settings: Value = match std::fs::read_to_string(&path) {
            Ok(s) if !s.trim().is_empty() => {
                serde_json::from_str(&s).with_context(|| format!("parsing {}", path.display()))?
            }
            _ => json!({}),
        };
        if !settings.is_object() {
            anyhow::bail!("{} is not a JSON object", path.display());
        }
        let env = settings
            .as_object_mut()
            .unwrap()
            .entry("env")
            .or_insert_with(|| json!({}));
        let env = env.as_object_mut().context("settings.env is not an object")?;
        for (k, v) in self.env(cfg, false) {
            env.insert(k, json!(v));
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if path.exists() {
            std::fs::copy(&path, path.with_extension("json.bak"))?;
        }
        std::fs::write(&path, serde_json::to_string_pretty(&settings)? + "\n")?;
        Ok(format!(
            "updated {} (backup: settings.json.bak). Restart Claude Code and pick \"{}\" in /model.",
            path.display(),
            cfg.server.model_name
        ))
    }
}
