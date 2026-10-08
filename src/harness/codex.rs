//! Codex CLI integration (OpenAI Responses protocol).
//!
//! Codex selects providers via `model_providers.<id>` in
//! `~/.codex/config.toml`; a profile bundles provider + model so
//! `codex --profile roulette` starts on the roulette model.

use std::path::PathBuf;

use anyhow::Result;

use super::{Harness, LaunchSpec, Protocol};
use crate::config::Config;

pub struct Codex;

pub const PROVIDER_ID: &str = "model_roulette";
pub const PROFILE: &str = "roulette";
pub const KEY_ENV: &str = "MODEL_ROULETTE_API_KEY";

impl Codex {
    pub fn config_path() -> PathBuf {
        let home = std::env::var("CODEX_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs::home_dir().unwrap_or_default().join(".codex"));
        home.join("config.toml")
    }

    fn provider_inline(cfg: &Config) -> String {
        let mut s = format!(
            "{{name=\"Model Roulette\",base_url=\"{}/v1\",wire_api=\"responses\"",
            cfg.base_url()
        );
        if cfg.server.api_key.is_some() {
            s.push_str(&format!(",env_key=\"{KEY_ENV}\""));
        }
        s.push('}');
        s
    }

    pub fn toml_snippet(cfg: &Config) -> String {
        let mut s = format!(
            "[model_providers.{PROVIDER_ID}]\nname = \"Model Roulette\"\nbase_url = \"{}/v1\"\nwire_api = \"responses\"\n",
            cfg.base_url()
        );
        if cfg.server.api_key.is_some() {
            s.push_str(&format!("env_key = \"{KEY_ENV}\"\n"));
        }
        s.push_str(&format!(
            "\n[profiles.{PROFILE}]\nmodel = \"{}\"\nmodel_provider = \"{PROVIDER_ID}\"\nmodel_context_window = {}\n",
            cfg.server.model_name, cfg.server.harness_context_tokens
        ));
        s
    }
}

impl Harness for Codex {
    fn id(&self) -> &'static str {
        "codex"
    }

    fn aliases(&self) -> &'static [&'static str] {
        &["codex-cli"]
    }

    fn display_name(&self) -> &'static str {
        "Codex CLI"
    }

    fn protocol(&self) -> Protocol {
        Protocol::OpenAiResponses
    }

    fn launch(&self, cfg: &Config, extra_args: &[String]) -> LaunchSpec {
        let mut args = vec![
            "-c".to_string(),
            format!("model_providers.{PROVIDER_ID}={}", Self::provider_inline(cfg)),
            "-c".to_string(),
            format!("model_provider=\"{PROVIDER_ID}\""),
            "-c".to_string(),
            format!("model=\"{}\"", cfg.server.model_name),
            "-c".to_string(),
            format!("model_context_window={}", cfg.server.harness_context_tokens),
        ];
        args.extend(extra_args.iter().cloned());
        LaunchSpec {
            program: "codex".into(),
            args,
            env: vec![(KEY_ENV.to_string(), cfg.client_key())],
            env_remove: vec![],
        }
    }

    fn setup_instructions(&self, cfg: &Config) -> String {
        format!(
            "Codex CLI\n\
             =========\n\
             One-off:\n\n    model-roulette launch codex [-- <codex args>]\n\n\
             Permanent (`model-roulette install codex` appends this to {path}):\n\n{snippet}\n\
             Then run `codex --profile {PROFILE}`.{key}",
            path = Self::config_path().display(),
            snippet = Self::toml_snippet(cfg).lines().map(|l| format!("    {l}")).collect::<Vec<_>>().join("\n"),
            key = if cfg.server.api_key.is_some() {
                format!("\nExport {KEY_ENV} with your server.api_key first.")
            } else {
                String::new()
            }
        )
    }

    fn install(&self, cfg: &Config) -> Result<String> {
        let path = Self::config_path();
        let existing = std::fs::read_to_string(&path).unwrap_or_default();
        if existing.contains(&format!("[model_providers.{PROVIDER_ID}]")) {
            return Ok(format!(
                "{} already has a [model_providers.{PROVIDER_ID}] section; left unchanged. Run `codex --profile {PROFILE}`.",
                path.display()
            ));
        }
        // Validate that the result is still TOML before writing.
        let mut combined = existing.clone();
        if !combined.is_empty() && !combined.ends_with('\n') {
            combined.push('\n');
        }
        combined.push_str("\n# Added by model-roulette\n");
        combined.push_str(&Self::toml_snippet(cfg));
        toml::from_str::<toml::Table>(&combined)
            .map_err(|e| anyhow::anyhow!("refusing to write {}: result would not parse: {e}", path.display()))?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if path.exists() {
            std::fs::copy(&path, path.with_extension("toml.bak"))?;
        }
        std::fs::write(&path, combined)?;
        Ok(format!("updated {}. Run `codex --profile {PROFILE}`.", path.display()))
    }
}
