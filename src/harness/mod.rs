//! Coding-harness integrations.
//!
//! A harness is a client program (Claude Code, Codex, ...) that talks to
//! model-roulette through one of the front-end protocols. Integrating a
//! harness means telling it to use the proxy: environment variables and CLI
//! flags for a one-off launch, or a persistent config change.
//!
//! To add a harness: implement [`Harness`] in a new module and register it in
//! [`registry`]. If it speaks Anthropic Messages, OpenAI Responses or OpenAI
//! Chat Completions, no proxy-side protocol work is needed.

pub mod claude_code;
pub mod codex;

use anyhow::Result;

use crate::config::Config;

/// Which client-facing API a harness uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    AnthropicMessages,
    OpenAiResponses,
    OpenAiChat,
}

impl Protocol {
    pub fn endpoint(self) -> &'static str {
        match self {
            Protocol::AnthropicMessages => "/v1/messages",
            Protocol::OpenAiResponses => "/v1/responses",
            Protocol::OpenAiChat => "/v1/chat/completions",
        }
    }
}

/// How to start the harness against the proxy.
#[derive(Debug, Clone, Default)]
pub struct LaunchSpec {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub env_remove: Vec<String>,
}

pub trait Harness: Send + Sync {
    /// CLI name, e.g. `claude-code`.
    fn id(&self) -> &'static str;
    /// Extra accepted names, e.g. `claude`.
    fn aliases(&self) -> &'static [&'static str] {
        &[]
    }
    fn display_name(&self) -> &'static str;
    fn protocol(&self) -> Protocol;
    /// Program + env + args to run the harness through the proxy, with the
    /// roulette model selected. `extra_args` are passed through.
    fn launch(&self, cfg: &Config, extra_args: &[String]) -> LaunchSpec;
    /// Human-readable instructions/snippet for manual setup.
    fn setup_instructions(&self, cfg: &Config) -> String;
    /// Persistently configure the harness so the roulette model shows up in
    /// its model picker. Returns a description of what changed.
    fn install(&self, cfg: &Config) -> Result<String>;
}

pub fn registry() -> Vec<Box<dyn Harness>> {
    vec![Box::new(claude_code::ClaudeCode), Box::new(codex::Codex)]
}

pub fn find(name: &str) -> Option<Box<dyn Harness>> {
    registry()
        .into_iter()
        .find(|h| h.id() == name || h.aliases().contains(&name))
}
