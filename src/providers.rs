//! Provider presets.
//!
//! Every upstream account belongs to a [`ProviderKind`]. A kind resolves to a
//! [`Preset`]: the wire protocol it speaks, its default base URL, sensible
//! default models and a handful of provider quirks the adapters need to know
//! about. Adding a provider that speaks an existing wire protocol is a matter
//! of adding a variant and a preset here.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Anthropic Messages API (Claude).
    Anthropic,
    /// OpenAI (GPT) via Chat Completions.
    Openai,
    /// Google Gemini via its OpenAI-compatible endpoint.
    Gemini,
    /// Meta Model API (Muse) via its OpenAI-compatible endpoint.
    Meta,
    /// DeepSeek via its OpenAI-compatible endpoint.
    Deepseek,
    /// xAI (Grok) via its OpenAI-compatible endpoint.
    Xai,
    /// Any other OpenAI Chat Completions compatible endpoint (OpenRouter,
    /// Mistral, Groq, Together, Ollama, vLLM, ...). Requires `base_url`.
    OpenaiCompatible,
    /// Any other Anthropic Messages compatible endpoint. Requires `base_url`.
    AnthropicCompatible,
}

/// The wire protocol an upstream speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// `POST {base}/v1/messages`
    Anthropic,
    /// `POST {base}/chat/completions`
    OpenAiChat,
}

#[derive(Debug, Clone)]
pub struct Preset {
    pub display: &'static str,
    pub wire: Wire,
    /// Base URL. For the Anthropic wire it excludes `/v1`; for the OpenAI wire
    /// it includes the version segment (e.g. `https://api.openai.com/v1`).
    pub base_url: Option<&'static str>,
    pub default_model: Option<&'static str>,
    pub default_compact_model: Option<&'static str>,
    pub api_key_env: Option<&'static str>,
    pub context_window: u64,
    pub max_output_tokens: Option<u64>,
    pub quirks: Quirks,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Quirks {
    /// Send `max_completion_tokens` instead of `max_tokens`.
    pub max_completion_tokens: bool,
    /// Never send `temperature`/`top_p` (reasoning models reject them).
    pub drop_sampling: bool,
    /// Echo `reasoning_content` back on assistant messages that carry tool
    /// calls (DeepSeek thinking mode requires it inside a tool loop).
    pub echo_reasoning_content: bool,
    /// Round-trip Gemini thought signatures on tool calls.
    pub gemini_thought_signatures: bool,
    /// Strip JSON-schema keywords Gemini rejects.
    pub gemini_schema: bool,
    /// Forward the client's `anthropic-beta` header and Claude-specific
    /// request fields (Anthropic wire only).
    pub forward_anthropic_extras: bool,
}

impl ProviderKind {
    pub fn preset(self) -> Preset {
        match self {
            ProviderKind::Anthropic => Preset {
                display: "Anthropic",
                wire: Wire::Anthropic,
                base_url: Some("https://api.anthropic.com"),
                default_model: Some("claude-sonnet-5-5"),
                default_compact_model: Some("claude-haiku-5-5"),
                api_key_env: Some("ANTHROPIC_API_KEY"),
                context_window: 200_000,
                max_output_tokens: None,
                quirks: Quirks {
                    forward_anthropic_extras: true,
                    ..Default::default()
                },
            },
            ProviderKind::Openai => Preset {
                display: "OpenAI",
                wire: Wire::OpenAiChat,
                base_url: Some("https://api.openai.com/v1"),
                default_model: Some("gpt-5.6"),
                default_compact_model: None,
                api_key_env: Some("OPENAI_API_KEY"),
                context_window: 400_000,
                max_output_tokens: None,
                quirks: Quirks {
                    max_completion_tokens: true,
                    drop_sampling: true,
                    ..Default::default()
                },
            },
            ProviderKind::Gemini => Preset {
                display: "Google Gemini",
                wire: Wire::OpenAiChat,
                base_url: Some("https://generativelanguage.googleapis.com/v1beta/openai"),
                default_model: Some("gemini-3.1-pro-preview"),
                default_compact_model: Some("gemini-3.5-flash"),
                api_key_env: Some("GEMINI_API_KEY"),
                context_window: 1_000_000,
                max_output_tokens: Some(65_536),
                quirks: Quirks {
                    gemini_thought_signatures: true,
                    gemini_schema: true,
                    ..Default::default()
                },
            },
            ProviderKind::Meta => Preset {
                display: "Meta (Muse)",
                wire: Wire::OpenAiChat,
                base_url: Some("https://api.meta.ai/v1"),
                default_model: Some("muse-spark-1.3"),
                default_compact_model: None,
                api_key_env: Some("META_API_KEY"),
                context_window: 1_000_000,
                max_output_tokens: None,
                quirks: Quirks::default(),
            },
            ProviderKind::Deepseek => Preset {
                display: "DeepSeek",
                wire: Wire::OpenAiChat,
                base_url: Some("https://api.deepseek.com/v1"),
                default_model: Some("deepseek-chat"),
                default_compact_model: Some("deepseek-chat"),
                api_key_env: Some("DEEPSEEK_API_KEY"),
                context_window: 128_000,
                max_output_tokens: Some(8_192),
                quirks: Quirks {
                    echo_reasoning_content: true,
                    ..Default::default()
                },
            },
            ProviderKind::Xai => Preset {
                display: "xAI (Grok)",
                wire: Wire::OpenAiChat,
                base_url: Some("https://api.x.ai/v1"),
                default_model: Some("grok-4.5"),
                default_compact_model: None,
                api_key_env: Some("XAI_API_KEY"),
                context_window: 256_000,
                max_output_tokens: None,
                quirks: Quirks {
                    drop_sampling: false,
                    ..Default::default()
                },
            },
            ProviderKind::OpenaiCompatible => Preset {
                display: "OpenAI-compatible",
                wire: Wire::OpenAiChat,
                base_url: None,
                default_model: None,
                default_compact_model: None,
                api_key_env: None,
                context_window: 128_000,
                max_output_tokens: None,
                quirks: Quirks::default(),
            },
            ProviderKind::AnthropicCompatible => Preset {
                display: "Anthropic-compatible",
                wire: Wire::Anthropic,
                base_url: None,
                default_model: None,
                default_compact_model: None,
                api_key_env: None,
                context_window: 128_000,
                max_output_tokens: None,
                quirks: Quirks::default(),
            },
        }
    }

    pub fn all() -> &'static [ProviderKind] {
        &[
            ProviderKind::Anthropic,
            ProviderKind::Openai,
            ProviderKind::Gemini,
            ProviderKind::Meta,
            ProviderKind::Deepseek,
            ProviderKind::Xai,
            ProviderKind::OpenaiCompatible,
            ProviderKind::AnthropicCompatible,
        ]
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Openai => "openai",
            ProviderKind::Gemini => "gemini",
            ProviderKind::Meta => "meta",
            ProviderKind::Deepseek => "deepseek",
            ProviderKind::Xai => "xai",
            ProviderKind::OpenaiCompatible => "openai_compatible",
            ProviderKind::AnthropicCompatible => "anthropic_compatible",
        }
    }
}
