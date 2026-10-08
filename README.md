# model-roulette

**One model name that rotates across all your LLM accounts.**

Pick `model-roulette` in Claude Code or Codex and keep working. When the
current account hits a rate limit or runs out of credits, model-roulette
compacts the conversation and continues on the next account (Anthropic,
OpenAI, Gemini, Meta, DeepSeek, xAI, or any compatible endpoint). The
harness never notices. Benched accounts come back into rotation when their
limit resets.

```
 Claude Code ─┐                       ┌─► claude   (Anthropic)    429 ✗ cooling 5m
              │   ┌───────────────┐   ├─► openai   (OpenAI)       ✓ ← session moved here
 Codex ───────┼──►│ model-roulette│───┼─► gemini   (Google)
              │   │  :8787        │   ├─► meta     (Muse)
 other ───────┘   └───────────────┘   └─► deepseek (DeepSeek)
```

Written in Rust (tokio + axum + reqwest) as a single small binary.

---

## Contents

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Harness setup](#harness-setup)
- [Providers](#providers)
- [Commands](#commands)
- [Testing without API keys](#testing-without-api-keys)
- [Limitations](#limitations)

## Quick start

```sh
# 1. Build & install
cargo install --path .

# 2. Create ~/.model-roulette/config.toml and list your accounts in order
model-roulette init
$EDITOR ~/.model-roulette/config.toml

# 3. Export the keys referenced by api_key_env
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... META_API_KEY=... DEEPSEEK_API_KEY=...

# 4a. One-off: start the harness wired to the proxy (starts the proxy if needed)
model-roulette launch claude-code
model-roulette launch codex

# 4b. Or run the proxy yourself and configure the harness permanently
model-roulette serve &
model-roulette install claude-code   # adds "model-roulette" to Claude Code's /model picker
model-roulette install codex         # adds a `roulette` profile:  codex --profile roulette
```

`model-roulette status` shows which accounts are ready, which are cooling
down (and for how long), and where each session currently is.

## How it works

### Rotation

* Accounts are tried **in the order listed** in the config.
* A **new session** starts on the first account that isn't cooling down.
* An **existing session sticks** to its current account (no needless
  switches, warm prompt caches). When that account fails, the session moves
  *forward* to the next available account, wrapping around the list.
* Failures are classified from status codes, error bodies and headers:

| Failure | Examples | What happens |
|---|---|---|
| Rate limited | 429, `rate_limit_error`, Gemini `RESOURCE_EXHAUSTED` with retry delay | benched until the provider's reset time (`retry-after`, `anthropic-ratelimit-*-reset`, `x-ratelimit-reset-*`, Gemini `retryDelay`); without a hint, 60s with exponential backoff |
| Out of credits / quota | 402, `insufficient_quota`, "credit balance is too low", "Insufficient Balance" | benched for `quota_cooldown_secs` (6h default) or the provider's hint |
| Account error | 401, 403, 404 (bad key, no access, unknown model) | benched for 24h |
| Transient | 5xx, 529 overloaded, network errors | benched for 20s |
| Context overflow | "prompt is too long", `context_length_exceeded` | compact and retry the **same** account |
| Bad request | other 400s | returned to the client; no failover |

* Accounts are also **benched pre-emptively** when a successful response's
  headers say the remaining request/token budget is `0`.
* Failover is **transparent** as long as nothing has been streamed to the
  client yet, which covers HTTP errors and in-stream errors that arrive
  before the first token. If a stream dies *after* output started, the
  client gets a normal retryable error (`overloaded_error` /
  `response.failed`). The account is benched, so the harness's automatic
  retry lands on the next account.
* If **every** account is cooling down, the proxy returns `429` with a
  `retry-after` for the earliest recovery, or waits up to
  `rotation.max_wait_secs` if you set it.
* Cooldowns and sessions are **persisted** to
  `~/.model-roulette/state.json`, so restarts and new sessions know which
  accounts are still benched.

### Compaction on switch

Moving a long conversation to a new account would make the new model
re-read everything, and maybe exhaust that account too. So when a session
switches accounts (or a request would overflow the target's context
window), model-roulette **compacts** it:

1. It picks a cut point that keeps the most recent
   `keep_recent_tokens` verbatim. The cut always lands on an assistant turn,
   so tool calls and results stay paired.
2. It summarizes everything before the cut **with a cheap model on an
   account that isn't rate limited**: normally the account being switched
   to, using its `compact_model` (e.g. Haiku / Gemini Flash). The account
   that just hit its limit is never asked to summarize.
3. Tool outputs are truncated (head + tail) before summarizing. Very long
   histories are summarized in a **rolling** fashion, chunk by chunk
   (`chunk_tokens`), so no single summarizer call has to read everything.
4. The user's most recent request is always kept **verbatim**.
5. If no summarizer is reachable at all, an **extractive fallback**
   (user messages, tool activity, latest notes) is used. A switch never
   fails because of compaction.

Harnesses keep sending their full history every turn and don't know about
any of this. So compaction is **virtual**: the proxy stores a *checkpoint*
(a hash of the compacted prefix plus the summary). Every later request
whose history starts with that prefix has it replaced by the summary before
going upstream, with no further summarizer calls. If the harness compacts
on its own (e.g. Claude Code `/compact`), the prefix no longer matches and
the checkpoint is dropped automatically. Sub-agents that share a session id
get separate checkpoints, keyed by the conversation's first message.

Small conversations (below `trigger_tokens`) are handed over as-is.

### Cross-provider details handled for you

* Anthropic ⇄ OpenAI Chat ⇄ OpenAI Responses translation: tool calls and
  results, images, system prompts, Claude Code's mid-conversation system
  messages, Codex namespaced, custom (freeform) and `local_shell` tools,
  and reasoning/thinking blocks.
* Claude **thinking signatures** are tracked per account. Blocks signed by
  another account are dropped before reaching a different Anthropic
  account, and thinking is disabled for that one request if Anthropic's
  "tool loop must start with thinking" rule would otherwise fail.
* **Gemini thought signatures** are cached by tool-call id and replayed;
  history from other models gets Google's documented skip-validator value.
* **DeepSeek `reasoning_content`** is echoed back inside tool loops.
* Tool names are mapped to each provider's charset/length limits and back.
* Gemini-incompatible JSON-schema keywords are stripped.

## Configuration

`~/.model-roulette/config.toml` (override with `--config` or
`MODEL_ROULETTE_CONFIG`; `MODEL_ROULETTE_HOME` moves the whole directory):

```toml
[server]
host = "127.0.0.1"
port = 8787
model_name = "model-roulette"            # what you select in the harness
fast_model_name = "model-roulette-fast"  # cheap background calls -> each account's fast_model
# api_key = "secret"                     # require this key from clients
unknown_models = "passthrough"           # passthrough | roulette | reject
passthrough_base_url = "https://api.anthropic.com"
harness_context_tokens = 200000          # context window advertised to harnesses

[rotation]
rate_limit_cooldown_secs = 60     # when no reset hint; doubles per consecutive 429
max_backoff_secs = 3600
quota_cooldown_secs = 21600
auth_cooldown_secs = 86400
transient_cooldown_secs = 20
max_wait_secs = 0                 # >0: wait for the first recovering account instead of 429
sticky = true                     # false: always start from the top of the list
preemptive = true                 # bench when headers say remaining budget is 0
request_timeout_secs = 600        # idle timeout for upstream streams

[compaction]
enabled = true
trigger_tokens = 12000            # compact on switch only above this size
keep_recent_tokens = 8000
chunk_tokens = 60000              # max transcript per summarizer call
summary_max_tokens = 4096
proactive_ratio = 0.85            # compact when a request exceeds 85% of the target's window
tool_result_max_chars = 2000
tool_input_max_chars = 1200
compactor_accounts = []           # e.g. ["gemini"] to always summarize on one account

[[accounts]]
id = "claude"                     # unique name
provider = "anthropic"            # see Providers
api_key_env = "ANTHROPIC_API_KEY" # or api_key = "..."
model = "claude-sonnet-5-5"
compact_model = "claude-haiku-5-5"
# fast_model = "claude-haiku-5-5"
# base_url = "https://..."        # override endpoint
# context_window = 200000
# max_output_tokens = 32000       # clamp max_tokens
# reasoning_effort = "medium"     # OpenAI-wire providers
# headers = { "x-foo" = "bar" }
# drop_fields = ["context_management"]
# auth = "bearer"                 # Anthropic-wire gateways that want Bearer auth
# enabled = false
```

Several accounts with the same provider are fine (e.g. two Anthropic keys
with different `api_key_env`s).

> **Model IDs in the presets and examples are starting points.** Provider
> model names change often; set `model` / `compact_model` to the IDs your
> accounts actually have.

## Harness setup

| Harness | Protocol | One-off | Permanent |
|---|---|---|---|
| Claude Code | Anthropic Messages (`/v1/messages`) | `model-roulette launch claude-code` | `model-roulette install claude-code`, then pick **model-roulette** in `/model` |
| Codex CLI | OpenAI Responses (`/v1/responses`) | `model-roulette launch codex` | `model-roulette install codex`, then `codex --profile roulette` |
| Anything else | OpenAI Chat Completions (`/v1/chat/completions`) or either of the above | point its base URL at `http://127.0.0.1:8787/v1`, model `model-roulette` | — |

`model-roulette setup <harness>` prints the exact settings if you'd rather
configure things by hand.

**Claude Code details.** `install` adds `ANTHROPIC_BASE_URL` and
`ANTHROPIC_CUSTOM_MODEL_OPTION` to `~/.claude/settings.json` (a backup is
written). Your normal login keeps working: requests for any *other* model
are passed through untouched to Anthropic with your own credentials
(`unknown_models = "passthrough"`). `launch` instead runs Claude Code fully
on the roulette: no Anthropic login needed, and background calls use the
fast lane. Claude Code sends a session id header, so stickiness and
checkpoints are per Claude Code session.

**Codex details.** `install` appends a `[model_providers.model_roulette]`
table to `~/.codex/config.toml` and writes `~/.codex/roulette.config.toml`
(the Codex ≥0.134 profile format). Codex's session id is used for
stickiness. Codex compacts locally for custom providers, so no
`/responses/compact` endpoint is needed.

### Adding a harness

Implement the `Harness` trait (`src/harness/mod.rs`) and register it in
`registry()`. If the harness speaks one of the three supported protocols,
no proxy changes are needed. See [AGENTS.md](AGENTS.md).

## Providers

| `provider` | Wire | Default base URL | Key env |
|---|---|---|---|
| `anthropic` | Anthropic Messages | `https://api.anthropic.com` | `ANTHROPIC_API_KEY` |
| `openai` | Chat Completions | `https://api.openai.com/v1` | `OPENAI_API_KEY` |
| `gemini` | Chat Completions (OpenAI-compatible) | `https://generativelanguage.googleapis.com/v1beta/openai` | `GEMINI_API_KEY` |
| `meta` | Chat Completions (OpenAI-compatible) | `https://api.meta.ai/v1` | `META_API_KEY` |
| `deepseek` | Chat Completions | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
| `xai` | Chat Completions | `https://api.x.ai/v1` | `XAI_API_KEY` |
| `openai_compatible` | Chat Completions | set `base_url` | set `api_key_env` |
| `anthropic_compatible` | Anthropic Messages | set `base_url` | set `api_key_env` |

`model-roulette providers` prints the presets.

## Commands

```
model-roulette init [--force]          write an example config
model-roulette serve [--port N]        run the proxy
model-roulette status [--json]         accounts, cooldowns, sessions
model-roulette reset [ACCOUNT]         clear cooldowns
model-roulette providers               list provider presets
model-roulette harnesses               list supported harnesses
model-roulette setup <harness>         print manual setup instructions
model-roulette install <harness>       configure a harness permanently
model-roulette launch <harness> [-- args]   run a harness through the proxy
model-roulette mock-upstream [--port]  fake provider for testing
```

HTTP endpoints: `POST /v1/messages`, `POST /v1/messages/count_tokens`,
`POST /v1/responses`, `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health`, `GET /roulette/status`, `POST /roulette/reset[?account=id]`.
Responses carry `x-model-roulette-account` / `x-model-roulette-model`
headers saying who answered. Logs go to stderr (`RUST_LOG=model_roulette=debug`
for more), or to `~/.model-roulette/proxy.log` under `launch`.

## Testing without API keys

`model-roulette mock-upstream` runs a fake provider that speaks both the
Anthropic and OpenAI wires. Its behaviour is chosen by the API key, for
example `api_key = "claude?rl_after=3&retry_after=300"` starts returning
429s after 3 requests. See `src/mock.rs` for all options (`quota`,
`fail=500`, `thought_sig`, `echo_reasoning`, `midstream_error`,
`ctx_limit=N`, ...). Send `CALL_TOOL <name> <json>` to make it call a tool.

```toml
[[accounts]]
id = "claude"
provider = "anthropic"
base_url = "http://127.0.0.1:9999"
api_key = "claude?rl_after=3&retry_after=300"

[[accounts]]
id = "openai"
provider = "openai"
base_url = "http://127.0.0.1:9999/v1"
api_key = "openai"
```

```sh
cargo test     # unit + end-to-end tests (proxy + mock providers over HTTP)
```

## Limitations

* **API keys only.** Consumer subscriptions (Claude Pro/Max, ChatGPT
  Plus/Pro) can't be pooled this way. Passthrough of your own Claude Code
  login for *non-roulette* models still works.
* **Provider-hosted tools** (Anthropic web search/code execution, OpenAI
  `web_search`, ...) only work on their own provider and are dropped when
  translating to another one.
* **Lossy across providers.** Reasoning/thinking isn't portable between
  providers. It's shown to the harness but not replayed to a different
  provider, so a new model only sees the visible conversation plus the
  summary.
* Token counts used for thresholds are estimates (~4 chars/token).
* Two requests from the *same* conversation that trigger a switch at the
  same moment may both compact; the second checkpoint simply replaces the
  first.
