# AGENTS.md

Guidance for AI agents (and humans) working on this repository. Read this
before changing code; keep it up to date when the design changes.

## What this is

`model-roulette` is a local HTTP proxy, written in Rust. Coding harnesses
(Claude Code, Codex, ...) select one model name, `model-roulette`. The
proxy serves it by rotating across many upstream provider accounts. When the
active account is rate limited or out of credits, the proxy benches it,
compacts the conversation, and continues on the next account. Harnesses
never see the switch.

## Commands

```sh
cargo build                      # debug build
cargo test                       # unit tests + tests/integration.rs (must stay green)
cargo clippy --all-targets       # must be warning-free
cargo fmt                        # rustfmt.toml: max_width = 100
cargo run -- mock-upstream       # fake provider on :9999 (see src/mock.rs)
cargo run -- serve --config <file>
```

No network or API keys are needed for tests: integration tests spin up the
proxy and the mock provider on random ports.

## Architecture

```
harness ──► frontend ──► canonical request ──► roulette (router) ──► upstream adapter ──► provider
        ◄── encoder  ◄── canonical events  ◄──────────────────────◄── stream converter ◄──
```

**Canonical format = the Anthropic Messages API.** Requests are Anthropic
request JSON (`serde_json::Value`). Responses are streams of Anthropic SSE
event objects (`message_start`, `content_block_*`, `message_delta`,
`message_stop`). It is the richest of the wires (typed blocks, thinking,
tool use/results), so translating through it loses the least. Front ends
convert *into* it; upstream adapters convert *out of* it.

| Module | Responsibility |
|---|---|
| `src/main.rs` | CLI (clap): init/serve/status/reset/setup/install/launch/mock-upstream |
| `src/config.rs` | TOML config types, defaults, validation, `EXAMPLE_CONFIG` |
| `src/providers.rs` | `ProviderKind` → `Preset` (wire, base URL, default models, quirks) |
| `src/canonical.rs` | helpers on canonical requests: digests/hashes, token estimates, `sanitize_messages`, `Accumulator` (events → message) |
| `src/ratelimit.rs` | classify upstream failures (`FailureKind`), parse reset hints from headers/bodies |
| `src/state.rs` | persisted state (cooldowns, sessions, checkpoints, signature & tool-extra caches), debounced JSON flush |
| `src/roulette.rs` | **the router**: account selection, stickiness, failover loop, peek-before-commit, compaction trigger, summarizer calls, status |
| `src/compaction.rs` | boundary choice, transcript rendering, chunking, prompts, fallback summary, checkpoint apply/make |
| `src/upstream/anthropic.rs` | Anthropic wire: request cleanup (foreign thinking, beta headers, tool ids), SSE passthrough |
| `src/upstream/openai_chat.rs` | Chat Completions wire: request conversion, `ChunkConverter` (chunks → canonical events), provider quirks |
| `src/frontend/anthropic.rs` | `/v1/messages`, `/count_tokens`, passthrough of non-roulette models |
| `src/frontend/responses.rs` | `/v1/responses` (Codex): input items/tools → canonical; `ResponsesEncoder` |
| `src/frontend/chat.rs` | `/v1/chat/completions` for generic harnesses |
| `src/harness/` | `Harness` trait + Claude Code and Codex integrations (launch env/args, install) |
| `src/server.rs` | axum router, `start()` (used by tests and `launch`), `serve()` |
| `src/mock.rs` | scriptable fake provider (Anthropic + OpenAI wires) for tests and demos |

## Key design decisions (keep these unless deliberately redesigning)

1. **Sticky sessions, forward rotation.** A session stays on its account
   until that account fails, then moves to the *next* account in config order
   (wrapping). New sessions start at the first available account. Sessions
   are keyed by the harness session id (`X-Claude-Code-Session-Id`, Codex
   `session-id`/`prompt_cache_key`). Without one, the conversation
   fingerprint is used.
2. **Peek before commit.** The router reads upstream events until the first
   content event before returning a stream to the front end. Any failure up
   to that point fails over transparently. After it, errors go to the client
   as retryable stream errors and the account is benched by `observe()`.
3. **Virtual compaction via checkpoints.** Harnesses resend full history
   every turn, so the proxy stores a `Checkpoint {covered, prefix_hash,
   summary}` per (session, conversation fingerprint) and rewrites matching
   prefixes on every request. `covered` counts **user/assistant turns
   only**: mid-conversation `role: "system"` messages, which Claude Code
   injects and regenerates, are excluded from hashing and counting.
   `message_digest` ignores thinking blocks, `cache_control`, whitespace and
   tool-result bodies, so harness-side edits (e.g. micro-compaction) don't
   break matches. A mismatch drops the checkpoint. That is safe: the proxy
   then just sends full history.
4. **Summarize on a healthy account with a cheap model.** Compaction never
   uses the account that just failed (it's benched). The order is the target
   account's `compact_model`, then the others (or
   `compaction.compactor_accounts`). Rolling chunked summaries bound
   per-call context. A deterministic extractive fallback guarantees a
   switch never fails because of compaction. The latest user request is
   appended verbatim if it fell inside the compacted part.
5. **Compaction triggers:** account switch with estimated size ≥
   `trigger_tokens`; request > `proactive_ratio` × target context window;
   `ContextOverflow` from upstream (compact, then retry the same account
   once).
6. **Provider-specific opaque data is cached by the proxy, never trusted
   from clients.** Claude thinking signatures are mapped to the account that
   produced them (`thinking_signatures`); foreign or unsigned thinking blocks
   are stripped before an Anthropic call. Gemini thought signatures and
   DeepSeek `reasoning_content` are cached by tool-call id
   (`tool_extras`).
7. **Client credentials are never forwarded upstream** except in explicit
   passthrough mode (non-roulette model names, Anthropic front end only).
   Account keys come from config/env only.

## Conventions

* Edition 2024; prefer let-chains over nested `if let`.
* JSON is handled as `serde_json::Value` with small helper fns; keep
  conversion code table-like and covered by unit tests next to it
  (`#[cfg(test)] mod tests` at the bottom of each file).
* Every behaviour change in routing/compaction/conversion needs a test:
  * a unit test for pure conversion logic, and
  * an integration test in `tests/integration.rs` when it spans HTTP
    (use the mock's key options to script failures).
* Don't log request bodies or credentials. Logs use `tracing` with
  structured fields.
* Never put model identifiers of the assistant that wrote a change into
  code or commits.

## Recipes

**Add a provider that speaks an existing wire:** add a `ProviderKind`
variant and its `Preset` in `src/providers.rs` (base URL, default models,
context window, key env, quirks). Add it to `ProviderKind::all()` and the
README table. If it needs a request/response quirk, add a flag to `Quirks`
and handle it in the adapter, with a mock option + test if it changes the
wire.

**Add a provider with a new wire:** add `Wire::X`, an adapter module under
`src/upstream/` exposing `send(http, store, Call) ->
Result<UpstreamResponse, UpstreamFailure>`. It must produce canonical
events, return pre-stream errors via `ratelimit::classify`, and yield
in-stream errors via `classify_stream_error`. Dispatch it in
`upstream::send`.

**Add a harness:** new module in `src/harness/` implementing `Harness`
(id, aliases, protocol, `launch`, `setup_instructions`, `install`).
Register it in `registry()`. If it speaks a new protocol, add a front end
under `src/frontend/` (request → canonical, canonical events → its stream
format) and route it in `src/server.rs`.

**Debugging a live session:** `RUST_LOG=model_roulette=debug
model-roulette serve`, `model-roulette status --json`, response headers
`x-model-roulette-account/model`, and the state file
`~/.model-roulette/state.json`.

## Verified behaviour (manual end-to-end runs)

Besides `cargo test`, these were exercised against the real harness
binaries with the mock providers:

* Claude Code 2.1.x (`claude -p` and the interactive TUI): `model-roulette`
  shows up in `/model` after `install claude-code`; Bash tool calls run
  through the proxy; a multi-turn `--continue` session rotated
  claude → openai → gemini with two LLM compactions, and tools kept working
  on every account.
* Codex CLI 0.16x (`codex exec`, `exec resume --last`, `--profile roulette`):
  `exec_command` tool calls, reasoning display, rotation across resumed
  turns.

Real provider APIs were not exercised in development (no keys); the
adapters follow each provider's documented wire format.
