//! End-to-end tests: real proxy + mock upstream providers over HTTP.

use std::net::SocketAddr;
use std::sync::Arc;

use model_roulette::config::{AccountConfig, Config, UnknownModelPolicy};
use model_roulette::mock::{self, MockState};
use model_roulette::providers::ProviderKind;
use model_roulette::server;
use serde_json::{Value, json};

struct Env {
    proxy: String,
    mock_addr: SocketAddr,
    mock: Arc<MockState>,
    http: reqwest::Client,
    running: server::Running,
    _dir: TempDir,
}

struct TempDir(std::path::PathBuf);
impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn account(id: &str, kind: ProviderKind, key: &str, base: &str) -> AccountConfig {
    AccountConfig {
        id: id.into(),
        provider: kind,
        model: Some(format!("{id}-model")),
        compact_model: Some(format!("{id}-cheap")),
        fast_model: Some(format!("{id}-fast")),
        api_key: Some(key.into()),
        api_key_env: None,
        base_url: Some(base.into()),
        auth: None,
        context_window: None,
        max_output_tokens: None,
        reasoning_effort: None,
        headers: Default::default(),
        drop_fields: vec![],
        enabled: true,
    }
}

/// `accounts`: (id, kind, key). Anthropic kinds get the root URL, the rest `/v1`.
async fn setup(accounts: &[(&str, ProviderKind, &str)], tweak: impl FnOnce(&mut Config)) -> Env {
    let (mock_addr, mock) = mock::spawn("127.0.0.1:0").await.unwrap();
    let dir = std::env::temp_dir().join(format!("mr-it-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut cfg = Config::default();
    cfg.server.port = 0;
    cfg.server.state_file = Some(dir.join("state.json"));
    cfg.server.passthrough_base_url = format!("http://{mock_addr}");
    cfg.compaction.trigger_tokens = 500;
    cfg.compaction.keep_recent_tokens = 400;
    for (id, kind, key) in accounts {
        let base = match kind.preset().wire {
            model_roulette::providers::Wire::Anthropic => format!("http://{mock_addr}"),
            _ => format!("http://{mock_addr}/v1"),
        };
        cfg.accounts.push(account(id, *kind, key, &base));
    }
    tweak(&mut cfg);
    let running = server::start(cfg).await.unwrap();
    Env {
        proxy: format!("http://{}", running.addr),
        mock_addr,
        mock,
        http: reqwest::Client::new(),
        running,
        _dir: TempDir(dir),
    }
}

impl Env {
    fn requests(&self) -> Vec<Value> {
        self.mock.requests.lock().unwrap().clone()
    }
    fn requests_for(&self, account: &str) -> Vec<Value> {
        self.requests().into_iter().filter(|r| r["account"] == account).collect()
    }
    fn clear(&self) {
        self.mock.requests.lock().unwrap().clear();
    }

    async fn anthropic(&self, session: &str, body: Value) -> (u16, reqwest::header::HeaderMap, Vec<Value>) {
        let resp = self
            .http
            .post(format!("{}/v1/messages", self.proxy))
            .header("x-claude-code-session-id", session)
            .header("anthropic-version", "2023-06-01")
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let headers = resp.headers().clone();
        let text = resp.text().await.unwrap();
        (status, headers, parse_sse(&text))
    }

    async fn responses(&self, session: &str, body: Value) -> (u16, Vec<Value>) {
        let resp = self
            .http
            .post(format!("{}/v1/responses", self.proxy))
            .header("session-id", session)
            .json(&body)
            .send()
            .await
            .unwrap();
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap();
        (status, parse_sse(&text))
    }

    async fn status(&self) -> Value {
        self.http.get(format!("{}/roulette/status", self.proxy)).send().await.unwrap().json().await.unwrap()
    }
}

fn parse_sse(text: &str) -> Vec<Value> {
    let parsed: Vec<Value> = text
        .split("\n\n")
        .filter_map(|frame| frame.lines().find_map(|l| l.strip_prefix("data: ")))
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect();
    if parsed.is_empty() {
        serde_json::from_str(text).map(|v| vec![v]).unwrap_or_default()
    } else {
        parsed
    }
}

fn text_of(events: &[Value]) -> String {
    events
        .iter()
        .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
        .filter_map(|e| e["delta"]["text"].as_str())
        .collect()
}

/// A long agentic conversation (well above the test compaction trigger).
fn long_history(turns: usize, tool_output: usize) -> Vec<Value> {
    let mut msgs = vec![json!({"role":"user","content":"Please refactor the parser module and add tests."})];
    for i in 0..turns {
        msgs.push(json!({"role":"assistant","content":[
            {"type":"thinking","thinking":"plan","signature":format!("sig-claude-{i}")},
            {"type":"text","text":format!("Step {i}: reading file")},
            {"type":"tool_use","id":format!("toolu_{i}"),"name":"Read","input":{"path":format!("src/file{i}.rs")}}]}));
        msgs.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":format!("toolu_{i}"),"content":"fn x() {}\n".repeat(tool_output / 10)}]}));
    }
    msgs.push(json!({"role":"assistant","content":[{"type":"text","text":"Done reading."}]}));
    msgs.push(json!({"role":"user","content":"Now continue with the refactor."}));
    msgs
}

fn req(messages: Vec<Value>) -> Value {
    json!({
        "model": "model-roulette",
        "max_tokens": 1024,
        "stream": true,
        "system": [{"type":"text","text":"You are a coding agent."}],
        "tools": [{"name":"Read","description":"read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
        "thinking": {"type":"adaptive"},
        "messages": messages
    })
}

#[tokio::test]
async fn rotates_compacts_and_sticks() {
    let env = setup(
        &[
            ("claude", ProviderKind::Anthropic, "claude?rl_after=2&retry_after=120"),
            ("openai", ProviderKind::Openai, "openai"),
            ("gemini", ProviderKind::Gemini, "gemini?thought_sig"),
        ],
        |_| {},
    )
    .await;

    // Two small requests go to the first account.
    for _ in 0..2 {
        let (st, h, ev) = env.anthropic("S1", req(vec![json!({"role":"user","content":"hello"})])).await;
        assert_eq!(st, 200);
        assert_eq!(h["x-model-roulette-account"], "claude");
        assert!(text_of(&ev).contains("[claude/claude-model]"), "{}", text_of(&ev));
    }

    // Third request: claude is rate limited -> compaction -> openai.
    env.clear();
    let history = long_history(12, 1500);
    let (st, h, ev) = env.anthropic("S1", req(history.clone())).await;
    assert_eq!(st, 200, "{ev:?}");
    assert_eq!(h["x-model-roulette-account"], "openai");
    assert!(text_of(&ev).contains("[openai/openai-model]"));

    let reqs = env.requests();
    assert_eq!(reqs[0]["account"], "claude"); // the 429
    // Summarizer call used openai's cheap model.
    let summarize = reqs.iter().find(|r| r["body"]["model"] == "openai-cheap").expect("summarizer call");
    let sum_prompt = summarize["body"]["messages"][1]["content"].as_str().unwrap();
    assert!(sum_prompt.contains("<transcript>") && sum_prompt.contains("Please refactor the parser module"));
    // The real request carries the summary instead of the full history.
    let main = reqs.iter().rev().find(|r| r["body"]["model"] == "openai-model").unwrap();
    let msgs = main["body"]["messages"].as_array().unwrap();
    assert_eq!(msgs[0]["role"], "system");
    let first_user = msgs[1]["content"].as_str().unwrap();
    assert!(first_user.contains("<context-summary>") && first_user.contains("MOCK-SUMMARY(openai)"), "{first_user}");
    assert!(msgs.len() < history.len(), "{} >= {}", msgs.len(), history.len());
    assert_eq!(msgs.last().unwrap()["content"], "Now continue with the refactor.");

    // Claude is cooling down for ~120s (retry-after).
    let status = env.status().await;
    let claude = &status["accounts"][0];
    assert_eq!(claude["available"], false);
    assert!(claude["cooldown_remaining_secs"].as_i64().unwrap() > 100);
    assert_eq!(status["recent_sessions"][0]["account"], "openai");
    assert_eq!(status["recent_sessions"][0]["switches"], 1);

    // Next turn of the same session: sticks to openai, reuses the checkpoint
    // (no new summarizer call) and still sends a compact history.
    env.clear();
    let mut grown = history.clone();
    grown.push(json!({"role":"assistant","content":[{"type":"text","text":"Refactored."}]}));
    grown.push(json!({"role":"user","content":"Great, now run the tests."}));
    let (st, h, _) = env.anthropic("S1", req(grown.clone())).await;
    assert_eq!(st, 200);
    assert_eq!(h["x-model-roulette-account"], "openai");
    let reqs = env.requests();
    assert_eq!(reqs.len(), 1, "no extra summarizer call expected");
    let msgs = reqs[0]["body"]["messages"].as_array().unwrap();
    assert!(msgs[1]["content"].as_str().unwrap().contains("MOCK-SUMMARY(openai)"));
    assert!(msgs.len() < grown.len());

    // A brand-new session skips the cooling account.
    let (_, h, _) = env.anthropic("S2", req(vec![json!({"role":"user","content":"new session"})])).await;
    assert_eq!(h["x-model-roulette-account"], "openai");

    // State survives a restart of the store (persisted to disk).
    env.running.roulette.store.flush().unwrap();
    let path = env.running.roulette.cfg.state_file();
    let persisted: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    assert!(persisted["accounts"]["claude"]["cooldown_until"].as_i64().unwrap() > 0);
    assert!(persisted["sessions"]["S1"]["conversations"].as_object().unwrap().values().any(|c| c["checkpoint"].is_object()));
}

#[tokio::test]
async fn switching_between_anthropic_accounts_drops_foreign_thinking() {
    let env = setup(
        &[
            ("c1", ProviderKind::Anthropic, "c1?rl_after=1"),
            ("c2", ProviderKind::Anthropic, "c2"),
        ],
        |c| c.compaction.trigger_tokens = 1_000_000, // no compaction: test signature handling
    )
    .await;
    let (_, h, ev) = env.anthropic("T", req(vec![json!({"role":"user","content":"hi"})])).await;
    assert_eq!(h["x-model-roulette-account"], "c1");
    // Build the next turn from c1's reply (thinking block signed by c1).
    let sig = ev.iter().find_map(|e| e["delta"]["signature"].as_str()).unwrap().to_string();
    let history = vec![
        json!({"role":"user","content":"hi"}),
        json!({"role":"assistant","content":[{"type":"thinking","thinking":"mock thinking","signature":sig},{"type":"text","text":"hello"}]}),
        json!({"role":"user","content":"next"}),
    ];
    env.clear();
    let (st, h, _) = env.anthropic("T", req(history)).await;
    assert_eq!(st, 200);
    assert_eq!(h["x-model-roulette-account"], "c2");
    let sent = env.requests_for("c2");
    let assistant = &sent[0]["body"]["messages"][1]["content"];
    assert!(assistant.as_array().unwrap().iter().all(|b| b["type"] != "thinking"), "{assistant}");
    // Thinking for the new turn is still requested.
    assert_eq!(sent[0]["body"]["thinking"]["type"], "adaptive");
}

#[tokio::test]
async fn quota_exhaustion_and_all_exhausted() {
    let env = setup(
        &[("broke", ProviderKind::Openai, "broke?quota"), ("limited", ProviderKind::Anthropic, "limited?rl_after=0&retry_after=45")],
        |_| {},
    )
    .await;
    let (st, h, ev) = env.anthropic("Q", req(vec![json!({"role":"user","content":"hi"})])).await;
    assert_eq!(st, 429, "{ev:?}");
    let retry: u64 = h["retry-after"].to_str().unwrap().parse().unwrap();
    assert!(retry >= 40 && retry <= 45, "retry-after {retry}");
    assert_eq!(ev[0]["error"]["type"], "rate_limit_error");
    let status = env.status().await;
    assert_eq!(status["accounts"][0]["last_failure"], "quota_exhausted");
    // Quota exhaustion uses the long quota cooldown.
    assert!(status["accounts"][0]["cooldown_remaining_secs"].as_i64().unwrap() > 3600);
    assert_eq!(status["accounts"][1]["last_failure"], "rate_limited");

    // Reset puts them back in rotation.
    env.http.post(format!("{}/roulette/reset", env.proxy)).send().await.unwrap();
    let status = env.status().await;
    assert_eq!(status["accounts"][0]["cooldown_remaining_secs"], 0);
}

#[tokio::test]
async fn context_overflow_triggers_compaction_on_same_account() {
    let env = setup(&[("small", ProviderKind::Anthropic, "small?ctx_limit=60000")], |c| {
        c.compaction.keep_recent_tokens = 2000;
    })
    .await;
    let history = long_history(20, 4000);
    let (st, h, ev) = env.anthropic("O", req(history)).await;
    assert_eq!(st, 200, "{ev:?}");
    assert_eq!(h["x-model-roulette-account"], "small");
    let reqs = env.requests();
    assert!(reqs.len() >= 3, "overflow, summarize, retry: {}", reqs.len());
    let last = reqs.last().unwrap();
    let first = last["body"]["messages"][0]["content"][0]["text"].as_str().unwrap();
    assert!(first.contains("MOCK-SUMMARY(small)"));
}

#[tokio::test]
async fn proactive_compaction_for_small_context_window() {
    let env = setup(&[("tiny", ProviderKind::Openai, "tiny")], |c| {
        c.accounts[0].context_window = Some(8_000);
        c.compaction.keep_recent_tokens = 1000;
    })
    .await;
    let (st, _, _) = env.anthropic("P", req(long_history(15, 3000))).await;
    assert_eq!(st, 200);
    let reqs = env.requests();
    assert_eq!(reqs.len(), 2, "summarize + request");
    assert_eq!(reqs[0]["body"]["model"], "tiny-cheap");
}

#[tokio::test]
async fn summarizer_failure_falls_back_to_extractive_summary() {
    let env = setup(
        &[("x", ProviderKind::Openai, "x"), ("y", ProviderKind::Openai, "y"), ("z", ProviderKind::Openai, "z?rl_after=0")],
        |c| c.compaction.compactor_accounts = vec!["z".into()], // summarizer always fails
    )
    .await;
    let (_, h, _) = env.anthropic("G", req(vec![json!({"role":"user","content":"Please refactor the parser module and add tests."})])).await;
    assert_eq!(h["x-model-roulette-account"], "x");
    env.running.roulette.store.bench("x", model_roulette::ratelimit::FailureKind::RateLimited, model_roulette::state::now_ts() + 100, "test");
    env.clear();
    let (st, h, _) = env.anthropic("G", req(long_history(10, 2000))).await;
    assert_eq!(st, 200);
    assert_eq!(h["x-model-roulette-account"], "y");
    let sent = env.requests_for("y");
    let first = sent[0]["body"]["messages"][1]["content"].as_str().unwrap();
    assert!(first.contains("Extractive summary") && first.contains("Please refactor the parser module"), "{first}");
    assert!(first.contains("Read×"));
}

#[tokio::test]
async fn codex_responses_roundtrip_with_tools_via_gemini() {
    let env = setup(&[("gemini", ProviderKind::Gemini, "gemini?thought_sig")], |_| {}).await;
    let tools = json!([
        {"type":"function","name":"exec_command","description":"run","strict":false,"parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"],"additionalProperties":false}},
        {"type":"namespace","name":"multi_agent_v1","tools":[{"type":"function","name":"close_agent","parameters":{"type":"object","properties":{}}}]},
        {"type":"custom","name":"apply_patch","description":"patch files","format":{"type":"grammar","syntax":"lark","definition":"start: /.+/"}},
        {"type":"web_search","external_web_access":false}
    ]);
    let input1 = json!([
        {"type":"message","role":"developer","content":[{"type":"input_text","text":"You are Codex."}]},
        {"type":"message","role":"user","content":[{"type":"input_text","text":"CALL_TOOL exec_command {\"cmd\":\"ls -la\"}"}]}
    ]);
    let body = json!({"model":"model-roulette","stream":true,"input":input1,"tools":tools,"tool_choice":"auto","parallel_tool_calls":true,
        "reasoning":{"summary":"auto"},"store":false,"include":["reasoning.encrypted_content"],"prompt_cache_key":"codex-sess"});
    let (st, ev) = env.responses("codex-sess", body).await;
    assert_eq!(st, 200, "{ev:?}");
    assert_eq!(ev[0]["type"], "response.created");
    let call = ev
        .iter()
        .find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "function_call")
        .expect("function call")["item"]
        .clone();
    assert_eq!(call["name"], "exec_command");
    assert_eq!(serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap()["cmd"], "ls -la");
    let done = ev.last().unwrap();
    assert_eq!(done["type"], "response.completed");
    assert!(done["response"]["usage"]["total_tokens"].as_u64().unwrap() > 0);

    // Gemini-specific schema keywords were stripped.
    let sent = env.requests();
    let fparams = &sent[0]["body"]["tools"][0]["function"]["parameters"];
    assert!(fparams.get("additionalProperties").is_none());
    let names: Vec<&str> = sent[0]["body"]["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"multi_agent_v1__close_agent") && names.contains(&"apply_patch"));

    // Turn 2: Codex sends back the call and its output. The mock rejects the
    // history unless the Gemini thought signature is round-tripped.
    let mut input2 = input1.as_array().unwrap().clone();
    input2.push(json!({"type":"reasoning","summary":[],"encrypted_content":null}));
    input2.push(json!({"type":"function_call","name":"exec_command","arguments":call["arguments"],"call_id":call["call_id"]}));
    input2.push(json!({"type":"function_call_output","call_id":call["call_id"],"output":"total 0\nfile.txt"}));
    let body2 = json!({"model":"model-roulette","stream":true,"input":input2,"tools":tools,"prompt_cache_key":"codex-sess"});
    let (st, ev) = env.responses("codex-sess", body2).await;
    assert_eq!(st, 200, "{ev:?}");
    let msg = ev
        .iter()
        .find(|e| e["type"] == "response.output_item.done" && e["item"]["type"] == "message")
        .expect("message")["item"]
        .clone();
    assert!(msg["content"][0]["text"].as_str().unwrap().contains("tool result received: total 0"));
    let sent = env.requests();
    let tc = &sent[1]["body"]["messages"][2]["tool_calls"][0];
    assert_eq!(tc["extra_content"]["google"]["thought_signature"], "gsig-1");

    // Namespaced and custom tools map back to Codex's item shapes.
    let input3 = json!([{"type":"message","role":"user","content":[{"type":"input_text","text":"CALL_TOOL multi_agent_v1__close_agent {\"id\":\"a1\"}"}]}]);
    let (_, ev) = env.responses("codex-2", json!({"model":"model-roulette","stream":true,"input":input3,"tools":tools})).await;
    let item = ev.iter().find(|e| e["type"] == "response.output_item.done").unwrap()["item"].clone();
    assert_eq!(item["name"], "close_agent");
    assert_eq!(item["namespace"], "multi_agent_v1");
    let input4 = json!([{"type":"message","role":"user","content":[{"type":"input_text","text":"CALL_TOOL apply_patch {\"input\":\"*** Begin Patch\\n*** End Patch\"}"}]}]);
    let (_, ev) = env.responses("codex-3", json!({"model":"model-roulette","stream":true,"input":input4,"tools":tools})).await;
    let item = ev.iter().find(|e| e["type"] == "response.output_item.done").unwrap()["item"].clone();
    assert_eq!(item["type"], "custom_tool_call");
    assert_eq!(item["input"], "*** Begin Patch\n*** End Patch");
}

#[tokio::test]
async fn deepseek_reasoning_content_is_echoed_in_tool_loops() {
    let env = setup(&[("ds", ProviderKind::Deepseek, "ds?reasoning&echo_reasoning")], |_| {}).await;
    let tools = json!([{"name":"Bash","description":"run","input_schema":{"type":"object","properties":{"command":{"type":"string"}}}}]);
    let mut body = req(vec![json!({"role":"user","content":"CALL_TOOL Bash {\"command\":\"pwd\"}"})]);
    body["tools"] = tools.clone();
    let (st, _, ev) = env.anthropic("D", body).await;
    assert_eq!(st, 200);
    // The reasoning is surfaced as a thinking block, the call as tool_use.
    assert!(ev.iter().any(|e| e["content_block"]["type"] == "thinking"));
    let tool = ev.iter().find(|e| e["content_block"]["type"] == "tool_use").unwrap()["content_block"].clone();
    let id = tool["id"].as_str().unwrap();
    // Next turn as Claude Code would send it (thinking block w/o signature).
    let mut body = req(vec![
        json!({"role":"user","content":"CALL_TOOL Bash {\"command\":\"pwd\"}"}),
        json!({"role":"assistant","content":[{"type":"thinking","thinking":"mock reasoning","signature":""},{"type":"tool_use","id":id,"name":"Bash","input":{"command":"pwd"}}]}),
        json!({"role":"user","content":[{"type":"tool_result","tool_use_id":id,"content":"/home/user"}]}),
    ]);
    body["tools"] = tools;
    let (st, _, ev) = env.anthropic("D", body).await;
    assert_eq!(st, 200, "{ev:?}");
    assert!(text_of(&ev).contains("tool result received: /home/user"));
    let sent = env.requests();
    assert_eq!(sent[1]["body"]["messages"][2]["reasoning_content"], "mock reasoning");
    // DeepSeek's preset caps max_tokens.
    assert_eq!(sent[1]["body"]["max_tokens"], 1024);
}

#[tokio::test]
async fn midstream_error_benches_account_and_next_request_moves_on() {
    let env = setup(
        &[("flaky", ProviderKind::Anthropic, "flaky?midstream_error"), ("steady", ProviderKind::Anthropic, "steady")],
        |_| {},
    )
    .await;
    let (st, _, ev) = env.anthropic("M", req(vec![json!({"role":"user","content":"hi"})])).await;
    assert_eq!(st, 200);
    assert_eq!(ev.last().unwrap()["type"], "error");
    assert_eq!(ev.last().unwrap()["error"]["type"], "overloaded_error");
    let (_, h, _) = env.anthropic("M", req(vec![json!({"role":"user","content":"hi"})])).await;
    assert_eq!(h["x-model-roulette-account"], "steady");
}

#[tokio::test]
async fn fast_lane_passthrough_models_and_count_tokens() {
    let env = setup(&[("a", ProviderKind::Anthropic, "a"), ("b", ProviderKind::Openai, "b")], |_| {}).await;
    // Fast lane uses fast_model.
    let mut body = req(vec![json!({"role":"user","content":"title please"})]);
    body["model"] = json!("model-roulette-fast");
    body["stream"] = json!(false);
    let resp: Value = env.http.post(format!("{}/v1/messages", env.proxy)).json(&body).send().await.unwrap().json().await.unwrap();
    assert_eq!(resp["type"], "message");
    assert!(resp["content"].as_array().unwrap().iter().any(|b| b["text"].as_str().unwrap_or("").contains("[a/a-fast]")));

    // Unknown models pass through with the client's own key.
    body["model"] = json!("claude-opus-5-5");
    body["stream"] = json!(true);
    let resp = env
        .http
        .post(format!("{}/v1/messages?beta=true", env.proxy))
        .header("x-api-key", "client-own-key")
        .json(&body)
        .send()
        .await
        .unwrap();
    let ev = parse_sse(&resp.text().await.unwrap());
    assert!(text_of(&ev).contains("[client-own-key/claude-opus-5-5]"));

    // count_tokens is answered locally for roulette models.
    body["model"] = json!("model-roulette");
    let ct: Value = env.http.post(format!("{}/v1/messages/count_tokens", env.proxy)).json(&body).send().await.unwrap().json().await.unwrap();
    assert!(ct["input_tokens"].as_u64().unwrap() > 10);

    // /v1/models lists the roulette models.
    let models: Value = env.http.get(format!("{}/v1/models", env.proxy)).send().await.unwrap().json().await.unwrap();
    assert_eq!(models["data"][0]["id"], "model-roulette");
}

#[tokio::test]
async fn chat_completions_frontend() {
    let env = setup(&[("a", ProviderKind::Anthropic, "a")], |c| c.server.unknown_models = UnknownModelPolicy::Reject).await;
    let body = json!({"model":"model-roulette","stream":true,"stream_options":{"include_usage":true},
        "messages":[{"role":"system","content":"sys"},{"role":"user","content":"CALL_TOOL get_weather {\"city\":\"Paris\"}"}],
        "tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]});
    let resp = env.http.post(format!("{}/v1/chat/completions", env.proxy)).json(&body).send().await.unwrap();
    let text = resp.text().await.unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"));
    let chunks = parse_sse(&text);
    let tc = chunks.iter().find(|c| c["choices"][0]["delta"]["tool_calls"].is_array()).unwrap();
    assert_eq!(tc["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "get_weather");
    assert!(chunks.iter().any(|c| c["choices"][0]["finish_reason"] == "tool_calls"));
    assert!(chunks.iter().any(|c| c["usage"]["total_tokens"].as_u64().is_some()));

    // Unknown model rejected by policy.
    let r = env.http.post(format!("{}/v1/chat/completions", env.proxy)).json(&json!({"model":"gpt-x","messages":[]})).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 404);
    let _ = env.mock_addr;
}

#[tokio::test]
async fn proxy_api_key_is_enforced() {
    let env = setup(&[("a", ProviderKind::Anthropic, "a")], |c| c.server.api_key = Some("secret".into())).await;
    let r = env.http.post(format!("{}/v1/messages", env.proxy)).json(&req(vec![json!({"role":"user","content":"x"})])).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 401);
    let r = env
        .http
        .post(format!("{}/v1/messages", env.proxy))
        .header("authorization", "Bearer secret")
        .json(&req(vec![json!({"role":"user","content":"x"})]))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status().as_u16(), 200);
}
