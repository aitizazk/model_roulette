//! Virtual compaction.
//!
//! Harnesses keep sending their full history on every request; they don't
//! know the proxy switched models. So compaction happens *inside the proxy*:
//! when a session moves to a new account (or a request would overflow the
//! target's context window), the older part of the conversation is summarized
//! once and stored as a [`Checkpoint`]. Every later request whose history
//! starts with the same messages gets that prefix replaced by the summary
//! before it is sent upstream. The harness never notices, and the new model
//! only reads the summary plus the most recent turns.
//!
//! The summary is produced by a cheap model (`compact_model`) on an account
//! that is *not* rate limited — normally the one we are switching to.
//! Huge histories are summarized in a rolling fashion so no single call has
//! to read everything, and tool outputs are truncated before summarizing.
//! If no model is available a deterministic extractive summary is used, so a
//! switch never fails because of compaction.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::canonical::{
    blocks, estimate_tokens, prefix_hash, role, tool_result_text, turn_count, turn_cut_index,
};
use crate::config::CompactionConfig;
use crate::state::{Checkpoint, now_ts, truncate};

pub const SUMMARY_OPEN: &str = "<context-summary>";
pub const SUMMARY_CLOSE: &str = "</context-summary>";

pub const SYSTEM_PROMPT: &str = "You are the context compactor for an AI coding agent. \
The agent's conversation is being handed over to a different model, which will only see \
your summary plus the most recent messages. Write a dense, factual summary that lets the \
next model continue the work seamlessly. Never invent facts. Preserve exact file paths, \
function names, commands, error messages, identifiers and the user's exact instructions.";

const STRUCTURE: &str = "Structure the summary with these sections:\n\
1. Primary request and intent - everything the user asked for, in detail.\n\
2. Key technical context - languages, frameworks, constraints, decisions made and why.\n\
3. Files and code - every file read/created/modified, what changed, and short critical snippets.\n\
4. Errors and fixes - problems hit and how they were resolved (or not).\n\
5. User messages - list every user message (paraphrase long ones, quote instructions verbatim).\n\
6. Progress - what is done, what is in progress.\n\
7. Next step - the exact next action the agent was about to take, with any pending tool call context.\n\
Output only the summary.";

/// The user message that replaces a compacted prefix.
pub fn summary_message(summary: &str) -> Value {
    json!({
        "role": "user",
        "content": [{
            "type": "text",
            "text": format!(
                "{SUMMARY_OPEN}\nThe earlier part of this conversation was compacted (the session was moved to a different model). \
    This summary replaces those earlier messages:\n\n{summary}\n{SUMMARY_CLOSE}\n\nContinue the work from where it left off; the most recent messages follow."
            )
        }]
    })
}

fn existing_summary(msg: &Value) -> Option<String> {
    if role(msg) != "user" {
        return None;
    }
    let text = blocks(msg).first()?.get("text")?.as_str()?.to_string();
    let start = text.find(SUMMARY_OPEN)?;
    let end = text.find(SUMMARY_CLOSE)?;
    let inner = &text[start + SUMMARY_OPEN.len()..end];
    // Drop the fixed preamble line.
    let inner = inner.split_once(":\n\n").map(|(_, s)| s).unwrap_or(inner);
    Some(inner.trim().to_string())
}

/// Replace the checkpointed prefix of `original` with its summary, if the
/// checkpoint matches this conversation.
///
/// `ck.covered` counts user/assistant turns, so mid-conversation system
/// messages (which harnesses add and regenerate) never break the match; any
/// inside the covered prefix are dropped along with it.
pub fn apply_checkpoint(original: &[Value], ck: &Checkpoint) -> Option<Vec<Value>> {
    let cut = checkpoint_cut(original, ck)?;
    let mut out = Vec::with_capacity(original.len() - cut + 1);
    out.push(summary_message(&ck.summary));
    out.extend_from_slice(&original[cut..]);
    Some(out)
}

/// Raw index in `original` where the checkpoint's covered prefix ends, if the
/// checkpoint matches and something remains after it.
pub fn checkpoint_cut(original: &[Value], ck: &Checkpoint) -> Option<usize> {
    if turn_count(original) <= ck.covered {
        return None;
    }
    if prefix_hash(original, ck.covered) != ck.prefix_hash {
        return None;
    }
    turn_cut_index(original, ck.covered)
}

/// Where to cut. `working` is the list that would be sent (possibly already
/// starting with a summary message). Returns `b` such that `working[..b]` gets
/// summarized and `working[b..]` (which starts with an assistant message) is
/// kept verbatim.
pub fn choose_boundary(working: &[Value], cfg: &CompactionConfig) -> Option<usize> {
    let candidates: Vec<usize> = (1..working.len())
        .filter(|&i| role(&working[i]) == "assistant")
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let mut suffix = vec![0u64; working.len() + 1];
    for i in (0..working.len()).rev() {
        suffix[i] = suffix[i + 1] + estimate_tokens(&working[i]);
    }
    let b = candidates
        .iter()
        .copied()
        .find(|&i| suffix[i] <= cfg.keep_recent_tokens)
        .unwrap_or(*candidates.last().unwrap());
    // Only worth it if the head is meaningfully large.
    let head = suffix[0] - suffix[b];
    let already_summary_only = b == 1 && existing_summary(&working[0]).is_some();
    if head < 1_500 || already_summary_only {
        return None;
    }
    Some(b)
}

/// Render messages as a plain-text transcript for the summarizer.
pub fn render_transcript(msgs: &[Value], cfg: &CompactionConfig) -> (Option<String>, Vec<String>) {
    let mut previous = None;
    let mut names: HashMap<String, String> = HashMap::new();
    let mut entries = Vec::new();
    for (i, m) in msgs.iter().enumerate() {
        if i == 0
            && let Some(s) = existing_summary(m)
        {
            previous = Some(s);
            continue;
        }
        let mut out = String::new();
        let r = role(m);
        for b in blocks(m) {
            match b.get("type").and_then(Value::as_str).unwrap_or("") {
                "text" => {
                    let t = b.get("text").and_then(Value::as_str).unwrap_or("");
                    out.push_str(&format!("[{}]\n{}\n", r.to_uppercase(), t.trim()));
                }
                "tool_use" => {
                    let name = b.get("name").and_then(Value::as_str).unwrap_or("?");
                    let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                    names.insert(id.to_string(), name.to_string());
                    let input = b.get("input").map(|v| v.to_string()).unwrap_or_default();
                    out.push_str(&format!(
                        "[ASSISTANT → tool call {name}] {}\n",
                        head_tail(&input, cfg.tool_input_max_chars)
                    ));
                }
                "tool_result" => {
                    let id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                    let name = names.get(id).cloned().unwrap_or_else(|| "tool".into());
                    let err = if b.get("is_error").and_then(Value::as_bool) == Some(true) {
                        " (error)"
                    } else {
                        ""
                    };
                    out.push_str(&format!(
                        "[TOOL RESULT {name}{err}]\n{}\n",
                        head_tail(&tool_result_text(&b), cfg.tool_result_max_chars)
                    ));
                }
                "image" => out.push_str(&format!("[{}] [image]\n", r.to_uppercase())),
                _ => {}
            }
        }
        if !out.is_empty() {
            entries.push(out);
        }
    }
    (previous, entries)
}

/// Keep the beginning and end of long text.
pub fn head_tail(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let half = max / 2;
    let head = &s[..s.floor_char_boundary(half)];
    let tail_start = s.ceil_char_boundary(s.len() - half);
    format!(
        "{head}\n…[{} chars omitted]…\n{}",
        s.len() - max,
        &s[tail_start..]
    )
}

/// Split transcript entries into chunks of at most `max_tokens`.
pub fn chunk_entries(entries: &[String], max_tokens: u64) -> Vec<String> {
    let max_chars = (max_tokens.max(1000) * 4) as usize;
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for e in entries {
        let e = if e.len() > max_chars {
            head_tail(e, max_chars)
        } else {
            e.clone()
        };
        if !cur.is_empty() && cur.len() + e.len() > max_chars {
            chunks.push(std::mem::take(&mut cur));
        }
        cur.push_str(&e);
        cur.push('\n');
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// Build the summarizer prompt for one chunk.
pub fn chunk_prompt(previous: Option<&str>, chunk: &str, part: usize, total: usize) -> String {
    let mut p = String::new();
    if let Some(prev) = previous {
        p.push_str("Summary of the conversation so far:\n<previous-summary>\n");
        p.push_str(prev);
        p.push_str("\n</previous-summary>\n\n");
        p.push_str(
            "Update that summary to also cover the following later part of the conversation",
        );
    } else {
        p.push_str("Summarize the following conversation");
    }
    if total > 1 {
        p.push_str(&format!(" (part {part} of {total})"));
    }
    p.push_str(".\n\n<transcript>\n");
    p.push_str(chunk);
    p.push_str("</transcript>\n\n");
    p.push_str(STRUCTURE);
    p
}

/// The most recent thing the user actually typed in `msgs` (ignoring tool
/// results and harness-injected `<system-reminder>` blocks).
pub fn latest_user_request(msgs: &[Value]) -> Option<String> {
    msgs.iter()
        .rev()
        .filter(|m| role(m) == "user")
        .find_map(|m| {
            let texts: Vec<String> = blocks(m)
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .map(str::trim)
                .filter(|t| {
                    !t.is_empty()
                        && !t.starts_with("<system-reminder>")
                        && !t.contains(SUMMARY_OPEN)
                })
                .map(str::to_string)
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        })
}

/// Append the user's latest request verbatim when compaction swallowed it,
/// so the next model sees the exact instruction it is working on.
pub fn with_latest_request(summary: String, head: &[Value], tail: &[Value]) -> String {
    if latest_user_request(tail).is_some() {
        return summary;
    }
    match latest_user_request(head) {
        Some(req) => format!(
            "{summary}\n\nThe user's most recent request (verbatim):\n{}",
            head_tail(&req, 6_000)
        ),
        None => summary,
    }
}

/// Deterministic summary used when no summarizer model is reachable.
pub fn fallback_summary(previous: Option<&str>, msgs: &[Value]) -> String {
    let mut out =
        String::from("(Extractive summary generated without a model; details may be missing.)\n\n");
    if let Some(prev) = previous {
        out.push_str("Earlier summary:\n");
        out.push_str(&head_tail(prev, 8_000));
        out.push_str("\n\n");
    }
    let mut users = Vec::new();
    let mut tool_counts: Vec<(String, usize)> = Vec::new();
    let mut recent_calls = Vec::new();
    let mut last_assistant = Vec::new();
    for m in msgs {
        for b in blocks(m) {
            match (role(m), b.get("type").and_then(Value::as_str).unwrap_or("")) {
                ("user", "text") => {
                    let t = b
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !t.is_empty() && !t.contains(SUMMARY_OPEN) {
                        users.push(truncate(&t, 600));
                    }
                }
                ("assistant", "text") => {
                    let t = b
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    if !t.is_empty() {
                        last_assistant.push(truncate(&t, 800));
                    }
                }
                ("assistant", "tool_use") => {
                    let name = b
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("?")
                        .to_string();
                    match tool_counts.iter_mut().find(|(n, _)| *n == name) {
                        Some((_, c)) => *c += 1,
                        None => tool_counts.push((name.clone(), 1)),
                    }
                    let input = b.get("input").map(|v| v.to_string()).unwrap_or_default();
                    recent_calls.push(format!("{name} {}", truncate(&input, 200)));
                }
                _ => {}
            }
        }
    }
    if !users.is_empty() {
        out.push_str("User messages:\n");
        let keep: Vec<&String> = if users.len() > 30 {
            users
                .iter()
                .take(10)
                .chain(users.iter().skip(users.len() - 20))
                .collect()
        } else {
            users.iter().collect()
        };
        for u in keep {
            out.push_str(&format!("- {u}\n"));
        }
        out.push('\n');
    }
    if !tool_counts.is_empty() {
        out.push_str("Tool usage: ");
        out.push_str(
            &tool_counts
                .iter()
                .map(|(n, c)| format!("{n}×{c}"))
                .collect::<Vec<_>>()
                .join(", "),
        );
        out.push_str("\nMost recent tool calls:\n");
        for c in recent_calls.iter().rev().take(20).rev() {
            out.push_str(&format!("- {c}\n"));
        }
        out.push('\n');
    }
    if !last_assistant.is_empty() {
        out.push_str("Latest assistant notes:\n");
        for a in last_assistant.iter().rev().take(3).rev() {
            out.push_str(&format!("- {a}\n"));
        }
    }
    out
}

/// Turn a summary over `working[..b]` into a checkpoint over the *original*
/// message list. `previous_cut` is the raw index in `original` where the
/// older checkpoint ended when `working` already began with its summary.
pub fn make_checkpoint(
    original: &[Value],
    b_in_working: usize,
    previous_cut: Option<usize>,
    summary: String,
    by_account: &str,
    method: String,
) -> Checkpoint {
    let raw_cut = match previous_cut {
        Some(c) => c + b_in_working - 1,
        None => b_in_working,
    };
    let covered = turn_count(&original[..raw_cut.min(original.len())]);
    Checkpoint {
        covered,
        prefix_hash: prefix_hash(original, covered),
        summary,
        created_at: now_ts(),
        by_account: by_account.to_string(),
        method,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(n: usize, size: usize) -> Vec<Value> {
        let mut v = vec![json!({"role":"user","content":"start the task"})];
        for i in 0..n {
            v.push(json!({"role":"assistant","content":[{"type":"text","text":format!("step {i}")},{"type":"tool_use","id":format!("t{i}"),"name":"Bash","input":{"cmd":"ls"}}]}));
            v.push(json!({"role":"user","content":[{"type":"tool_result","tool_use_id":format!("t{i}"),"content":"x".repeat(size)}]}));
        }
        v
    }

    #[test]
    fn boundary_and_checkpoint_roundtrip() {
        let cfg = CompactionConfig {
            keep_recent_tokens: 3000,
            ..Default::default()
        };
        let original = conv(20, 4000);
        let b = choose_boundary(&original, &cfg).unwrap();
        assert_eq!(role(&original[b]), "assistant");
        assert!(b > 10);
        let ck = make_checkpoint(&original, b, None, "S1".into(), "a", "test".into());
        let rewritten = apply_checkpoint(&original, &ck).unwrap();
        assert_eq!(rewritten.len(), original.len() - b + 1);
        assert!(existing_summary(&rewritten[0]).unwrap().contains("S1"));

        // The conversation grows; the checkpoint still applies.
        let mut grown = original.clone();
        grown.extend(conv(3, 10).into_iter().skip(1));
        assert!(apply_checkpoint(&grown, &ck).is_some());

        // A second compaction on top of the first maps back to original indices.
        let working = apply_checkpoint(&grown, &ck).unwrap();
        let cfg2 = CompactionConfig {
            keep_recent_tokens: 100,
            ..Default::default()
        };
        if let Some(b2) = choose_boundary(&working, &cfg2) {
            let ck2 = make_checkpoint(
                &grown,
                b2,
                checkpoint_cut(&grown, &ck),
                "S2".into(),
                "b",
                "test".into(),
            );
            let rw2 = apply_checkpoint(&grown, &ck2).unwrap();
            assert_eq!(rw2[1], working[b2]);
        }

        // A different conversation doesn't match.
        let mut other = conv(25, 10);
        other[0] = json!({"role":"user","content":"a different task"});
        assert!(apply_checkpoint(&other, &ck).is_none());
    }

    #[test]
    fn system_messages_do_not_break_checkpoints() {
        let cfg = CompactionConfig {
            keep_recent_tokens: 3000,
            ..Default::default()
        };
        let mut original = conv(20, 4000);
        original.insert(3, json!({"role":"system","content":"reminder v1"}));
        let b = choose_boundary(&original, &cfg).unwrap();
        let ck = make_checkpoint(&original, b, None, "S".into(), "a", "t".into());
        // Next turn: the old system message is gone, a new one is appended.
        let mut next: Vec<Value> = original
            .iter()
            .filter(|m| role(m) != "system")
            .cloned()
            .collect();
        next.push(json!({"role":"system","content":"reminder v2"}));
        let rw = apply_checkpoint(&next, &ck).expect("still matches");
        assert_eq!(rw[1], original[b]);
        assert_eq!(rw.last().unwrap()["content"], "reminder v2");
    }

    #[test]
    fn latest_request_is_kept_verbatim() {
        let head = vec![
            json!({"role":"user","content":[{"type":"text","text":"<system-reminder>x</system-reminder>"},{"type":"text","text":"fix the bug in parser.rs"}]}),
            json!({"role":"assistant","content":"ok"}),
            json!({"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"data"}]}),
        ];
        let tail = vec![json!({"role":"assistant","content":"working"})];
        let s = with_latest_request("SUM".into(), &head, &tail);
        assert!(s.ends_with("fix the bug in parser.rs"), "{s}");
        let tail2 = vec![
            json!({"role":"assistant","content":"a"}),
            json!({"role":"user","content":"new ask"}),
        ];
        assert_eq!(with_latest_request("SUM".into(), &head, &tail2), "SUM");
    }

    #[test]
    fn small_conversations_are_not_compacted() {
        let cfg = CompactionConfig::default();
        assert!(choose_boundary(&conv(2, 10), &cfg).is_none());
    }

    #[test]
    fn transcript_and_chunks() {
        let cfg = CompactionConfig::default();
        let msgs = conv(10, 10_000);
        let (prev, entries) = render_transcript(&msgs, &cfg);
        assert!(prev.is_none());
        assert!(entries.iter().any(|e| e.contains("[TOOL RESULT Bash]")));
        assert!(entries.iter().all(|e| e.len() < 3_000));
        let chunks = chunk_entries(&entries, 1_000);
        assert!(chunks.len() > 1);
        let p = chunk_prompt(Some("old"), &chunks[1], 2, chunks.len());
        assert!(p.contains("<previous-summary>") && p.contains("part 2"));
        let fb = fallback_summary(Some("older summary"), &msgs);
        assert!(fb.contains("start the task") && fb.contains("Bash×10"));
    }

    #[test]
    fn head_tail_is_char_safe() {
        let s = "é".repeat(1000);
        let t = head_tail(&s, 101);
        assert!(t.contains("omitted"));
    }
}
