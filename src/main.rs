use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use model_roulette::config::{Config, EXAMPLE_CONFIG};
use model_roulette::harness;
use model_roulette::providers::ProviderKind;
use model_roulette::roulette::Roulette;
use model_roulette::state::StateStore;
use serde_json::Value;

#[derive(Parser)]
#[command(
    name = "model-roulette",
    version,
    about = "One model name that rotates across your LLM accounts when they hit rate limits."
)]
struct Cli {
    /// Config file (default: ~/.model-roulette/config.toml or $MODEL_ROULETTE_CONFIG)
    #[arg(long, short, global = true, env = "MODEL_ROULETTE_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write an example config file.
    Init {
        #[arg(long)]
        force: bool,
    },
    /// Run the proxy in the foreground.
    Serve {
        /// Override the port from the config.
        #[arg(long)]
        port: Option<u16>,
    },
    /// Show accounts, cooldowns and recent sessions.
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Clear cooldowns (all accounts, or one).
    Reset { account: Option<String> },
    /// List supported provider kinds.
    Providers,
    /// List supported coding harnesses.
    Harnesses,
    /// Print setup instructions for a harness.
    Setup { harness: String },
    /// Persistently configure a harness to offer the roulette model.
    Install { harness: String },
    /// Start a harness wired to the proxy (starting the proxy if needed).
    /// Pass harness arguments after `--`.
    Launch {
        harness: String,
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Run a fake upstream provider for testing (see src/mock.rs).
    MockUpstream {
        #[arg(long, default_value_t = 9999)]
        port: u16,
    },
}

fn config_path(cli: &Cli) -> PathBuf {
    cli.config.clone().unwrap_or_else(Config::default_path)
}

fn load_config(cli: &Cli) -> Result<Config> {
    let path = config_path(cli);
    if !path.exists() {
        bail!(
            "no config at {} — run `model-roulette init` first",
            path.display()
        );
    }
    Config::load(&path)
}

fn init_tracing(to_file: Option<PathBuf>) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("model_roulette=info"));
    let builder = tracing_subscriber::fmt().with_env_filter(filter);
    match to_file.and_then(|p| {
        if let Some(d) = p.parent() {
            let _ = std::fs::create_dir_all(d);
        }
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .ok()
    }) {
        Some(file) => builder
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(file))
            .init(),
        None => builder.with_writer(std::io::stderr).init(),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    let log_file =
        matches!(cli.cmd, Cmd::Launch { .. }).then(|| Config::default_dir().join("proxy.log"));
    init_tracing(log_file);
    match run(cli).await {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    match &cli.cmd {
        Cmd::Init { force } => {
            let path = config_path(&cli);
            if path.exists() && !force {
                bail!(
                    "{} already exists (use --force to overwrite)",
                    path.display()
                );
            }
            if let Some(d) = path.parent() {
                std::fs::create_dir_all(d)?;
            }
            std::fs::write(&path, EXAMPLE_CONFIG)?;
            println!(
                "wrote {}\nEdit the [[accounts]] list, export your API keys, then run `model-roulette serve`.",
                path.display()
            );
        }
        Cmd::Serve { port } => {
            let mut cfg = load_config(&cli)?;
            if let Some(p) = port {
                cfg.server.port = *p;
            }
            model_roulette::server::serve(cfg).await?;
        }
        Cmd::Status { json } => {
            let cfg = load_config(&cli)?;
            let status = fetch_status(&cfg).await.unwrap_or_else(|| {
                let store = StateStore::open(Some(cfg.state_file()));
                Roulette::new(cfg.clone(), store)
                    .map(|r| r.status())
                    .unwrap_or(Value::Null)
            });
            if *json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                print_status(&status);
            }
        }
        Cmd::Reset { account } => {
            let cfg = load_config(&cli)?;
            let url = match account {
                Some(a) => format!("{}/roulette/reset?account={a}", cfg.base_url()),
                None => format!("{}/roulette/reset", cfg.base_url()),
            };
            let http = reqwest::Client::new();
            let live = http
                .post(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
                .is_ok();
            if !live {
                let store = StateStore::open(Some(cfg.state_file()));
                store.reset_account(account.as_deref());
                store.flush()?;
            }
            println!(
                "cooldowns cleared{}",
                account
                    .as_ref()
                    .map(|a| format!(" for {a}"))
                    .unwrap_or_default()
            );
        }
        Cmd::Providers => {
            println!(
                "{:<22} {:<22} {:<26} key env var",
                "provider", "name", "default model"
            );
            for k in ProviderKind::all() {
                let p = k.preset();
                println!(
                    "{:<22} {:<22} {:<26} {}",
                    k.as_str(),
                    p.display,
                    p.default_model.unwrap_or("(set model)"),
                    p.api_key_env.unwrap_or("(set api_key_env)")
                );
            }
        }
        Cmd::Harnesses => {
            for h in harness::registry() {
                println!(
                    "{:<12} {:<12} {:?} (aliases: {})",
                    h.id(),
                    h.display_name(),
                    h.protocol(),
                    h.aliases().join(", ")
                );
            }
        }
        Cmd::Setup { harness: name } => {
            let cfg = load_config(&cli).unwrap_or_default();
            let h = harness::find(name).with_context(|| format!("unknown harness '{name}'"))?;
            println!("{}", h.setup_instructions(&cfg));
        }
        Cmd::Install { harness: name } => {
            let cfg = load_config(&cli)?;
            let h = harness::find(name).with_context(|| format!("unknown harness '{name}'"))?;
            println!("{}", h.install(&cfg)?);
        }
        Cmd::Launch {
            harness: name,
            args,
        } => {
            let cfg = load_config(&cli)?;
            let h = harness::find(name).with_context(|| format!("unknown harness '{name}'"))?;
            let _server = if healthy(&cfg).await {
                None
            } else {
                let running = model_roulette::server::start(cfg.clone())
                    .await
                    .context("starting the proxy (is the port in use?)")?;
                eprintln!(
                    "model-roulette: proxy started on http://{} (log: {})",
                    running.addr,
                    Config::default_dir().join("proxy.log").display()
                );
                Some(running)
            };
            let spec = h.launch(&cfg, args);
            let mut cmd = tokio::process::Command::new(&spec.program);
            cmd.args(&spec.args);
            for k in &spec.env_remove {
                cmd.env_remove(k);
            }
            for (k, v) in &spec.env {
                cmd.env(k, v);
            }
            let status = cmd.status().await.with_context(|| {
                format!(
                    "running `{}` — is {} installed?",
                    spec.program,
                    h.display_name()
                )
            })?;
            if let Some(s) = &_server {
                let _ = s.roulette.store.flush();
            }
            return Ok(ExitCode::from(
                status.code().unwrap_or(1).clamp(0, 255) as u8
            ));
        }
        Cmd::MockUpstream { port } => {
            let (addr, _) = model_roulette::mock::spawn(&format!("127.0.0.1:{port}")).await?;
            println!(
                "mock upstream on http://{addr} (anthropic: /v1/messages, openai: /v1/chat/completions)"
            );
            tokio::signal::ctrl_c().await?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

async fn healthy(cfg: &Config) -> bool {
    reqwest::Client::new()
        .get(format!("{}/health", cfg.base_url()))
        .timeout(Duration::from_millis(800))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

async fn fetch_status(cfg: &Config) -> Option<Value> {
    let resp = reqwest::Client::new()
        .get(format!("{}/roulette/status", cfg.base_url()))
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .ok()?;
    resp.json().await.ok()
}

fn print_status(s: &Value) {
    println!("model: {}", s["model"].as_str().unwrap_or("?"));
    println!();
    println!(
        "{:<3} {:<16} {:<20} {:<26} {:<12} {:>8} {:>8}",
        "#", "account", "provider", "model", "state", "reqs", "fails"
    );
    for (i, a) in s["accounts"]
        .as_array()
        .cloned()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        let state = if !a["enabled"].as_bool().unwrap_or(true) {
            "disabled".to_string()
        } else if !a["has_key"].as_bool().unwrap_or(true) {
            "no key".to_string()
        } else if a["available"].as_bool().unwrap_or(false) {
            "ready".to_string()
        } else {
            format!(
                "cooling {}s",
                a["cooldown_remaining_secs"].as_i64().unwrap_or(0)
            )
        };
        println!(
            "{:<3} {:<16} {:<20} {:<26} {:<12} {:>8} {:>8}",
            i + 1,
            a["id"].as_str().unwrap_or(""),
            a["provider"].as_str().unwrap_or(""),
            a["model"].as_str().unwrap_or(""),
            state,
            a["requests"].as_u64().unwrap_or(0),
            a["failures"].as_u64().unwrap_or(0)
        );
        if let Some(e) = a["last_error"].as_str()
            && !a["available"].as_bool().unwrap_or(true)
        {
            println!("    last error: {e}");
        }
    }
    let sessions = s["recent_sessions"].as_array().cloned().unwrap_or_default();
    if !sessions.is_empty() {
        println!();
        println!(
            "recent sessions ({} total):",
            s["session_count"].as_u64().unwrap_or(0)
        );
        for x in sessions.iter().take(10) {
            println!(
                "  {:<40} on {:<16} switches {:<3} compactions {}",
                x["id"].as_str().unwrap_or(""),
                x["account"].as_str().unwrap_or("-"),
                x["switches"].as_u64().unwrap_or(0),
                x["compactions"].as_u64().unwrap_or(0)
            );
        }
    }
}
