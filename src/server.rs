//! HTTP server wiring.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Query, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::json;

use crate::config::Config;
use crate::frontend::{AppState, anthropic, chat, responses};
use crate::roulette::Roulette;
use crate::state::StateStore;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/messages", post(anthropic::messages))
        .route("/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .route("/v1/responses", post(responses::responses))
        .route("/responses", post(responses::responses))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/chat/completions", post(chat::chat_completions))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route("/health", get(|| async { axum::Json(json!({"ok": true})) }))
        .route("/roulette/status", get(status))
        .route("/roulette/reset", post(reset))
        .layer(DefaultBodyLimit::max(256 * 1024 * 1024))
        .with_state(state)
}

async fn models(State(app): State<AppState>) -> Response {
    let s = &app.roulette.cfg.server;
    let created = 1_767_225_600; // 2026-01-01
    let entry = |id: &str, name: &str| {
        json!({
            "id": id, "object": "model", "type": "model", "display_name": name,
            "created": created, "created_at": "2026-01-01T00:00:00Z", "owned_by": "model-roulette"
        })
    };
    let data = vec![entry(&s.model_name, "Model Roulette"), entry(&s.fast_model_name, "Model Roulette (fast)")];
    axum::Json(json!({
        "object": "list",
        "data": data,
        "has_more": false,
        "first_id": s.model_name,
        "last_id": s.fast_model_name,
    }))
    .into_response()
}

async fn status(State(app): State<AppState>) -> Response {
    axum::Json(app.roulette.status()).into_response()
}

#[derive(serde::Deserialize)]
struct ResetQuery {
    account: Option<String>,
}

async fn reset(State(app): State<AppState>, Query(q): Query<ResetQuery>) -> Response {
    app.roulette.store.reset_account(q.account.as_deref());
    axum::Json(json!({"ok": true})).into_response()
}

pub struct Running {
    pub addr: SocketAddr,
    pub roulette: Arc<Roulette>,
    pub handle: tokio::task::JoinHandle<()>,
}

/// Build the roulette and start serving on `cfg.server.host:port`
/// (port 0 picks a free port). Returns once the socket is bound.
pub async fn start(cfg: Config) -> anyhow::Result<Running> {
    let store = StateStore::open(Some(cfg.state_file()));
    store.spawn_flusher();
    let bind = format!("{}:{}", cfg.server.host, cfg.server.port);
    let roulette = Arc::new(Roulette::new(cfg, Arc::clone(&store))?);
    for a in &roulette.accounts {
        if !roulette.usable(a) {
            let why = if !a.cfg.enabled {
                "disabled".to_string()
            } else {
                format!("no API key (set {})", a.cfg.api_key_env_name().unwrap_or_else(|| "api_key".into()))
            };
            tracing::warn!(account = a.id(), "account not in rotation: {why}");
        }
    }
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    let addr = listener.local_addr()?;
    let app = router(AppState { roulette: Arc::clone(&roulette) });
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!("server error: {e}");
        }
    });
    Ok(Running { addr, roulette, handle })
}

/// Run until Ctrl-C, flushing state on exit.
pub async fn serve(cfg: Config) -> anyhow::Result<()> {
    let running = start(cfg).await?;
    let r = &running.roulette;
    tracing::info!(
        "model-roulette listening on http://{} — model \"{}\" over {} account(s): {}",
        running.addr,
        r.cfg.server.model_name,
        r.accounts.iter().filter(|a| r.usable(a)).count(),
        r.accounts.iter().map(|a| format!("{} ({}/{})", a.id(), a.cfg.provider.as_str(), a.cfg.model())).collect::<Vec<_>>().join(" → ")
    );
    tokio::signal::ctrl_c().await?;
    tracing::info!("shutting down");
    let _ = running.roulette.store.flush();
    Ok(())
}
