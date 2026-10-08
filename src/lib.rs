//! model-roulette: one model name that rotates across many provider accounts.
//!
//! See README.md for usage and AGENTS.md for the architecture.

pub mod canonical;
pub mod compaction;
pub mod config;
pub mod frontend;
pub mod harness;
pub mod mock;
pub mod providers;
pub mod ratelimit;
pub mod roulette;
pub mod server;
pub mod state;
pub mod upstream;
