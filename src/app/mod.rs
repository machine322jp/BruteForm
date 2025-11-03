// アプリケーション層

pub mod board_evaluation;
pub mod chain_extension;
pub mod chain_operations;
pub mod chain_play;
pub mod event_adapter;
pub mod state;
pub mod stats;
pub mod ui;

// 旧実装（連鎖生成機能で使用中、新規コードでは使用しないこと）
mod bruteforce_search;

pub use state::{App, Mode};
pub use stats::{Message, Stats};
