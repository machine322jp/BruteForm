// アプリケーション状態

use std::sync::{Arc, atomic::AtomicBool};
use crossbeam_channel::Receiver;
use crate::model::Cell;
use crate::constants::W;
use super::{Stats, Message};
use super::chain_play::ChainPlay;

/// 画面モード
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    BruteForce,
    ChainPlay,
}

/// アプリケーション状態
pub struct App {
    pub board: Vec<Cell>,
    pub threshold: u32,
    pub lru_k: u32,
    pub out_path: Option<std::path::PathBuf>,
    pub out_name: String,
    pub stop_progress_plateau: f32,
    pub exact_four_only: bool,
    pub profile_enabled: bool,
    pub running: bool,
    pub abort_flag: Arc<AtomicBool>,
    pub rx: Option<Receiver<Message>>,
    pub stats: Stats,
    pub preview: Option<[[u16; W]; 4]>,
    pub log_lines: Vec<String>,
    pub mode: Mode,
    pub cp: ChainPlay,
    pub verbose_logging: bool,
}

impl Default for App {
    fn default() -> Self {
        let mut board = vec![Cell::Blank; W * 14];
        for y in 0..3 {
            for x in 0..W {
                board[y * W + x] = Cell::Any;
            }
        }
        Self {
            board,
            threshold: 7,
            lru_k: 300,
            out_path: None,
            out_name: "results.jsonl".to_string(),
            stop_progress_plateau: 0.0,
            exact_four_only: false,
            profile_enabled: false,
            running: false,
            abort_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            stats: Stats::default(),
            preview: None,
            log_lines: vec!["待機中".into()],
            mode: Mode::BruteForce,
            cp: ChainPlay::default(),
            verbose_logging: false,
        }
    }
}

impl App {
    pub fn push_log(&mut self, s: String) {
        self.log_lines.push(s);
        if self.log_lines.len() > 500 {
            let cut = self.log_lines.len() - 500;
            self.log_lines.drain(0..cut);
        }
    }
}
