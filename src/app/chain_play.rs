// 連鎖生成モード関連

use std::time::Instant;
use rand::Rng;
use crate::constants::W;

/// アニメーション段階
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum AnimPhase {
    AfterErase,
    AfterFall,
}

#[derive(Clone, Copy)]
pub struct AnimState {
    pub phase: AnimPhase,
    pub since: Instant,
}

/// 保存状態
#[derive(Clone, Copy)]
pub struct SavedState {
    pub cols: [[u16; W]; 4],
    pub pair_index: usize,
}

/// 向き
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Orient {
    Up,
    Right,
    Down,
    Left,
}

/// 連鎖生成モード状態
pub struct ChainPlay {
    pub cols: [[u16; W]; 4],
    pub pair_seq: Vec<(u8, u8)>,
    pub pair_index: usize,
    pub undo_stack: Vec<SavedState>,
    pub anim: Option<AnimState>,
    pub lock: bool,
    pub erased_cols: Option<[[u16; W]; 4]>,
    pub next_cols: Option<[[u16; W]; 4]>,
    pub target_board: [[u16; W]; 4],
}

impl Default for ChainPlay {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut seq = Vec::with_capacity(128);
        for _ in 0..128 {
            seq.push((rng.gen_range(0..4), rng.gen_range(0..4)));
        }
        let cols = [[0u16; W]; 4];
        let s0 = SavedState { cols, pair_index: 0 };
        Self {
            cols,
            pair_seq: seq,
            pair_index: 0,
            undo_stack: vec![s0],
            anim: None,
            lock: false,
            erased_cols: None,
            next_cols: None,
            target_board: [[0u16; W]; 4],
        }
    }
}
