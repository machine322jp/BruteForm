// 連鎖生成モード関連の構造体定義

use crate::constants::W;
use rand::Rng;
use std::time::Instant;

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
    // 追撃最適化（ビーム探索）パラメータ
    pub beam_width: usize,
    pub max_depth: usize,
    // 連鎖検出結果
    pub target_chain_info: Option<Vec<crate::domain::chain_legacy::detector::ChainStep>>,
    // 周囲9マスのキャッシュ（1連鎖目と最終連鎖用）
    pub around_cells_cache: Option<(
        std::collections::HashSet<(usize, usize)>,
        std::collections::HashSet<(usize, usize)>,
    )>,
    // 削除したフリートップの位置（頭伸ばし用）
    pub removed_freetop: Option<(usize, usize)>,
}

impl Default for ChainPlay {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut seq = Vec::with_capacity(128);
        for _ in 0..128 {
            seq.push((rng.gen_range(0..4), rng.gen_range(0..4)));
        }
        let cols = [[0u16; W]; 4];
        let s0 = SavedState {
            cols,
            pair_index: 0,
        };
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
            beam_width: 10,
            max_depth: 8,
            target_chain_info: None,
            around_cells_cache: None,
            removed_freetop: None,
        }
    }
}
