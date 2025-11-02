// 状態管理（undo/redo）

use crate::app::chain_play::{ChainPlay, SavedState};

/// 状態管理のユーティリティ
pub struct StateManager;

impl StateManager {
    /// 1手戻る
    pub fn undo(cp: &mut ChainPlay) {
        if cp.undo_stack.len() > 1 && !cp.lock {
            cp.anim = None;
            cp.erased_cols = None;
            cp.next_cols = None;
            // 現在の状態を削除
            cp.undo_stack.pop();
            // 直前の状態を取得（削除はしない）
            if let Some(last) = cp.undo_stack.last().copied() {
                cp.cols = last.cols;
                cp.pair_index = last.pair_index;
            }
        }
    }

    /// 初手に戻る
    pub fn reset_to_initial(cp: &mut ChainPlay) {
        if cp.lock {
            return;
        }
        cp.anim = None;
        cp.erased_cols = None;
        cp.next_cols = None;
        if let Some(first) = cp.undo_stack.first().copied() {
            cp.cols = first.cols;
            cp.pair_index = first.pair_index;
            cp.undo_stack.clear();
            cp.undo_stack.push(first);
        }
        cp.lock = false;
    }

    /// 状態を保存
    pub fn save_state(cp: &mut ChainPlay) {
        cp.undo_stack.push(SavedState {
            cols: cp.cols,
            pair_index: cp.pair_index,
        });
    }

    /// 現在のペアを取得
    pub fn current_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = cp.pair_index % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }

    /// 次のペアを取得
    pub fn next_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = (cp.pair_index + 1) % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }

    /// 次の次のペアを取得
    pub fn dnext_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = (cp.pair_index + 2) % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }
}
