// 連鎖操作の後方互換レイヤー（新しいモジュール構造への橋渡し）

use crate::app::chain_play::{
    Animation, ChainPlay, Orient, PiecePlacer, StateManager, TargetOperations,
};
use crate::constants::W;

/// 連鎖操作のユーティリティ関数群（後方互換性のため）
pub struct ChainOperations;

impl ChainOperations {
    // ===== ペア取得 =====

    pub fn current_pair(cp: &ChainPlay) -> (u8, u8) {
        StateManager::current_pair(cp)
    }

    pub fn next_pair(cp: &ChainPlay) -> (u8, u8) {
        StateManager::next_pair(cp)
    }

    pub fn dnext_pair(cp: &ChainPlay) -> (u8, u8) {
        StateManager::dnext_pair(cp)
    }

    // ===== 状態管理 =====

    pub fn undo(cp: &mut ChainPlay) {
        StateManager::undo(cp);
    }

    pub fn reset_to_initial(cp: &mut ChainPlay) {
        StateManager::reset_to_initial(cp);
    }

    // ===== 盤面操作 =====

    pub fn col_height(cols: &[[u16; W]; 4], x: usize) -> usize {
        PiecePlacer::col_height(cols, x)
    }

    pub fn place_with(cp: &mut ChainPlay, x: usize, orient: Orient, pair: (u8, u8)) {
        PiecePlacer::place_with(cp, x, orient, pair);
    }

    pub fn place_random(cp: &mut ChainPlay) -> Result<(), String> {
        PiecePlacer::place_random(cp)
    }

    pub fn check_and_start_chain(cp: &mut ChainPlay) {
        Animation::check_and_start_chain(cp);
    }

    pub fn step_animation(cp: &mut ChainPlay) {
        Animation::step_animation(cp);
    }

    // ===== 目標盤面操作 =====

    pub fn update_target(cp: &mut ChainPlay, beam_width: usize, max_depth: u8) -> String {
        TargetOperations::update_target(cp, beam_width, max_depth)
    }

    pub fn detect_target_chain(cp: &mut ChainPlay) -> Result<String, String> {
        TargetOperations::detect_target_chain(cp)
    }

    pub fn place_target(
        cp: &mut ChainPlay,
        target_board: &[[u16; W]; 4],
    ) -> Result<(usize, Orient, i32), String> {
        TargetOperations::place_target(cp, target_board)
    }
}
