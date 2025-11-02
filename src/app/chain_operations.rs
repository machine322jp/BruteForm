// 連鎖生成モードの操作ロジック

use std::collections::HashSet;
use crate::constants::{W, H, MASK14};
use crate::app::chain_play::{ChainPlay, SavedState, AnimState, AnimPhase, Orient};
use crate::app::board_evaluation::BoardEvaluator;
use crate::search::board::{compute_erase_mask_cols, apply_clear_no_fall, apply_given_clear_and_fall};
use crate::chain::{cols_to_board, Detector};
use crate::vlog;

/// 連鎖操作のユーティリティ関数群
pub struct ChainOperations;

impl ChainOperations {
    // ===== ペア取得 =====
    
    pub fn current_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = cp.pair_index % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }

    pub fn next_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = (cp.pair_index + 1) % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }

    pub fn dnext_pair(cp: &ChainPlay) -> (u8, u8) {
        let idx = (cp.pair_index + 2) % cp.pair_seq.len().max(1);
        cp.pair_seq[idx]
    }

    // ===== 状態管理 =====
    
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

    // ===== 盤面操作 =====
    
    pub fn col_height(cols: &[[u16; W]; 4], x: usize) -> usize {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
        occ.count_ones() as usize
    }

    fn set_cell(cols: &mut [[u16; W]; 4], x: usize, y: usize, color: u8) {
        if x >= W || y >= H {
            return;
        }
        let bit = 1u16 << y;
        let c = (color as usize).min(3);
        cols[c][x] |= bit;
    }

    pub fn place_with(cp: &mut ChainPlay, x: usize, orient: Orient, pair: (u8, u8)) {
        match orient {
            Orient::Up => {
                let h = Self::col_height(&cp.cols, x);
                Self::set_cell(&mut cp.cols, x, h, pair.0);
                Self::set_cell(&mut cp.cols, x, h + 1, pair.1);
            }
            Orient::Down => {
                let h = Self::col_height(&cp.cols, x);
                Self::set_cell(&mut cp.cols, x, h, pair.1);
                Self::set_cell(&mut cp.cols, x, h + 1, pair.0);
            }
            Orient::Right => {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x + 1);
                Self::set_cell(&mut cp.cols, x, h0, pair.0);
                Self::set_cell(&mut cp.cols, x + 1, h1, pair.1);
            }
            Orient::Left => {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x - 1);
                Self::set_cell(&mut cp.cols, x, h0, pair.0);
                Self::set_cell(&mut cp.cols, x - 1, h1, pair.1);
            }
        }
        if !cp.pair_seq.is_empty() {
            cp.pair_index = (cp.pair_index + 1) % cp.pair_seq.len();
        }
    }

    pub fn place_random(cp: &mut ChainPlay) -> Result<(), String> {
        if cp.lock || cp.anim.is_some() {
            return Err("操作がロックされています".to_string());
        }

        let pair = Self::current_pair(cp);
        let mut rng = rand::thread_rng();

        let mut moves: Vec<(usize, Orient)> = Vec::new();
        for x in 0..W {
            let h = Self::col_height(&cp.cols, x);
            if h + 1 < H {
                moves.push((x, Orient::Up));
                moves.push((x, Orient::Down));
            }
            if x + 1 < W {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x + 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Right));
                }
            }
            if x >= 1 {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x - 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Left));
                }
            }
        }
        if moves.is_empty() {
            return Err("置ける場所がありません".to_string());
        }
        let (x, orient) = moves[rand::Rng::gen_range(&mut rng, 0..moves.len())];

        Self::place_with(cp, x, orient, pair);
        Self::check_and_start_chain(cp);
        
        // 手を打った後に状態を保存
        cp.undo_stack.push(SavedState {
            cols: cp.cols,
            pair_index: cp.pair_index,
        });
        
        Ok(())
    }

    pub fn check_and_start_chain(cp: &mut ChainPlay) {
        let clear = compute_erase_mask_cols(&cp.cols);
        let any = (0..W).any(|x| clear[x] != 0);
        if !any {
            return;
        }
        cp.lock = true;
        let erased = apply_clear_no_fall(&cp.cols, &clear);
        let next = apply_given_clear_and_fall(&cp.cols, &clear);
        cp.erased_cols = Some(erased);
        cp.next_cols = Some(next);
        cp.cols = erased;
        cp.anim = Some(AnimState {
            phase: AnimPhase::AfterErase,
            since: std::time::Instant::now(),
        });
    }

    pub fn step_animation(cp: &mut ChainPlay) {
        let Some(anim) = cp.anim else { return };
        let elapsed = anim.since.elapsed();
        if elapsed < std::time::Duration::from_millis(500) {
            return;
        }
        match anim.phase {
            AnimPhase::AfterErase => {
                if let Some(next) = cp.next_cols.take() {
                    cp.cols = next;
                }
                cp.anim = Some(AnimState {
                    phase: AnimPhase::AfterFall,
                    since: std::time::Instant::now(),
                });
            }
            AnimPhase::AfterFall => {
                let clear = compute_erase_mask_cols(&cp.cols);
                let any = (0..W).any(|x| clear[x] != 0);
                if !any {
                    cp.anim = None;
                    cp.erased_cols = None;
                    cp.next_cols = None;
                    cp.lock = false;
                } else {
                    let erased = apply_clear_no_fall(&cp.cols, &clear);
                    let next = apply_given_clear_and_fall(&cp.cols, &clear);
                    cp.cols = erased;
                    cp.erased_cols = Some(erased);
                    cp.next_cols = Some(next);
                    cp.anim = Some(AnimState {
                        phase: AnimPhase::AfterErase,
                        since: std::time::Instant::now(),
                    });
                }
            }
        }
    }

    // ===== 目標盤面操作 =====
    
    pub fn update_target(cp: &mut ChainPlay, beam_width: usize, max_depth: u8) -> String {
        if cp.lock || cp.anim.is_some() {
            return "操作がロックされています".to_string();
        }
        // 実盤面から右盤面相当（目標盤面）を生成
        let bw = beam_width.max(1);
        let md_u8 = max_depth.max(1).min(255);
        let (target, chain) = crate::chain::compute_target_from_actual_with_params(&cp.cols, bw, md_u8);
        cp.target_board = target;
        format!(
            "目標盤面を更新しました（推定最大連鎖: {} / ビーム幅: {} / 最大深さ: {}）",
            chain, bw, md_u8
        )
    }

    pub fn detect_target_chain(cp: &mut ChainPlay) -> Result<String, String> {
        if cp.lock || cp.anim.is_some() {
            return Err("操作がロックされています".to_string());
        }
        
        // 目標盤面が空かチェック
        let target_empty = (0..W).all(|x| {
            cp.target_board[0][x] == 0
                && cp.target_board[1][x] == 0
                && cp.target_board[2][x] == 0
                && cp.target_board[3][x] == 0
        });
        if target_empty {
            return Err("目標盤面が設定されていません".to_string());
        }
        
        // 目標盤面をBoardに変換
        let board = cols_to_board(&cp.target_board);
        let mut detector = Detector::new(board);
        let chain_count = detector.simulate_chain();
        
        // 連鎖検出時はフリートップ削除をリセット
        cp.removed_freetop = None;
        
        if chain_count > 0 {
            cp.target_chain_info = Some(detector.chain_history.clone());
            
            // 周囲9マスを計算してキャッシュ
            let mut first_chain_cells = HashSet::new();
            let mut last_chain_cells = HashSet::new();
            
            // 1連鎖目（元の盤面での位置）
            for group in &detector.chain_history[0].original_groups {
                for &(x, y) in group {
                    first_chain_cells.insert((x, y));
                }
            }
            
            // 最終連鎖（元の盤面での位置）
            let last_idx = detector.chain_history.len() - 1;
            for group in &detector.chain_history[last_idx].original_groups {
                for &(x, y) in group {
                    last_chain_cells.insert((x, y));
                }
            }
            
            // BFSで周囲9マスを収集（頭伸ばし用）
            let around_first = BoardEvaluator::collect_empty_cells_bfs(&cp.target_board, &first_chain_cells, 9);
            let around_last = BoardEvaluator::collect_empty_cells_bfs(&cp.target_board, &last_chain_cells, 9);
            cp.around_cells_cache = Some((around_first, around_last));
            
            Ok(format!(
                "目標盤面の連鎖を検出: {}連鎖（1連鎖目=黄色, {}連鎖目=オレンジ）",
                chain_count, chain_count
            ))
        } else {
            cp.target_chain_info = None;
            cp.around_cells_cache = None;
            Err("目標盤面で連鎖が発生しませんでした".to_string())
        }
    }

    pub fn place_target(
        cp: &mut ChainPlay,
        target_board: &[[u16; W]; 4],
    ) -> Result<(usize, Orient, i32), String> {
        if cp.lock || cp.anim.is_some() {
            return Err("操作がロックされています".to_string());
        }

        let target_empty = (0..W).all(|x| {
            target_board[0][x] == 0
                && target_board[1][x] == 0
                && target_board[2][x] == 0
                && target_board[3][x] == 0
        });
        if target_empty {
            return Err("目標盤面が設定されていません".to_string());
        }
        
        // デバッグ：現在のtarget_boardのハッシュを出力
        let current_hash: u64 = target_board.iter()
            .flat_map(|color_col| color_col.iter())
            .fold(0u64, |acc, &val| acc.wrapping_mul(31).wrapping_add(val as u64));
        vlog!("[目標配置] 参照するtarget_boardのハッシュ={}", current_hash);

        let pair = Self::current_pair(cp);

        let mut moves: Vec<(usize, Orient, i32)> = Vec::new();
        for x in 0..W {
            let h = Self::col_height(&cp.cols, x);
            if h + 1 < H {
                let score_up = BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Up, pair);
                let score_down = BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Down, pair);
                moves.push((x, Orient::Up, score_up));
                moves.push((x, Orient::Down, score_down));
            }
            if x + 1 < W {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x + 1);
                if h0 < H && h1 < H {
                    let score = BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Right, pair);
                    moves.push((x, Orient::Right, score));
                }
            }
            if x >= 1 {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x - 1);
                if h0 < H && h1 < H {
                    let score = BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Left, pair);
                    moves.push((x, Orient::Left, score));
                }
            }
        }
        if moves.is_empty() {
            return Err("置ける場所がありません".to_string());
        }

        let best = moves.iter().max_by_key(|&&(_, _, score)| score).unwrap();
        let (x, orient, score) = *best;

        if score <= 0 {
            return Err("目標に寄与する手がありません（ランダム配置を推奨）".to_string());
        }

        Self::place_with(cp, x, orient, pair);
        Self::check_and_start_chain(cp);
        
        // 手を打った後に状態を保存
        cp.undo_stack.push(SavedState {
            cols: cp.cols,
            pair_index: cp.pair_index,
        });
        
        Ok((x, orient, score))
    }
}
