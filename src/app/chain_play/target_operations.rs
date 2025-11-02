// 目標盤面の操作

use super::animation::Animation;
use super::piece_placer::PiecePlacer;
use super::state_manager::StateManager;
use crate::app::board_evaluation::BoardEvaluator;
use crate::app::chain_play::{ChainPlay, Orient};
use crate::chain::{cols_to_board, Detector};
use crate::constants::{H, W};
use crate::vlog;
use std::collections::HashSet;

/// 目標盤面操作のユーティリティ
pub struct TargetOperations;

impl TargetOperations {
    /// 目標盤面を更新
    pub fn update_target(cp: &mut ChainPlay, beam_width: usize, max_depth: u8) -> String {
        if cp.lock || cp.anim.is_some() {
            return "操作がロックされています".to_string();
        }
        // 実盤面から右盤面相当（目標盤面）を生成
        let bw = beam_width.max(1);
        let md_u8 = max_depth.max(1).min(255);
        let (target, chain) =
            crate::chain::compute_target_from_actual_with_params(&cp.cols, bw, md_u8);
        cp.target_board = target;
        format!(
            "目標盤面を更新しました（推定最大連鎖: {} / ビーム幅: {} / 最大深さ: {}）",
            chain, bw, md_u8
        )
    }

    /// 目標盤面の連鎖を検出
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
            let around_first =
                BoardEvaluator::collect_empty_cells_bfs(&cp.target_board, &first_chain_cells, 9);
            let around_last =
                BoardEvaluator::collect_empty_cells_bfs(&cp.target_board, &last_chain_cells, 9);
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

    /// 目標盤面に近づく手を打つ
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
        let current_hash: u64 = target_board
            .iter()
            .flat_map(|color_col| color_col.iter())
            .fold(0u64, |acc, &val| {
                acc.wrapping_mul(31).wrapping_add(val as u64)
            });
        vlog!("[目標配置] 参照するtarget_boardのハッシュ={}", current_hash);

        let pair = StateManager::current_pair(cp);

        let mut moves: Vec<(usize, Orient, i32)> = Vec::new();
        for x in 0..W {
            let h = PiecePlacer::col_height(&cp.cols, x);
            if h + 1 < H {
                let score_up =
                    BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Up, pair);
                let score_down =
                    BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Down, pair);
                moves.push((x, Orient::Up, score_up));
                moves.push((x, Orient::Down, score_down));
            }
            if x + 1 < W {
                let h0 = PiecePlacer::col_height(&cp.cols, x);
                let h1 = PiecePlacer::col_height(&cp.cols, x + 1);
                if h0 < H && h1 < H {
                    let score =
                        BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Right, pair);
                    moves.push((x, Orient::Right, score));
                }
            }
            if x >= 1 {
                let h0 = PiecePlacer::col_height(&cp.cols, x);
                let h1 = PiecePlacer::col_height(&cp.cols, x - 1);
                if h0 < H && h1 < H {
                    let score =
                        BoardEvaluator::eval_move(&cp.cols, target_board, x, Orient::Left, pair);
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

        PiecePlacer::place_with(cp, x, orient, pair);
        Animation::check_and_start_chain(cp);

        // 手を打った後に状態を保存
        StateManager::save_state(cp);

        Ok((x, orient, score))
    }
}
