// 連鎖の頭伸ばしロジック

use crate::app::board_evaluation::BoardEvaluator;
use crate::app::bruteforce_search::BruteforceSearch;
use crate::domain::chain_legacy::{board_to_cols, cols_to_board, Board, ChainStep, Detector};
use crate::constants::{H, W};
use crate::vlog;
use std::collections::{HashMap, HashSet};

/// 連鎖の頭伸ばしに関する操作
pub struct ChainExtension;

impl ChainExtension {
    /// 1連鎖目のフリートップを削除
    ///
    /// # 引数
    /// - `target_board`: 目標盤面（ミュータブル、削除されたマスが更新される）
    /// - `chain_info`: 連鎖情報（1連鎖目のグループ情報が必要）
    /// - `around_cache`: 周囲マスキャッシュ（更新される）
    ///
    /// # 戻り値
    /// - `Ok((usize, usize))`: 削除したマスの座標 (x, y)
    /// - `Err(String)`: エラーメッセージ
    pub fn remove_first_chain_freetop(
        target_board: &mut [[u16; W]; 4],
        chain_info: &[ChainStep],
        around_cache: &mut Option<(HashSet<(usize, usize)>, HashSet<(usize, usize)>)>,
    ) -> Result<(usize, usize), String> {
        if chain_info.is_empty() {
            return Err("連鎖情報が空です".to_string());
        }

        // 1連鎖目で消えるマスを取得
        let first_chain_cells: HashSet<(usize, usize)> = chain_info[0]
            .original_groups
            .iter()
            .flat_map(|group| group.iter().copied())
            .collect();

        // 各列の1連鎖目ぷよの最上部を取得
        let mut col_max_y = HashMap::new();
        for &(x, y) in &first_chain_cells {
            col_max_y
                .entry(x)
                .and_modify(|max_y| {
                    if y > *max_y {
                        *max_y = y;
                    }
                })
                .or_insert(y);
        }

        // 盤面を取得して、その上にぷよがないかチェック
        let board = cols_to_board(target_board);

        // フリートップ列（1連鎖目の最上部の上にぷよがない列）を特定
        let mut freetop_cols: Vec<(usize, usize)> = Vec::new();
        for (&x, &max_y) in col_max_y.iter() {
            // この列の最上部の上にぷよがあるかチェック
            let mut has_puyo_above = false;
            for y in (max_y + 1)..H {
                if board[y][x].is_some() {
                    has_puyo_above = true;
                    break;
                }
            }

            // 上にぷよがない = フリートップ列
            if !has_puyo_above {
                freetop_cols.push((x, max_y));
            }
        }

        if freetop_cols.is_empty() {
            return Err(
                "1連鎖目にフリートップ列がありません（全ての列で上にぷよが乗っています）"
                    .to_string(),
            );
        }

        // フリートップ列を左から選ぶ
        freetop_cols.sort_by_key(|&(x, _)| x);
        let (remove_col, remove_y) = freetop_cols[0];

        vlog!(
            "[フリートップ削除] 候補: {:?}, 選択: ({}, {})",
            freetop_cols,
            remove_col,
            remove_y
        );

        // 目標盤面から該当マスを削除
        let bit = 1u16 << remove_y;
        for c in 0..4 {
            target_board[c][remove_col] &= !bit;
        }

        // 連鎖情報は保持したまま、周囲7マスに更新（2マス分を頭伸ばし用に残す）
        if !chain_info.is_empty() {
            let mut first_chain_cells_updated = HashSet::new();

            // 1連鎖目（削除後も残っているマス）
            for group in &chain_info[0].original_groups {
                for &(x, y) in group {
                    // 削除したマスを除外
                    if x != remove_col || y != remove_y {
                        first_chain_cells_updated.insert((x, y));
                    }
                }
            }

            // 周囲7マスを収集（削除したマスとその上の2マスを除く）
            let around_first = BoardEvaluator::collect_empty_cells_bfs(
                target_board,
                &first_chain_cells_updated,
                7,
            );
            if let Some((_, ref last)) = around_cache {
                *around_cache = Some((around_first, last.clone()));
            }
        }

        Ok((remove_col, remove_y))
    }

    /// 頭伸ばし（連鎖数を増やす）
    ///
    /// # 引数
    /// - `target_board`: 現在の目標盤面（削除後の状態）
    /// - `baseline`: ベースライン連鎖数（この値を超える配置を探す）
    /// - `removed_freetop`: 削除したフリートップの座標 (x, y)
    /// - `chain_info`: 削除前の連鎖情報
    /// - `around_cache`: 周囲マスキャッシュ（削除前の情報）
    ///
    /// # 戻り値
    /// - `Ok((i32, [[u16; W]; 4]))`: (新しい連鎖数, 新しい目標盤面)
    /// - `Err(String)`: エラーメッセージ
    pub fn extend_head(
        target_board: &[[u16; W]; 4],
        baseline: i32,
        removed_freetop: (usize, usize),
        chain_info: &[ChainStep],
        around_cache: &Option<(HashSet<(usize, usize)>, HashSet<(usize, usize)>)>,
    ) -> Result<(i32, [[u16; W]; 4]), String> {
        let (rx, ry) = removed_freetop;

        // 削除後の盤面での連鎖数も確認
        let base_board = cols_to_board(target_board);
        let mut baseline_detector = Detector::new(base_board.clone());
        let baseline_after_remove = baseline_detector.simulate_chain() as i32;

        vlog!(
            "[頭伸ばし] 削除前の連鎖数={}, 削除後の連鎖数={}",
            baseline,
            baseline_after_remove
        );

        // 1連鎖目に消えるぷよの位置を特定（削除前の情報）
        let mut first_chain_puyos = HashSet::new();
        for group in &chain_info[0].original_groups {
            for &(x, y) in group {
                first_chain_puyos.insert((x, y));
            }
        }

        // 各列の1連鎖目ぷよの最上部を計算
        let mut first_chain_top: HashMap<usize, usize> = HashMap::new();
        for &(x, y) in first_chain_puyos.iter() {
            first_chain_top
                .entry(x)
                .and_modify(|max_y| *max_y = (*max_y).max(y))
                .or_insert(y);
        }

        vlog!(
            "[頭伸ばし] 削除前の1連鎖目の列と最上部={:?}",
            {
                let mut v: Vec<_> = first_chain_top.iter().map(|(x, y)| (*x, *y)).collect();
                v.sort_unstable();
                v
            }
        );

        // 候補位置：周囲9マス（削除前の周囲マス + 削除したマス）
        let Some((ref around_first_before_remove, _)) = around_cache else {
            return Err("周囲マス情報がありません".to_string());
        };

        // 削除前の周囲9マスを使用し、削除後の盤面で空マスのみフィルタ
        let mut candidate_positions = HashSet::new();
        for &(x, y) in around_first_before_remove.iter() {
            // 削除後の盤面で空マスか確認
            if base_board[y][x].is_none() {
                candidate_positions.insert((x, y));
            }
        }

        // 削除したマスも候補に追加（最重要！）
        candidate_positions.insert((rx, ry));

        // 削除マスの上も候補に追加（9マス目）
        if ry + 1 < H && base_board[ry + 1][rx].is_none() {
            candidate_positions.insert((rx, ry + 1));
        }

        vlog!(
            "[頭伸ばし] 削除前周囲マス数={}, 削除マス含む候補位置数={}",
            around_first_before_remove.len(),
            candidate_positions.len()
        );

        // 候補列を収集
        let mut cand_cols = HashSet::new();
        for &(x, _) in candidate_positions.iter() {
            cand_cols.insert(x);
        }

        vlog!(
            "[頭伸ばし] 候補位置数={} 候補列={:?}",
            candidate_positions.len(),
            {
                let mut v: Vec<_> = cand_cols.iter().copied().collect();
                v.sort_unstable();
                v
            }
        );
        vlog!("[頭伸ばし] 削除マス=({}, {})", rx, ry);

        // 候補位置を配列に変換
        let mut cand_positions: Vec<(usize, usize)> = candidate_positions.iter().copied().collect();
        cand_positions.sort();

        vlog!("[頭伸ばし] 候補位置={:?}", cand_positions);

        // 完全総当たり：各位置に4色または空欄を配置（5^N通り）
        let total_combinations = 5usize.pow(cand_positions.len() as u32);
        vlog!(
            "[頭伸ばし] 総当たり開始: {}^{} = {} 通り（空欄含む）",
            5,
            cand_positions.len(),
            total_combinations
        );

        let max_results = if total_combinations > 1_000_000 {
            100
        } else {
            total_combinations
        };

        let results_raw = BruteforceSearch::search_all_colors(
            &base_board,
            &cand_positions,
            baseline,
            max_results,
            &first_chain_top,
        );

        let mut all_results: Vec<(i32, Board, (usize, usize))> = results_raw
            .into_iter()
            .map(|(chain, board, placed)| {
                let seed = placed.first().map(|(x, y, _)| (*x, *y)).unwrap_or((0, 0));
                (chain, board, seed)
            })
            .collect();

        // 連鎖数降順でソート
        all_results.sort_by(|a, b| b.0.cmp(&a.0));
        let results = all_results;

        vlog!(
            "[頭伸ばし] find_best_arrangement: results={} 件",
            results.len()
        );

        // 結果の詳細をログ出力（最初の10件）
        for (i, (ch, _, seed)) in results.iter().enumerate().take(10) {
            vlog!("[頭伸ばし] result[{}]: chain={}, seed={:?}", i, ch, seed);
        }

        // 連鎖が伸びる最良案を採用
        for (ch, board_pre, seed) in results {
            vlog!(
                "[頭伸ばし] 候補確認: chain={} vs baseline={} seed={:?}",
                ch,
                baseline,
                seed
            );
            if ch > baseline {
                vlog!(
                    "[頭伸ばし] 採用: chain={} > baseline={} seed={:?}",
                    ch,
                    baseline,
                    seed
                );
                let new_cols = board_to_cols(&board_pre);

                // 更新前のハッシュを計算（デバッグ用）
                let old_hash: u64 = target_board
                    .iter()
                    .flat_map(|color_col| color_col.iter())
                    .fold(0u64, |acc, &val| {
                        acc.wrapping_mul(31).wrapping_add(val as u64)
                    });

                // 更新後のハッシュを計算（デバッグ用）
                let new_hash: u64 = new_cols
                    .iter()
                    .flat_map(|color_col| color_col.iter())
                    .fold(0u64, |acc, &val| {
                        acc.wrapping_mul(31).wrapping_add(val as u64)
                    });

                vlog!(
                    "[頭伸ばし] target_board更新: old_hash={}, new_hash={}",
                    old_hash,
                    new_hash
                );

                return Ok((ch, new_cols));
            }
        }

        vlog!("[頭伸ばし] 不採用: baseline={} を超える候補なし", baseline);
        Err(format!("連鎖は{}連鎖のまま", baseline))
    }
}
