// 総当たり探索ロジック

use crate::app::board_evaluation::BoardEvaluator;
use crate::domain::chain_legacy::{apply_gravity, board_to_cols, Board, CellData, Detector, IterId};
use crate::constants::{H, W};
use crate::vlog;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// 総当たり探索の結果
pub type SearchResult = (i32, Board, Vec<(usize, usize, u8)>);

/// 総当たり探索エンジン
pub struct BruteforceSearch;

impl BruteforceSearch {
    /// 二項係数 nCk を計算
    #[allow(dead_code)]
    pub fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        let k = k.min(n - k);
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// 候補位置に各色のぷよを配置する総当たり（重力適用後に連鎖検出）
    /// 各位置に0〜3の色または空欄を割り当てる全組み合わせを試す（5^N通り）
    /// フリートップ列が1列以上残っている配置のみ有効
    /// 並列化版：先頭の候補で分岐して並列処理
    pub fn search_all_colors(
        base_board: &Board,
        candidates: &[(usize, usize)],
        baseline: i32,
        max_results: usize,
        first_chain_top: &HashMap<usize, usize>,
    ) -> Vec<SearchResult> {
        if candidates.is_empty() {
            return Vec::new();
        }

        // 候補が少ない場合は単一スレッドで実行
        if candidates.len() <= 3 {
            return Self::search_single_thread(
                base_board,
                candidates,
                baseline,
                max_results,
                first_chain_top,
            );
        }

        // 並列化：先頭の候補位置で分岐（5通り：空欄 + 4色）
        let first_pos = candidates[0];
        let rest_candidates = &candidates[1..];

        vlog!(
            "  [並列総当たり] 先頭={:?}, 残り={} 候補",
            first_pos,
            rest_candidates.len()
        );

        // 5通りの初期配置を並列処理（重複排除付き）
        let initial_placements: Vec<Option<u8>> = vec![None, Some(0), Some(1), Some(2), Some(3)];

        let all_results: Vec<Vec<SearchResult>> = initial_placements
            .par_iter()
            .map(|&first_color| {
                let mut local_placed = Vec::new();
                if let Some(color) = first_color {
                    local_placed.push((first_pos.0, first_pos.1, color));
                }

                let mut local_results = Vec::new();
                let mut local_trial_count = 0usize;
                let mut local_seen_hashes = HashSet::new(); // 重複排除用

                Self::dfs_deduplicated(
                    base_board,
                    rest_candidates,
                    0,
                    &mut local_placed,
                    &mut local_results,
                    baseline,
                    max_results / 5 + 1, // 各スレッドに上限を分配
                    &mut local_trial_count,
                    first_chain_top,
                    &mut local_seen_hashes,
                );

                vlog!(
                    "  [並列総当たり] 初期配置={:?}: 試行数={}, 結果={}, 重複排除={}",
                    first_color,
                    local_trial_count,
                    local_results.len(),
                    local_seen_hashes.len()
                );

                local_results
            })
            .collect();

        // 結果を統合してソート（グローバル重複排除）
        let mut global_seen = HashSet::new();
        let mut combined_results: Vec<SearchResult> = Vec::new();

        for result in all_results.into_iter().flatten() {
            let cols = board_to_cols(&result.1);
            let hash = BoardEvaluator::compute_board_hash(&cols);

            if global_seen.insert(hash) {
                combined_results.push(result);
            }
        }

        combined_results.sort_by(|a, b| b.0.cmp(&a.0)); // 連鎖数降順

        if combined_results.len() > max_results {
            combined_results.truncate(max_results);
        }

        vlog!(
            "  [並列総当たり] 統合完了: 結果={} 件",
            combined_results.len()
        );

        combined_results
    }

    /// 単一スレッド版の総当たり（候補数が少ない場合用）
    fn search_single_thread(
        base_board: &Board,
        candidates: &[(usize, usize)],
        baseline: i32,
        max_results: usize,
        first_chain_top: &HashMap<usize, usize>,
    ) -> Vec<SearchResult> {
        let mut results = Vec::new();
        let mut trial_count = 0usize;
        let mut placed = Vec::new();
        let mut seen_hashes = HashSet::new();

        Self::dfs_deduplicated(
            base_board,
            candidates,
            0,
            &mut placed,
            &mut results,
            baseline,
            max_results,
            &mut trial_count,
            first_chain_top,
            &mut seen_hashes,
        );

        vlog!(
            "  [単一総当たり] dfs完了: 試行数={}, ベースライン超え={} 件, 重複排除={}",
            trial_count,
            results.len(),
            seen_hashes.len()
        );

        results
    }

    /// DFS本体（重複排除版）
    fn dfs_deduplicated(
        base_board: &Board,
        candidates: &[(usize, usize)],
        idx: usize,
        placed: &mut Vec<(usize, usize, u8)>,
        results: &mut Vec<SearchResult>,
        baseline: i32,
        max_results: usize,
        trial_count: &mut usize,
        first_chain_top: &HashMap<usize, usize>,
        seen_hashes: &mut HashSet<u64>,
    ) {
        if idx == candidates.len() {
            // 全位置に色を割り当て完了

            // 空の配置は無効
            if placed.is_empty() {
                return;
            }

            // フリートップ制約チェック：配置するぷよは各列の最上部でなければならない
            let mut board = base_board.clone();

            // 列ごとの最上部の高さを計算
            let mut col_top: [Option<usize>; 6] = [None; 6];
            for x in 0..W {
                for y in (0..H).rev() {
                    if board[y][x].is_some() {
                        col_top[x] = Some(y + 1); // 最上部の1つ上
                        break;
                    }
                }
                if col_top[x].is_none() {
                    col_top[x] = Some(0); // 空列
                }
            }

            // 配置するぷよが各列のフリートップかチェック
            let mut col_expected_top = col_top.clone();
            let mut valid = true;

            // y座標が小さい順（底から上）にソート
            let mut sorted_placed = placed.clone();
            sorted_placed.sort_by(|a, b| a.1.cmp(&b.1));

            for &(x, y, _color) in sorted_placed.iter() {
                if let Some(expected_y) = col_expected_top[x] {
                    if y != expected_y {
                        // フリートップでない位置に配置しようとしている
                        valid = false;
                        break;
                    }
                    // 次のフリートップを更新
                    col_expected_top[x] = Some(expected_y + 1);
                } else {
                    valid = false;
                    break;
                }
            }

            if !valid {
                return; // この配置は無効
            }

            // 有効な配置のみ実行（配置後に重力適用して連鎖シミュレーション）
            for &(x, y, color) in placed.iter() {
                if board[y][x].is_none() {
                    board[y][x] = Some(CellData {
                        color,
                        iter: IterId(0),
                        original_pos: Some((x, y)),
                    });
                }
            }

            // 重力適用
            apply_gravity(&mut board);

            // 連鎖検出
            let mut detector = Detector::new(board.clone());
            let chain = detector.simulate_chain();

            // 1連鎖目の連結数チェック：5個未満（つまり4個のみ）
            if chain > 0 && !detector.chain_history.is_empty() {
                for group in &detector.chain_history[0].original_groups {
                    if group.len() >= 5 {
                        return; // 5個以上は無効
                    }
                }
            }

            // フリートップ列チェック：配置後に1連鎖目が発生し、フリートップ列が1列以上あるか
            if chain > 0 && !detector.chain_history.is_empty() {
                // 1連鎖目に消えるぷよの位置を取得
                let mut new_first_chain_puyos = HashSet::new();
                for group in &detector.chain_history[0].original_groups {
                    for &(x, y) in group {
                        new_first_chain_puyos.insert((x, y));
                    }
                }

                // 各列の1連鎖目ぷよの最上部を計算
                let mut new_first_chain_top: HashMap<usize, usize> = HashMap::new();
                for &(x, y) in new_first_chain_puyos.iter() {
                    new_first_chain_top
                        .entry(x)
                        .and_modify(|max_y| *max_y = (*max_y).max(y))
                        .or_insert(y);
                }

                // フリートップ列が1列以上あるかチェック
                let mut has_freetop = false;
                for (&x, &first_chain_max_y) in new_first_chain_top.iter() {
                    let freetop_start_y = first_chain_max_y + 1;

                    // その上にぷよがあるか
                    let mut has_puyo_above = false;
                    for y in freetop_start_y..H {
                        if board[y][x].is_some() {
                            has_puyo_above = true;
                            break;
                        }
                    }

                    // その上に何もない = フリートップ列
                    if !has_puyo_above {
                        // さらに、候補位置内にfreetop_start_yがあるか
                        for &(cx, cy) in candidates {
                            if cx == x && cy == freetop_start_y {
                                has_freetop = true;
                                break;
                            }
                        }
                    }

                    if has_freetop {
                        break;
                    }
                }

                if !has_freetop {
                    return; // フリートップ列がない配置は無効
                }
            } else {
                // 連鎖が発生しない場合は無効
                return;
            }

            *trial_count += 1;

            // デバッグ：全ての結果をログ出力（最初の10件のみ）
            if *trial_count <= 10 {
                vlog!(
                    "  [総当たり] trial={}, placed={:?}, chain={} (baseline={})",
                    trial_count,
                    placed,
                    chain,
                    baseline
                );
            } else if *trial_count % 10000 == 0 {
                vlog!("  [総当たり] 試行中... {} / ? 件", trial_count);
            }

            // デバッグ：baseline以上（=含む）の結果を最初の5件表示
            if chain >= baseline && results.len() < 5 {
                vlog!(
                    "  [総当たり] ★baseline以上: trial={}, chain={}, placed={:?}",
                    trial_count,
                    chain,
                    placed
                );
            }

            // ベースラインを超える場合のみ結果に追加（重複チェック付き）
            if chain > baseline {
                let cols = board_to_cols(&board);
                let hash = BoardEvaluator::compute_board_hash(&cols);

                if seen_hashes.insert(hash) {
                    results.push((chain, board, placed.clone()));
                    if results.len() >= max_results {
                        return; // 上限到達で打ち切り
                    }
                }
            }
            return;
        }

        // 現在の位置に各色を試す（空欄も含む）
        let (x, y) = candidates[idx];

        // 早期枝刈り：結果が上限に達したら打ち切り
        if results.len() >= max_results {
            return;
        }

        // 空欄を試す
        Self::dfs_deduplicated(
            base_board,
            candidates,
            idx + 1,
            placed,
            results,
            baseline,
            max_results,
            trial_count,
            first_chain_top,
            seen_hashes,
        );
        if results.len() >= max_results {
            return;
        }

        // 各色を試す
        for color in 0..4u8 {
            placed.push((x, y, color));
            Self::dfs_deduplicated(
                base_board,
                candidates,
                idx + 1,
                placed,
                results,
                baseline,
                max_results,
                trial_count,
                first_chain_top,
                seen_hashes,
            );
            placed.pop();

            if results.len() >= max_results {
                return; // 上限到達で打ち切り
            }
        }
    }
}
