// 盤面評価とBFS探索のロジック

use crate::app::chain_play::Orient;
use crate::constants::{H, W};
use crate::search::hash::canonical_hash64_fast;
use crate::vlog;
use std::collections::{HashSet, VecDeque};

/// 盤面評価ユーティリティ
pub struct BoardEvaluator;

impl BoardEvaluator {
    /// 指定された手を評価し、目標盤面との一致度をスコアで返す
    pub fn eval_move(
        actual_cols: &[[u16; W]; 4],
        target_board: &[[u16; W]; 4],
        x: usize,
        orient: Orient,
        pair: (u8, u8),
    ) -> i32 {
        let mut test_cols = *actual_cols;

        // 配置する2つのぷよの位置と色を記録
        let mut placed_puyos: Vec<(usize, usize, u8)> = Vec::new();

        match orient {
            Orient::Up => {
                let h = Self::col_height(&test_cols, x);
                if h + 1 >= H {
                    return -1000;
                }
                let bit0 = 1u16 << h;
                let bit1 = 1u16 << (h + 1);
                test_cols[pair.0 as usize][x] |= bit0;
                test_cols[pair.1 as usize][x] |= bit1;
                placed_puyos.push((x, h, pair.0));
                placed_puyos.push((x, h + 1, pair.1));
            }
            Orient::Down => {
                let h = Self::col_height(&test_cols, x);
                if h + 1 >= H {
                    return -1000;
                }
                let bit0 = 1u16 << h;
                let bit1 = 1u16 << (h + 1);
                test_cols[pair.1 as usize][x] |= bit0;
                test_cols[pair.0 as usize][x] |= bit1;
                placed_puyos.push((x, h, pair.1));
                placed_puyos.push((x, h + 1, pair.0));
            }
            Orient::Right => {
                let h0 = Self::col_height(&test_cols, x);
                let h1 = Self::col_height(&test_cols, x + 1);
                if h0 >= H || h1 >= H || x + 1 >= W {
                    return -1000;
                }
                test_cols[pair.0 as usize][x] |= 1u16 << h0;
                test_cols[pair.1 as usize][x + 1] |= 1u16 << h1;
                placed_puyos.push((x, h0, pair.0));
                placed_puyos.push((x + 1, h1, pair.1));
            }
            Orient::Left => {
                if x == 0 {
                    return -1000;
                }
                let h0 = Self::col_height(&test_cols, x);
                let h1 = Self::col_height(&test_cols, x - 1);
                if h0 >= H || h1 >= H {
                    return -1000;
                }
                test_cols[pair.0 as usize][x] |= 1u16 << h0;
                test_cols[pair.1 as usize][x - 1] |= 1u16 << h1;
                placed_puyos.push((x, h0, pair.0));
                placed_puyos.push((x - 1, h1, pair.1));
            }
        }

        let mut score = 0i32;
        for c in 0..4 {
            for col_x in 0..W {
                let matched = test_cols[c][col_x] & target_board[c][col_x];
                score += matched.count_ones() as i32;
            }
        }

        // 配置したぷよが目標盤面の別の色を上書きしていないかチェック
        for (px, py, color) in placed_puyos {
            let bit = 1u16 << py;
            // この位置に目標盤面でぷよがあるか確認
            for target_color in 0..4u8 {
                if (target_board[target_color as usize][px] & bit) != 0 {
                    // 目標盤面にぷよがある
                    if target_color != color {
                        // 違う色を上書きしている → 大幅減点
                        score -= 100;
                    }
                    break;
                }
            }
        }

        score
    }

    /// 列の高さ（既存ぷよの数）を計算
    pub fn col_height(cols: &[[u16; W]; 4], x: usize) -> usize {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & crate::constants::MASK14;
        occ.count_ones() as usize
    }

    /// 盤面のハッシュ値を計算（正規化版：ミラー対称を考慮）
    pub fn compute_board_hash(cols: &[[u16; W]; 4]) -> u64 {
        let (hash, _mirror) = canonical_hash64_fast(cols);
        hash
    }

    /// 空マスが床または既存ぷよ（または収集済み空マス/連鎖マス）と隣接しているかチェック
    pub fn is_placeable_empty(
        target_board: &[[u16; W]; 4],
        x: usize,
        y: usize,
        collected: &HashSet<(usize, usize)>,
        chain_cells: &HashSet<(usize, usize)>,
    ) -> bool {
        // 空マスでない場合はfalse
        let bit = 1u16 << y;
        let is_empty = target_board[0][x] & bit == 0
            && target_board[1][x] & bit == 0
            && target_board[2][x] & bit == 0
            && target_board[3][x] & bit == 0;

        if !is_empty {
            return false;
        }

        // 床（y=0）の場合は常にtrue
        if y == 0 {
            return true;
        }

        // 下に既存ぷよがあるかチェック
        let bit_below = 1u16 << (y - 1);
        let has_puyo_below = target_board[0][x] & bit_below != 0
            || target_board[1][x] & bit_below != 0
            || target_board[2][x] & bit_below != 0
            || target_board[3][x] & bit_below != 0;

        // または下に収集済み空マス/連鎖マスがあるか
        let has_collected_below =
            collected.contains(&(x, y - 1)) || chain_cells.contains(&(x, y - 1));

        has_puyo_below || has_collected_below
    }

    /// BFSで連鎖マスから隣接する「配置可能な空マス」を距離順に収集（最大max_count個）
    pub fn collect_empty_cells_bfs(
        target_board: &[[u16; W]; 4],
        chain_cells: &HashSet<(usize, usize)>,
        max_count: usize,
    ) -> HashSet<(usize, usize)> {
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<((usize, usize), usize)> = VecDeque::new(); // (位置, 距離)

        // 連鎖マスから開始（距離0）
        for &pos in chain_cells {
            visited.insert(pos);
        }

        vlog!("[周囲9マス探索] 連鎖マス数: {}", chain_cells.len());

        // 連鎖マスに隣接する配置可能な空マスをキューに追加（距離1）
        for &(cx, cy) in chain_cells {
            let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
            for (dx, dy) in dirs {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < W as isize && ny >= 0 && ny < H as isize {
                    let nxu = nx as usize;
                    let nyu = ny as usize;

                    if visited.contains(&(nxu, nyu)) {
                        continue;
                    }

                    // 必ずvisitedに追加（空マスでなくても）
                    visited.insert((nxu, nyu));

                    // 配置可能な空マスかチェック（床または既存ぷよ/収集済みマス/連鎖マスと隣接）
                    if Self::is_placeable_empty(target_board, nxu, nyu, &result, chain_cells) {
                        queue.push_back(((nxu, nyu), 1));
                        result.insert((nxu, nyu));
                        if result.len() <= 20 {
                            vlog!(
                                "[周囲9マス探索] 距離1: ({}, {}) 収集数={}",
                                nxu,
                                nyu,
                                result.len()
                            );
                        }

                        if result.len() >= max_count {
                            vlog!("[周囲9マス探索] 最大{}マス到達、探索終了", max_count);
                            return result;
                        }
                    }
                }
            }
        }

        vlog!(
            "[周囲9マス探索] 距離1探索完了、キューサイズ={}",
            queue.len()
        );

        // BFSで更に隣接する配置可能な空マスを探索
        let mut iteration_count = 0;
        while let Some(((cx, cy), dist)) = queue.pop_front() {
            iteration_count += 1;
            if iteration_count > 10000 {
                vlog!("[周囲9マス探索] 警告: 10000回反復、強制終了");
                break;
            }

            let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
            for (dx, dy) in dirs {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < W as isize && ny >= 0 && ny < H as isize {
                    let nxu = nx as usize;
                    let nyu = ny as usize;

                    if visited.contains(&(nxu, nyu)) {
                        continue;
                    }

                    // 必ずvisitedに追加（空マスでなくても）
                    visited.insert((nxu, nyu));

                    // 配置可能な空マスかチェック（床または既存ぷよ/収集済みマス/連鎖マスと隣接）
                    if Self::is_placeable_empty(target_board, nxu, nyu, &result, chain_cells) {
                        queue.push_back(((nxu, nyu), dist + 1));
                        result.insert((nxu, nyu));
                        if result.len() <= 20 {
                            vlog!(
                                "[周囲9マス探索] 距離{}: ({}, {}) 収集数={}",
                                dist + 1,
                                nxu,
                                nyu,
                                result.len()
                            );
                        }

                        if result.len() >= max_count {
                            vlog!("[周囲9マス探索] 最大{}マス到達、探索終了", max_count);
                            return result;
                        }
                    }
                }
            }
        }

        vlog!(
            "[周囲9マス探索] 探索完了、収集数={} 反復回数={}",
            result.len(),
            iteration_count
        );
        result
    }
}
