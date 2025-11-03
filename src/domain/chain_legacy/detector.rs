use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use super::grid::{
    apply_gravity, board_to_cols, find_bottom_empty, find_groups_4plus, get_connected_cells,
    in_range, remove_groups, Board, CellData, IterId,
};
use crate::constants::{H, W};
use crate::domain::board::bitboard::pack_cols;
use crate::domain::board::hash::canonical_hash64_fast;
use crate::vlog;

const LOG_DET_VERBOSE: bool = false;
const LOG_DET_CHAIN_DETAIL: bool = false; // 連鎖シミュレーションの詳細ログ（盤面表示など）

// 色番号を漢字に変換
fn color_name(color: u8) -> &'static str {
    match color {
        0 => "赤",
        1 => "緑",
        2 => "青",
        3 => "黄",
        _ => "?",
    }
}

#[derive(Clone, Debug)]
pub enum RejectKind {
    FirstChainMultipleGroups,
    FirstChainNoLatestAdd,
    MixedIterationInGroup,
}

/// 連鎖ごとの消去情報
#[derive(Clone, Debug)]
pub struct ChainStep {
    pub chain_num: usize,                          // 連鎖番号（1始まり）
    pub groups: Vec<Vec<(usize, usize)>>,          // 消えたグループ（現在の位置）
    pub original_groups: Vec<Vec<(usize, usize)>>, // 消えたグループ（元の盤面での位置）
}

pub struct Detector {
    pub field: Board,
    pub last_reject: Option<RejectKind>,
    pub chain_history: Vec<ChainStep>, // 連鎖履歴
}

impl Detector {
    pub fn new(mut field: Board) -> Self {
        // 全セルに元の位置を設定（まだ設定されていない場合）
        for y in 0..H {
            for x in 0..W {
                if let Some(ref mut cell) = field[y][x] {
                    if cell.original_pos.is_none() {
                        cell.original_pos = Some((x, y));
                    }
                }
            }
        }
        Self {
            field,
            last_reject: None,
            chain_history: Vec::new(),
        }
    }

    fn get_connected_cells(&self, sx: usize, sy: usize) -> Vec<(usize, usize)> {
        get_connected_cells(&self.field, sx, sy)
    }

    pub fn reflect_to(&self, external: &mut Board) {
        for y in 0..H {
            for x in 0..W {
                external[y][x] = self.field[y][x];
            }
        }
    }

    pub fn simulate_chain(&mut self) -> i32 {
        self.last_reject = None;
        self.chain_history.clear();
        let mut chain_count: i32 = 0;

        // ビットボードで高速事前チェック：4個以上の色がなければ早期リターン
        let cols = board_to_cols(&self.field);
        let bb = pack_cols(&cols);
        let has_potential = (bb[0].count_ones() >= 4)
            || (bb[1].count_ones() >= 4)
            || (bb[2].count_ones() >= 4)
            || (bb[3].count_ones() >= 4);

        if !has_potential {
            return 0;
        }

        // デバッグ: 初期盤面を出力
        if LOG_DET_CHAIN_DETAIL {
            vlog!("[検出器/DEBUG] 連鎖シミュレーション開始:");
            for y in (0..H).rev() {
                let mut line = format!("  y={:2}: ", y);
                for x in 0..W {
                    if let Some(c) = self.field[y][x] {
                        line.push_str(&format!("{}(i{}) ", color_name(c.color), c.iter.0));
                    } else {
                        line.push_str("  .   ");
                    }
                }
                vlog!("{}", line);
            }
        }

        loop {
            let groups = find_groups_4plus(&self.field);
            if groups.is_empty() {
                if LOG_DET_CHAIN_DETAIL {
                    vlog!(
                        "[検出器/DEBUG] 連鎖{}後: 消せるグループなし → 終了",
                        chain_count
                    );
                }
                break;
            }

            if LOG_DET_CHAIN_DETAIL {
                vlog!(
                    "[検出器/DEBUG] 連鎖{}: {}個のグループ検出",
                    chain_count,
                    groups.len()
                );
            }

            if chain_count == 0 {
                // 1連鎖目: 複数の独立グループが同時に消える構成を不採用
                if groups.len() > 1 {
                    // デバッグ: 各グループの詳細を出力
                    if LOG_DET_CHAIN_DETAIL {
                        vlog!("[検出器/DEBUG] 1連鎖目に{}個のグループ検出:", groups.len());
                        for (i, g) in groups.iter().enumerate() {
                            let mut colors = std::collections::HashSet::new();
                            let mut iters = std::collections::HashSet::new();
                            for &(x, y) in g {
                                if let Some(cell) = self.field[y][x] {
                                    colors.insert(cell.color);
                                    iters.insert(cell.iter.0);
                                }
                            }
                            vlog!(
                                "  グループ{}: サイズ={}, 色={:?}, iter={:?}, 位置={:?}",
                                i,
                                g.len(),
                                colors,
                                iters,
                                g
                            );
                        }
                    }
                    if LOG_DET_VERBOSE {
                        vlog!(
                            "[検出器] 1連鎖目に複数グループ同時消し→不採用: groups={}",
                            groups.len()
                        );
                    }
                    self.last_reject = Some(RejectKind::FirstChainMultipleGroups);
                    return -1;
                }
                // 1連鎖目: "グループ内" に異なる iteration が混在する場合も不採用
                for g in &groups {
                    let mut ids_g: HashSet<u8> = HashSet::new();
                    for &(x, y) in g {
                        if let Some(cell) = self.field[y][x] {
                            if cell.iter.0 != 0 {
                                ids_g.insert(cell.iter.0);
                            }
                        }
                    }
                    if ids_g.len() > 1 {
                        if LOG_DET_VERBOSE {
                            vlog!("[検出器] 同一グループ内で異iteration混在→不採用");
                        }
                        self.last_reject = Some(RejectKind::MixedIterationInGroup);
                        return -1;
                    }
                }
            }

            // 連鎖履歴に記録（元の位置も含む）
            let mut original_groups = Vec::new();
            for group in &groups {
                let mut orig_group = Vec::new();
                for &(x, y) in group {
                    if let Some(cell) = self.field[y][x] {
                        if let Some(orig_pos) = cell.original_pos {
                            orig_group.push(orig_pos);
                        } else {
                            // original_posがない場合は現在位置を使用
                            orig_group.push((x, y));
                        }
                    }
                }
                original_groups.push(orig_group);
            }

            self.chain_history.push(ChainStep {
                chain_num: (chain_count + 1) as usize,
                groups: groups.clone(),
                original_groups,
            });

            // グループを消去して重力適用
            remove_groups(&mut self.field, &groups);
            chain_count += 1;
            if LOG_DET_CHAIN_DETAIL {
                vlog!(
                    "[検出器/DEBUG] {}連鎖目: グループ消去後、重力適用前",
                    chain_count
                );
            }
            apply_gravity(&mut self.field);

            // 重力適用後の盤面を出力
            if LOG_DET_CHAIN_DETAIL {
                vlog!("[検出器/DEBUG] {}連鎖目: 重力適用後の盤面:", chain_count);
                for y in (0..H).rev() {
                    let mut line = format!("  y={:2}: ", y);
                    for x in 0..W {
                        if let Some(c) = self.field[y][x] {
                            line.push_str(&format!("{}(i{}) ", color_name(c.color), c.iter.0));
                        } else {
                            line.push_str("  .   ");
                        }
                    }
                    vlog!("{}", line);
                }
            }
        }
        chain_count
    }

    /// 禁止ルール（1連鎖目の同時消し・最新追加関与など）を無視し、
    /// 物理的に連鎖を最後まで適用して連鎖数を返す。
    /// leftover計算（A探索の基準盤面生成）専用の安全なシミュレーション。
    pub fn simulate_chain_physical(&mut self) -> i32 {
        let mut chain_count: i32 = 0;
        loop {
            let groups = find_groups_4plus(&self.field);
            if groups.is_empty() {
                break;
            }
            chain_count += 1;
            remove_groups(&mut self.field, &groups);
            apply_gravity(&mut self.field);
        }
        chain_count
    }

    /// 直前の simulate_chain での棄却理由（あれば）を取得してクリア
    pub fn take_last_reject(&mut self) -> Option<RejectKind> {
        self.last_reject.take()
    }

    /// Python: check_and_place_puyos_for_color
    /// 追加配置Bを必要数だけ探索し、最良候補を逐次適用
    pub fn check_and_place_puyos_for_color(
        &mut self,
        x: usize,
        y: usize,
        base_color: u8,
        blocked_columns: Option<&HashSet<usize>>,
        _previous_additions: Option<&HashMap<(usize, usize), CellData>>,
        iteration: u8,
    ) -> bool {
        // vlog!(
        //     "[検出器] 追加配置探索開始: iter={}, 基点=({},{}), 色={}, blocked={}",
        //     iteration,
        //     x,
        //     y,
        //     color_name(base_color),
        //     blocked_columns.map(|b| b.len()).unwrap_or(0)
        // );
        if x >= W || y >= H || self.field[y][x].is_some() {
            return false;
        }

        // 隣接同色グループ収集（右・左・上下）
        let mut adj_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
        for (dx, dy) in dirs {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if !in_range(nx, ny) {
                continue;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            if let Some(c) = self.field[nyu][nxu] {
                if c.color == base_color {
                    let g = self.get_connected_cells(nxu, nyu);
                    // 既存と重複しないグループのみ追加
                    if !adj_groups.iter().any(|ex| intersects(&g, ex)) {
                        adj_groups.push(g);
                    }
                }
            }
        }
        if adj_groups.is_empty() {
            vlog!("[検出器] 隣接する同色ぷよがないため中断");
            return false;
        }

        // Aグループと許可列
        use std::collections::BTreeSet;
        let mut puyo_a: BTreeSet<(usize, usize)> = BTreeSet::new();
        for g in &adj_groups {
            for &p in g {
                puyo_a.insert(p);
            }
        }
        let mut adjacency_base: HashSet<(usize, usize)> = puyo_a.iter().copied().collect();
        let mut allowed_cols: HashSet<usize> = HashSet::new();
        for &(cx, _cy) in &adjacency_base {
            allowed_cols.insert(cx);
            if cx > 0 {
                allowed_cols.insert(cx - 1);
            }
            if cx + 1 < W {
                allowed_cols.insert(cx + 1);
            }
        }

        let total_adjacent: usize = adj_groups.iter().map(|g| g.len()).sum();
        let effective_adjacent = if total_adjacent < 4 {
            total_adjacent
        } else {
            3
        };
        let needed = 4usize.saturating_sub(effective_adjacent);

        let mut placed_positions: Vec<(usize, usize)> = Vec::new();
        vlog!(
            "[検出器] A集合|隣接点数={} / 許可列={:?} / 必要追加={}",
            adjacency_base.len(),
            {
                let mut v: Vec<_> = allowed_cols.iter().copied().collect();
                v.sort_unstable();
                v
            },
            needed
        );
        for _i in 0..needed {
            let mut best_chain: i32 = -1;
            let mut best_field: Option<Board> = None;
            let mut best_pos: Option<(usize, usize)> = None;

            let mut cols: Vec<usize> = allowed_cols.iter().copied().collect();
            cols.sort_unstable();
            for col in cols {
                if let Some(bl) = blocked_columns {
                    if bl.contains(&col) {
                        continue;
                    }
                }
                let Some(cand_y) = find_bottom_empty(&self.field, col) else {
                    continue;
                };

                let mut cand = self.field.clone();
                cand[cand_y][col] = Some(CellData {
                    color: base_color,
                    iter: IterId(iteration),
                    original_pos: None,
                });
                let mut tmp = cand.clone();
                apply_gravity(&mut tmp);

                // 最終着地点を推定（新規セルかつ元が空）
                // y=0 が底なので、底から上へ走査して最下段の新規位置を特定
                let mut final_y: Option<usize> = None;
                for yy in 0..H {
                    if tmp[yy][col].is_some() && self.field[yy][col].is_none() {
                        final_y = Some(yy);
                        break;
                    }
                }
                let Some(final_y) = final_y else {
                    vlog!("[検出器] 列{}: 新規落下位置を特定できずスキップ", col);
                    continue;
                };

                // Aグループと隣接か
                if !is_adjacent_to(&adjacency_base, col, final_y) {
                    vlog!(
                        "[検出器] 列{}: Aグループに隣接しないため却下 (y={})",
                        col,
                        final_y
                    );
                    continue;
                }

                // 連鎖評価
                let mut det = Detector::new(tmp.clone());
                let chain = det.simulate_chain();
                vlog!("[検出器] 列{}: 候補(y={}) → chain={}", col, final_y, chain);
                if chain > best_chain {
                    best_chain = chain;
                    best_field = Some(tmp);
                    best_pos = Some((col, final_y));
                }
            }

            let Some(bf) = best_field else {
                vlog!("[検出器] 追加候補が見つからず中断");
                return false;
            };
            let (col, fy) = best_pos.unwrap();
            self.field = bf;
            adjacency_base.insert((col, fy));
            allowed_cols.insert(col);
            if col > 0 {
                allowed_cols.insert(col - 1);
            }
            if col + 1 < W {
                allowed_cols.insert(col + 1);
            }
            placed_positions.push((col, fy));
            vlog!("[検出器] 追加確定: 列{} y={}", col, fy);
        }
        vlog!("[検出器] 追加完了: {}個", placed_positions.len());
        true
    }

    /// 総当たりで「必要追加数」を複数列に分配して探索する版
    /// 例: 必要3個 → [colAに1, colBに2] などを再帰で列挙
    pub fn check_and_place_puyos_for_color_bruteforce(
        &mut self,
        x: usize,
        y: usize,
        base_color: u8,
        blocked_columns: Option<&HashSet<usize>>,
        _previous_additions: Option<&HashMap<(usize, usize), CellData>>,
        iteration: u8,
    ) -> bool {
        // vlog!(
        //     "[検出器/BF] 追加配置(総当たり)開始: iter={}, 基点=({},{}), 色={}, blocked={}",
        //     iteration, x, y, color_name(base_color), blocked_columns.map(|b| b.len()).unwrap_or(0)
        // );
        if x >= W || y >= H || self.field[y][x].is_some() {
            return false;
        }

        // 隣接同色グループ収集（右・左・上下）
        use std::collections::BTreeSet;
        let mut adj_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
        for (dx, dy) in dirs {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if !in_range(nx, ny) {
                continue;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            if let Some(c) = self.field[nyu][nxu] {
                if c.color == base_color {
                    let g = self.get_connected_cells(nxu, nyu);
                    if !adj_groups.iter().any(|ex| intersects(&g, ex)) {
                        adj_groups.push(g);
                    }
                }
            }
        }
        if adj_groups.is_empty() {
            vlog!("[検出器/BF] 隣接する同色ぷよがないため中断");
            return false;
        }

        let mut puyo_a: BTreeSet<(usize, usize)> = BTreeSet::new();
        for g in &adj_groups {
            for &p in g {
                puyo_a.insert(p);
            }
        }
        let mut adjacency_base: HashSet<(usize, usize)> = puyo_a.iter().copied().collect();
        let mut allowed_cols: HashSet<usize> = HashSet::new();
        for &(cx, _cy) in &adjacency_base {
            allowed_cols.insert(cx);
            if cx > 0 {
                allowed_cols.insert(cx - 1);
            }
            if cx + 1 < W {
                allowed_cols.insert(cx + 1);
            }
        }

        let total_adjacent: usize = adj_groups.iter().map(|g| g.len()).sum();
        let effective_adjacent = if total_adjacent < 4 {
            total_adjacent
        } else {
            3
        };
        let needed = 4usize.saturating_sub(effective_adjacent);
        vlog!(
            "        → A集合|隣接点数={} / 許可列={:?} / 必要追加={}",
            adjacency_base.len(),
            {
                let mut v: Vec<_> = allowed_cols.iter().copied().collect();
                v.sort_unstable();
                v
            },
            needed
        );
        if needed == 0 {
            return true;
        }

        // 再帰で総当たり探索
        fn dfs(
            cur_field: &Board,
            adjacency_base: &HashSet<(usize, usize)>,
            allowed_cols: &HashSet<usize>,
            needed: usize,
            base_color: u8,
            iteration: u8,
            blocked_columns: Option<&HashSet<usize>>,
            depth: usize,
        ) -> Option<(i32, Board, Vec<(usize, usize)>)> {
            let indent = "  ".repeat(depth);
            if depth == 0 {
                let mut cols_dbg: Vec<_> = allowed_cols.iter().copied().collect();
                cols_dbg.sort_unstable();
                vlog!(
                    "[検出器/BF-DFS] depth={} 開始: needed={} / 許可列={:?} / A|B点数={}",
                    depth,
                    needed,
                    cols_dbg,
                    adjacency_base.len()
                );
            }
            if needed == 0 {
                // スコアは連鎖後で評価するが、返す盤面は「連鎖前（追加+重力適用済み）」を保持する
                let mut det = Detector::new(cur_field.clone());
                let chain = det.simulate_chain();
                vlog!("[検出器/BF-DFS]{} 葉: chain={}", indent, chain);
                return Some((chain, cur_field.clone(), Vec::new()));
            }

            let mut best: Option<(i32, Board, Vec<(usize, usize)>)> = None;
            let mut cols: Vec<usize> = allowed_cols.iter().copied().collect();
            cols.sort_unstable();
            for col in cols {
                if let Some(bl) = blocked_columns {
                    if bl.contains(&col) {
                        vlog!(
                            "[検出器/BF-DFS]{} 列{}: ブロック列のためスキップ",
                            indent,
                            col
                        );
                        continue;
                    }
                }
                let Some(cand_y) = find_bottom_empty(cur_field, col) else {
                    vlog!("[検出器/BF-DFS]{} 列{}: 空き無しでスキップ", indent, col);
                    continue;
                };
                let mut cand = cur_field.clone();
                cand[cand_y][col] = Some(CellData {
                    color: base_color,
                    iter: IterId(iteration),
                    original_pos: None,
                });
                let mut tmp = cand.clone();
                apply_gravity(&mut tmp);
                // 新規落下位置
                let mut final_y: Option<usize> = None;
                for yy in 0..H {
                    if tmp[yy][col].is_some() && cur_field[yy][col].is_none() {
                        final_y = Some(yy);
                        break;
                    }
                }
                let Some(final_y) = final_y else {
                    vlog!("[検出器/BF-DFS]{} 列{}: 落下位置特定できず", indent, col);
                    continue;
                };
                if !is_adjacent_to(adjacency_base, col, final_y) {
                    vlog!(
                        "[検出器/BF-DFS]{} 列{}: A∪Bに非隣接 (y={})",
                        indent,
                        col,
                        final_y
                    );
                    continue;
                }

                vlog!(
                    "[検出器/BF-DFS]{} 列{}: 追加 → y={} (残りneeded={})",
                    indent,
                    col,
                    final_y,
                    needed
                );

                let mut next_adj = adjacency_base.clone();
                next_adj.insert((col, final_y));
                let mut next_allowed = allowed_cols.clone();
                next_allowed.insert(col);
                if col > 0 {
                    next_allowed.insert(col - 1);
                }
                if col + 1 < W {
                    next_allowed.insert(col + 1);
                }

                if let Some((sc, bf, mut seq)) = dfs(
                    &tmp,
                    &next_adj,
                    &next_allowed,
                    needed - 1,
                    base_color,
                    iteration,
                    blocked_columns,
                    depth + 1,
                ) {
                    let mut take = (sc, bf, seq);
                    take.2.push((col, final_y));
                    if best.as_ref().map(|b| b.0).unwrap_or(i32::MIN) < sc {
                        vlog!(
                            "[検出器/BF-DFS]{} ベスト更新: chain={} at 列{} y={} (seq_len={})",
                            indent,
                            sc,
                            col,
                            final_y,
                            take.2.len()
                        );
                        best = Some(take);
                    }
                }
            }
            best
        }

        if let Some((best_chain, best_field, seq)) = dfs(
            &self.field,
            &adjacency_base,
            &allowed_cols,
            needed,
            base_color,
            iteration,
            blocked_columns,
            0,
        ) {
            if best_chain < 0 {
                vlog!(
                    "[検出器/BF] 最良候補は異iteration同時消しのため不採用 (chain={})",
                    best_chain
                );
                return false;
            }
            vlog!(
                "[検出器/BF] 追加確定(総当たり): chain={} / 配置={:?}",
                best_chain,
                seq
            );
            self.field = best_field;
            return true;
        }
        vlog!("[検出器/BF] 候補なし");
        false
    }

    /// 総当たりで「必要追加数」を複数列に分配し、到達可能な盤面をできるだけ多く収集する版
    /// 返すBoardは「連鎖前（追加＋重力適用済み）」の盤面。スコア（chain）は別途 simulate_chain の結果。
    pub fn enumerate_additions_for_color_bruteforce(
        &self,
        x: usize,
        y: usize,
        base_color: u8,
        blocked_columns: Option<&HashSet<usize>>,
        _previous_additions: Option<&HashMap<(usize, usize), CellData>>,
        iteration: u8,
        max_keep: usize,
    ) -> Vec<(i32, Board, Vec<(usize, usize)>)> {
        vlog!(
            "      [検出器] 基点=({},{}), 色={}",
            x,
            y,
            color_name(base_color)
        );
        if x >= W || y >= H || self.field[y][x].is_some() {
            return Vec::new();
        }

        // 隣接同色グループ収集（右・左・上下）
        use std::collections::BTreeSet;
        let mut adj_groups: Vec<Vec<(usize, usize)>> = Vec::new();
        let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
        for (dx, dy) in dirs {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if !in_range(nx, ny) {
                continue;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            if let Some(c) = self.field[nyu][nxu] {
                if c.color == base_color {
                    let g = self.get_connected_cells(nxu, nyu);
                    if !adj_groups.iter().any(|ex| intersects(&g, ex)) {
                        adj_groups.push(g);
                    }
                }
            }
        }
        if adj_groups.is_empty() {
            vlog!("        → 隣接する同色ぷよがないため中断");
            return Vec::new();
        }

        let mut puyo_a: BTreeSet<(usize, usize)> = BTreeSet::new();
        for g in &adj_groups {
            for &p in g {
                puyo_a.insert(p);
            }
        }
        let mut adjacency_base: HashSet<(usize, usize)> = puyo_a.iter().copied().collect();

        // 許可列: 基点の列とその左右、および隣接グループの列とその左右
        let mut allowed_cols: HashSet<usize> = HashSet::new();
        // 基点の列とその左右を追加
        allowed_cols.insert(x);
        if x > 0 {
            allowed_cols.insert(x - 1);
        }
        if x + 1 < W {
            allowed_cols.insert(x + 1);
        }
        // 隣接グループの各セルの列とその左右を追加
        for &(cx, _cy) in &adjacency_base {
            allowed_cols.insert(cx);
            if cx > 0 {
                allowed_cols.insert(cx - 1);
            }
            if cx + 1 < W {
                allowed_cols.insert(cx + 1);
            }
        }

        let total_adjacent: usize = adj_groups.iter().map(|g| g.len()).sum();
        let effective_adjacent = if total_adjacent < 4 {
            total_adjacent
        } else {
            3
        };
        let needed = 4usize.saturating_sub(effective_adjacent);
        vlog!(
            "        → A集合|隣接点数={} / 許可列={:?} / 必要追加={}",
            adjacency_base.len(),
            {
                let mut v: Vec<_> = allowed_cols.iter().copied().collect();
                v.sort_unstable();
                v
            },
            needed
        );
        if needed == 0 {
            // 追加不要。評価だけした1件を返す（現状の方針に合わせて chain=simulate_chain）
            let mut det = Detector::new(self.field.clone());
            let chain = det.simulate_chain();
            return vec![(chain, self.field.clone(), Vec::new())];
        }

        // 並列化：needed >= 2 の場合、最初の1個目の配置候補で分岐（重複排除付き）
        if needed >= 2 {
            let mut cols: Vec<usize> = allowed_cols.iter().copied().collect();
            cols.sort_unstable();

            // グローバル重複排除用
            let global_seen = Arc::new(Mutex::new(HashSet::new()));

            // 各列での最初の配置を並列処理
            let all_results: Vec<Vec<(i32, Board, Vec<(usize, usize)>)>> = cols
                .par_iter()
                .filter_map(|&col| {
                    let global_seen = Arc::clone(&global_seen);
                    if let Some(bl) = blocked_columns {
                        if bl.contains(&col) {
                            return None;
                        }
                    }
                    let cand_y = find_bottom_empty(&self.field, col)?;

                    let mut cand = self.field.clone();
                    cand[cand_y][col] = Some(CellData {
                        color: base_color,
                        iter: IterId(iteration),
                        original_pos: None,
                    });
                    let mut tmp = cand.clone();
                    apply_gravity(&mut tmp);

                    // 新規落下位置
                    let mut final_y: Option<usize> = None;
                    for yy in 0..H {
                        if tmp[yy][col].is_some() && self.field[yy][col].is_none() {
                            final_y = Some(yy);
                            break;
                        }
                    }
                    let final_y = final_y?;

                    if !is_adjacent_to(&adjacency_base, col, final_y) {
                        return None;
                    }

                    // この配置から残りを探索
                    let mut next_adj = adjacency_base.clone();
                    next_adj.insert((col, final_y));
                    let mut next_allowed = allowed_cols.clone();
                    next_allowed.insert(col);
                    if col > 0 {
                        next_allowed.insert(col - 1);
                    }
                    if col + 1 < W {
                        next_allowed.insert(col + 1);
                    }

                    let mut local_results = Vec::new();
                    let mut path = vec![(col, final_y)];
                    let mut local_seen = HashSet::new();

                    dfs_collect_serial_deduplicated(
                        &tmp,
                        &next_adj,
                        &next_allowed,
                        needed - 1,
                        base_color,
                        iteration,
                        blocked_columns,
                        0,
                        &mut path,
                        &mut local_results,
                        &mut local_seen,
                    );

                    // グローバル重複チェック
                    let mut global_lock = global_seen.lock().unwrap();
                    let filtered: Vec<_> = local_results
                        .into_iter()
                        .filter(|(_, board, _)| {
                            let hash = compute_board_hash_simple(board);
                            global_lock.insert(hash)
                        })
                        .collect();
                    drop(global_lock);

                    if filtered.is_empty() {
                        None
                    } else {
                        Some(filtered)
                    }
                })
                .collect();

            // 結果を統合
            let mut results: Vec<(i32, Board, Vec<(usize, usize)>)> =
                all_results.into_iter().flatten().collect();

            // スコア降順で並べ、max_keepにトリミング
            results.sort_by(|a, b| b.0.cmp(&a.0));
            if max_keep > 0 && results.len() > max_keep {
                results.truncate(max_keep);
            }
            let dedup_count = Arc::try_unwrap(global_seen)
                .unwrap()
                .into_inner()
                .unwrap()
                .len();
            vlog!(
                "        → 並列列挙完了: {}件, 重複排除={}",
                results.len(),
                dedup_count
            );
            return results;
        }

        // needed == 1 の場合は単純にループ
        let mut results: Vec<(i32, Board, Vec<(usize, usize)>)> = Vec::new();
        let mut cols: Vec<usize> = allowed_cols.iter().copied().collect();
        cols.sort_unstable();

        for col in cols {
            if let Some(bl) = blocked_columns {
                if bl.contains(&col) {
                    continue;
                }
            }
            let Some(cand_y) = find_bottom_empty(&self.field, col) else {
                continue;
            };
            let mut cand = self.field.clone();
            cand[cand_y][col] = Some(CellData {
                color: base_color,
                iter: IterId(iteration),
                original_pos: None,
            });
            let mut tmp = cand.clone();
            apply_gravity(&mut tmp);

            let mut final_y: Option<usize> = None;
            for yy in 0..H {
                if tmp[yy][col].is_some() && self.field[yy][col].is_none() {
                    final_y = Some(yy);
                    break;
                }
            }
            let Some(final_y) = final_y else {
                continue;
            };
            if !is_adjacent_to(&adjacency_base, col, final_y) {
                continue;
            }

            let mut det = Detector::new(tmp.clone());
            let chain = det.simulate_chain();
            results.push((chain, tmp, vec![(col, final_y)]));
        }

        // スコア降順で並べ、max_keepにトリミング
        results.sort_by(|a, b| b.0.cmp(&a.0));
        if max_keep > 0 && results.len() > max_keep {
            results.truncate(max_keep);
        }
        vlog!("        → 列挙完了: {}件", results.len());
        results
    }
}

// シリアル版のDFS収集（重複排除付き）
fn dfs_collect_serial_deduplicated(
    cur_field: &Board,
    adjacency_base: &HashSet<(usize, usize)>,
    allowed_cols: &HashSet<usize>,
    needed: usize,
    base_color: u8,
    iteration: u8,
    blocked_columns: Option<&HashSet<usize>>,
    _depth: usize,
    path: &mut Vec<(usize, usize)>,
    out: &mut Vec<(i32, Board, Vec<(usize, usize)>)>,
    seen: &mut HashSet<u64>,
) {
    if needed == 0 {
        let hash = compute_board_hash_simple(cur_field);
        if seen.insert(hash) {
            let mut det = Detector::new(cur_field.clone());
            let chain = det.simulate_chain();
            out.push((chain, cur_field.clone(), path.clone()));
        }
        return;
    }
    let mut cols: Vec<usize> = allowed_cols.iter().copied().collect();
    cols.sort_unstable();
    for col in cols {
        if let Some(bl) = blocked_columns {
            if bl.contains(&col) {
                continue;
            }
        }
        let Some(cand_y) = find_bottom_empty(cur_field, col) else {
            continue;
        };
        let mut cand = cur_field.clone();
        cand[cand_y][col] = Some(CellData {
            color: base_color,
            iter: IterId(iteration),
            original_pos: None,
        });
        let mut tmp = cand.clone();
        apply_gravity(&mut tmp);
        // 新規落下位置
        let mut final_y: Option<usize> = None;
        for yy in 0..H {
            if tmp[yy][col].is_some() && cur_field[yy][col].is_none() {
                final_y = Some(yy);
                break;
            }
        }
        let Some(final_y) = final_y else {
            continue;
        };
        if !is_adjacent_to(adjacency_base, col, final_y) {
            continue;
        }

        let mut next_adj = adjacency_base.clone();
        next_adj.insert((col, final_y));
        let mut next_allowed = allowed_cols.clone();
        next_allowed.insert(col);
        if col > 0 {
            next_allowed.insert(col - 1);
        }
        if col + 1 < W {
            next_allowed.insert(col + 1);
        }

        path.push((col, final_y));
        dfs_collect_serial_deduplicated(
            &tmp,
            &next_adj,
            &next_allowed,
            needed - 1,
            base_color,
            iteration,
            blocked_columns,
            0,
            path,
            out,
            seen,
        );
        path.pop();
    }
}

/// 盤面の正規化ハッシュ計算（ミラー対称を考慮）
fn compute_board_hash_simple(board: &Board) -> u64 {
    // Boardをcols形式に変換してから正規化ハッシュを適用
    let cols = board_to_cols(board);
    let (hash, _mirror) = canonical_hash64_fast(&cols);
    hash
}

fn intersects(a: &Vec<(usize, usize)>, b: &Vec<(usize, usize)>) -> bool {
    let sa: HashSet<(usize, usize)> = a.iter().copied().collect();
    b.iter().any(|p| sa.contains(p))
}

fn is_adjacent_to(base: &HashSet<(usize, usize)>, x: usize, y: usize) -> bool {
    const DIRS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (dx, dy) in DIRS {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || ny < 0 {
            continue;
        }
        let nxu = nx as usize;
        let nyu = ny as usize;
        if nxu < W && nyu < H && base.contains(&(nxu, nyu)) {
            return true;
        }
    }
    false
}
