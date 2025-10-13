use std::collections::{HashMap, HashSet};

use crate::constants::{W, H};
use super::grid::{Board, find_bottom_empty, in_range, CellData};
use crate::vlog;

const LOG_GEN_VERBOSE: bool = true;
use super::detector::Detector;

pub struct Generator {
    pub original_field: Board,
    pub allow_full_column: bool,
}

impl Generator {
    pub fn new(field: Board) -> Self {
        Self { original_field: field, allow_full_column: false }
    }
    pub fn with_full_column(mut self, allow: bool) -> Self { self.allow_full_column = allow; self }

    pub fn find_best_arrangement(
        &self,
        target_color: Option<u8>,
        blocked_columns: Option<&HashSet<usize>>,
        preferred_columns: Option<&HashSet<usize>>,
        previous_additions: Option<&HashMap<(usize,usize), CellData>>,
        iteration: u8,
    ) -> Vec<(i32, Board, (usize, usize))> {
        // 最終返却用（chain, board, seed座標）
        let mut candidates: Vec<(i32, Board, (usize, usize))> = Vec::new();
        // ランキング用に追加数も保持（chain, adds, board, seed座標）
        let mut ranked: Vec<(i32, usize, Board, (usize, usize))> = Vec::new();
        let mut cand_cols: Vec<usize> = Vec::new();
        if let Some(pref) = preferred_columns { let mut v: Vec<_> = pref.iter().copied().collect(); v.sort_unstable(); cand_cols.extend(v); }
        for x in 0..W {
            if let Some(blocked) = blocked_columns { if blocked.contains(&x) { continue; } }
            if let Some(pref) = preferred_columns { if pref.contains(&x) { continue; } }
            cand_cols.push(x);
        }
        if LOG_GEN_VERBOSE {
            vlog!(
                "  [生成器] iter={} 候補列={:?}（blocked={}） target_color={:?}",
                iteration,
                cand_cols,
                blocked_columns.map(|b| b.len()).unwrap_or(0),
                target_color
            );
        }

        for &x in &cand_cols {
            let bottom_y = find_bottom_empty(&self.original_field, x).or_else(|| if self.allow_full_column { Some(0) } else { None });
            let Some(bottom_y) = bottom_y else { continue; };
            if LOG_GEN_VERBOSE { vlog!("    [生成器] 列{}: 最下段空きy={}", x, bottom_y); }

            // 試行色: target_color が指定されていればそれに限定。未指定(None)時のみ近傍隣接色を採用
            let colors_to_try: Vec<u8> = if let Some(tc) = target_color {
                vec![tc]
            } else {
                let mut set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
                for c in self.get_neighbor_colors(x, bottom_y) { set.insert(c); }
                set.into_iter().collect()
            };
            if colors_to_try.is_empty() { continue; }
            if LOG_GEN_VERBOSE { vlog!("    [生成器] 列{}: 試行色={:?}", x, colors_to_try); }

            for color in colors_to_try {
                // 列挙版: 多通りの追加配置を候補として収集
                let det = Detector::new(self.original_field.clone());
                let max_keep = 64usize; // 返却上限（必要に応じて拡張可）
                let many = det.enumerate_additions_for_color_bruteforce(
                    x, bottom_y, color,
                    blocked_columns,
                    previous_additions,
                    iteration,
                    max_keep,
                );
                if many.is_empty() {
                    if LOG_GEN_VERBOSE { vlog!("      [生成器] 列{}: 色{} 配置不可（スキップ）", x, color); }
                    continue;
                }
                if LOG_GEN_VERBOSE { vlog!("      [生成器] 列{}: 色{} → {}通り", x, color, many.len()); }
                for (chain, board_pre_chain, seq) in many {
                    let adds = seq.len();
                    ranked.push((chain, adds, board_pre_chain, (x, bottom_y)));
                }
            }
        }

        // chain 降順, adds 昇順で並べ替え
        ranked.sort_by(|a,b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // 返却型にマッピング
        candidates = ranked.into_iter().map(|(ch, _adds, bd, xy)| (ch, bd, xy)).collect();

        // if let Some((best_chain, _f, (bx, by))) = candidates.first() {
        //     vlog!("  [生成器] 最良候補: 列{} y={} chain={}（計{}件）", bx, by, best_chain, candidates.len());
        // } else if LOG_GEN_VERBOSE {
        //     vlog!("  [生成器] 候補0件");
        // }
        candidates
    }

    fn get_neighbor_colors(&self, x: usize, y: usize) -> Vec<u8> {
        let dirs = [(1isize,0isize),(-1,0),(0,1),(0,-1)];
        let mut s: HashSet<u8> = HashSet::new();
        for (dx,dy) in dirs {
            let nx = x as isize + dx; let ny = y as isize + dy;
            if !in_range(nx, ny) { continue; }
            let nxu = nx as usize; let nyu = ny as usize;
            if let Some(c) = self.original_field[nyu][nxu] { s.insert(c.color); }
        }
        s.into_iter().collect()
    }
}
