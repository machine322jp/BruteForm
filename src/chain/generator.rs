use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

use super::grid::{board_to_cols, find_bottom_empty, in_range, Board, CellData};
use crate::constants::{H, W};
use crate::search::hash::canonical_hash64_fast;
use crate::vlog;

const LOG_GEN_VERBOSE: bool = true;
use super::detector::Detector;

pub struct Generator {
    pub original_field: Board,
    pub allow_full_column: bool,
}

impl Generator {
    pub fn new(field: Board) -> Self {
        Self {
            original_field: field,
            allow_full_column: false,
        }
    }
    pub fn with_full_column(mut self, allow: bool) -> Self {
        self.allow_full_column = allow;
        self
    }

    pub fn find_best_arrangement(
        &self,
        target_color: Option<u8>,
        blocked_columns: Option<&HashSet<usize>>,
        preferred_columns: Option<&HashSet<usize>>,
        previous_additions: Option<&HashMap<(usize, usize), CellData>>,
        iteration: u8,
    ) -> Vec<(i32, Board, (usize, usize))> {
        let mut cand_cols: Vec<usize> = Vec::new();
        if let Some(pref) = preferred_columns {
            let mut v: Vec<_> = pref.iter().copied().collect();
            v.sort_unstable();
            cand_cols.extend(v);
        }
        for x in 0..W {
            if let Some(blocked) = blocked_columns {
                if blocked.contains(&x) {
                    continue;
                }
            }
            if let Some(pref) = preferred_columns {
                if pref.contains(&x) {
                    continue;
                }
            }
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

        // 並列化：各候補列での評価を並列処理
        let all_ranked: Vec<Vec<(i32, usize, Board, (usize, usize))>> = cand_cols
            .par_iter()
            .filter_map(|&x| {
                let bottom_y = find_bottom_empty(&self.original_field, x).or_else(|| {
                    if self.allow_full_column {
                        Some(0)
                    } else {
                        None
                    }
                })?;
                if LOG_GEN_VERBOSE {
                    vlog!("    [生成器] 列{}: 最下段空きy={}", x, bottom_y);
                }

                // 試行色: target_color が指定されていればそれに限定。未指定(None)時のみ近傍隣接色を採用
                let colors_to_try: Vec<u8> = if let Some(tc) = target_color {
                    vec![tc]
                } else {
                    let mut set: std::collections::BTreeSet<u8> = std::collections::BTreeSet::new();
                    for c in self.get_neighbor_colors(x, bottom_y) {
                        set.insert(c);
                    }
                    set.into_iter().collect()
                };
                if colors_to_try.is_empty() {
                    return None;
                }
                if LOG_GEN_VERBOSE {
                    vlog!("    [生成器] 列{}: 試行色={:?}", x, colors_to_try);
                }

                let mut local_ranked: Vec<(i32, usize, Board, (usize, usize))> = Vec::new();

                for color in colors_to_try {
                    // 列挙版: 多通りの追加配置を候補として収集
                    let det = Detector::new(self.original_field.clone());
                    let max_keep = 64usize; // 返却上限（必要に応じて拡張可）
                    let many = det.enumerate_additions_for_color_bruteforce(
                        x,
                        bottom_y,
                        color,
                        blocked_columns,
                        previous_additions,
                        iteration,
                        max_keep,
                    );
                    if many.is_empty() {
                        if LOG_GEN_VERBOSE {
                            vlog!("      [生成器] 列{}: 色{} 配置不可（スキップ）", x, color);
                        }
                        continue;
                    }
                    if LOG_GEN_VERBOSE {
                        vlog!("      [生成器] 列{}: 色{} → {}通り", x, color, many.len());
                    }
                    for (chain, board_pre_chain, seq) in many {
                        let adds = seq.len();
                        local_ranked.push((chain, adds, board_pre_chain, (x, bottom_y)));
                    }
                }

                if local_ranked.is_empty() {
                    None
                } else {
                    Some(local_ranked)
                }
            })
            .collect();

        // 結果を統合（重複排除付き）
        let mut seen_hashes = std::collections::HashSet::new();
        let mut ranked: Vec<(i32, usize, Board, (usize, usize))> = Vec::new();

        for item in all_ranked.into_iter().flatten() {
            let hash = compute_board_hash_for_generator(&item.2);
            if seen_hashes.insert(hash) {
                ranked.push(item);
            }
        }

        // chain 降順, adds 昇順で並べ替え
        ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        // 返却型にマッピング
        let candidates: Vec<(i32, Board, (usize, usize))> = ranked
            .into_iter()
            .map(|(ch, _adds, bd, xy)| (ch, bd, xy))
            .collect();

        // if let Some((best_chain, _f, (bx, by))) = candidates.first() {
        //     vlog!("  [生成器] 最良候補: 列{} y={} chain={}（計{}件）", bx, by, best_chain, candidates.len());
        // } else if LOG_GEN_VERBOSE {
        //     vlog!("  [生成器] 候補0件");
        // }
        candidates
    }

    fn get_neighbor_colors(&self, x: usize, y: usize) -> Vec<u8> {
        let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
        let mut s: HashSet<u8> = HashSet::new();
        for (dx, dy) in dirs {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if !in_range(nx, ny) {
                continue;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            if let Some(c) = self.original_field[nyu][nxu] {
                s.insert(c.color);
            }
        }
        s.into_iter().collect()
    }
}

/// 盤面の正規化ハッシュ計算（Generator用、ミラー対称を考慮）
fn compute_board_hash_for_generator(board: &Board) -> u64 {
    // Boardをcols形式に変換してから正規化ハッシュを適用
    let cols = board_to_cols(board);
    let (hash, _mirror) = canonical_hash64_fast(&cols);
    hash
}
