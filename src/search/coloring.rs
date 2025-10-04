// 彩色と列生成ロジック

use std::collections::HashMap;
use std::time::{Duration, Instant};
use num_bigint::BigUint;
use num_traits::{One, Zero};
use crate::model::TCell;
use crate::constants::{W, H};

/// 抽象情報（ラベルと隣接関係）
pub struct AbstractInfo {
    pub labels: Vec<char>,
    pub adj: Vec<Vec<usize>>,
}

/// 入力から抽象情報を構築
pub fn build_abstract_info(board: &[char]) -> AbstractInfo {
    let mut labels = Vec::new();
    for &v in board {
        if ('A'..='M').contains(&v) && !labels.contains(&v) {
            labels.push(v);
        }
    }
    let mut label_idx = HashMap::new();
    for (i, &c) in labels.iter().enumerate() {
        label_idx.insert(c, i);
    }
    let n = labels.len();
    let mut adj = vec![Vec::<usize>::new(); n];
    let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
    for x in 0..W {
        for y in 0..H {
            let v = board[y * W + x];
            if !('A'..='M').contains(&v) {
                continue;
            }
            let id = label_idx[&v];
            for (dx, dy) in dirs {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize {
                    continue;
                }
                let w = board[(ny as usize) * W + (nx as usize)];
                if ('A'..='M').contains(&w) && w != v {
                    let nb = label_idx[&w];
                    if !adj[id].contains(&nb) {
                        adj[id].push(nb);
                    }
                }
            }
        }
    }
    AbstractInfo { labels, adj }
}

/// DSATUR風彩色列挙
pub fn enumerate_colorings_fast(info: &AbstractInfo) -> Vec<Vec<u8>> {
    let n = info.labels.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    // 隣接を bitset に
    let mut adj = vec![0u16; n];
    for (v, adjv) in adj.iter_mut().enumerate() {
        let mut m = 0u16;
        for &u in &info.adj[v] {
            m |= 1u16 << u;
        }
        *adjv = m;
    }

    let mut color = vec![4u8; n];
    let mut used_mask = vec![0u8; n];

    let mut out = Vec::new();
    fn dfs(
        vleft: usize,
        total_n: usize,
        adj: &[u16],
        color: &mut [u8],
        used_mask: &mut [u8],
        out: &mut Vec<Vec<u8>>,
        max_used: u8,
    ) {
        if vleft == 0 {
            out.push(color.to_vec());
            return;
        }

        let mut pick = None;
        let mut best_sat = -1i32;
        let mut best_deg = -1i32;
        for v in 0..color.len() {
            if color[v] != 4 {
                continue;
            }
            let sat = used_mask[v].count_ones() as i32;
            let deg = adj[v].count_ones() as i32;
            if sat > best_sat || (sat == best_sat && deg > best_deg) {
                best_sat = sat;
                best_deg = deg;
                pick = Some(v);
            }
        }
        let v = pick.unwrap();

        let forbid = used_mask[v];
        let mut new_color_limit = (max_used + 1).min(3);
        if vleft == total_n {
            new_color_limit = 0;
        }
        for c in 0u8..=new_color_limit {
            if ((forbid >> c) & 1) != 0 {
                continue;
            }
            color[v] = c;

            let mut touched = 0u16;
            let mut nb = adj[v];
            while nb != 0 {
                let u = nb.trailing_zeros() as usize;
                nb &= nb - 1;
                if color[u] == 4 {
                    used_mask[u] |= 1u8 << c;
                    touched |= 1u16 << u;
                }
            }
            let next_max_used = if c > max_used { c } else { max_used };
            dfs(vleft - 1, total_n, adj, color, used_mask, out, next_max_used);

            color[v] = 4;
            let mut t = touched;
            while t != 0 {
                let u = t.trailing_zeros() as usize;
                t &= t - 1;
                used_mask[u] &= !(1u8 << c);
            }
        }
    }
    dfs(n, n, &adj, &mut color, &mut used_mask, &mut out, 0);
    out
}

/// 彩色をテンプレートに適用
pub fn apply_coloring_to_template(base: &[char], map: &HashMap<char, u8>) -> Vec<TCell> {
    base.iter()
        .map(|&v| {
            if ('A'..='M').contains(&v) {
                TCell::Fixed(map[&v])
            } else if v == 'N' {
                TCell::Any
            } else if v == 'X' {
                TCell::Any4
            } else if v == '.' {
                TCell::Blank
            } else if ('0'..='3').contains(&v) {
                TCell::Fixed(v as u8 - b'0')
            } else {
                TCell::Blank
            }
        })
        .collect()
}

/// 列DP（通り数）
pub fn count_column_candidates_dp(col: &[TCell]) -> BigUint {
    let mut dp0 = BigUint::one();
    let mut dp1 = BigUint::zero();
    for &cell in col.iter().take(H) {
        let mut ndp0 = BigUint::zero();
        let mut ndp1 = BigUint::zero();
        match cell {
            TCell::Blank => {
                ndp1 += &dp0;
                ndp1 += &dp1;
            }
            TCell::Any4 => {
                if !dp0.is_zero() {
                    ndp0 += dp0.clone() * BigUint::from(4u32);
                }
            }
            TCell::Any => {
                ndp1 += &dp0;
                ndp1 += &dp1;
                if !dp0.is_zero() {
                    ndp0 += dp0 * BigUint::from(4u32);
                }
            }
            TCell::Fixed(_) => {
                ndp0 += &dp0;
            }
        }
        dp0 = ndp0;
        dp1 = ndp1;
    }
    dp0 + dp1
}

/// 列ストリーミング列挙
pub fn stream_column_candidates<F: FnMut([u16; 4])>(col: &[TCell], mut yield_masks: F) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        yield_masks: &mut F,
    ) {
        if y >= H {
            yield_masks(*masks);
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    rec(0, false, col, &mut masks, &mut yield_masks);
}

/// 列ストリーミング列挙（計測版）
pub fn stream_column_candidates_timed<F: FnMut([u16; 4])>(
    col: &[TCell],
    enum_time: &mut Duration,
    mut yield_masks: F,
) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        enum_time: &mut Duration,
        last_start: &mut Instant,
        yield_masks: &mut F,
    ) {
        if y >= H {
            *enum_time += last_start.elapsed();
            yield_masks(*masks);
            *last_start = Instant::now();
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, enum_time, last_start, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, enum_time, last_start, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    let mut last_start = Instant::now();
    rec(
        0,
        false,
        col,
        &mut masks,
        enum_time,
        &mut last_start,
        &mut yield_masks,
    );
    *enum_time += last_start.elapsed();
}

/// 列候補生成のハイブリッド
pub enum ColGen {
    Pre(Vec<[u16; 4]>),
    Stream(Vec<TCell>),
}

pub fn build_colgen(col: &[TCell], cnt: &BigUint) -> ColGen {
    if cnt.bits() <= 11 {
        let mut v = Vec::new();
        stream_column_candidates(col, |m| v.push(m));
        ColGen::Pre(v)
    } else {
        ColGen::Stream(col.to_vec())
    }
}

/// 列の最大埋め込み数を計算
#[inline(always)]
pub fn compute_max_fill(col: &[TCell]) -> u8 {
    let mut cnt: u8 = 0;
    for &cell in col.iter().take(H) {
        match cell {
            TCell::Blank => break,
            _ => cnt += 1,
        }
    }
    cnt
}
