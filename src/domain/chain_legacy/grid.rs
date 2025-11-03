use crate::constants::{H, W};
use crate::domain::board::bitboard::{fall_cols_fast, pack_cols};

/// 追加配置の世代(イテレーション)識別子
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IterId(pub u8); // 0: 初期盤面, 1..: 追加配置

/// セル情報: 色インデックス(0..=3)とイテレーションID
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellData {
    pub color: u8,
    pub iter: IterId,
    pub original_pos: Option<(usize, usize)>, // 連鎖開始前の元の位置
}

pub type Board = Vec<Vec<Option<CellData>>>; // [H][W]

pub fn empty_board() -> Board {
    vec![vec![None; W]; H]
}

#[inline]
pub fn in_range(x: isize, y: isize) -> bool {
    x >= 0 && (x as usize) < W && y >= 0 && (y as usize) < H
}

#[inline]
pub fn get_color(cell: Option<CellData>) -> Option<u8> {
    cell.map(|c| c.color)
}

/// 重力適用（ビットボード最適化版）
pub fn apply_gravity(board: &mut Board) {
    // Boardをcols形式に変換
    let cols = board_to_cols(board);

    // BMI2最適化された落下処理を適用
    let fallen = fall_cols_fast(&cols);

    // 結果をBoard形式に戻す（元の位置情報を保持）
    apply_fallen_to_board(board, &cols, &fallen);
}

/// 落下後の盤面をBoardに反映（元の位置情報を保持）
fn apply_fallen_to_board(board: &mut Board, old_cols: &[[u16; W]; 4], new_cols: &[[u16; W]; 4]) {
    // 各列で落下前後の対応を追跡
    for x in 0..W {
        // 各色のセルを集める
        let mut cells_by_color: [Vec<CellData>; 4] =
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];

        // 落下前のセルを色ごとに集める（下から上へ）
        for y in 0..H {
            if let Some(cell) = board[y][x] {
                cells_by_color[cell.color as usize].push(cell);
            }
        }

        // 列をクリア
        for y in 0..H {
            board[y][x] = None;
        }

        // 落下後の位置にセルを配置
        for color in 0..4 {
            let mut bits = new_cols[color][x];
            let mut cell_idx = 0;
            let mut y = 0usize;

            while bits != 0 {
                if bits & 1 != 0 {
                    if cell_idx < cells_by_color[color].len() {
                        board[y][x] = Some(cells_by_color[color][cell_idx]);
                        cell_idx += 1;
                    }
                }
                bits >>= 1;
                y += 1;
            }
        }
    }
}

pub fn find_bottom_empty(board: &Board, col: usize) -> Option<usize> {
    // もっとも低い空き（底から探索）
    for y in 0..H {
        if board[y][col].is_none() {
            return Some(y);
        }
    }
    None
}

pub fn get_connected_cells(board: &Board, sx: usize, sy: usize) -> Vec<(usize, usize)> {
    use std::collections::VecDeque;
    let base = match board[sy][sx] {
        Some(c) => c.color,
        None => return vec![],
    };
    let mut vis = vec![vec![false; W]; H];
    let mut q = VecDeque::new();
    let mut out = Vec::new();
    vis[sy][sx] = true;
    q.push_back((sx, sy));
    out.push((sx, sy));
    const DIRS: [(isize, isize); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    while let Some((x, y)) = q.pop_front() {
        for (dx, dy) in DIRS {
            let nx = x as isize + dx;
            let ny = y as isize + dy;
            if !in_range(nx, ny) {
                continue;
            }
            let nxu = nx as usize;
            let nyu = ny as usize;
            if vis[nyu][nxu] {
                continue;
            }
            if let Some(c) = board[nyu][nxu] {
                if c.color == base {
                    vis[nyu][nxu] = true;
                    q.push_back((nxu, nyu));
                    out.push((nxu, nyu));
                }
            }
        }
    }
    out
}

/// 4個以上の連結グループを検出（ビットボード最適化版）
pub fn find_groups_4plus(board: &Board) -> Vec<Vec<(usize, usize)>> {
    // 高速事前チェック：ビットボードで各色の個数を確認
    let cols = board_to_cols(board);
    let bb = pack_cols(&cols);

    // どの色も4個未満なら早期リターン
    let has_potential = (bb[0].count_ones() >= 4)
        || (bb[1].count_ones() >= 4)
        || (bb[2].count_ones() >= 4)
        || (bb[3].count_ones() >= 4);

    if !has_potential {
        return Vec::new();
    }

    // 4個以上の色がある場合、詳細な連結判定を行う
    let mut vis = vec![vec![false; W]; H];
    let mut found = Vec::new();
    for y in 0..H {
        for x in 0..W {
            if board[y][x].is_none() || vis[y][x] {
                continue;
            }
            let g = get_connected_cells(board, x, y);
            for &(gx, gy) in &g {
                vis[gy][gx] = true;
            }
            if g.len() >= 4 {
                found.push(g);
            }
        }
    }
    found
}

pub fn remove_groups(board: &mut Board, groups: &[Vec<(usize, usize)>]) {
    for grp in groups {
        for &(x, y) in grp {
            board[y][x] = None;
        }
    }
}

/// 既存UIの [[u16; W]; 4] 形式から Board へ変換（iter=0を付与）
pub fn cols_to_board(cols: &[[u16; W]; 4]) -> Board {
    let mut b = empty_board();
    for c in 0..4 {
        for x in 0..W {
            let mut bits = cols[c][x];
            let mut y = 0usize;
            while bits != 0 {
                if bits & 1 != 0 {
                    b[y][x] = Some(CellData {
                        color: c as u8,
                        iter: IterId(0),
                        original_pos: Some((x, y)), // 元の位置を記録
                    });
                }
                bits >>= 1;
                y += 1;
            }
        }
    }
    // 重なりは上書きされるが、入力は色ごとに排他的である想定
    b
}

/// Board から [[u16; W]; 4] へ変換（色のみ反映）
pub fn board_to_cols(board: &Board) -> [[u16; W]; 4] {
    let mut cols = [[0u16; W]; 4];
    for y in 0..H {
        for x in 0..W {
            if let Some(cell) = board[y][x] {
                let bit = 1u16 << y;
                let c = (cell.color as usize).min(3);
                cols[c][x] |= bit;
            }
        }
    }
    cols
}
