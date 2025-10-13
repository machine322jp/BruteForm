use crate::constants::{W, H};

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

pub fn apply_gravity(board: &mut Board) {
    // y=0 が底。底から上へ集め、底から上へ詰める
    for x in 0..W {
        let mut stack: Vec<CellData> = Vec::with_capacity(H);
        for y in 0..H { // bottom -> top
            if let Some(c) = board[y][x] { stack.push(c); }
        }
        let mut it = stack.into_iter();
        for y in 0..H { // bottom -> top
            board[y][x] = it.next();
        }
    }
}

pub fn find_bottom_empty(board: &Board, col: usize) -> Option<usize> {
    // もっとも低い空き（底から探索）
    for y in 0..H {
        if board[y][col].is_none() { return Some(y); }
    }
    None
}

pub fn get_connected_cells(board: &Board, sx: usize, sy: usize) -> Vec<(usize, usize)> {
    use std::collections::VecDeque;
    let base = match board[sy][sx] { Some(c) => c.color, None => return vec![] };
    let mut vis = vec![vec![false; W]; H];
    let mut q = VecDeque::new();
    let mut out = Vec::new();
    vis[sy][sx] = true;
    q.push_back((sx, sy));
    out.push((sx, sy));
    const DIRS: [(isize, isize); 4] = [(1,0),(-1,0),(0,1),(0,-1)];
    while let Some((x,y)) = q.pop_front() {
        for (dx,dy) in DIRS {
            let nx = x as isize + dx; let ny = y as isize + dy;
            if !in_range(nx, ny) { continue; }
            let nxu = nx as usize; let nyu = ny as usize;
            if vis[nyu][nxu] { continue; }
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

pub fn find_groups_4plus(board: &Board) -> Vec<Vec<(usize,usize)>> {
    let mut vis = vec![vec![false; W]; H];
    let mut found = Vec::new();
    for y in 0..H {
        for x in 0..W {
            if board[y][x].is_none() || vis[y][x] { continue; }
            let g = get_connected_cells(board, x, y);
            for &(gx, gy) in &g { vis[gy][gx] = true; }
            if g.len() >= 4 { found.push(g); }
        }
    }
    found
}

pub fn remove_groups(board: &mut Board, groups: &[Vec<(usize,usize)>]) {
    for grp in groups {
        for &(x,y) in grp { board[y][x] = None; }
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
