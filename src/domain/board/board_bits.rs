// ビットボード型 - 高速な盤面表現

use crate::constants::{H, W};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// ビットボード型（6×14=84マスを u128 にパック）
pub type BB = u128;

/// 4色のビットボードで盤面を表現
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BoardBits {
    colors: [BB; 4],
}

impl BoardBits {
    /// 新しい空のビットボードを作成
    pub fn new() -> Self {
        Self { colors: [0; 4] }
    }

    /// 指定色のビットボードを取得
    pub fn get_color(&self, color: usize) -> BB {
        self.colors[color]
    }

    /// 指定色のビットボードを設定
    pub fn set_color(&mut self, color: usize, bits: BB) {
        self.colors[color] = bits;
    }

    /// すべての色が空かどうか
    pub fn is_empty(&self) -> bool {
        self.colors.iter().all(|&c| c == 0)
    }

    /// すべてのセルを含むビットボードを取得
    pub fn all_cells(&self) -> BB {
        self.colors[0] | self.colors[1] | self.colors[2] | self.colors[3]
    }

    /// セルを取得（範囲外またはブランクならNone）
    pub fn get(&self, x: usize, y: usize) -> Option<u8> {
        if x >= W || y >= H {
            return None;
        }

        let bit_pos = x * H + y;
        for color in 0..4 {
            if (self.colors[color] >> bit_pos) & 1 != 0 {
                return Some(color as u8);
            }
        }
        None
    }

    /// セルを設定
    pub fn set(&mut self, x: usize, y: usize, color: u8) -> Result<()> {
        if x >= W || y >= H {
            return Err(anyhow!("座標が範囲外: ({}, {})", x, y));
        }
        if color >= 4 {
            return Err(anyhow!("色が不正: {}", color));
        }

        let bit_pos = x * H + y;
        // 既存の色をクリア
        for c in 0..4 {
            self.colors[c] &= !(1u128 << bit_pos);
        }
        // 新しい色を設定
        self.colors[color as usize] |= 1u128 << bit_pos;

        Ok(())
    }

    /// セルをクリア（ブランクにする）
    pub fn clear(&mut self, x: usize, y: usize) -> Result<()> {
        if x >= W || y >= H {
            return Err(anyhow!("座標が範囲外: ({}, {})", x, y));
        }

        let bit_pos = x * H + y;
        for c in 0..4 {
            self.colors[c] &= !(1u128 << bit_pos);
        }

        Ok(())
    }
}

impl Default for BoardBits {
    fn default() -> Self {
        Self::new()
    }
}

/// Board型からBoardBitsへの変換
impl From<&super::board::Board> for BoardBits {
    fn from(board: &super::board::Board) -> Self {
        let mut bits = BoardBits::new();
        
        for y in 0..H {
            for x in 0..W {
                if let Some(super::cell::Cell::Fixed(color)) = board.get(x, y) {
                    let _ = bits.set(x, y, color);
                }
            }
        }
        
        bits
    }
}

impl From<super::board::Board> for BoardBits {
    fn from(board: super::board::Board) -> Self {
        Self::from(&board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_board_is_empty() {
        let board = BoardBits::new();
        assert!(board.is_empty());
        assert_eq!(board.all_cells(), 0);
    }

    #[test]
    fn set_and_get_work() {
        let mut board = BoardBits::new();
        board.set(0, 0, 0).unwrap();
        assert_eq!(board.get(0, 0), Some(0));

        board.set(1, 2, 2).unwrap();
        assert_eq!(board.get(1, 2), Some(2));
    }

    #[test]
    fn set_overwrites_previous_color() {
        let mut board = BoardBits::new();
        board.set(0, 0, 0).unwrap();
        board.set(0, 0, 1).unwrap();
        assert_eq!(board.get(0, 0), Some(1));
    }

    #[test]
    fn clear_removes_cell() {
        let mut board = BoardBits::new();
        board.set(0, 0, 2).unwrap();
        board.clear(0, 0).unwrap();
        assert_eq!(board.get(0, 0), None);
    }

    #[test]
    fn set_out_of_bounds_fails() {
        let mut board = BoardBits::new();
        assert!(board.set(W, 0, 0).is_err());
        assert!(board.set(0, H, 0).is_err());
    }

    #[test]
    fn set_invalid_color_fails() {
        let mut board = BoardBits::new();
        assert!(board.set(0, 0, 4).is_err());
    }

    #[test]
    fn all_cells_returns_union() {
        let mut board = BoardBits::new();
        board.set(0, 0, 0).unwrap();
        board.set(1, 1, 1).unwrap();
        board.set(2, 2, 2).unwrap();

        let all = board.all_cells();
        assert_ne!(all, 0);
        assert!(all & (1 << 0) != 0); // (0,0) のビット
        assert!(all & (1 << (1 * H + 1)) != 0); // (1,1) のビット
    }
}
