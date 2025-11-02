// Board型 - 6×14の盤面を表現

use crate::constants::{H, W};
use crate::domain::board::cell::Cell;
use anyhow::{anyhow, Result};

/// 6×14の盤面を表現する型（UI用）
#[derive(Clone, Debug, PartialEq)]
pub struct Board {
    cells: [Cell; W * H],
}

impl Board {
    /// 新しい空の盤面を作成
    pub fn new() -> Self {
        Self {
            cells: [Cell::Blank; W * H],
        }
    }

    /// セルを取得（範囲外はNone）
    pub fn get(&self, x: usize, y: usize) -> Option<Cell> {
        if x >= W || y >= H {
            return None;
        }
        Some(self.cells[y * W + x])
    }

    /// セルを設定
    pub fn set(&mut self, x: usize, y: usize, cell: Cell) -> Result<()> {
        if x >= W || y >= H {
            return Err(anyhow!("座標が範囲外: ({}, {})", x, y));
        }
        self.cells[y * W + x] = cell;
        Ok(())
    }

    /// 盤面全体への直接アクセス（読み取り専用）
    pub fn cells(&self) -> &[Cell; W * H] {
        &self.cells
    }

    /// 盤面の妥当性を検証
    pub fn validate(&self) -> Result<()> {
        // 1. 隣接するAbs系セルのチェック
        for y in 0..H {
            for x in 0..W {
                if let Some(Cell::Abs(i)) = self.get(x, y) {
                    // 右方向をチェック
                    if x + 1 < W {
                        if let Some(Cell::Abs(j)) = self.get(x + 1, y) {
                            if i == j {
                                return Err(anyhow!("隣接するAbsセルが同じ値: ({}, {})", x, y));
                            }
                        }
                    }
                    // 下方向をチェック
                    if y + 1 < H {
                        if let Some(Cell::Abs(j)) = self.get(x, y + 1) {
                            if i == j {
                                return Err(anyhow!("隣接するAbsセルが同じ値: ({}, {})", x, y));
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// デフォルトテンプレート（下3行がAny）
    pub fn with_default_template() -> Self {
        let mut board = Self::new();
        for y in 0..3 {
            for x in 0..W {
                let _ = board.set(x, y, Cell::Any);
            }
        }
        board
    }

    /// 文字列表現から構築
    pub fn from_string(s: &str) -> Result<Self> {
        let mut board = Self::new();
        let chars: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();

        if chars.len() != W * H {
            return Err(anyhow!("文字数が不正: 期待{}、実際{}", W * H, chars.len()));
        }

        for (i, &ch) in chars.iter().enumerate() {
            let x = i % W;
            let y = i / W;
            let cell = Cell::from_char(ch)?;
            board.set(x, y, cell)?;
        }

        Ok(board)
    }

    /// 文字列表現に変換
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(W * H);
        for y in 0..H {
            for x in 0..W {
                s.push(self.get(x, y).unwrap().to_char());
            }
        }
        s
    }
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

/// BoardBits型からBoardへの変換
impl From<&super::board_bits::BoardBits> for Board {
    fn from(bits: &super::board_bits::BoardBits) -> Self {
        let mut board = Board::new();
        
        for y in 0..H {
            for x in 0..W {
                if let Some(color) = bits.get(x, y) {
                    let _ = board.set(x, y, Cell::Fixed(color));
                }
            }
        }
        
        board
    }
}

impl From<super::board_bits::BoardBits> for Board {
    fn from(bits: super::board_bits::BoardBits) -> Self {
        Self::from(&bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_board_is_all_blank() {
        let board = Board::new();
        for y in 0..H {
            for x in 0..W {
                assert_eq!(board.get(x, y), Some(Cell::Blank));
            }
        }
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let board = Board::new();
        assert_eq!(board.get(W, 0), None);
        assert_eq!(board.get(0, H), None);
    }

    #[test]
    fn set_and_get_work() {
        let mut board = Board::new();
        board.set(2, 3, Cell::Fixed(1)).unwrap();
        assert_eq!(board.get(2, 3), Some(Cell::Fixed(1)));
    }

    #[test]
    fn set_out_of_bounds_fails() {
        let mut board = Board::new();
        assert!(board.set(W, 0, Cell::Blank).is_err());
        assert!(board.set(0, H, Cell::Blank).is_err());
    }

    #[test]
    fn validate_rejects_adjacent_abs_horizontal() {
        let mut board = Board::new();
        board.set(0, 0, Cell::Abs(5)).unwrap();
        board.set(1, 0, Cell::Abs(5)).unwrap();

        assert!(board.validate().is_err());
    }

    #[test]
    fn validate_rejects_adjacent_abs_vertical() {
        let mut board = Board::new();
        board.set(0, 0, Cell::Abs(5)).unwrap();
        board.set(0, 1, Cell::Abs(5)).unwrap();

        assert!(board.validate().is_err());
    }

    #[test]
    fn validate_accepts_different_abs() {
        let mut board = Board::new();
        board.set(0, 0, Cell::Abs(5)).unwrap();
        board.set(1, 0, Cell::Abs(6)).unwrap();

        assert!(board.validate().is_ok());
    }

    #[test]
    fn default_template_has_any_in_bottom_rows() {
        let board = Board::with_default_template();
        for y in 0..3 {
            for x in 0..W {
                assert_eq!(board.get(x, y), Some(Cell::Any));
            }
        }
        for y in 3..H {
            for x in 0..W {
                assert_eq!(board.get(x, y), Some(Cell::Blank));
            }
        }
    }

    #[test]
    fn from_string_parses_correctly() {
        let s = "・".repeat(W * H);
        let board = Board::from_string(&s).unwrap();
        assert_eq!(board.get(0, 0), Some(Cell::Blank));
    }

    #[test]
    fn from_string_rejects_wrong_length() {
        let s = "・・・";
        assert!(Board::from_string(&s).is_err());
    }

    #[test]
    fn to_string_roundtrip() {
        let mut board = Board::new();
        board.set(0, 0, Cell::Fixed(0)).unwrap();
        board.set(1, 0, Cell::Any).unwrap();
        board.set(2, 0, Cell::Abs(5)).unwrap();

        let s = board.to_string();
        let board2 = Board::from_string(&s).unwrap();
        assert_eq!(board, board2);
    }

    #[test]
    fn convert_to_board_bits_only_fixed() {
        use super::super::board_bits::BoardBits;
        
        let mut board = Board::new();
        board.set(0, 0, Cell::Fixed(0)).unwrap();
        board.set(1, 1, Cell::Fixed(2)).unwrap();
        board.set(2, 2, Cell::Fixed(3)).unwrap();
        
        let bits = BoardBits::from(&board);
        assert_eq!(bits.get(0, 0), Some(0));
        assert_eq!(bits.get(1, 1), Some(2));
        assert_eq!(bits.get(2, 2), Some(3));
        assert_eq!(bits.get(3, 3), None);
    }
    
    #[test]
    fn convert_from_board_bits() {
        use super::super::board_bits::BoardBits;
        
        let mut bits = BoardBits::new();
        bits.set(0, 0, 1).unwrap();
        bits.set(2, 3, 2).unwrap();
        
        let board = Board::from(&bits);
        assert_eq!(board.get(0, 0), Some(Cell::Fixed(1)));
        assert_eq!(board.get(2, 3), Some(Cell::Fixed(2)));
        assert_eq!(board.get(1, 1), Some(Cell::Blank));
    }
    
    #[test]
    fn roundtrip_board_boardbits_conversion() {
        use super::super::board_bits::BoardBits;
        
        let mut board = Board::new();
        board.set(0, 0, Cell::Fixed(0)).unwrap();
        board.set(1, 1, Cell::Fixed(1)).unwrap();
        board.set(2, 2, Cell::Fixed(2)).unwrap();
        board.set(3, 3, Cell::Fixed(3)).unwrap();
        
        let bits = BoardBits::from(&board);
        let board2 = Board::from(&bits);
        
        // Fixed セルのみが保持される
        assert_eq!(board2.get(0, 0), Some(Cell::Fixed(0)));
        assert_eq!(board2.get(1, 1), Some(Cell::Fixed(1)));
        assert_eq!(board2.get(2, 2), Some(Cell::Fixed(2)));
        assert_eq!(board2.get(3, 3), Some(Cell::Fixed(3)));
    }
}
