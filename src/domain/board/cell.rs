// セル型定義（ドメイン層）

use anyhow::{anyhow, Result};

/// UI で使用するセル型（盤面入力用）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Blank,     // '・'
    Any,       // 'N' (0個以上)
    Any4,      // 'X' (1個以上)
    Abs(u8),   // 0..=12 = 'A'..'M'
    Fixed(u8), // 0..=3 = 'R','G','B','Y'
}

/// テンプレート用セル型（探索処理用）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TCell {
    Blank,
    Any,
    Any4,
    Fixed(u8),
}

impl Cell {
    /// 文字からCellに変換
    pub fn from_char(ch: char) -> Result<Self> {
        match ch {
            '・' | '.' => Ok(Cell::Blank),
            'N' => Ok(Cell::Any),
            'X' => Ok(Cell::Any4),
            'R' => Ok(Cell::Fixed(0)),
            'G' => Ok(Cell::Fixed(1)),
            'B' => Ok(Cell::Fixed(2)),
            'Y' => Ok(Cell::Fixed(3)),
            'A' | 'C'..='F' | 'H'..='M' => Ok(Cell::Abs((ch as u8) - b'A')),
            _ => Err(anyhow!("不正な文字: {}", ch)),
        }
    }

    /// Cellを文字に変換
    pub fn to_char(self) -> char {
        match self {
            Cell::Blank => '・',
            Cell::Any => 'N',
            Cell::Any4 => 'X',
            Cell::Abs(i) => (b'A' + i) as char,
            Cell::Fixed(0) => 'R',
            Cell::Fixed(1) => 'G',
            Cell::Fixed(2) => 'B',
            Cell::Fixed(3) => 'Y',
            Cell::Fixed(_) => '?',
        }
    }

    /// セルをラベル文字に変換（UI互換用）
    pub fn label_char(self) -> char {
        match self {
            Cell::Blank => '.',
            Cell::Any => 'N',
            Cell::Any4 => 'X',
            Cell::Abs(i) => (b'A' + i) as char,
            Cell::Fixed(c) => (b'0' + c) as char,
        }
    }
}

impl TCell {
    /// CellからTCellに変換
    pub fn from_cell(cell: Cell) -> Self {
        match cell {
            Cell::Blank => TCell::Blank,
            Cell::Any => TCell::Any,
            Cell::Any4 => TCell::Any4,
            Cell::Abs(_) => TCell::Any, // Absは探索時にはAnyとして扱う
            Cell::Fixed(c) => TCell::Fixed(c),
        }
    }
}

/// Abs 系のサイクル（A→B→...→M→A）
pub fn cycle_abs(c: Cell) -> Cell {
    match c {
        Cell::Blank | Cell::Any | Cell::Any4 => Cell::Abs(0),
        Cell::Abs(i) => Cell::Abs(((i as usize + 1) % 13) as u8),
        Cell::Fixed(_) => Cell::Abs(0),
    }
}

/// Any ↔ Any4 のサイクル
pub fn cycle_any(c: Cell) -> Cell {
    match c {
        Cell::Any => Cell::Any4,
        Cell::Any4 => Cell::Any,
        _ => Cell::Any,
    }
}

/// Fixed のサイクル（R→G→B→Y→R）
pub fn cycle_fixed(c: Cell) -> Cell {
    match c {
        Cell::Fixed(v) => Cell::Fixed(((v as usize + 1) % 4) as u8),
        _ => Cell::Fixed(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_char_converts_correctly() {
        assert_eq!(Cell::from_char('.').unwrap(), Cell::Blank);
        assert_eq!(Cell::from_char('・').unwrap(), Cell::Blank);
        assert_eq!(Cell::from_char('N').unwrap(), Cell::Any);
        assert_eq!(Cell::from_char('X').unwrap(), Cell::Any4);
        assert_eq!(Cell::from_char('A').unwrap(), Cell::Abs(0));
        assert_eq!(Cell::from_char('M').unwrap(), Cell::Abs(12));
        assert_eq!(Cell::from_char('R').unwrap(), Cell::Fixed(0));
        assert_eq!(Cell::from_char('Y').unwrap(), Cell::Fixed(3));
    }

    #[test]
    fn from_char_rejects_invalid() {
        assert!(Cell::from_char('Z').is_err());
        assert!(Cell::from_char('9').is_err());
    }

    #[test]
    fn to_char_roundtrip() {
        let cells = vec![
            Cell::Blank,
            Cell::Any,
            Cell::Any4,
            Cell::Abs(0),
            Cell::Abs(5),
            Cell::Abs(12),
            Cell::Fixed(0),
            Cell::Fixed(1),
            Cell::Fixed(2),
            Cell::Fixed(3),
        ];

        for cell in cells {
            let ch = cell.to_char();
            assert_eq!(Cell::from_char(ch).unwrap(), cell);
        }
    }

    #[test]
    fn cycle_abs_works() {
        assert_eq!(cycle_abs(Cell::Blank), Cell::Abs(0));
        assert_eq!(cycle_abs(Cell::Abs(0)), Cell::Abs(1));
        assert_eq!(cycle_abs(Cell::Abs(12)), Cell::Abs(0));
    }

    #[test]
    fn cycle_any_toggles() {
        assert_eq!(cycle_any(Cell::Any), Cell::Any4);
        assert_eq!(cycle_any(Cell::Any4), Cell::Any);
        assert_eq!(cycle_any(Cell::Blank), Cell::Any);
    }

    #[test]
    fn cycle_fixed_rotates() {
        assert_eq!(cycle_fixed(Cell::Fixed(0)), Cell::Fixed(1));
        assert_eq!(cycle_fixed(Cell::Fixed(3)), Cell::Fixed(0));
    }
}
