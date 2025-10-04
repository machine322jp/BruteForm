// セル型定義

use egui::Color32;

/// UI で使用するセル型（盤面入力用）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Blank,      // '.'
    Any,        // 'N' (空白 or 色)
    Any4,       // 'X' (色のみ)
    Abs(u8),    // 0..12 = 'A'..'M'
    Fixed(u8),  // 0..=3 = '0'..'3' (RGBY固定)
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
    /// セルをラベル文字に変換
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

/// セルのスタイル情報を取得（表示テキスト、塗りつぶし色、ストローク）
pub fn cell_style(c: Cell) -> (String, Color32, egui::Stroke) {
    match c {
        Cell::Blank => (
            "・".to_string(),
            Color32::WHITE,
            egui::Stroke::new(1.0, Color32::LIGHT_GRAY),
        ),
        Cell::Any => (
            "N".to_string(),
            Color32::from_rgb(254, 243, 199),
            egui::Stroke::new(1.0, Color32::from_rgb(245, 158, 11)),
        ),
        Cell::Any4 => (
            "X".to_string(),
            Color32::from_rgb(220, 252, 231),
            egui::Stroke::new(1.0, Color32::from_rgb(22, 163, 74)),
        ),
        Cell::Abs(i) => {
            let ch = (b'A' + i) as char;
            (
                ch.to_string(),
                Color32::from_rgb(238, 242, 255),
                egui::Stroke::new(1.0, Color32::from_rgb(99, 102, 241)),
            )
        }
        Cell::Fixed(i) => {
            // 0:R, 1:G, 2:B, 3:Y（表示は R/G/B/Y）
            match i {
                0 => (
                    "R".to_string(),
                    Color32::from_rgb(254, 226, 226),
                    egui::Stroke::new(1.0, Color32::from_rgb(239, 68, 68)),
                ),
                1 => (
                    "G".to_string(),
                    Color32::from_rgb(220, 252, 231),
                    egui::Stroke::new(1.0, Color32::from_rgb(34, 197, 94)),
                ),
                2 => (
                    "B".to_string(),
                    Color32::from_rgb(219, 234, 254),
                    egui::Stroke::new(1.0, Color32::from_rgb(59, 130, 246)),
                ),
                3 => (
                    "Y".to_string(),
                    Color32::from_rgb(254, 249, 195),
                    egui::Stroke::new(1.0, Color32::from_rgb(234, 179, 8)),
                ),
                _ => (
                    "?".to_string(),
                    Color32::LIGHT_GRAY,
                    egui::Stroke::new(1.0, Color32::DARK_GRAY),
                ),
            }
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
