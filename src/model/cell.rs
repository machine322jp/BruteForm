// セル型定義（UI層向けヘルパー）
// Cell型の本体はdomain::board::cellに移行済み

use crate::domain::board::Cell;
use egui::Color32;

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

// cycle関数はドメイン層から再エクスポート
pub use crate::domain::board::cell::{cycle_abs, cycle_any, cycle_fixed};
