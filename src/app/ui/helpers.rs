// UI描画用のヘルパー関数

use crate::constants::{H, W};
use crate::model::cell_style;
use crate::profiling::{fmt_dur_ms, ProfileTotals};
use egui::{Color32, RichText, Vec2};

/// カラーパレット（4色のぷよ）
pub const COLOR_PALETTE: [Color32; 4] = [
    Color32::from_rgb(239, 68, 68),  // red
    Color32::from_rgb(34, 197, 94),  // green
    Color32::from_rgb(59, 130, 246), // blue
    Color32::from_rgb(234, 179, 8),  // yellow
];

/// ペアプレビューの描画
pub fn draw_pair_preview(ui: &mut egui::Ui, pair: (u8, u8)) {
    let sz = Vec2::new(18.0, 18.0);
    ui.vertical(|ui| {
        let (txt1, fill1, stroke1) = cell_style(crate::model::Cell::Fixed(pair.1));
        let (txt0, fill0, stroke0) = cell_style(crate::model::Cell::Fixed(pair.0));
        let top = egui::Button::new(RichText::new(txt1).size(11.0))
            .min_size(sz)
            .fill(fill1)
            .stroke(stroke1);
        let bot = egui::Button::new(RichText::new(txt0).size(11.0))
            .min_size(sz)
            .fill(fill0)
            .stroke(stroke0);
        let _ = ui.add(top);
        let _ = ui.add(bot);
    });
}

/// プレビュー描画（列形式の盤面）
pub fn draw_preview(ui: &mut egui::Ui, cols: &[[u16; W]; 4]) {
    let cell = 16.0_f32;
    let gap = 1.0_f32;

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // すべてのマスを描画（背景 + ぷよ）
    for x in 0..W {
        for y in 0..H {
            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));

            let bit = 1u16 << y;
            let fill = if cols[0][x] & bit != 0 {
                COLOR_PALETTE[0]
            } else if cols[1][x] & bit != 0 {
                COLOR_PALETTE[1]
            } else if cols[2][x] & bit != 0 {
                COLOR_PALETTE[2]
            } else if cols[3][x] & bit != 0 {
                COLOR_PALETTE[3]
            } else {
                // 空マスは薄いグレー
                Color32::from_rgb(240, 240, 240)
            };

            painter.rect_filled(r, 3.0, fill);
            // グリッド線（境界）
            painter.rect_stroke(r, 3.0, egui::Stroke::new(0.5, Color32::from_gray(200)));
        }
    }
}

/// プロファイル表示
pub fn show_profile_table(ui: &mut egui::Ui, p: &ProfileTotals) {
    ui.monospace(format!(
        "I/O 書き込み合計: {}",
        fmt_dur_ms(p.io_write_total)
    ));
    ui.add_space(4.0);
    egui::Grid::new("profile-grid")
        .striped(true)
        .num_columns(16)
        .show(ui, |ui| {
            ui.monospace("深さ");
            ui.monospace("nodes");
            ui.monospace("cand");
            ui.monospace("pruned");
            ui.monospace("leaves");
            ui.monospace("pre_thres");
            ui.monospace("pre_e1ng");
            ui.monospace("L-hit");
            ui.monospace("G-hit");
            ui.monospace("Miss");
            ui.monospace("gen");
            ui.monospace("assign");
            ui.monospace("upper");
            ui.monospace("hash");
            ui.monospace("reach");
            ui.monospace("serial");
            ui.end_row();

            for d in 0..=W {
                let t = p.dfs_times[d];
                let c = p.dfs_counts[d];
                ui.monospace(format!("{}", d));
                ui.monospace(format!("{}", c.nodes));
                ui.monospace(format!("{}", c.cand_generated));
                ui.monospace(format!("{}", c.pruned_upper));
                ui.monospace(format!("{}", c.leaves));
                ui.monospace(format!("{}", c.leaf_pre_tshort));
                ui.monospace(format!("{}", c.leaf_pre_e1_impossible));
                ui.monospace(format!("{}", c.memo_lhit));
                ui.monospace(format!("{}", c.memo_ghit));
                ui.monospace(format!("{}", c.memo_miss));
                ui.monospace(fmt_dur_ms(t.gen_candidates));
                ui.monospace(fmt_dur_ms(t.assign_cols));
                ui.monospace(fmt_dur_ms(t.upper_bound));
                ui.monospace(fmt_dur_ms(t.leaf_hash));
                ui.monospace(fmt_dur_ms(t.leaf_memo_miss_compute));
                ui.monospace(fmt_dur_ms(t.out_serialize));
                ui.end_row();
            }
        });
}
