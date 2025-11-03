// 総当たり探索タブのUI

use egui::{Color32, RichText, Vec2};
use num_traits::ToPrimitive;

use crate::app::App;
use crate::app::ui::cell_style;
use crate::constants::{H, W};
use crate::domain::board::cell::{cycle_abs, cycle_any, cycle_fixed};
use crate::domain::board::Cell;

pub struct BruteforceUI;

impl BruteforceUI {
    pub fn draw_controls(app: &mut App, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("入力と操作");
            ui.label(
                "左クリック: A→B→…→M / 中クリック: N→X→N / 右クリック: ・（空白） / Shift+左: RGBY",
            );

            ui.horizontal_wrapped(|ui| {
                ui.add(
                    egui::DragValue::new(&mut app.threshold)
                        .clamp_range(1..=19)
                        .speed(0.1),
                );
                ui.label("連鎖閾値");
                ui.add_space(8.0);
                ui.add(
                    egui::DragValue::new(&mut app.lru_k)
                        .clamp_range(10..=1000)
                        .speed(1.0),
                );
                ui.label("形キャッシュ上限(千)");
            });

            ui.horizontal_wrapped(|ui| {
                ui.add(
                    egui::DragValue::new(&mut app.stop_progress_plateau)
                        .clamp_range(0.0..=1.0)
                        .speed(0.01),
                );
                ui.label("早期終了: 進捗停滞比 (0=無効, 例 0.10)");
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(
                    &mut app.exact_four_only,
                    "4個消しモード（5個以上で消えたら除外）",
                );
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut app.profile_enabled, "計測を有効化（軽量）");
            });

            ui.horizontal(|ui| {
                if ui
                    .add_enabled(!app.running, egui::Button::new("Run"))
                    .clicked()
                {
                    app.start_run();
                }
                if ui
                    .add_enabled(app.running, egui::Button::new("Stop"))
                    .clicked()
                {
                    if let Some(handle) = &app.search_handle {
                        handle.abort();
                    }
                }
            });

            ui.horizontal(|ui| {
                ui.label("出力ファイル:");
                ui.text_edit_singleline(&mut app.out_name);
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("保存先の選択")
                        .set_file_name(&app.out_name)
                        .save_file()
                    {
                        app.out_path = Some(path.clone());
                        app.out_name = path
                            .file_name()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .into();
                    }
                }
            });
        });
    }

    pub fn draw_stats(app: &App, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("実行・進捗");
            let pct = {
                let total = app.stats.total.to_f64().unwrap_or(0.0);
                let done = app.stats.done.to_f64().unwrap_or(0.0);
                if total > 0.0 { (done / total * 100.0).clamp(0.0, 100.0) } else { 0.0 }
            };
            ui.label(format!("進捗: {:.1}%", pct));
            ui.add(egui::ProgressBar::new((pct / 100.0) as f32).show_percentage());

            ui.add_space(4.0);
            ui.monospace(format!(
                "探索中: {} / 代表集合: {} / 出力件数: {} / 展開節点: {} / 枝刈り: {} / 総組合せ(厳密): {} / 完了: {} / 速度: {:.1} nodes/s / メモ: L-hit={} G-hit={} Miss={} / 形キャッシュ: 上限={} 実={}",
                if app.stats.searching { "YES" } else { "NO" },
                app.stats.unique,
                app.stats.output,
                app.stats.nodes,
                app.stats.pruned,
                &app.stats.total,
                &app.stats.done,
                app.stats.rate,
                app.stats.memo_hit_local,
                app.stats.memo_hit_global,
                app.stats.memo_miss,
                app.stats.lru_limit,
                app.stats.memo_len
            ));
        });
    }

    pub fn draw_board(app: &mut App, ui: &mut egui::Ui) {
        ui.label("盤面（左: A→B→…→M / 中: N↔X / 右: ・ / Shift+左: RGBY）");
        ui.add_space(6.0);

        let cell_size = Vec2::new(28.0, 28.0);
        let gap = 2.0;

        let shift_pressed = ui.input(|i| i.modifiers.shift);
        for y in (0..H).rev() {
            ui.horizontal(|ui| {
                for x in 0..W {
                    let i = y * W + x;
                    let (text, fill, stroke) = cell_style(app.board[i]);
                    let btn = egui::Button::new(RichText::new(text).size(12.0))
                        .min_size(cell_size)
                        .fill(fill)
                        .stroke(stroke);
                    let resp = ui.add(btn);
                    if resp.clicked_by(egui::PointerButton::Primary) {
                        if shift_pressed {
                            app.board[i] = cycle_fixed(app.board[i]);
                        } else {
                            app.board[i] = cycle_abs(app.board[i]);
                        }
                    }
                    if resp.clicked_by(egui::PointerButton::Middle) {
                        app.board[i] = cycle_any(app.board[i]);
                    }
                    if resp.clicked_by(egui::PointerButton::Secondary) {
                        app.board[i] = Cell::Blank;
                    }
                    ui.add_space(gap);
                }
            });
            ui.add_space(gap);
        }
    }
}
