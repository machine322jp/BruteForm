// 連鎖生成タブのUI

use egui::{Color32, RichText};
use std::collections::HashSet;

use super::helpers::{draw_pair_preview, draw_preview, COLOR_PALETTE};
use crate::app::chain_extension::ChainExtension;
use crate::app::chain_operations::ChainOperations;
use crate::app::App;
use crate::constants::{H, W};

pub struct ChainplayUI;

impl ChainplayUI {
    pub fn draw_controls(app: &mut App, ui: &mut egui::Ui) {
        // CPU機能検出（初回のみ）
        static CPU_INFO_SHOWN: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !CPU_INFO_SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let bmi2 = std::is_x86_feature_detected!("bmi2");
                let popcnt = std::is_x86_feature_detected!("popcnt");
                app.push_log(format!(
                    "CPU最適化: BMI2={} / POPCNT={}",
                    if bmi2 { "有効" } else { "無効" },
                    if popcnt { "有効" } else { "無効" }
                ));
                if bmi2 {
                    app.push_log("ビットボード処理が高速化されています".to_string());
                }
            }
        }

        ui.group(|ui| {
            ui.label("連鎖生成 — 操作");
            let cur = ChainOperations::current_pair(&app.cp);
            let nxt = ChainOperations::next_pair(&app.cp);
            let dnx = ChainOperations::dnext_pair(&app.cp);

            ui.horizontal(|ui| {
                ui.label("現在手:");
                draw_pair_preview(ui, cur);
                ui.add_space(12.0);
                ui.label("Next:");
                draw_pair_preview(ui, nxt);
                ui.add_space(12.0);
                ui.label("Next2:");
                draw_pair_preview(ui, dnx);
            });

            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let can_ops = !app.cp.lock && app.cp.anim.is_none();
                if ui
                    .add_enabled(can_ops, egui::Button::new("ランダム配置"))
                    .clicked()
                {
                    if let Err(e) = ChainOperations::place_random(&mut app.cp) {
                        app.push_log(e);
                    }
                }
                if ui
                    .add_enabled(can_ops, egui::Button::new("目標配置"))
                    .clicked()
                {
                    let target_board = app.cp.target_board.clone();
                    match ChainOperations::place_target(&mut app.cp, &target_board) {
                        Ok(_) => {}
                        Err(e) => app.push_log(e),
                    }
                }
                if ui
                    .add_enabled(can_ops, egui::Button::new("目標盤面更新"))
                    .clicked()
                {
                    let beam_width = app.cp.beam_width;
                    let max_depth = app.cp.max_depth as u8;
                    let msg = ChainOperations::update_target(&mut app.cp, beam_width, max_depth);
                    app.push_log(msg);
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !app.cp.lock && app.cp.anim.is_none();
                if ui
                    .add_enabled(can_ops, egui::Button::new("連鎖検出（目標盤面）"))
                    .clicked()
                {
                    match ChainOperations::detect_target_chain(&mut app.cp) {
                        Ok(msg) => app.push_log(msg),
                        Err(e) => app.push_log(e),
                    }
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !app.cp.lock && app.cp.anim.is_none();
                let has_chain_info = app.cp.target_chain_info.is_some();
                if ui
                    .add_enabled(
                        can_ops && has_chain_info,
                        egui::Button::new("1連鎖目フリートップ削除"),
                    )
                    .clicked()
                {
                    Self::handle_remove_freetop(app);
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !app.cp.lock && app.cp.anim.is_none();
                let has_removed = app.cp.removed_freetop.is_some();
                if ui
                    .add_enabled(can_ops && has_removed, egui::Button::new("頭伸ばし"))
                    .clicked()
                {
                    Self::handle_extend_head(app);
                }
            });
            ui.add_space(6.0);
            ui.group(|ui| {
                ui.label("探索パラメータ（右盤面生成）");
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        egui::DragValue::new(&mut app.cp.beam_width)
                            .clamp_range(1..=64)
                            .speed(1.0),
                    );
                    ui.label("ビーム幅");
                    ui.add_space(12.0);
                    ui.add(
                        egui::DragValue::new(&mut app.cp.max_depth)
                            .clamp_range(1..=20)
                            .speed(1.0),
                    );
                    ui.label("最大深さ");
                });
            });
            ui.horizontal(|ui| {
                let prev = app.verbose_logging;
                ui.checkbox(&mut app.verbose_logging, "詳細ログ出力");
                if prev != app.verbose_logging {
                    // グローバルフラグを更新
                    if app.verbose_logging {
                        crate::logging::enable_verbose_logging();
                    } else {
                        crate::logging::disable_verbose_logging();
                    }
                }
                if app.verbose_logging {
                    ui.label(
                        RichText::new("（debug_log.txt に詳細ログを出力）")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                } else {
                    ui.label(
                        RichText::new("（ログ出力を抑制して高速化）")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !app.cp.lock && app.cp.anim.is_none();
                if ui
                    .add_enabled(
                        can_ops && app.cp.undo_stack.len() > 1,
                        egui::Button::new("戻る"),
                    )
                    .clicked()
                {
                    ChainOperations::undo(&mut app.cp);
                }
                if ui
                    .add_enabled(
                        can_ops && app.cp.undo_stack.len() > 1,
                        egui::Button::new("初手に戻る"),
                    )
                    .clicked()
                {
                    ChainOperations::reset_to_initial(&mut app.cp);
                }
            });
            ui.label(if app.cp.lock {
                "連鎖中…（操作ロック）"
            } else {
                "待機中"
            });
        });
    }

    pub fn draw_board(app: &mut App, ui: &mut egui::Ui) {
        ui.label("連鎖生成 — 実盤面");
        ui.add_space(6.0);
        draw_preview(ui, &app.cp.cols);

        ui.add_space(12.0);
        ui.label("目標盤面（クリックで R→G→B→Y→空白 の順で遷移）");
        ui.add_space(6.0);
        Self::draw_target_board_editable(app, ui);
    }

    fn handle_remove_freetop(app: &mut App) {
        if app.cp.lock || app.cp.anim.is_some() {
            return;
        }

        // 連鎖情報がない場合は何もしない
        let Some(ref chain_info) = app.cp.target_chain_info else {
            app.push_log(
                "連鎖情報がありません。先に「連鎖検出（目標盤面）」を実行してください".into(),
            );
            return;
        };

        match ChainExtension::remove_first_chain_freetop(
            &mut app.cp.target_board,
            chain_info,
            &mut app.cp.around_cells_cache,
        ) {
            Ok((remove_col, remove_y)) => {
                // 削除した位置を保存（頭伸ばし用）
                app.cp.removed_freetop = Some((remove_col, remove_y));
                app.push_log(format!(
                    "1連鎖目のフリートップを削除: 列={}, y={} (頭伸ばし準備完了)",
                    remove_col, remove_y
                ));
            }
            Err(e) => {
                app.push_log(e);
            }
        }
    }

    fn handle_extend_head(app: &mut App) {
        if app.cp.lock || app.cp.anim.is_some() {
            return;
        }

        // 連鎖情報とフリートップ削除情報をクローン（借用を避ける）
        let chain_info = match app.cp.target_chain_info.as_ref() {
            Some(info) => info.clone(),
            None => {
                app.push_log("連鎖情報がありません。先に「連鎖検出（目標盤面）」→「1連鎖目フリートップ削除」を実行してください".into());
                return;
            }
        };

        let Some(removed_freetop) = app.cp.removed_freetop else {
            app.push_log("フリートップ削除がされていません。先に「1連鎖目フリートップ削除」を実行してください".into());
            return;
        };

        // ベースライン：削除前の連鎖数（目標は削除前を超えること）
        let baseline = chain_info.len() as i32;

        // 候補位置の数を計算して警告表示
        let around_first_count = app
            .cp
            .around_cells_cache
            .as_ref()
            .map(|(first, _)| first.len())
            .unwrap_or(0);
        let total_combinations = 5usize.pow((around_first_count + 2).min(20) as u32);

        if total_combinations > 1_000_000 {
            app.push_log(format!(
                "警告: 総当たり数が多すぎる可能性があります（候補位置数={}）",
                around_first_count + 2
            ));
        }

        // 候補列を収集（ログ表示用）
        let mut cand_cols = HashSet::new();
        if let Some((ref around_first, _)) = app.cp.around_cells_cache {
            for &(x, _) in around_first.iter() {
                cand_cols.insert(x);
            }
        }
        cand_cols.insert(removed_freetop.0);

        app.push_log(format!(
            "頭伸ばし開始: ベースライン={}連鎖, 候補列数={}",
            baseline,
            cand_cols.len()
        ));

        let target_board = app.cp.target_board;
        let around_cache = app.cp.around_cells_cache.clone();

        match ChainExtension::extend_head(
            &target_board,
            baseline,
            removed_freetop,
            &chain_info,
            &around_cache,
        ) {
            Ok((best_chain, new_cols)) => {
                app.cp.target_board = new_cols;

                // キャッシュクリア
                app.cp.target_chain_info = None;
                app.cp.around_cells_cache = None;
                app.cp.removed_freetop = None;

                app.push_log(format!(
                    "頭伸ばし成功: {}連鎖 → {}連鎖（候補列={:?}、削除列={}）★目標盤面を更新しました",
                    baseline, best_chain,
                    {
                        let mut v: Vec<_> = cand_cols.iter().copied().collect();
                        v.sort_unstable();
                        v
                    },
                    removed_freetop.0
                ));
            }
            Err(e) => {
                app.push_log(format!("頭伸ばし失敗: {}", e));
            }
        }
    }

    fn draw_target_board_editable(app: &mut App, ui: &mut egui::Ui) {
        let cell = 16.0_f32;
        let gap = 1.0_f32;

        let width = W as f32 * cell + (W - 1) as f32 * gap;
        let height = H as f32 * cell + (H - 1) as f32 * gap;

        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
        let painter = ui.painter_at(rect);

        // 強調表示対象のマスを収集
        let mut first_chain_cells = HashSet::new();
        let mut last_chain_cells = HashSet::new();
        let around_first_9_cells;
        let around_last_9_cells;

        if let Some(ref chain_info) = app.cp.target_chain_info {
            if !chain_info.is_empty() {
                // 1連鎖目（元の盤面での位置を使用）
                for group in &chain_info[0].original_groups {
                    for &(x, y) in group {
                        first_chain_cells.insert((x, y));
                    }
                }

                // 最終連鎖（元の盤面での位置を使用）
                let last_idx = chain_info.len() - 1;
                for group in &chain_info[last_idx].original_groups {
                    for &(x, y) in group {
                        last_chain_cells.insert((x, y));
                    }
                }

                // キャッシュから周囲9マスを取得
                if let Some((ref first, ref last)) = app.cp.around_cells_cache {
                    around_first_9_cells = first.clone();
                    around_last_9_cells = last.clone();
                } else {
                    around_first_9_cells = HashSet::new();
                    around_last_9_cells = HashSet::new();
                }
            } else {
                around_first_9_cells = HashSet::new();
                around_last_9_cells = HashSet::new();
            }
        } else {
            around_first_9_cells = HashSet::new();
            around_last_9_cells = HashSet::new();
        }

        for y in 0..H {
            for x in 0..W {
                let mut cidx: Option<usize> = None;
                let bit = 1u16 << y;
                if app.cp.target_board[0][x] & bit != 0 {
                    cidx = Some(0);
                } else if app.cp.target_board[1][x] & bit != 0 {
                    cidx = Some(1);
                } else if app.cp.target_board[2][x] & bit != 0 {
                    cidx = Some(2);
                } else if app.cp.target_board[3][x] & bit != 0 {
                    cidx = Some(3);
                }

                let fill = cidx.map(|k| COLOR_PALETTE[k]).unwrap_or(Color32::WHITE);

                // 強調表示の適用
                let x0 = rect.min.x + x as f32 * (cell + gap);
                let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
                let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));

                // まず通常の色で塗る
                painter.rect_filled(r, 3.0, fill);

                // 強調表示のオーバーレイ（優先順位: 実際の連鎖セル > 周囲マス）
                if first_chain_cells.contains(&(x, y)) {
                    // 1連鎖目: 黄色の太枠
                    painter.rect_stroke(
                        r,
                        3.0,
                        egui::Stroke::new(3.0, Color32::from_rgb(255, 215, 0)),
                    );
                } else if last_chain_cells.contains(&(x, y)) {
                    // 最終連鎖: オレンジの太枠
                    painter.rect_stroke(
                        r,
                        3.0,
                        egui::Stroke::new(3.0, Color32::from_rgb(255, 140, 0)),
                    );
                } else if around_first_9_cells.contains(&(x, y)) {
                    // 1連鎖目の周囲9マス: 薄い黄色の細枠
                    painter.rect_stroke(
                        r,
                        3.0,
                        egui::Stroke::new(1.5, Color32::from_rgb(255, 255, 150)),
                    );
                } else if around_last_9_cells.contains(&(x, y)) {
                    // 最終連鎖の周囲9マス: 薄いオレンジの細枠
                    painter.rect_stroke(
                        r,
                        3.0,
                        egui::Stroke::new(1.5, Color32::from_rgb(255, 200, 150)),
                    );
                }
            }
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let rel = pos - rect.min;
                let click_x = (rel.x / (cell + gap)) as usize;
                let click_y = ((rect.max.y - pos.y) / (cell + gap)) as usize;

                if click_x < W && click_y < H {
                    Self::toggle_target_cell(app, click_x, click_y);
                }
            }
        }
    }

    fn toggle_target_cell(app: &mut App, x: usize, y: usize) {
        let bit = 1u16 << y;

        let current = if app.cp.target_board[0][x] & bit != 0 {
            Some(0)
        } else if app.cp.target_board[1][x] & bit != 0 {
            Some(1)
        } else if app.cp.target_board[2][x] & bit != 0 {
            Some(2)
        } else if app.cp.target_board[3][x] & bit != 0 {
            Some(3)
        } else {
            None
        };

        for c in 0..4 {
            app.cp.target_board[c][x] &= !bit;
        }

        match current {
            None => app.cp.target_board[0][x] |= bit,
            Some(0) => app.cp.target_board[1][x] |= bit,
            Some(1) => app.cp.target_board[2][x] |= bit,
            Some(2) => app.cp.target_board[3][x] |= bit,
            Some(3) => {}
            Some(_) => {}
        }
    }
}
