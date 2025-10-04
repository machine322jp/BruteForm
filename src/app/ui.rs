// GUI実装

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use crossbeam_channel::unbounded;
use egui::{Color32, RichText, Vec2};
use rand::Rng;
use num_traits::ToPrimitive;

use crate::constants::{W, H};
use crate::model::{Cell, cell_style, cycle_abs, cycle_any, cycle_fixed};
use crate::app::{App, Mode, Message, Stats};
use crate::app::chain_play::{ChainPlay, SavedState, AnimState, AnimPhase, Orient};
use crate::profiling::{ProfileTotals, has_profile_any, fmt_dur_ms};
use crate::search::run_search;
use crate::search::board::{apply_clear_no_fall, apply_erase_and_fall_cols};

/// ペアプレビューの描画
fn draw_pair_preview(ui: &mut egui::Ui, pair: (u8, u8)) {
    let sz = Vec2::new(18.0, 18.0);
    ui.vertical(|ui| {
        let (txt1, fill1, stroke1) = cell_style(Cell::Fixed(pair.1));
        let (txt0, fill0, stroke0) = cell_style(Cell::Fixed(pair.0));
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

/// プレビュー描画
fn draw_preview(ui: &mut egui::Ui, cols: &[[u16; W]; 4]) {
    let cell = 16.0_f32;
    let gap = 1.0_f32;

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let palette = [
        Color32::from_rgb(239, 68, 68),   // red
        Color32::from_rgb(34, 197, 94),   // green
        Color32::from_rgb(59, 130, 246),  // blue
        Color32::from_rgb(234, 179, 8),   // yellow
    ];

    // すべてのマスを描画（背景 + ぷよ）
    for x in 0..W {
        for y in 0..H {
            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));

            let bit = 1u16 << y;
            let fill = if cols[0][x] & bit != 0 {
                palette[0]
            } else if cols[1][x] & bit != 0 {
                palette[1]
            } else if cols[2][x] & bit != 0 {
                palette[2]
            } else if cols[3][x] & bit != 0 {
                palette[3]
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
fn show_profile_table(ui: &mut egui::Ui, p: &ProfileTotals) {
    ui.monospace(format!("I/O 書き込み合計: {}", fmt_dur_ms(p.io_write_total)));
    ui.add_space(4.0);
    egui::Grid::new("profile-grid").striped(true).num_columns(16).show(ui, |ui| {
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

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.mode == Mode::ChainPlay {
            self.cp_step_animation();
        }

        if let Some(rx) = self.rx.take() {
            let mut keep_rx = true;

            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Message::Log(s) => self.push_log(s),
                    Message::Preview(p) => self.preview = Some(p),
                    Message::Progress(mut st) => {
                        let prof = self.stats.profile.clone();
                        st.profile = prof;
                        self.stats = st;
                    }
                    Message::Finished(mut st) => {
                        let prof = self.stats.profile.clone();
                        st.profile = prof;
                        self.stats = st;
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log("完了".into());
                        keep_rx = false;
                    }
                    Message::Error(e) => {
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log(format!("エラー: {e}"));
                        keep_rx = false;
                    }
                    Message::TimeDelta(td) => {
                        self.stats.profile.add_delta(&td);
                    }
                }
            }

            if keep_rx {
                self.rx = Some(rx);
            }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI（列ストリーミング＋LRU形キャッシュ＋並列化＋計測＋追撃最適化）");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::BruteForce, "総当たり");
                ui.selectable_value(&mut self.mode, Mode::ChainPlay, "連鎖生成");
            });
        });

        egui::SidePanel::left("left").min_width(420.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
                if self.mode == Mode::BruteForce {
                    self.draw_bruteforce_controls(ui);
                } else {
                    self.draw_chainplay_controls(ui);
                }

                ui.separator();

                if self.mode == Mode::BruteForce && !self.running && self.profile_enabled && has_profile_any(&self.stats.profile) {
                    ui.group(|ui| {
                        ui.label("処理時間（累積）");
                        show_profile_table(ui, &self.stats.profile);
                    });
                    ui.separator();
                }

                if self.mode == Mode::BruteForce {
                    ui.label("プレビュー（E1直前の落下後盤面）");
                    ui.add_space(4.0);
                    if let Some(cols) = &self.preview {
                        draw_preview(ui, cols);
                    } else {
                        ui.label(RichText::new("（実行中に更新表示）").italics().color(Color32::GRAY));
                    }
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    ui.label("ログ");
                    for line in &self.log_lines {
                        ui.monospace(line);
                    }
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    self.draw_bruteforce_stats(ui);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                if self.mode == Mode::BruteForce {
                    self.draw_bruteforce_board(ui);
                } else {
                    self.draw_chainplay_board(ui);
                }
            });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl App {
    // ===== 連鎖生成モード：ユーティリティ =====
    fn cp_current_pair(&self) -> (u8, u8) {
        let idx = self.cp.pair_index % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }

    fn cp_next_pair(&self) -> (u8, u8) {
        let idx = (self.cp.pair_index + 1) % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }

    fn cp_dnext_pair(&self) -> (u8, u8) {
        let idx = (self.cp.pair_index + 2) % self.cp.pair_seq.len().max(1);
        self.cp.pair_seq[idx]
    }

    fn cp_undo(&mut self) {
        if self.cp.undo_stack.len() > 1 && !self.cp.lock {
            self.cp.anim = None;
            self.cp.erased_cols = None;
            self.cp.next_cols = None;
            // 現在の状態を削除
            self.cp.undo_stack.pop();
            // 直前の状態を取得（削除はしない）
            if let Some(last) = self.cp.undo_stack.last().copied() {
                self.cp.cols = last.cols;
                self.cp.pair_index = last.pair_index;
            }
        }
    }

    fn cp_reset_to_initial(&mut self) {
        if self.cp.lock {
            return;
        }
        self.cp.anim = None;
        self.cp.erased_cols = None;
        self.cp.next_cols = None;
        if let Some(first) = self.cp.undo_stack.first().copied() {
            self.cp.cols = first.cols;
            self.cp.pair_index = first.pair_index;
            self.cp.undo_stack.clear();
            self.cp.undo_stack.push(first);
        }
        self.cp.lock = false;
    }

    fn cp_place_random(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }

        let pair = self.cp_current_pair();
        let mut rng = rand::thread_rng();

        let mut moves: Vec<(usize, Orient)> = Vec::new();
        for x in 0..W {
            let h = self.cp_col_height(x);
            if h + 1 < H {
                moves.push((x, Orient::Up));
                moves.push((x, Orient::Down));
            }
            if x + 1 < W {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Right));
                }
            }
            if x >= 1 {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Left));
                }
            }
        }
        if moves.is_empty() {
            self.push_log("置ける場所がありません".into());
            return;
        }
        let (x, orient) = moves[rng.gen_range(0..moves.len())];

        self.cp_place_with(x, orient, pair);
        self.cp_check_and_start_chain();
        
        // 手を打った後に状態を保存
        self.cp.undo_stack.push(SavedState {
            cols: self.cp.cols,
            pair_index: self.cp.pair_index,
        });
    }

    fn cp_place_with(&mut self, x: usize, orient: Orient, pair: (u8, u8)) {
        match orient {
            Orient::Up => {
                let h = self.cp_col_height(x);
                self.cp_set_cell(x, h, pair.0);
                self.cp_set_cell(x, h + 1, pair.1);
            }
            Orient::Down => {
                let h = self.cp_col_height(x);
                self.cp_set_cell(x, h, pair.1);
                self.cp_set_cell(x, h + 1, pair.0);
            }
            Orient::Right => {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                self.cp_set_cell(x, h0, pair.0);
                self.cp_set_cell(x + 1, h1, pair.1);
            }
            Orient::Left => {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                self.cp_set_cell(x, h0, pair.0);
                self.cp_set_cell(x - 1, h1, pair.1);
            }
        }
        if !self.cp.pair_seq.is_empty() {
            self.cp.pair_index = (self.cp.pair_index + 1) % self.cp.pair_seq.len();
        }
    }

    fn cp_col_height(&self, x: usize) -> usize {
        let occ = (self.cp.cols[0][x] | self.cp.cols[1][x] | self.cp.cols[2][x] | self.cp.cols[3][x]) & crate::constants::MASK14;
        occ.count_ones() as usize
    }

    fn cp_set_cell(&mut self, x: usize, y: usize, color: u8) {
        if x >= W || y >= H {
            return;
        }
        let bit = 1u16 << y;
        let c = (color as usize).min(3);
        self.cp.cols[c][x] |= bit;
    }

    fn cp_check_and_start_chain(&mut self) {
        use crate::search::board::compute_erase_mask_cols;
        let clear = compute_erase_mask_cols(&self.cp.cols);
        let any = (0..W).any(|x| clear[x] != 0);
        if !any {
            return;
        }
        self.cp.lock = true;
        let erased = apply_clear_no_fall(&self.cp.cols, &clear);
        let next = crate::search::board::apply_given_clear_and_fall(&self.cp.cols, &clear);
        self.cp.erased_cols = Some(erased);
        self.cp.next_cols = Some(next);
        self.cp.cols = erased;
        self.cp.anim = Some(AnimState {
            phase: AnimPhase::AfterErase,
            since: std::time::Instant::now(),
        });
    }

    fn cp_step_animation(&mut self) {
        use crate::search::board::compute_erase_mask_cols;
        let Some(anim) = self.cp.anim else { return };
        let elapsed = anim.since.elapsed();
        if elapsed < Duration::from_millis(500) {
            return;
        }
        match anim.phase {
            AnimPhase::AfterErase => {
                if let Some(next) = self.cp.next_cols.take() {
                    self.cp.cols = next;
                }
                self.cp.anim = Some(AnimState {
                    phase: AnimPhase::AfterFall,
                    since: std::time::Instant::now(),
                });
            }
            AnimPhase::AfterFall => {
                let clear = compute_erase_mask_cols(&self.cp.cols);
                let any = (0..W).any(|x| clear[x] != 0);
                if !any {
                    self.cp.anim = None;
                    self.cp.erased_cols = None;
                    self.cp.next_cols = None;
                    self.cp.lock = false;
                } else {
                    let erased = apply_clear_no_fall(&self.cp.cols, &clear);
                    let next = crate::search::board::apply_given_clear_and_fall(&self.cp.cols, &clear);
                    self.cp.cols = erased;
                    self.cp.erased_cols = Some(erased);
                    self.cp.next_cols = Some(next);
                    self.cp.anim = Some(AnimState {
                        phase: AnimPhase::AfterErase,
                        since: std::time::Instant::now(),
                    });
                }
            }
        }
    }

    fn cp_place_target(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }

        let target_empty = (0..W).all(|x| {
            self.cp.target_board[0][x] == 0
                && self.cp.target_board[1][x] == 0
                && self.cp.target_board[2][x] == 0
                && self.cp.target_board[3][x] == 0
        });
        if target_empty {
            self.push_log("目標盤面が設定されていません".into());
            return;
        }

        let pair = self.cp_current_pair();

        let mut moves: Vec<(usize, Orient, i32)> = Vec::new();
        for x in 0..W {
            let h = self.cp_col_height(x);
            if h + 1 < H {
                let score_up = self.eval_move(x, Orient::Up, pair);
                let score_down = self.eval_move(x, Orient::Down, pair);
                moves.push((x, Orient::Up, score_up));
                moves.push((x, Orient::Down, score_down));
            }
            if x + 1 < W {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                if h0 < H && h1 < H {
                    let score = self.eval_move(x, Orient::Right, pair);
                    moves.push((x, Orient::Right, score));
                }
            }
            if x >= 1 {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                if h0 < H && h1 < H {
                    let score = self.eval_move(x, Orient::Left, pair);
                    moves.push((x, Orient::Left, score));
                }
            }
        }
        if moves.is_empty() {
            self.push_log("置ける場所がありません".into());
            return;
        }

        let best = moves.iter().max_by_key(|&&(_, _, score)| score).unwrap();
        let (x, orient, score) = *best;

        if score <= 0 {
            self.push_log("目標に寄与する手がありません（ランダム配置を推奨）".into());
            return;
        }

        self.cp_place_with(x, orient, pair);
        self.cp_check_and_start_chain();
        
        // 手を打った後に状態を保存
        self.cp.undo_stack.push(SavedState {
            cols: self.cp.cols,
            pair_index: self.cp.pair_index,
        });
    }

    fn eval_move(&self, x: usize, orient: Orient, pair: (u8, u8)) -> i32 {
        let mut test_cols = self.cp.cols;

        match orient {
            Orient::Up => {
                let h = self.cp_col_height(x);
                if h + 1 >= H {
                    return -1000;
                }
                let bit0 = 1u16 << h;
                let bit1 = 1u16 << (h + 1);
                test_cols[pair.0 as usize][x] |= bit0;
                test_cols[pair.1 as usize][x] |= bit1;
            }
            Orient::Down => {
                let h = self.cp_col_height(x);
                if h + 1 >= H {
                    return -1000;
                }
                let bit0 = 1u16 << h;
                let bit1 = 1u16 << (h + 1);
                test_cols[pair.1 as usize][x] |= bit0;
                test_cols[pair.0 as usize][x] |= bit1;
            }
            Orient::Right => {
                let h0 = self.test_col_height(&test_cols, x);
                let h1 = self.test_col_height(&test_cols, x + 1);
                if h0 >= H || h1 >= H || x + 1 >= W {
                    return -1000;
                }
                test_cols[pair.0 as usize][x] |= 1u16 << h0;
                test_cols[pair.1 as usize][x + 1] |= 1u16 << h1;
            }
            Orient::Left => {
                if x == 0 {
                    return -1000;
                }
                let h0 = self.test_col_height(&test_cols, x);
                let h1 = self.test_col_height(&test_cols, x - 1);
                if h0 >= H || h1 >= H {
                    return -1000;
                }
                test_cols[pair.0 as usize][x] |= 1u16 << h0;
                test_cols[pair.1 as usize][x - 1] |= 1u16 << h1;
            }
        }

        let mut score = 0i32;
        for c in 0..4 {
            for col_x in 0..W {
                let matched = test_cols[c][col_x] & self.cp.target_board[c][col_x];
                score += matched.count_ones() as i32;
            }
        }
        score
    }

    fn test_col_height(&self, cols: &[[u16; W]; 4], x: usize) -> usize {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & crate::constants::MASK14;
        occ.count_ones() as usize
    }

    fn draw_target_board_editable(&mut self, ui: &mut egui::Ui) {
        let cell = 16.0_f32;
        let gap = 1.0_f32;

        let width = W as f32 * cell + (W - 1) as f32 * gap;
        let height = H as f32 * cell + (H - 1) as f32 * gap;

        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
        let painter = ui.painter_at(rect);

        let palette = [
            Color32::from_rgb(239, 68, 68),
            Color32::from_rgb(34, 197, 94),
            Color32::from_rgb(59, 130, 246),
            Color32::from_rgb(234, 179, 8),
        ];

        for y in 0..H {
            for x in 0..W {
                let mut cidx: Option<usize> = None;
                let bit = 1u16 << y;
                if self.cp.target_board[0][x] & bit != 0 {
                    cidx = Some(0);
                } else if self.cp.target_board[1][x] & bit != 0 {
                    cidx = Some(1);
                } else if self.cp.target_board[2][x] & bit != 0 {
                    cidx = Some(2);
                } else if self.cp.target_board[3][x] & bit != 0 {
                    cidx = Some(3);
                }

                let fill = cidx.map(|k| palette[k]).unwrap_or(Color32::WHITE);

                let x0 = rect.min.x + x as f32 * (cell + gap);
                let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
                let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
                painter.rect_filled(r, 3.0, fill);
            }
        }

        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let rel = pos - rect.min;
                let click_x = (rel.x / (cell + gap)) as usize;
                let click_y = ((rect.max.y - pos.y) / (cell + gap)) as usize;

                if click_x < W && click_y < H {
                    self.toggle_target_cell(click_x, click_y);
                }
            }
        }
    }

    fn toggle_target_cell(&mut self, x: usize, y: usize) {
        let bit = 1u16 << y;

        let current = if self.cp.target_board[0][x] & bit != 0 {
            Some(0)
        } else if self.cp.target_board[1][x] & bit != 0 {
            Some(1)
        } else if self.cp.target_board[2][x] & bit != 0 {
            Some(2)
        } else if self.cp.target_board[3][x] & bit != 0 {
            Some(3)
        } else {
            None
        };

        for c in 0..4 {
            self.cp.target_board[c][x] &= !bit;
        }

        match current {
            None => self.cp.target_board[0][x] |= bit,
            Some(0) => self.cp.target_board[1][x] |= bit,
            Some(1) => self.cp.target_board[2][x] |= bit,
            Some(2) => self.cp.target_board[3][x] |= bit,
            Some(3) => {}
            Some(_) => {}
        }
    }

    fn draw_bruteforce_controls(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("入力と操作");
            ui.label("左クリック: A→B→…→M / 中クリック: N→X→N / 右クリック: ・（空白） / Shift+左: RGBY");

            ui.horizontal_wrapped(|ui| {
                ui.add(egui::DragValue::new(&mut self.threshold).clamp_range(1..=19).speed(0.1));
                ui.label("連鎖閾値");
                ui.add_space(8.0);
                ui.add(egui::DragValue::new(&mut self.lru_k).clamp_range(10..=1000).speed(1.0));
                ui.label("形キャッシュ上限(千)");
            });

            ui.horizontal_wrapped(|ui| {
                ui.add(egui::DragValue::new(&mut self.stop_progress_plateau).clamp_range(0.0..=1.0).speed(0.01));
                ui.label("早期終了: 進捗停滞比 (0=無効, 例 0.10)");
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.exact_four_only, "4個消しモード（5個以上で消えたら除外）");
            });

            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.profile_enabled, "計測を有効化（軽量）");
            });

            ui.horizontal(|ui| {
                if ui.add_enabled(!self.running, egui::Button::new("Run")).clicked() {
                    self.start_run();
                }
                if ui.add_enabled(self.running, egui::Button::new("Stop")).clicked() {
                    self.abort_flag.store(true, Ordering::Relaxed);
                }
            });

            ui.horizontal(|ui| {
                ui.label("出力ファイル:");
                ui.text_edit_singleline(&mut self.out_name);
                if ui.button("Browse…").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_title("保存先の選択")
                        .set_file_name(&self.out_name)
                        .save_file()
                    {
                        self.out_path = Some(path.clone());
                        self.out_name = path.file_name().unwrap_or_default().to_string_lossy().into();
                    }
                }
            });
        });
    }

    fn draw_chainplay_controls(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("連鎖生成 — 操作");
            let cur = self.cp_current_pair();
            let nxt = self.cp_next_pair();
            let dnx = self.cp_dnext_pair();

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
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                if ui.add_enabled(can_ops, egui::Button::new("ランダム配置")).clicked() {
                    self.cp_place_random();
                }
                if ui.add_enabled(can_ops, egui::Button::new("目標配置")).clicked() {
                    self.cp_place_target();
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                if ui.add_enabled(can_ops && self.cp.undo_stack.len() > 1, egui::Button::new("戻る")).clicked() {
                    self.cp_undo();
                }
                if ui.add_enabled(can_ops && self.cp.undo_stack.len() > 1, egui::Button::new("初手に戻る")).clicked() {
                    self.cp_reset_to_initial();
                }
            });
            ui.label(if self.cp.lock { "連鎖中…（操作ロック）" } else { "待機中" });
        });
    }

    fn draw_bruteforce_stats(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.label("実行・進捗");
            let pct = {
                let total = self.stats.total.to_f64().unwrap_or(0.0);
                let done = self.stats.done.to_f64().unwrap_or(0.0);
                if total > 0.0 { (done / total * 100.0).clamp(0.0, 100.0) } else { 0.0 }
            };
            ui.label(format!("進捗: {:.1}%", pct));
            ui.add(egui::ProgressBar::new((pct / 100.0) as f32).show_percentage());

            ui.add_space(4.0);
            ui.monospace(format!(
                "探索中: {} / 代表集合: {} / 出力件数: {} / 展開節点: {} / 枝刈り: {} / 総組合せ(厳密): {} / 完了: {} / 速度: {:.1} nodes/s / メモ: L-hit={} G-hit={} Miss={} / 形キャッシュ: 上限={} 実={}",
                if self.stats.searching { "YES" } else { "NO" },
                self.stats.unique,
                self.stats.output,
                self.stats.nodes,
                self.stats.pruned,
                &self.stats.total,
                &self.stats.done,
                self.stats.rate,
                self.stats.memo_hit_local,
                self.stats.memo_hit_global,
                self.stats.memo_miss,
                self.stats.lru_limit,
                self.stats.memo_len
            ));
        });
    }

    fn draw_bruteforce_board(&mut self, ui: &mut egui::Ui) {
        ui.label("盤面（左: A→B→…→M / 中: N↔X / 右: ・ / Shift+左: RGBY）");
        ui.add_space(6.0);

        let cell_size = Vec2::new(28.0, 28.0);
        let gap = 2.0;

        let shift_pressed = ui.input(|i| i.modifiers.shift);
        for y in (0..H).rev() {
            ui.horizontal(|ui| {
                for x in 0..W {
                    let i = y * W + x;
                    let (text, fill, stroke) = cell_style(self.board[i]);
                    let btn = egui::Button::new(RichText::new(text).size(12.0))
                        .min_size(cell_size)
                        .fill(fill)
                        .stroke(stroke);
                    let resp = ui.add(btn);
                    if resp.clicked_by(egui::PointerButton::Primary) {
                        if shift_pressed {
                            self.board[i] = cycle_fixed(self.board[i]);
                        } else {
                            self.board[i] = cycle_abs(self.board[i]);
                        }
                    }
                    if resp.clicked_by(egui::PointerButton::Middle) {
                        self.board[i] = cycle_any(self.board[i]);
                    }
                    if resp.clicked_by(egui::PointerButton::Secondary) {
                        self.board[i] = Cell::Blank;
                    }
                    ui.add_space(gap);
                }
            });
            ui.add_space(gap);
        }
    }

    fn draw_chainplay_board(&mut self, ui: &mut egui::Ui) {
        ui.label("連鎖生成 — 実盤面");
        ui.add_space(6.0);
        draw_preview(ui, &self.cp.cols);
        
        ui.add_space(12.0);
        ui.label("目標盤面（クリックで R→G→B→Y→空白 の順で遷移）");
        ui.add_space(6.0);
        self.draw_target_board_editable(ui);
    }

    fn start_run(&mut self) {
        let threshold = self.threshold.clamp(1, 19);
        let lru_limit = (self.lru_k.clamp(10, 1000) as usize) * 1000;
        let outfile = if let Some(p) = &self.out_path {
            p.clone()
        } else {
            std::path::PathBuf::from(&self.out_name)
        };
        let board_chars: Vec<char> = self.board.iter().map(|c| c.label_char()).collect();

        let (tx, rx) = unbounded::<Message>();
        self.rx = Some(rx);
        self.running = true;
        self.preview = None;
        self.log_lines.clear();
        self.stats.profile = ProfileTotals::default();

        let abort = self.abort_flag.clone();
        abort.store(false, Ordering::Relaxed);

        let stop_progress_plateau = self.stop_progress_plateau.clamp(0.0, 1.0);
        let exact_four_only = self.exact_four_only;
        let profile_enabled = self.profile_enabled;

        self.push_log(format!(
            "出力: JSONL / 形キャッシュ上限 ≈ {} 形 / 保存先: {} / 進捗停滞比={:.2} / 4個消しモード={} / 計測={}",
            lru_limit,
            outfile.display(),
            stop_progress_plateau,
            if exact_four_only { "ON" } else { "OFF" },
            if profile_enabled { "ON" } else { "OFF" },
        ));

        thread::spawn(move || {
            if let Err(e) = run_search(
                board_chars,
                threshold,
                lru_limit,
                outfile,
                tx.clone(),
                abort,
                stop_progress_plateau,
                exact_four_only,
                profile_enabled,
            ) {
                let _ = tx.send(Message::Error(format!("{e:?}")));
            }
        });
    }
}
