// GUI実装

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use crossbeam_channel::unbounded;
use egui::{Color32, RichText, Vec2};
use rand::Rng;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use std::collections::HashSet;

use crate::constants::{W, H};
use crate::model::{Cell, cell_style, cycle_abs, cycle_any, cycle_fixed};
use crate::app::{App, Mode, Message, Stats};
use crate::app::chain_play::{ChainPlay, SavedState, AnimState, AnimPhase, Orient};
use crate::profiling::{ProfileTotals, has_profile_any, fmt_dur_ms};
use crate::search::run_search;
use crate::search::board::{apply_clear_no_fall, apply_erase_and_fall_cols};
use crate::search::hash::canonical_hash64_fast;
use crate::vlog;

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
        // 初回起動時にverbose loggingの状態を同期
        static INITIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !INITIALIZED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            // ログファイルを初期化（カレントディレクトリに出力）
            let log_path = "debug_log.txt";
            
            // カレントディレクトリを取得して絶対パスを構築
            let full_path = std::env::current_dir()
                .ok()
                .and_then(|p| Some(p.join(log_path)))
                .unwrap_or_else(|| std::path::PathBuf::from(log_path));
            
            if let Err(e) = crate::logging::init_log_file(log_path) {
                eprintln!("ログファイルの初期化に失敗: {}", e);
            } else {
                println!("デバッグログを {} に出力します", full_path.display());
            }
            
            if self.verbose_logging {
                crate::logging::enable_verbose_logging();
            } else {
                crate::logging::disable_verbose_logging();
            }
        }
        
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

                ui.label("ログ");
                for line in &self.log_lines {
                    ui.monospace(line);
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

    fn cp_update_target(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }
        // 実盤面から右盤面相当（目標盤面）を生成
        let bw = self.cp.beam_width.max(1);
        let md_u8 = (self.cp.max_depth.max(1).min(255)) as u8;
        let (target, chain) = crate::chain::compute_target_from_actual_with_params(&self.cp.cols, bw, md_u8);
        self.cp.target_board = target;
        self.push_log(format!(
            "目標盤面を更新しました（推定最大連鎖: {} / ビーム幅: {} / 最大深さ: {}）",
            chain, bw, md_u8
        ));
    }

    fn cp_remove_first_chain_freetop(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }
        
        // 連鎖情報がない場合は何もしない
        let Some(ref chain_info) = self.cp.target_chain_info else {
            self.push_log("連鎖情報がありません。先に「連鎖検出（目標盤面）」を実行してください".into());
            return;
        };
        
        if chain_info.is_empty() {
            self.push_log("連鎖情報が空です".into());
            return;
        }
        
        // 1連鎖目で消えるマスを取得
        let first_chain_cells: std::collections::HashSet<(usize, usize)> = 
            chain_info[0].original_groups.iter()
                .flat_map(|group| group.iter().copied())
                .collect();
        
        // 各列の1連鎖目ぷよの最上部を取得
        let mut col_max_y = std::collections::HashMap::new();
        for &(x, y) in &first_chain_cells {
            col_max_y.entry(x)
                .and_modify(|max_y| { if y > *max_y { *max_y = y; } })
                .or_insert(y);
        }
        
        // 盤面を取得して、その上にぷよがないかチェック
        let board = crate::chain::cols_to_board(&self.cp.target_board);
        
        // フリートップ列（1連鎖目の最上部の上にぷよがない列）を特定
        let mut freetop_cols: Vec<(usize, usize)> = Vec::new();
        for (&x, &max_y) in col_max_y.iter() {
            // この列の最上部の上にぷよがあるかチェック
            let mut has_puyo_above = false;
            for y in (max_y + 1)..crate::constants::H {
                if board[y][x].is_some() {
                    has_puyo_above = true;
                    break;
                }
            }
            
            // 上にぷよがない = フリートップ列
            if !has_puyo_above {
                freetop_cols.push((x, max_y));
            }
        }
        
        if freetop_cols.is_empty() {
            self.push_log("1連鎖目にフリートップ列がありません（全ての列で上にぷよが乗っています）".into());
            return;
        }
        
        // フリートップ列を左から選ぶ
        freetop_cols.sort_by_key(|&(x, _)| x);
        let (remove_col, remove_y) = freetop_cols[0];
        
        vlog!("[フリートップ削除] 候補: {:?}, 選択: ({}, {})", freetop_cols, remove_col, remove_y);
        
        // 目標盤面から該当マスを削除
        let bit = 1u16 << remove_y;
        for c in 0..4 {
            self.cp.target_board[c][remove_col] &= !bit;
        }
        
        // 削除した位置を保存（頭伸ばし用）
        self.cp.removed_freetop = Some((remove_col, remove_y));
        
        // 連鎖情報は保持したまま、周囲7マスに更新（2マス分を頭伸ばし用に残す）
        if let Some(ref chain_info) = self.cp.target_chain_info {
            if !chain_info.is_empty() {
                use std::collections::HashSet;
                let mut first_chain_cells = HashSet::new();
                
                // 1連鎖目（削除後も残っているマス）
                for group in &chain_info[0].original_groups {
                    for &(x, y) in group {
                        // 削除したマスを除外
                        if x != remove_col || y != remove_y {
                            first_chain_cells.insert((x, y));
                        }
                    }
                }
                
                // 周囲7マスを収集（削除したマスとその上の2マスを除く）
                let around_first = self.collect_empty_cells_bfs(&first_chain_cells, 7);
                if let Some((_, ref last)) = self.cp.around_cells_cache {
                    self.cp.around_cells_cache = Some((around_first, last.clone()));
                }
            }
        }
        
        self.push_log(format!(
            "1連鎖目のフリートップを削除: 列={}, y={} (頭伸ばし準備完了)",
            remove_col, remove_y
        ));
    }

    /// 二項係数 nCk を計算
    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }
        let k = k.min(n - k);
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// 候補位置に各色のぷよを配置する総当たり（重力適用後に連鎖検出）
    /// 各位置に0〜3の色または空欄を割り当てる全組み合わせを試す（5^N通り）
    /// フリートップ列が1列以上残っている配置のみ有効
    /// 並列化版：先頭の候補で分岐して並列処理
    fn try_add_puyos_bruteforce_all_colors(
        &self,
        base_board: &crate::chain::Board,
        candidates: &[(usize, usize)],
        baseline: i32,
        max_results: usize,
        first_chain_top: &std::collections::HashMap<usize, usize>,
    ) -> Vec<(i32, crate::chain::Board, Vec<(usize, usize, u8)>)> {
        if candidates.is_empty() {
            return Vec::new();
        }
        
        // 候補が少ない場合は単一スレッドで実行
        if candidates.len() <= 3 {
            return self.try_add_puyos_bruteforce_single_thread(
                base_board, candidates, baseline, max_results, first_chain_top
            );
        }
        
        // 並列化：先頭の候補位置で分岐（5通り：空欄 + 4色）
        let first_pos = candidates[0];
        let rest_candidates = &candidates[1..];
        
        vlog!("  [並列総当たり] 先頭={:?}, 残り={} 候補", first_pos, rest_candidates.len());
        
        // 5通りの初期配置を並列処理（重複排除付き）
        let initial_placements: Vec<Option<u8>> = vec![None, Some(0), Some(1), Some(2), Some(3)];
        
        let all_results: Vec<Vec<(i32, crate::chain::Board, Vec<(usize, usize, u8)>)>> = initial_placements
            .par_iter()
            .map(|&first_color| {
                let mut local_placed = Vec::new();
                if let Some(color) = first_color {
                    local_placed.push((first_pos.0, first_pos.1, color));
                }
                
                let mut local_results = Vec::new();
                let mut local_trial_count = 0usize;
                let mut local_seen_hashes = HashSet::new();  // 重複排除用
                
                Self::dfs_bruteforce_deduplicated(
                    base_board,
                    rest_candidates,
                    0,
                    &mut local_placed,
                    &mut local_results,
                    baseline,
                    max_results / 5 + 1,  // 各スレッドに上限を分配
                    &mut local_trial_count,
                    first_chain_top,
                    &mut local_seen_hashes,
                );
                
                vlog!("  [並列総当たり] 初期配置={:?}: 試行数={}, 結果={}, 重複排除={}", 
                         first_color, local_trial_count, local_results.len(), local_seen_hashes.len());
                
                local_results
            })
            .collect();
        
        // 結果を統合してソート（グローバル重複排除）
        let mut global_seen = HashSet::new();
        let mut combined_results: Vec<(i32, crate::chain::Board, Vec<(usize, usize, u8)>)> = Vec::new();
        
        for result in all_results.into_iter().flatten() {
            let cols = crate::chain::board_to_cols(&result.1);
            let hash = Self::compute_board_hash(&cols);
            
            if global_seen.insert(hash) {
                combined_results.push(result);
            }
        }
        
        combined_results.sort_by(|a, b| b.0.cmp(&a.0));  // 連鎖数降順
        
        if combined_results.len() > max_results {
            combined_results.truncate(max_results);
        }
        
        vlog!("  [並列総当たり] 統合完了: 結果={} 件", combined_results.len());
        
        combined_results
    }
    
    /// 単一スレッド版の総当たり（候補数が少ない場合用）
    fn try_add_puyos_bruteforce_single_thread(
        &self,
        base_board: &crate::chain::Board,
        candidates: &[(usize, usize)],
        baseline: i32,
        max_results: usize,
        first_chain_top: &std::collections::HashMap<usize, usize>,
    ) -> Vec<(i32, crate::chain::Board, Vec<(usize, usize, u8)>)> {
        let mut results = Vec::new();
        let mut trial_count = 0usize;
        let mut placed = Vec::new();
        let mut seen_hashes = HashSet::new();
        
        Self::dfs_bruteforce_deduplicated(
            base_board, candidates, 0, &mut placed, &mut results, 
            baseline, max_results, &mut trial_count, first_chain_top, &mut seen_hashes
        );
        
        vlog!("  [単一総当たり] dfs完了: 試行数={}, ベースライン超え={} 件, 重複排除={}", 
                 trial_count, results.len(), seen_hashes.len());
        
        results
    }
    
    /// 盤面のハッシュ値を計算（正規化版：ミラー対称を考慮）
    fn compute_board_hash(cols: &[[u16; W]; 4]) -> u64 {
        // 総当たりタブと同じ正規化ハッシュを使用
        // ミラー対称の盤面は同じハッシュになる
        let (hash, _mirror) = canonical_hash64_fast(cols);
        hash
    }
    
    /// DFS本体（重複排除版）
    fn dfs_bruteforce_deduplicated(
        base_board: &crate::chain::Board,
        candidates: &[(usize, usize)],
        idx: usize,
        placed: &mut Vec<(usize, usize, u8)>,
        results: &mut Vec<(i32, crate::chain::Board, Vec<(usize, usize, u8)>)>,
        baseline: i32,
        max_results: usize,
        trial_count: &mut usize,
        first_chain_top: &std::collections::HashMap<usize, usize>,
        seen_hashes: &mut HashSet<u64>,
    ) {
            if idx == candidates.len() {
                // 全位置に色を割り当て完了
                
                // 空の配置は無効
                if placed.is_empty() {
                    return;
                }
                
                // フリートップ制約チェック：配置するぷよは各列の最上部でなければならない
                let mut board = base_board.clone();
                
                // 列ごとの最上部の高さを計算
                let mut col_top: [Option<usize>; 6] = [None; 6];
                for x in 0..crate::constants::W {
                    for y in (0..crate::constants::H).rev() {
                        if board[y][x].is_some() {
                            col_top[x] = Some(y + 1); // 最上部の1つ上
                            break;
                        }
                    }
                    if col_top[x].is_none() {
                        col_top[x] = Some(0); // 空列
                    }
                }
                
                // 配置するぷよが各列のフリートップかチェック
                let mut col_expected_top = col_top.clone();
                let mut valid = true;
                
                // y座標が小さい順（底から上）にソート
                let mut sorted_placed = placed.clone();
                sorted_placed.sort_by(|a, b| a.1.cmp(&b.1));
                
                for &(x, y, _color) in sorted_placed.iter() {
                    if let Some(expected_y) = col_expected_top[x] {
                        if y != expected_y {
                            // フリートップでない位置に配置しようとしている
                            valid = false;
                            break;
                        }
                        // 次のフリートップを更新
                        col_expected_top[x] = Some(expected_y + 1);
                    } else {
                        valid = false;
                        break;
                    }
                }
                
                if !valid {
                    return; // この配置は無効
                }
                
                // 有効な配置のみ実行（配置後に重力適用して連鎖シミュレーション）
                for &(x, y, color) in placed.iter() {
                    if board[y][x].is_none() {
                        board[y][x] = Some(crate::chain::CellData {
                            color,
                            iter: crate::chain::IterId(0),
                            original_pos: Some((x, y)),
                        });
                    }
                }
                
                // 重力適用
                crate::chain::apply_gravity(&mut board);
                
                // 連鎖検出
                let mut detector = crate::chain::Detector::new(board.clone());
                let chain = detector.simulate_chain();
                
                // フリートップ列チェック：配置後に1連鎖目が発生し、フリートップ列が1列以上あるか
                if chain > 0 && !detector.chain_history.is_empty() {
                    // 1連鎖目に消えるぷよの位置を取得
                    let mut new_first_chain_puyos = std::collections::HashSet::new();
                    for group in &detector.chain_history[0].original_groups {
                        for &(x, y) in group {
                            new_first_chain_puyos.insert((x, y));
                        }
                    }
                    
                    // 各列の1連鎖目ぷよの最上部を計算
                    let mut new_first_chain_top: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
                    for &(x, y) in new_first_chain_puyos.iter() {
                        new_first_chain_top.entry(x).and_modify(|max_y| *max_y = (*max_y).max(y)).or_insert(y);
                    }
                    
                    // フリートップ列が1列以上あるかチェック
                    let mut has_freetop = false;
                    for (&x, &first_chain_max_y) in new_first_chain_top.iter() {
                        let freetop_start_y = first_chain_max_y + 1;
                        
                        // その上にぷよがあるか
                        let mut has_puyo_above = false;
                        for y in freetop_start_y..crate::constants::H {
                            if board[y][x].is_some() {
                                has_puyo_above = true;
                                break;
                            }
                        }
                        
                        // その上に何もない = フリートップ列
                        if !has_puyo_above {
                            // さらに、候補位置内にfreetop_start_yがあるか
                            for &(cx, cy) in candidates {
                                if cx == x && cy == freetop_start_y {
                                    has_freetop = true;
                                    break;
                                }
                            }
                        }
                        
                        if has_freetop {
                            break;
                        }
                    }
                    
                    if !has_freetop {
                        return; // フリートップ列がない配置は無効
                    }
                } else {
                    // 連鎖が発生しない場合は無効
                    return;
                }
                
                *trial_count += 1;
                
                // デバッグ：全ての結果をログ出力（最初の10件のみ）
                if *trial_count <= 10 {
                    vlog!("  [総当たり] trial={}, placed={:?}, chain={} (baseline={})", 
                             trial_count, placed, chain, baseline);
                } else if *trial_count % 10000 == 0 {
                    vlog!("  [総当たり] 試行中... {} / ? 件", trial_count);
                }
                
                // デバッグ：baseline以上（=含む）の結果を最初の5件表示
                if chain >= baseline && results.len() < 5 {
                    vlog!("  [総当たり] ★baseline以上: trial={}, chain={}, placed={:?}", 
                             trial_count, chain, placed);
                }
                
                // ベースラインを超える場合のみ結果に追加（重複チェック付き）
                if chain > baseline {
                    let cols = crate::chain::board_to_cols(&board);
                    let hash = Self::compute_board_hash(&cols);
                    
                    if seen_hashes.insert(hash) {
                        results.push((chain, board, placed.clone()));
                        if results.len() >= max_results {
                            return; // 上限到達で打ち切り
                        }
                    }
                }
                return;
            }
            
            // 現在の位置に各色を試す（空欄も含む）
            let (x, y) = candidates[idx];
            
            // 早期枝刈り：結果が上限に達したら打ち切り
            if results.len() >= max_results {
                return;
            }
            
            // 空欄を試す
            Self::dfs_bruteforce_deduplicated(base_board, candidates, idx + 1, placed, results, baseline, max_results, trial_count, first_chain_top, seen_hashes);
            if results.len() >= max_results {
                return;
            }
            
            // 各色を試す
            for color in 0..4u8 {
                placed.push((x, y, color));
                Self::dfs_bruteforce_deduplicated(base_board, candidates, idx + 1, placed, results, baseline, max_results, trial_count, first_chain_top, seen_hashes);
                placed.pop();
                
                if results.len() >= max_results {
                    return; // 上限到達で打ち切り
                }
            }
    }

    fn cp_extend_head(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }
        
        // 連鎖情報とフリートップ削除情報が必要
        let Some(ref chain_info) = self.cp.target_chain_info else {
            self.push_log("連鎖情報がありません。先に「連鎖検出（目標盤面）」→「1連鎖目フリートップ削除」を実行してください".into());
            return;
        };
        
        let Some((rx, ry)) = self.cp.removed_freetop else {
            self.push_log("フリートップ削除がされていません。先に「1連鎖目フリートップ削除」を実行してください".into());
            return;
        };
        
        // ベースライン：削除前の連鎖数（目標は削除前を超えること）
        let baseline_before_remove = chain_info.len() as i32;
        
        // 削除後の盤面での連鎖数も確認
        let base_board = crate::chain::cols_to_board(&self.cp.target_board);
        let mut baseline_detector = crate::chain::Detector::new(base_board.clone());
        let baseline_after_remove = baseline_detector.simulate_chain() as i32;
        
        vlog!("[頭伸ばし] 削除前の連鎖数={}, 削除後の連鎖数={}", 
                 baseline_before_remove, baseline_after_remove);
        
        // 目標は削除前の連鎖数を超えること
        let baseline = baseline_before_remove;
        
        // 1連鎖目に消えるぷよの位置を特定（削除前の情報）
        let mut first_chain_puyos = std::collections::HashSet::new();
        for group in &chain_info[0].original_groups {
            for &(x, y) in group {
                first_chain_puyos.insert((x, y));
            }
        }
        
        // 各列の1連鎖目ぷよの最上部を計算
        let mut first_chain_top: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(x, y) in first_chain_puyos.iter() {
            first_chain_top.entry(x).and_modify(|max_y| *max_y = (*max_y).max(y)).or_insert(y);
        }
        
        vlog!("[頭伸ばし] 削除前の1連鎖目の列と最上部={:?}", {
            let mut v: Vec<_> = first_chain_top.iter().map(|(x, y)| (*x, *y)).collect();
            v.sort_unstable();
            v
        });
        
        // 候補位置：周囲9マス（削除前の周囲マス + 削除したマス）
        let Some((ref around_first_before_remove, _)) = self.cp.around_cells_cache else {
            self.push_log("周囲マス情報がありません".into());
            return;
        };
        
        // 削除前の周囲9マスを使用し、削除後の盤面で空マスのみフィルタ
        let mut candidate_positions = std::collections::HashSet::new();
        for &(x, y) in around_first_before_remove.iter() {
            // 削除後の盤面で空マスか確認
            if base_board[y][x].is_none() {
                candidate_positions.insert((x, y));
            }
        }
        
        // 削除したマスも候補に追加（最重要！）
        candidate_positions.insert((rx, ry));
        
        // 削除マスの上も候補に追加（9マス目）
        if ry + 1 < crate::constants::H && base_board[ry + 1][rx].is_none() {
            candidate_positions.insert((rx, ry + 1));
        }
        
        vlog!("[頭伸ばし] 削除前周囲マス数={}, 削除マス含む候補位置数={}", 
                 around_first_before_remove.len(), candidate_positions.len());
        
        // 候補列を収集
        let mut cand_cols = std::collections::HashSet::new();
        for &(x, _) in candidate_positions.iter() {
            cand_cols.insert(x);
        }
        
        vlog!("[頭伸ばし] 候補位置数={} 候補列={:?}", candidate_positions.len(), {
            let mut v: Vec<_> = cand_cols.iter().copied().collect();
            v.sort_unstable();
            v
        });
        vlog!("[頭伸ばし] 削除マス=({}, {})", rx, ry);
        
        self.push_log(format!(
            "頭伸ばし開始: ベースライン={}連鎖, 候補位置={}個, 候補列数={}",
            baseline, candidate_positions.len(), cand_cols.len()
        ));
        
        // 候補位置を配列に変換
        let mut cand_positions: Vec<(usize, usize)> = candidate_positions.iter().copied().collect();
        cand_positions.sort();
        
        vlog!("[頭伸ばし] 候補位置={:?}", cand_positions);
        
        // 完全総当たり：各位置に4色または空欄を配置（5^N通り）
        let total_combinations = 5usize.pow(cand_positions.len() as u32);
        vlog!("[頭伸ばし] 総当たり開始: {}^{} = {} 通り（空欄含む）", 5, cand_positions.len(), total_combinations);
        
        let max_results = if total_combinations > 1_000_000 {
            self.push_log(format!(
                "警告: 総当たり数が多すぎます（{}通り）。結果を100件に制限します。",
                total_combinations
            ));
            100
        } else {
            total_combinations
        };
        
        let results_raw = self.try_add_puyos_bruteforce_all_colors(
            &base_board,
            &cand_positions,
            baseline,
            max_results,
            &first_chain_top,
        );
        
        let mut all_results: Vec<(i32, crate::chain::Board, (usize, usize))> = results_raw.into_iter()
            .map(|(chain, board, placed)| {
                let seed = placed.first().map(|(x, y, _)| (*x, *y)).unwrap_or((0, 0));
                (chain, board, seed)
            })
            .collect();
        
        // 連鎖数降順でソート
        all_results.sort_by(|a, b| b.0.cmp(&a.0));
        let results = all_results;
        
        vlog!("[頭伸ばし] find_best_arrangement: results={} 件", results.len());
        
        // 結果の詳細をログ出力（最初の10件）
        for (i, (ch, _, seed)) in results.iter().enumerate().take(10) {
            vlog!("[頭伸ばし] result[{}]: chain={}, seed={:?}", i, ch, seed);
        }
        
        // 連鎖が伸びる最良案を採用
        let mut adopted: Option<(i32, crate::chain::Board)> = None;
        for (ch, board_pre, seed) in results {
            vlog!("[頭伸ばし] 候補確認: chain={} vs baseline={} seed={:?}", ch, baseline, seed);
            if ch > baseline {
                vlog!("[頭伸ばし] 採用: chain={} > baseline={} seed={:?}", ch, baseline, seed);
                adopted = Some((ch, board_pre));
                break;
            }
        }
        
        if let Some((best_chain, board_pre)) = adopted {
            let new_cols = crate::chain::board_to_cols(&board_pre);
            
            // 更新前のハッシュを計算（デバッグ用）
            let old_hash: u64 = self.cp.target_board.iter()
                .flat_map(|color_col| color_col.iter())
                .fold(0u64, |acc, &val| acc.wrapping_mul(31).wrapping_add(val as u64));
            
            self.cp.target_board = new_cols;
            
            // 更新後のハッシュを計算（デバッグ用）
            let new_hash: u64 = self.cp.target_board.iter()
                .flat_map(|color_col| color_col.iter())
                .fold(0u64, |acc, &val| acc.wrapping_mul(31).wrapping_add(val as u64));
            
            vlog!("[頭伸ばし] target_board更新: old_hash={}, new_hash={}", old_hash, new_hash);
            
            // キャッシュクリア
            self.cp.target_chain_info = None;
            self.cp.around_cells_cache = None;
            self.cp.removed_freetop = None;
            self.push_log(format!(
                "頭伸ばし成功: {}連鎖 → {}連鎖（候補列={:?}、削除列={}）★目標盤面を更新しました",
                baseline, best_chain,
                {
                    let mut v: Vec<_> = cand_cols.iter().copied().collect();
                    v.sort_unstable(); v
                }, rx
            ));
        } else {
            vlog!("[頭伸ばし] 不採用: baseline={} を超える候補なし", baseline);
            self.push_log(format!("頭伸ばし失敗: 連鎖は{}連鎖のまま", baseline));
        }
    }

    fn cp_detect_target_chain(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() {
            return;
        }
        
        // 目標盤面が空かチェック
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
        
        // 目標盤面をBoardに変換
        let board = crate::chain::cols_to_board(&self.cp.target_board);
        let mut detector = crate::chain::Detector::new(board);
        let chain_count = detector.simulate_chain();
        
        // 連鎖検出時はフリートップ削除をリセット
        self.cp.removed_freetop = None;
        
        if chain_count > 0 {
            self.cp.target_chain_info = Some(detector.chain_history.clone());
            
            // 周囲9マスを計算してキャッシュ
            use std::collections::HashSet;
            let mut first_chain_cells = HashSet::new();
            let mut last_chain_cells = HashSet::new();
            
            // 1連鎖目（元の盤面での位置）
            for group in &detector.chain_history[0].original_groups {
                for &(x, y) in group {
                    first_chain_cells.insert((x, y));
                }
            }
            
            // 最終連鎖（元の盤面での位置）
            let last_idx = detector.chain_history.len() - 1;
            for group in &detector.chain_history[last_idx].original_groups {
                for &(x, y) in group {
                    last_chain_cells.insert((x, y));
                }
            }
            
            // BFSで周囲9マスを収集（頭伸ばし用）
            let around_first = self.collect_empty_cells_bfs(&first_chain_cells, 9);
            let around_last = self.collect_empty_cells_bfs(&last_chain_cells, 9);
            self.cp.around_cells_cache = Some((around_first, around_last));
            
            // デバッグ: 各連鎖の情報をログ出力
            for (idx, step) in detector.chain_history.iter().enumerate() {
                self.push_log(format!(
                    "  連鎖[idx={}] chain_num={} グループ数={}",
                    idx, step.chain_num, step.groups.len()
                ));
            }
            
            self.push_log(format!(
                "目標盤面の連鎖を検出: {}連鎖（1連鎖目=黄色, {}連鎖目=オレンジ）",
                chain_count, chain_count
            ));
        } else {
            self.cp.target_chain_info = None;
            self.cp.around_cells_cache = None;
            self.push_log("目標盤面で連鎖が発生しませんでした".into());
        }
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
        
        // デバッグ：現在のtarget_boardのハッシュを出力
        let current_hash: u64 = self.cp.target_board.iter()
            .flat_map(|color_col| color_col.iter())
            .fold(0u64, |acc, &val| acc.wrapping_mul(31).wrapping_add(val as u64));
        vlog!("[目標配置] 参照するtarget_boardのハッシュ={}", current_hash);

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
        
        // 配置する2つのぷよの位置と色を記録
        let mut placed_puyos: Vec<(usize, usize, u8)> = Vec::new();

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
                placed_puyos.push((x, h, pair.0));
                placed_puyos.push((x, h + 1, pair.1));
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
                placed_puyos.push((x, h, pair.1));
                placed_puyos.push((x, h + 1, pair.0));
            }
            Orient::Right => {
                let h0 = self.test_col_height(&test_cols, x);
                let h1 = self.test_col_height(&test_cols, x + 1);
                if h0 >= H || h1 >= H || x + 1 >= W {
                    return -1000;
                }
                test_cols[pair.0 as usize][x] |= 1u16 << h0;
                test_cols[pair.1 as usize][x + 1] |= 1u16 << h1;
                placed_puyos.push((x, h0, pair.0));
                placed_puyos.push((x + 1, h1, pair.1));
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
                placed_puyos.push((x, h0, pair.0));
                placed_puyos.push((x - 1, h1, pair.1));
            }
        }

        let mut score = 0i32;
        for c in 0..4 {
            for col_x in 0..W {
                let matched = test_cols[c][col_x] & self.cp.target_board[c][col_x];
                score += matched.count_ones() as i32;
            }
        }
        
        // 配置したぷよが目標盤面の別の色を上書きしていないかチェック
        for (px, py, color) in placed_puyos {
            let bit = 1u16 << py;
            // この位置に目標盤面でぷよがあるか確認
            for target_color in 0..4u8 {
                if (self.cp.target_board[target_color as usize][px] & bit) != 0 {
                    // 目標盤面にぷよがある
                    if target_color != color {
                        // 違う色を上書きしている → 大幅減点
                        score -= 100;
                    }
                    break;
                }
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

        // 強調表示対象のマスを収集
        use std::collections::HashSet;
        let mut first_chain_cells = HashSet::new();
        let mut last_chain_cells = HashSet::new();
        let around_first_9_cells;
        let around_last_9_cells;
        
        if let Some(ref chain_info) = self.cp.target_chain_info {
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
                if let Some((ref first, ref last)) = self.cp.around_cells_cache {
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
                if self.cp.target_board[0][x] & bit != 0 {
                    cidx = Some(0);
                } else if self.cp.target_board[1][x] & bit != 0 {
                    cidx = Some(1);
                } else if self.cp.target_board[2][x] & bit != 0 {
                    cidx = Some(2);
                } else if self.cp.target_board[3][x] & bit != 0 {
                    cidx = Some(3);
                }

                let mut fill = cidx.map(|k| palette[k]).unwrap_or(Color32::WHITE);
                
                // 強調表示の適用
                let x0 = rect.min.x + x as f32 * (cell + gap);
                let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
                let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
                
                // まず通常の色で塗る
                painter.rect_filled(r, 3.0, fill);
                
                // 強調表示のオーバーレイ（優先順位: 実際の連鎖セル > 周囲マス）
                if first_chain_cells.contains(&(x, y)) {
                    // 1連鎖目: 黄色の太枠
                    painter.rect_stroke(r, 3.0, egui::Stroke::new(3.0, Color32::from_rgb(255, 215, 0)));
                } else if last_chain_cells.contains(&(x, y)) {
                    // 最終連鎖: オレンジの太枠
                    painter.rect_stroke(r, 3.0, egui::Stroke::new(3.0, Color32::from_rgb(255, 140, 0)));
                } else if around_first_9_cells.contains(&(x, y)) {
                    // 1連鎖目の周囲9マス: 薄い黄色の細枠
                    painter.rect_stroke(r, 3.0, egui::Stroke::new(1.5, Color32::from_rgb(255, 255, 150)));
                } else if around_last_9_cells.contains(&(x, y)) {
                    // 最終連鎖の周囲9マス: 薄いオレンジの細枠
                    painter.rect_stroke(r, 3.0, egui::Stroke::new(1.5, Color32::from_rgb(255, 200, 150)));
                }
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

    /// 空マスが床または既存ぷよ（または収集済み空マス/連鎖マス）と隣接しているかチェック
    fn is_placeable_empty(&self, x: usize, y: usize, collected: &std::collections::HashSet<(usize, usize)>, chain_cells: &std::collections::HashSet<(usize, usize)>) -> bool {
        // 空マスでない場合はfalse
        let bit = 1u16 << y;
        let is_empty = self.cp.target_board[0][x] & bit == 0
            && self.cp.target_board[1][x] & bit == 0
            && self.cp.target_board[2][x] & bit == 0
            && self.cp.target_board[3][x] & bit == 0;
        
        if !is_empty {
            return false;
        }
        
        // 床（y=0）の場合は常にtrue
        if y == 0 {
            return true;
        }
        
        // 下に既存ぷよがあるかチェック
        let bit_below = 1u16 << (y - 1);
        let has_puyo_below = self.cp.target_board[0][x] & bit_below != 0
            || self.cp.target_board[1][x] & bit_below != 0
            || self.cp.target_board[2][x] & bit_below != 0
            || self.cp.target_board[3][x] & bit_below != 0;
        
        // または下に収集済み空マス/連鎖マスがあるか
        let has_collected_below = collected.contains(&(x, y - 1)) || chain_cells.contains(&(x, y - 1));
        
        has_puyo_below || has_collected_below
    }

    /// BFSで連鎖マスから隣接する「配置可能な空マス」を距離順に収集（最大max_count個）
    fn collect_empty_cells_bfs(&self, chain_cells: &std::collections::HashSet<(usize, usize)>, max_count: usize) -> std::collections::HashSet<(usize, usize)> {
        use std::collections::{VecDeque, HashSet};
        
        let mut result = HashSet::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<((usize, usize), usize)> = VecDeque::new(); // (位置, 距離)
        
        // 連鎖マスから開始（距離0）
        for &pos in chain_cells {
            visited.insert(pos);
        }
        
        vlog!("[周囲9マス探索] 連鎖マス数: {}", chain_cells.len());
        
        // 連鎖マスに隣接する配置可能な空マスをキューに追加（距離1）
        for &(cx, cy) in chain_cells {
            let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
            for (dx, dy) in dirs {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < W as isize && ny >= 0 && ny < H as isize {
                    let nxu = nx as usize;
                    let nyu = ny as usize;
                    
                    if visited.contains(&(nxu, nyu)) {
                        continue;
                    }
                    
                    // 必ずvisitedに追加（空マスでなくても）
                    visited.insert((nxu, nyu));
                    
                    // 配置可能な空マスかチェック（床または既存ぷよ/収集済みマス/連鎖マスと隣接）
                    if self.is_placeable_empty(nxu, nyu, &result, chain_cells) {
                        queue.push_back(((nxu, nyu), 1));
                        result.insert((nxu, nyu));
                        if result.len() <= 20 {
                            vlog!("[周囲9マス探索] 距離1: ({}, {}) 収集数={}", nxu, nyu, result.len());
                        }
                        
                        if result.len() >= max_count {
                            vlog!("[周囲9マス探索] 最大{}マス到達、探索終了", max_count);
                            return result;
                        }
                    }
                }
            }
        }
        
        vlog!("[周囲9マス探索] 距離1探索完了、キューサイズ={}", queue.len());
        
        // BFSで更に隣接する配置可能な空マスを探索
        let mut iteration_count = 0;
        while let Some(((cx, cy), dist)) = queue.pop_front() {
            iteration_count += 1;
            if iteration_count > 10000 {
                vlog!("[周囲9マス探索] 警告: 10000回反復、強制終了");
                break;
            }
            
            let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
            for (dx, dy) in dirs {
                let nx = cx as isize + dx;
                let ny = cy as isize + dy;
                if nx >= 0 && nx < W as isize && ny >= 0 && ny < H as isize {
                    let nxu = nx as usize;
                    let nyu = ny as usize;
                    
                    if visited.contains(&(nxu, nyu)) {
                        continue;
                    }
                    
                    // 必ずvisitedに追加（空マスでなくても）
                    visited.insert((nxu, nyu));
                    
                    // 配置可能な空マスかチェック（床または既存ぷよ/収集済みマス/連鎖マスと隣接）
                    if self.is_placeable_empty(nxu, nyu, &result, chain_cells) {
                        queue.push_back(((nxu, nyu), dist + 1));
                        result.insert((nxu, nyu));
                        if result.len() <= 20 {
                            vlog!("[周囲9マス探索] 距離{}: ({}, {}) 収集数={}", dist + 1, nxu, nyu, result.len());
                        }
                        
                        if result.len() >= max_count {
                            vlog!("[周囲9マス探索] 最大{}マス到達、探索終了", max_count);
                            return result;
                        }
                    }
                }
            }
        }
        
        vlog!("[周囲9マス探索] 探索完了、収集数={} 反復回数={}", result.len(), iteration_count);
        result
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
        // CPU機能検出（初回のみ）
        static CPU_INFO_SHOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !CPU_INFO_SHOWN.swap(true, std::sync::atomic::Ordering::Relaxed) {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let bmi2 = std::is_x86_feature_detected!("bmi2");
                let popcnt = std::is_x86_feature_detected!("popcnt");
                self.push_log(format!("CPU最適化: BMI2={} / POPCNT={}", 
                    if bmi2 { "有効" } else { "無効" },
                    if popcnt { "有効" } else { "無効" }
                ));
                if bmi2 {
                    self.push_log("ビットボード処理が高速化されています".to_string());
                }
            }
        }
        
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
                if ui.add_enabled(can_ops, egui::Button::new("目標盤面更新")).clicked() {
                    self.cp_update_target();
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                if ui.add_enabled(can_ops, egui::Button::new("連鎖検出（目標盤面）")).clicked() {
                    self.cp_detect_target_chain();
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                let has_chain_info = self.cp.target_chain_info.is_some();
                if ui.add_enabled(can_ops && has_chain_info, egui::Button::new("1連鎖目フリートップ削除")).clicked() {
                    self.cp_remove_first_chain_freetop();
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                let has_removed = self.cp.removed_freetop.is_some();
                if ui.add_enabled(can_ops && has_removed, egui::Button::new("頭伸ばし")).clicked() {
                    self.cp_extend_head();
                }
            });
            ui.add_space(6.0);
            ui.group(|ui| {
                ui.label("探索パラメータ（右盤面生成）");
                ui.horizontal_wrapped(|ui| {
                    ui.add(egui::DragValue::new(&mut self.cp.beam_width).clamp_range(1..=64).speed(1.0));
                    ui.label("ビーム幅");
                    ui.add_space(12.0);
                    ui.add(egui::DragValue::new(&mut self.cp.max_depth).clamp_range(1..=20).speed(1.0));
                    ui.label("最大深さ");
                });
            });
            ui.horizontal(|ui| {
                let prev = self.verbose_logging;
                ui.checkbox(&mut self.verbose_logging, "詳細ログ出力");
                if prev != self.verbose_logging {
                    // グローバルフラグを更新
                    if self.verbose_logging {
                        crate::logging::enable_verbose_logging();
                    } else {
                        crate::logging::disable_verbose_logging();
                    }
                }
                if self.verbose_logging {
                    ui.label(RichText::new("（debug_log.txt に詳細ログを出力）").small().color(egui::Color32::GRAY));
                } else {
                    ui.label(RichText::new("（ログ出力を抑制して高速化）").small().color(egui::Color32::GRAY));
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
