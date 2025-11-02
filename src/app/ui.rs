// GUI実装

use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;
use crossbeam_channel::unbounded;
use egui::{Color32, RichText, Vec2};
use num_traits::ToPrimitive;

use crate::constants::{W, H};
use crate::model::{Cell, cell_style, cycle_abs, cycle_any, cycle_fixed};
use crate::app::{App, Mode, Message};
use crate::app::ui_helpers::{draw_pair_preview, draw_preview, show_profile_table, COLOR_PALETTE};
use crate::app::board_evaluation::BoardEvaluator;
use crate::app::bruteforce_search::BruteforceSearch;
use crate::app::chain_operations::ChainOperations;
use crate::profiling::{ProfileTotals, has_profile_any};
use crate::search::run_search;
use crate::vlog;

// ヘルパー関数はui_helpersモジュールに移動しました

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
            ChainOperations::step_animation(&mut self.cp);
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
    // 連鎖操作関連メソッドは chain_operations.rs に移動しました

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
                let around_first = BoardEvaluator::collect_empty_cells_bfs(&self.cp.target_board, &first_chain_cells, 7);
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

    // 総当たり探索関連メソッドは bruteforce_search.rs に移動しました

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
        
        let results_raw = BruteforceSearch::search_all_colors(
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

    // これらのメソッドは chain_operations.rs に移動しました
    // - cp_detect_target_chain, cp_place_with, cp_col_height, cp_set_cell
    // - cp_check_and_start_chain, cp_step_animation, cp_place_target

    // eval_move と test_col_height は board_evaluation.rs に移動しました

    fn draw_target_board_editable(&mut self, ui: &mut egui::Ui) {
        let cell = 16.0_f32;
        let gap = 1.0_f32;

        let width = W as f32 * cell + (W - 1) as f32 * gap;
        let height = H as f32 * cell + (H - 1) as f32 * gap;

        let (rect, response) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::click());
        let painter = ui.painter_at(rect);

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

    // is_placeable_empty と collect_empty_cells_bfs は board_evaluation.rs に移動しました

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
            let cur = ChainOperations::current_pair(&self.cp);
            let nxt = ChainOperations::next_pair(&self.cp);
            let dnx = ChainOperations::dnext_pair(&self.cp);

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
                    if let Err(e) = ChainOperations::place_random(&mut self.cp) {
                        self.push_log(e);
                    }
                }
                if ui.add_enabled(can_ops, egui::Button::new("目標配置")).clicked() {
                    let target_board = self.cp.target_board.clone();
                    match ChainOperations::place_target(&mut self.cp, &target_board) {
                        Ok(_) => {},
                        Err(e) => self.push_log(e),
                    }
                }
                if ui.add_enabled(can_ops, egui::Button::new("目標盤面更新")).clicked() {
                    let beam_width = self.cp.beam_width;
                    let max_depth = self.cp.max_depth as u8;
                    let msg = ChainOperations::update_target(&mut self.cp, beam_width, max_depth);
                    self.push_log(msg);
                }
            });
            ui.horizontal(|ui| {
                let can_ops = !self.cp.lock && self.cp.anim.is_none();
                if ui.add_enabled(can_ops, egui::Button::new("連鎖検出（目標盤面）")).clicked() {
                    match ChainOperations::detect_target_chain(&mut self.cp) {
                        Ok(msg) => self.push_log(msg),
                        Err(e) => self.push_log(e),
                    }
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
                    ChainOperations::undo(&mut self.cp);
                }
                if ui.add_enabled(can_ops && self.cp.undo_stack.len() > 1, egui::Button::new("初手に戻る")).clicked() {
                    ChainOperations::reset_to_initial(&mut self.cp);
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
