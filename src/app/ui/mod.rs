// UIモジュールのエントリポイント

pub mod bruteforce;
pub mod chainplay;
pub mod helpers;

// cell_styleを公開（UI用ヘルパー）
pub use helpers::cell_style;

use crossbeam_channel::unbounded;
use std::thread;
use std::time::Duration;

use crate::app::chain_operations::ChainOperations;
use crate::app::{App, Message, Mode};
use crate::application::bruteforce::BruteforceService;
use crate::profiling::{has_profile_any, ProfileTotals};

use bruteforce::BruteforceUI;
use chainplay::ChainplayUI;

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 初回起動時にverbose loggingの状態を同期
        static INITIALIZED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
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
                        self.push_log("完了".into());
                        keep_rx = false;
                    }
                    Message::Error(e) => {
                        self.running = false;
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

        egui::SidePanel::left("left")
            .min_width(420.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing = egui::Vec2::new(8.0, 8.0);
                        if self.mode == Mode::BruteForce {
                            BruteforceUI::draw_controls(self, ui);
                        } else {
                            ChainplayUI::draw_controls(self, ui);
                        }

                        ui.separator();

                        if self.mode == Mode::BruteForce
                            && !self.running
                            && self.profile_enabled
                            && has_profile_any(&self.stats.profile)
                        {
                            ui.group(|ui| {
                                ui.label("処理時間（累積）");
                                helpers::show_profile_table(ui, &self.stats.profile);
                            });
                            ui.separator();
                        }

                        if self.mode == Mode::BruteForce {
                            ui.label("プレビュー（E1直前の落下後盤面）");
                            ui.add_space(4.0);
                            if let Some(cols) = &self.preview {
                                helpers::draw_preview(ui, cols);
                            } else {
                                ui.label(
                                    egui::RichText::new("（実行中に更新表示）")
                                        .italics()
                                        .color(egui::Color32::GRAY),
                                );
                            }
                        }

                        ui.separator();

                        ui.label("ログ");
                        for line in &self.log_lines {
                            ui.monospace(line);
                        }

                        ui.separator();

                        if self.mode == Mode::BruteForce {
                            BruteforceUI::draw_stats(self, ui);
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if self.mode == Mode::BruteForce {
                        BruteforceUI::draw_board(self, ui);
                    } else {
                        ChainplayUI::draw_board(self, ui);
                    }
                });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl App {
    pub fn start_run(&mut self) {
        use crate::domain::search::{SearchConfig, ChainCount, CacheSize, Ratio};
        
        // SearchConfigを構築（Value Objectsを使用）
        let threshold = ChainCount::new(self.threshold.clamp(1, 19))
            .unwrap_or_else(|_| ChainCount::new(7).unwrap());
        let lru_limit = CacheSize::new_in_thousands(self.lru_k.clamp(10, 1000))
            .unwrap_or_else(|_| CacheSize::new_in_thousands(300).unwrap());
        let stop_progress_plateau = Ratio::new(self.stop_progress_plateau.clamp(0.0, 1.0))
            .unwrap_or_else(|_| Ratio::zero());
        
        let config = SearchConfig {
            threshold,
            lru_limit,
            exact_four_only: self.exact_four_only,
            stop_progress_plateau,
            profile_enabled: self.profile_enabled,
        };
        
        let outfile = if let Some(p) = &self.out_path {
            p.clone()
        } else {
            std::path::PathBuf::from(&self.out_name)
        };

        // Board を構築
        let board = match crate::domain::board::Board::try_from_slice(&self.board) {
            Ok(b) => b,
            Err(e) => {
                self.push_log(format!("盤面構築エラー: {e:?}"));
                return;
            }
        };

        self.push_log(format!(
            "出力: JSONL / 形キャッシュ上限 ≈ {} 形 / 保存先: {} / 進捗停滞比={:.2} / 4個消しモード={} / 計測={}",
            config.lru_limit.get(),
            outfile.display(),
            config.stop_progress_plateau.get(),
            if config.exact_four_only { "ON" } else { "OFF" },
            if config.profile_enabled { "ON" } else { "OFF" },
        ));

        // BruteforceServiceを使って検索を開始
        match BruteforceService::start_search_async(&board, config, outfile) {
            Ok((handle, event_rx)) => {
                // SearchEvent → Message 変換用のチャンネル
                let (msg_tx, msg_rx) = unbounded::<Message>();
                
                self.rx = Some(msg_rx);
                self.running = true;
                self.preview = None;
                self.log_lines.clear();
                self.stats.profile = ProfileTotals::default();
                self.search_handle = Some(handle);

                // イベント変換スレッド
                thread::spawn(move || {
                    while let Ok(event) = event_rx.recv() {
                        let msg = crate::app::event_adapter::search_event_to_message(event);
                        if msg_tx.send(msg).is_err() {
                            break;
                        }
                    }
                });
            }
            Err(e) => {
                self.push_log(format!("検索開始エラー: {e:?}"));
            }
        }
    }
}
