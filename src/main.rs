use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use dashmap::{DashMap, DashSet};
use eframe::egui;
use egui::{Color32, RichText, Vec2};
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rand::Rng;
use rayon::prelude::*;
use serde::Serialize;

// u64 キー専用のノーハッシュ（高速化）
use nohash_hasher::BuildNoHashHasher;
type U64Map<V> = std::collections::HashMap<u64, V, BuildNoHashHasher<u64>>;
type U64Set = std::collections::HashSet<u64, BuildNoHashHasher<u64>>;
type DU64Map<V> = DashMap<u64, V, BuildNoHashHasher<u64>>;
type DU64Set = DashSet<u64, BuildNoHashHasher<u64>>;

// ====== 盤面定数 ======
const W: usize = 6;
const H: usize = 14;
const MASK14: u16 = (1u16 << H) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Blank,     // '.'
    Any,       // 'N' (空白 or 色)
    Any4,      // 'X' (色のみ)
    Abs(u8),   // 0..12 = 'A'..'M'
    Color(u8), // 具体色 0..3 = RGBY
}

// ---- 計測（DFS 深さ×フェーズ） -------------------------

#[derive(Default, Clone, Copy)]
struct DfsDepthTimes {
    gen_candidates: Duration,
    assign_cols: Duration,
    upper_bound: Duration,

    // 葉専用
    leaf_fall_pre: Duration,
    leaf_hash: Duration,
    leaf_memo_get: Duration,          // 今回の最適化後はほぼ0のまま
    leaf_memo_miss_compute: Duration, // 到達判定（reaches_t...）
    out_serialize: Duration,
}
#[derive(Default, Clone, Copy)]
struct DfsDepthCounts {
    nodes: u64,
    cand_generated: u64,
    pruned_upper: u64,
    leaves: u64,
    // 葉早期リターン（落下や到達判定より前）
    leaf_pre_tshort: u64,        // 4T 未満で不可能
    leaf_pre_e1_impossible: u64, // E1 不可能（4連結なし）
    memo_lhit: u64,              // 以降は基本0（残しつつ非使用）
    memo_ghit: u64,
    memo_miss: u64,
}
#[derive(Default, Clone)]
struct ProfileTotals {
    dfs_times: [DfsDepthTimes; W + 1],
    dfs_counts: [DfsDepthCounts; W + 1],
    io_write_total: Duration,
}
#[derive(Default, Clone)]
struct TimeDelta {
    dfs_times: [DfsDepthTimes; W + 1],
    dfs_counts: [DfsDepthCounts; W + 1],
    io_write_total: Duration,
}

impl ProfileTotals {
    fn add_delta(&mut self, d: &TimeDelta) {
        for i in 0..=W {
            let a = &mut self.dfs_times[i];
            let b = &d.dfs_times[i];
            a.gen_candidates += b.gen_candidates;
            a.assign_cols += b.assign_cols;
            a.upper_bound += b.upper_bound;
            a.leaf_fall_pre += b.leaf_fall_pre;
            a.leaf_hash += b.leaf_hash;
            a.leaf_memo_get += b.leaf_memo_get;
            a.leaf_memo_miss_compute += b.leaf_memo_miss_compute;
            a.out_serialize += b.out_serialize;

            let ac = &mut self.dfs_counts[i];
            let bc = &d.dfs_counts[i];
            ac.nodes += bc.nodes;
            ac.cand_generated += bc.cand_generated;
            ac.pruned_upper += bc.pruned_upper;
            ac.leaves += bc.leaves;
            ac.leaf_pre_tshort += bc.leaf_pre_tshort;
            ac.leaf_pre_e1_impossible += bc.leaf_pre_e1_impossible;
            ac.memo_lhit += bc.memo_lhit;
            ac.memo_ghit += bc.memo_ghit;
            ac.memo_miss += bc.memo_miss;
        }
        self.io_write_total += d.io_write_total;
    }
}

fn time_delta_has_any(d: &TimeDelta) -> bool {
    if d.io_write_total != Duration::ZERO {
        return true;
    }
    for i in 0..=W {
        let t = d.dfs_times[i];
        if t.gen_candidates != Duration::ZERO
            || t.assign_cols != Duration::ZERO
            || t.upper_bound != Duration::ZERO
            || t.leaf_fall_pre != Duration::ZERO
            || t.leaf_hash != Duration::ZERO
            || t.leaf_memo_get != Duration::ZERO
            || t.leaf_memo_miss_compute != Duration::ZERO
            || t.out_serialize != Duration::ZERO
        {
            return true;
        }
        let c = d.dfs_counts[i];
        if c.nodes != 0
            || c.cand_generated != 0
            || c.pruned_upper != 0
            || c.leaves != 0
            || c.leaf_pre_tshort != 0
            || c.leaf_pre_e1_impossible != 0
            || c.memo_lhit != 0
            || c.memo_ghit != 0
            || c.memo_miss != 0
        {
            return true;
        }
    }
    false
}

// pl を適用した後で、指定列 x の E1 必要セルが全て埋まっているか（= その列の E1 が完成したか）
#[inline(always)]
fn is_e1_col_complete_after(
    pl: &Placement,
    pre: &[[u16; W]; 4],
    e1: &[u16; W],
    board_cols: &[[u16; W]; 4],
    x: usize,
) -> bool {
    let mut after = *board_cols;
    apply_placement(&mut after, pl);
    let e1col = e1[x] & MASK14;
    if e1col == 0 {
        return false;
    }
    for c in 0..4 {
        let needed = pre[c][x] & e1col;
        if needed != 0 {
            let remaining = needed & !after[c][x];
            if remaining != 0 {
                return false;
            }
        }
    }
    true
}

// 現在の board_cols において、列 x の E1 必要セルが全て埋まっているか
#[inline(always)]
fn is_e1_col_complete(
    pre: &[[u16; W]; 4],
    e1: &[u16; W],
    board_cols: &[[u16; W]; 4],
    x: usize,
) -> bool {
    let e1col = e1[x] & MASK14;
    if e1col == 0 {
        return false;
    }
    for c in 0..4 {
        let needed = pre[c][x] & e1col;
        if needed != 0 {
            let remaining = needed & !board_cols[c][x];
            if remaining != 0 {
                return false;
            }
        }
    }
    true
}

// 計測マクロ：enabled 時のみ計測
macro_rules! prof {
    ($enabled:expr, $slot:expr, $e:expr) => {{
        if $enabled {
            let __t0 = Instant::now();
            let __r = $e;
            $slot += __t0.elapsed();
            __r
        } else {
            $e
        }
    }};
}

// ---- ここまで計測 ---------------------------------------

#[derive(Serialize)]
#[allow(dead_code)]
struct OutputLine {
    key: String,
    hash: u32,
    chains: u32,
    pre_chain_board: Vec<String>,
    example_mapping: HashMap<char, u8>,
    mirror: bool,
}

// ====== GUI アプリ状態 ======
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    BruteForce,
    SmallChain,
}

#[derive(Clone)]
struct ScResult {
    pre: [[u16; W]; 4],
    e_masks: Vec<[u16; W]>,
    union: [u16; W],
    key64: u64,
    mirror: bool,
}

#[derive(Clone, Copy)]
struct Placement {
    positions: [(usize, usize, u8); 2], // (x, y, color)
}

#[derive(Clone)]
struct SuggestionCandidate {
    target_idx: usize,
    pre: [[u16; W]; 4],
    union: [u16; W],
    place: Placement,
    contrib: u8, // 2 or 1
}

struct App {
    board: Vec<Cell>, // y*W+x, y=0が最下段
    threshold: u32,
    lru_k: u32,
    out_path: Option<std::path::PathBuf>,
    out_name: String,

    // 早期終了（進捗停滞）
    stop_progress_plateau: f32, // 0..=1

    // 切替: 4個消しモード（5個以上で消去が起きた瞬間に除外）
    exact_four_only: bool,

    // 計測 ON/OFF
    profile_enabled: bool,

    // 実行状態
    running: bool,
    abort_flag: Arc<AtomicBool>,
    rx: Option<Receiver<Message>>,
    stats: Stats,
    preview: Option<[[u16; W]; 4]>,
    log_lines: Vec<String>,

    // タブ
    tab: Tab,

    // 小連鎖生成 用の状態
    sc_pair: (u8, u8),         // 現在のツモ（0..3, 0..3）
    sc_results: Vec<ScResult>, // 総当たりで得た形（メモリ保持）
    sc_prop_index: usize,      // 互換用（未使用／将来用）

    // 目指す形（固定プレビュー）
    sc_target_preview: Option<[[u16; W]; 4]>,

    // 提案候補管理
    sc_candidates: Vec<SuggestionCandidate>,
    sc_candidate_idx: usize,
    sc_suggestion: Option<SuggestionCandidate>,
    sc_last_pair: (u8, u8),
    sc_last_results_len: usize,

    // ★追加：目指す形の選択インデックス
    sc_target_idx: Option<usize>,
    sc_last_target_idx: Option<usize>,

    // 診断表示
    sc_diag_msg: Option<String>,
    sc_diag_cells: Vec<(usize, usize)>,

    // 自動一括実行（探索→目指す形→提案→採用）
    sc_auto_active: bool,
    sc_auto_wait_result: bool,
    sc_auto_cursor: usize,

    // N/T 自動増加モード
    sc_auto_nt_mode: bool,
    sc_auto_nt_hand_count: u32,
}

#[derive(Default, Clone)]
struct Stats {
    searching: bool,
    unique: u64,
    output: u64,
    nodes: u64,
    pruned: u64,
    memo_hit_local: u64,
    memo_hit_global: u64,
    memo_miss: u64,
    total: BigUint,
    done: BigUint,
    rate: f64,
    memo_len: usize,
    lru_limit: usize,

    // 計測結果合計（UI は終了/停止時に表示）
    profile: ProfileTotals,
}

#[derive(Clone, Copy, Default)]
struct StatDelta {
    nodes: u64,
    leaves: u64,
    outputs: u64,
    pruned: u64,
    lhit: u64,
    ghit: u64,
    mmiss: u64,
}

// ====== ここ：メッセージ種類（★逐次配信用バリアントを追加） ======
enum Message {
    Log(String),
    Preview([[u16; W]; 4]),
    Progress(Stats),
    Finished(Stats),
    Error(String),
    // 追加：時間の増分メッセージ
    TimeDelta(TimeDelta),
    // 小連鎖探索の完了（メモリ保持結果：従来版互換）
    SmallChainFinished(Vec<ScResult>),
    // ★ 新規：1件見つけるたび即送る／終了通知
    SmallChainFound(ScResult),
    SmallChainDone,
}

impl Default for App {
    fn default() -> Self {
        let mut board = vec![Cell::Blank; W * H];
        // デフォルトは「下3段をN」
        for y in 0..3 {
            for x in 0..W {
                board[y * W + x] = Cell::Any;
            }
        }
        let mut rng = rand::thread_rng();
        let sc_pair = (rng.gen_range(0..4), rng.gen_range(0..4));
        Self {
            board,
            threshold: 7,
            lru_k: 300,
            out_path: None,
            out_name: "results.jsonl".to_string(),
            stop_progress_plateau: 0.0, // 無効（0.10などにすると有効）
            exact_four_only: false,
            profile_enabled: false,
            running: false,
            abort_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            stats: Stats::default(),
            preview: None,
            log_lines: vec!["待機中".into()],

            tab: Tab::BruteForce,
            sc_pair,
            sc_results: Vec::new(),
            sc_prop_index: 0,
            sc_target_preview: None,
            sc_candidates: Vec::new(),
            sc_candidate_idx: 0,
            sc_suggestion: None,
            sc_last_pair: sc_pair,
            sc_last_results_len: 0,

            // 追加
            sc_target_idx: None,
            sc_last_target_idx: None,

            sc_diag_msg: None,
            sc_diag_cells: Vec::new(),

            // 自動一括
            sc_auto_active: false,
            sc_auto_wait_result: false,
            sc_auto_cursor: 0,

            // N/T 自動増加モード
            sc_auto_nt_mode: false,
            sc_auto_nt_hand_count: 0,
        }
    }
}

fn install_japanese_fonts(ctx: &egui::Context) {
    use egui::{FontData, FontDefinitions, FontFamily};

    let mut fonts = FontDefinitions::default();

    // Windows フォント候補（存在したものを最初に採用）
    let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".to_string());
    let fontdir = std::path::Path::new(&windir).join("Fonts");
    let candidates = [
        "meiryo.ttc",   // Meiryo
        "meiryob.ttc",  // Meiryo UI（環境による）
        "YuGothR.ttc",  // 游ゴシック（Regular）
        "YuGothM.ttc",  // 游ゴシック（Medium）
        "YuGothB.ttc",  // 游ゴシック（Bold）
        "YuGothUI.ttc", // 游ゴシック UI
        "YuGothU.ttc",  // 旧表記の可能性
        "msgothic.ttc", // MS ゴシック（最終手段）
        "msmincho.ttc", // MS 明朝（最終手段）
    ];

    let mut loaded = false;
    for name in candidates.iter() {
        let path = fontdir.join(name);
        if let Ok(bytes) = std::fs::read(&path) {
            let key = format!("jp-{}", name.to_lowercase());
            fonts
                .font_data
                .insert(key.clone(), FontData::from_owned(bytes));
            fonts
                .families
                .get_mut(&FontFamily::Proportional)
                .unwrap()
                .insert(0, key.clone());
            fonts
                .families
                .get_mut(&FontFamily::Monospace)
                .unwrap()
                .insert(0, key.clone());
            loaded = true;
            break;
        }
    }

    if loaded {
        ctx.set_fonts(fonts);
    } else {
        eprintln!(
            "日本語フォントを見つけられませんでした。C:\\Windows\\Fonts を確認してください。"
        );
    }
}

// ====== eframe エントリ ======
fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(980.0, 760.0)),
        ..Default::default()
    };

    eframe::run_native(
        "ぷよぷよ 連鎖形（総当たり/小連鎖）— Rust GUI",
        options,
        Box::new(|cc| {
            install_japanese_fonts(&cc.egui_ctx);
            Box::new(App::default())
        }),
    )
    .map_err(|e| anyhow!("GUI起動に失敗: {e}"))
}

// ===== App ユーティリティ =====
impl App {
    fn push_log(&mut self, s: String) {
        self.log_lines.push(s);
        if self.log_lines.len() > 500 {
            let cut = self.log_lines.len() - 500;
            self.log_lines.drain(0..cut);
        }
    }
}

// ===== App メソッド（小連鎖） =====
impl App {
    // ===== 小連鎖探索の起動（★逐次ストリーミング＋再押下で中断＆リセット） =====
    fn start_small_chain(&mut self) {
        // 既存の処理が走っていたら中断（総当たり/小連鎖どちらでも）
        if self.running {
            self.push_log("[小連鎖] 実行中を中断します…".into());
            self.abort_flag.store(true, Ordering::Relaxed);
            self.rx = None; // 古いチャネルは破棄
        }

        // 小連鎖用の状態をリセット
        self.sc_results.clear();
        self.sc_candidates.clear();
        self.sc_candidate_idx = 0;
        self.sc_suggestion = None;
        self.sc_target_preview = None;
        self.sc_diag_msg = None;
        self.sc_diag_cells.clear();

        // ★ 目指す形の選択状態もリセット
        self.sc_target_idx = None;
        self.sc_last_target_idx = None;

        // 条件
        let threshold = self.threshold.clamp(1, 19);
        let exact_four_only = self.exact_four_only;

        // 盤面を char 配列へ
        let board_chars: Vec<char> = self.board.iter().map(|c| c.label_char()).collect();

        self.push_log(format!(
            "[小連鎖] 探索開始: T={} / 4個消しモード={}",
            threshold,
            if exact_four_only { "ON" } else { "OFF" }
        ));

        // 新しいチャネルを用意
        let (tx, rx) = unbounded::<Message>();
        self.rx = Some(rx);
        // 進捗の初期化
        self.stats = Stats::default();
        self.running = true;
        self.abort_flag.store(false, Ordering::Relaxed);
        let abort = self.abort_flag.clone();
        let pair_colors = self.sc_pair;

        // 逐次ストリーミング版の探索を起動
        thread::spawn(move || {
            if let Err(e) = run_small_chain_search_streaming(
                board_chars,
                threshold,
                exact_four_only,
                pair_colors,
                abort,
                tx.clone(),
            ) {
                let _ = tx.send(Message::Error(format!("[小連鎖] 失敗: {e:#}")));
            } else {
                let _ = tx.send(Message::SmallChainDone);
            }
        });
    }

    /// ★ 目指す形を“選択/ローテーション”して固定プレビューへ反映
    fn select_or_cycle_target(&mut self) {
        if self.sc_results.is_empty() {
            self.push_log("[小連鎖] まず『小連鎖探索開始』で形を収集してください。".into());
            return;
        }
        let next = match self.sc_target_idx {
            None => 0,
            Some(i) => (i + 1) % self.sc_results.len(),
        };
        self.sc_target_idx = Some(next);
        self.sc_last_target_idx = self.sc_target_idx;

        // プレビュー固定＆候補は一旦リセット
        let sc = &self.sc_results[next];
        self.sc_target_preview = Some(sc.pre);
        self.sc_candidates.clear();
        self.sc_candidate_idx = 0;
        self.sc_suggestion = None;
        self.sc_diag_msg = None;
        self.sc_diag_cells.clear();

        self.push_log(format!(
            "[小連鎖] 『目指す形』を #{} に設定しました。『置き方を提案』でこの形に寄与する配置を探します。",
            next
        ));
    }

    // ===== 置き方の提案（★目指す形を固定したうえで計算） =====
    fn compute_or_advance_proposal(&mut self) {
        if self.sc_results.is_empty() {
            self.push_log("[小連鎖] まずは『小連鎖探索開始』で形を収集してください。".into());
            return;
        }
        let Some(target_idx) = self.sc_target_idx else {
            self.push_log("[小連鎖] 先に『目指す形を提案』で目指す形を選んでください。".into());
            return;
        };

        // 入力（ペア or 結果 or 目指す形）が変わったら候補を再構築
        if self.sc_candidates.is_empty()
            || self.sc_last_pair != self.sc_pair
            || self.sc_last_results_len != self.sc_results.len()
            || self.sc_last_target_idx != self.sc_target_idx
        {
            let cols_exist = build_cols_from_board_colors(&self.board);
            let heights = column_heights(&cols_exist);
            let c0 = self.sc_pair.0;
            let c1 = self.sc_pair.1;
            let placements = enumerate_all_placements_with_walls(&heights, c0, c1);

            let sc = &self.sc_results[target_idx];

            // E1（1連鎖目）関連の順序制約用前計算
            let e1_mask_opt: Option<[u16; W]> = sc.e_masks.get(0).cloned();
            let (open_flags, top_y_arr) = if let Some(ref e1m) = e1_mask_opt {
                compute_open_top_info(&sc.pre, e1m)
            } else {
                ([false; W], [0usize; W])
            };
            let has_open = open_flags.iter().any(|&b| b);
            // 事前に「既に完成している E1 列」を把握（pl とは独立なので1回だけ計算）
            let complete_before: [bool; W] = if let Some(ref e1m) = e1_mask_opt {
                let mut arr = [false; W];
                for x in 0..W {
                    arr[x] = is_e1_col_complete(&sc.pre, e1m, &cols_exist, x);
                }
                arr
            } else {
                [false; W]
            };

            let mut cands: Vec<SuggestionCandidate> = Vec::new();
            // 診断用カウンタ
            let mut total_trials: u32 = 0;
            let mut cnt_no_contrib: u32 = 0;
            let mut cnt_blocked_union: u32 = 0;
            let mut cnt_immediate_ge5: u32 = 0;
            let mut cnt_immediate_len_ng: u32 = 0;
            let mut cnt_ordering_violation: u32 = 0;
            let mut diag_cells_set: HashSet<(usize, usize)> = HashSet::new();

            for pl in &placements {
                total_trials += 1;

                // 任意yで寄与判定
                let contrib = contribution_on_pre(pl, &sc.pre);
                if contrib == 0 {
                    cnt_no_contrib += 1;
                    continue;
                }

                // 1個寄与ならゴミの位置は union に入らないこと
                if contrib == 1 {
                    if let Some((gx, gy, _)) = garbage_piece_on_pre(pl, &sc.pre) {
                        let bit = 1u16 << gy;
                        if (sc.union[gx] & bit) != 0 {
                            cnt_blocked_union += 1;
                            diag_cells_set.insert((gx, gy));
                            continue;
                        }
                    }
                }

                // 直後に消える置き方は、連鎖長がちょうどTのときのみ許可。
                // exact_four_only のとき、5個以上が絡む初回消去は不許可。
                let mut after = cols_exist.clone();
                apply_placement(&mut after, pl);
                let (clear, had_ge5, _had_four) = compute_erase_mask_and_flags(&after);
                let any_clear = (0..W).any(|x| clear[x] != 0);
                if any_clear {
                    if self.exact_four_only && had_ge5 {
                        cnt_immediate_ge5 += 1;
                        continue;
                    }
                    let len = chain_len(after, self.exact_four_only);
                    if len != self.threshold {
                        cnt_immediate_len_ng += 1;
                        continue;
                    }
                }

                // 順序制約：E1 の open-top 列について、『今回の配置でその列のE1が完成する』場合に、
                // まだ他列のE1が未完成のままなら除外する。
                if has_open {
                    if let Some(ref e1m) = e1_mask_opt {
                        // 今回で完成する open-top 列を列挙
                        let mut completed_cols: Vec<usize> = Vec::new();
                        for x in 0..W {
                            if !open_flags[x] {
                                continue;
                            }
                            if complete_before[x] {
                                continue;
                            }
                            if is_e1_col_complete_after(pl, &sc.pre, e1m, &cols_exist, x) {
                                completed_cols.push(x);
                            }
                        }
                        if !completed_cols.is_empty() {
                            if any_other_e1_incomplete_after(
                                pl,
                                &sc.pre,
                                e1m,
                                &cols_exist,
                                &completed_cols,
                            ) {
                                cnt_ordering_violation += 1;
                                for &x in &completed_cols {
                                    diag_cells_set.insert((x, top_y_arr[x]));
                                }
                                continue;
                            }
                        }
                    }
                }

                cands.push(SuggestionCandidate {
                    target_idx,
                    pre: sc.pre,
                    union: sc.union,
                    place: *pl,
                    contrib: contrib as u8,
                });
            }

            if cands.is_empty() {
                self.sc_suggestion = None;
                // ターゲットは維持
                self.sc_target_preview = Some(sc.pre);
                // 診断を構築
                let mut parts: Vec<String> = Vec::new();
                parts.push(format!("試行: {} 件", total_trials));
                if cnt_no_contrib > 0 {
                    parts.push(format!("寄与0: {} 件", cnt_no_contrib));
                }
                if cnt_blocked_union > 0 {
                    parts.push(format!("ゴミがEマスクに衝突: {} 件", cnt_blocked_union));
                }
                if cnt_immediate_ge5 > 0 {
                    parts.push(format!(
                        "初回に5個以上の消去が発生: {} 件",
                        cnt_immediate_ge5
                    ));
                }
                if cnt_immediate_len_ng > 0 {
                    parts.push(format!("直後消去だが連鎖長≠T: {} 件", cnt_immediate_len_ng));
                }
                if cnt_ordering_violation > 0 {
                    parts.push(format!(
                        "順序制約（E1のopen-top最上段を先に埋めない）で除外: {} 件",
                        cnt_ordering_violation
                    ));
                }
                if parts.is_empty() {
                    parts.push("（要因不明：内部条件により除外）".into());
                }
                self.sc_diag_msg = Some(format!("[診断] {}", parts.join(" / ")));
                // 強調表示セル（上限64）
                self.sc_diag_cells = diag_cells_set.into_iter().take(64).collect();
                self.push_log(
                    "[小連鎖] 今のツモで寄与する置き方は見つかりませんでした。診断を表示します。"
                        .into(),
                );
                return;
            }

            // 並べ替え：2個寄与を優先、次により低い段を優先
            cands.sort_by(|a, b| {
                use std::cmp::Ordering::*;
                let k1 = b.contrib.cmp(&a.contrib);
                if k1 != Equal {
                    return k1;
                }
                let ay = min_placement_y(&a.place).cmp(&min_placement_y(&b.place));
                if ay != Equal {
                    return ay;
                }
                Equal
            });

            self.sc_candidates = cands;
            self.sc_candidate_idx = 0;
            self.sc_last_pair = self.sc_pair;
            self.sc_last_results_len = self.sc_results.len();
            self.sc_last_target_idx = self.sc_target_idx;
            // 候補があるので診断はクリア / 目指す形プレビューは固定のまま
            self.sc_diag_msg = None;
            self.sc_diag_cells.clear();
        } else {
            // 次の候補へ
            if !self.sc_candidates.is_empty() {
                self.sc_candidate_idx = (self.sc_candidate_idx + 1) % self.sc_candidates.len();
            }
        }

        if let Some(cand) = self.sc_candidates.get(self.sc_candidate_idx).cloned() {
            self.sc_suggestion = Some(cand.clone());
            // 目指す形は固定（念のため再表示）
            if let Some(idx) = self.sc_target_idx {
                self.sc_target_preview = Some(self.sc_results[idx].pre);
            }
            self.push_log(format!(
                "[小連鎖] 提案: 目指す形#{} / 寄与={} / 配置=({},{})&({},{})",
                cand.target_idx,
                cand.contrib,
                cand.place.positions[0].0,
                cand.place.positions[0].1,
                cand.place.positions[1].0,
                cand.place.positions[1].1,
            ));
        }
    }

    // ===== 提案の適用 =====
    fn apply_current_proposal(&mut self) {
        let Some(sug) = self.sc_suggestion.clone() else {
            self.push_log("[小連鎖] 提案がありません。『置き方を提案』を押してください。".into());
            return;
        };
        // セルを具体色で確定
        for &(x, y, col) in &sug.place.positions {
            if x < W && y < H {
                let idx = y * W + x;
                self.board[idx] = Cell::Color(col);
            }
        }

        // N/T 自動増加モード: 1手ごとに N を最小列へ2個追加、3手ごとに T を +1
        if self.sc_auto_nt_mode {
            for _ in 0..2 {
                if let Some((nx, ny)) = self.add_n_to_lowest_column() {
                    self.push_log(format!(
                        "[小連鎖][NT] 列{} の y={} に N を追加しました。",
                        nx, ny
                    ));
                } else {
                    self.push_log("[小連鎖][NT] N を追加できる列がありません（全列満杯）".into());
                    break;
                }
            }
            // 手数カウントと T 増加（3手ごと）
            self.sc_auto_nt_hand_count = self.sc_auto_nt_hand_count.saturating_add(1);
            if self.sc_auto_nt_hand_count % 3 == 0 {
                let before = self.threshold;
                self.threshold = (self.threshold + 1).min(19);
                if self.threshold != before {
                    self.push_log(format!(
                        "[小連鎖][NT] 3手経過のため T を {} に増やしました。",
                        self.threshold
                    ));
                }
            }
        }

        // ツモ色をランダムに変える（目指す形は維持）
        let mut rng = rand::thread_rng();
        self.sc_pair = (rng.gen_range(0..4), rng.gen_range(0..4));
        self.sc_suggestion = None;
        self.sc_candidates.clear();
        self.push_log(
            "[小連鎖] 提案を適用しました。ツモをランダムに更新（目指す形は維持）。".into(),
        );
    }

    // === N/T 自動増加用ヘルパ ===
    fn col_nonblank_count(&self, x: usize) -> usize {
        let mut cnt = 0usize;
        for y in 0..H {
            let c = self.board[y * W + x];
            if !matches!(c, Cell::Blank) {
                cnt += 1;
            }
        }
        cnt
    }

    // 非Blankセル数（Color/N/Abs/X）合計が最小の列を選び、最下段の空きに N を追加
    // 追加に成功したら (x, y) を返す
    fn add_n_to_lowest_column(&mut self) -> Option<(usize, usize)> {
        // 列 x を非Blankセル数で昇順に並べる（左優先）
        let mut idxs: Vec<usize> = (0..W).collect();
        idxs.sort_by_key(|&x| self.col_nonblank_count(x));

        for &x in &idxs {
            // 最下段の空きセル（y=0..H-1）を探す
            for y in 0..H {
                let i = y * W + x;
                if matches!(self.board[i], Cell::Blank) {
                    self.board[i] = Cell::Any; // 'N'
                    return Some((x, y));
                }
            }
        }
        None
    }

    // ===== 自動実行（探索→目指す形→提案→採用） =====
    fn start_sc_auto(&mut self) {
        // 自動状態初期化
        self.sc_auto_active = true;
        self.sc_auto_wait_result = false;
        self.sc_auto_cursor = 0;
        self.push_log("[小連鎖][自動] 自動実行を開始します。".into());

        // まず探索を開始（逐次で結果が届く）
        self.start_small_chain();
        self.sc_auto_wait_result = true;
    }

    fn stop_sc_auto(&mut self) {
        self.sc_auto_active = false;
        self.sc_auto_wait_result = false;
        self.push_log("[小連鎖][自動] 自動実行を停止しました。".into());
    }

    fn sc_auto_tick(&mut self) {
        // 探索結果待ち
        if self.sc_auto_wait_result {
            if self.sc_results.is_empty() {
                // まだ結果が来ていない。探索が終わっているなら停止。
                if !self.running {
                    if self.sc_auto_nt_mode {
                        // NTモード中は、見つからなかった場合でもNを最大2個追加して再探索
                        let mut added = 0usize;
                        for _ in 0..2 {
                            if let Some((nx, ny)) = self.add_n_to_lowest_column() {
                                self.push_log(format!("[小連鎖][自動][NT] 形が見つからなかったため 列{} の y={} に N を追加しました。", nx, ny));
                                added += 1;
                            } else {
                                break;
                            }
                        }
                        if added == 0 {
                            self.push_log(
                                "[小連鎖][自動][NT] N を追加できる余地がないため自動を停止します。"
                                    .into(),
                            );
                            self.stop_sc_auto();
                        } else {
                            self.push_log(format!(
                                "[小連鎖][自動][NT] Nを{}個追加したので再探索を開始します…",
                                added
                            ));
                            self.start_small_chain();
                            self.sc_auto_wait_result = true;
                        }
                    } else {
                        self.push_log(
                            "[小連鎖][自動] 探索で形が見つかりませんでした。自動を停止します。"
                                .into(),
                        );
                        self.stop_sc_auto();
                    }
                }
                return;
            }
            // 1件でも来たら進行
            self.sc_auto_wait_result = false;
            if self.sc_target_idx.is_none() {
                self.select_or_cycle_target();
            }
        }

        // 目指す形が未選択 かつ 結果があるなら選ぶ
        if self.sc_target_idx.is_none() && !self.sc_results.is_empty() {
            self.select_or_cycle_target();
        }

        // 候補が未計算なら計算
        if self.sc_suggestion.is_none() {
            self.compute_or_advance_proposal();
        }

        // それでも出ない場合は目指す形を順に切り替えながら探索
        if self.sc_suggestion.is_none() && !self.sc_results.is_empty() {
            let total = self.sc_results.len();
            let mut tried = 0usize;
            while tried < total {
                self.select_or_cycle_target();
                self.compute_or_advance_proposal();
                if self.sc_suggestion.is_some() {
                    break;
                }
                tried += 1;
            }
            if self.sc_suggestion.is_none() {
                // 現在の候補群では不可。探索が継続中なら新着の形を待つ。
                if self.running {
                    self.push_log("[小連鎖][自動] 今のツモでは現時点の『目指す形』では置き方が見つかりません。新しい形の到着を待ちます…".into());
                    self.sc_auto_wait_result = true;
                    return;
                } else {
                    if self.sc_auto_nt_mode {
                        // 結果はあるが置き方が見つからなかった。Nを最大2個追加して再探索
                        let mut added = 0usize;
                        for _ in 0..2 {
                            if let Some((nx, ny)) = self.add_n_to_lowest_column() {
                                self.push_log(format!("[小連鎖][自動][NT] 置き方が見つからなかったため 列{} の y={} に N を追加しました。", nx, ny));
                                added += 1;
                            } else {
                                break;
                            }
                        }
                        if added == 0 {
                            self.push_log(
                                "[小連鎖][自動][NT] N を追加できる余地がないため自動を停止します。"
                                    .into(),
                            );
                            self.stop_sc_auto();
                            return;
                        } else {
                            self.push_log(format!(
                                "[小連鎖][自動][NT] Nを{}個追加したので再探索を開始します…",
                                added
                            ));
                            self.start_small_chain();
                            self.sc_auto_wait_result = true;
                            return;
                        }
                    } else {
                        self.push_log("[小連鎖][自動] 探索が終了し、どの『目指す形』でも置き方が見つかりませんでした。自動を停止します。".into());
                        self.stop_sc_auto();
                        return;
                    }
                }
            }
        }

        // 提案が得られたら即採用して終了
        if self.sc_suggestion.is_some() {
            self.apply_current_proposal();
            self.push_log("[小連鎖][自動] 提案を自動適用しました。".into());
            self.sc_diag_msg = None;
            self.sc_diag_cells.clear();
            self.stop_sc_auto();
        }
    }
}

fn has_profile_any(p: &ProfileTotals) -> bool {
    if p.io_write_total != Duration::ZERO {
        return true;
    }
    for i in 0..=W {
        let t = p.dfs_times[i];
        if t.gen_candidates != Duration::ZERO
            || t.assign_cols != Duration::ZERO
            || t.upper_bound != Duration::ZERO
            || t.leaf_fall_pre != Duration::ZERO
            || t.leaf_hash != Duration::ZERO
            || t.leaf_memo_get != Duration::ZERO
            || t.leaf_memo_miss_compute != Duration::ZERO
            || t.out_serialize != Duration::ZERO
        {
            return true;
        }
        let c = p.dfs_counts[i];
        if c.nodes != 0
            || c.cand_generated != 0
            || c.pruned_upper != 0
            || c.leaves != 0
            || c.leaf_pre_tshort != 0
            || c.leaf_pre_e1_impossible != 0
            || c.memo_lhit != 0
            || c.memo_ghit != 0
            || c.memo_miss != 0
        {
            return true;
        }
    }
    false
}

fn fmt_dur_ms(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.3} ms", ms)
    } else {
        format!("{:.1} ms", ms)
    }
}

fn show_profile_table(ui: &mut egui::Ui, p: &ProfileTotals) {
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
            ui.monospace("fall");
            ui.monospace("hash");
            ui.monospace("memo_get/miss_compute/out");
            ui.end_row();

            for d in 0..=W {
                let c = p.dfs_counts[d];
                let t = p.dfs_times[d];
                ui.monospace(format!("{:>2}", d));
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
                ui.monospace(fmt_dur_ms(t.leaf_fall_pre));
                ui.monospace(fmt_dur_ms(t.leaf_hash));
                ui.monospace(format!(
                    "{} / {} / {}",
                    fmt_dur_ms(t.leaf_memo_get),
                    fmt_dur_ms(t.leaf_memo_miss_compute),
                    fmt_dur_ms(t.out_serialize),
                ));
                ui.end_row();
            }
        });
}

#[derive(Clone, Copy)]
enum TCell {
    Blank,
    Any,
    Any4,
    Fixed(u8),
    // 'N' の下2段制約（小連鎖用）：許可色のビットマスク（bit0..bit3）
    AnyMask(u8),
}

impl Cell {
    fn label_char(self) -> char {
        match self {
            Cell::Blank => '.',
            Cell::Any => 'N',
            Cell::Any4 => 'X',
            Cell::Abs(i) => (b'A' + i) as char,
            Cell::Color(i) => (b'0' + (i.min(3))) as char,
        }
    }
}

fn cell_style(c: Cell) -> (String, Color32, egui::Stroke) {
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
        Cell::Color(i) => {
            let idx = (i as usize).min(3);
            let palette = [
                Color32::from_rgb(96, 165, 250),  // 0
                Color32::from_rgb(244, 114, 182), // 1
                Color32::from_rgb(251, 191, 36),  // 2
                Color32::from_rgb(52, 211, 153),  // 3
            ];
            (
                format!("{}", idx),
                palette[idx],
                egui::Stroke::new(1.0, Color32::from_rgb(37, 99, 235)),
            )
        }
    }
}
fn cycle_abs(c: Cell) -> Cell {
    match c {
        Cell::Blank | Cell::Any | Cell::Any4 => Cell::Abs(0),
        Cell::Abs(i) => Cell::Abs(((i as usize + 1) % 13) as u8),
        Cell::Color(_) => Cell::Abs(0),
    }
}
fn cycle_any(c: Cell) -> Cell {
    match c {
        Cell::Any => Cell::Any4,
        Cell::Any4 => Cell::Any,
        _ => Cell::Any,
    }
}

fn cycle_color(c: Cell) -> Cell {
    match c {
        Cell::Color(i) => Cell::Color(((i as usize + 1) % 4) as u8),
        _ => Cell::Color(0),
    }
}

fn draw_preview(ui: &mut egui::Ui, cols: &[[u16; W]; 4]) {
    let cell = 16.0_f32; // 1マスのサイズ
    let gap = 1.0_f32; // マス間の隙間

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let palette = [
        Color32::from_rgb(96, 165, 250),  // blue
        Color32::from_rgb(244, 114, 182), // pink
        Color32::from_rgb(251, 191, 36),  // yellow
        Color32::from_rgb(52, 211, 153),  // green
    ];

    for y in 0..H {
        for x in 0..W {
            let mut cidx: Option<usize> = None;
            let bit = 1u16 << y;
            if cols[0][x] & bit != 0 {
                cidx = Some(0);
            } else if cols[1][x] & bit != 0 {
                cidx = Some(1);
            } else if cols[2][x] & bit != 0 {
                cidx = Some(2);
            } else if cols[3][x] & bit != 0 {
                cidx = Some(3);
            }

            let fill = cidx.map(|k| palette[k]).unwrap_or(Color32::WHITE);

            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
            painter.rect_filled(r, 3.0, fill);
        }
    }
}

fn draw_preview_with_suggestion(
    ui: &mut egui::Ui,
    cols: &[[u16; W]; 4],
    highlight: &[(usize, usize, u8); 2],
) {
    let cell = 16.0_f32; // 1マスのサイズ（draw_preview と合わせる）
    let gap = 1.0_f32; // マス間の隙間

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    let palette = [
        Color32::from_rgb(96, 165, 250),  // blue
        Color32::from_rgb(244, 114, 182), // pink
        Color32::from_rgb(251, 191, 36),  // yellow
        Color32::from_rgb(52, 211, 153),  // green
    ];

    // 基本グリッドの塗り
    for y in 0..H {
        for x in 0..W {
            let mut cidx: Option<usize> = None;
            let bit = 1u16 << y;
            if cols[0][x] & bit != 0 {
                cidx = Some(0);
            } else if cols[1][x] & bit != 0 {
                cidx = Some(1);
            } else if cols[2][x] & bit != 0 {
                cidx = Some(2);
            } else if cols[3][x] & bit != 0 {
                cidx = Some(3);
            }

            let fill = cidx.map(|k| palette[k]).unwrap_or(Color32::WHITE);

            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
            painter.rect_filled(r, 3.0, fill);
        }
    }

    // 提案配置セルの強調枠
    let stroke = egui::Stroke::new(2.0, Color32::BLACK);
    for &(x, y, _col) in highlight.iter() {
        if x < W && y < H {
            let x0 = rect.min.x + x as f32 * (cell + gap);
            let y0 = rect.max.y - ((y + 1) as f32 * cell + y as f32 * gap);
            let r = egui::Rect::from_min_size(egui::pos2(x0, y0), egui::vec2(cell, cell));
            painter.rect_stroke(r, 3.0, stroke);
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 受信メッセージ処理（安全：take→処理→必要なら戻す）
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
                    Message::SmallChainFinished(v) => {
                        // 互換用：今回の逐次モードでは基本使わないが残す
                        self.sc_results = v;
                        self.sc_last_results_len = self.sc_results.len();
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log(format!("[小連鎖] 形を {} 件 取得", self.sc_results.len()));
                        keep_rx = false;
                    }
                    // ★ 逐次ストリーム：1件見つかったら即受け取る（目指す形は自動決定しない）
                    Message::SmallChainFound(r) => {
                        let was_empty = self.sc_results.is_empty();
                        self.sc_results.push(r.clone());
                        self.sc_last_results_len = self.sc_results.len();

                        if was_empty {
                            self.push_log(
                                "[小連鎖] 1件目の形を見つけました。『目指す形を提案』で形を選んでから『置き方を提案』してください。".into()
                            );
                        }
                    }
                    // ★ 探索スレッドの終了
                    Message::SmallChainDone => {
                        self.running = false;
                        self.abort_flag.store(false, Ordering::Relaxed);
                        self.push_log(format!(
                            "[小連鎖] 探索終了（累計 {} 件）",
                            self.sc_results.len()
                        ));
                        keep_rx = false;
                    }
                }
            }

            if keep_rx {
                self.rx = Some(rx);
            }
        }

        // 自動進行（小連鎖：探索→目指す形→提案→採用）
        if self.sc_auto_active {
            self.sc_auto_tick();
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("ぷよぷよ 連鎖形（総当たり/小連鎖）— Rust GUI（列ストリーミング＋LRU形キャッシュ＋並列化＋計測＋追撃最適化）");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.tab, Tab::BruteForce, "総当たり");
                ui.radio_value(&mut self.tab, Tab::SmallChain, "小連鎖生成");
            });
        });

        // 左ペイン全体をひとつの ScrollArea でまとめる
        egui::SidePanel::left("left").min_width(420.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);

                match self.tab {
                    Tab::BruteForce => {
                        // ── 入力と操作（総当たり） ─────────────────────────────────
                        ui.group(|ui| {
                            ui.label("入力と操作（総当たり）");
                            ui.label("左クリック: A→B→…→M / Shift+左クリック: 色0..3 / 中クリック: N↔X / 右クリック: ・");

                            ui.horizontal_wrapped(|ui| {
                                ui.add(egui::DragValue::new(&mut self.threshold).clamp_range(1..=19).speed(0.1));
                                ui.label("連鎖閾値");
                                ui.add_space(8.0);
                                ui.add(egui::DragValue::new(&mut self.lru_k).clamp_range(10..=1000).speed(1.0));
                                ui.label("形キャッシュ上限(千)");
                            });

                            // ★ 進捗停滞 早期終了
                            ui.horizontal_wrapped(|ui| {
                                ui.add(
                                    egui::DragValue::new(&mut self.stop_progress_plateau)
                                        .clamp_range(0.0..=1.0)
                                        .speed(0.01),
                                );
                                ui.label("早期終了: 進捗停滞比 (0=無効, 例 0.10)");
                            });

                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.exact_four_only, "4個消しモード（5個以上で消えたら除外）");
                            });

                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.profile_enabled, "計測を有効化（軽量）");
                            });

                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(!self.running, egui::Button::new("Run"))
                                    .clicked()
                                {
                                    self.start_run();
                                }
                                if ui
                                    .add_enabled(self.running, egui::Button::new("Stop"))
                                    .clicked()
                                {
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
                                        self.out_name = path
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .into();
                                    }
                                }
                            });
                        });

                        ui.separator();

                        // ── 処理時間（累積） ────────────────────────────────────────
                        if !self.running && self.profile_enabled && has_profile_any(&self.stats.profile) {
                            ui.group(|ui| {
                                ui.label("処理時間（累積）");
                                show_profile_table(ui, &self.stats.profile);
                            });
                            ui.separator();
                        }

                        // ── プレビュー ─────────────────────────────────────────────
                        ui.label("プレビュー（E1直前の落下後盤面）");
                        ui.add_space(4.0);
                        if let Some(cols) = &self.preview {
                            draw_preview(ui, cols);
                        } else {
                            ui.label(
                                egui::RichText::new("（実行中に更新表示）")
                                    .italics()
                                    .color(Color32::GRAY),
                            );
                        }

                        ui.separator();

                        // ── ログ ──────────────────────────────────────────────────
                        ui.label("ログ");
                        for line in &self.log_lines {
                            ui.monospace(line);
                        }

                        ui.separator();

                        // ── 実行・進捗 ────────────────────────────────────────────
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
                    Tab::SmallChain => {
                        ui.group(|ui| {
                            ui.label("入力と操作（小連鎖生成）");
                            ui.label("左クリック: A→B→…→M / Shift+左クリック: 色0..3 / 中クリック: N↔X / 右クリック: ・");
                            ui.horizontal_wrapped(|ui| {
                                let mut t_val = self.threshold as u32;
                                ui.label("T（連鎖数）");
                                if ui.add(egui::DragValue::new(&mut t_val).clamp_range(1..=19).speed(0.1)).changed() {
                                    self.threshold = t_val as u32;
                                }
                                ui.add_space(8.0);
                                ui.checkbox(&mut self.exact_four_only, "4個消しモード（5個以上で消えたら除外）");
                            });

                            // N/T 自動増加モードの切替
                            ui.horizontal_wrapped(|ui| {
                                ui.checkbox(&mut self.sc_auto_nt_mode, "N/T自動増加モード");
                                if self.sc_auto_nt_mode {
                                    ui.label(egui::RichText::new("（1手毎にNを最小列へ2個追加／3手毎にTを+1）").italics().color(Color32::GRAY));
                                }
                            });

                            ui.add_space(4.0);
                            // 現在のツモ表示とランダム化
                            ui.horizontal(|ui| {
                                ui.label("現在のツモ: ");
                                let (c0, c1) = self.sc_pair;
                                let palette = [
                                    Color32::from_rgb(96, 165, 250),
                                    Color32::from_rgb(244, 114, 182),
                                    Color32::from_rgb(251, 191, 36),
                                    Color32::from_rgb(52, 211, 153),
                                ];
                                let btn_sz = Vec2::new(24.0, 24.0);
                                ui.add_sized(btn_sz, egui::Button::new(RichText::new(format!("{}", c0)).strong()).fill(palette[c0 as usize]));
                                ui.label("+");
                                ui.add_sized(btn_sz, egui::Button::new(RichText::new(format!("{}", c1)).strong()).fill(palette[c1 as usize]));
                                if ui.button("ランダム").clicked() {
                                    let mut rng = rand::thread_rng();
                                    self.sc_pair = (rng.gen_range(0..4), rng.gen_range(0..4));
                                }
                            });

                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if ui.button("小連鎖探索開始").clicked() {
                                    self.start_small_chain();
                                }
                                if ui.button("目指す形を提案").clicked() {
                                    self.select_or_cycle_target();
                                }
                                if ui.button("置き方を提案").clicked() {
                                    self.compute_or_advance_proposal();
                                }
                                if ui.button("提案を採用").clicked() {
                                    self.apply_current_proposal();
                                }
                                let auto_label = if self.sc_auto_active { "自動停止" } else { "自動（探索→提案→採用）" };
                                if ui.button(auto_label).clicked() {
                                    if self.sc_auto_active { self.stop_sc_auto(); } else { self.start_sc_auto(); }
                                }
                            });
                        });

                        ui.separator();

                        // ── 実行・進捗（小連鎖） ────────────────────────────────────
                        ui.group(|ui| {
                            ui.label("実行・進捗（小連鎖）");
                            let pct = {
                                let total = self.stats.total.to_f64().unwrap_or(0.0);
                                let done = self.stats.done.to_f64().unwrap_or(0.0);
                                if total > 0.0 { (done / total * 100.0).clamp(0.0, 100.0) } else { 0.0 }
                            };
                            ui.label(format!("進捗: {:.1}%", pct));
                            ui.add(egui::ProgressBar::new((pct / 100.0) as f32).show_percentage());

                            ui.add_space(4.0);
                            ui.monospace(format!(
                                "探索中: {} / 総組合せ(厳密): {} / 完了(葉): {} / 速度: {:.1} leaves/s",
                                if self.stats.searching { "YES" } else { "NO" },
                                &self.stats.total,
                                &self.stats.done,
                                self.stats.rate,
                            ));
                        });

                        ui.separator();

                        ui.label("この形を目指しています（例図）");
                        ui.add_space(4.0);
                        if let Some(cols) = &self.sc_target_preview {
                            draw_preview(ui, cols);
                        } else {
                            ui.label(RichText::new("（『目指す形を提案』で選択すると表示）").italics().color(Color32::GRAY));
                        }

                        ui.separator();

                        // 提案プレビュー
                        ui.label("置き方の提案プレビュー（適用後）");
                        ui.add_space(4.0);
                        if let Some(sug) = &self.sc_suggestion {
                            let mut cols = build_cols_from_board_colors(&self.board);
                            apply_placement(&mut cols, &sug.place);
                            let hl = [sug.place.positions[0], sug.place.positions[1]];
                            draw_preview_with_suggestion(ui, &cols, &hl);
                        } else {
                            ui.label(RichText::new("（『置き方を提案』で候補を計算するとここに表示）").italics().color(Color32::GRAY));
                        }

                        ui.separator();

                        // 診断表示
                        ui.label("診断");
                        if let Some(msg) = &self.sc_diag_msg {
                            ui.monospace(msg);
                            if !self.sc_diag_cells.is_empty() {
                                ui.label(RichText::new("赤枠がブロックされたマスです").color(Color32::RED));
                            }
                        } else {
                            ui.label(RichText::new("（候補が見つからなかった場合に要因を表示）").italics().color(Color32::GRAY));
                        }

                        // ログ
                        ui.label("ログ");
                        for line in &self.log_lines {
                            ui.monospace(line);
                        }
                    }
                }
            });
        });

        // 盤面側もスクロール可能に
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.label("盤面（左: A→B→…→M / Shift+左: 色0..3 / 中: N↔X / 右: ・）");
                    ui.add_space(6.0);

                    let cell_size = Vec2::new(28.0, 28.0);
                    let gap = 2.0;

                    for y in (0..H).rev() {
                        ui.horizontal(|ui| {
                            for x in 0..W {
                                let i = y * W + x;
                                let (text, fill, mut stroke) = cell_style(self.board[i]);
                                // 診断セルを赤枠で強調
                                if self
                                    .sc_diag_cells
                                    .iter()
                                    .any(|&(dx, dy)| dx == x && dy == y)
                                {
                                    stroke = egui::Stroke::new(2.5, Color32::RED);
                                }
                                let btn = egui::Button::new(RichText::new(text).size(12.0))
                                    .min_size(cell_size)
                                    .fill(fill)
                                    .stroke(stroke);
                                let resp = ui.add(btn);
                                if resp.clicked_by(egui::PointerButton::Primary) {
                                    let shift = ui.input(|inp| inp.modifiers.shift);
                                    if shift {
                                        self.board[i] = cycle_color(self.board[i]);
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
                });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}
impl App {
    fn start_run(&mut self) {
        // 準備
        let threshold = self.threshold.clamp(1, 19);
        let lru_limit = (self.lru_k.clamp(10, 1000) as usize) * 1000;
        let outfile = if let Some(p) = &self.out_path {
            p.clone()
        } else {
            std::path::PathBuf::from(&self.out_name)
        };
        // 盤面を文字配列へ
        let board_chars: Vec<char> = self.board.iter().map(|c| c.label_char()).collect();

        let (tx, rx) = unbounded::<Message>();
        self.rx = Some(rx);
        self.running = true;
        self.preview = None;
        self.log_lines.clear();
        self.stats.profile = ProfileTotals::default();

        let abort = self.abort_flag.clone();
        abort.store(false, Ordering::Relaxed);

        // 停滞比
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

// ======== ここから：ビットボード最適化版 ========

// 6×14=84 マスを u128 にパック
type BB = u128;
const COL_BITS: usize = H; // 14

// 各端マスク（列境界/上下境界の越境を防ぐ）
const fn top_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS + (COL_BITS - 1));
        x += 1;
    }
    m
}
const fn bottom_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS);
        x += 1;
    }
    m
}
const TOP_MASK: BB = top_mask();
const BOTTOM_MASK: BB = bottom_mask();
const LEFTCOL_MASK: BB = (1u128 << COL_BITS) - 1;
const RIGHTCOL_MASK: BB = ((1u128 << COL_BITS) - 1) << ((W - 1) * COL_BITS);

// ★ 盤面全体の有効マスク
const fn board_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= ((1u128 << COL_BITS) - 1) << (x * COL_BITS);
        x += 1;
    }
    m
}
const BOARD_MASK: BB = board_mask();

#[inline(always)]
fn pack_cols(cols: &[[u16; W]; 4]) -> [BB; 4] {
    let mut out = [0u128; 4];
    for c in 0..4 {
        let mut acc: BB = 0;
        for x in 0..W {
            acc |= (cols[c][x] as BB) << (x * COL_BITS);
        }
        out[c] = acc;
    }
    out
}

#[inline(always)]
fn unpack_mask_to_cols(mask: BB) -> [u16; W] {
    let mut out = [0u16; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = ((mask >> (x * COL_BITS)) as u16) & MASK14;
    }
    out
}

#[inline(always)]
fn neighbors(bits: BB) -> BB {
    let v_up = (bits & !TOP_MASK) << 1;
    let v_down = (bits & !BOTTOM_MASK) >> 1;
    let h_left = (bits & !LEFTCOL_MASK) >> COL_BITS;
    let h_right = (bits & !RIGHTCOL_MASK) << COL_BITS;
    v_up | v_down | h_left | h_right
}

// ======== 以降：探索ロジック（CLI版相当） ========

// 入力（A..M/N/X/./0..3）を元に抽象情報
struct AbstractInfo {
    labels: Vec<char>,
    adj: Vec<Vec<usize>>,
}

fn build_abstract_info(board: &[char]) -> AbstractInfo {
    let mut labels = Vec::new();
    for &v in board {
        if ('A'..='M').contains(&v) && !labels.contains(&v) {
            labels.push(v);
        }
    }
    let mut label_idx = HashMap::new();
    for (i, &c) in labels.iter().enumerate() {
        label_idx.insert(c, i);
    }
    let n = labels.len();
    let mut adj = vec![Vec::<usize>::new(); n];
    let dirs = [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)];
    for x in 0..W {
        for y in 0..H {
            let v = board[y * W + x];
            if !('A'..='M').contains(&v) {
                continue;
            }
            let id = label_idx[&v];
            for (dx, dy) in dirs {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize {
                    continue;
                }
                let w = board[(ny as usize) * W + (nx as usize)];
                if ('A'..='M').contains(&w) && w != v {
                    let nb = label_idx[&w];
                    if !adj[id].contains(&nb) {
                        adj[id].push(nb);
                    }
                }
            }
        }
    }
    AbstractInfo { labels, adj }
}

// 4) 彩色: DSATUR 風で高速化（ビット集合）
fn enumerate_colorings_fast(info: &AbstractInfo) -> Vec<Vec<u8>> {
    let n = info.labels.len();
    if n == 0 {
        return vec![Vec::new()];
    }

    // 隣接を bitset に
    let mut adj = vec![0u16; n];
    for (v, adjv) in adj.iter_mut().enumerate() {
        let mut m = 0u16;
        for &u in &info.adj[v] {
            m |= 1u16 << u;
        }
        *adjv = m;
    }

    // DSATUR: 彩色飽和度最大→次数最大
    let mut color = vec![4u8; n]; // 0..=3, 4=未彩色
    let mut used_mask = vec![0u8; n]; // 4bit: 近傍で使われた色

    let mut out = Vec::new();
    fn dfs(
        vleft: usize,
        total_n: usize,
        adj: &[u16],
        color: &mut [u8],
        used_mask: &mut [u8],
        out: &mut Vec<Vec<u8>>,
        max_used: u8, // 既に使われた最大色（0始まり）。新規色は max_used+1 のみ許可
    ) {
        if vleft == 0 {
            out.push(color.to_vec());
            return;
        }

        // 次に塗る頂点を選択（DSATUR）
        let mut pick = None;
        let mut best_sat = -1i32;
        let mut best_deg = -1i32;
        for v in 0..color.len() {
            if color[v] != 4 {
                continue;
            }
            let sat = used_mask[v].count_ones() as i32;
            let deg = adj[v].count_ones() as i32;
            if sat > best_sat || (sat == best_sat && deg > best_deg) {
                best_sat = sat;
                best_deg = deg;
                pick = Some(v);
            }
        }
        let v = pick.unwrap();

        // 使える色を列挙（4色から used を除く）+ 対称性破り
        let forbid = used_mask[v];
        let mut new_color_limit = (max_used + 1).min(3);
        if vleft == total_n {
            new_color_limit = 0;
        } // 最初の1手は 0 のみ
        for c in 0u8..=new_color_limit {
            if ((forbid >> c) & 1) != 0 {
                continue;
            }
            color[v] = c;

            // 近傍の used_mask を更新
            let mut touched = 0u16;
            let mut nb = adj[v];
            while nb != 0 {
                let u = nb.trailing_zeros() as usize;
                nb &= nb - 1;
                if color[u] == 4 {
                    used_mask[u] |= 1u8 << c;
                    touched |= 1u16 << u;
                }
            }
            let next_max_used = if c > max_used { c } else { max_used };
            dfs(
                vleft - 1,
                total_n,
                adj,
                color,
                used_mask,
                out,
                next_max_used,
            );

            // ロールバック
            color[v] = 4;
            let mut t = touched;
            while t != 0 {
                let u = t.trailing_zeros() as usize;
                t &= t - 1;
                used_mask[u] &= !(1u8 << c);
            }
        }
    }
    dfs(n, n, &adj, &mut color, &mut used_mask, &mut out, 0);
    out
}

fn apply_coloring_to_template(base: &[char], map: &HashMap<char, u8>) -> Vec<TCell> {
    base.iter()
        .map(|&v| {
            if ('A'..='M').contains(&v) {
                TCell::Fixed(map[&v])
            } else if v == 'N' {
                TCell::Any
            } else if v == 'X' {
                TCell::Any4
            } else if v == '.' {
                TCell::Blank
            } else if ('0'..='3').contains(&v) {
                TCell::Fixed(v as u8 - b'0')
            } else {
                TCell::Blank
            }
        })
        .collect()
}

// ★ 小連鎖用：下2段の 'N' を現在手の色（2色）に制限したテンプレート
fn apply_coloring_to_template_with_bottom2_pair(
    base: &[char],
    map: &HashMap<char, u8>,
    pair: (u8, u8),
) -> Vec<TCell> {
    let allow_mask: u8 = (1u8 << (pair.0.min(3))) | (1u8 << (pair.1.min(3)));
    // 行優先のインデックス templ[y*W + x] に合わせて代入していく
    let mut out = vec![TCell::Blank; W * H];
    for x in 0..W {
        let mut n_used_in_col = 0u8; // その列で下から数えて N を2つまで制限
        for y in 0..H {
            // y=0 が最下段
            let idx = y * W + x;
            let v = base[idx];
            let cell = if ('A'..='M').contains(&v) {
                TCell::Fixed(map[&v])
            } else if v == 'N' {
                if n_used_in_col < 2 {
                    n_used_in_col += 1;
                    TCell::AnyMask(allow_mask)
                } else {
                    TCell::Any
                }
            } else if v == 'X' {
                TCell::Any4
            } else if v == '.' {
                TCell::Blank
            } else if ('0'..='3').contains(&v) {
                TCell::Fixed(v as u8 - b'0')
            } else {
                TCell::Blank
            };
            out[idx] = cell;
        }
    }
    out
}

// 列DP（通り数）
fn count_column_candidates_dp(col: &[TCell]) -> BigUint {
    let mut dp0 = BigUint::one(); // belowBlank=false
    let mut dp1 = BigUint::zero(); // belowBlank=true
    for &cell in col.iter().take(H) {
        let mut ndp0 = BigUint::zero();
        let mut ndp1 = BigUint::zero();
        match cell {
            TCell::Blank => {
                ndp1 += &dp0;
                ndp1 += &dp1;
            }
            TCell::Any4 => {
                if !dp0.is_zero() {
                    ndp0 += dp0.clone() * BigUint::from(4u32);
                }
            }
            TCell::Any => {
                ndp1 += &dp0;
                ndp1 += &dp1;
                if !dp0.is_zero() {
                    ndp0 += dp0 * BigUint::from(4u32);
                }
            }
            TCell::AnyMask(mask) => {
                ndp1 += &dp0;
                ndp1 += &dp1;
                if !dp0.is_zero() {
                    let k: u32 = (mask & 0x0F).count_ones();
                    if k != 0 {
                        ndp0 += dp0 * BigUint::from(k);
                    }
                }
            }
            TCell::Fixed(_) => {
                ndp0 += &dp0;
            }
        }
        dp0 = ndp0;
        dp1 = ndp1;
    }
    dp0 + dp1
}

// 列ストリーミング列挙（従来版）
fn stream_column_candidates<F: FnMut([u16; 4])>(col: &[TCell], mut yield_masks: F) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        yield_masks: &mut F,
    ) {
        if y >= H {
            yield_masks(*masks);
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::AnyMask(mask) => {
                rec(y + 1, true, col, masks, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        if ((mask >> c) & 1) != 0 {
                            masks[c] |= 1 << y;
                            rec(y + 1, false, col, masks, yield_masks);
                            masks[c] &= !(1 << y);
                        }
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    rec(0, false, col, &mut masks, &mut yield_masks);
}

// 列ストリーミング列挙（計測版：列挙時間＝再帰本体、yield 時間は除外）
fn stream_column_candidates_timed<F: FnMut([u16; 4])>(
    col: &[TCell],
    enum_time: &mut Duration,
    mut yield_masks: F,
) {
    fn rec<F: FnMut([u16; 4])>(
        y: usize,
        below_blank: bool,
        col: &[TCell],
        masks: &mut [u16; 4],
        enum_time: &mut Duration,
        last_start: &mut Instant,
        yield_masks: &mut F,
    ) {
        if y >= H {
            *enum_time += last_start.elapsed();
            yield_masks(*masks);
            *last_start = Instant::now();
            return;
        }
        match col[y] {
            TCell::Blank => rec(y + 1, true, col, masks, enum_time, last_start, yield_masks),
            TCell::Any4 => {
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::Any => {
                rec(y + 1, true, col, masks, enum_time, last_start, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        masks[c] |= 1 << y;
                        rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                        masks[c] &= !(1 << y);
                    }
                }
            }
            TCell::AnyMask(mask) => {
                rec(y + 1, true, col, masks, enum_time, last_start, yield_masks);
                if !below_blank {
                    for c in 0..4 {
                        if ((mask >> c) & 1) != 0 {
                            masks[c] |= 1 << y;
                            rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                            masks[c] &= !(1 << y);
                        }
                    }
                }
            }
            TCell::Fixed(c) => {
                if !below_blank {
                    masks[c as usize] |= 1 << y;
                    rec(y + 1, false, col, masks, enum_time, last_start, yield_masks);
                    masks[c as usize] &= !(1 << y);
                }
            }
        }
    }
    let mut masks = [0u16; 4];
    let mut last_start = Instant::now();
    rec(
        0,
        false,
        col,
        &mut masks,
        enum_time,
        &mut last_start,
        &mut yield_masks,
    );
    *enum_time += last_start.elapsed();
}

// 5) 列候補生成のハイブリッド（小さい列だけ前展開）
enum ColGen {
    Pre(Vec<[u16; 4]>),
    Stream(Vec<TCell>),
}
fn build_colgen(col: &[TCell], cnt: &BigUint) -> ColGen {
    if cnt.bits() <= 11 {
        let mut v = Vec::new();
        stream_column_candidates(col, |m| v.push(m));
        ColGen::Pre(v)
    } else {
        ColGen::Stream(col.to_vec())
    }
}

// ====== 落下：スカラ版と PEXT/PDEP 版 ======
#[inline(always)]
fn fall_cols(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] & MASK14;
        let c1 = cols_in[1][x] & MASK14;
        let c2 = cols_in[2][x] & MASK14;
        let c3 = cols_in[3][x] & MASK14;
        let mut occ = c0 | c1 | c2 | c3;

        let mut dst: usize = 0;
        while occ != 0 {
            let bit = occ & occ.wrapping_neg();
            let color = if (c0 & bit) != 0 {
                0
            } else if (c1 & bit) != 0 {
                1
            } else if (c2 & bit) != 0 {
                2
            } else {
                3
            };
            out[color][x] |= 1u16 << dst;
            dst += 1;
            occ &= occ - 1;
        }
    }

    out
}

#[inline(always)]
fn fall_cols_fast(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("bmi2") {
            unsafe {
                return fall_cols_bmi2(cols_in);
            }
        }
    }
    fall_cols(cols_in)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn fall_cols_bmi2(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_pdep_u32, _pext_u32};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_pdep_u32, _pext_u32};

    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] as u32;
        let c1 = cols_in[1][x] as u32;
        let c2 = cols_in[2][x] as u32;
        let c3 = cols_in[3][x] as u32;

        let occ = (c0 | c1 | c2 | c3) & (MASK14 as u32);
        let k = occ.count_ones();
        if k == 0 {
            continue;
        }
        let base = (1u32 << k) - 1;

        let s0 = _pext_u32(c0, occ);
        let s1 = _pext_u32(c1, occ);
        let s2 = _pext_u32(c2, occ);
        let s3 = _pext_u32(c3, occ);

        out[0][x] = _pdep_u32(s0, base) as u16;
        out[1][x] = _pdep_u32(s1, base) as u16;
        out[2][x] = _pdep_u32(s2, base) as u16;
        out[3][x] = _pdep_u32(s3, base) as u16;
    }

    out
}

// 連結抽出（旧ビット列版：他箇所でも使用しているので残す）
#[allow(dead_code)]
#[inline(always)]
fn component_from_seed_cols(s: &[u16; W], seed_x: usize, seed_bits: u16) -> [u16; W] {
    let mut comp = [0u16; W];
    let mut frontier = [0u16; W];
    comp[seed_x] = seed_bits;
    frontier[seed_x] = seed_bits;
    loop {
        let mut changed = false;
        let mut next = [0u16; W];
        for x in 0..W {
            let mut nb = ((frontier[x] << 1) & MASK14) | (frontier[x] >> 1);
            if x > 0 {
                nb |= frontier[x - 1];
            }
            if x + 1 < W {
                nb |= frontier[x + 1];
            }
            next[x] = nb & s[x];
        }
        for x in 0..W {
            let add = next[x] & !comp[x];
            if add != 0 {
                comp[x] |= add;
                frontier[x] = add;
                changed = true;
            } else {
                frontier[x] = 0;
            }
        }
        if !changed {
            break;
        }
    }
    comp
}

#[inline(always)]
fn compute_erase_mask_cols(cols: &[[u16; W]; 4]) -> [u16; W] {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }

        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
            }
            s &= !comp;
        }
    }

    unpack_mask_to_cols(clear_all)
}

// ★ 4個消しモード用：マスク + “5個以上あったか” + “4個があったか”
#[inline(always)]
fn compute_erase_mask_and_flags(cols: &[[u16; W]; 4]) -> ([u16; W], bool, bool) {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;
    let mut had_ge5 = false;
    let mut had_four = false;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }
        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
                if sz == 4 {
                    had_four = true;
                } else {
                    had_ge5 = true;
                }
            }
            s &= !comp;
        }
    }

    (unpack_mask_to_cols(clear_all), had_ge5, had_four)
}

#[inline(always)]
fn apply_given_clear_and_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut next = [[0u16; W]; 4];

    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (next_col, pre_col) in next.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(next_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    fall_cols_fast(&next)
}

// ★ 追加：どの色か一色でも“4つ以上”あれば次の消去が起こりうる
#[inline(always)]
fn any_color_has_four(cols: &[[u16; W]; 4]) -> bool {
    let bb = pack_cols(cols);
    (bb[0].count_ones() >= 4)
        || (bb[1].count_ones() >= 4)
        || (bb[2].count_ones() >= 4)
        || (bb[3].count_ones() >= 4)
}

#[inline(always)]
fn apply_erase_and_fall_cols(cols: &[[u16; W]; 4]) -> (bool, [[u16; W]; 4]) {
    if !any_color_has_four(cols) {
        return (false, *cols);
    }

    let clear = compute_erase_mask_cols(cols);
    let any = (0..W).any(|x| clear[x] != 0);
    if !any {
        (false, *cols)
    } else {
        (true, apply_given_clear_and_fall(cols, &clear))
    }
}

// ★ 4個消しモード用
enum StepExact {
    NoClear,
    Cleared([[u16; W]; 4]),
    Illegal,
}

#[inline(always)]
fn apply_erase_and_fall_exact4(cols: &[[u16; W]; 4]) -> StepExact {
    if !any_color_has_four(cols) {
        return StepExact::NoClear;
    }
    let (clear, had_ge5, had_four) = compute_erase_mask_and_flags(cols);
    if had_ge5 {
        return StepExact::Illegal;
    }
    let any = (0..W).any(|x| clear[x] != 0);
    if !any || !had_four {
        StepExact::NoClear
    } else {
        StepExact::Cleared(apply_given_clear_and_fall(cols, &clear))
    }
}
// 列 x への 4色一括代入（ループ展開）
#[inline(always)]
fn assign_col_unrolled(cols: &mut [[u16; W]; 4], x: usize, masks: &[u16; 4]) {
    debug_assert!(x < W);
    cols[0][x] = masks[0];
    cols[1][x] = masks[1];
    cols[2][x] = masks[2];
    cols[3][x] = masks[3];
}

// 列 x をゼロクリア（ループ展開）
#[inline(always)]
fn clear_col_unrolled(cols: &mut [[u16; W]; 4], x: usize) {
    debug_assert!(x < W);
    cols[0][x] = 0;
    cols[1][x] = 0;
    cols[2][x] = 0;
    cols[3][x] = 0;
}

// E1単一連結 + 追加条件 + T到達（最適化版）
#[inline(always)]
fn reaches_t_from_pre_single_e1(pre: &[[u16; W]; 4], t: u32, exact_four_only: bool) -> bool {
    if exact_four_only {
        let mut potential: u32 = 0;
        for col in pre.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < t {
            return false;
        }
    }

    let bb_pre = pack_cols(pre);
    let (clear_bb, total_cnt) = {
        let mut clr: BB = 0;
        let mut tot: u32 = 0;
        for &bb in bb_pre.iter() {
            if bb.count_ones() < 4 {
                continue;
            }
            let mut s = bb;
            while s != 0 {
                let seed = s & (!s + 1);
                let mut comp = seed;
                let mut frontier = seed;
                loop {
                    let grow = neighbors(frontier) & bb & !comp;
                    if grow == 0 {
                        break;
                    }
                    comp |= grow;
                    frontier = grow;
                }
                let sz = comp.count_ones();
                if sz >= 4 {
                    clr |= comp;
                    tot = tot.saturating_add(sz);
                }
                s &= !comp;
            }
        }
        (clr, tot)
    };

    let total = total_cnt;
    if total == 0 {
        return false;
    }

    // exact4: 初回消去が4以外なら不成立
    if exact_four_only && total != 4 {
        return false;
    }

    // 空白隣接とオーバーハングの簡易チェック
    let occ_bb = bb_pre[0] | bb_pre[1] | bb_pre[2] | bb_pre[3];
    let blank_bb = BOARD_MASK & !occ_bb;
    if neighbors(clear_bb) & blank_bb == 0 {
        return false;
    }

    let mut ok_overhang = false;
    for x in 0..W {
        let clear_col: u16 = ((clear_bb >> (x * COL_BITS)) as u16) & MASK14;
        if clear_col == 0 {
            continue;
        }
        let occ_col: u16 = ((occ_bb >> (x * COL_BITS)) as u16) & MASK14;

        let top_y = 15 - clear_col.leading_zeros() as usize;
        let above = (occ_col & !clear_col) >> (top_y + 1);
        let run = (above.trailing_ones()) as usize;
        if run == 0 {
            ok_overhang = true;
            break;
        }
    }
    if !ok_overhang {
        return false;
    }

    // E1 単一連結チェック
    let seed = clear_bb & (!clear_bb + 1);
    let mut comp = seed;
    let mut frontier = seed;
    loop {
        let grow = neighbors(frontier) & clear_bb & !comp;
        if grow == 0 {
            break;
        }
        comp |= grow;
        frontier = grow;
    }
    if comp.count_ones() != total {
        return false;
    }

    // 初回消去後の盤面
    let mut cur;
    {
        let mut work = [[0u16; W]; 4];
        let clear_cols = unpack_mask_to_cols(clear_bb);
        for x in 0..W {
            let inv = (!clear_cols[x]) & MASK14;
            work[0][x] = pre[0][x] & inv;
            work[1][x] = pre[1][x] & inv;
            work[2][x] = pre[2][x] & inv;
            work[3][x] = pre[3][x] & inv;
        }
        cur = fall_cols_fast(&work);
    }
    if t == 1 {
        return true;
    }

    // 残り (t-1) 連鎖のポテンシャル上限
    {
        let mut potential: u32 = 0;
        for col in cur.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < (t - 1) {
            return false;
        }
    }

    if !exact_four_only {
        for _ in 2..=t {
            let (erased, next) = apply_erase_and_fall_cols(&cur);
            if !erased {
                return false;
            }
            cur = next;
        }
        true
    } else {
        for _ in 2..=t {
            match apply_erase_and_fall_exact4(&cur) {
                StepExact::Illegal => return false,
                StepExact::NoClear => return false,
                StepExact::Cleared(next) => {
                    cur = next;
                }
            }
        }
        true
    }
}

// ========== 追撃最適化：占有比較の u128 化 & ハッシュ ==========
#[inline(always)]
fn choose_mirror_by_occupancy(cols: &[[u16; W]; 4]) -> Option<bool> {
    let mut packed: u128 = 0;
    let mut rev: u128 = 0;
    for x in 0..W {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) as u128;
        packed |= occ << (x * 16);
        rev |= occ << ((W - 1 - x) * 16);
    }
    if packed < rev {
        Some(false)
    } else if packed > rev {
        Some(true)
    } else {
        None
    }
}

#[inline(always)]
fn map_code_lut(entry: &mut u8, next: &mut u8) -> u64 {
    if *entry == u8::MAX {
        *entry = *next;
        *next = next.wrapping_add(1);
    }
    *entry as u64
}

#[inline(always)]
fn canonical_hash64_oriented_bits(cols: &[[u16; W]; 4], mirror: bool) -> u64 {
    const P: u64 = 1099511628211;
    const O: u64 = 14695981039346656037;

    let mut p_pow = [1u64; 15];
    for i in 1..15 {
        p_pow[i] = p_pow[i - 1].wrapping_mul(P);
    }
    #[inline(always)]
    fn mul_pow(h: u64, pp: &[u64; 15], k: usize) -> u64 {
        h.wrapping_mul(pp[k])
    }

    let mut h = O;
    let mut map: [u8; 4] = [u8::MAX; 4];
    let mut next: u8 = 1;

    if !mirror {
        for xi in 0..W {
            let c0 = cols[0][xi];
            let c1 = cols[1][xi];
            let c2 = cols[2][xi];
            let c3 = cols[3][xi];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            let mut prev_y: i32 = -1;
            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }
            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    } else {
        for xr in (0..W).rev() {
            let c0 = cols[0][xr];
            let c1 = cols[1][xr];
            let c2 = cols[2][xr];
            let c3 = cols[3][xr];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            let mut prev_y: i32 = -1;
            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }
            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    }
    h
}

// 占有で決まらない（左右対称）場合のみ、両向き計算して小さい方
#[inline(always)]
fn canonical_hash64_fast(cols: &[[u16; W]; 4]) -> (u64, bool) {
    if let Some(mirror) = choose_mirror_by_occupancy(cols) {
        let h = canonical_hash64_oriented_bits(cols, mirror);
        (h, mirror)
    } else {
        let h0 = canonical_hash64_oriented_bits(cols, false);
        let h1 = canonical_hash64_oriented_bits(cols, true);
        if h0 <= h1 {
            (h0, false)
        } else {
            (h1, true)
        }
    }
}

fn encode_canonical_string(cols: &[[u16; W]; 4], mirror: bool) -> String {
    let mut map: [u8; 4] = [0; 4];
    let mut next: u8 = b'A';
    let mut s = String::with_capacity(W * H);
    if !mirror {
        for x in 0..W {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    } else {
        for x in (0..W).rev() {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    }
    s
}

fn serialize_board_from_cols(cols: &[[u16; W]; 4]) -> Vec<String> {
    let mut rows = Vec::with_capacity(H);
    for y in 0..H {
        let mut line = String::with_capacity(W);
        for x in 0..W {
            let bit = 1u16 << y;
            let ch = if cols[0][x] & bit != 0 {
                '0'
            } else if cols[1][x] & bit != 0 {
                '1'
            } else if cols[2][x] & bit != 0 {
                '2'
            } else if cols[3][x] & bit != 0 {
                '3'
            } else {
                '.'
            };
            line.push(ch);
        }
        rows.push(line);
    }
    rows
}
fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in s.as_bytes() {
        h ^= b as u32;
        h = h
            .wrapping_add(h << 1)
            .wrapping_add(h << 4)
            .wrapping_add(h << 7)
            .wrapping_add(h << 8)
            .wrapping_add(h << 24);
    }
    h
}

// 3) JSON を手組みで生成
#[inline(always)]
fn make_json_line_str(
    key: &str,
    hash: u32,
    chains: u32,
    rows: &[String],
    mapping: &HashMap<char, u8>,
    mirror: bool,
) -> String {
    let mut s = String::with_capacity(256);
    s.push('{');
    s.push_str(r#""key":"#);
    s.push('"');
    for ch in key.chars() {
        if ch == '"' {
            s.push('\\');
        }
        s.push(ch);
    }
    s.push('"');
    s.push(',');

    s.push_str(r#""hash":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", hash));
    s.push(',');
    s.push_str(r#""chains":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", chains));
    s.push(',');

    s.push_str(r#""pre_chain_board":["#);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        for ch in row.chars() {
            if ch == '"' {
                s.push('\\');
            }
            s.push(ch);
        }
        s.push('"');
    }
    s.push_str("],");

    s.push_str(r#""example_mapping":{"#);
    let mut keys: Vec<_> = mapping.keys().copied().collect();
    keys.sort_unstable();
    let mut first = true;
    for k in keys {
        if !first {
            s.push(',');
        }
        first = false;
        s.push('"');
        s.push(k);
        s.push_str(r#"":"#);
        let _ = std::fmt::write(&mut s, format_args!("{}", mapping[&k]));
    }
    s.push_str("},");

    s.push_str(r#""mirror":"#);
    s.push_str(if mirror { "true" } else { "false" });
    s.push('}');
    s
}

// 近似LRU（ローカル専用）
struct ApproxLru {
    limit: usize,
    map: U64Map<bool>, // 現方針では未参照でもOK
    q: VecDeque<u64>,
}
impl ApproxLru {
    fn new(limit: usize) -> Self {
        let cap = (limit.saturating_mul(11) / 10).max(16);
        let map: U64Map<bool> =
            std::collections::HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());
        let q = VecDeque::with_capacity(cap);
        Self { limit, map, q }
    }
    #[allow(dead_code)]
    fn get(&self, k: u64) -> Option<bool> {
        self.map.get(&k).copied()
    }
    #[allow(dead_code)]
    fn insert(&mut self, k: u64, v: bool) {
        use std::collections::hash_map::Entry;
        match self.map.entry(k) {
            Entry::Vacant(e) => {
                e.insert(v);
                self.q.push_back(k);
                let cap = (self.limit as f64 * 1.1) as usize;
                if self.q.len() > cap {
                    let to_delete = self.q.len() - self.limit;
                    for _ in 0..to_delete {
                        if let Some(kk) = self.q.pop_front() {
                            self.map.remove(&kk);
                        }
                    }
                }
            }
            Entry::Occupied(mut e) => {
                e.insert(v);
            }
        }
    }
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.map.len()
    }
}

// DFS（列順最適化、ハイブリッド列候補、ローカルLRU、グローバル一意集合、バッチ出力）
#[allow(clippy::too_many_arguments)]
fn dfs_combine_parallel(
    depth: usize,
    cols0: &mut [[u16; W]; 4],
    gens: &[ColGen; W],
    order: &[usize],
    threshold: u32,
    exact_four_only: bool,
    _memo: &mut ApproxLru,
    local_output_once: &mut U64Set,
    global_output_once: &Arc<DU64Set>,
    _global_memo: &Arc<DU64Map<bool>>,
    map_label_to_color: &HashMap<char, u8>,
    batch: &mut Vec<String>,
    batch_sender: &Sender<Vec<String>>,
    stat_sender: &Sender<StatDelta>,
    // 計測
    profile_enabled: bool,
    time_batch: &mut TimeDelta,

    nodes_batch: &mut u64,
    leaves_batch: &mut u64,
    outputs_batch: &mut u64,
    pruned_batch: &mut u64,
    lhit_batch: &mut u64,
    ghit_batch: &mut u64,
    mmiss_batch: &mut u64,
    preview_ok: bool,
    preview_tx: &Sender<Message>,
    last_preview: &mut Instant,
    _lru_limit: usize,
    t0: Instant,
    abort: &AtomicBool,
    placed_total: u32,
    remain_suffix: &[u16],
) -> Result<()> {
    if abort.load(Ordering::Relaxed) {
        return Ok(());
    }
    *nodes_batch += 1;
    if profile_enabled {
        time_batch.dfs_counts[depth].nodes += 1;
    }

    // 葉
    if depth == W {
        *leaves_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].leaves += 1;
        }

        if placed_total < 4 * threshold {
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_tshort += 1;
            }
            return Ok(());
        }
        if !any_color_has_four(cols0) {
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_e1_impossible += 1;
            }
            return Ok(());
        }

        let pre = *cols0;

        if profile_enabled {
            time_batch.dfs_counts[depth].memo_miss += 1;
        }
        *mmiss_batch += 1;
        let reached = prof!(
            profile_enabled,
            time_batch.dfs_times[depth].leaf_memo_miss_compute,
            { reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only) }
        );
        if !reached {
            if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
                && (*nodes_batch > 0
                    || *leaves_batch > 0
                    || *outputs_batch > 0
                    || *pruned_batch > 0
                    || *lhit_batch > 0
                    || *ghit_batch > 0
                    || *mmiss_batch > 0)
            {
                let _ = stat_sender.send(StatDelta {
                    nodes: *nodes_batch,
                    leaves: *leaves_batch,
                    outputs: *outputs_batch,
                    pruned: *pruned_batch,
                    lhit: *lhit_batch,
                    ghit: *ghit_batch,
                    mmiss: *mmiss_batch,
                });
                *nodes_batch = 0;
                *leaves_batch = 0;
                *outputs_batch = 0;
                *pruned_batch = 0;
                *lhit_batch = 0;
                *ghit_batch = 0;
                *mmiss_batch = 0;

                if profile_enabled && time_delta_has_any(time_batch) {
                    let td = time_batch.clone();
                    *time_batch = TimeDelta::default();
                    let _ = preview_tx.send(Message::TimeDelta(td));
                }
            }
            return Ok(());
        }

        // 正規化キー
        let (key64, mirror) = prof!(profile_enabled, time_batch.dfs_times[depth].leaf_hash, {
            canonical_hash64_fast(&pre)
        });

        if !local_output_once.contains(&key64) {
            // 近似的なグローバル一意集合
            const OUTPUT_SET_CAP: usize = 2_000_000;
            const OUTPUT_SAMPLE_MASK: u64 = 0x7; // 1/8 サンプリング
            const OUTPUT_SAMPLE_MATCH: u64 = 0;

            let under_cap = global_output_once.len() < OUTPUT_SET_CAP;
            let sampled = (key64 & OUTPUT_SAMPLE_MASK) == OUTPUT_SAMPLE_MATCH;
            let should_insert_global = preview_ok && under_cap && sampled;

            let is_new = if should_insert_global {
                global_output_once.insert(key64)
            } else {
                !global_output_once.contains(&key64)
            };

            if is_new {
                local_output_once.insert(key64);

                if preview_ok && last_preview.elapsed() >= Duration::from_millis(3000) {
                    let _ = preview_tx.send(Message::Preview(pre));
                    *last_preview = Instant::now();
                }

                // 出力整形
                let line = prof!(
                    profile_enabled,
                    time_batch.dfs_times[depth].out_serialize,
                    {
                        let key_str = encode_canonical_string(&pre, mirror);
                        let hash = fnv1a32(&key_str);
                        let rows = serialize_board_from_cols(&pre);
                        make_json_line_str(
                            &key_str,
                            hash,
                            threshold,
                            &rows,
                            map_label_to_color,
                            mirror,
                        )
                    }
                );
                batch.push(line);
                *outputs_batch += 1;

                if batch.len() >= 2048 {
                    let out = std::mem::take(batch);
                    let _ = batch_sender.send(out);
                }
            }
        }

        // 統計フラッシュ
        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    // 上界枝刈り
    let pruned_now = prof!(profile_enabled, time_batch.dfs_times[depth].upper_bound, {
        let placed = placed_total;
        let remain_cap = *remain_suffix.get(depth).unwrap_or(&0) as u32;
        placed + remain_cap < 4 * threshold
    });
    if pruned_now {
        *pruned_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].pruned_upper += 1;
        }
        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    let x = order[depth];
    match &gens[x] {
        ColGen::Pre(v) => {
            if profile_enabled {
                time_batch.dfs_counts[depth].cand_generated += v.len() as u64;
            }
            for &masks in v {
                if abort.load(Ordering::Relaxed) {
                    return Ok(());
                }
                prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                    assign_col_unrolled(cols0, x, &masks);
                });
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                let _ = dfs_combine_parallel(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    _memo,
                    local_output_once,
                    global_output_once,
                    _global_memo,
                    map_label_to_color,
                    batch,
                    batch_sender,
                    stat_sender,
                    profile_enabled,
                    time_batch,
                    nodes_batch,
                    leaves_batch,
                    outputs_batch,
                    pruned_batch,
                    lhit_batch,
                    ghit_batch,
                    mmiss_batch,
                    preview_ok,
                    preview_tx,
                    last_preview,
                    _lru_limit,
                    t0,
                    abort,
                    placed_total + add,
                    remain_suffix,
                );
                clear_col_unrolled(cols0, x);
            }
        }
        ColGen::Stream(colv) => {
            if profile_enabled {
                let mut enum_time = Duration::ZERO;
                stream_column_candidates_timed(colv, &mut enum_time, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    time_batch.dfs_counts[depth].cand_generated += 1;

                    prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                        assign_col_unrolled(cols0, x, &masks);
                    });

                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
                time_batch.dfs_times[depth].gen_candidates += enum_time;
            } else {
                stream_column_candidates(colv, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    assign_col_unrolled(cols0, x, &masks);
                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
            }
        }
    }

    Ok(())
}
// 検索メイン（並列化＋シングル writer スレッド＋集約 Progress）
#[allow(clippy::too_many_arguments)]
fn run_search(
    base_board: Vec<char>,
    threshold: u32,
    lru_limit: usize,
    outfile: std::path::PathBuf,
    tx: Sender<Message>,
    abort: Arc<AtomicBool>,
    stop_progress_plateau: f32,
    exact_four_only: bool,
    profile_enabled: bool,
) -> Result<()> {
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        let _ = tx.send(Message::Log(
            "抽象ラベルの4色彩色が存在しないため、探索を終了します。".into(),
        ));
        let _ = tx.send(Message::Finished(Stats::default()));
        return Ok(());
    }
    let _ = tx.send(Message::Log(format!(
        "抽象ラベル={} / 彩色候補={} / 4個消しモード={} / 計測={}",
        info.labels.iter().collect::<String>(),
        colorings.len(),
        if exact_four_only { "ON" } else { "OFF" },
        if profile_enabled { "ON" } else { "OFF" }
    )));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let bmi2 = std::is_x86_feature_detected!("bmi2");
        let popcnt = std::is_x86_feature_detected!("popcnt");
        let _ = tx.send(Message::Log(format!(
            "CPU features: bmi2={} / popcnt={}",
            bmi2, popcnt
        )));
    }

    // 厳密総数（列DP）とメタ構築
    type Meta = (HashMap<char, u8>, [ColGen; W], [u8; W], Vec<usize>);
    let mut metas: Vec<Meta> = Vec::new();
    let mut total = BigUint::zero();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut cols: [Vec<TCell>; W] = array_init(|_| Vec::with_capacity(H));
        for x in 0..W {
            for y in 0..H {
                cols[x].push(templ[y * W + x]);
            }
        }
        let mut prod = BigUint::one();
        let mut impossible = false;
        let mut counts: [BigUint; W] = array_init(|_| BigUint::zero());
        let mut max_fill_arr: [u8; W] = [0; W];
        for x in 0..W {
            let cnt = count_column_candidates_dp(&cols[x]);
            if cnt.is_zero() {
                impossible = true;
                break;
            }
            counts[x] = cnt.clone();
            max_fill_arr[x] = compute_max_fill(&cols[x]);
            prod *= cnt;
        }
        if !impossible {
            total += prod;
            let mut order: Vec<usize> = (0..W).collect();
            order.sort_by(|&a, &b| counts[a].cmp(&counts[b]));
            let gens: [ColGen; W] = array_init(|x| build_colgen(&cols[x], &counts[x]));
            metas.push((map, gens, max_fill_arr, order));
        }
    }
    let _ = tx.send(Message::Log(format!(
        "厳密な総組合せ（列制約適用）: {}",
        total
    )));

    // writer
    let (wtx, wrx) = unbounded::<Vec<String>>();
    let tx_for_writer = tx.clone();
    let writer_handle = {
        let outfile = outfile.clone();
        thread::spawn(move || -> Result<()> {
            let mut io_time = Duration::ZERO;
            let file = File::create(&outfile)
                .with_context(|| format!("出力を作成できません: {}", outfile.display()))?;
            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
            while let Ok(batch) = wrx.recv() {
                let t0 = Instant::now();
                for line in batch {
                    writer.write_all(line.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
                io_time += t0.elapsed();
            }
            writer.flush()?;
            if io_time != Duration::ZERO {
                let mut td = TimeDelta::default();
                td.io_write_total = io_time;
                let _ = tx_for_writer.send(Message::TimeDelta(td));
            }
            Ok(())
        })
    };

    // 集約 Progress
    let (stx, srx) = unbounded::<StatDelta>();
    let t0 = Instant::now();
    let tx_progress = tx.clone();
    let total_clone = total.clone();
    let abort_for_agg = abort.clone();
    let global_output_once: Arc<DU64Set> =
        Arc::new(DU64Set::with_hasher(BuildNoHashHasher::default()));
    let global_memo: Arc<DU64Map<bool>> =
        Arc::new(DU64Map::with_hasher(BuildNoHashHasher::default()));

    let global_memo_for_agg = global_memo.clone();
    let lru_limit_for_agg = lru_limit;
    let agg_handle = thread::spawn(move || {
        let mut nodes: u64 = 0;
        let mut outputs: u64 = 0;
        let mut done = BigUint::zero();
        let mut pruned: u64 = 0;
        let mut lhit: u64 = 0;
        let mut ghit: u64 = 0;
        let mut mmiss: u64 = 0;
        let mut last_send = Instant::now();

        let mut last_output: u64 = 0;
        let mut progress_at_last_output: f64 = 0.0;
        let plateau: f64 = stop_progress_plateau as f64;

        loop {
            match srx.recv_timeout(Duration::from_millis(500)) {
                Ok(d) => {
                    nodes += d.nodes;
                    outputs += d.outputs;
                    if d.leaves > 0 {
                        done += BigUint::from(d.leaves);
                    }
                    pruned += d.pruned;
                    lhit += d.lhit;
                    ghit += d.ghit;
                    mmiss += d.mmiss;

                    if outputs > last_output {
                        last_output = outputs;
                        if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                            if tt > 0.0 {
                                progress_at_last_output = (td / tt).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    let dt = t0.elapsed().as_secs_f64();
                    let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                    let memo_len = global_memo_for_agg.len();
                    let st = Stats {
                        searching: false,
                        unique: outputs,
                        output: outputs,
                        nodes,
                        pruned,
                        memo_hit_local: lhit,
                        memo_hit_global: ghit,
                        memo_miss: mmiss,
                        total: total_clone.clone(),
                        done: done.clone(),
                        rate,
                        memo_len,
                        lru_limit: lru_limit_for_agg,
                        profile: ProfileTotals::default(),
                    };
                    let _ = tx_progress.send(Message::Finished(st));
                    break;
                }
            }

            if plateau > 0.0 {
                if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                    if tt > 0.0 {
                        let p = (td / tt).clamp(0.0, 1.0);
                        if p - progress_at_last_output >= plateau {
                            let msg = format!(
                                "早期終了: 進捗が {:.1}% 進む間に新規出力がありませんでした（しきい値 {:.1}%）",
                                (p - progress_at_last_output) * 100.0,
                                plateau * 100.0
                            );
                            let _ = tx_progress.send(Message::Log(msg));
                            abort_for_agg.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }

            if last_send.elapsed() >= Duration::from_millis(500) {
                let dt = t0.elapsed().as_secs_f64();
                let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                let memo_len = global_memo_for_agg.len();
                let st = Stats {
                    searching: true,
                    unique: outputs,
                    output: outputs,
                    nodes,
                    pruned,
                    memo_hit_local: lhit,
                    memo_hit_global: ghit,
                    memo_miss: mmiss,
                    total: total_clone.clone(),
                    done: done.clone(),
                    rate,
                    memo_len,
                    lru_limit: lru_limit_for_agg,
                    profile: ProfileTotals::default(),
                };
                let _ = tx_progress.send(Message::Progress(st));
                last_send = Instant::now();
            }
        }
    });

    // metas を並列探索
    metas.par_iter().enumerate().try_for_each(
        |(i, (map_label_to_color, gens, max_fill, order))| -> Result<()> {
            if abort.load(Ordering::Relaxed) {
                return Ok(());
            }

            let preview_ok = i == 0;
            let first_x = order[0];
            let mut first_candidates: Vec<[u16; 4]> = Vec::new();
            match &gens[first_x] {
                ColGen::Pre(v) => first_candidates.extend_from_slice(v),
                ColGen::Stream(colv) => {
                    stream_column_candidates(colv, |m| first_candidates.push(m));
                }
            }

            let mut remain_suffix: Vec<u16> = vec![0; W + 1];
            for d in (0..W).rev() {
                remain_suffix[d] = remain_suffix[d + 1] + max_fill[order[d]] as u16;
            }

            let threads = rayon::current_num_threads().max(1);
            if first_candidates.len() >= threads {
                first_candidates
                    .par_iter()
                    .try_for_each(|masks| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        let mut cols0 = [[0u16; W]; 4];
                        for c in 0..4 {
                            cols0[c][first_x] = masks[c];
                        }
                        let mut memo =
                            ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
                        let mut local_output_once: U64Set = U64Set::default();
                        let mut batch: Vec<String> = Vec::with_capacity(256);

                        let mut nodes_batch: u64 = 0;
                        let mut leaves_batch: u64 = 0;
                        let mut outputs_batch: u64 = 0;
                        let mut pruned_batch: u64 = 0;
                        let mut lhit_batch: u64 = 0;
                        let mut ghit_batch: u64 = 0;
                        let mut mmiss_batch: u64 = 0;
                        let mut last_preview = Instant::now();

                        let mut time_batch = TimeDelta::default();

                        let placed_first: u32 = (0..4).map(|c| masks[c].count_ones()).sum();
                        let _ = dfs_combine_parallel(
                            1,
                            &mut cols0,
                            gens,
                            order,
                            threshold,
                            exact_four_only,
                            &mut memo,
                            &mut local_output_once,
                            &global_output_once,
                            &global_memo,
                            map_label_to_color,
                            &mut batch,
                            &wtx,
                            &stx,
                            profile_enabled,
                            &mut time_batch,
                            &mut nodes_batch,
                            &mut leaves_batch,
                            &mut outputs_batch,
                            &mut pruned_batch,
                            &mut lhit_batch,
                            &mut ghit_batch,
                            &mut mmiss_batch,
                            preview_ok,
                            &tx,
                            &mut last_preview,
                            lru_limit,
                            t0,
                            &abort,
                            placed_first,
                            &remain_suffix,
                        );

                        if !batch.is_empty() {
                            let _ = wtx.send(batch);
                        }
                        if nodes_batch > 0
                            || leaves_batch > 0
                            || outputs_batch > 0
                            || pruned_batch > 0
                            || lhit_batch > 0
                            || ghit_batch > 0
                            || mmiss_batch > 0
                        {
                            let _ = stx.send(StatDelta {
                                nodes: nodes_batch,
                                leaves: leaves_batch,
                                outputs: outputs_batch,
                                pruned: pruned_batch,
                                lhit: lhit_batch,
                                ghit: ghit_batch,
                                mmiss: mmiss_batch,
                            });
                        }
                        if profile_enabled && time_delta_has_any(&time_batch) {
                            let _ = tx.send(Message::TimeDelta(time_batch));
                        }
                        Ok(())
                    })?;
            } else {
                let second_x = order[1];
                let mut second_candidates: Vec<[u16; 4]> = Vec::new();
                match &gens[second_x] {
                    ColGen::Pre(v) => second_candidates.extend_from_slice(v),
                    ColGen::Stream(colv) => {
                        stream_column_candidates(colv, |m| second_candidates.push(m));
                    }
                }

                second_candidates
                    .par_iter()
                    .try_for_each(|m2| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        for masks in &first_candidates {
                            if abort.load(Ordering::Relaxed) {
                                break;
                            }

                            let mut cols0 = [[0u16; W]; 4];
                            for c in 0..4 {
                                cols0[c][first_x] = masks[c];
                                cols0[c][second_x] = m2[c];
                            }
                            let mut memo =
                                ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
                            let mut local_output_once: U64Set = U64Set::default();
                            let mut batch: Vec<String> = Vec::with_capacity(256);

                            let mut nodes_batch: u64 = 0;
                            let mut leaves_batch: u64 = 0;
                            let mut outputs_batch: u64 = 0;
                            let mut pruned_batch: u64 = 0;
                            let mut lhit_batch: u64 = 0;
                            let mut ghit_batch: u64 = 0;
                            let mut mmiss_batch: u64 = 0;
                            let mut last_preview = Instant::now();

                            let mut time_batch = TimeDelta::default();

                            let placed2: u32 = (0..4)
                                .map(|c| masks[c].count_ones() + m2[c].count_ones())
                                .sum();
                            let _ = dfs_combine_parallel(
                                2,
                                &mut cols0,
                                gens,
                                order,
                                threshold,
                                exact_four_only,
                                &mut memo,
                                &mut local_output_once,
                                &global_output_once,
                                &global_memo,
                                map_label_to_color,
                                &mut batch,
                                &wtx,
                                &stx,
                                profile_enabled,
                                &mut time_batch,
                                &mut nodes_batch,
                                &mut leaves_batch,
                                &mut outputs_batch,
                                &mut pruned_batch,
                                &mut lhit_batch,
                                &mut ghit_batch,
                                &mut mmiss_batch,
                                preview_ok,
                                &tx,
                                &mut last_preview,
                                lru_limit,
                                t0,
                                &abort,
                                placed2,
                                &remain_suffix,
                            );

                            if !batch.is_empty() {
                                let _ = wtx.send(batch);
                            }
                            if nodes_batch > 0
                                || leaves_batch > 0
                                || outputs_batch > 0
                                || pruned_batch > 0
                                || lhit_batch > 0
                                || ghit_batch > 0
                                || mmiss_batch > 0
                            {
                                let _ = stx.send(StatDelta {
                                    nodes: nodes_batch,
                                    leaves: leaves_batch,
                                    outputs: outputs_batch,
                                    pruned: pruned_batch,
                                    lhit: lhit_batch,
                                    ghit: ghit_batch,
                                    mmiss: mmiss_batch,
                                });
                            }
                            if profile_enabled && time_delta_has_any(&time_batch) {
                                let _ = tx.send(Message::TimeDelta(time_batch));
                            }
                        }
                        Ok(())
                    })?;
            }
            Ok(())
        },
    )?;

    drop(wtx);
    drop(stx);
    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow!("writer join error"))?;
    writer_result?;
    agg_handle.join().map_err(|_| anyhow!("agg join error"))?;

    Ok(())
}

// ユーティリティ：配列初期化（const generics）
fn array_init<T, F: FnMut(usize) -> T, const N: usize>(mut f: F) -> [T; N] {
    use std::mem::MaybeUninit;
    let mut data: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for (i, slot) in data.iter_mut().enumerate() {
        slot.write(f(i));
    }
    unsafe { std::mem::transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&data) }
}

// 列テンプレートから、最大で何マス埋められるか（下から Blank までの連続非 Blank 数）
#[inline(always)]
fn compute_max_fill(col: &[TCell]) -> u8 {
    let mut cnt: u8 = 0;
    for &cell in col.iter().take(H) {
        match cell {
            TCell::Blank => break,
            _ => cnt = cnt.saturating_add(1),
        }
    }
    cnt
}

// ===== ここから 小連鎖用 ヘルパ =====

fn build_cols_from_board_colors(board: &[Cell]) -> [[u16; W]; 4] {
    let mut cols = [[0u16; W]; 4];
    for y in 0..H {
        for x in 0..W {
            let idx = y * W + x;
            if let Cell::Color(c) = board[idx] {
                let bit = 1u16 << y;
                cols[c as usize][x] |= bit;
            }
        }
    }
    cols
}

fn column_heights(cols: &[[u16; W]; 4]) -> [usize; W] {
    let mut hts = [0usize; W];
    for x in 0..W {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
        hts[x] = occ.count_ones() as usize;
    }
    hts
}

#[derive(Clone, Copy)]
enum Ori {
    Up,
    Right,
    Down,
    Left,
}

fn enumerate_all_placements_with_walls(heights: &[usize; W], c0: u8, c1: u8) -> Vec<Placement> {
    let mut out = Vec::new();
    for x in 0..W {
        let left_ok = x > 0;
        let right_ok = x + 1 < W;
        // Up
        {
            let y0 = heights[x];
            if y0 + 1 < H {
                out.push(Placement {
                    positions: [(x, y0, c0), (x, y0 + 1, c1)],
                });
            }
        }
        // Down
        {
            let y0 = heights[x];
            if y0 + 1 < H {
                out.push(Placement {
                    positions: [(x, y0, c1), (x, y0 + 1, c0)],
                });
            }
        }
        // 横置き
        if left_ok && right_ok {
            let yx = heights[x];
            // Left
            let yL = heights[x - 1];
            if yx < H && yL < H {
                out.push(Placement {
                    positions: [(x, yx, c0), (x - 1, yL, c1)],
                });
            }
            // Right
            let yR = heights[x + 1];
            if yx < H && yR < H {
                out.push(Placement {
                    positions: [(x, yx, c0), (x + 1, yR, c1)],
                });
            }
        } else if right_ok {
            // x==0
            let yx = heights[x];
            let yR = heights[x + 1];
            if yx < H && yR < H {
                out.push(Placement {
                    positions: [(x, yx, c0), (x + 1, yR, c1)],
                });
            }
        } else if left_ok {
            // x==W-1
            let yx = heights[x];
            let yL = heights[x - 1];
            if yx < H && yL < H {
                out.push(Placement {
                    positions: [(x, yx, c0), (x - 1, yL, c1)],
                });
            }
        }
    }
    out
}

// 旧（下2段限定）の便宜関数は残しておく（互換/検証用）
fn required_colors_bottom2(pre: &[[u16; W]; 4]) -> [[Option<u8>; 2]; W] {
    let mut req: [[Option<u8>; 2]; W] = [[None; 2]; W];
    for x in 0..W {
        for y in 0..2.min(H) {
            let bit = 1u16 << y;
            let mut color: Option<u8> = None;
            for c in 0..4 {
                if (pre[c][x] & bit) != 0 {
                    color = Some(c as u8);
                    break;
                }
            }
            req[x][y] = color;
        }
    }
    req
}

// （★新）寄与判定：pre内の任意座標と一致すれば寄与
#[inline(always)]
fn contribution_on_pre(pl: &Placement, pre: &[[u16; W]; 4]) -> u32 {
    let mut contrib = 0u32;
    for &(x, y, col) in &pl.positions {
        if x < W && y < H {
            let bit = 1u16 << y;
            if (pre[col as usize][x] & bit) != 0 {
                contrib += 1;
            }
        }
    }
    contrib
}

// （★新）1個寄与の場合の“ゴミ”片
#[inline(always)]
fn garbage_piece_on_pre(pl: &Placement, pre: &[[u16; W]; 4]) -> Option<(usize, usize, u8)> {
    let mut match_idx: Option<usize> = None;
    for (i, &(x, y, col)) in pl.positions.iter().enumerate() {
        if x < W && y < H {
            let bit = 1u16 << y;
            if (pre[col as usize][x] & bit) != 0 {
                match_idx = Some(i);
                break;
            }
        }
    }
    match match_idx {
        Some(0) => Some(pl.positions[1]),
        Some(1) => Some(pl.positions[0]),
        _ => None,
    }
}

fn apply_placement(cols: &mut [[u16; W]; 4], pl: &Placement) {
    for &(x, y, col) in &pl.positions {
        if x < W && y < H {
            cols[col as usize][x] |= 1u16 << y;
        }
    }
}

#[inline(always)]
fn min_placement_y(pl: &Placement) -> usize {
    let a = pl.positions[0].1;
    let b = pl.positions[1].1;
    if a < b {
        a
    } else {
        b
    }
}

fn has_immediate_clear(cols: &[[u16; W]; 4], exact_four_only: bool) -> bool {
    if !any_color_has_four(cols) {
        return false;
    }
    if !exact_four_only {
        let clear = compute_erase_mask_cols(cols);
        (0..W).any(|x| clear[x] != 0)
    } else {
        let (clear, had_ge5, had_four) = compute_erase_mask_and_flags(cols);
        had_four && !had_ge5 && (0..W).any(|x| clear[x] != 0)
    }
}

fn chain_len(mut cur: [[u16; W]; 4], exact_four_only: bool) -> u32 {
    let mut k = 0u32;
    if !exact_four_only {
        loop {
            let (erased, next) = apply_erase_and_fall_cols(&cur);
            if !erased {
                break;
            }
            cur = next;
            k += 1;
        }
    } else {
        loop {
            match apply_erase_and_fall_exact4(&cur) {
                StepExact::NoClear => break,
                StepExact::Illegal => {
                    k = 0;
                    break;
                } // 不正は 0 扱い
                StepExact::Cleared(next) => {
                    cur = next;
                    k += 1;
                }
            }
        }
    }
    k
}

fn simulate_chain_full(mut cur: [[u16; W]; 4], exact_four_only: bool) -> (u32, [[u16; W]; 4]) {
    if !exact_four_only {
        let mut k = 0u32;
        loop {
            let (erased, next) = apply_erase_and_fall_cols(&cur);
            if !erased {
                break;
            }
            cur = next;
            k = k.saturating_add(1);
        }
        (k, cur)
    } else {
        let mut k = 0u32;
        loop {
            match apply_erase_and_fall_exact4(&cur) {
                StepExact::NoClear => break,
                StepExact::Illegal => return (0, cur),
                StepExact::Cleared(next) => {
                    cur = next;
                    k = k.saturating_add(1);
                }
            }
        }
        (k, cur)
    }
}

#[inline(always)]
fn column_height_at(cols: &[[u16; W]; 4], x: usize) -> usize {
    let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
    occ.count_ones() as usize
}

#[inline(always)]
fn is_cell_occupied(cols: &[[u16; W]; 4], x: usize, y: usize) -> bool {
    let bit = 1u16 << y;
    ((cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & bit) != 0
}

fn drop_same_color(cols: &mut [[u16; W]; 4], x: usize, color: u8, count: usize) {
    let mut y = column_height_at(cols, x);
    for _ in 0..count {
        if y >= H {
            break;
        }
        cols[color as usize][x] |= 1u16 << y;
        y += 1;
    }
}

struct Scenario {
    chain_len: u32,
    after_board: [[u16; W]; 4],
    drop_x: usize,
    drop_color: u8,
    drop_count: usize,
}

fn find_best_scenario(board: &[[u16; W]; 4], exact_four_only: bool) -> Option<Scenario> {
    let mut best: Option<Scenario> = None;
    for color in 0..4 {
        for x in 0..W {
            let mut mask = board[color][x] & MASK14;
            while mask != 0 {
                let bit = mask & (!mask + 1);
                let y = bit.trailing_zeros() as usize;
                mask &= mask - 1;

                for (dx, dy) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let nx = x as isize + dx;
                    let ny = y as isize + dy;
                    if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize {
                        continue;
                    }
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if is_cell_occupied(board, nx, ny) {
                        continue;
                    }

                    let height = column_height_at(board, nx);
                    if height > ny {
                        continue;
                    }
                    let required_fill = ny + 1 - height;

                    let comp = component_from_seed_cols(&board[color], x, 1u16 << y);
                    let comp_size: usize = comp.iter().map(|&m| m.count_ones() as usize).sum();
                    if comp_size >= 4 {
                        continue;
                    }
                    let needed_for_four = 4usize.saturating_sub(comp_size);
                    let mut count = required_fill;
                    if count < needed_for_four {
                        count = needed_for_four;
                    }
                    if count == 0 {
                        continue;
                    }
                    if height + count > H {
                        continue;
                    }

                    let mut test = *board;
                    drop_same_color(&mut test, nx, color as u8, count);
                    let (len, after) = simulate_chain_full(test, exact_four_only);
                    if len == 0 {
                        continue;
                    }

                    let update = match &best {
                        None => true,
                        Some(b) => {
                            if len > b.chain_len {
                                true
                            } else if len == b.chain_len {
                                if count < b.drop_count {
                                    true
                                } else if count == b.drop_count {
                                    if nx < b.drop_x {
                                        true
                                    } else if nx == b.drop_x {
                                        (color as u8) < b.drop_color
                                    } else {
                                        false
                                    }
                                } else {
                                    false
                                }
                            } else {
                                false
                            }
                        }
                    };

                    if update {
                        best = Some(Scenario {
                            chain_len: len,
                            after_board: after,
                            drop_x: nx,
                            drop_color: color as u8,
                            drop_count: count,
                        });
                    }
                }
            }
        }
    }
    best
}

struct GreedyResult {
    board: [[u16; W]; 4],
    chain_length: u32,
}

fn build_greedy_chain(
    initial: &[[u16; W]; 4],
    exact_four_only: bool,
    threshold: u32,
) -> Option<GreedyResult> {
    let mut simulation = *initial;
    let mut events: Vec<(usize, u8, usize)> = Vec::new();
    let mut total_chain: u32 = 0;

    loop {
        if total_chain >= threshold {
            break;
        }
        let Some(scn) = find_best_scenario(&simulation, exact_four_only) else {
            break;
        };
        if scn.chain_len == 0 {
            break;
        }
        total_chain = total_chain.saturating_add(scn.chain_len);
        events.push((scn.drop_x, scn.drop_color, scn.drop_count));
        simulation = scn.after_board;
    }

    let mut final_board = *initial;
    for (x, color, count) in events.iter() {
        drop_same_color(&mut final_board, *x, *color, *count);
    }
    let (chain_len, _) = simulate_chain_full(final_board, exact_four_only);
    if chain_len == 0 {
        return None;
    }
    Some(GreedyResult {
        board: final_board,
        chain_length: chain_len,
    })
}

fn simulate_e_masks(
    pre: &[[u16; W]; 4],
    t: u32,
    exact_four_only: bool,
) -> (Vec<[u16; W]>, [u16; W]) {
    // e_masks: 各連鎖ステップで『その時点の座標系』で消えるマスク（従来通り）
    // union:   『初期 pre 座標系』で、最終的に消える全セルの合併（修正）
    let mut masks: Vec<[u16; W]> = Vec::new();
    let mut union_pre: [u16; W] = [0; W];

    // 各列を長さ H のベクタ（index = 現在の y）で表す。
    // 要素は Some((color, pre_y)) または None。
    let mut cols_state: [Vec<Option<(u8, u8)>>; W] = array_init(|_| vec![None; H]);
    for x in 0..W {
        for y in 0..H {
            let bit = 1u16 << y;
            for c in 0..4 {
                if (pre[c][x] & bit) != 0 {
                    cols_state[x][y] = Some((c as u8, y as u8));
                    break;
                }
            }
        }
    }

    // 現在の盤面 cur を cols_state から構築
    #[inline(always)]
    fn build_cur_from_state(state: &[Vec<Option<(u8, u8)>>; W]) -> [[u16; W]; 4] {
        let mut cols = [[0u16; W]; 4];
        for x in 0..W {
            for y in 0..H {
                if let Some((col, _pre_y)) = state[x][y] {
                    cols[col as usize][x] |= 1u16 << y;
                }
            }
        }
        cols
    }

    // 連鎖を最大 t 回まで進める
    for _step in 0..t {
        let mut cur = build_cur_from_state(&cols_state);

        // 現在の盤面から消去マスクを取得（座標系は現在）
        let clear: [u16; W];
        if !exact_four_only {
            clear = compute_erase_mask_cols(&cur);
            if (0..W).all(|x| clear[x] == 0) {
                break;
            }
        } else {
            let (clr, had_ge5, had_four) = compute_erase_mask_and_flags(&cur);
            if had_ge5 || !had_four || (0..W).all(|x| clr[x] == 0) {
                break;
            }
            clear = clr;
        }

        // e_masks は従来通り、現座標系での消去マスクを保持
        masks.push(clear);

        // クリアされる駒の『元の pre 座標』を union_pre に積む
        for x in 0..W {
            // 現在の y で消える位置を抽出（列マスク）
            let mut to_remove: [bool; H] = [false; H];
            let mut bits = clear[x];
            while bits != 0 {
                let b = bits & (!bits + 1);
                let y_cur = b.trailing_zeros() as usize;
                if y_cur < H {
                    to_remove[y_cur] = true;
                }
                bits &= bits - 1;
            }

            // union_pre に元座標を反映しつつ、該当要素を除去（None に）
            for y in 0..H {
                if to_remove[y] {
                    if let Some((_col, pre_y)) = cols_state[x][y] {
                        union_pre[x] |= 1u16 << (pre_y as u16);
                        cols_state[x][y] = None;
                    }
                }
            }

            // 一括落下：下から詰め直す
            let mut new_col: Vec<Option<(u8, u8)>> = Vec::with_capacity(H);
            for y in 0..H {
                if let Some(v) = cols_state[x][y] {
                    new_col.push(Some(v));
                }
            }
            while new_col.len() < H {
                new_col.push(None);
            }
            cols_state[x] = new_col;
        }
    }

    (masks, union_pre)
}

// ---- E1（1連鎖目）に関する順序制約ヘルパ ----
// E1 の各列について、
//  - open_top[x]: その列の E1 で最上段セルの直上が空いている（= 上に何も乗っていない）か
//  - top_y[x]:    その列における E1 の最上段 y
#[inline(always)]
fn compute_open_top_info(pre: &[[u16; W]; 4], e1: &[u16; W]) -> ([bool; W], [usize; W]) {
    let mut open = [false; W];
    let mut topy = [0usize; W];
    for x in 0..W {
        let clear_col = e1[x] & MASK14;
        if clear_col == 0 {
            continue;
        }
        let occ_col = (pre[0][x] | pre[1][x] | pre[2][x] | pre[3][x]) & MASK14;
        let ty = 15 - clear_col.leading_zeros() as usize; // 最上段の y（0..H-1）
        topy[x] = ty.min(H - 1);
        if ty + 1 >= H {
            // 盤外は空とみなす
            open[x] = true;
        } else {
            let bit_above = 1u16 << (ty + 1);
            open[x] = (occ_col & bit_above) == 0;
        }
    }
    (open, topy)
}

// 配置 pl が、E1 の open-top 列の「最上段 E1 セル」を含むか判定し、該当列 x を列挙
#[inline(always)]
fn collect_open_top_hits(
    pl: &Placement,
    pre: &[[u16; W]; 4],
    e1: &[u16; W],
    open: &[bool; W],
    top_y: &[usize; W],
) -> Vec<usize> {
    let mut xs: Vec<usize> = Vec::new();
    for &(x, y, col) in &pl.positions {
        if x < W && y < H && open[x] {
            let bit = 1u16 << y;
            if (e1[x] & bit) != 0 && y == top_y[x] && (pre[col as usize][x] & bit) != 0 {
                if !xs.contains(&x) {
                    xs.push(x);
                }
            }
        }
    }
    xs
}

// pl を適用した後で、E1 のうち（除外列以外で）未完成の列が残るか
#[inline(always)]
fn any_other_e1_incomplete_after(
    pl: &Placement,
    pre: &[[u16; W]; 4],
    e1: &[u16; W],
    board_cols: &[[u16; W]; 4],
    exclude_cols: &[usize],
) -> bool {
    let mut after = *board_cols;
    apply_placement(&mut after, pl);

    // 除外列集合（O(W) で十分）
    for x in 0..W {
        let e1col = e1[x] & MASK14;
        if e1col == 0 {
            continue;
        }
        if exclude_cols.iter().any(|&y| y == x) {
            continue;
        }

        // 列 x の E1 必要セルが全て埋まっているか
        let mut complete = true;
        for c in 0..4 {
            let needed = pre[c][x] & e1col;
            if needed != 0 {
                let remaining = needed & !after[c][x];
                if remaining != 0 {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            return true;
        }
    }
    false
}

fn dfs_collect_preboards(
    depth: usize,
    cols0: &mut [[u16; W]; 4],
    gens: &[ColGen; W],
    order: &[usize],
    threshold: u32,
    exact_four_only: bool,
    placed_total: u32,
    remain_suffix: &[u16],
    local_once: &mut U64Set,
    out: &mut Vec<ScResult>,
) {
    // 上界
    let placed = placed_total;
    let remain_cap = *remain_suffix.get(depth).unwrap_or(&0) as u32;
    if placed + remain_cap < 4 * threshold {
        return;
    }

    if depth == W {
        // 葉前の早期除外
        if placed_total < 4 * threshold {
            return;
        }
        if !any_color_has_four(cols0) {
            return;
        }
        let pre = *cols0;
        if !reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only) {
            return;
        }
        let (key64, mirror) = canonical_hash64_fast(&pre);
        if local_once.contains(&key64) {
            return;
        }
        local_once.insert(key64);
        let (e_masks, union) = simulate_e_masks(&pre, threshold, exact_four_only);
        out.push(ScResult {
            pre,
            e_masks,
            union,
            key64,
            mirror,
        });
        return;
    }

    let x = order[depth];
    match &gens[x] {
        ColGen::Pre(v) => {
            for &masks in v {
                assign_col_unrolled(cols0, x, &masks);
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                dfs_collect_preboards(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    placed_total + add,
                    remain_suffix,
                    local_once,
                    out,
                );
                clear_col_unrolled(cols0, x);
            }
        }
        ColGen::Stream(colv) => {
            stream_column_candidates(colv, |masks| {
                assign_col_unrolled(cols0, x, &masks);
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                dfs_collect_preboards(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    placed_total + add,
                    remain_suffix,
                    local_once,
                    out,
                );
                clear_col_unrolled(cols0, x);
            });
        }
    }
}

// 互換：まとめて収集（※今回のストリーミング対応では未使用）
fn run_small_chain_search(
    base_board: Vec<char>,
    threshold: u32,
    exact_four_only: bool,
    abort: Arc<AtomicBool>,
) -> Result<Vec<ScResult>> {
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        return Ok(Vec::new());
    }

    let mut metas: Vec<(HashMap<char, u8>, [ColGen; W], [u8; W], Vec<usize>)> = Vec::new();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut cols: [Vec<TCell>; W] = array_init(|_| Vec::with_capacity(H));
        for x in 0..W {
            for y in 0..H {
                cols[x].push(templ[y * W + x]);
            }
        }
        let mut impossible = false;
        let mut counts: [BigUint; W] = array_init(|_| BigUint::zero());
        let mut max_fill_arr: [u8; W] = [0; W];
        for x in 0..W {
            let cnt = count_column_candidates_dp(&cols[x]);
            if cnt.is_zero() {
                impossible = true;
                break;
            }
            counts[x] = cnt.clone();
            max_fill_arr[x] = compute_max_fill(&cols[x]);
        }
        if !impossible {
            let mut order: Vec<usize> = (0..W).collect();
            order.sort_by(|&a, &b| counts[a].cmp(&counts[b]));
            let gens: [ColGen; W] = array_init(|x| build_colgen(&cols[x], &counts[x]));
            metas.push((map, gens, max_fill_arr, order));
        }
    }

    let mut results: Vec<ScResult> = Vec::new();
    for (_map, gens, max_fill, order) in metas.into_iter() {
        if abort.load(Ordering::Relaxed) {
            break;
        }
        let first_x = order[0];
        let mut first_candidates: Vec<[u16; 4]> = Vec::new();
        match &gens[first_x] {
            ColGen::Pre(v) => first_candidates.extend_from_slice(v),
            ColGen::Stream(colv) => {
                stream_column_candidates(colv, |m| first_candidates.push(m));
            }
        }
        let mut remain_suffix: Vec<u16> = vec![0; W + 1];
        for d in (0..W).rev() {
            remain_suffix[d] = remain_suffix[d + 1] + max_fill[order[d]] as u16;
        }

        for masks in &first_candidates {
            if abort.load(Ordering::Relaxed) {
                break;
            }
            let mut cols0 = [[0u16; W]; 4];
            for c in 0..4 {
                cols0[c][first_x] = masks[c];
            }
            let placed_first: u32 = (0..4).map(|c| masks[c].count_ones()).sum();
            let mut local_once: U64Set = U64Set::default();
            dfs_collect_preboards(
                1,
                &mut cols0,
                &gens,
                &order,
                threshold,
                exact_four_only,
                placed_first,
                &remain_suffix,
                &mut local_once,
                &mut results,
            );
        }
    }
    Ok(results)
}
// ===== ストリーミング版：見つけ次第UIへ流す DFS =====
fn stream_collect_preboards_streaming(
    depth: usize,
    cols0: &mut [[u16; W]; 4],
    gens: &[ColGen; W],
    order: &[usize],
    threshold: u32,
    exact_four_only: bool,
    placed_total: u32,
    remain_suffix: &[u16],
    // 走査全体での一意集合（キー64）
    global_once: &mut U64Set,
    // 進捗の蓄積（最後に SmallChainFinished でまとめて返す用）
    out_accum: &mut Vec<ScResult>,
    // 中間結果を逐次UIへ届ける
    tx: &Sender<Message>,
    // 中断トリガ（「小連鎖探索開始」を再押下で true になる想定）
    abort: &AtomicBool,
    // 現在手の色と、列ごとの『その列に存在する N/X/ラベル(A..M) の中で下2つ』の位置マスク
    pair_colors: (u8, u8),
    bottom2_n_mask: &[u16; W],
    // 進捗（葉の処理数）を集計スレッドへ送る
    progress_tx: &Sender<u64>,
    progress_batch: &mut u64,
) {
    if abort.load(Ordering::Relaxed) {
        return;
    }

    // 上界枝刈り：残り列の最大充填で 4T 未満なら打ち切り
    let remain_cap = *remain_suffix.get(depth).unwrap_or(&0) as u32;
    if placed_total + remain_cap < 4 * threshold {
        return;
    }

    if depth == W {
        // 葉に到達（全組合せに対する進捗を1増加）
        *progress_batch = progress_batch.saturating_add(1);
        if *progress_batch >= 4096 {
            let n = std::mem::take(progress_batch);
            let _ = progress_tx.send(n);
        }
        // 葉直前の軽量チェック
        if placed_total < 4 * threshold {
            return;
        }
        if !any_color_has_four(cols0) {
            return;
        }

        let pre = *cols0;

        // まず T 連鎖到達性を確認（高速判定）
        if !reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only) {
            return;
        }

        // 下2段（N/X/ラベルの底から2つ）に現在手の色が1個以上含まれることを要求（全体で1個以上）
        {
            let (c0, c1) = (pair_colors.0.min(3) as usize, pair_colors.1.min(3) as usize);
            let mut any_in_bottom2 = false;
            for x in 0..W {
                let m = bottom2_n_mask[x] & MASK14;
                let used = (pre[c0][x] | pre[c1][x]) & m;
                if used != 0 {
                    any_in_bottom2 = true;
                    break;
                }
            }
            if !any_in_bottom2 {
                return;
            }
        }

        // 消去セルの合併（union）を算出し、そこに現在手の色が1個以上含まれることを要求
        let (e_masks, union) = simulate_e_masks(&pre, threshold, exact_four_only);
        {
            let (c0, c1) = (pair_colors.0.min(3) as usize, pair_colors.1.min(3) as usize);
            let mut any_used = false;
            for x in 0..W {
                let m = union[x] & MASK14;
                let used = (pre[c0][x] | pre[c1][x]) & m;
                if used != 0 {
                    any_used = true;
                    break;
                }
            }
            if !any_used {
                return;
            }
        }

        // 一意化キー（条件を満たしたもののみ登録）
        let (key64, mirror) = canonical_hash64_fast(&pre);
        if global_once.contains(&key64) {
            return;
        }
        global_once.insert(key64);

        // 付帯情報（e_masks/union）は上で算出済み
        let sc = ScResult {
            pre,
            e_masks,
            union,
            key64,
            mirror,
        };

        // 逐次: 見つけ次第 UI へ
        let _ = tx.send(Message::SmallChainFound(sc.clone()));
        // 保持（終了時にまとめて Finished で返す）
        out_accum.push(sc);
        return;
    }

    let x = order[depth];
    match &gens[x] {
        ColGen::Pre(v) => {
            for &masks in v {
                if abort.load(Ordering::Relaxed) {
                    return;
                }
                assign_col_unrolled(cols0, x, &masks);
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                stream_collect_preboards_streaming(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    placed_total + add,
                    remain_suffix,
                    global_once,
                    out_accum,
                    tx,
                    abort,
                    pair_colors,
                    bottom2_n_mask,
                    progress_tx,
                    progress_batch,
                );
                clear_col_unrolled(cols0, x);
                if abort.load(Ordering::Relaxed) {
                    return;
                }
            }
        }
        ColGen::Stream(colv) => {
            stream_column_candidates(colv, |masks| {
                if abort.load(Ordering::Relaxed) {
                    return;
                }
                assign_col_unrolled(cols0, x, &masks);
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                stream_collect_preboards_streaming(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    placed_total + add,
                    remain_suffix,
                    global_once,
                    out_accum,
                    tx,
                    abort,
                    pair_colors,
                    bottom2_n_mask,
                    progress_tx,
                    progress_batch,
                );
                clear_col_unrolled(cols0, x);
            });
        }
    }
}

// ===== 小連鎖探索（ストリーミング）=====
// 1件目が見つかったら即座に Message::SmallChainFound を tx に流し、
// 全探索完了 or 中断時に Message::SmallChainFinished(これまでのVec) を送る。
fn run_small_chain_search_streaming(
    base_board: Vec<char>,
    threshold: u32,
    exact_four_only: bool,
    pair_colors: (u8, u8),
    abort: Arc<AtomicBool>,
    tx: Sender<Message>,
) -> Result<()> {
    // 抽象→4色彩色
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        // 空なら空配列で Finished
        let _ = tx.send(Message::Log(
            "[小連鎖] 彩色候補なし（条件に合致する形なし）".into(),
        ));
        let _ = tx.send(Message::SmallChainFinished(Vec::new()));
        return Ok(());
    }

    let _ = tx.send(Message::Log(format!(
        "[小連鎖] ストリーミング探索開始: ラベル={} / 彩色候補={} / T={} / 4個消し={}",
        info.labels.iter().collect::<String>(),
        colorings.len(),
        threshold,
        if exact_four_only { "ON" } else { "OFF" },
    )));

    // 事前に彩色済み盤面と必須セルを構築
    let mut instances: Vec<([[u16; W]; 4], [u16; W])> = Vec::new();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut board_cols = [[0u16; W]; 4];
        let mut required_nonblank = [0u16; W];
        for x in 0..W {
            for y in 0..H {
                let idx = y * W + x;
                match templ[idx] {
                    TCell::Fixed(c) => {
                        board_cols[c as usize][x] |= 1u16 << y;
                    }
                    TCell::Any4 => {
                        required_nonblank[x] |= 1u16 << y;
                    }
                    _ => {}
                }
            }
        }
        instances.push((board_cols, required_nonblank));
    }

    let total: BigUint = BigUint::from(instances.len() as u64);

    // 進捗集約スレッド（小連鎖用）：葉ごとの進捗を受け取り、UIへ Progress を送る
    use crossbeam_channel::unbounded as unbounded_progress;
    let (ptx, prx) = unbounded_progress::<u64>();
    let tx_progress = tx.clone();
    let total_clone = total.clone();
    let t0 = Instant::now();
    thread::spawn(move || {
        let mut done = BigUint::zero();
        let mut leaves_count: u64 = 0;
        let mut last_send = Instant::now();
        loop {
            match prx.recv_timeout(Duration::from_millis(500)) {
                Ok(n) => {
                    leaves_count = leaves_count.saturating_add(n);
                    done += BigUint::from(n);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    // 最終更新（searching=false）
                    let dt = t0.elapsed().as_secs_f64();
                    let rate = if dt > 0.0 {
                        leaves_count as f64 / dt
                    } else {
                        0.0
                    };
                    let st = Stats {
                        searching: false,
                        unique: 0,
                        output: 0,
                        nodes: leaves_count,
                        pruned: 0,
                        memo_hit_local: 0,
                        memo_hit_global: 0,
                        memo_miss: 0,
                        total: total_clone.clone(),
                        done: done.clone(),
                        rate,
                        memo_len: 0,
                        lru_limit: 0,
                        profile: ProfileTotals::default(),
                    };
                    let _ = tx_progress.send(Message::Progress(st));
                    break;
                }
            }

            if last_send.elapsed() >= Duration::from_millis(500) {
                let dt = t0.elapsed().as_secs_f64();
                let rate = if dt > 0.0 {
                    leaves_count as f64 / dt
                } else {
                    0.0
                };
                let st = Stats {
                    searching: true,
                    unique: 0,
                    output: 0,
                    nodes: leaves_count,
                    pruned: 0,
                    memo_hit_local: 0,
                    memo_hit_global: 0,
                    memo_miss: 0,
                    total: total_clone.clone(),
                    done: done.clone(),
                    rate,
                    memo_len: 0,
                    lru_limit: 0,
                    profile: ProfileTotals::default(),
                };
                let _ = tx_progress.send(Message::Progress(st));
                last_send = Instant::now();
            }
        }
    });

    // 逐次出力の蓄積（最後に SmallChainFinished で返す）
    let mut all_results: Vec<ScResult> = Vec::new();
    // 走査全体での一意キー集合（ミラー統一済み key64）
    let mut global_once: U64Set = U64Set::default();

    // 列ごとの『その列に存在する N/X/ラベル(A..M) の中で底から2つ』の位置マスク（y=0が最下段）
    let mut bottom2_n_mask: [u16; W] = [0; W];
    for x in 0..W {
        let mut cnt = 0u8;
        for y in 0..H {
            let ch = base_board[y * W + x];
            if ch == 'N' || ch == 'X' || ('A'..='M').contains(&ch) {
                if cnt < 2 {
                    bottom2_n_mask[x] |= 1u16 << y;
                    cnt += 1;
                }
            }
        }
    }

    let mut prog_batch_local: u64 = 0;

    for (board_cols, required_nonblank) in instances.into_iter() {
        if abort.load(Ordering::Relaxed) {
            break;
        }

        if let Some(result) = build_greedy_chain(&board_cols, exact_four_only, threshold) {
            let GreedyResult {
                board: pre,
                chain_length,
            } = result;

            if chain_length >= threshold {
                let mut ok_required = true;
                for x in 0..W {
                    let occ = (pre[0][x] | pre[1][x] | pre[2][x] | pre[3][x]) & MASK14;
                    if (required_nonblank[x] & !occ) != 0 {
                        ok_required = false;
                        break;
                    }
                }

                if ok_required {
                    let mut placed_total: u32 = 0;
                    for c in 0..4 {
                        for &m in &pre[c] {
                            placed_total = placed_total.saturating_add(m.count_ones() as u32);
                        }
                    }

                    if placed_total >= 4 * threshold
                        && any_color_has_four(&pre)
                        && reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only)
                    {
                        let (pc0, pc1) =
                            (pair_colors.0.min(3) as usize, pair_colors.1.min(3) as usize);
                        let mut any_in_bottom2 = false;
                        for x in 0..W {
                            let m = bottom2_n_mask[x] & MASK14;
                            if m == 0 {
                                continue;
                            }
                            let used = (pre[pc0][x] | pre[pc1][x]) & m;
                            if used != 0 {
                                any_in_bottom2 = true;
                                break;
                            }
                        }

                        if any_in_bottom2 {
                            let (e_masks, union) =
                                simulate_e_masks(&pre, threshold, exact_four_only);
                            let mut any_used = false;
                            for x in 0..W {
                                let m = union[x] & MASK14;
                                let used = (pre[pc0][x] | pre[pc1][x]) & m;
                                if used != 0 {
                                    any_used = true;
                                    break;
                                }
                            }

                            if any_used {
                                let (key64, mirror) = canonical_hash64_fast(&pre);
                                if !global_once.contains(&key64) {
                                    global_once.insert(key64);
                                    let sc = ScResult {
                                        pre,
                                        e_masks,
                                        union,
                                        key64,
                                        mirror,
                                    };
                                    let _ = tx.send(Message::SmallChainFound(sc.clone()));
                                    all_results.push(sc);
                                }
                            }
                        }
                    }
                }
            }
        }

        prog_batch_local = prog_batch_local.saturating_add(1);
        if prog_batch_local >= 64 {
            let _ = ptx.send(prog_batch_local);
            prog_batch_local = 0;
        }
        if abort.load(Ordering::Relaxed) {
            break;
        }
    }

    // 残バッチをフラッシュ
    if prog_batch_local > 0 {
        let _ = ptx.send(prog_batch_local);
    }

    // 終了通知（中断でも、ここまでの結果を返す）
    let _ = tx.send(Message::SmallChainFinished(all_results));
    Ok(())
}
