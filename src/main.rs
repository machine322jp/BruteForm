

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufWriter, BufReader, BufRead, Write};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use eframe::egui;
use egui::{Color32, RichText, Vec2};
use num_bigint::BigUint;
use num_traits::{One, Zero, ToPrimitive};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use serde_json;
use dashmap::{DashMap, DashSet};
use rand::Rng;

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
    Blank,   // '.'
    Any,     // 'N' (空白 or 色)
    Any4,    // 'X' (色のみ)
    Abs(u8), // 0..12 = 'A'..'M'
    Fixed(u8), // 0..=3 = '0'..'3' (RGBY固定)
}

// 方式B: 直前ステップ（最後の追加の直前）で既に free-top 列がE1で消える状態だったなら true
#[inline(always)]
fn precompleted_free_top_e1_on_last_step(
    original: &[[u16; W]; 4],
    additions: &Vec<(usize, u8)>,
) -> bool {
    let n = additions.len();
    if n == 0 { return false; }
    // 最終盤面
    let merged_final = merge_additions_onto(*original, additions);
    let clear_final = compute_erase_mask_cols(&merged_final);
    let ft_final = first_clear_free_top_cols(&merged_final);

    // 直前盤面（最後の追加を除く）
    let mut pre_adds: Vec<(usize, u8)> = additions.clone();
    pre_adds.pop();
    let merged_pre = merge_additions_onto(*original, &pre_adds);
    let clear_pre = compute_erase_mask_cols(&merged_pre);
    let ft_pre = first_clear_free_top_cols(&merged_pre);

    // final で free-top かつ E1 の列が、pre でも free-top かつ E1 なら「直前に既に完成していた」とみなす
    for x in 0..W {
        if ft_final[x] && (clear_final[x] & MASK14) != 0 {
            if ft_pre[x] && (clear_pre[x] & MASK14) != 0 {
                return true;
            }
        }
    }
    false
}

#[inline(always)]
fn tie_break_cmp(a: (usize, Orient), b: (usize, Orient), board: &[[u16; W]; 4], policy: &TieBreakPolicy) -> std::cmp::Ordering {
    if !policy.apply_initial { return std::cmp::Ordering::Equal; }
    let ab = tie_break_choose(a, b, board, policy);
    let ba = tie_break_choose(b, a, board, policy);
    if ab && !ba { std::cmp::Ordering::Less }
    else if ba && !ab { std::cmp::Ordering::Greater }
    else { std::cmp::Ordering::Equal }
}

// 共通ヘルパー: 候補集合からタイブレーク規則に従って最良手を1つ選ぶ
#[inline(always)]
fn select_best_move(cands: &[(usize, Orient)], board: &[[u16; W]; 4], policy: &TieBreakPolicy) -> Option<(usize, Orient)> {
    if cands.is_empty() { return None; }
    let mut best = cands[0];
    for &mv in &cands[1..] {
        if tie_break_cmp(mv, best, board, policy) == std::cmp::Ordering::Less {
            best = mv;
        }
    }
    Some(best)
}

// 共通ヘルパー: スコア最大を優先し、同点ならタイブレーク
#[inline(always)]
fn select_best_scored_i32(cands: &[((usize, Orient), i32)], board: &[[u16; W]; 4], policy: &TieBreakPolicy) -> Option<(usize, Orient)> {
    if cands.is_empty() { return None; }
    let mut best = cands[0];
    for &item in &cands[1..] {
        if item.1 > best.1 {
            best = item;
        } else if item.1 == best.1 {
            if tie_break_cmp(item.0, best.0, board, policy) == std::cmp::Ordering::Less {
                best = item;
            }
        }
    }
    Some(best.0)
}

// 追加群のうち、「free-top 列」に置かれ、かつ E1 消去に参加しているセルが1つでもあれば true
#[inline(always)]
fn any_addition_on_free_top_involved_in_e1(
    original: &[[u16; W]; 4],
    additions: &Vec<(usize, u8)>,
) -> bool {
    if additions.is_empty() { return false; }
    let merged = merge_additions_onto(*original, additions);
    if count_first_clear_groups(&merged) == 0 || !first_clear_has_free_top(&merged) {
        return false;
    }
    let ft = first_clear_free_top_cols(&merged);
    let clear = compute_erase_mask_cols(&merged);
    // 列ごとの追加の個数を数えながら y を算出
    let mut per_col_counts = [0usize; W];
    for &(cx, _color) in additions {
        if cx >= W { continue; }
        per_col_counts[cx] += 1;
        let y = col_height(original, cx) + per_col_counts[cx] - 1;
        if ft[cx] && (clear[cx] & (1u16 << y)) != 0 {
            return true;
        }
    }
    false
}

// 1連鎖目で消える塊に対し、「その列の上に何も乗っていない」列を列挙
#[inline(always)]
fn first_clear_free_top_cols(cols: &[[u16; W]; 4]) -> [bool; W] {
    let mut res = [false; W];
    let clear = compute_erase_mask_cols(cols);
    for x in 0..W {
        let m = clear[x] & MASK14;
        if m == 0 { continue; }
        // この列の塊の最上段 y を求める
        let mut top_y_opt = None;
        for y in (0..H).rev() {
            if (m & (1u16 << y)) != 0 { top_y_opt = Some(y); break; }
        }
        if let Some(top_y) = top_y_opt {
            let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
            let above_mask: u16 = (!((1u16 << (top_y + 1)) - 1)) & MASK14;
            if (occ & above_mask) == 0 { res[x] = true; }
        }
    }
    res
}

#[inline(always)]
fn apply_clear_no_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];
    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (out_col, pre_col) in out.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(out_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    out
}

fn draw_pair_preview(ui: &mut egui::Ui, pair: (u8, u8)) {
    let sz = Vec2::new(18.0, 18.0);
    ui.vertical(|ui| {
        // 表示規約: デフォルトは軸ぷよの上に子ぷよ（axis below, child above 表示）
        let (txt1, fill1, stroke1) = cell_style(Cell::Fixed(pair.1)); // child (上)
        let (txt0, fill0, stroke0) = cell_style(Cell::Fixed(pair.0)); // axis (下)
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

// ---- 計測（DFS 深さ×フェーズ） -------------------------

#[derive(Default, Clone, Copy)]
struct DfsDepthTimes {
    gen_candidates: Duration,
    assign_cols: Duration,
    upper_bound: Duration,

    // 葉専用
    leaf_fall_pre: Duration,
    leaf_hash: Duration,
    leaf_memo_get: Duration,           // 今回の最適化後はほぼ0のまま
    leaf_memo_miss_compute: Duration,  // 到達判定（reaches_t...）
    out_serialize: Duration,
}
#[derive(Default, Clone, Copy)]
struct DfsDepthCounts {
    nodes: u64,
    cand_generated: u64,
    pruned_upper: u64,
    leaves: u64,
    // 葉早期リターン（落下や到達判定より前）
    leaf_pre_tshort: u64,          // 4T 未満で不可能
    leaf_pre_e1_impossible: u64,   // E1 不可能（4連結なし）
    memo_lhit: u64, // 以降は基本0（残しつつ非使用）
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
    // 1連鎖目に free-top 列が1つ以上ある形に限定する
    require_free_top_e1: bool,

    // 実行状態
    running: bool,
    abort_flag: Arc<AtomicBool>,
    rx: Option<Receiver<Message>>,
    stats: Stats,
    preview: Option<[[u16; W]; 4]>,
    log_lines: Vec<String>,

    // 画面モード
    mode: Mode,
    // 連鎖生成モードの状態
    cp: ChainPlay,
    // 連鎖生成（目標配置）: 非同期探索用
    cp_rx: Option<Receiver<CpMessage>>, // 連鎖生成モード専用のメッセージ受信
    cp_abort_flag: Arc<AtomicBool>,     // 連鎖生成モード専用の中断フラグ
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

enum Message {
    Log(String),
    Preview([[u16; W]; 4]),
    Progress(Stats),
    Finished(Stats),
    Error(String),
    // 追加：時間の増分メッセージ
    TimeDelta(TimeDelta),
}

// 連鎖生成（目標配置）専用メッセージ
#[derive(Clone, Copy, PartialEq, Eq)]
enum FoundKind { TargetExact, Make3Group, MakeDiagonalSameColor, Make2Group, OneChainMaxGroup, FullClearFallback, JsonlPattern, JsonlAvoid }
enum CpMessage {
    Log(String),
    Depth(usize),
    Found { x: usize, orient: Orient, pair: (u8, u8), kind: FoundKind, fc_preview: Option<[[u16; W]; 4]>, fc_chain: Option<u32> },
    // JSONL 形が完成したことの通知（以降、保全モジュールは停止するためのフラグに使う）
    JsonlCompleted { key: String },
    // 目指す JSONL 形の表示更新
    TargetShape { rows: Vec<String>, key: String },
    FinishedNone,
    Error(String),
}

// 連鎖生成：配置制約（モジュール化）
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum CpConstraintKind {
    // JSONL の形を目指す（最優先）。貢献手がなければ次の制約へ。
    JsonlFollow,
    // 0寄与かつ再検索でもヒットなしのとき、合致率が一定以上の形を保全するため、干渉（他色での上書き）を避ける手を採用
    // しきい値はパーセント（0..=100）
    JsonlAvoidInterfere { min_ratio_pct: u8 },
    // 深さ 1..=max_depth の IDDFS で「＝ target」達成なら採用
    TargetExactWithinDepth { max_depth: usize },
    // 現在手で3連結を作れる手があれば採用（即時消去は避ける）
    Make3GroupPrefer { avoid_diag_down23: bool },
    // 現在手で、同色の斜め隣接を作れる手があれば採用（即時消去は避ける）
    MakeDiagonalSameColorPrefer,
    // 現在手で2連結を作れる手があれば採用（即時消去は避ける）
    Make2GroupPrefer { avoid_diag_down23: bool },
    // 1連鎖で発火し、かつ1連鎖目の連結サイズが最大となる手を優先（即時消去は必要条件）
    OneChainMaxGroupPrefer { avoid_diag_down23: bool },
    // 全消し検出（ビーム）で最大連鎖になる初手を採用
    FullClearFallback { beam_depth: usize, beam_width: usize, pair_lookahead: usize },
}

// ── 同率タイ時の優先ルール ───────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TieBreakRule {
    // 設置列の高さが低い方を優先（横置きは関与列の最小高さ）
    MinPlacementHeight,
    // 左端（xが小さい）を優先
    LeftmostColumn,
    // 垂直配置（Up/Down）を優先
    PreferVertical,
    // 同色の斜め隣接を優先（現在ペア色の情報がない場合は無効）
    PreferDiagonalSameColor,
    // 3連結を作る手を優先（直前盤面での斜め下2/3同色連結がある既存ぷよへの接続は避ける）
    PreferMake3GroupAvoidDiagDown23,
    // 2連結を作る手を優先（直前盤面での斜め下2/3同色連結がある既存ぷよへの接続は避ける）
    PreferMake2GroupAvoidDiagDown23,
}

#[derive(Clone, Serialize, Deserialize)]
struct TieBreakPolicy {
    // どの局面で適用するか（まずは初手選択のみ）
    apply_initial: bool,
    // 優先ルールの並び（上から順に適用）
    rules: Vec<TieBreakRule>,
}

impl Default for TieBreakPolicy {
    fn default() -> Self {
        Self {
            apply_initial: true,
            rules: vec![
                TieBreakRule::PreferDiagonalSameColor,
                TieBreakRule::PreferMake3GroupAvoidDiagDown23,
                TieBreakRule::PreferMake2GroupAvoidDiagDown23,
                TieBreakRule::MinPlacementHeight,
                TieBreakRule::LeftmostColumn,
            ],
        }
    }
}

// 連鎖生成モジュール設定（保存用）
#[derive(Serialize, Deserialize, Clone)]
struct CpSettings {
    constraints: Vec<CpConstraintKind>,
    tie_break: TieBreakPolicy,
}

#[inline(always)]
fn tie_rule_label(rule: TieBreakRule) -> &'static str {
    match rule {
        TieBreakRule::MinPlacementHeight => "最小設置列高を優先",
        TieBreakRule::LeftmostColumn => "左端（xが小さい）を優先",
        TieBreakRule::PreferVertical => "垂直配置を優先",
        TieBreakRule::PreferDiagonalSameColor => "同色の斜め隣接を優先",
        TieBreakRule::PreferMake3GroupAvoidDiagDown23 => "3連結を優先（斜め下2/3回避）",
        TieBreakRule::PreferMake2GroupAvoidDiagDown23 => "2連結を優先（斜め下2/3回避）",
    }
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
        Self {
            board,
            threshold: 7,
            lru_k: 300,
            out_path: None,
            out_name: "results.jsonl".to_string(),
            stop_progress_plateau: 0.0, // 無効（0.10などにすると有効）
            exact_four_only: false,
            profile_enabled: false,
            require_free_top_e1: false,
            running: false,
            abort_flag: Arc::new(AtomicBool::new(false)),
            rx: None,
            stats: Stats::default(),
            preview: None,
            log_lines: vec!["待機中".into()],
            mode: Mode::BruteForce,
            cp: ChainPlay::default(),
            cp_rx: None,
            cp_abort_flag: Arc::new(AtomicBool::new(false)),
        }
    }
}

// 画面モード
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    BruteForce,
    ChainPlay,
}

// 連鎖生成：配置向き
#[derive(Clone, Copy, PartialEq, Eq)]
enum Orient { Up, Right, Down, Left }

// 連鎖生成モード：保存状態
#[derive(Clone, Copy)]
struct SavedState {
    cols: [[u16; W]; 4],
    pair_index: usize,
}

// アニメーション段階
#[derive(Clone, Copy, PartialEq, Eq)]
enum AnimPhase {
    AfterErase,
    AfterFall,
}

#[derive(Clone, Copy)]
struct AnimState {
    phase: AnimPhase,
    since: Instant,
}

// 連鎖生成モードのワーク
struct ChainPlay {
    cols: [[u16; W]; 4],
    pair_seq: Vec<(u8, u8)>, // 軸, 子（0..=3）
    pair_index: usize,
    undo_stack: Vec<SavedState>,
    anim: Option<AnimState>,
    lock: bool, // 連鎖アニメ中ロック
    // アニメ表示用：消去直後の盤面と、落下後の次盤面
    erased_cols: Option<[[u16; W]; 4]>,
    next_cols: Option<[[u16; W]; 4]>,
    // 全消しプレビュー
    best_preview: Option<[[u16; W]; 4]>,
    best_chain: u32,
    // 目標配置（探索）用
    target_chain: u32,     // 目標連鎖数（＝で判定）
    searching: bool,       // 探索中フラグ
    search_depth: usize,   // 現在の探索深さ（UI表示）
    last_found_kind: Option<FoundKind>, // 直前の採用理由
    // 配置制約（適用順リスト）
    constraints: Vec<CpConstraintKind>,
    // 同率タイ時の優先ルール
    tie_break: TieBreakPolicy,
    // JSONL 追従: 現在の目標形（表示用）
    target_shape_rows: Option<Vec<String>>, // 数字行列（1個除外済み）
    target_shape_key: Option<String>,        // 元の key（抽象ラベル）
    // JSONL ファイルパス（未設定なら out_path/out_name を使用）
    jsonl_path: Option<std::path::PathBuf>,
    // JSONL 完成検知（永続フラグ）
    jsonl_completed: bool,
    jsonl_completed_key: Option<String>,
}

impl Default for ChainPlay {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let mut seq = Vec::with_capacity(128);
        for _ in 0..128 {
            seq.push((rng.gen_range(0..4), rng.gen_range(0..4)));
        }
        let cols = [[0u16; W]; 4];
        let s0 = SavedState { cols, pair_index: 0 };
        Self {
            cols,
            pair_seq: seq,
            pair_index: 0,
            undo_stack: vec![s0],
            anim: None,
            lock: false,
            erased_cols: None,
            next_cols: None,
            best_preview: None,
            best_chain: 0,
            target_chain: 4,
            searching: false,
            search_depth: 0,
            last_found_kind: None,
            constraints: vec![
                CpConstraintKind::JsonlFollow,
                CpConstraintKind::JsonlAvoidInterfere { min_ratio_pct: 50 },
                CpConstraintKind::TargetExactWithinDepth { max_depth: 5 },
                CpConstraintKind::Make3GroupPrefer { avoid_diag_down23: true },
                CpConstraintKind::MakeDiagonalSameColorPrefer,
                CpConstraintKind::Make2GroupPrefer { avoid_diag_down23: true },
                CpConstraintKind::OneChainMaxGroupPrefer { avoid_diag_down23: true },
                CpConstraintKind::FullClearFallback { beam_depth: 8, beam_width: 16, pair_lookahead: 5 },
            ],
            tie_break: TieBreakPolicy::default(),
            target_shape_rows: None,
            target_shape_key: None,
            jsonl_path: None,
            jsonl_completed: false,
            jsonl_completed_key: None,
        }
    }
}

// ...

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
            fonts.font_data.insert(key.clone(), FontData::from_owned(bytes));
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
        eprintln!("日本語フォントを見つけられませんでした。C:\\Windows\\Fonts を確認してください。");
    }
}

// ====== eframe エントリ ======
fn main() -> Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size(egui::vec2(980.0, 760.0)),
        ..Default::default()
    };

    eframe::run_native(
        "ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI",
        options,
        Box::new(|cc| {
            install_japanese_fonts(&cc.egui_ctx);
            Box::new(App::new())
        }),
    )
    .map_err(|e| anyhow!("GUI起動に失敗: {e}"))
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 連鎖アニメーション進行
        if self.mode == Mode::ChainPlay {
            self.cp_step_animation();
        }
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
                }
            }

            if keep_rx {
                self.rx = Some(rx);
            }
        }

        // 連鎖生成（目標配置）: 受信メッセージ処理
        if let Some(rx) = self.cp_rx.take() {
            let mut keep_rx = true;
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    CpMessage::Log(s) => self.push_log(s),
                    CpMessage::Depth(d) => {
                        self.cp.search_depth = d;
                    }
                    CpMessage::JsonlCompleted { key } => {
                        self.cp.jsonl_completed = true;
                        self.cp.jsonl_completed_key = Some(key.clone());
                        self.cp.target_shape_rows = None;
                        self.cp.target_shape_key = Some(key.clone());
                        self.push_log(format!("目標配置: JSONL 形が完成（以降の保全は停止）: {}", key));
                    }
                    CpMessage::TargetShape { rows, key } => {
                        self.cp.target_shape_rows = Some(rows);
                        self.cp.target_shape_key = Some(key);
                    }
                    CpMessage::Found { x, orient, pair, kind, fc_preview, fc_chain } => {
                        // 成功: 現在状態に適用（スナップショット→配置→連鎖チェック）
                        self.cp.undo_stack.push(SavedState { cols: self.cp.cols, pair_index: self.cp.pair_index });
                        self.cp_place_with(x, orient, pair);
                        self.cp_check_and_start_chain();
                        self.cp.last_found_kind = Some(kind);
                        // フォールバック時は全消しプレビューも更新
                        if let FoundKind::FullClearFallback = kind {
                            if let Some(pv) = fc_preview { self.cp.best_preview = Some(pv); }
                            if let Some(tc) = fc_chain { self.cp.best_chain = tc; }
                        }
                        // ログはリセット前の深さを利用
                        let depth_shown = self.cp.search_depth;
                        match kind {
                            FoundKind::TargetExact => self.push_log(format!("目標配置: 深さ{}で目標＝{}連鎖を達成", depth_shown, self.cp.target_chain)),
                            FoundKind::Make3Group => self.push_log("目標配置: 3連結を作成できる手を採用".into()),
                            FoundKind::MakeDiagonalSameColor => self.push_log("目標配置: 同色の斜め隣接を作れる手を採用".into()),
                            FoundKind::Make2Group => self.push_log("目標配置: 2連結を作成できる手を採用".into()),
                            FoundKind::OneChainMaxGroup => self.push_log("目標配置: 1連鎖（最大連結）で発火".into()),
                            FoundKind::FullClearFallback => self.push_log("目標配置: 目標未達 → 全消し検出の最大連鎖で配置（fallback）".into()),
                            FoundKind::JsonlPattern => self.push_log("目標配置: JSONL 形に寄与する手を採用".into()),
                            FoundKind::JsonlAvoid => self.push_log("目標配置: JSONL 形を邪魔しない手を採用".into()),
                        }
                        // JSONL 以外（ただし保全手 JsonlAvoid と、形保持の OneChainMaxGroup は除く）を採用した場合は目標形表示をクリア
                        if kind != FoundKind::JsonlPattern && kind != FoundKind::JsonlAvoid && kind != FoundKind::OneChainMaxGroup {
                            self.cp.target_shape_rows = None;
                            self.cp.target_shape_key = None;
                        }
                        self.cp.searching = false;
                        self.cp.search_depth = 0;
                        self.cp_abort_flag.store(false, Ordering::Relaxed);
                        keep_rx = false; // 終了
                    }
                    CpMessage::FinishedNone => {
                        self.push_log("目標配置: 見つかりませんでした".into());
                        // 目指す形をリセット
                        self.cp.target_shape_rows = None;
                        self.cp.target_shape_key = None;
                        self.cp.searching = false;
                        self.cp.search_depth = 0;
                        self.cp_abort_flag.store(false, Ordering::Relaxed);
                        keep_rx = false;
                    }
                    CpMessage::Error(e) => {
                        self.push_log(format!("目標配置: エラー: {e}"));
                        self.cp.searching = false;
                        self.cp.search_depth = 0;
                        self.cp_abort_flag.store(false, Ordering::Relaxed);
                        keep_rx = false;
                    }
                }
            }
            if keep_rx { self.cp_rx = Some(rx); }
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.heading("ぷよぷよ 連鎖形 総当たり（6×14）— Rust GUI（列ストリーミング＋LRU形キャッシュ＋並列化＋計測＋追撃最適化）");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.mode, Mode::BruteForce, "総当たり");
                ui.selectable_value(&mut self.mode, Mode::ChainPlay, "連鎖生成");
            });
        });

        // 左ペイン全体をひとつの ScrollArea でまとめる
        egui::SidePanel::left("left").min_width(420.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
                if self.mode == Mode::BruteForce {
                    // ── 入力と操作（総当たり） ─────────────────────────────
                    ui.group(|ui| {
                        ui.label("入力と操作");
                        ui.label("左クリック: A→B→…→M / 中クリック: N→X→N / 右クリック: ・（空白） / Shift+左: RGBY");

                        ui.horizontal_wrapped(|ui| {
                            ui.add(egui::DragValue::new(&mut self.threshold).clamp_range(1..=19).speed(0.1));
                            ui.label("連鎖閾値");
                            ui.add_space(8.0);
                            // 同率タイ UI は連鎖生成モードに移動
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
                            ui.checkbox(&mut self.require_free_top_e1, "1連鎖目に free-top 列が1つ以上ある形に限定");
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
                } else {
                    // ── 連鎖生成モードの操作 ─────────────────────────────
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
                            if ui.add_enabled(can_ops && !self.cp.searching, egui::Button::new("ランダム配置")).clicked() {
                                self.cp_place_random();
                            }
                            if ui.add_enabled(can_ops && !self.cp.searching && self.cp.undo_stack.len() > 1, egui::Button::new("戻る")).clicked() {
                                self.cp_undo();
                            }
                            if ui.add_enabled(can_ops && !self.cp.searching, egui::Button::new("初手に戻る")).clicked() {
                                self.cp_reset_to_initial();
                            }
                        });

                        // 目標連鎖数と探索操作
                        ui.add_space(6.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.add(egui::DragValue::new(&mut self.cp.target_chain).clamp_range(1..=19).speed(0.2));
                            ui.label("目標連鎖数（＝で判定）");
                            let can_ops = !self.cp.lock && self.cp.anim.is_none();
                            if ui.add_enabled(can_ops && !self.cp.searching, egui::Button::new("目標配置")).clicked() {
                                self.cp_start_target_search();
                            }
                            if ui.add_enabled(self.cp.searching, egui::Button::new("停止")).clicked() {
                                self.cp_abort_flag.store(true, Ordering::Relaxed);
                            }
                            // JSONL ファイルの選択／解除
                            if ui.add_enabled(!self.cp.searching, egui::Button::new("JSONL選択")).clicked() {
                                if let Some(path) = rfd::FileDialog::new()
                                    .add_filter("JSON Lines", &["jsonl"]).pick_file() {
                                    self.cp.jsonl_path = Some(path.clone());
                                    self.push_log(format!("JSONLを選択: {}", path.display()));
                                }
                            }
                            if ui.add_enabled(!self.cp.searching && self.cp.jsonl_path.is_some(), egui::Button::new("JSONL解除")).clicked() {
                                self.cp.jsonl_path = None;
                                self.push_log("JSONL選択をクリア".into());
                            }
                        });
                        // 現在のJSONL読み込み元の表示
                        {
                            let cur = if let Some(p) = &self.cp.jsonl_path {
                                format!("選択中: {}", p.display())
                            } else if let Some(dir) = &self.out_path {
                                let d = dir.join(&self.out_name);
                                format!("既定: {}", d.display())
                            } else {
                                format!("既定: {}", &self.out_name)
                            };
                            ui.label(egui::RichText::new(cur).small().color(Color32::GRAY));
                        }
                        if self.cp.searching {
                            ui.label(format!("探索中… 深さ {}", self.cp.search_depth));
                        } else {
                            ui.label(if self.cp.lock { "連鎖中…（操作ロック）" } else { "待機中" });
                        }
                        if let Some(kind) = self.cp.last_found_kind {
                            match kind {
                                FoundKind::TargetExact => ui.label("直前: 目標連鎖＝達成で配置"),
                                FoundKind::Make3Group => ui.label("直前: 3連結を作成できる手を採用"),
                                FoundKind::MakeDiagonalSameColor => ui.label("直前: 同色の斜め隣接を作れる手を採用"),
                                FoundKind::Make2Group => ui.label("直前: 2連結を作成できる手を採用"),
                                FoundKind::OneChainMaxGroup => ui.label("直前: 1連鎖（最大連結）で発火"),
                                FoundKind::FullClearFallback => ui.label("直前: 全消し検出の最大連鎖で配置（fallback）"),
                                FoundKind::JsonlPattern => ui.label("直前: JSONL 形に寄与する手を採用"),
                                FoundKind::JsonlAvoid => ui.label("直前: JSONL 形を邪魔しない手を採用"),
                            };
                        }
                        ui.add_space(8.0);
                        ui.group(|ui| {
                            ui.label("配置制約（適用順）");
                            let can_edit = !self.cp.searching && self.cp.anim.is_none() && !self.cp.lock;
                            let len = self.cp.constraints.len();
                            let mut move_up_idx: Option<usize> = None;
                            let mut move_down_idx: Option<usize> = None;
                            let mut new_constraints = self.cp.constraints.clone();
                            for i in 0..len {
                                let mut c = new_constraints[i];
                                // 行ごとにユニークなID（インデックス）を使用して衝突回避
                                ui.push_id(i, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.monospace(format!("{:>2}.", i + 1));
                                        match c {
                                            CpConstraintKind::JsonlFollow => {
                                                ui.label("JSONLの形を目指す");
                                            }
                                            CpConstraintKind::JsonlAvoidInterfere { min_ratio_pct } => {
                                                ui.label("JSONLを保全（干渉回避）");
                                                let mut r = min_ratio_pct as i32;
                                                let resp = ui.add_enabled(can_edit, egui::DragValue::new(&mut r).clamp_range(0..=100).speed(1.0));
                                                if resp.changed() {
                                                    let rr = r.clamp(0, 100) as u8;
                                                    c = CpConstraintKind::JsonlAvoidInterfere { min_ratio_pct: rr };
                                                }
                                            }
                                            CpConstraintKind::TargetExactWithinDepth { max_depth } => {
                                                ui.label("目標＝達成（IDDFS）");
                                                let mut md = max_depth as i32;
                                                let resp = ui.add_enabled(can_edit, egui::DragValue::new(&mut md).clamp_range(1..=5).speed(0.1));
                                                if resp.changed() {
                                                    let md2 = md.clamp(1, 5) as usize;
                                                    c = CpConstraintKind::TargetExactWithinDepth { max_depth: md2 };
                                                }
                                            }
                                            CpConstraintKind::Make3GroupPrefer { avoid_diag_down23 } => {
                                                ui.label("現在手で3連結を作る");
                                                let mut av = avoid_diag_down23;
                                                let resp = ui.checkbox(&mut av, "斜め下2/3同色がある既存ぷよへの連結を避ける");
                                                if resp.changed() {
                                                    c = CpConstraintKind::Make3GroupPrefer { avoid_diag_down23: av };
                                                }
                                            }
                                            CpConstraintKind::MakeDiagonalSameColorPrefer => {
                                                ui.label("現在手で同色の斜めを作る");
                                            }
                                            CpConstraintKind::Make2GroupPrefer { avoid_diag_down23 } => {
                                                ui.label("現在手で2連結を作る");
                                                let mut av = avoid_diag_down23;
                                                let resp = ui.checkbox(&mut av, "斜め下2/3同色がある既存ぷよへの連結を避ける");
                                                if resp.changed() {
                                                    c = CpConstraintKind::Make2GroupPrefer { avoid_diag_down23: av };
                                                }
                                            }
                                            CpConstraintKind::OneChainMaxGroupPrefer { avoid_diag_down23 } => {
                                                ui.label("1連鎖（最大連結）で発火");
                                                let mut av = avoid_diag_down23;
                                                let resp = ui.checkbox(&mut av, "消去後にJSONL形に侵襲する落下を避ける");
                                                if resp.changed() {
                                                    c = CpConstraintKind::OneChainMaxGroupPrefer { avoid_diag_down23: av };
                                                }
                                            }
                                            CpConstraintKind::FullClearFallback { beam_depth, beam_width, pair_lookahead } => {
                                                ui.label("全消し検出（ビーム）");
                                                ui.monospace("深さ:");
                                                let mut bd = beam_depth as i32;
                                                let bd_resp = ui.add_enabled(can_edit, egui::DragValue::new(&mut bd).clamp_range(1..=16).speed(0.1));
                                                ui.monospace(" 幅:");
                                                let mut bw = beam_width as i32;
                                                let bw_resp = ui.add_enabled(can_edit, egui::DragValue::new(&mut bw).clamp_range(1..=64).speed(0.1));
                                                ui.monospace(" 先読み手数(ペア):");
                                                let mut pl = pair_lookahead as i32;
                                                let pl_resp = ui.add_enabled(can_edit, egui::DragValue::new(&mut pl).clamp_range(0..=10).speed(0.1));
                                                if bd_resp.changed() || bw_resp.changed() {
                                                    let bd2 = bd.clamp(1, 16) as usize;
                                                    let bw2 = bw.clamp(1, 64) as usize;
                                                    c = CpConstraintKind::FullClearFallback { beam_depth: bd2, beam_width: bw2, pair_lookahead };
                                                }
                                                if pl_resp.changed() {
                                                    let pl2 = pl.clamp(0, 10) as usize;
                                                    c = match c { CpConstraintKind::FullClearFallback { beam_depth: bd2, beam_width: bw2, .. } => CpConstraintKind::FullClearFallback { beam_depth: bd2, beam_width: bw2, pair_lookahead: pl2 }, _ => c };
                                                }
                                            }
                                        }
                                        // 並べ替えボタン（スワップは後でまとめて）
                                        if ui.add_enabled(can_edit && i > 0, egui::Button::new("↑")).clicked() {
                                            move_up_idx = Some(i);
                                        }
                                        if ui.add_enabled(can_edit && i + 1 < len, egui::Button::new("↓")).clicked() {
                                            move_down_idx = Some(i);
                                        }
                                    });
                                });
                                new_constraints[i] = c;
                            }
                            // ループ後にスワップを適用（途中で配列を変えない）
                            if let Some(idx) = move_up_idx { new_constraints.swap(idx, idx - 1); }
                            else if let Some(idx) = move_down_idx { new_constraints.swap(idx, idx + 1); }
                            self.cp.constraints = new_constraints;
                            ui.horizontal(|ui| {
                                if ui.add_enabled(can_edit, egui::Button::new("デフォルト順に戻す")).clicked() {
                                    self.cp.constraints = vec![
                                        CpConstraintKind::TargetExactWithinDepth { max_depth: 5 },
                                        CpConstraintKind::Make3GroupPrefer { avoid_diag_down23: true },
                                        CpConstraintKind::MakeDiagonalSameColorPrefer,
                                        CpConstraintKind::Make2GroupPrefer { avoid_diag_down23: true },
                                        CpConstraintKind::OneChainMaxGroupPrefer { avoid_diag_down23: true },
                                        CpConstraintKind::FullClearFallback { beam_depth: 8, beam_width: 16, pair_lookahead: 5 },
                                    ];
                                }
                            });
                        });
                        // ── 同率タイ時の優先ルール ──────────────────────
                        ui.group(|ui| {
                            ui.label("同率タイ時の優先ルール");
                            let can_edit = !self.cp.searching && self.cp.anim.is_none() && !self.cp.lock;
                            ui.checkbox(&mut self.cp.tie_break.apply_initial, "初手選択時に適用");
                            let len = self.cp.tie_break.rules.len();
                            let mut move_up_idx: Option<usize> = None;
                            let mut move_down_idx: Option<usize> = None;
                            let mut remove_idx: Option<usize> = None;
                            let mut new_rules = self.cp.tie_break.rules.clone();
                            for i in 0..len {
                                let mut r = new_rules[i];
                                ui.push_id(format!("tie-rule-{}", i), |ui| {
                                    ui.horizontal(|ui| {
                                        ui.monospace(format!("{:>2}.", i + 1));
                                        egui::ComboBox::from_id_source("rule")
                                            .selected_text(tie_rule_label(r))
                                            .show_ui(ui, |ui| {
                                                if ui.selectable_label(r == TieBreakRule::MinPlacementHeight, tie_rule_label(TieBreakRule::MinPlacementHeight)).clicked() { r = TieBreakRule::MinPlacementHeight; }
                                                if ui.selectable_label(r == TieBreakRule::LeftmostColumn, tie_rule_label(TieBreakRule::LeftmostColumn)).clicked() { r = TieBreakRule::LeftmostColumn; }
                                                if ui.selectable_label(r == TieBreakRule::PreferVertical, tie_rule_label(TieBreakRule::PreferVertical)).clicked() { r = TieBreakRule::PreferVertical; }
                                                if ui.selectable_label(r == TieBreakRule::PreferDiagonalSameColor, tie_rule_label(TieBreakRule::PreferDiagonalSameColor)).clicked() { r = TieBreakRule::PreferDiagonalSameColor; }
                                                if ui.selectable_label(r == TieBreakRule::PreferMake3GroupAvoidDiagDown23, tie_rule_label(TieBreakRule::PreferMake3GroupAvoidDiagDown23)).clicked() { r = TieBreakRule::PreferMake3GroupAvoidDiagDown23; }
                                                if ui.selectable_label(r == TieBreakRule::PreferMake2GroupAvoidDiagDown23, tie_rule_label(TieBreakRule::PreferMake2GroupAvoidDiagDown23)).clicked() { r = TieBreakRule::PreferMake2GroupAvoidDiagDown23; }
                                            });
                                        if ui.add_enabled(can_edit && i > 0, egui::Button::new("↑")).clicked() { move_up_idx = Some(i); }
                                        if ui.add_enabled(can_edit && i + 1 < len, egui::Button::new("↓")).clicked() { move_down_idx = Some(i); }
                                        if ui.add_enabled(can_edit, egui::Button::new("×")).clicked() { remove_idx = Some(i); }
                                    });
                                });
                                new_rules[i] = r;
                            }
                            if let Some(idx) = remove_idx { new_rules.remove(idx); }
                            if let Some(idx) = move_up_idx { new_rules.swap(idx, idx - 1); }
                            else if let Some(idx) = move_down_idx { new_rules.swap(idx, idx + 1); }
                            self.cp.tie_break.rules = new_rules;
                            ui.horizontal(|ui| {
                                if ui.add_enabled(can_edit, egui::Button::new("追加")).clicked() {
                                    self.cp.tie_break.rules.push(TieBreakRule::MinPlacementHeight);
                                }
                                if ui.add_enabled(can_edit, egui::Button::new("デフォルトに戻す")).clicked() {
                                    self.cp.tie_break = TieBreakPolicy::default();
                                }
                                if ui.add_enabled(can_edit, egui::Button::new("設定を保存")).clicked() {
                                    if let Err(e) = self.cp_save_settings() { self.push_log(format!("設定保存に失敗: {e}")); }
                                }
                                if ui.add_enabled(can_edit, egui::Button::new("設定を読込")).clicked() {
                                    if let Err(e) = self.cp_load_settings() { self.push_log(format!("設定読込に失敗: {e}")); }
                                }
                            });
                        });
                        ui.add_space(8.0);
                        egui::CollapsingHeader::new("ルールの説明（隠れルール）")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label(egui::RichText::new("JSONL 追従").strong());
                                ui.label("・表示用の形は、1連鎖目の free-top 列の最上段 1 セルを除外して判定します");
                                ui.label("・即時消去が起きる手は除外します");
                                ui.label("・注記: 除外セル（表示上の1セル）に置く場合は軽い減点（-10）があります");
                                ui.label("・2個寄与が最優先。1個寄与の場合、もう1個は '.'（非ラベル）上にのみ配置します");
                                ui.label("・既存/将来の同色直交連結でラベル領域に干渉する置き方は避けます");
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("JSONL 保全（干渉回避）").strong());
                                ui.label("・寄与手が無い場合、合致率しきい値以上の形を保全する手を選びます（デフォルト 50%）");
                                ui.label("・free-top 列の消去塊の最上段より上には置きません（前提崩しを回避）");
                                ui.label("・'.' への配置を優先し、ラベル位置は既に割当済みの一致色のみ許容します");
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("free-top / E1 の一般ルール").strong());
                                ui.label("・1連鎖目は単一グループかつ free-top 列が存在、最大連結サイズは 5 以下を重視します");
                                ui.label("・自分の追加ぷよが free-top 列の E1 消去に参加してしまう候補は避けます");
                                ui.label("・初手ペアが free-top E1 の消去成分に含まれる候補も避けます");
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("その他の制約").strong());
                                ui.label("・目標＝達成（IDDFS）、3連結/2連結優先、同色斜め優先、1連鎖（最大連結）など");
                                ui.label("・3/2連結優先では、既存ぷよの斜め下 2/3 同色への接続を避けるオプションがあります");
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("タイブレーク（デフォルト順）").strong());
                                ui.label("・同色斜め優先 → 3連結優先（斜め下2/3回避） → 2連結優先（斜め下2/3回避） → 最小設置列高 → 左端");
                                ui.add_space(6.0);
                                ui.label(egui::RichText::new("4個消しモード").strong());
                                ui.label("・初回消去が 4 以外なら不成立。E1 は空白隣接とオーバーハング条件を満たす必要があります");
                            });
                        ui.horizontal(|ui| {
                            if ui.button("全消しプレビュー更新").clicked() {
                                self.cp_update_full_clear_preview();
                            }
                            if self.cp.best_preview.is_some() {
                                ui.monospace(format!("MaxChain: {}", self.cp.best_chain));
                            }
                        });
                    });
                }

                ui.separator();

                // ── 処理時間（累積） ────────────────────────────────────────
                if self.mode == Mode::BruteForce && !self.running && self.profile_enabled && has_profile_any(&self.stats.profile) {
                    ui.group(|ui| {
                        ui.label("処理時間（累積）");
                        show_profile_table(ui, &self.stats.profile);
                    });
                    ui.separator();
                }

                // ── プレビュー ─────────────────────────────────────────────
                if self.mode == Mode::BruteForce {
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
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    // ── ログ ──────────────────────────────────────────────
                    ui.label("ログ");
                    for line in &self.log_lines {
                        ui.monospace(line);
                    }
                }

                ui.separator();

                if self.mode == Mode::BruteForce {
                    // ── 実行・進捗 ────────────────────────────────────────
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
            });
        });

        // 盤面側もスクロール可能に
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().auto_shrink([false, false]).show(ui, |ui| {
                if self.mode == Mode::BruteForce {
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
                } else {
                    ui.label("連鎖生成 — 盤面");
                    ui.add_space(6.0);
                    draw_preview(ui, &self.cp.cols);
                    ui.add_space(12.0);
                    ui.label("目標形（JSONL）");
                    if let Some(rows) = &self.cp.target_shape_rows {
                        let cols = rows_digits_to_cols(rows);
                        draw_preview(ui, &cols);
                    } else {
                        ui.label(egui::RichText::new("（JSONL 形：未選択）").italics().color(Color32::GRAY));
                    }
                    ui.add_space(12.0);
                    ui.label("全消しプレビュー");
                    if let Some(pv) = &self.cp.best_preview {
                        draw_preview(ui, pv);
                    } else {
                        ui.label(
                            egui::RichText::new("（ボタンで更新）").italics().color(Color32::GRAY),
                        );
                    }
                }
            });
        });

        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

// 連鎖生成モード：実装（App メソッド）
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
            // スナップショットは「配置前」に push 済みなので、
            // 直近のスナップショットそのものへ復元すれば 1 手 Undo になる。
            if let Some(last) = self.cp.undo_stack.pop() {
                self.cp.cols = last.cols;
                self.cp.pair_index = last.pair_index;
            }
            // 戻る後の盤面で JSONL 完成状態を再評価（形キーがある場合のみ）
            if let Some(ref pk) = self.cp.target_shape_key {
                // JSONL パス（優先: 手動選択したファイル → 既定の出力先/ファイル名）
                let jsonl_path = if let Some(p) = &self.cp.jsonl_path {
                    p.clone()
                } else if let Some(dir) = &self.out_path {
                    dir.join(&self.out_name)
                } else {
                    std::path::PathBuf::from(&self.out_name)
                };
                if let Ok(entries) = load_jsonl_entries(&jsonl_path) {
                    if let Some(ent) = entries.iter().find(|e| &e.key == pk) {
                        let shape_cols = rows_digits_to_cols(&ent.pre_chain_board);
                        if let Some((rx, ry)) = choose_removed_cell_for_display(&shape_cols) {
                            let mut labels_mod = ent.labels.clone();
                            labels_mod[ry][rx] = '.';
                            let done = is_shape_fully_filled(&self.cp.cols, &labels_mod);
                            self.cp.jsonl_completed = done;
                            self.cp.jsonl_completed_key = if done { Some(ent.key.clone()) } else { None };
                        }
                    }
                }
            }
        }
    }

    fn cp_reset_to_initial(&mut self) {
        if self.cp.lock { return; }
        self.cp.anim = None;
        self.cp.erased_cols = None;
        self.cp.next_cols = None;
        // 初手状態かどうか（スナップショットが初期のみ、かつ盤・インデックスが初期と一致）
        let at_initial = if let Some(first) = self.cp.undo_stack.first().copied() {
            self.cp.undo_stack.len() == 1 && self.cp.cols == first.cols && self.cp.pair_index == first.pair_index
        } else { false };

        if at_initial {
            // 初手状態での「初手に戻る」→ ツモ列をランダム再生成
            let mut rng = rand::thread_rng();
            self.cp.pair_seq.clear();
            for _ in 0..128 {
                self.cp.pair_seq.push((rng.gen_range(0..4), rng.gen_range(0..4)));
            }
            self.push_log("ツモ列を再生成しました".into());
            // 盤・インデックスはすでに初期と一致
        } else if let Some(first) = self.cp.undo_stack.first().copied() {
            // 通常の初手リセット
            self.cp.cols = first.cols;
            self.cp.pair_index = first.pair_index;
            // 初期スナップショットだけ残す
            self.cp.undo_stack.clear();
            self.cp.undo_stack.push(first);
        }
        // JSONL 完成フラグは解除（完成誤認を避ける）
        self.cp.jsonl_completed = false;
        self.cp.jsonl_completed_key = None;
        self.cp.lock = false;
    }

    fn cp_place_random(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() { return; }
        // 事前スナップショット
        self.cp.undo_stack.push(SavedState { cols: self.cp.cols, pair_index: self.cp.pair_index });

        let pair = self.cp_current_pair();
        let mut rng = rand::thread_rng();

        // 有効手集合を列挙
        let mut moves: Vec<(usize, Orient)> = Vec::new();
        for x in 0..W {
            // 垂直（Up/Down）: 同一列に2個置けるか
            let h = self.cp_col_height(x);
            if h + 1 < H { moves.push((x, Orient::Up)); moves.push((x, Orient::Down)); }
            // 右
            if x + 1 < W {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x + 1);
                if h0 < H && h1 < H { moves.push((x, Orient::Right)); }
            }
            // 左
            if x >= 1 {
                let h0 = self.cp_col_height(x);
                let h1 = self.cp_col_height(x - 1);
                if h0 < H && h1 < H { moves.push((x, Orient::Left)); }
            }
        }
        if moves.is_empty() {
            // 置けない（詰み）: スナップショットは維持するがインデックスは進めない
            self.push_log("置ける場所がありません".into());
            let _ = self.cp.undo_stack.pop();
            return;
        }
        let (x, orient) = moves[rng.gen_range(0..moves.len())];

        self.cp_place_with(x, orient, pair);
        // 連鎖開始チェック
        self.cp_check_and_start_chain();
    }

    fn cp_place_with(&mut self, x: usize, orient: Orient, pair: (u8, u8)) {
        // 盤面に2つの色を追加（重力後の最下段に直接置く）
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
        // 次の手へ
        if !self.cp.pair_seq.is_empty() {
            self.cp.pair_index = (self.cp.pair_index + 1) % self.cp.pair_seq.len();
        }
    }

    fn cp_col_height(&self, x: usize) -> usize {
        let occ = (self.cp.cols[0][x] | self.cp.cols[1][x] | self.cp.cols[2][x] | self.cp.cols[3][x]) & MASK14;
        occ.count_ones() as usize
    }

    fn cp_set_cell(&mut self, x: usize, y: usize, color: u8) {
        if x >= W || y >= H { return; }
        let bit = 1u16 << y;
        let c = (color as usize).min(3);
        self.cp.cols[c][x] |= bit;
    }

    fn cp_check_and_start_chain(&mut self) {
        // 4連結抽出
        let clear = compute_erase_mask_cols(&self.cp.cols);
        let any = (0..W).any(|x| clear[x] != 0);
        if !any {
            // 連鎖なし
            return;
        }
        // 連鎖開始：操作ロック
        self.cp.lock = true;
        // 消去直後表示と次盤面作成
        let erased = apply_clear_no_fall(&self.cp.cols, &clear);
        let next = apply_given_clear_and_fall(&self.cp.cols, &clear);
        self.cp.erased_cols = Some(erased);
        self.cp.next_cols = Some(next);
        self.cp.cols = erased; // 消えた状態を表示
        self.cp.anim = Some(AnimState { phase: AnimPhase::AfterErase, since: Instant::now() });
    }

    // ===== 全消しプレビュー更新 =====
    fn cp_update_full_clear_preview(&mut self) {
        // Python版 iterative_chain_clearing に相当するビームサーチを使用
        let original = self.cp.cols;
        let max_depth = 8usize;
        let beam_width = 16usize;
        if let Some((adds, total_chain)) = cp_iterative_chain_clearing_beam(original, max_depth, beam_width, None) {
            let merged = merge_additions_onto(original, &adds);
            self.cp.best_preview = Some(merged);
            self.cp.best_chain = total_chain;
        } else {
            // 厳格条件を満たす解が見つからないときは、緩和版ビームでベストをプレビュー
            let (adds, total_chain) = cp_iterative_chain_clearing_beam_relaxed(original, max_depth, beam_width);
            let merged = merge_additions_onto(original, &adds);
            self.cp.best_preview = Some(merged);
            self.cp.best_chain = total_chain;
        }
    }

    fn cp_step_animation(&mut self) {
        let Some(anim) = self.cp.anim else { return; };
        let elapsed = anim.since.elapsed();
        if elapsed < Duration::from_millis(500) { return; }
        match anim.phase {
            AnimPhase::AfterErase => {
                if let Some(next) = self.cp.next_cols.take() {
                    self.cp.cols = next;
                }
                self.cp.anim = Some(AnimState { phase: AnimPhase::AfterFall, since: Instant::now() });
                // 次の消去準備は AfterFall 経由
            }
            AnimPhase::AfterFall => {
                // 次の連鎖チェック
                let clear = compute_erase_mask_cols(&self.cp.cols);
                let any = (0..W).any(|x| clear[x] != 0);
                if !any {
                    // 完了
                    self.cp.anim = None;
                    self.cp.erased_cols = None;
                    self.cp.next_cols = None;
                    self.cp.lock = false;
                } else {
                    let erased = apply_clear_no_fall(&self.cp.cols, &clear);
                    let next = apply_given_clear_and_fall(&self.cp.cols, &clear);
                    self.cp.cols = erased;
                    self.cp.erased_cols = Some(erased);
                    self.cp.next_cols = Some(next);
                    self.cp.anim = Some(AnimState { phase: AnimPhase::AfterErase, since: Instant::now() });
                }
            }
        }
    }
}

// ===== 連鎖生成モード：目標配置（IDDFS 非同期探索） =====
impl App {
    fn cp_start_target_search(&mut self) {
        if self.cp.lock || self.cp.anim.is_some() || self.cp.searching { return; }
        // 準備
        let original = self.cp.cols;
        let pair_seq = self.cp.pair_seq.clone();
        let pair_index = self.cp.pair_index;
        let target = self.cp.target_chain.max(1);
        let constraints = self.cp.constraints.clone();
        let tie_break = self.cp.tie_break.clone();
        let preferred_key = self.cp.target_shape_key.clone();
        let jsonl_completed = self.cp.jsonl_completed;
        let _jsonl_completed_key = self.cp.jsonl_completed_key.clone();
        // JSONL パス（優先: 手動選択したファイル → 既定の出力先/ファイル名）
        let jsonl_path = if let Some(p) = &self.cp.jsonl_path {
            p.clone()
        } else if let Some(dir) = &self.out_path {
            dir.join(&self.out_name)
        } else {
            std::path::PathBuf::from(&self.out_name)
        };
        // 読み込み（UI スレッドで実行してから move）
        let jsonl_entries = match load_jsonl_entries(&jsonl_path) {
            Ok(v) => v,
            Err(e) => {
                self.push_log(format!("JSONL 読み込み失敗: {}", e));
                Vec::new()
            }
        };

        let (tx, rx) = unbounded::<CpMessage>();
        self.cp_rx = Some(rx);
        self.cp.searching = true;
        self.cp.search_depth = 0;
        self.cp_abort_flag.store(false, Ordering::Relaxed);

        let abort = self.cp_abort_flag.clone();
        thread::spawn(move || {
            // 制約リストの順序で適用
            let pair_len = pair_seq.len().max(1);
            let start_pair = pair_seq[pair_index % pair_len];
            let _ = tx.send(CpMessage::Log("目標配置探索を開始".into()));
            // UI から受け取った JSONL 完成フラグ（ローカルコピー）
            let mut jsonl_completed_flag = jsonl_completed;

            for constraint in constraints.iter() {
                if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                match *constraint {
                    CpConstraintKind::JsonlFollow => {
                        // 事前: JSONL 完成済みなら全スキップ
                        let mut skip_jsonl = jsonl_completed_flag;
                        // さらに、現在の優先形（表示中の形）が「free-top 最上段以外はすべて埋まっている」なら、このモジュールをスキップ
                        if let Some(ref pk) = preferred_key {
                            if let Some(ent_pk) = jsonl_entries.iter().find(|e| &e.key == pk) {
                                let shape_cols = rows_digits_to_cols(&ent_pk.pre_chain_board);
                                if let Some((rx, ry)) = choose_removed_cell_for_display(&shape_cols) {
                                    let mut labels_mod = ent_pk.labels.clone();
                                    labels_mod[ry][rx] = '.';
                                    if !skip_jsonl && is_shape_fully_filled(&original, &labels_mod) {
                                        skip_jsonl = true;
                                        if !jsonl_completed_flag {
                                            let _ = tx.send(CpMessage::JsonlCompleted { key: ent_pk.key.clone() });
                                            jsonl_completed_flag = true;
                                        }
                                    }
                                }
                            }
                        }
                        if skip_jsonl { /* 次の制約へ */ } else {
                        // 1) 形の候補を順に試し、現盤面から作成可能か（抽象ラベル）をチェック
                        // 2) 最初に「寄与する手」が見つかった形を採用し、その場で Found を送って終了
                        let mut found_any = false;
                        // 優先: 前回選択した形
                        let mut indices: Vec<usize> = (0..jsonl_entries.len()).collect();
                        if let Some(ref pk) = preferred_key {
                            if let Some(pos) = jsonl_entries.iter().position(|e| &e.key == pk) {
                                if pos != 0 { indices.remove(pos); indices.insert(0, pos); }
                            }
                        }
                        'shape_loop: for &i in indices.iter() {
                            let ent = &jsonl_entries[i];
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            // pre_chain_board からビットボード
                            let shape_cols = rows_digits_to_cols(&ent.pre_chain_board);
                            // E1 の free-top 列から 1 個除外（表示用）
                            let removed = choose_removed_cell_for_display(&shape_cols);
                            if removed.is_none() { continue; }
                            let removed = removed.unwrap();
                            // ラベルグリッド（抽象ラベル）に除外を反映
                            let mut labels_mod = ent.labels.clone();
                            labels_mod[removed.1][removed.0] = '.';
                            // 形が現盤面から作れるか（既存配置で矛盾がないか）
                            if !pattern_compatible_with_board(&original, &labels_mod) { continue; }

                            // 候補手をスコアリング
                            let moves = cp_generate_moves_from_cols(&original);
                            let mut scored: Vec<((usize, Orient), i32)> = Vec::new();
                            for (x, orient) in moves {
                                if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                                if let Some(((ax, ay), (bx, by))) = cp_positions_for_move(&original, x, orient) {
                                    // 即時消去が起きる手はスキップ（形づくり優先）
                                    let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                                    let clear = compute_erase_mask_cols(&after1);
                                    if (0..W).any(|cx| clear[cx] != 0) { continue; }
                                    let place_ok = evaluate_jsonl_contribution(
                                        &original,
                                        (ax, ay), (bx, by),
                                        start_pair,
                                        &labels_mod,
                                        removed,
                                    );
                                    if let Some(score) = place_ok {
                                        scored.push(((x, orient), score));
                                    }
                                }
                            }
                            if !scored.is_empty() {
                                // UI に目標形を通知（この形で寄与可能なときのみ）
                                let mut shown = ent.pre_chain_board.clone();
                                let (rx, ry) = removed;
                                if ry < shown.len() && rx < shown[ry].len() {
                                    let mut row = shown[ry].chars().collect::<Vec<char>>();
                                    row[rx] = '.';
                                    shown[ry] = row.into_iter().collect();
                                }
                                let _ = tx.send(CpMessage::TargetShape { rows: shown.clone(), key: ent.key.clone() });
                            }
                            if let Some((bx, bo)) = select_best_scored_i32(&scored, &original, &tie_break) {
                                let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::JsonlPattern, fc_preview: None, fc_chain: None });
                                found_any = true;
                                break 'shape_loop;
                            }
                            // 寄与手なし → 次の形を試す
                        }
                        if found_any { return; }
                        }
                        // 見つからなければ次の制約へフォールスルー
                    }
                    CpConstraintKind::JsonlAvoidInterfere { min_ratio_pct } => {
                        // 事前: JSONL 完成済みなら全スキップ
                        let mut skip_jsonl = jsonl_completed_flag;
                        // さらに、現在の優先形（表示中の形）が「free-top 最上段以外はすべて埋まっている」なら、このモジュールをスキップ
                        if let Some(ref pk) = preferred_key {
                            if let Some(ent_pk) = jsonl_entries.iter().find(|e| &e.key == pk) {
                                let shape_cols = rows_digits_to_cols(&ent_pk.pre_chain_board);
                                if let Some((rx, ry)) = choose_removed_cell_for_display(&shape_cols) {
                                    let mut labels_mod = ent_pk.labels.clone();
                                    labels_mod[ry][rx] = '.';
                                    if !skip_jsonl && is_shape_fully_filled(&original, &labels_mod) {
                                        skip_jsonl = true;
                                        if !jsonl_completed_flag {
                                            let _ = tx.send(CpMessage::JsonlCompleted { key: ent_pk.key.clone() });
                                            jsonl_completed_flag = true;
                                        }
                                    }
                                }
                            }
                        }
                        if skip_jsonl { /* 次の制約へ */ } else {
                        // 形が全く寄与できない場合、現局面と合致率が高い形を保全するため、
                        // その形を邪魔しない（他色でラベル位置を潰さない）手を選ぶ
                        let mut best_move: Option<(usize, Orient, i32, usize)> = None; // (x,orient,score,shape_index)
                        // 優先: 直前の形
                        let mut indices: Vec<usize> = (0..jsonl_entries.len()).collect();
                        if let Some(ref pk) = preferred_key {
                            if let Some(pos) = jsonl_entries.iter().position(|e| &e.key == pk) {
                                if pos != 0 { indices.remove(pos); indices.insert(0, pos); }
                            }
                        }
                        for &i in indices.iter() {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            let ent = &jsonl_entries[i];
                            // free-top 列から 1 個除外した形で保全を評価する
                            let shape_cols = rows_digits_to_cols(&ent.pre_chain_board);
                            let removed = if let Some(rc) = choose_removed_cell_for_display(&shape_cols) { rc } else { continue; };
                            let mut labels_mod = ent.labels.clone();
                            labels_mod[removed.1][removed.0] = '.';
                            if let Some((col_to_label, label_to_col, ratio)) = compute_mapping_and_ratio(&original, &labels_mod) {
                                if (ratio * 100.0) < (min_ratio_pct as f32) { continue; }
                                // 候補手のうち、形に干渉しないものを評価
                                let moves = cp_generate_moves_from_cols(&original);
                                // free-top 列の top_y を計算（この上には置かない）
                                let clear_shape = compute_erase_mask_cols(&shape_cols);
                                let ft_shape = first_clear_free_top_cols(&shape_cols);
                                let mut top_y: [Option<usize>; W] = [None; W];
                                for x2 in 0..W {
                                    if !ft_shape[x2] { continue; }
                                    let m = clear_shape[x2] & MASK14;
                                    if m == 0 { continue; }
                                    for y2 in (0..H).rev() {
                                        if (m & (1u16 << y2)) != 0 { top_y[x2] = Some(y2); break; }
                                    }
                                }
                                for (x, orient) in moves {
                                    if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                                    // 即時消去手は避ける
                                    let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                                    let clear = compute_erase_mask_cols(&after1);
                                    if (0..W).any(|cx| clear[cx] != 0) { continue; }
                                    if let Some(((ax, ay), (bx, by))) = cp_positions_for_move(&original, x, orient) {
                                        // free-top 列の上側には置かない（形の前提を壊さない）
                                        if let Some(ty) = top_y[ax] { if ay > ty { continue; } }
                                        if let Some(ty) = top_y[bx] { if by > ty { continue; } }
                                        if move_avoids_interference(
                                            &original, (ax, ay), (bx, by), start_pair, &labels_mod, &col_to_label, &label_to_col
                                        ) {
                                            // スコア: '.' に置くほど良い（完全回避を優先）
                                            let mut sc = 0i32;
                                            let la = labels_mod[ay][ax];
                                            let lb = labels_mod[by][bx];
                                            if la == '.' { sc += 10; } else { sc += 3; }
                                            if lb == '.' { sc += 10; } else { sc += 3; }
                                            match best_move {
                                                None => best_move = Some((x, orient, sc, i)),
                                                Some((_bx,_bo, bsc, _bi)) => {
                                                    if sc > bsc || (sc == bsc && tie_break_choose_with_pair((x,orient), (_bx,_bo), &original, start_pair, &tie_break)) {
                                                        best_move = Some((x, orient, sc, i));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if let Some((bx, bo, _sc, bi)) = best_move {
                            // UI に保全対象の形を表示（free-top 列から1個除外した形）
                            let ent = &jsonl_entries[bi];
                            let shape_cols = rows_digits_to_cols(&ent.pre_chain_board);
                            if let Some((rx, ry)) = choose_removed_cell_for_display(&shape_cols) {
                                let mut shown = ent.pre_chain_board.clone();
                                if ry < shown.len() && rx < shown[ry].len() {
                                    let mut row = shown[ry].chars().collect::<Vec<char>>();
                                    row[rx] = '.';
                                    shown[ry] = row.into_iter().collect();
                                }
                                let _ = tx.send(CpMessage::TargetShape { rows: shown, key: ent.key.clone() });
                            } else {
                                // 念のためフォールバック（通常は到達しない）
                                let _ = tx.send(CpMessage::TargetShape { rows: ent.pre_chain_board.clone(), key: ent.key.clone() });
                            }
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::JsonlAvoid, fc_preview: None, fc_chain: None });
                            return;
                        }
                        }
                        // 見つからなければ次の制約へ
                    }
                    CpConstraintKind::TargetExactWithinDepth { max_depth } => {
                        for depth in 1..=max_depth.max(1) {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            let _ = tx.send(CpMessage::Depth(depth));
                            let moves = cp_generate_moves_from_cols(&original);
                            let mut cands: Vec<(usize, Orient)> = Vec::new();
                            for (x, orient) in moves {
                                if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                                let after = cp_apply_move_pure(&original, x, orient, start_pair);
                                let (gain, leftover) = simulate_chain_and_final(after);
                                if gain == target { cands.push((x, orient)); continue; }
                                if depth > 1 {
                                    // 重要: 再帰には連鎖を解決した安定盤面（leftover）を渡す
                                    if dfs_exact_reachable(leftover, &pair_seq, pair_index + 1, depth - 1, target, &abort) {
                                        cands.push((x, orient));
                                    }
                                }
                            }
                            if let Some((bx, bo)) = select_best_move_with_pair(&cands, &original, start_pair, &tie_break) {
                                let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::TargetExact, fc_preview: None, fc_chain: None });
                                return;
                            }
                        }
                    }
                    CpConstraintKind::Make3GroupPrefer { avoid_diag_down23 } => {
                        let moves = cp_generate_moves_from_cols(&original);
                        let mut cands: Vec<(usize, Orient)> = Vec::new();
                        for (x, orient) in moves {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            if let Some((pos_axis, pos_child)) = cp_positions_for_move(&original, x, orient) {
                                let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                                // 即時消去が起きる手はスキップ
                                let clear = compute_erase_mask_cols(&after1);
                                if (0..W).any(|cx| clear[cx] != 0) { continue; }
                                // 斜め下 2/3 連結回避オプション
                                if avoid_diag_down23 {
                                    // 既存ぷよへの連結を避けたい場合は、事前盤面 original で評価
                                    if avoid_connect_if_neighbor_has_diag_down_23(&original, start_pair.0, pos_axis) { continue; }
                                    if avoid_connect_if_neighbor_has_diag_down_23(&original, start_pair.1, pos_child) { continue; }
                                }
                                let sz_a = comp_size_for_color_at(&after1, start_pair.0, pos_axis.0, pos_axis.1);
                                let sz_b = comp_size_for_color_at(&after1, start_pair.1, pos_child.0, pos_child.1);
                                if sz_a == 3 || sz_b == 3 { cands.push((x, orient)); }
                            }
                        }
                        if let Some((bx, bo)) = select_best_move_with_pair(&cands, &original, start_pair, &tie_break) {
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::Make3Group, fc_preview: None, fc_chain: None });
                            return;
                        }
                    }
                    CpConstraintKind::MakeDiagonalSameColorPrefer => {
                        let moves = cp_generate_moves_from_cols(&original);
                        let mut cands: Vec<(usize, Orient)> = Vec::new();
                        for (x, orient) in moves {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                            // 即時消去が起きる手はスキップ
                            let clear = compute_erase_mask_cols(&after1);
                            if (0..W).any(|cx| clear[cx] != 0) { continue; }
                            if diagonal_same_color_for_move(&original, x, orient, start_pair) {
                                cands.push((x, orient));
                            }
                        }
                        if let Some((bx, bo)) = select_best_move_with_pair(&cands, &original, start_pair, &tie_break) {
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::MakeDiagonalSameColor, fc_preview: None, fc_chain: None });
                            return;
                        }
                    }
                    CpConstraintKind::Make2GroupPrefer { avoid_diag_down23 } => {
                        let moves = cp_generate_moves_from_cols(&original);
                        let mut cands: Vec<(usize, Orient)> = Vec::new();
                        for (x, orient) in moves {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            if let Some((pos_axis, pos_child)) = cp_positions_for_move(&original, x, orient) {
                                let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                                // 即時消去が起きる手はスキップ
                                let clear = compute_erase_mask_cols(&after1);
                                if (0..W).any(|cx| clear[cx] != 0) { continue; }
                                // 斜め下 2/3 連結回避オプション
                                if avoid_diag_down23 {
                                    if avoid_connect_if_neighbor_has_diag_down_23(&original, start_pair.0, pos_axis) { continue; }
                                    if avoid_connect_if_neighbor_has_diag_down_23(&original, start_pair.1, pos_child) { continue; }
                                }
                                let sz_a = comp_size_for_color_at(&after1, start_pair.0, pos_axis.0, pos_axis.1);
                                let sz_b = comp_size_for_color_at(&after1, start_pair.1, pos_child.0, pos_child.1);
                                if sz_a == 2 || sz_b == 2 { cands.push((x, orient)); }
                            }
                        }
                        if let Some((bx, bo)) = select_best_move_with_pair(&cands, &original, start_pair, &tie_break) {
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::Make2Group, fc_preview: None, fc_chain: None });
                            return;
                        }
                    }
                    CpConstraintKind::OneChainMaxGroupPrefer { avoid_diag_down23 } => {
                        let moves = cp_generate_moves_from_cols(&original);
                        // JSONL 形ガード用マスク（ラベル位置を1にする）。preferred_key がある場合のみ作る。
                        let guard_mask_opt: Option<[u16; W]> = if avoid_diag_down23 {
                            if let Some(ref pk) = preferred_key {
                                if let Some(ent) = jsonl_entries.iter().find(|e| &e.key == pk) {
                                    let mut mask = [0u16; W];
                                    for y in 0..H { for x in 0..W { if ent.labels[y][x] != '.' { mask[x] |= 1u16 << y; } } }
                                    Some(mask)
                                } else { None }
                            } else { None }
                        } else { None };

                        let mut scored: Vec<((usize, Orient), i32)> = Vec::new();
                        for (x, orient) in moves {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            if cp_positions_for_move(&original, x, orient).is_none() { continue; }
                            let after1 = cp_apply_move_pure(&original, x, orient, start_pair);
                            // まず即時消去が必要
                            let clear = compute_erase_mask_cols(&after1);
                            if (0..W).all(|cx| clear[cx] == 0) { continue; }
                            // 総連鎖数がちょうど 1 であること（落下まで反映）
                            let (cc, final_board) = simulate_chain_and_final(after1);
                            if cc != 1 { continue; }

                            // JSONL 形への侵襲回避: 最終盤面で新規にラベル位置が埋まるなら拒否
                            if let Some(guard) = guard_mask_opt {
                                let mut invaded = false;
                                for cx in 0..W {
                                    let orig_occ = (original[0][cx] | original[1][cx] | original[2][cx] | original[3][cx]) & guard[cx];
                                    let fin_occ  = (final_board[0][cx] | final_board[1][cx] | final_board[2][cx] | final_board[3][cx]) & guard[cx];
                                    if (fin_occ & !orig_occ) != 0 { invaded = true; break; }
                                }
                                if invaded { continue; }
                            }

                            // スコア: 1連鎖目で消える塊の最大連結サイズ（after1 基準）
                            let best_sz = first_clear_largest_size(&after1) as i32;
                            scored.push(((x, orient), best_sz));
                        }
                        if let Some((bx, bo)) = select_best_scored_i32_with_pair(&scored, &original, start_pair, &tie_break) {
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::OneChainMaxGroup, fc_preview: None, fc_chain: None });
                            return;
                        }
                    }
                    CpConstraintKind::FullClearFallback { beam_depth, beam_width, pair_lookahead } => {
                        let moves = cp_generate_moves_from_cols(&original);
                        if moves.is_empty() { continue; }
                        let mut best: Option<(usize, Orient, u32, Option<[[u16; W]; 4]>, Option<u32>)> = None;
                        for (x, orient) in moves {
                            if abort.load(Ordering::Relaxed) { let _ = tx.send(CpMessage::FinishedNone); return; }
                            let first_pos = cp_positions_for_move(&original, x, orient);
                            let after1 = cp_apply_move_pure(&original, x, orient, start_pair);

                            // 初手（ペア）で E1 が発生し、かつ free-top 列の上に直置きして完成させる手は回避
                            if let Some(((ax, ay), (bx, by))) = first_pos {
                                let clear = compute_erase_mask_cols(&after1);
                                let groups = count_first_clear_groups(&after1);
                                if groups >= 1 && first_clear_has_free_top(&after1) {
                                    let ft = first_clear_free_top_cols(&after1);
                                    let abit = 1u16 << ay;
                                    let bbit = 1u16 << by;
                                    let a_on_clear = (clear[ax] & abit) != 0;
                                    let b_on_clear = (clear[bx] & bbit) != 0;
                                    if (ft[ax] && a_on_clear) || (ft[bx] && b_on_clear) {
                                        // この候補はスキップ
                                        continue;
                                    }
                                }
                            }
                            let (score, fc_prev, fc_tc) = if pair_lookahead > 0 {
                                cp_evaluate_with_pair_lookahead(
                                    after1,
                                    &pair_seq,
                                    pair_index + 1,
                                    pair_lookahead,
                                    beam_depth.max(1),
                                    beam_width.max(1),
                                    &abort,
                                    first_pos,
                                )
                            } else if let Some((adds, total_chain)) = cp_iterative_chain_clearing_beam(after1, beam_depth.max(1), beam_width.max(1), first_pos) {
                                let merged = merge_additions_onto(after1, &adds);
                                (total_chain, Some(merged), Some(total_chain))
                            } else {
                                (simulate_chain_count_simple(after1), None, None)
                            };
                            match &mut best {
                                None => best = Some((x, orient, score, fc_prev, fc_tc)),
                                Some((cbx, cbo, bs, _bpv, _btc)) => {
                                    if score > *bs {
                                        *bs = score; best = Some((x, orient, score, fc_prev, fc_tc));
                                    } else if score == *bs && tie_break.apply_initial {
                                        // タイブレーク適用（trueなら新候補に置き換える）
                                        if tie_break_choose_with_pair((x, orient), (*cbx, *cbo), &original, start_pair, &tie_break) {
                                            best = Some((x, orient, score, fc_prev, fc_tc));
                                        }
                                    }
                                }
                            }
                        }
                        if let Some((bx, bo, _sc, bpv, btc)) = best {
                            let _ = tx.send(CpMessage::Found { x: bx, orient: bo, pair: start_pair, kind: FoundKind::FullClearFallback, fc_preview: bpv, fc_chain: btc });
                            return;
                        }
                    }
                }
            }
            // どの制約でも見つからず
            let _ = tx.send(CpMessage::FinishedNone);
        });
    }
}

// ===== JSONL 形 追従の実装補助 =====

#[derive(Deserialize, Clone)]
struct JsonlLineIn {
    key: String,
    hash: u32,
    chains: u32,
    pre_chain_board: Vec<String>,
    example_mapping: HashMap<char, u8>,
    mirror: bool,
}

#[derive(Clone)]
struct JsonlEntry {
    key: String,
    pre_chain_board: Vec<String>,
    labels: Vec<Vec<char>>, // H rows × W cols, from key ('.' or 'A'..) and de-mirrored to match rows
}

fn load_jsonl_entries(path: &std::path::Path) -> Result<Vec<JsonlEntry>> {
    if !path.exists() { return Ok(Vec::new()); }
    let f = File::open(path).with_context(|| format!("JSONL を開けませんでした: {}", path.display()))?;
    let rdr = BufReader::new(f);
    let mut out = Vec::new();
    for line in rdr.lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let v: JsonlLineIn = serde_json::from_str(&line)
            .with_context(|| "JSONL 1行の解析に失敗")?;
        let mut labels = key_to_label_grid(&v.key);
        // key は canonical orientation（mirror 適用済み）のため、rows と合わせるには mirror=true の場合に左右反転する
        if v.mirror {
            for y in 0..H {
                labels[y].reverse();
            }
        }
        out.push(JsonlEntry { key: v.key, pre_chain_board: v.pre_chain_board, labels });
    }
    Ok(out)
}

#[inline(always)]
fn key_to_label_grid(key: &str) -> Vec<Vec<char>> {
    // key は x 外側, y 内側で push されている（encode_canonical_string 参照）
    let mut grid = vec![vec!['.'; W]; H];
    let chars: Vec<char> = key.chars().collect();
    if chars.len() != W * H { return grid; }
    for x in 0..W {
        for y in 0..H {
            let idx = x * H + y;
            grid[y][x] = chars[idx];
        }
    }
    grid
}

#[inline(always)]
fn rows_digits_to_cols(rows: &Vec<String>) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];
    let hh = rows.len().min(H);
    for y in 0..hh {
        let row = rows[y].as_bytes();
        for x in 0..W.min(row.len()) {
            let ch = row[x];
            if ch >= b'0' && ch <= b'3' {
                let c = (ch - b'0') as usize;
                out[c][x] |= 1u16 << y;
            }
        }
    }
    out
}

#[inline(always)]
fn pattern_compatible_with_board(board: &[[u16; W]; 4], labels: &Vec<Vec<char>>) -> bool {
    let mut col_to_label: HashMap<u8, char> = HashMap::new();
    let mut label_to_col: HashMap<char, u8> = HashMap::new();
    for x in 0..W {
        for y in 0..H {
            let bit = 1u16 << y;
            let mut c_opt: Option<u8> = None;
            for c in 0..4 { if (board[c][x] & bit) != 0 { c_opt = Some(c as u8); break; } }
            if let Some(c) = c_opt {
                let lab = labels[y][x];
                if lab == '.' { continue; }
                if let Some(&l0) = col_to_label.get(&c) { if l0 != lab { return false; } } else {
                    if let Some(&c0) = label_to_col.get(&lab) { if c0 != c { return false; } }
                    col_to_label.insert(c, lab);
                    label_to_col.insert(lab, c);
                }
            }
        }
    }
    true
}

#[inline(always)]
fn choose_removed_cell_for_display(cols: &[[u16; W]; 4]) -> Option<(usize, usize)> {
    // 1連鎖目の free-top 列から、消える塊のうち一番上の 1 個を除外
    let clear = compute_erase_mask_cols(cols);
    if !first_clear_has_free_top(cols) { return None; }
    let ft = first_clear_free_top_cols(cols);
    for x in 0..W {
        if !ft[x] { continue; }
        let mut col_mask = clear[x] & MASK14;
        if col_mask == 0 { continue; }
        // 最上段（y の大きい）を選ぶ
        let mut sel_y: Option<usize> = None;
        for y in (0..H).rev() {
            if (col_mask & (1u16 << y)) != 0 { sel_y = Some(y); break; }
        }
        if let Some(y) = sel_y { return Some((x, y)); }
    }
    None
}

// 評価: JSONL 形への寄与スコアを返す（なければ None）
// ルール: 2個寄与=OK、1個寄与=もう1個は形の '.' 上に置くこと
// スコア: 2個寄与=100、1個寄与=40、さらに除外セルを埋めるなら +20
#[inline(always)]
fn evaluate_jsonl_contribution(
    board: &[[u16; W]; 4],
    pos_a: (usize, usize),
    pos_b: (usize, usize),
    pair: (u8, u8),
    labels: &Vec<Vec<char>>, // from key
    removed_cell: (usize, usize),
) -> Option<i32> {
    // 既存配置との整合性をマップに反映
    let mut col_to_label: HashMap<u8, char> = HashMap::new();
    let mut label_to_col: HashMap<char, u8> = HashMap::new();
    for x in 0..W {
        for y in 0..H {
            let bit = 1u16 << y;
            for c in 0..4 {
                if (board[c][x] & bit) != 0 {
                    let lab = labels[y][x];
                    if lab == '.' { continue; }
                    let cu = c as u8;
                    if let Some(&l0) = col_to_label.get(&cu) { if l0 != lab { return None; } } else {
                        if let Some(&c0) = label_to_col.get(&lab) { if c0 != cu { return None; } }
                        col_to_label.insert(cu, lab);
                        label_to_col.insert(lab, cu);
                    }
                }
            }
        }
    }

    // 置き先のラベル
    let (ax, ay) = pos_a; let (bx, by) = pos_b;
    let la = labels[ay][ax];
    let lb = labels[by][bx];
    let a_is_label = la != '.';
    let b_is_label = lb != '.';

    // 2個寄与
    if a_is_label && b_is_label {
        if !unify_color_label(pair.0, la, &mut col_to_label, &mut label_to_col) { return None; }
        if !unify_color_label(pair.1, lb, &mut col_to_label, &mut label_to_col) { return None; }
        // removed_cell は表示上の除外候補。原則として避ける（軽いペナルティ）。
        let mut score = 100;
        if (ax, ay) == removed_cell || (bx, by) == removed_cell { score -= 10; }
        return Some(score);
    }
    // 1個寄与: もう1つは '.' 上に置く
    if a_is_label ^ b_is_label {
        let (px, py, col, lab, other, other_col) = if a_is_label { (ax, ay, pair.0, la, (bx, by), pair.1) } else { (bx, by, pair.1, lb, (ax, ay), pair.0) };
        // other はパターン '.' である必要
        if labels[other.1][other.0] != '.' { return None; }
        if !unify_color_label(col, lab, &mut col_to_label, &mut label_to_col) { return None; }
        // '.' へ置く側が、既存の形（ラベル領域に既に置かれている同色）に直交隣接して連結しないこと
        if connects_to_shape(board, labels, other.0, other.1, other_col) { return None; }
        // 将来その隣接ラベルが同色になり得て連結してしまう置き方も避ける
        if would_future_connect_to_shape(labels, other.0, other.1, other_col, &col_to_label, &label_to_col) { return None; }
        // removed_cell は避ける（軽いペナルティ）
        let mut score = 40;
        if (px, py) == removed_cell { score -= 10; }
        return Some(score);
    }
    None
}

#[inline(always)]
fn unify_color_label(col: u8, lab: char, col_to_label: &mut HashMap<u8, char>, label_to_col: &mut HashMap<char, u8>) -> bool {
    if let Some(&l0) = col_to_label.get(&col) { if l0 != lab { return false; } } else {
        if let Some(&c0) = label_to_col.get(&lab) { if c0 != col { return false; } }
        col_to_label.insert(col, lab);
        label_to_col.insert(lab, col);
    }
    true
}

// 既存盤面に基づいて、色<->ラベルの対応を構築し、合致率を返す
// 合致率 = 「ラベル付きセルのうち、現在すでに正しい色が置かれている数」/「現在ラベル付きセルに置かれている総数」
// 一貫した対応が作れない場合は None
fn compute_mapping_and_ratio(
    board: &[[u16; W]; 4],
    labels: &Vec<Vec<char>>,
) -> Option<(HashMap<u8, char>, HashMap<char, u8>, f32)> {
    let mut col_to_label: HashMap<u8, char> = HashMap::new();
    let mut label_to_col: HashMap<char, u8> = HashMap::new();
    // まず対応を構築（占有かつラベル付きセルのみ）
    for x in 0..W {
        for y in 0..H {
            let lab = labels[y][x];
            if lab == '.' { continue; }
            let bit = 1u16 << y;
            let mut color_opt: Option<u8> = None;
            for c in 0..4 {
                if (board[c][x] & bit) != 0 { color_opt = Some(c as u8); break; }
            }
            if let Some(col) = color_opt {
                // 一貫性チェック
                if let Some(&l0) = col_to_label.get(&col) { if l0 != lab { return None; } }
                if let Some(&c0) = label_to_col.get(&lab) { if c0 != col { return None; } }
                col_to_label.insert(col, lab);
                label_to_col.insert(lab, col);
            }
        }
    }
    // 分母・分子の計算
    let mut denom: u32 = 0;
    let mut num: u32 = 0;
    for x in 0..W {
        for y in 0..H {
            let lab = labels[y][x];
            if lab == '.' { continue; }
            let bit = 1u16 << y;
            let mut color_opt: Option<u8> = None;
            for c in 0..4 {
                if (board[c][x] & bit) != 0 { color_opt = Some(c as u8); break; }
            }
            if let Some(col) = color_opt {
                denom += 1;
                if let Some(&mapped_col) = label_to_col.get(&lab) {
                    if mapped_col == col { num += 1; }
                }
            }
        }
    }
    if denom == 0 { return None; }
    let ratio = num as f32 / denom as f32;
    Some((col_to_label, label_to_col, ratio))
}

// 形への干渉を避ける手かどうかを判定
// ポリシー: '.' には自由に置いて良い。ラベル位置には、既に割当済みの色で置く場合のみ許容。
// ラベルにまだ割当が無い場合は、そのラベル位置へは置かない（干渉リスクを避ける）。
#[inline(always)]
fn connects_to_shape(board: &[[u16; W]; 4], labels: &Vec<Vec<char>>, x: usize, y: usize, color: u8) -> bool {
    let dirs: &[(isize, isize)] = &[(-1,0),(1,0),(0,-1),(0,1)];
    for (dx, dy) in dirs {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { continue; }
        let (nxu, nyu) = (nx as usize, ny as usize);
        if labels[nyu][nxu] == '.' { continue; }
        let bit = 1u16 << nyu;
        if (board[color as usize][nxu] & bit) != 0 { return true; }
    }
    false
}

// 未来にそのラベルへ同色が割り当てられ得るなら（＝将来的に直交隣接で連結し得るなら）true
#[inline(always)]
fn would_future_connect_to_shape(
    labels: &Vec<Vec<char>>,
    x: usize,
    y: usize,
    color: u8,
    col_to_label: &HashMap<u8, char>,
    label_to_col: &HashMap<char, u8>,
) -> bool {
    let dirs: &[(isize, isize)] = &[(-1,0),(1,0),(0,-1),(0,1)];
    for (dx, dy) in dirs {
        let nx = x as isize + dx;
        let ny = y as isize + dy;
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { continue; }
        let (nxu, nyu) = (nx as usize, ny as usize);
        let lab = labels[nyu][nxu];
        if lab == '.' { continue; }
        if let Some(&mc) = label_to_col.get(&lab) {
            if mc == color { return true; }
        } else {
            // ラベル未割当。color が他ラベルに固定されていなければ、このラベルに将来割り当てられ得る
            if let Some(&mlab) = col_to_label.get(&color) {
                if mlab == lab { return true; } // 既にこのラベルに紐付く色
            } else {
                // color も未使用 → このラベルへ割当可能性あり
                return true;
            }
        }
    }
    false
}

fn move_avoids_interference(
    board: &[[u16; W]; 4],
    pos_a: (usize, usize),
    pos_b: (usize, usize),
    pair: (u8, u8),
    labels: &Vec<Vec<char>>,
    col_to_label: &HashMap<u8, char>,
    label_to_col: &HashMap<char, u8>,
) -> bool {
    let (ax, ay) = pos_a; let (bx, by) = pos_b;
    let la = labels[ay][ax];
    let lb = labels[by][bx];
    // A 側: ラベル位置には置かない（完全に干渉を避ける）
    if la != '.' { return false; }
    // B 側
    if lb != '.' { return false; }
    // '.' へ置く場合でも、同色で JSONL ラベル領域に隣接（直交）して連結するのは避ける
    if connects_to_shape(board, labels, ax, ay, pair.0) { return false; }
    if connects_to_shape(board, labels, bx, by, pair.1) { return false; }
    // 将来、その隣接ラベルが同色になり得て連結してしまう置き方も避ける
    if would_future_connect_to_shape(labels, ax, ay, pair.0, col_to_label, label_to_col) { return false; }
    if would_future_connect_to_shape(labels, bx, by, pair.1, col_to_label, label_to_col) { return false; }
    true
}

// labels（表示用の1個除外適用済み）に対し、全ラベル位置へ既に正しい色が置かれているか
// すべて埋まっていれば true（＝ 形は「free-top 最上段以外がすでに埋まった」状態）
fn is_shape_fully_filled(board: &[[u16; W]; 4], labels: &Vec<Vec<char>>) -> bool {
    let mut col_to_label: HashMap<u8, char> = HashMap::new();
    let mut label_to_col: HashMap<char, u8> = HashMap::new();
    for x in 0..W {
        for y in 0..H {
            let lab = labels[y][x];
            if lab == '.' { continue; }
            let bit = 1u16 << y;
            let mut color_opt: Option<u8> = None;
            for c in 0..4 { if (board[c][x] & bit) != 0 { color_opt = Some(c as u8); break; } }
            let Some(col) = color_opt else { return false; };
            if !unify_color_label(col, lab, &mut col_to_label, &mut label_to_col) { return false; }
        }
    }
    true
}

// 単手の合法手を列挙（最大 22 通り）
fn cp_generate_moves_from_cols(cols: &[[u16; W]; 4]) -> Vec<(usize, Orient)> {
    let mut moves: Vec<(usize, Orient)> = Vec::new();
    for x in 0..W {
        let h = col_height(cols, x);
        if h + 1 < H {
            moves.push((x, Orient::Up));
            moves.push((x, Orient::Down));
        }
        if x + 1 < W {
            let h0 = h;
            let h1 = col_height(cols, x + 1);
            if h0 < H && h1 < H { moves.push((x, Orient::Right)); }
        }
        if x >= 1 {
            let h0 = h;
            let h1 = col_height(cols, x - 1);
            if h0 < H && h1 < H { moves.push((x, Orient::Left)); }
        }
    }
    moves
}

// 純関数版の配置適用（UI状態を変更しない）
fn cp_apply_move_pure(cols: &[[u16; W]; 4], x: usize, orient: Orient, pair: (u8, u8)) -> [[u16; W]; 4] {
    let mut out = *cols;
    match orient {
        Orient::Up => {
            let h = col_height(&out, x);
            if h < H { out[pair.0 as usize][x] |= 1u16 << h; }
            if h + 1 < H { out[pair.1 as usize][x] |= 1u16 << (h + 1); }
        }
        Orient::Down => {
            let h = col_height(&out, x);
            if h < H { out[pair.1 as usize][x] |= 1u16 << h; }
            if h + 1 < H { out[pair.0 as usize][x] |= 1u16 << (h + 1); }
        }
        Orient::Right => {
            let h0 = col_height(&out, x);
            let h1 = if x + 1 < W { col_height(&out, x + 1) } else { H };
            if h0 < H { out[pair.0 as usize][x] |= 1u16 << h0; }
            if x + 1 < W && h1 < H { out[pair.1 as usize][x + 1] |= 1u16 << h1; }
        }
        Orient::Left => {
            let h0 = col_height(&out, x);
            let h1 = if x >= 1 { col_height(&out, x - 1) } else { H };
            if h0 < H { out[pair.0 as usize][x] |= 1u16 << h0; }
            if x >= 1 && h1 < H { out[pair.1 as usize][x - 1] |= 1u16 << h1; }
        }
    }
    out
}

// 深さ優先探索（depth_left 手以内のどこかで ちょうど target 連鎖を達成）
fn dfs_exact_equal(
    cols: [[u16; W]; 4],
    pair_seq: &[(u8, u8)],
    idx: usize,
    depth_left: usize,
    target: u32,
    abort: &Arc<AtomicBool>,
) -> Option<(usize, Orient)> {
    if abort.load(Ordering::Relaxed) { return None; }
    let pair_len = pair_seq.len().max(1);
    let pair = pair_seq[idx % pair_len];

    // 候補手を列挙
    let moves = cp_generate_moves_from_cols(&cols);
    // まず 1 手見たときの評価を計算
    let mut evals: Vec<((usize, Orient), u32, [[u16; W]; 4])> = Vec::with_capacity(moves.len());
    for &(x, orient) in &moves {
        if abort.load(Ordering::Relaxed) { return None; }
        let after = cp_apply_move_pure(&cols, x, orient, pair);
        let (gain, leftover) = simulate_chain_and_final(after);
        evals.push(((x, orient), gain, leftover));
    }
    if depth_left == 1 {
        if let Some(((x, orient), _g, _)) = evals.iter().find(|(_, g, _)| *g == target) {
            return Some((*x, *orient));
        }
        return None;
    }
    // 深掘り（ヒューリスティクス: 大きい g を先に）
    evals.sort_by(|a, b| b.1.cmp(&a.1));
    for ((x, orient), g, leftover) in evals {
        if abort.load(Ordering::Relaxed) { return None; }
        if g == target { return Some((x, orient)); }
        if let Some((_fx, _fo)) = dfs_exact_equal(leftover, pair_seq, idx + 1, depth_left - 1, target, abort) {
            // ルート手として現在の (x, orient) を返す
            return Some((x, orient));
        }
    }
    None
}

// ちょうど target 連鎖が depth_left 手以内のどこかで到達可能か（根手は既に打たれている想定）
fn dfs_exact_reachable(
    cols: [[u16; W]; 4],
    pair_seq: &[(u8, u8)],
    idx: usize,
    depth_left: usize,
    target: u32,
    abort: &Arc<AtomicBool>,
) -> bool {
    if abort.load(Ordering::Relaxed) { return false; }
    let pair_len = pair_seq.len().max(1);
    let pair = pair_seq[idx % pair_len];

    let moves = cp_generate_moves_from_cols(&cols);
    // まず 1 手見たときの評価を計算
    let mut evals: Vec<((usize, Orient), u32, [[u16; W]; 4])> = Vec::with_capacity(moves.len());
    for &(x, orient) in &moves {
        if abort.load(Ordering::Relaxed) { return false; }
        let after = cp_apply_move_pure(&cols, x, orient, pair);
        let (gain, leftover) = simulate_chain_and_final(after);
        if gain == target { return true; }
        evals.push(((x, orient), gain, leftover));
    }
    if depth_left <= 1 { return false; }
    // 深掘り（ヒューリスティクス: 大きい g を先に）
    evals.sort_by(|a, b| b.1.cmp(&a.1));
    for ((_x, _orient), _g, leftover) in evals {
        if abort.load(Ordering::Relaxed) { return false; }
        if dfs_exact_reachable(leftover, pair_seq, idx + 1, depth_left - 1, target, abort) {
            return true;
        }
    }
    false
}


// ===== App ユーティリティ =====
impl App {
    // 新規作成（設定の自動読込を行う）
    fn new() -> Self {
        let mut app = App::default();
        let _ = app.cp_load_settings();
        app
    }

    // 設定ファイルパス（実行ディレクトリ直下）
    fn cp_settings_path() -> std::path::PathBuf {
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join("cp_settings.json")
    }

    // モジュール並びと設定値の保存
    fn cp_save_settings(&mut self) -> Result<()> {
        let settings = CpSettings { constraints: self.cp.constraints.clone(), tie_break: self.cp.tie_break.clone() };
        let path = Self::cp_settings_path();
        let f = File::create(&path).with_context(|| format!("設定ファイルを作成できません: {}", path.display()))?;
        let mut w = BufWriter::new(f);
        serde_json::to_writer_pretty(&mut w, &settings).with_context(|| "設定のシリアライズに失敗")?;
        w.flush().ok();
        self.push_log(format!("設定を保存しました: {}", path.display()));
        Ok(())
    }

    // モジュール並びと設定値の読込
    fn cp_load_settings(&mut self) -> Result<()> {
        let path = Self::cp_settings_path();
        if !path.exists() { return Ok(()); }
        let f = File::open(&path).with_context(|| format!("設定ファイルを開けません: {}", path.display()))?;
        let r = BufReader::new(f);
        let loaded: CpSettings = serde_json::from_reader(r).with_context(|| "設定のデシリアライズに失敗")?;
        self.cp.constraints = loaded.constraints;
        self.cp.tie_break = loaded.tie_break;
        self.push_log(format!("設定を読み込みました: {}", path.display()));
        Ok(())
    }
    fn push_log(&mut self, s: String) {
        self.log_lines.push(s);
        if self.log_lines.len() > 500 {
            let cut = self.log_lines.len() - 500;
            self.log_lines.drain(0..cut);
        }
    }

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
        let require_free_top_e1 = self.require_free_top_e1;
        let profile_enabled = self.profile_enabled;

        self.push_log(format!(
            "出力: JSONL / 形キャッシュ上限 ≈ {} 形 / 保存先: {} / 進捗停滞比={:.2} / 4個消しモード={} / free-top E1限定={} / 計測={}",
            lru_limit,
            outfile.display(),
            stop_progress_plateau,
            if exact_four_only { "ON" } else { "OFF" },
            if require_free_top_e1 { "ON" } else { "OFF" },
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
                require_free_top_e1,
                profile_enabled,
            ) {
                let _ = tx.send(Message::Error(format!("{e:?}")));
            }
        });
    }
}

fn has_profile_any(p: &ProfileTotals) -> bool {
    if p.io_write_total != Duration::ZERO { return true; }
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
        { return true; }
        let c = p.dfs_counts[i];
        if c.nodes != 0 || c.cand_generated != 0 || c.pruned_upper != 0
            || c.leaves != 0
            || c.leaf_pre_tshort != 0
            || c.leaf_pre_e1_impossible != 0
            || c.memo_lhit != 0 || c.memo_ghit != 0 || c.memo_miss != 0 { return true; }
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
            ui.monospace(format!("{} / {} / {}",
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
}

impl Cell {
    fn label_char(self) -> char {
        match self {
            Cell::Blank => '.',
            Cell::Any => 'N',
            Cell::Any4 => 'X',
            Cell::Abs(i) => (b'A' + i) as char,
            Cell::Fixed(c) => (b'0' + c) as char,
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
fn cycle_abs(c: Cell) -> Cell {
    match c {
        Cell::Blank | Cell::Any | Cell::Any4 => Cell::Abs(0),
        Cell::Abs(i) => Cell::Abs(((i as usize + 1) % 13) as u8),
        Cell::Fixed(_) => Cell::Abs(0),
    }
}
fn cycle_any(c: Cell) -> Cell {
    match c {
        Cell::Any => Cell::Any4,
        Cell::Any4 => Cell::Any,
        _ => Cell::Any,
    }
}
fn cycle_fixed(c: Cell) -> Cell {
    match c {
        Cell::Fixed(v) => Cell::Fixed(((v as usize + 1) % 4) as u8),
        _ => Cell::Fixed(0),
    }
}

fn draw_preview(ui: &mut egui::Ui, cols: &[[u16; W]; 4]) {
    let cell = 16.0_f32; // 1マスのサイズ
    let gap = 1.0_f32; // マス間の隙間

    let width = W as f32 * cell + (W - 1) as f32 * gap;
    let height = H as f32 * cell + (H - 1) as f32 * gap;

    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let painter = ui.painter_at(rect);

    // 0=R, 1=G, 2=B, 3=Y
    let palette = [
        Color32::from_rgb(239, 68, 68),   // red
        Color32::from_rgb(34, 197, 94),   // green
        Color32::from_rgb(59, 130, 246),  // blue
        Color32::from_rgb(234, 179, 8),   // yellow
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
            if color[v] != 4 { continue; }
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
        if vleft == total_n { new_color_limit = 0; } // 最初の1手は 0 のみ
        for c in 0u8..=new_color_limit {
            if ((forbid >> c) & 1) != 0 { continue; }
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
            dfs(vleft - 1, total_n, adj, color, used_mask, out, next_max_used);

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
    rec(0, false, col, &mut masks, enum_time, &mut last_start, &mut yield_masks);
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
            unsafe { return fall_cols_bmi2(cols_in); }
        }
    }
    fall_cols(cols_in)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn fall_cols_bmi2(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_pdep_u32, _pext_u32};
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_pdep_u32, _pext_u32};

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

// ===== 全消しプレビュー用ヘルパー =====
#[inline(always)]
fn cols_is_empty(cols: &[[u16; W]; 4]) -> bool {
    for c in 0..4 {
        for x in 0..W {
            if cols[c][x] != 0 { return false; }
        }
    }
    true
}

#[inline(always)]
fn col_height(cols: &[[u16; W]; 4], x: usize) -> usize {
    let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
    occ.count_ones() as usize
}

// ===== 連結サイズ計測と配置位置算出のヘルパー =====
#[inline(always)]
fn comp_size_for_color_at(cols: &[[u16; W]; 4], color: u8, x: usize, y: usize) -> u32 {
    if x >= W || y >= H { return 0; }
    let bb = pack_cols(cols);
    let mask = bb[color as usize];
    let seed: BB = 1u128 << (x * COL_BITS + y);
    if (mask & seed) == 0 { return 0; }
    let mut comp = seed;
    loop {
        let grow = neighbors(comp) & mask & !comp;
        if grow == 0 { break; }
        comp |= grow;
    }
    comp.count_ones() as u32
}

// pair=(axis,child) の配置先座標（置く直前の cols に対して）
#[inline(always)]
fn cp_positions_for_move(cols: &[[u16; W]; 4], x: usize, orient: Orient) -> Option<((usize, usize), (usize, usize))> {
    match orient {
        Orient::Up => {
            let h = col_height(cols, x);
            if h + 1 >= H { return None; }
            Some(((x, h), (x, h + 1)))
        }
        Orient::Down => {
            let h = col_height(cols, x);
            if h + 1 >= H { return None; }
            Some(((x, h + 1), (x, h)))
        }
        Orient::Right => {
            if x + 1 >= W { return None; }
            let h0 = col_height(cols, x);
            let h1 = col_height(cols, x + 1);
            if h0 >= H || h1 >= H { return None; }
            Some(((x, h0), (x + 1, h1)))
        }
        Orient::Left => {
            if x == 0 { return None; }
            let h0 = col_height(cols, x);
            let h1 = col_height(cols, x - 1);
            if h0 >= H || h1 >= H { return None; }
            Some(((x, h0), (x - 1, h1)))
        }
    }
}

#[inline(always)]
fn simulate_chain_count_simple(mut cols: [[u16; W]; 4]) -> u32 {
    let mut cc: u32 = 0;
    loop {
        let (cleared, next) = apply_erase_and_fall_cols(&cols);
        if !cleared { break; }
        cols = next;
        cc += 1;
    }
    cc
}

// ── タイブレーク：設置先高さの評価 ─────────────────────────────────────
#[inline(always)]
fn placement_height_metric(board: &[[u16; W]; 4], x: usize, orient: Orient) -> usize {
    match orient {
        Orient::Up | Orient::Down => col_height(board, x),
        Orient::Right => {
            let h0 = col_height(board, x);
            let h1 = if x + 1 < W { col_height(board, x + 1) } else { H };
            h0.min(h1)
        }
        Orient::Left => {
            let h0 = col_height(board, x);
            let h1 = if x >= 1 { col_height(board, x - 1) } else { H };
            h0.min(h1)
        }
    }
}

#[inline(always)]
fn orient_is_vertical(o: Orient) -> bool { matches!(o, Orient::Up | Orient::Down) }

#[inline(always)]
fn tie_break_choose(candidate: (usize, Orient), current: (usize, Orient), board: &[[u16; W]; 4], policy: &TieBreakPolicy) -> bool {
    let (cx, co) = current;
    let (nx, no) = candidate;
    for &rule in &policy.rules {
        match rule {
            TieBreakRule::MinPlacementHeight => {
                let ch = placement_height_metric(board, cx, co);
                let nh = placement_height_metric(board, nx, no);
                if nh < ch { return true; }
                if nh > ch { return false; }
            }
            TieBreakRule::LeftmostColumn => {
                if nx < cx { return true; }
                if nx > cx { return false; }
            }
            TieBreakRule::PreferVertical => {
                let cv = orient_is_vertical(co);
                let nv = orient_is_vertical(no);
                if nv && !cv { return true; }
                if !nv && cv { return false; }
            }
            // ペア情報なしの評価関数では無効化（no-op）。
            // ペア情報ありの tie_break_choose_with_pair を使用してください。
            TieBreakRule::PreferDiagonalSameColor => {}
            // 新規ルール（ペア情報なしでは評価できないため no-op）
            TieBreakRule::PreferMake3GroupAvoidDiagDown23 => {}
            TieBreakRule::PreferMake2GroupAvoidDiagDown23 => {}
        }
    }
    false
}

// ── タイブレーク（ペア情報あり）：同色の斜め隣接優先を評価可能にする ─────────────
#[inline(always)]
fn tie_break_choose_with_pair(
    candidate: (usize, Orient),
    current: (usize, Orient),
    board: &[[u16; W]; 4],
    pair: (u8, u8),
    policy: &TieBreakPolicy,
) -> bool {
    let (cx, co) = current;
    let (nx, no) = candidate;
    for &rule in &policy.rules {
        match rule {
            TieBreakRule::MinPlacementHeight => {
                let ch = placement_height_metric(board, cx, co);
                let nh = placement_height_metric(board, nx, no);
                if nh < ch { return true; }
                if nh > ch { return false; }
            }
            TieBreakRule::LeftmostColumn => {
                if nx < cx { return true; }
                if nx > cx { return false; }
            }
            TieBreakRule::PreferVertical => {
                let cv = orient_is_vertical(co);
                let nv = orient_is_vertical(no);
                if nv && !cv { return true; }
                if !nv && cv { return false; }
            }
            TieBreakRule::PreferDiagonalSameColor => {
                let cv = diagonal_same_color_for_move(board, cx, co, pair);
                let nv = diagonal_same_color_for_move(board, nx, no, pair);
                if nv && !cv { return true; }
                if !nv && cv { return false; }
            }
            TieBreakRule::PreferMake3GroupAvoidDiagDown23 => {
                let mut cand3 = false;
                if let Some((cpos_a, cpos_b)) = cp_positions_for_move(board, nx, no) {
                    // 斜め下 2/3 連結回避（既存ぷよへの接続を避ける）
                    if !avoid_connect_if_neighbor_has_diag_down_23(board, pair.0, cpos_a)
                        && !avoid_connect_if_neighbor_has_diag_down_23(board, pair.1, cpos_b)
                    {
                        let after = cp_apply_move_pure(board, nx, no, pair);
                        let sa = comp_size_for_color_at(&after, pair.0, cpos_a.0, cpos_a.1);
                        let sb = comp_size_for_color_at(&after, pair.1, cpos_b.0, cpos_b.1);
                        cand3 = sa == 3 || sb == 3;
                    }
                }
                let mut curr3 = false;
                if let Some((ppos_a, ppos_b)) = cp_positions_for_move(board, cx, co) {
                    if !avoid_connect_if_neighbor_has_diag_down_23(board, pair.0, ppos_a)
                        && !avoid_connect_if_neighbor_has_diag_down_23(board, pair.1, ppos_b)
                    {
                        let after = cp_apply_move_pure(board, cx, co, pair);
                        let sa = comp_size_for_color_at(&after, pair.0, ppos_a.0, ppos_a.1);
                        let sb = comp_size_for_color_at(&after, pair.1, ppos_b.0, ppos_b.1);
                        curr3 = sa == 3 || sb == 3;
                    }
                }
                if cand3 && !curr3 { return true; }
                if !cand3 && curr3 { return false; }
            }
            TieBreakRule::PreferMake2GroupAvoidDiagDown23 => {
                let mut cand2 = false;
                if let Some((cpos_a, cpos_b)) = cp_positions_for_move(board, nx, no) {
                    if !avoid_connect_if_neighbor_has_diag_down_23(board, pair.0, cpos_a)
                        && !avoid_connect_if_neighbor_has_diag_down_23(board, pair.1, cpos_b)
                    {
                        let after = cp_apply_move_pure(board, nx, no, pair);
                        let sa = comp_size_for_color_at(&after, pair.0, cpos_a.0, cpos_a.1);
                        let sb = comp_size_for_color_at(&after, pair.1, cpos_b.0, cpos_b.1);
                        cand2 = sa == 2 || sb == 2;
                    }
                }
                let mut curr2 = false;
                if let Some((ppos_a, ppos_b)) = cp_positions_for_move(board, cx, co) {
                    if !avoid_connect_if_neighbor_has_diag_down_23(board, pair.0, ppos_a)
                        && !avoid_connect_if_neighbor_has_diag_down_23(board, pair.1, ppos_b)
                    {
                        let after = cp_apply_move_pure(board, cx, co, pair);
                        let sa = comp_size_for_color_at(&after, pair.0, ppos_a.0, ppos_a.1);
                        let sb = comp_size_for_color_at(&after, pair.1, ppos_b.0, ppos_b.1);
                        curr2 = sa == 2 || sb == 2;
                    }
                }
                if cand2 && !curr2 { return true; }
                if !cand2 && curr2 { return false; }
            }
        }
    }
    false
}

#[inline(always)]
fn tie_break_cmp_with_pair(
    a: (usize, Orient),
    b: (usize, Orient),
    board: &[[u16; W]; 4],
    pair: (u8, u8),
    policy: &TieBreakPolicy,
) -> std::cmp::Ordering {
    if !policy.apply_initial { return std::cmp::Ordering::Equal; }
    let ab = tie_break_choose_with_pair(a, b, board, pair, policy);
    let ba = tie_break_choose_with_pair(b, a, board, pair, policy);
    if ab && !ba { std::cmp::Ordering::Less }
    else if ba && !ab { std::cmp::Ordering::Greater }
    else { std::cmp::Ordering::Equal }
}

// 共通ヘルパー（ペアあり）: 候補集合からタイブレーク規則に従って最良手を1つ選ぶ
#[inline(always)]
fn select_best_move_with_pair(
    cands: &[(usize, Orient)],
    board: &[[u16; W]; 4],
    pair: (u8, u8),
    policy: &TieBreakPolicy,
) -> Option<(usize, Orient)> {
    if cands.is_empty() { return None; }
    let mut best = cands[0];
    for &mv in &cands[1..] {
        if tie_break_cmp_with_pair(mv, best, board, pair, policy) == std::cmp::Ordering::Less {
            best = mv;
        }
    }
    Some(best)
}

// 共通ヘルパー（ペアあり）: スコア最大を優先し、同点ならタイブレーク
#[inline(always)]
fn select_best_scored_i32_with_pair(
    cands: &[((usize, Orient), i32)],
    board: &[[u16; W]; 4],
    pair: (u8, u8),
    policy: &TieBreakPolicy,
) -> Option<(usize, Orient)> {
    if cands.is_empty() { return None; }
    let mut best = cands[0];
    for &item in &cands[1..] {
        if item.1 > best.1 {
            best = item;
        } else if item.1 == best.1 {
            if tie_break_cmp_with_pair(item.0, best.0, board, pair, policy) == std::cmp::Ordering::Less {
                best = item;
            }
        }
    }
    Some(best.0)
}

// 先読み手数（ペア）で盤面を拡張し、その後ビーム（単ぷよ追加）で評価
#[inline(always)]
fn cp_evaluate_with_pair_lookahead(
    after_first: [[u16; W]; 4],
    pair_seq: &[(u8, u8)],
    mut next_pair_idx: usize,
    lookahead_pairs: usize,
    beam_depth: usize,
    beam_width: usize,
    abort: &Arc<AtomicBool>,
    first_pair_positions: Option<((usize, usize), (usize, usize))>,
) -> (u32, Option<[[u16; W]; 4]>, Option<u32>) {
    if abort.load(Ordering::Relaxed) {
        return (simulate_chain_count_simple(after_first), None, None);
    }

    // ペア先読み：ビーム幅は beam_width を流用
    #[derive(Clone)]
    struct PairNode { board: [[u16; W]; 4], idx: usize }
    let mut frontier: Vec<PairNode> = vec![PairNode { board: after_first, idx: next_pair_idx }];

    let pair_len = pair_seq.len().max(1);
    for _d in 0..lookahead_pairs {
        if abort.load(Ordering::Relaxed) { break; }
        let mut candidates: Vec<(i32, PairNode)> = Vec::new();
        for node in &frontier {
            if abort.load(Ordering::Relaxed) { break; }
            let pair = pair_seq[node.idx % pair_len];
            let moves = cp_generate_moves_from_cols(&node.board);
            for (x, orient) in moves {
                if abort.load(Ordering::Relaxed) { break; }
                let b2 = cp_apply_move_pure(&node.board, x, orient, pair);
                let h = simulate_chain_count_simple(b2) as i32; // 簡易ヒューリスティック
                candidates.push((h, PairNode { board: b2, idx: node.idx + 1 }));
            }
        }
        if candidates.is_empty() { break; }
        // 上位 beam_width のみ残す
        candidates.sort_by(|a, b| b.0.cmp(&a.0));
        let keep = candidates.into_iter().take(beam_width.max(1)).map(|(_, n)| n).collect::<Vec<_>>();
        frontier = keep;
    }

    // 葉でビーム（単ぷよ追加）評価
    let mut best_score: i32 = -1;
    let mut best_prev: Option<[[u16; W]; 4]> = None;
    let mut best_chain: Option<u32> = None;
    for node in &frontier {
        if abort.load(Ordering::Relaxed) { break; }
        if let Some((adds, total_chain)) = cp_iterative_chain_clearing_beam(node.board, beam_depth, beam_width, first_pair_positions) {
            let merged = merge_additions_onto(node.board, &adds);
            if (total_chain as i32) > best_score {
                best_score = total_chain as i32;
                best_prev = Some(merged);
                best_chain = Some(total_chain);
            } else if (total_chain as i32) == best_score {
                // tie-break: 盤面の占有が少ない方
                let cur_occ = board_occupied_count(best_prev.as_ref().unwrap_or(&node.board));
                let new_occ = board_occupied_count(&merged);
                if new_occ < cur_occ {
                    best_prev = Some(merged);
                    best_chain = Some(total_chain);
                }
            }
        } else {
            let sc = simulate_chain_count_simple(node.board) as i32;
            if sc > best_score { best_score = sc; best_prev = None; best_chain = None; }
        }
    }

    if best_score < 0 {
        (simulate_chain_count_simple(after_first), None, None)
    } else {
        (best_score as u32, best_prev, best_chain)
    }
}

#[inline(always)]
fn simulate_chain_and_final(mut cols: [[u16; W]; 4]) -> (u32, [[u16; W]; 4]) {
    let mut cc: u32 = 0;
    loop {
        let (cleared, next) = apply_erase_and_fall_cols(&cols);
        if !cleared { break; }
        cols = next;
        cc += 1;
    }
    (cc, cols)
}

// 1連鎖目で消えるグループ数（全色合計）
#[inline(always)]
fn count_first_clear_groups(cols: &[[u16; W]; 4]) -> u32 {
    let bb = pack_cols(cols);
    let mut groups: u32 = 0;
    for &mask in bb.iter() {
        if mask.count_ones() < 4 { continue; }
        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 { break; }
                comp |= grow;
                frontier = grow;
            }
            if comp.count_ones() >= 4 { groups += 1; }
            s &= !comp;
        }
    }
    groups
}

// 座標 (x,y) の近傍（上下左右）に存在する色を列挙
fn neighbor_colors_at(cols: &[[u16; W]; 4], x: usize, y: usize) -> [bool; 4] {
    let mut used = [false; 4];
    let mut check = |nx: isize, ny: isize| {
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { return; }
        let nxu = nx as usize; let nyu = ny as usize;
        let bit = 1u16 << nyu;
        for c in 0..4 {
            if (cols[c][nxu] & bit) != 0 { used[c] = true; }
        }
    };
    check(x as isize - 1, y as isize);
    check(x as isize + 1, y as isize);
    check(x as isize, y as isize - 1);
    check(x as isize, y as isize + 1);
    used
}

// 斜め方向（4近傍の対角）に同色が存在するか
#[inline(always)]
fn has_diagonal_same_color(cols: &[[u16; W]; 4], color: u8, x: usize, y: usize) -> bool {
    let cidx = color as usize;
    let mut check = |nx: isize, ny: isize| -> bool {
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { return false; }
        let nxu = nx as usize; let nyu = ny as usize;
        let bit = 1u16 << nyu;
        (cols[cidx][nxu] & bit) != 0
    };
    check(x as isize - 1, y as isize - 1)
        || check(x as isize + 1, y as isize - 1)
        || check(x as isize - 1, y as isize + 1)
        || check(x as isize + 1, y as isize + 1)
}

// 与えられた手 (x,orient) を現在盤面に適用したとき、ペアのどちらかが同色の斜め隣接を持つか（判定は現盤面基準）
#[inline(always)]
fn diagonal_same_color_for_move(board: &[[u16; W]; 4], x: usize, orient: Orient, pair: (u8, u8)) -> bool {
    if let Some(((ax, ay), (bx, by))) = cp_positions_for_move(board, x, orient) {
        has_diagonal_same_color(board, pair.0, ax, ay) || has_diagonal_same_color(board, pair.1, bx, by)
    } else { false }
}

// 与えた座標に同色が存在するか
#[inline(always)]
fn has_same_color_at(cols: &[[u16; W]; 4], color: u8, x: usize, y: usize) -> bool {
    if x >= W || y >= H { return false; }
    let bit = 1u16 << y;
    (cols[color as usize][x] & bit) != 0
}

// 置こうとしている位置 (px,py) と同色の既存ぷよ（上下左右のどれか）に連結する場合、
// その既存ぷよの「斜め下（左下 or 右下）」に同色の2連結または3連結が存在するなら true
#[inline(always)]
fn avoid_connect_if_neighbor_has_diag_down_23(cols: &[[u16; W]; 4], color: u8, pos: (usize, usize)) -> bool {
    let (px, py) = pos;
    // 既存同色との連結先を列挙（現盤面）
    let mut neighs: [(isize, isize); 4] = [
        (px as isize - 1, py as isize),
        (px as isize + 1, py as isize),
        (px as isize, py as isize - 1),
        (px as isize, py as isize + 1),
    ];
    for &(nx, ny) in &neighs {
        if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { continue; }
        let nxu = nx as usize; let nyu = ny as usize;
        if !has_same_color_at(cols, color, nxu, nyu) { continue; }
        // 既存ぷよ (nxu,nyu) の斜め下2方向
        let diag = [ (nx - 1, ny - 1), (nx + 1, ny - 1) ];
        for &(dx, dy) in &diag {
            if dx < 0 || dx >= W as isize || dy < 0 || dy >= H as isize { continue; }
            let dxu = dx as usize; let dyu = dy as usize;
            if !has_same_color_at(cols, color, dxu, dyu) { continue; }
            let sz = comp_size_for_color_at(cols, color, dxu, dyu);
            if sz == 2 || sz == 3 { return true; }
        }
    }
    false
}

// 指定した色レイヤ s の seeds（座標群）から連結成分をユニオン
fn union_components_from_seeds(s: &[u16; W], seeds: &[(usize, usize)]) -> [u16; W] {
    let mut acc = [0u16; W];
    for &(sx, sy) in seeds {
        let bit = 1u16 << sy;
        if (s[sx] & bit) == 0 { continue; }
        let comp = component_from_seed_cols(s, sx, bit);
        for x in 0..W { acc[x] |= comp[x]; }
    }
    acc
}

#[inline(always)]
fn is_adjacent_to_group(group: &[u16; W], x: usize, y: usize) -> bool {
    let bit = 1u16 << y;
    if x > 0 && (group[x - 1] & bit) != 0 { return true; }
    if x + 1 < W && (group[x + 1] & bit) != 0 { return true; }
    if y > 0 && (group[x] & (bit >> 1)) != 0 { return true; }
    if y + 1 < H && (group[x] & (bit << 1)) != 0 { return true; }
    false
}

fn allowed_cols_from_group(group: &[u16; W]) -> [bool; W] {
    let mut allow = [false; W];
    for x in 0..W {
        if group[x] != 0 {
            allow[x] = true;
            if x > 0 { allow[x - 1] = true; }
            if x + 1 < W { allow[x + 1] = true; }
        }
    }
    allow
}

// 追加分を元盤面へ順番に積む（列ごとに高さを進める）
fn merge_additions_onto(original: [[u16; W]; 4], adds: &Vec<(usize, u8)>) -> [[u16; W]; 4] {
    let mut out = original;
    let mut heights = [0usize; W];
    for x in 0..W { heights[x] = col_height(&out, x); }
    for &(x, c) in adds {
        if x >= W { continue; }
        if heights[x] >= H { continue; }
        out[c as usize][x] |= 1u16 << heights[x];
        heights[x] += 1;
    }
    out
}

// 1手ぶんの最良拡張（leftover を基準）。
// 返り値: (この手での連鎖数, 追加後の盤面, この手で新規追加した (col,color) 群)
fn find_best_single_step_extension(
    leftover: &[[u16; W]; 4],
    original: &[[u16; W]; 4],
    prev_additions: &Vec<(usize, u8)>,
    blocked_cols_opt: Option<&[bool; W]>,
) -> Option<(u32, [[u16; W]; 4], Vec<(usize, u8)>)> {
    let blocked = blocked_cols_opt.unwrap_or(&[false; W]);

    let mut best_chain: i32 = -1;
    let mut best_after: Option<[[u16; W]; 4]> = None;
    let mut best_adds: Vec<(usize, u8)> = Vec::new();

    for x in 0..W {
        let y = col_height(leftover, x);
        if y >= H { continue; }

        let used = neighbor_colors_at(leftover, x, y);
        for color in 0u8..4u8 {
            if !used[color as usize] { continue; }

            // 近傍同色を seeds に集め、成分ユニオン（puyoA）
            let mut seeds: Vec<(usize, usize)> = Vec::new();
            let mut push_if = |nx: isize, ny: isize| {
                if nx < 0 || nx >= W as isize || ny < 0 || ny >= H as isize { return; }
                let nxu = nx as usize; let nyu = ny as usize;
                let bit = 1u16 << nyu;
                if (leftover[color as usize][nxu] & bit) != 0 { seeds.push((nxu, nyu)); }
            };
            push_if(x as isize - 1, y as isize);
            push_if(x as isize + 1, y as isize);
            push_if(x as isize, y as isize - 1);
            push_if(x as isize, y as isize + 1);

            let puyo_a = union_components_from_seeds(&leftover[color as usize], &seeds);
            let mut groups: Vec<[u16; W]> = Vec::new();
            // 重複除去しつつ近傍の各成分を収集
            for &(sx, sy) in &seeds {
                let comp = component_from_seed_cols(&leftover[color as usize], sx, 1u16 << sy);
                if comp.iter().all(|&m| m == 0) { continue; }
                let mut dup = false;
                for g in &groups {
                    let mut inter = false;
                    for ix in 0..W { if (g[ix] & comp[ix]) != 0 { inter = true; break; } }
                    if inter { dup = true; break; }
                }
                if !dup { groups.push(comp); }
            }
            let mut total_adj: u32 = 0;
            for g in &groups { total_adj += g.iter().map(|&m| m.count_ones()).sum::<u32>(); }
            let effective_adj = total_adj.min(3);
            let needed = 4u32.saturating_sub(effective_adj) as usize; // 0..=3

            let mut allow = allowed_cols_from_group(&puyo_a);
            let mut temp = *leftover;
            let mut additions_step: Vec<(usize, u8)> = Vec::new();
            let mut adjacency = puyo_a.clone();

            let mut placed: usize = 0;
            for _ in 0..needed {
                let mut cand_best_chain: i32 = -1;
                let mut cand_best_cols: Option<[[u16; W]; 4]> = None;
                let mut cand_best_col: Option<usize> = None;

                for col in 0..W {
                    if !allow[col] || blocked[col] { continue; }
                    let h = col_height(&temp, col);
                    if h >= H { continue; }
                    if !is_adjacent_to_group(&adjacency, col, h) { continue; }

                    let mut t2 = temp;
                    t2[color as usize][col] |= 1u16 << h;

                    // マージ後の初手同時消し（>1 グループ）にならないかをチェック
                    let mut merged_adds = prev_additions.clone();
                    merged_adds.extend_from_slice(&additions_step);
                    merged_adds.push((col, color));
                    let merged_board = merge_additions_onto(*original, &merged_adds);
                    let fg = count_first_clear_groups(&merged_board);

                    let cc_tmp = simulate_chain_count_simple(t2);
                    if cc_tmp >= 1 && fg > 1 { continue; }

                    if (cc_tmp as i32) > cand_best_chain {
                        cand_best_chain = cc_tmp as i32;
                        cand_best_cols = Some(t2);
                        cand_best_col = Some(col);
                    }
                }

                let Some(col) = cand_best_col else { break; };
                let t2 = cand_best_cols.unwrap();
                temp = t2;
                let h = col_height(&temp, col).saturating_sub(1);
                if col < W { additions_step.push((col, color)); }

                // 隣接集合/許容列の更新
                if col > 0 { allow[col - 1] = true; }
                allow[col] = true;
                if col + 1 < W { allow[col + 1] = true; }
                adjacency[col] |= 1u16 << h;
                placed += 1;
            }

            if placed == needed {
                let cc = simulate_chain_count_simple(temp);
                if (cc as i32) > best_chain {
                    best_chain = cc as i32;
                    best_after = Some(temp);
                    best_adds = additions_step;
                }
            }
        }
    }

    if best_chain < 0 { None } else { Some((best_chain as u32, best_after.unwrap(), best_adds)) }
}

// ====== ビームサーチ型：iterative_chain_clearing（python版相当の移植） ======
#[derive(Clone)]
struct BeamNode {
    leftover: [[u16; W]; 4],
    additions: Vec<(usize, u8)>,
    total_chain: u32,
    first_chain_locked: bool,
}

#[inline(always)]
fn board_occupied_count(cols: &[[u16; W]; 4]) -> u32 {
    let mut s: u32 = 0;
    for c in 0..4 {
        for x in 0..W { s = s.saturating_add(cols[c][x].count_ones()); }
    }
    s
}

// 初手の最大塊サイズ（グループ数>0のときの最大 comp サイズ）
fn first_clear_largest_size(cols: &[[u16; W]; 4]) -> u32 {
    let bb = pack_cols(cols);
    let mut best: u32 = 0;
    for &mask in bb.iter() {
        if mask.count_ones() < 4 { continue; }
        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 { break; }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 { best = best.max(sz as u32); }
            s &= !comp;
        }
    }
    best
}

// 1連鎖目で消える塊のうち、少なくとも1列は「その列の上に何も乗っていない（列の最上段が塊に含まれる）」を満たすか
fn first_clear_has_free_top(cols: &[[u16; W]; 4]) -> bool {
    let clear = compute_erase_mask_cols(cols);
    // 1つも消えないなら満たせない
    if (0..W).all(|x| clear[x] == 0) { return false; }
    for x in 0..W {
        let m = clear[x] & MASK14;
        if m == 0 { continue; }
        // この列の塊の最上段 y を求める
        let mut top_y_opt = None;
        for y in (0..H).rev() {
            if (m & (1u16 << y)) != 0 { top_y_opt = Some(y); break; }
        }
        if let Some(top_y) = top_y_opt {
            let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
            let above_mask: u16 = (!((1u16 << (top_y + 1)) - 1)) & MASK14;
            if (occ & above_mask) == 0 {
                return true;
            }
        }
    }
    false
}

fn expand_beam_node(
    original: &[[u16; W]; 4],
    node: &BeamNode,
    first_pair_positions: Option<((usize, usize), (usize, usize))>,
) -> Vec<BeamNode> {
    let mut out: Vec<BeamNode> = Vec::new();
    // 1個追加の全候補（隣接色のみ）
    for x in 0..W {
        let y = col_height(&node.leftover, x);
        if y >= H { continue; }
        let used = neighbor_colors_at(&node.leftover, x, y);
        // 隣接色がなければスキップ（ヒューリスティック）
        if !used.iter().any(|&u| u) { continue; }
        for color in 0u8..4u8 {
            if !used[color as usize] { continue; }

            let mut after = node.leftover;
            after[color as usize][x] |= 1u16 << y;

            let (gain, next_leftover) = simulate_chain_and_final(after);

            // マージ側（初期盤面 + 全追加）での 1連鎖目を評価
            let mut merged_adds = node.additions.clone();
            merged_adds.push((x, color));
            let merged_board = merge_additions_onto(*original, &merged_adds);
            let groups = count_first_clear_groups(&merged_board);
            if groups > 1 {
                // フォールバック: 同じ色を隣接列に移して合流を避け、かつ仮配置側では連鎖が起きることを要求
                let mut separated = false;
                for dx in [-1isize, 1] {
                    let ax = x as isize + dx;
                    if ax < 0 || ax >= W as isize { continue; }
                    let axu = ax as usize;
                    let ay = col_height(&node.leftover, axu);
                    if ay >= H { continue; }
                    // 仮配置時点で同色隣接（連鎖の芽）
                    let used2 = neighbor_colors_at(&node.leftover, axu, ay);
                    if !used2[color as usize] { continue; }

                    let mut after2 = node.leftover;
                    after2[color as usize][axu] |= 1u16 << ay;
                    let (gain2, next_leftover2) = simulate_chain_and_final(after2);

                    // 代替のマージ側で 1グループかつ free-top、かつ仮配置で1連鎖以上
                    let mut merged_adds2 = node.additions.clone();
                    merged_adds2.push((axu, color));
                    let merged_board2 = merge_additions_onto(*original, &merged_adds2);
                    let groups2 = count_first_clear_groups(&merged_board2);
                    if gain2 >= 1 && groups2 == 1 && first_clear_has_free_top(&merged_board2)
                        && first_clear_largest_size(&merged_board2) <= 5 {
                        // 追加制約: free-top 列かつ、新規追加セルが E1 消去に参加している場合は回避
                        let ft2 = first_clear_free_top_cols(&merged_board2);
                        if ft2[axu] {
                            let occ_total = merged_adds2.iter().filter(|(cx, _)| *cx == axu).count();
                            let y_new = col_height(original, axu) + occ_total.saturating_sub(1);
                            let clear2 = compute_erase_mask_cols(&merged_board2);
                            if (clear2[axu] & (1u16 << y_new)) != 0 { continue; }
                        }
                        // 追加制約: 初手ペアが free-top 列の消去成分に含まれている場合も回避
                        if let Some(((px, py), (qx, qy))) = first_pair_positions {
                            let clear2 = compute_erase_mask_cols(&merged_board2);
                            let pa = (clear2[px] & (1u16 << py)) != 0;
                            let qb = (clear2[qx] & (1u16 << qy)) != 0;
                            if (ft2[px] && pa) || (ft2[qx] && qb) { continue; }
                        }
                        let mut additions = node.additions.clone();
                        additions.push((axu, color));
                        if precompleted_free_top_e1_on_last_step(original, &additions) { continue; }
                        let first_locked = node.first_chain_locked || count_first_clear_groups(&merged_board2) > 0 || gain2 >= 1;
                        out.push(BeamNode {
                            leftover: next_leftover2,
                            additions,
                            total_chain: node.total_chain.saturating_add(gain2),
                            first_chain_locked: first_locked,
                        });
                        separated = true;
                    }
                }
                if !separated {
                    // 回避不能ならこの候補は捨てる
                    continue;
                } else {
                    // 代替を push 済み。元の配置はスキップ
                    continue;
                }
            } else if groups == 1 {
                // 条件: 少なくとも1列は塊の上に何も乗っていない かつ 連結数 <= 5（merged_board で判定）
                if !(first_clear_has_free_top(&merged_board) && first_clear_largest_size(&merged_board) <= 5) {
                    // 隣接列に置き方を変えて、この条件を満たせるか試す
                    let mut accepted = false;
                    for dx in [-1isize, 1] {
                        let ax = x as isize + dx;
                        if ax < 0 || ax >= W as isize { continue; }
                        let axu = ax as usize;
                        let ay = col_height(&node.leftover, axu);
                        if ay >= H { continue; }
                        let used2 = neighbor_colors_at(&node.leftover, axu, ay);
                        if !used2[color as usize] { continue; }

                        let mut after2 = node.leftover;
                        after2[color as usize][axu] |= 1u16 << ay;
                        let (gain2, next_leftover2) = simulate_chain_and_final(after2);

                        let mut merged_adds2 = node.additions.clone();
                        merged_adds2.push((axu, color));
                        let merged_board2 = merge_additions_onto(*original, &merged_adds2);
                        let groups2 = count_first_clear_groups(&merged_board2);
                        if groups2 == 1 && first_clear_has_free_top(&merged_board2)
                            && first_clear_largest_size(&merged_board2) <= 5 {
                            // 追加制約: free-top 列かつ、新規追加セルが E1 消去に参加している場合は回避
                            let ft2 = first_clear_free_top_cols(&merged_board2);
                            if ft2[axu] {
                                let occ_total = merged_adds2.iter().filter(|(cx, _)| *cx == axu).count();
                                let y_new = col_height(original, axu) + occ_total.saturating_sub(1);
                                let clear2 = compute_erase_mask_cols(&merged_board2);
                                if (clear2[axu] & (1u16 << y_new)) != 0 { continue; }
                            }
                            // 追加制約: 初手ペアが free-top 列の消去成分に含まれている場合も回避
                            if let Some(((px, py), (qx, qy))) = first_pair_positions {
                                let clear2 = compute_erase_mask_cols(&merged_board2);
                                let pa = (clear2[px] & (1u16 << py)) != 0;
                                let qb = (clear2[qx] & (1u16 << qy)) != 0;
                                if (ft2[px] && pa) || (ft2[qx] && qb) { continue; }
                            }
                            let mut additions = node.additions.clone();
                            additions.push((axu, color));
                            if precompleted_free_top_e1_on_last_step(original, &additions) { continue; }
                            let first_locked = node.first_chain_locked || count_first_clear_groups(&merged_board2) > 0 || gain2 >= 1;
                            out.push(BeamNode {
                                leftover: next_leftover2,
                                additions,
                                total_chain: node.total_chain.saturating_add(gain2),
                                first_chain_locked: first_locked,
                            });
                            accepted = true;
                        }
                    }
                    if !accepted { continue; } else { continue; }
                } else {
                    // 条件自体は満たしているが、free-top 列に自分で置いたケースを回避
                    let ft = first_clear_free_top_cols(&merged_board);
                    if ft[x] {
                        // 隣接列へ移せるなら移す
                        let mut moved = false;
                        for dx in [-1isize, 1] {
                            let ax = x as isize + dx;
                            if ax < 0 || ax >= W as isize { continue; }
                            let axu = ax as usize;
                            let ay = col_height(&node.leftover, axu);
                            if ay >= H { continue; }
                            let used2 = neighbor_colors_at(&node.leftover, axu, ay);
                            if !used2[color as usize] { continue; }
                            let mut after2 = node.leftover;
                            after2[color as usize][axu] |= 1u16 << ay;
                            let (gain2, next_leftover2) = simulate_chain_and_final(after2);
                            let mut merged_adds2 = node.additions.clone();
                            merged_adds2.push((axu, color));
                            let merged_board2 = merge_additions_onto(*original, &merged_adds2);
                            let groups2 = count_first_clear_groups(&merged_board2);
                            if groups2 == 1 && first_clear_has_free_top(&merged_board2)
                                && first_clear_largest_size(&merged_board2) <= 5 {
                                let ft2 = first_clear_free_top_cols(&merged_board2);
                                if ft2[axu] {
                                    let occ_total = merged_adds2.iter().filter(|(cx, _)| *cx == axu).count();
                                    let y_new = col_height(original, axu) + occ_total.saturating_sub(1);
                                    let clear2 = compute_erase_mask_cols(&merged_board2);
                                    if (clear2[axu] & (1u16 << y_new)) != 0 { continue; }
                                }
                                if let Some(((px, py), (qx, qy))) = first_pair_positions {
                                    let clear2 = compute_erase_mask_cols(&merged_board2);
                                    let pa = (clear2[px] & (1u16 << py)) != 0;
                                    let qb = (clear2[qx] & (1u16 << qy)) != 0;
                                    if (ft2[px] && pa) || (ft2[qx] && qb) { continue; }
                                }
                                let mut additions = node.additions.clone();
                                additions.push((axu, color));
                                if precompleted_free_top_e1_on_last_step(original, &additions) { continue; }
                                let first_locked = node.first_chain_locked || count_first_clear_groups(&merged_board2) > 0 || gain2 >= 1;
                                out.push(BeamNode {
                                    leftover: next_leftover2,
                                    additions,
                                    total_chain: node.total_chain.saturating_add(gain2),
                                    first_chain_locked: first_locked,
                                });
                                moved = true;
                            }
                        }
                        if !moved { continue; } else { continue; }
                    }
                }
            }

            let mut additions = node.additions.clone();
            additions.push((x, color));

            // 初手が発生したか（merged側でグループが >=1 でも true とする）
            let first_locked = if node.first_chain_locked {
                true
            } else {
                count_first_clear_groups(&merged_board) > 0 || gain >= 1
            };

            // 通常受理：初手ペア／追加群が free-top 列の消去成分に含まれていないか最終確認
            if let Some(((px, py), (qx, qy))) = first_pair_positions {
                let clear = compute_erase_mask_cols(&merged_board);
                if first_clear_has_free_top(&merged_board) {
                    let ft = first_clear_free_top_cols(&merged_board);
                    let pa = (clear[px] & (1u16 << py)) != 0;
                    let qb = (clear[qx] & (1u16 << qy)) != 0;
                    if (ft[px] && pa) || (ft[qx] && qb) {
                        continue;
                    }
                }
            }
            if any_addition_on_free_top_involved_in_e1(original, &additions) { continue; }
            out.push(BeamNode {
                leftover: next_leftover,
                additions,
                total_chain: node.total_chain.saturating_add(gain),
                first_chain_locked: first_locked,
            });
        }
    }
    out
}

fn cp_iterative_chain_clearing_beam(
    original: [[u16; W]; 4],
    max_depth: usize,
    beam_width: usize,
    first_pair_positions: Option<((usize, usize), (usize, usize))>,
) -> Option<(Vec<(usize, u8)>, u32)> {
    // 自然連鎖を先に適用
    let (baseline, leftover0) = simulate_chain_and_final(original);

    let mut beam: Vec<BeamNode> = vec![BeamNode {
        leftover: leftover0,
        additions: Vec::new(),
        total_chain: baseline,
        first_chain_locked: baseline > 0,
    }];

    let mut best = beam[0].clone();
    let mut best_valid: Option<BeamNode> = None; // g==1 かつ free-top を満たした最良

    for _d in 0..max_depth {
        // 目標達成
        if cols_is_empty(&best.leftover) { break; }

        let mut cand_all: Vec<BeamNode> = Vec::new();
        for node in &beam {
            let ex = expand_beam_node(&original, node, first_pair_positions);
            cand_all.extend(ex);
        }
        if cand_all.is_empty() { break; }

        // 有効解（g==1 & free-top）を収集（トリミング前）
        for n in &cand_all {
            let merged = merge_additions_onto(original, &n.additions);
            let g = count_first_clear_groups(&merged);
            if g == 1 && first_clear_has_free_top(&merged) && first_clear_largest_size(&merged) <= 5 {
                if precompleted_free_top_e1_on_last_step(&original, &n.additions) { continue; }
                // 追加ゲート: 直近の単ぷよが free-top 列の E1 に参加している場合は拒否
                if let Some(&(lx, _lc)) = n.additions.last() {
                    let ft = first_clear_free_top_cols(&merged);
                    if ft[lx] {
                        let occ_total = n.additions.iter().filter(|(cx, _)| *cx == lx).count();
                        let y_new = col_height(&original, lx) + occ_total.saturating_sub(1);
                        let clear = compute_erase_mask_cols(&merged);
                        if (clear[lx] & (1u16 << y_new)) != 0 { continue; }
                    }
                }
                match &mut best_valid {
                    None => best_valid = Some(n.clone()),
                    Some(cur) => {
                        if n.total_chain > cur.total_chain {
                            *cur = n.clone();
                        } else if n.total_chain == cur.total_chain {
                            let co = board_occupied_count(&cur.leftover);
                            let no = board_occupied_count(&n.leftover);
                            if no < co { *cur = n.clone(); }
                        }
                    }
                }
            }
        }

        // スコアでソート
        cand_all.sort_by(|a, b| {
            // occupancy は小さい方が良い
            let occ_a = board_occupied_count(&a.leftover);
            let occ_b = board_occupied_count(&b.leftover);
            // 初手最大塊サイズは初手未ロック時のみ評価
            let a_first = if a.first_chain_locked { 0 } else { first_clear_largest_size(&merge_additions_onto(original, &a.additions)) };
            let b_first = if b.first_chain_locked { 0 } else { first_clear_largest_size(&merge_additions_onto(original, &b.additions)) };

            // 並べ替えキー（降順）: total_chain, a_first, -additions_len, -occ
            (b.total_chain.cmp(&a.total_chain))
                .then(b_first.cmp(&a_first))
                .then(a.additions.len().cmp(&b.additions.len()))
                .then(occ_a.cmp(&occ_b))
        });

        // 上位 beam_width
        if cand_all.len() > beam_width { cand_all.truncate(beam_width); }
        beam = cand_all;

        // ベスト更新（全消し優先, 次いで total_chain）
        for n in &beam {
            if let Some(&(lx, _lc)) = n.additions.last() {
                let merged = merge_additions_onto(original, &n.additions);
                if precompleted_free_top_e1_on_last_step(&original, &n.additions) { continue; }
                let ft = first_clear_free_top_cols(&merged);
                if ft[lx] {
                    let occ_total = n.additions.iter().filter(|(cx, _)| *cx == lx).count();
                    let y_new = col_height(&original, lx) + occ_total.saturating_sub(1);
                    let clear = compute_erase_mask_cols(&merged);
                    if (clear[lx] & (1u16 << y_new)) != 0 { continue; }
                }
            }
            if cols_is_empty(&n.leftover) {
                best = n.clone();
                break;
            }
            if n.total_chain > best.total_chain {
                best = n.clone();
            } else if n.total_chain == best.total_chain {
                // tie: occupancy 少ない方
                let bo = board_occupied_count(&best.leftover);
                let no = board_occupied_count(&n.leftover);
                if no < bo { best = n.clone(); }
            }
        }
    }

    // 最優先は探索中に見つかった「g==1 & free-top」を満たすノード
    if let Some(v) = best_valid {
        return Some((v.additions, v.total_chain));
    }

    // 次点: 終了時点の beam から条件を満たすもの（保険）
    let mut final_best_opt: Option<BeamNode> = None;
    for n in &beam {
        let merged = merge_additions_onto(original, &n.additions);
        let g = count_first_clear_groups(&merged);
        if g == 1 && first_clear_has_free_top(&merged) && first_clear_largest_size(&merged) <= 5 {
            if precompleted_free_top_e1_on_last_step(&original, &n.additions) { continue; }
            // 追加ゲート: 直近の単ぷよが free-top 列の E1 に参加している場合は拒否
            if let Some(&(lx, _lc)) = n.additions.last() {
                let ft = first_clear_free_top_cols(&merged);
                if ft[lx] {
                    let occ_total = n.additions.iter().filter(|(cx, _)| *cx == lx).count();
                    let y_new = col_height(&original, lx) + occ_total.saturating_sub(1);
                    let clear = compute_erase_mask_cols(&merged);
                    if (clear[lx] & (1u16 << y_new)) != 0 { continue; }
                }
            }
            match &mut final_best_opt {
                None => final_best_opt = Some(n.clone()),
                Some(cur) => {
                    if n.total_chain > cur.total_chain {
                        *cur = n.clone();
                    } else if n.total_chain == cur.total_chain {
                        let fo = board_occupied_count(&cur.leftover);
                        let no = board_occupied_count(&n.leftover);
                        if no < fo { *cur = n.clone(); }
                    }
                }
            }
        }
    }
    if let Some(v) = final_best_opt { return Some((v.additions, v.total_chain)); }

    // 条件を満たす解が見つからなければ None（プレビュー更新しない）
    None
}

// ====== ビームサーチ（緩和版）：プレビュー用のフォールバック ======
// 厳格条件（free-top/g==1/各種ゲート）を適用せず、単に total_chain 最大を目指す。
// 必ず何かしらの解（現在の best）を返す。
fn cp_iterative_chain_clearing_beam_relaxed(
    original: [[u16; W]; 4],
    max_depth: usize,
    beam_width: usize,
) -> (Vec<(usize, u8)>, u32) {
    let (baseline, leftover0) = simulate_chain_and_final(original);
    let mut beam: Vec<BeamNode> = vec![BeamNode {
        leftover: leftover0,
        additions: Vec::new(),
        total_chain: baseline,
        first_chain_locked: baseline > 0,
    }];

    let mut best = beam[0].clone();
    for _d in 0..max_depth {
        if cols_is_empty(&best.leftover) { break; }

        let mut cand_all: Vec<BeamNode> = Vec::new();
        for node in &beam {
            for x in 0..W {
                let y = col_height(&node.leftover, x);
                if y >= H { continue; }
                let used = neighbor_colors_at(&node.leftover, x, y);
                let any_used = used.iter().any(|&u| u);
                for color in 0u8..4u8 {
                    // 隣接色が存在する列は隣接色のみ、そうでなければ全色を許容（停滞回避）
                    if any_used && !used[color as usize] { continue; }

                    let mut after = node.leftover;
                    after[color as usize][x] |= 1u16 << y;
                    let (gain, next_leftover) = simulate_chain_and_final(after);

                    let mut additions = node.additions.clone();
                    additions.push((x, color));
                    cand_all.push(BeamNode {
                        leftover: next_leftover,
                        additions,
                        total_chain: node.total_chain.saturating_add(gain),
                        first_chain_locked: node.first_chain_locked || gain >= 1,
                    });
                }
            }
        }

        if cand_all.is_empty() { break; }
        cand_all.sort_by(|a, b| {
            let occ_a = board_occupied_count(&a.leftover);
            let occ_b = board_occupied_count(&b.leftover);
            (b.total_chain.cmp(&a.total_chain))
                .then(occ_a.cmp(&occ_b))
                .then(a.additions.len().cmp(&b.additions.len()))
        });
        if cand_all.len() > beam_width { cand_all.truncate(beam_width); }
        beam = cand_all;

        for n in &beam {
            if cols_is_empty(&n.leftover) {
                best = n.clone();
                break;
            }
            if n.total_chain > best.total_chain {
                best = n.clone();
            } else if n.total_chain == best.total_chain {
                let bo = board_occupied_count(&best.leftover);
                let no = board_occupied_count(&n.leftover);
                if no < bo { best = n.clone(); }
            }
        }
    }

    (best.additions, best.total_chain)
}

// 列 x への 4色一括代入（ループ展開）
#[inline(always)]
fn assign_col_unrolled(cols: &mut [[u16; W]; 4], x: usize, masks: &[u16; 4]) {
    // 安全のためのデバッグアサート（最適化時は消える）
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
fn reaches_t_from_pre_single_e1(pre: &[[u16; W]; 4], t: u32, exact_four_only: bool, require_free_top_e1: bool) -> bool {
    // 追加フィルタ: 1連鎖目にfree-top列が1つ以上（必要な場合）
    if require_free_top_e1 && !first_clear_has_free_top(pre) {
        return false;
    }
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
            if bb.count_ones() < 4 { continue; }
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

    // exact4 の場合、初回消去が 4 以外なら即不成立（以降の高コスト判定を回避）
    if exact_four_only && total != 4 {
        return false;
    }

    // 先に空白隣接とオーバーハングの簡易チェックで早期棄却
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

        if run <= 1 {
            ok_overhang = true;
            break;
        }
    }
    if !ok_overhang {
        return false;
    }

    // E1 単一連結チェック（clear_bb が 1 コンポーネントか）
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

    // 残り (t-1) 連鎖のポテンシャル上限チェック
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

// ========== 追撃最適化：占有比較の u128 化 & ハッシュの占有ビット走査 ==========

// 占有パターンの16bit整列パックを作って左右比較（u128 で一発）
// Some(false)=正方向, Some(true)=ミラー, None=完全同一（左右対称）
#[inline(always)]
fn choose_mirror_by_occupancy(cols: &[[u16; W]; 4]) -> Option<bool> {
    // 各列の占有（14bit）を 16bit チャンクに入れて 96bit（u128）に連結
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

// LUT 風：色マッピングの更新（未割当なら next を払い出し）
#[inline(always)]
fn map_code_lut(entry: &mut u8, next: &mut u8) -> u64 {
    if *entry == u8::MAX {
        *entry = *next;
        *next = next.wrapping_add(1);
    }
    *entry as u64
}

// ====== ここを差し替え：空白を P^k でまとめ掛けする高速版 ======
#[inline(always)]
fn canonical_hash64_oriented_bits(cols: &[[u16; W]; 4], mirror: bool) -> u64 {
    const P: u64 = 1099511628211;
    const O: u64 = 14695981039346656037;

    // P^k（k=0..14）をその場で構築（到達葉のみで実行されるため十分軽量）
    let mut p_pow = [1u64; 15];
    for i in 1..15 {
        p_pow[i] = p_pow[i - 1].wrapping_mul(P);
    }
    #[inline(always)]
    fn mul_pow(h: u64, pp: &[u64; 15], k: usize) -> u64 {
        debug_assert!(k < 15);
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
                // 列が空：空白14個ぶん一気に掛ける
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            // 直前の占有 y（最初は -1 とみなす）
            let mut prev_y: i32 = -1;

            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                // 占有間の空白をまとめて掛ける
                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                // 色コード（初出→1, 次→2...）
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

            // 列末尾の空白をまとめて掛ける
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

// 占有で決まらない（左右対称）場合のみ、両向き計算して小さい方を採用
#[inline(always)]
fn canonical_hash64_fast(cols: &[[u16; W]; 4]) -> (u64, bool) {
    if let Some(mirror) = choose_mirror_by_occupancy(cols) {
        let h = canonical_hash64_oriented_bits(cols, mirror);
        (h, mirror)
    } else {
        let h0 = canonical_hash64_oriented_bits(cols, false);
        let h1 = canonical_hash64_oriented_bits(cols, true);
        if h0 <= h1 { (h0, false) } else { (h1, true) }
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

// 3) JSON を手組みで生成（serde_json を避ける）
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
    map: U64Map<bool>, // 使わない（今回の方針では未参照でもOK）
    q: VecDeque<u64>,
}
impl ApproxLru {
    fn new(limit: usize) -> Self {
        let cap = (limit.saturating_mul(11) / 10).max(16);
        let map: U64Map<bool> = std::collections::HashMap::with_capacity_and_hasher(
            cap,
            BuildNoHashHasher::default(),
        );
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
    require_free_top_e1: bool,
    _memo: &mut ApproxLru, // 現方針では実質未使用（陽性のみ挿入なら使用可）
    local_output_once: &mut U64Set,
    global_output_once: &Arc<DU64Set>,
    _global_memo: &Arc<DU64Map<bool>>, // get を廃止
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

    // ==== 葉はここで処理（上界チェックはスキップ！）====
    if depth == W {
        *leaves_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].leaves += 1;
        }

        // ★ 早期リターン（落下や到達判定より前）
        if placed_total < 4 * threshold {
            // 4T 未満はどう頑張っても T 連鎖に届かない
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_tshort += 1;
            }
            return Ok(());
        }
        if !any_color_has_four(cols0) {
            // E1 不可能
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_e1_impossible += 1;
            }
            return Ok(());
        }

        // ★ 初回 fall をスキップ（列生成で既に重力正規化済み）
        let pre = *cols0;

        // ==== 先に到達判定のみ実行（ミス計上＋計測は miss_compute に積む）====
        if profile_enabled {
            time_batch.dfs_counts[depth].memo_miss += 1;
        }
        *mmiss_batch += 1;
        let reached = prof!(profile_enabled, time_batch.dfs_times[depth].leaf_memo_miss_compute, {
            reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only, require_free_top_e1)
        });
        if !reached {
            // 統計フラッシュ（従来通り）
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

        // ==== 到達した場合にだけ正規化キー生成（計測は leaf_hash に積む）====
        let (key64, mirror) = prof!(profile_enabled, time_batch.dfs_times[depth].leaf_hash, {
            canonical_hash64_fast(&pre)
        });

        // （必要なら）陽性だけローカル LRU に入れる
        // if _lru_limit > 0 { _memo.insert(key64, true); }

        if !local_output_once.contains(&key64) {
            // 近似的なグローバル一意集合で重複回避
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
                let line = prof!(profile_enabled, time_batch.dfs_times[depth].out_serialize, {
                    let key_str = encode_canonical_string(&pre, mirror);
                    let hash = fnv1a32(&key_str);
                    let rows = serialize_board_from_cols(&pre);
                    make_json_line_str(&key_str, hash, threshold, &rows, map_label_to_color, mirror)
                });
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

    // ===== 上界枝刈り（非葉のみ）=====
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
            && (*nodes_batch > 0 || *leaves_batch > 0 || *outputs_batch > 0 || *pruned_batch > 0 || *lhit_batch > 0 || *ghit_batch > 0 || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta { nodes: *nodes_batch, leaves: *leaves_batch, outputs: *outputs_batch, pruned: *pruned_batch, lhit: *lhit_batch, ghit: *ghit_batch, mmiss: *mmiss_batch });
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
                    require_free_top_e1,
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
                        require_free_top_e1,
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
                        require_free_top_e1,
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
    require_free_top_e1: bool,
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
        "抽象ラベル={} / 彩色候補={} / 4個消しモード={} / free-top E1限定={} / 計測={}",
        info.labels.iter().collect::<String>(),
        colorings.len(),
        if exact_four_only { "ON" } else { "OFF" },
        if require_free_top_e1 { "ON" } else { "OFF" },
        if profile_enabled { "ON" } else { "OFF" }
    )));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let bmi2 = std::is_x86_feature_detected!("bmi2");
        let popcnt = std::is_x86_feature_detected!("popcnt");
        let _ = tx.send(Message::Log(format!(
            "CPU features: bmi2={} / popcnt={}", bmi2, popcnt
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
    let global_output_once: Arc<DU64Set> = Arc::new(DU64Set::with_hasher(BuildNoHashHasher::default()));
    // グローバル memo は get を廃止（挿入もしない方針）
    let global_memo: Arc<DU64Map<bool>> = Arc::new(DU64Map::with_hasher(BuildNoHashHasher::default()));

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
    metas
        .par_iter()
        .enumerate()
        .try_for_each(|(i, (map_label_to_color, gens, max_fill, order))| -> Result<()> {
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
                        if abort.load(Ordering::Relaxed) { return Ok(()); }
                        let mut cols0 = [[0u16; W]; 4];
                        for c in 0..4 { cols0[c][first_x] = masks[c]; }
                        let mut memo = ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
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
                            require_free_top_e1,
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

                        if !batch.is_empty() { let _ = wtx.send(batch); }
                        if nodes_batch > 0 || leaves_batch > 0 || outputs_batch > 0 || pruned_batch > 0 || lhit_batch > 0 || ghit_batch > 0 || mmiss_batch > 0 {
                            let _ = stx.send(StatDelta { nodes: nodes_batch, leaves: leaves_batch, outputs: outputs_batch, pruned: pruned_batch, lhit: lhit_batch, ghit: ghit_batch, mmiss: mmiss_batch });
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
                        if abort.load(Ordering::Relaxed) { return Ok(()); }
                        for masks in &first_candidates {
                            if abort.load(Ordering::Relaxed) { break; }

                            let mut cols0 = [[0u16; W]; 4];
                            for c in 0..4 {
                                cols0[c][first_x] = masks[c];
                                cols0[c][second_x] = m2[c];
                            }
                            let mut memo = ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
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

                            let placed2: u32 = (0..4).map(|c| masks[c].count_ones() + m2[c].count_ones()).sum();
                            let _ = dfs_combine_parallel(
                                2,
                                &mut cols0,
                                gens,
                                order,
                                threshold,
                                exact_four_only,
                                require_free_top_e1,
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

                            if !batch.is_empty() { let _ = wtx.send(batch); }
                            if nodes_batch > 0 || leaves_batch > 0 || outputs_batch > 0 || pruned_batch > 0 || lhit_batch > 0 || ghit_batch > 0 || mmiss_batch > 0 {
                                let _ = stx.send(StatDelta { nodes: nodes_batch, leaves: leaves_batch, outputs: outputs_batch, pruned: pruned_batch, lhit: lhit_batch, ghit: ghit_batch, mmiss: mmiss_batch });
                            }
                            if profile_enabled && time_delta_has_any(&time_batch) {
                                let _ = tx.send(Message::TimeDelta(time_batch));
                            }
                        }
                        Ok(())
                    })?;
            }
            Ok(())
        })?;

    drop(wtx);
    drop(stx);
    let writer_result = writer_handle.join().map_err(|_| anyhow!("writer join error"))?;
    writer_result?;
    agg_handle.join().map_err(|_| anyhow!("agg join error"))?;

    Ok(())
}

// ユーティリティ：配列初期化（const generics）
fn array_init<T, F: FnMut(usize) -> T, const N: usize>(mut f: F) -> [T; N] {
    use std::mem::MaybeUninit;
    let mut data: [MaybeUninit<T>; N] =
        unsafe { MaybeUninit::uninit().assume_init() };
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