// 計測モジュール

use crate::constants::W;
use std::time::Duration;

/// DFS 深さごとの処理時間
#[derive(Default, Clone, Copy)]
pub struct DfsDepthTimes {
    pub gen_candidates: Duration,
    pub assign_cols: Duration,
    pub upper_bound: Duration,
    // 葉専用
    pub leaf_fall_pre: Duration,
    pub leaf_hash: Duration,
    pub leaf_memo_get: Duration,
    pub leaf_memo_miss_compute: Duration,
    pub out_serialize: Duration,
}

/// DFS 深さごとのカウンタ
#[derive(Default, Clone, Copy)]
pub struct DfsDepthCounts {
    pub nodes: u64,
    pub cand_generated: u64,
    pub pruned_upper: u64,
    pub leaves: u64,
    pub leaf_pre_tshort: u64,
    pub leaf_pre_e1_impossible: u64,
    pub memo_lhit: u64,
    pub memo_ghit: u64,
    pub memo_miss: u64,
}

/// 計測結果の合計
#[derive(Default, Clone)]
pub struct ProfileTotals {
    pub dfs_times: [DfsDepthTimes; W + 1],
    pub dfs_counts: [DfsDepthCounts; W + 1],
    pub io_write_total: Duration,
}

/// 時間増分メッセージ用
#[derive(Default, Clone)]
pub struct TimeDelta {
    pub dfs_times: [DfsDepthTimes; W + 1],
    pub dfs_counts: [DfsDepthCounts; W + 1],
    pub io_write_total: Duration,
}

impl ProfileTotals {
    /// 増分を加算
    pub fn add_delta(&mut self, d: &TimeDelta) {
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

/// TimeDelta に何らかのデータがあるかチェック
pub fn time_delta_has_any(d: &TimeDelta) -> bool {
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

/// 計測マクロ：enabled 時のみ計測
#[macro_export]
macro_rules! prof {
    ($enabled:expr, $slot:expr, $e:expr) => {{
        if $enabled {
            let __t0 = std::time::Instant::now();
            let __r = $e;
            $slot += __t0.elapsed();
            __r
        } else {
            $e
        }
    }};
}

/// ProfileTotals に何らかのデータがあるかチェック
pub fn has_profile_any(p: &ProfileTotals) -> bool {
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

/// Duration をミリ秒文字列に整形
pub fn fmt_dur_ms(d: Duration) -> String {
    let ms = d.as_secs_f64() * 1000.0;
    if ms < 1.0 {
        format!("{:.3} ms", ms)
    } else {
        format!("{:.1} ms", ms)
    }
}
