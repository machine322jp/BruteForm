// 統計・メッセージ型定義

use crate::constants::W;
use crate::profiling::{ProfileTotals, TimeDelta};
use num_bigint::BigUint;

/// 探索進捗統計
#[derive(Default, Clone)]
pub struct Stats {
    pub searching: bool,
    pub unique: u64,
    pub output: u64,
    pub nodes: u64,
    pub pruned: u64,
    pub memo_hit_local: u64,
    pub memo_hit_global: u64,
    pub memo_miss: u64,
    pub total: BigUint,
    pub done: BigUint,
    pub rate: f64,
    pub memo_len: usize,
    pub lru_limit: usize,
    pub profile: ProfileTotals,
}

/// 統計の増分
#[derive(Clone, Copy, Default)]
pub struct StatDelta {
    pub nodes: u64,
    pub leaves: u64,
    pub outputs: u64,
    pub pruned: u64,
    pub lhit: u64,
    pub ghit: u64,
    pub mmiss: u64,
}

/// 探索スレッドからのメッセージ
pub enum Message {
    Log(String),
    Preview([[u16; W]; 4]),
    Progress(Stats),
    Finished(Stats),
    Error(String),
    TimeDelta(TimeDelta),
}
