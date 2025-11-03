// 総当たり検索のイベント定義（UI層に依存しない）

use crate::constants::W;
use crate::profiling::TimeDelta;
use num_bigint::BigUint;

/// 統計の増分（探索エンジン内部で使用）
#[derive(Clone, Copy, Default, Debug)]
pub struct StatDelta {
    pub nodes: u64,
    pub leaves: u64,
    pub outputs: u64,
    pub pruned: u64,
    pub lhit: u64,
    pub ghit: u64,
    pub mmiss: u64,
}

/// 検索進捗の統計情報
#[derive(Clone, Debug)]
pub struct SearchProgress {
    pub searching: bool,
    pub unique_results: u64,
    pub output_count: u64,
    pub nodes_searched: u64,
    pub pruned_count: u64,
    pub memo_hit_local: u64,
    pub memo_hit_global: u64,
    pub memo_miss: u64,
    pub total_combinations: BigUint,
    pub completed_combinations: BigUint,
    pub search_rate: f64,
    pub memo_size: usize,
    pub lru_limit: usize,
}

impl Default for SearchProgress {
    fn default() -> Self {
        Self {
            searching: false,
            unique_results: 0,
            output_count: 0,
            nodes_searched: 0,
            pruned_count: 0,
            memo_hit_local: 0,
            memo_hit_global: 0,
            memo_miss: 0,
            total_combinations: BigUint::from(0u32),
            completed_combinations: BigUint::from(0u32),
            search_rate: 0.0,
            memo_size: 0,
            lru_limit: 0,
        }
    }
}

/// 検索エンジンからのイベント
#[derive(Clone, Debug)]
pub enum SearchEvent {
    /// ログメッセージ
    Log(String),
    /// プレビュー（盤面スナップショット）
    Preview([[u16; W]; 4]),
    /// 進捗更新
    Progress(SearchProgress),
    /// 検索完了
    Finished(SearchProgress),
    /// エラー発生
    Error(String),
    /// プロファイル情報（計測データ）
    ProfileDelta(TimeDelta),
}
