// 検索結果の定義

use super::config::ChainCount;
use crate::domain::board::BoardBits;
use serde::{Deserialize, Serialize};

/// 単一の検索結果
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub board: BoardBits,
    pub chain_count: ChainCount,
    pub hash: u64,
    pub is_mirror: bool,
}

/// 検索サマリー
#[derive(Clone, Debug)]
pub struct SearchSummary {
    pub unique_count: u64,
    pub total_nodes: u64,
    pub pruned_count: u64,
    pub elapsed_seconds: f64,
    pub nodes_per_second: f64,
}

impl SearchSummary {
    pub fn new() -> Self {
        Self {
            unique_count: 0,
            total_nodes: 0,
            pruned_count: 0,
            elapsed_seconds: 0.0,
            nodes_per_second: 0.0,
        }
    }
}

impl Default for SearchSummary {
    fn default() -> Self {
        Self::new()
    }
}
