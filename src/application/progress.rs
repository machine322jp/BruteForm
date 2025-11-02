// 進捗管理

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 進捗メッセージの種類
#[derive(Clone, Debug)]
pub enum ProgressMessage {
    /// 検索開始
    Started { total_work: u64 },
    /// 進捗更新
    Progress {
        completed: u64,
        nodes_searched: u64,
        results_found: u64,
    },
    /// 検索完了
    Completed {
        total_nodes: u64,
        total_results: u64,
        elapsed: Duration,
    },
    /// エラー発生
    Error { message: String },
    /// 中断
    Aborted,
}

/// 進捗統計
#[derive(Clone, Debug, Default)]
pub struct ProgressStats {
    pub nodes_searched: u64,
    pub leaves_reached: u64,
    pub results_found: u64,
    pub pruned: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// 進捗マネージャー
pub struct ProgressManager {
    stats: Arc<ProgressStats>,
    abort_flag: Arc<AtomicBool>,
    nodes_searched: Arc<AtomicU64>,
    results_found: Arc<AtomicU64>,
    start_time: Instant,
}

impl ProgressManager {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(ProgressStats::default()),
            abort_flag: Arc::new(AtomicBool::new(false)),
            nodes_searched: Arc::new(AtomicU64::new(0)),
            results_found: Arc::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }

    /// 検索中断フラグを取得
    pub fn abort_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.abort_flag)
    }

    /// 検索を中断
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }

    /// 中断されたかチェック
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }

    /// ノード数を追加
    pub fn add_nodes(&self, count: u64) {
        self.nodes_searched.fetch_add(count, Ordering::Relaxed);
    }

    /// 結果数を追加
    pub fn add_results(&self, count: u64) {
        self.results_found.fetch_add(count, Ordering::Relaxed);
    }

    /// 現在の統計を取得
    pub fn get_stats(&self) -> ProgressStats {
        ProgressStats {
            nodes_searched: self.nodes_searched.load(Ordering::Relaxed),
            results_found: self.results_found.load(Ordering::Relaxed),
            leaves_reached: 0,
            pruned: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// 経過時間を取得
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// 検索速度（ノード/秒）を取得
    pub fn nodes_per_second(&self) -> f64 {
        let nodes = self.nodes_searched.load(Ordering::Relaxed) as f64;
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            nodes / elapsed
        } else {
            0.0
        }
    }

    /// リセット
    pub fn reset(&mut self) {
        self.abort_flag.store(false, Ordering::Relaxed);
        self.nodes_searched.store(0, Ordering::Relaxed);
        self.results_found.store(0, Ordering::Relaxed);
        self.start_time = Instant::now();
    }
}

impl Default for ProgressManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manager_starts_clean() {
        let mgr = ProgressManager::new();
        assert!(!mgr.is_aborted());
        assert_eq!(mgr.get_stats().nodes_searched, 0);
        assert_eq!(mgr.get_stats().results_found, 0);
    }

    #[test]
    fn can_abort() {
        let mgr = ProgressManager::new();
        assert!(!mgr.is_aborted());
        mgr.abort();
        assert!(mgr.is_aborted());
    }

    #[test]
    fn can_track_nodes() {
        let mgr = ProgressManager::new();
        mgr.add_nodes(100);
        mgr.add_nodes(50);
        assert_eq!(mgr.get_stats().nodes_searched, 150);
    }

    #[test]
    fn can_track_results() {
        let mgr = ProgressManager::new();
        mgr.add_results(5);
        mgr.add_results(3);
        assert_eq!(mgr.get_stats().results_found, 8);
    }

    #[test]
    fn reset_clears_state() {
        let mut mgr = ProgressManager::new();
        mgr.add_nodes(100);
        mgr.abort();

        mgr.reset();
        assert!(!mgr.is_aborted());
        assert_eq!(mgr.get_stats().nodes_searched, 0);
    }

    #[test]
    fn nodes_per_second_calculation() {
        let mgr = ProgressManager::new();
        mgr.add_nodes(1000);
        std::thread::sleep(Duration::from_millis(100));

        let nps = mgr.nodes_per_second();
        assert!(nps > 0.0);
    }
}
