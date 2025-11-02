// 総当たり検索のUI状態管理

use std::path::PathBuf;
use crate::domain::board::Board;
use crate::domain::search::SearchConfig;
use crate::application::progress::ProgressStats;

/// 検索の実行状態
#[derive(Clone, Debug, PartialEq)]
pub enum SearchStatus {
    /// 待機中
    Idle,
    /// 実行中
    Running,
    /// 完了
    Completed,
    /// エラー
    Error(String),
}

impl Default for SearchStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// 総当たり検索のUI状態
#[derive(Clone, Debug)]
pub struct BruteforceState {
    /// テンプレート盤面
    pub template: Board,
    /// 検索設定
    pub config: SearchConfig,
    /// 出力ファイルパス
    pub output_path: Option<PathBuf>,
    /// 検索状態
    pub status: SearchStatus,
    /// 進捗統計
    pub progress: Option<ProgressStats>,
    /// 結果数
    pub result_count: u64,
    /// 開始時刻（秒）
    pub started_at: Option<f64>,
    /// 完了時刻（秒）
    pub completed_at: Option<f64>,
}

impl BruteforceState {
    pub fn new() -> Self {
        Self {
            template: Board::with_default_template(),
            config: SearchConfig::default(),
            output_path: None,
            status: SearchStatus::Idle,
            progress: None,
            result_count: 0,
            started_at: None,
            completed_at: None,
        }
    }

    /// 検索を開始
    pub fn start_search(&mut self, output_path: PathBuf) {
        self.output_path = Some(output_path);
        self.status = SearchStatus::Running;
        self.progress = None;
        self.result_count = 0;
        self.started_at = Some(current_time_secs());
        self.completed_at = None;
    }

    /// 検索を完了
    pub fn complete_search(&mut self, result_count: u64) {
        self.status = SearchStatus::Completed;
        self.result_count = result_count;
        self.completed_at = Some(current_time_secs());
    }

    /// エラー発生
    pub fn set_error(&mut self, message: String) {
        self.status = SearchStatus::Error(message);
        self.completed_at = Some(current_time_secs());
    }

    /// 進捗を更新
    pub fn update_progress(&mut self, stats: ProgressStats) {
        self.progress = Some(stats);
        if let Some(stats) = &self.progress {
            self.result_count = stats.results_found;
        }
    }

    /// リセット
    pub fn reset(&mut self) {
        self.status = SearchStatus::Idle;
        self.progress = None;
        self.result_count = 0;
        self.started_at = None;
        self.completed_at = None;
    }

    /// 実行中かチェック
    pub fn is_running(&self) -> bool {
        matches!(self.status, SearchStatus::Running)
    }

    /// 完了したかチェック
    pub fn is_completed(&self) -> bool {
        matches!(self.status, SearchStatus::Completed)
    }

    /// エラーかチェック
    pub fn is_error(&self) -> bool {
        matches!(self.status, SearchStatus::Error(_))
    }

    /// 経過時間（秒）を取得
    pub fn elapsed_time(&self) -> Option<f64> {
        self.started_at.map(|start| {
            let end = self.completed_at.unwrap_or_else(current_time_secs);
            end - start
        })
    }
}

impl Default for BruteforceState {
    fn default() -> Self {
        Self::new()
    }
}

/// 現在時刻を秒で取得
fn current_time_secs() -> f64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state_is_idle() {
        let state = BruteforceState::new();
        assert!(matches!(state.status, SearchStatus::Idle));
        assert!(!state.is_running());
        assert!(!state.is_completed());
    }

    #[test]
    fn start_search_changes_status() {
        let mut state = BruteforceState::new();
        state.start_search(PathBuf::from("output.json"));

        assert!(state.is_running());
        assert!(state.started_at.is_some());
        assert_eq!(state.result_count, 0);
    }

    #[test]
    fn complete_search_updates_state() {
        let mut state = BruteforceState::new();
        state.start_search(PathBuf::from("output.json"));
        state.complete_search(100);

        assert!(state.is_completed());
        assert_eq!(state.result_count, 100);
        assert!(state.completed_at.is_some());
    }

    #[test]
    fn set_error_marks_as_error() {
        let mut state = BruteforceState::new();
        state.set_error("Test error".to_string());

        assert!(state.is_error());
        assert!(matches!(state.status, SearchStatus::Error(_)));
    }

    #[test]
    fn reset_clears_state() {
        let mut state = BruteforceState::new();
        state.start_search(PathBuf::from("output.json"));
        state.complete_search(50);

        state.reset();

        assert!(matches!(state.status, SearchStatus::Idle));
        assert_eq!(state.result_count, 0);
        assert!(state.started_at.is_none());
    }

    #[test]
    fn elapsed_time_calculation() {
        let mut state = BruteforceState::new();
        state.start_search(PathBuf::from("output.json"));
        
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        state.complete_search(10);
        
        let elapsed = state.elapsed_time();
        assert!(elapsed.is_some());
        assert!(elapsed.unwrap() >= 0.1);
    }
}
