// 連鎖生成のUI状態管理

use crate::domain::board::Board;
use crate::application::chainplay::service::ChainPlayConfig;

/// 連鎖生成モード
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChainPlayMode {
    /// 新規生成
    Generate,
    /// 頭伸ばし
    Extend,
    /// ビーム探索
    BeamSearch,
}

impl Default for ChainPlayMode {
    fn default() -> Self {
        Self::Generate
    }
}

/// 連鎖生成の実行状態
#[derive(Clone, Debug, PartialEq)]
pub enum GenerationStatus {
    /// 待機中
    Idle,
    /// 実行中
    Running,
    /// 完了
    Completed { results_count: usize },
    /// エラー
    Error(String),
}

impl Default for GenerationStatus {
    fn default() -> Self {
        Self::Idle
    }
}

/// 連鎖生成候補
#[derive(Clone, Debug)]
pub struct ChainCandidate {
    /// 盤面
    pub board: Board,
    /// 連鎖数
    pub chain_count: u32,
    /// スコア
    pub score: f64,
}

/// 連鎖生成のUI状態
#[derive(Clone, Debug)]
pub struct ChainPlayState {
    /// ベース盤面
    pub base_board: Board,
    /// 生成モード
    pub mode: ChainPlayMode,
    /// 設定
    pub config: ChainPlayConfig,
    /// 状態
    pub status: GenerationStatus,
    /// 候補一覧
    pub candidates: Vec<ChainCandidate>,
    /// 選択中の候補インデックス
    pub selected_index: Option<usize>,
    /// 追加ペア数（頭伸ばしモード用）
    pub additional_pairs: usize,
}

impl ChainPlayState {
    pub fn new() -> Self {
        Self {
            base_board: Board::new(),
            mode: ChainPlayMode::default(),
            config: ChainPlayConfig::default(),
            status: GenerationStatus::Idle,
            candidates: Vec::new(),
            selected_index: None,
            additional_pairs: 1,
        }
    }

    /// 生成を開始
    pub fn start_generation(&mut self) {
        self.status = GenerationStatus::Running;
        self.candidates.clear();
        self.selected_index = None;
    }

    /// 生成を完了
    pub fn complete_generation(&mut self, candidates: Vec<ChainCandidate>) {
        let count = candidates.len();
        self.candidates = candidates;
        self.status = GenerationStatus::Completed { results_count: count };
        
        // 最初の候補を自動選択
        if !self.candidates.is_empty() {
            self.selected_index = Some(0);
        }
    }

    /// エラー発生
    pub fn set_error(&mut self, message: String) {
        self.status = GenerationStatus::Error(message);
    }

    /// モードを変更
    pub fn set_mode(&mut self, mode: ChainPlayMode) {
        if self.mode != mode {
            self.mode = mode;
            self.reset();
        }
    }

    /// 候補を選択
    pub fn select_candidate(&mut self, index: usize) {
        if index < self.candidates.len() {
            self.selected_index = Some(index);
        }
    }

    /// 選択中の候補を取得
    pub fn selected_candidate(&self) -> Option<&ChainCandidate> {
        self.selected_index
            .and_then(|idx| self.candidates.get(idx))
    }

    /// 次の候補を選択
    pub fn select_next(&mut self) {
        if let Some(idx) = self.selected_index {
            if idx + 1 < self.candidates.len() {
                self.selected_index = Some(idx + 1);
            }
        }
    }

    /// 前の候補を選択
    pub fn select_previous(&mut self) {
        if let Some(idx) = self.selected_index {
            if idx > 0 {
                self.selected_index = Some(idx - 1);
            }
        }
    }

    /// リセット
    pub fn reset(&mut self) {
        self.status = GenerationStatus::Idle;
        self.candidates.clear();
        self.selected_index = None;
    }

    /// 実行中かチェック
    pub fn is_running(&self) -> bool {
        matches!(self.status, GenerationStatus::Running)
    }

    /// 完了したかチェック
    pub fn is_completed(&self) -> bool {
        matches!(self.status, GenerationStatus::Completed { .. })
    }

    /// エラーかチェック
    pub fn is_error(&self) -> bool {
        matches!(self.status, GenerationStatus::Error(_))
    }

    /// 結果があるかチェック
    pub fn has_results(&self) -> bool {
        !self.candidates.is_empty()
    }
}

impl Default for ChainPlayState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_candidate(chain_count: u32) -> ChainCandidate {
        ChainCandidate {
            board: Board::new(),
            chain_count,
            score: chain_count as f64 * 10.0,
        }
    }

    #[test]
    fn new_state_is_idle() {
        let state = ChainPlayState::new();
        assert!(matches!(state.status, GenerationStatus::Idle));
        assert!(!state.is_running());
        assert!(!state.has_results());
    }

    #[test]
    fn start_generation_clears_candidates() {
        let mut state = ChainPlayState::new();
        state.candidates.push(test_candidate(5));
        
        state.start_generation();
        
        assert!(state.is_running());
        assert!(state.candidates.is_empty());
    }

    #[test]
    fn complete_generation_updates_state() {
        let mut state = ChainPlayState::new();
        let candidates = vec![test_candidate(5), test_candidate(6)];
        
        state.complete_generation(candidates);
        
        assert!(state.is_completed());
        assert_eq!(state.candidates.len(), 2);
        assert_eq!(state.selected_index, Some(0));
    }

    #[test]
    fn select_candidate_works() {
        let mut state = ChainPlayState::new();
        state.candidates = vec![test_candidate(5), test_candidate(6), test_candidate(7)];
        
        state.select_candidate(1);
        assert_eq!(state.selected_index, Some(1));
        
        let selected = state.selected_candidate();
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().chain_count, 6);
    }

    #[test]
    fn select_next_and_previous() {
        let mut state = ChainPlayState::new();
        state.candidates = vec![test_candidate(5), test_candidate(6), test_candidate(7)];
        state.selected_index = Some(1);
        
        state.select_next();
        assert_eq!(state.selected_index, Some(2));
        
        state.select_previous();
        assert_eq!(state.selected_index, Some(1));
    }

    #[test]
    fn set_mode_resets_state() {
        let mut state = ChainPlayState::new();
        state.candidates.push(test_candidate(5));
        state.status = GenerationStatus::Completed { results_count: 1 };
        
        state.set_mode(ChainPlayMode::Extend);
        
        assert!(matches!(state.status, GenerationStatus::Idle));
        assert!(state.candidates.is_empty());
    }

    #[test]
    fn reset_clears_everything() {
        let mut state = ChainPlayState::new();
        state.candidates = vec![test_candidate(5)];
        state.selected_index = Some(0);
        state.status = GenerationStatus::Completed { results_count: 1 };
        
        state.reset();
        
        assert!(matches!(state.status, GenerationStatus::Idle));
        assert!(state.candidates.is_empty());
        assert!(state.selected_index.is_none());
    }
}
