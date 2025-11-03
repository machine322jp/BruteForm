// 総当たり検索サービス

use anyhow::{Result, Context, anyhow};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use crossbeam_channel::{unbounded, Receiver};

use crate::domain::board::Board;
use crate::domain::search::SearchConfig;
use crate::application::bruteforce::event::SearchEvent;
use crate::application::bruteforce::search::run_search;

/// 検索ハンドル（中断制御のみ）
pub struct SearchHandle {
    abort_flag: Arc<AtomicBool>,
}

impl SearchHandle {
    /// 検索を中断
    pub fn abort(&self) {
        self.abort_flag.store(true, Ordering::Relaxed);
    }

    /// 中断されたかチェック
    pub fn is_aborted(&self) -> bool {
        self.abort_flag.load(Ordering::Relaxed)
    }
}

/// 総当たり検索を管理するサービス
pub struct BruteforceService;

impl BruteforceService {
    /// 入力の検証
    fn validate_inputs(
        board: &Board,
        config: &SearchConfig,
        output_path: &Path,
    ) -> Result<()> {
        // 盤面の妥当性チェック
        board.validate().context("盤面が不正です")?;

        // 設定の妥当性チェック
        config.validate().context("検索設定が不正です")?;

        // 出力パスのディレクトリが存在するかチェック
        if let Some(parent) = output_path.parent() {
            if !parent.exists() {
                return Err(anyhow!(
                    "出力ディレクトリが存在しません: {}",
                    parent.display()
                ));
            }
        }

        Ok(())
    }

    /// 検索を開始（同期版・テスト用）
    pub fn start_search(
        board: &Board,
        config: SearchConfig,
        output_path: &Path,
    ) -> Result<SearchHandle> {
        // 事前検証
        Self::validate_inputs(board, &config, output_path)
            .context("入力の検証に失敗しました")?;

        // ハンドルを返す
        let abort_flag = Arc::new(AtomicBool::new(false));
        Ok(SearchHandle { abort_flag })
    }

    /// 検索を非同期で開始（UI用メインAPI）
    pub fn start_search_async(
        board: &Board,
        config: SearchConfig,
        output_path: PathBuf,
    ) -> Result<(SearchHandle, Receiver<SearchEvent>)> {
        // 1. 事前検証
        Self::validate_inputs(board, &config, &output_path)
            .context("入力の検証に失敗しました")?;

        // 2. Board を Vec<char> に変換
        let board_chars: Vec<char> = board.cells()
            .iter()
            .map(|cell| cell.label_char())
            .collect();

        // 3. イベントチャンネルを作成
        let (event_tx, event_rx) = unbounded::<SearchEvent>();
        
        // 4. 中断フラグを作成
        let abort_flag = Arc::new(AtomicBool::new(false));
        let abort_flag_clone = Arc::clone(&abort_flag);

        // 5. 検索スレッドを起動
        thread::spawn(move || {
            if let Err(e) = run_search(
                board_chars,
                &config,
                output_path,
                event_tx.clone(),
                abort_flag_clone,
            ) {
                let _ = event_tx.send(SearchEvent::Error(format!("{e:?}")));
            }
        });

        // 6. ハンドルとレシーバーを返す
        let handle = SearchHandle { abort_flag };
        
        Ok((handle, event_rx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::search::{ChainCount, CacheSize, Ratio};

    fn test_config() -> SearchConfig {
        SearchConfig {
            threshold: ChainCount::new(5).unwrap(),
            lru_limit: CacheSize::new_in_thousands(100).unwrap(),
            exact_four_only: false,
            stop_progress_plateau: Ratio::zero(),
            profile_enabled: false,
        }
    }

    #[test]
    fn validate_accepts_valid_board() {
        let board = Board::new();
        let config = test_config();
        
        // 一時ディレクトリを使用（有効なパス）
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_output.json");
        
        let result = BruteforceService::validate_inputs(&board, &config, &path);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_rejects_invalid_board() {
        let mut board = Board::new();
        // 隣接するAbsセルを配置（不正な盤面）
        board.set(0, 0, crate::domain::board::Cell::Abs(5)).unwrap();
        board.set(1, 0, crate::domain::board::Cell::Abs(5)).unwrap();

        let config = test_config();
        let path = Path::new(".");

        let result = BruteforceService::validate_inputs(&board, &config, path);
        assert!(result.is_err());
    }

    #[test]
    fn search_handle_can_abort() {
        let abort_flag = Arc::new(AtomicBool::new(false));
        let handle = SearchHandle {
            abort_flag,
        };

        assert!(!handle.is_aborted());
        handle.abort();
        assert!(handle.is_aborted());
    }

    #[test]
    fn start_search_returns_handle() {
        let board = Board::new();
        let config = test_config();
        
        // 一時ディレクトリに出力（テスト用）
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_output.jsonl");

        let result = BruteforceService::start_search(&board, config, &path);
        assert!(result.is_ok());
        let handle = result.unwrap();
        assert!(!handle.is_aborted());
    }
}
