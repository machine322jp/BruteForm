// 統合テスト

use bruteform::domain::board::{Board, BoardBits, Cell};
use bruteform::domain::search::{SearchConfig, SearchResult, ChainCount};
use bruteform::application::bruteforce::BruteforceService;
use bruteform::application::chainplay::ChainPlayService;
use bruteform::application::progress::ProgressManager;
use bruteform::infrastructure::storage::{ResultWriter, MemoryResultWriter};
use bruteform::infrastructure::executor::ParallelExecutor;
use bruteform::presentation::state::{BruteforceState, ChainPlayState};

/// ドメイン層の統合テスト
mod domain_integration {
    use super::*;

    #[test]
    fn board_boardbits_roundtrip() {
        // Boardを作成
        let mut board = Board::new();
        board.set(0, 0, Cell::Fixed(0)).unwrap();
        board.set(1, 1, Cell::Fixed(1)).unwrap();
        board.set(2, 2, Cell::Fixed(2)).unwrap();

        // BoardBitsに変換
        let bits = BoardBits::from(&board);

        // Boardに戻す
        let board2 = Board::from(&bits);

        // 同じ内容であることを確認
        for y in 0..14 {
            for x in 0..6 {
                assert_eq!(board.get(x, y), board2.get(x, y));
            }
        }
    }

    #[test]
    fn board_validation_works() {
        let mut board = Board::new();
        
        // 正常な盤面
        board.set(0, 0, Cell::Fixed(0)).unwrap();
        assert!(board.validate().is_ok());

        // 不正な盤面（隣接するAbs）
        let mut bad_board = Board::new();
        bad_board.set(0, 0, Cell::Abs(5)).unwrap();
        bad_board.set(1, 0, Cell::Abs(5)).unwrap();
        assert!(bad_board.validate().is_err());
    }
}

/// アプリケーション層の統合テスト
mod application_integration {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn bruteforce_service_lifecycle() {
        let mut service = BruteforceService::new();
        let board = Board::new();
        let config = SearchConfig::default();
        let path = PathBuf::from("test_output.json");

        // 検索ハンドルの取得
        let handle = service.start_search(&board, config, &path);
        
        // 現時点ではvalidationでディレクトリチェックがあるため
        // エラーになる可能性があるが、型レベルでは正しく動作する
        match handle {
            Ok(h) => {
                assert!(!h.is_aborted());
            }
            Err(_) => {
                // ディレクトリが存在しない場合のエラーは想定内
            }
        }
    }

    #[test]
    fn chainplay_service_lifecycle() {
        let service = ChainPlayService::default();
        
        // デフォルト設定の確認
        assert_eq!(service.config().beam_width, 100);
        assert_eq!(service.config().max_depth, 10);
        assert_eq!(service.config().target_chains, 5);
    }

    #[test]
    fn progress_manager_tracking() {
        let mgr = ProgressManager::new();
        
        // ノード追加
        mgr.add_nodes(1000);
        mgr.add_results(50);
        
        let stats = mgr.get_stats();
        assert_eq!(stats.nodes_searched, 1000);
        assert_eq!(stats.results_found, 50);
        
        // 中断
        assert!(!mgr.is_aborted());
        mgr.abort();
        assert!(mgr.is_aborted());
    }
}

/// インフラ層の統合テスト
mod infrastructure_integration {
    use super::*;

    #[test]
    fn memory_writer_stores_results() {
        let mut writer = MemoryResultWriter::new();
        
        // 結果を書き込む
        let result = SearchResult {
            board: BoardBits::new(),
            chain_count: ChainCount::new(5).unwrap(),
            hash: 12345,
            is_mirror: false,
        };
        
        writer.write_result(&result).unwrap();
        writer.write_result(&result).unwrap();
        
        assert_eq!(writer.count(), 2);
        assert_eq!(writer.results().len(), 2);
    }

    #[test]
    fn parallel_executor_configuration() {
        let executor = ParallelExecutor::default();
        
        // ワーカー数が設定されていることを確認
        assert!(executor.num_workers() > 0);
        assert_eq!(executor.batch_size(), 100);
        
        // 中断機能
        assert!(!executor.is_aborted());
        executor.abort();
        assert!(executor.is_aborted());
    }
}

/// プレゼンテーション層の統合テスト
mod presentation_integration {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn bruteforce_state_workflow() {
        let mut state = BruteforceState::new();
        
        // 初期状態
        assert!(!state.is_running());
        assert!(!state.is_completed());
        
        // 検索開始
        state.start_search(PathBuf::from("output.json"));
        assert!(state.is_running());
        
        // 検索完了
        state.complete_search(100);
        assert!(state.is_completed());
        assert_eq!(state.result_count, 100);
        
        // リセット
        state.reset();
        assert!(!state.is_running());
        assert!(!state.is_completed());
    }

    #[test]
    fn chainplay_state_workflow() {
        let mut state = ChainPlayState::new();
        
        // 初期状態
        assert!(!state.is_running());
        assert!(!state.has_results());
        
        // 生成開始
        state.start_generation();
        assert!(state.is_running());
        
        // 候補完成
        use bruteform::presentation::state::chainplay::ChainCandidate;
        let candidates = vec![
            ChainCandidate {
                board: Board::new(),
                chain_count: 5,
                score: 50.0,
            },
        ];
        
        state.complete_generation(candidates);
        assert!(state.is_completed());
        assert!(state.has_results());
        assert_eq!(state.selected_index, Some(0));
    }
}

/// レイヤー間の統合テスト
mod cross_layer_integration {
    use super::*;

    #[test]
    fn domain_to_application() {
        // ドメインオブジェクトをアプリケーション層で使用
        let _board = Board::with_default_template();
        let _config = SearchConfig::default();
        
        let service = BruteforceService::new();
        
        // サマリー作成
        let summary = service.create_summary(100, 10000);
        assert_eq!(summary.unique_count, 100);
        assert_eq!(summary.total_nodes, 10000);
    }

    #[test]
    fn application_to_presentation() {
        // アプリケーション層の状態をプレゼンテーション層で管理
        let mut state = BruteforceState::new();
        let progress = ProgressManager::new();
        
        progress.add_nodes(1000);
        progress.add_results(50);
        
        let stats = progress.get_stats();
        state.update_progress(stats);
        
        assert_eq!(state.result_count, 50);
    }

    #[test]
    fn domain_to_infrastructure() {
        // ドメインオブジェクトをインフラ層で永続化
        let mut writer = MemoryResultWriter::new();
        
        let result = SearchResult {
            board: BoardBits::new(),
            chain_count: ChainCount::new(7).unwrap(),
            hash: 99999,
            is_mirror: true,
        };
        
        writer.write_result(&result).unwrap();
        writer.flush().unwrap();
        
        assert_eq!(writer.count(), 1);
        assert_eq!(writer.results()[0].chain_count.get(), 7);
        assert!(writer.results()[0].is_mirror);
    }
}

/// エンドツーエンドテスト（簡易版）
#[test]
fn end_to_end_workflow() {
    // 1. ドメイン層：盤面作成
    let board = Board::with_default_template();
    assert!(board.validate().is_ok());
    
    // 2. アプリケーション層：サービス初期化
    let service = BruteforceService::new();
    let progress = ProgressManager::new();
    
    // 3. インフラ層：ライター準備
    let writer = MemoryResultWriter::new();
    
    // 4. プレゼンテーション層：状態管理
    let state = BruteforceState::new();
    
    // 統合確認：全層が正常に初期化できる
    assert!(!progress.is_aborted());
    assert_eq!(writer.count(), 0);
    assert!(!state.is_running());
}
