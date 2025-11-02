// 連鎖生成サービス

use anyhow::{Result, Context};
use crate::domain::board::Board;
use crate::domain::chain::ChainResult;

/// 連鎖生成の設定
#[derive(Clone, Debug)]
pub struct ChainPlayConfig {
    /// ビーム幅
    pub beam_width: usize,
    /// 最大深さ
    pub max_depth: usize,
    /// 目標連鎖数
    pub target_chains: u32,
}

impl Default for ChainPlayConfig {
    fn default() -> Self {
        Self {
            beam_width: 100,
            max_depth: 10,
            target_chains: 5,
        }
    }
}

/// 連鎖生成結果
#[derive(Clone, Debug)]
pub struct ChainPlayResult {
    /// 生成された盤面
    pub board: Board,
    /// 連鎖結果
    pub chain: ChainResult,
    /// スコア（評価値）
    pub score: f64,
}

/// 連鎖生成を管理するサービス
pub struct ChainPlayService {
    config: ChainPlayConfig,
}

impl ChainPlayService {
    pub fn new(config: ChainPlayConfig) -> Self {
        Self { config }
    }

    /// 設定の検証
    fn validate_config(&self) -> Result<()> {
        if self.config.beam_width == 0 {
            return Err(anyhow::anyhow!("ビーム幅は1以上である必要があります"));
        }
        if self.config.max_depth == 0 {
            return Err(anyhow::anyhow!("最大深さは1以上である必要があります"));
        }
        if self.config.target_chains == 0 {
            return Err(anyhow::anyhow!("目標連鎖数は1以上である必要があります"));
        }
        Ok(())
    }

    /// ベース盤面から連鎖を生成
    pub fn generate_chain(
        &self,
        base_board: &Board,
    ) -> Result<Vec<ChainPlayResult>> {
        // 設定の検証
        self.validate_config().context("設定が不正です")?;

        // 盤面の検証
        base_board.validate().context("ベース盤面が不正です")?;

        // TODO: 実際のビーム探索実装
        // 現時点では空の結果を返す
        Ok(Vec::new())
    }

    /// 頭伸ばし（既存の盤面に追加ぷよを配置）
    pub fn extend_chain(
        &self,
        base_board: &Board,
        additional_pairs: usize,
    ) -> Result<Vec<ChainPlayResult>> {
        // 設定の検証
        self.validate_config().context("設定が不正です")?;

        // 盤面の検証
        base_board.validate().context("ベース盤面が不正です")?;

        if additional_pairs == 0 {
            return Err(anyhow::anyhow!("追加ペア数は1以上である必要があります"));
        }

        // TODO: 実際の頭伸ばし実装
        Ok(Vec::new())
    }

    /// ビーム探索による最適解探索
    pub fn beam_search(
        &self,
        base_board: &Board,
    ) -> Result<Option<ChainPlayResult>> {
        // 設定の検証
        self.validate_config().context("設定が不正です")?;

        // TODO: 実際のビーム探索実装
        Ok(None)
    }

    /// 設定を更新
    pub fn update_config(&mut self, config: ChainPlayConfig) {
        self.config = config;
    }

    /// 現在の設定を取得
    pub fn config(&self) -> &ChainPlayConfig {
        &self.config
    }
}

impl Default for ChainPlayService {
    fn default() -> Self {
        Self::new(ChainPlayConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_service_has_default_config() {
        let service = ChainPlayService::default();
        assert_eq!(service.config().beam_width, 100);
        assert_eq!(service.config().max_depth, 10);
        assert_eq!(service.config().target_chains, 5);
    }

    #[test]
    fn validate_config_rejects_zero_beam_width() {
        let config = ChainPlayConfig {
            beam_width: 0,
            max_depth: 10,
            target_chains: 5,
        };
        let service = ChainPlayService::new(config);
        assert!(service.validate_config().is_err());
    }

    #[test]
    fn validate_config_rejects_zero_max_depth() {
        let config = ChainPlayConfig {
            beam_width: 100,
            max_depth: 0,
            target_chains: 5,
        };
        let service = ChainPlayService::new(config);
        assert!(service.validate_config().is_err());
    }

    #[test]
    fn validate_config_accepts_valid() {
        let service = ChainPlayService::default();
        assert!(service.validate_config().is_ok());
    }

    #[test]
    fn can_update_config() {
        let mut service = ChainPlayService::default();
        let new_config = ChainPlayConfig {
            beam_width: 200,
            max_depth: 15,
            target_chains: 7,
        };

        service.update_config(new_config.clone());
        assert_eq!(service.config().beam_width, 200);
        assert_eq!(service.config().max_depth, 15);
        assert_eq!(service.config().target_chains, 7);
    }

    #[test]
    fn generate_chain_validates_board() {
        let service = ChainPlayService::default();
        let mut board = Board::new();
        // 不正な盤面（隣接するAbsセル）
        board.set(0, 0, crate::domain::board::Cell::Abs(5)).unwrap();
        board.set(1, 0, crate::domain::board::Cell::Abs(5)).unwrap();

        let result = service.generate_chain(&board);
        assert!(result.is_err());
    }

    #[test]
    fn extend_chain_rejects_zero_pairs() {
        let service = ChainPlayService::default();
        let board = Board::new();

        let result = service.extend_chain(&board, 0);
        assert!(result.is_err());
    }
}
