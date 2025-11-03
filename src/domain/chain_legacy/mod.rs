// ドメイン層 - 連鎖検出・生成ロジック（chain/から移行）

pub mod beam;
pub mod detector;
pub mod generator;
pub mod grid;

pub use detector::{ChainStep, Detector};
pub use generator::Generator;
pub use grid::{apply_gravity, board_to_cols, cols_to_board, Board, CellData, IterId};
