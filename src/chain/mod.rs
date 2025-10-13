// 連鎖検出/生成ロジック（Python zenkesi.py の移植）

pub mod grid;
pub mod detector;
pub mod generator;
pub mod beam;

pub use beam::{compute_target_from_actual, compute_target_from_actual_with_params, iterative_chain_clearing};
pub use detector::{ChainStep, Detector};
pub use grid::{Board, CellData, IterId, cols_to_board, board_to_cols, apply_gravity};
pub use generator::Generator;
