// 連鎖検出/生成ロジック（Python zenkesi.py の移植）

pub mod beam;
pub mod detector;
pub mod generator;
pub mod grid;

pub use beam::{
    compute_target_from_actual, compute_target_from_actual_with_params, iterative_chain_clearing,
};
pub use detector::{ChainStep, Detector};
pub use generator::Generator;
pub use grid::{apply_gravity, board_to_cols, cols_to_board, Board, CellData, IterId};
