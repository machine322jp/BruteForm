// モデル層（レガシー - ドメイン層へ移行中）

pub mod cell;

// ドメイン層のCell型を再エクスポート（後方互換性）
pub use crate::domain::board::{Cell, TCell};
pub use cell::{cell_style, cycle_abs, cycle_any, cycle_fixed};
