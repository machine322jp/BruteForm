// ぷよぷよ連鎖形総当たり - ライブラリモジュール

pub mod constants;
pub mod model;
pub mod profiling;
pub mod search;
pub mod app;

// 外部クレートの再エクスポート
pub use anyhow::{Result, anyhow, Context};
pub use num_bigint::BigUint;
pub use num_traits::{One, Zero, ToPrimitive};

// 主要な型を再エクスポート
pub use constants::{W, H, MASK14};
pub use model::{Cell, TCell};
pub use app::{App, Stats, Message};
