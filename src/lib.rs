// ぷよぷよ連鎖形総当たり - ライブラリモジュール

pub mod constants;
pub mod domain;         // ドメイン層
pub mod application;    // アプリケーション層
pub mod infrastructure; // インフラ層
pub mod presentation;   // プレゼンテーション層
pub mod model;
pub mod profiling;
pub mod search;
pub mod app;
pub mod chain;
pub mod logging;

// 外部クレートの再エクスポート
pub use anyhow::{anyhow, Context, Result};
pub use num_bigint::BigUint;
pub use num_traits::{One, ToPrimitive, Zero};

// 主要な型を再エクスポート
pub use app::{App, Message, Stats};
pub use constants::{H, MASK14, W};
pub use model::{Cell, TCell};
