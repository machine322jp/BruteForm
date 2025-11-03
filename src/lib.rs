// ぷよぷよ連鎖形総当たり - ライブラリモジュール

// === クリーンアーキテクチャ層 ===
pub mod domain;         // ドメイン層: ビジネスロジックの核心（Board, Cell, ChainResult等）
pub mod application;    // アプリケーション層: ユースケース（BruteforceService, ChainPlayService）
pub mod infrastructure; // インフラ層: 技術的実装（並列実行、ファイル書き込み）
pub mod presentation;   // プレゼンテーション層: UI状態管理（BruteforceState, ChainPlayState）

// === UI層 ===
pub mod app;      // UI層: eframe/eguiアプリケーションとUI描画（presentation層が状態管理を担当）

// === ユーティリティ ===
pub mod constants; // 盤面定数とグローバル型定義
pub mod profiling; // パフォーマンス計測
pub mod logging;   // ロギング機能

// 外部クレートの再エクスポート
pub use anyhow::{anyhow, Context, Result};
pub use num_bigint::BigUint;
pub use num_traits::{One, ToPrimitive, Zero};

// 主要な型を再エクスポート
pub use app::{App, Message, Stats};
pub use constants::{H, MASK14, W};
pub use domain::board::{Cell, TCell};
