use std::sync::atomic::{AtomicBool, Ordering};

/// グローバルな詳細ログフラグ
pub static VERBOSE_LOGGING: AtomicBool = AtomicBool::new(false);

/// 詳細ログを有効にする
pub fn enable_verbose_logging() {
    VERBOSE_LOGGING.store(true, Ordering::Relaxed);
}

/// 詳細ログを無効にする
pub fn disable_verbose_logging() {
    VERBOSE_LOGGING.store(false, Ordering::Relaxed);
}

/// 詳細ログが有効かチェック
pub fn is_verbose() -> bool {
    VERBOSE_LOGGING.load(Ordering::Relaxed)
}

/// 詳細ログ出力マクロ
#[macro_export]
macro_rules! vlog {
    ($($arg:tt)*) => {
        if $crate::logging::is_verbose() {
            println!($($arg)*);
        }
    };
}
