use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// グローバルな詳細ログフラグ
pub static VERBOSE_LOGGING: AtomicBool = AtomicBool::new(false);

/// ログファイルのグローバルハンドル
static LOG_FILE: Mutex<Option<std::fs::File>> = Mutex::new(None);

/// ログファイルを初期化する
pub fn init_log_file(path: &str) -> std::io::Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)?;

    let mut log_file = LOG_FILE.lock().unwrap();
    *log_file = Some(file);
    Ok(())
}

/// ログをファイルに書き込む
pub fn write_log(message: String) {
    if let Ok(mut log_file) = LOG_FILE.lock() {
        if let Some(ref mut file) = *log_file {
            let _ = writeln!(file, "{}", message);
            let _ = file.flush(); // 即座にフラッシュして確実に書き込む
        }
    }
}

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

/// 詳細ログ出力マクロ（ファイル出力）
#[macro_export]
macro_rules! vlog {
    ($($arg:tt)*) => {
        if $crate::logging::is_verbose() {
            let message = format!($($arg)*);
            $crate::logging::write_log(message);
        }
    };
}
