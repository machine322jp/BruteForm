// インフラ層 - 外部システムとの接続、技術的実装

pub mod storage;
pub mod executor;

pub use storage::ResultWriter;
pub use executor::ParallelExecutor;
