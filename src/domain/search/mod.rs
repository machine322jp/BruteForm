// 検索関連のドメインモデル

pub mod config;
pub mod result;

pub use config::{CacheSize, ChainCount, Ratio, SearchConfig};
pub use result::{SearchResult, SearchSummary};
