// 検索設定のValue Objects

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// 連鎖数を表すValue Object
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChainCount(u32);

impl ChainCount {
    pub fn new(count: u32) -> Result<Self> {
        if count == 0 {
            return Err(anyhow!("連鎖数は1以上である必要があります"));
        }
        if count > 20 {
            return Err(anyhow!("連鎖数が大きすぎます: {}", count));
        }
        Ok(Self(count))
    }

    pub fn get(&self) -> u32 {
        self.0
    }
}

/// キャッシュサイズを表すValue Object（千単位）
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CacheSize(usize);

impl CacheSize {
    pub fn new_in_thousands(k: u32) -> Result<Self> {
        if k == 0 {
            return Err(anyhow!("キャッシュサイズは1以上"));
        }
        if k > 10000 {
            return Err(anyhow!("キャッシュサイズが大きすぎます"));
        }
        Ok(Self((k as usize) * 1000))
    }

    pub fn get(&self) -> usize {
        self.0
    }
}

/// 比率を表すValue Object (0.0 ~ 1.0)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Ratio(f32);

impl Ratio {
    pub fn new(value: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(anyhow!("比率は0.0~1.0の範囲"));
        }
        Ok(Self(value))
    }

    pub fn get(&self) -> f32 {
        self.0
    }

    pub fn zero() -> Self {
        Self(0.0)
    }
}

/// 検索設定のValue Object
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchConfig {
    pub threshold: ChainCount,
    pub lru_limit: CacheSize,
    pub exact_four_only: bool,
    pub stop_progress_plateau: Ratio,
    pub profile_enabled: bool,
}

impl SearchConfig {
    pub fn validate(&self) -> Result<()> {
        // Value Objectsで既に検証済み
        Ok(())
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            threshold: ChainCount::new(7).unwrap(),
            lru_limit: CacheSize::new_in_thousands(300).unwrap(),
            exact_four_only: false,
            stop_progress_plateau: Ratio::zero(),
            profile_enabled: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chain_count_rejects_zero() {
        assert!(ChainCount::new(0).is_err());
    }

    #[test]
    fn chain_count_accepts_valid() {
        assert!(ChainCount::new(7).is_ok());
        assert_eq!(ChainCount::new(7).unwrap().get(), 7);
    }

    #[test]
    fn chain_count_rejects_too_large() {
        assert!(ChainCount::new(21).is_err());
    }

    #[test]
    fn ratio_rejects_out_of_range() {
        assert!(Ratio::new(-0.1).is_err());
        assert!(Ratio::new(1.1).is_err());
    }

    #[test]
    fn ratio_accepts_valid() {
        assert!(Ratio::new(0.0).is_ok());
        assert!(Ratio::new(0.5).is_ok());
        assert!(Ratio::new(1.0).is_ok());
    }

    #[test]
    fn cache_size_rejects_zero() {
        assert!(CacheSize::new_in_thousands(0).is_err());
    }

    #[test]
    fn cache_size_accepts_valid() {
        let cache = CacheSize::new_in_thousands(300).unwrap();
        assert_eq!(cache.get(), 300000);
    }
}
