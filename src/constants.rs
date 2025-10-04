// 監面定数とユーティリティ型定義

use nohash_hasher::BuildNoHashHasher;
pub use dashmap::{DashMap, DashSet};

/// ====== 監面定数 ======
pub const W: usize = 6;
pub const H: usize = 14;
pub const MASK14: u16 = (1u16 << H) - 1;

// ビットボード用
pub type BB = u128;
const COL_BITS: usize = H;

const fn board_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= ((1u128 << COL_BITS) - 1) << (x * COL_BITS);
        x += 1;
    }
    m
}
pub const BOARD_MASK: BB = board_mask();

// u64 キー専用のノーハッシュ（高速化）
pub type U64Map<V> = std::collections::HashMap<u64, V, BuildNoHashHasher<u64>>;
pub type U64Set = std::collections::HashSet<u64, BuildNoHashHasher<u64>>;
pub type DU64Map<V> = DashMap<u64, V, BuildNoHashHasher<u64>>;
pub type DU64Set = DashSet<u64, BuildNoHashHasher<u64>>;
