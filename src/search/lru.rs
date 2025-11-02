// LRUキャッシュ

use crate::constants::U64Map;
use nohash_hasher::BuildNoHashHasher;
use std::collections::VecDeque;

/// 近似LRU（ローカル専用）
pub struct ApproxLru {
    pub limit: usize,
    pub map: U64Map<bool>,
    pub q: VecDeque<u64>,
}

impl ApproxLru {
    pub fn new(limit: usize) -> Self {
        let cap = (limit.saturating_mul(11) / 10).max(16);
        let map: U64Map<bool> =
            std::collections::HashMap::with_capacity_and_hasher(cap, BuildNoHashHasher::default());
        let q = VecDeque::with_capacity(cap);
        Self { limit, map, q }
    }

    #[allow(dead_code)]
    pub fn get(&self, k: u64) -> Option<bool> {
        self.map.get(&k).copied()
    }

    #[allow(dead_code)]
    pub fn insert(&mut self, k: u64, v: bool) {
        use std::collections::hash_map::Entry;
        match self.map.entry(k) {
            Entry::Vacant(e) => {
                e.insert(v);
                self.q.push_back(k);
                let cap = (self.limit as f64 * 1.1) as usize;
                if self.q.len() > cap {
                    let to_delete = self.q.len() - self.limit;
                    for _ in 0..to_delete {
                        if let Some(kk) = self.q.pop_front() {
                            self.map.remove(&kk);
                        }
                    }
                }
            }
            Entry::Occupied(mut e) => {
                e.insert(v);
            }
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.map.len()
    }
}

/// 配列初期化ヘルパー
pub fn array_init<T, F: FnMut(usize) -> T, const N: usize>(mut f: F) -> [T; N] {
    use std::mem::MaybeUninit;
    let mut data: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };
    for (i, slot) in data.iter_mut().enumerate() {
        slot.write(f(i));
    }
    unsafe { std::mem::transmute_copy::<[MaybeUninit<T>; N], [T; N]>(&data) }
}
