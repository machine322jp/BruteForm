// ペア配置ロジック

use super::animation::Animation;
use super::state_manager::StateManager;
use crate::app::chain_play::{ChainPlay, Orient};
use crate::constants::{H, MASK14, W};

/// ペア配置のユーティリティ
pub struct PiecePlacer;

impl PiecePlacer {
    /// 列の高さを取得
    pub fn col_height(cols: &[[u16; W]; 4], x: usize) -> usize {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) & MASK14;
        occ.count_ones() as usize
    }

    /// セルに色を設定
    fn set_cell(cols: &mut [[u16; W]; 4], x: usize, y: usize, color: u8) {
        if x >= W || y >= H {
            return;
        }
        let bit = 1u16 << y;
        let c = (color as usize).min(3);
        cols[c][x] |= bit;
    }

    /// 指定位置にペアを配置
    pub fn place_with(cp: &mut ChainPlay, x: usize, orient: Orient, pair: (u8, u8)) {
        match orient {
            Orient::Up => {
                let h = Self::col_height(&cp.cols, x);
                Self::set_cell(&mut cp.cols, x, h, pair.0);
                Self::set_cell(&mut cp.cols, x, h + 1, pair.1);
            }
            Orient::Down => {
                let h = Self::col_height(&cp.cols, x);
                Self::set_cell(&mut cp.cols, x, h, pair.1);
                Self::set_cell(&mut cp.cols, x, h + 1, pair.0);
            }
            Orient::Right => {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x + 1);
                Self::set_cell(&mut cp.cols, x, h0, pair.0);
                Self::set_cell(&mut cp.cols, x + 1, h1, pair.1);
            }
            Orient::Left => {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x - 1);
                Self::set_cell(&mut cp.cols, x, h0, pair.0);
                Self::set_cell(&mut cp.cols, x - 1, h1, pair.1);
            }
        }
        if !cp.pair_seq.is_empty() {
            cp.pair_index = (cp.pair_index + 1) % cp.pair_seq.len();
        }
    }

    /// ランダム配置
    pub fn place_random(cp: &mut ChainPlay) -> Result<(), String> {
        if cp.lock || cp.anim.is_some() {
            return Err("操作がロックされています".to_string());
        }

        let pair = StateManager::current_pair(cp);
        let mut rng = rand::thread_rng();

        let mut moves: Vec<(usize, Orient)> = Vec::new();
        for x in 0..W {
            let h = Self::col_height(&cp.cols, x);
            if h + 1 < H {
                moves.push((x, Orient::Up));
                moves.push((x, Orient::Down));
            }
            if x + 1 < W {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x + 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Right));
                }
            }
            if x >= 1 {
                let h0 = Self::col_height(&cp.cols, x);
                let h1 = Self::col_height(&cp.cols, x - 1);
                if h0 < H && h1 < H {
                    moves.push((x, Orient::Left));
                }
            }
        }
        if moves.is_empty() {
            return Err("置ける場所がありません".to_string());
        }
        let (x, orient) = moves[rand::Rng::gen_range(&mut rng, 0..moves.len())];

        Self::place_with(cp, x, orient, pair);
        Animation::check_and_start_chain(cp);

        // 手を打った後に状態を保存
        StateManager::save_state(cp);

        Ok(())
    }
}
