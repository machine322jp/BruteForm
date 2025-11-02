// 枝刈りと到達判定

use crate::constants::{BOARD_MASK, H, MASK14, W};
use crate::search::board::{
    any_color_has_four, apply_erase_and_fall_cols, apply_erase_and_fall_exact4, fall_cols_fast,
    pack_cols, unpack_mask_to_cols, StepExact, BB,
};

const COL_BITS: usize = H;

/// 隣接ビット取得
#[inline(always)]
fn neighbors(bits: BB) -> BB {
    const TOP_MASK: BB = top_mask();
    const BOTTOM_MASK: BB = bottom_mask();
    const LEFTCOL_MASK: BB = (1u128 << COL_BITS) - 1;
    const RIGHTCOL_MASK: BB = ((1u128 << COL_BITS) - 1) << ((W - 1) * COL_BITS);

    let v_up = (bits & !TOP_MASK) << 1;
    let v_down = (bits & !BOTTOM_MASK) >> 1;
    let h_left = (bits & !LEFTCOL_MASK) >> COL_BITS;
    let h_right = (bits & !RIGHTCOL_MASK) << COL_BITS;
    v_up | v_down | h_left | h_right
}

const fn top_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS + (COL_BITS - 1));
        x += 1;
    }
    m
}

const fn bottom_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= 1u128 << (x * COL_BITS);
        x += 1;
    }
    m
}

/// E1単一連結 + T到達判定
#[inline(always)]
pub fn reaches_t_from_pre_single_e1(pre: &[[u16; W]; 4], t: u32, exact_four_only: bool) -> bool {
    if exact_four_only {
        let mut potential: u32 = 0;
        for col in pre.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < t {
            return false;
        }
    }

    let bb_pre = pack_cols(pre);
    let (clear_bb, total_cnt) = {
        let mut clr: BB = 0;
        let mut tot: u32 = 0;
        for &bb in bb_pre.iter() {
            if bb.count_ones() < 4 {
                continue;
            }
            let mut s = bb;
            while s != 0 {
                let seed = s & (!s + 1);
                let mut comp = seed;
                let mut frontier = seed;
                loop {
                    let grow = neighbors(frontier) & bb & !comp;
                    if grow == 0 {
                        break;
                    }
                    comp |= grow;
                    frontier = grow;
                }
                let sz = comp.count_ones();
                if sz >= 4 {
                    clr |= comp;
                    tot = tot.saturating_add(sz);
                }
                s &= !comp;
            }
        }
        (clr, tot)
    };

    let total = total_cnt;
    if total == 0 {
        return false;
    }

    if exact_four_only && total != 4 {
        return false;
    }

    let occ_bb = bb_pre[0] | bb_pre[1] | bb_pre[2] | bb_pre[3];
    let blank_bb = BOARD_MASK & !occ_bb;
    if neighbors(clear_bb) & blank_bb == 0 {
        return false;
    }

    let mut ok_overhang = false;
    for x in 0..W {
        let clear_col: u16 = ((clear_bb >> (x * COL_BITS)) as u16) & MASK14;
        if clear_col == 0 {
            continue;
        }
        let occ_col: u16 = ((occ_bb >> (x * COL_BITS)) as u16) & MASK14;

        let top_y = 15 - clear_col.leading_zeros() as usize;
        let above = (occ_col & !clear_col) >> (top_y + 1);
        let run = (above.trailing_ones()) as usize;

        if run <= 1 {
            ok_overhang = true;
            break;
        }
    }
    if !ok_overhang {
        return false;
    }

    // E1 単一連結チェック
    let seed = clear_bb & (!clear_bb + 1);
    let mut comp = seed;
    let mut frontier = seed;
    loop {
        let grow = neighbors(frontier) & clear_bb & !comp;
        if grow == 0 {
            break;
        }
        comp |= grow;
        frontier = grow;
    }
    if comp.count_ones() != total {
        return false;
    }

    let mut cur;
    {
        let mut work = [[0u16; W]; 4];
        let clear_cols = unpack_mask_to_cols(clear_bb);
        for x in 0..W {
            let inv = (!clear_cols[x]) & MASK14;
            work[0][x] = pre[0][x] & inv;
            work[1][x] = pre[1][x] & inv;
            work[2][x] = pre[2][x] & inv;
            work[3][x] = pre[3][x] & inv;
        }
        cur = fall_cols_fast(&work);
    }

    if t == 1 {
        return true;
    }

    {
        let mut potential: u32 = 0;
        for col in cur.iter() {
            let cnt: u32 = col.iter().map(|&m| m.count_ones()).sum();
            potential = potential.saturating_add(cnt / 4);
        }
        if potential < (t - 1) {
            return false;
        }
    }

    if !exact_four_only {
        for _ in 2..=t {
            let (erased, next) = apply_erase_and_fall_cols(&cur);
            if !erased {
                return false;
            }
            cur = next;
        }
        true
    } else {
        for _ in 2..=t {
            match apply_erase_and_fall_exact4(&cur) {
                StepExact::Illegal => return false,
                StepExact::NoClear => return false,
                StepExact::Cleared(next) => {
                    cur = next;
                }
            }
        }
        true
    }
}
