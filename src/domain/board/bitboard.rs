// ビットボード処理

use crate::constants::{H, MASK14, W};

/// ビットボード型（6×14=84マスを u128 にパック）
pub type BB = u128;

const COL_BITS: usize = H; // 14

// 各端マスク（列境界/上下境界の越境を防ぐ）
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

const TOP_MASK: BB = top_mask();
const BOTTOM_MASK: BB = bottom_mask();
const LEFTCOL_MASK: BB = (1u128 << COL_BITS) - 1;
const RIGHTCOL_MASK: BB = ((1u128 << COL_BITS) - 1) << ((W - 1) * COL_BITS);

const fn board_mask() -> BB {
    let mut m: BB = 0;
    let mut x = 0;
    while x < W {
        m |= ((1u128 << COL_BITS) - 1) << (x * COL_BITS);
        x += 1;
    }
    m
}

#[allow(dead_code)]
const BOARD_MASK: BB = board_mask();

/// 列形式から BB 形式へパック
#[inline(always)]
pub fn pack_cols(cols: &[[u16; W]; 4]) -> [BB; 4] {
    let mut out = [0u128; 4];
    for c in 0..4 {
        let mut acc: BB = 0;
        for x in 0..W {
            acc |= (cols[c][x] as BB) << (x * COL_BITS);
        }
        out[c] = acc;
    }
    out
}

/// BB から列形式へアンパック
#[inline(always)]
pub fn unpack_mask_to_cols(mask: BB) -> [u16; W] {
    let mut out = [0u16; W];
    for (x, o) in out.iter_mut().enumerate() {
        *o = ((mask >> (x * COL_BITS)) as u16) & MASK14;
    }
    out
}

/// 隣接ビット取得
#[inline(always)]
fn neighbors(bits: BB) -> BB {
    let v_up = (bits & !TOP_MASK) << 1;
    let v_down = (bits & !BOTTOM_MASK) >> 1;
    let h_left = (bits & !LEFTCOL_MASK) >> COL_BITS;
    let h_right = (bits & !RIGHTCOL_MASK) << COL_BITS;
    v_up | v_down | h_left | h_right
}

/// 落下処理（スカラ版）
#[inline(always)]
pub fn fall_cols(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] & MASK14;
        let c1 = cols_in[1][x] & MASK14;
        let c2 = cols_in[2][x] & MASK14;
        let c3 = cols_in[3][x] & MASK14;
        let mut occ = c0 | c1 | c2 | c3;

        let mut dst: usize = 0;
        while occ != 0 {
            let bit = occ & occ.wrapping_neg();
            let color = if (c0 & bit) != 0 {
                0
            } else if (c1 & bit) != 0 {
                1
            } else if (c2 & bit) != 0 {
                2
            } else {
                3
            };
            out[color][x] |= 1u16 << dst;
            dst += 1;
            occ &= occ - 1;
        }
    }

    out
}

/// 落下処理（BMI2最適化版）
#[inline(always)]
pub fn fall_cols_fast(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("bmi2") {
            unsafe {
                return fall_cols_bmi2(cols_in);
            }
        }
    }
    fall_cols(cols_in)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "bmi2")]
unsafe fn fall_cols_bmi2(cols_in: &[[u16; W]; 4]) -> [[u16; W]; 4] {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{_pdep_u32, _pext_u32};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{_pdep_u32, _pext_u32};

    let mut out = [[0u16; W]; 4];

    for x in 0..W {
        let c0 = cols_in[0][x] as u32;
        let c1 = cols_in[1][x] as u32;
        let c2 = cols_in[2][x] as u32;
        let c3 = cols_in[3][x] as u32;

        let occ = (c0 | c1 | c2 | c3) & (MASK14 as u32);
        let k = occ.count_ones();
        if k == 0 {
            continue;
        }
        let base = (1u32 << k) - 1;

        let s0 = _pext_u32(c0, occ);
        let s1 = _pext_u32(c1, occ);
        let s2 = _pext_u32(c2, occ);
        let s3 = _pext_u32(c3, occ);

        out[0][x] = _pdep_u32(s0, base) as u16;
        out[1][x] = _pdep_u32(s1, base) as u16;
        out[2][x] = _pdep_u32(s2, base) as u16;
        out[3][x] = _pdep_u32(s3, base) as u16;
    }

    out
}

/// 4連結消去マスクを計算
#[inline(always)]
pub fn compute_erase_mask_cols(cols: &[[u16; W]; 4]) -> [u16; W] {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }

        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
            }
            s &= !comp;
        }
    }

    unpack_mask_to_cols(clear_all)
}

/// 4個消しモード用：消去マスク + フラグ
#[inline(always)]
pub fn compute_erase_mask_and_flags(cols: &[[u16; W]; 4]) -> ([u16; W], bool, bool) {
    let bb = pack_cols(cols);
    let mut clear_all: BB = 0;
    let mut had_ge5 = false;
    let mut had_four = false;

    for &mask in bb.iter() {
        if mask.count_ones() < 4 {
            continue;
        }
        let mut s = mask;
        while s != 0 {
            let seed = s & (!s + 1);
            let mut comp = seed;
            let mut frontier = seed;
            loop {
                let grow = neighbors(frontier) & mask & !comp;
                if grow == 0 {
                    break;
                }
                comp |= grow;
                frontier = grow;
            }
            let sz = comp.count_ones();
            if sz >= 4 {
                clear_all |= comp;
                if sz == 4 {
                    had_four = true;
                } else {
                    had_ge5 = true;
                }
            }
            s &= !comp;
        }
    }

    (unpack_mask_to_cols(clear_all), had_ge5, had_four)
}

/// 消去せずにマスククリア
#[inline(always)]
pub fn apply_clear_no_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut out = [[0u16; W]; 4];
    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (out_col, pre_col) in out.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(out_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    out
}

/// 消去して落下
#[inline(always)]
pub fn apply_given_clear_and_fall(pre: &[[u16; W]; 4], clear: &[u16; W]) -> [[u16; W]; 4] {
    let mut next = [[0u16; W]; 4];

    let mut keep = [0u16; W];
    for (x, k) in keep.iter_mut().enumerate() {
        *k = (!clear[x]) & MASK14;
    }
    for (next_col, pre_col) in next.iter_mut().zip(pre.iter()) {
        for (&k, (cell, &p)) in keep.iter().zip(next_col.iter_mut().zip(pre_col.iter())) {
            *cell = p & k;
        }
    }
    fall_cols_fast(&next)
}

/// どの色か4つ以上あるか
#[inline(always)]
pub fn any_color_has_four(cols: &[[u16; W]; 4]) -> bool {
    let bb = pack_cols(cols);
    (bb[0].count_ones() >= 4)
        || (bb[1].count_ones() >= 4)
        || (bb[2].count_ones() >= 4)
        || (bb[3].count_ones() >= 4)
}

/// 消去と落下を適用
#[inline(always)]
pub fn apply_erase_and_fall_cols(cols: &[[u16; W]; 4]) -> (bool, [[u16; W]; 4]) {
    if !any_color_has_four(cols) {
        return (false, *cols);
    }

    let clear = compute_erase_mask_cols(cols);
    let any = (0..W).any(|x| clear[x] != 0);
    if !any {
        (false, *cols)
    } else {
        (true, apply_given_clear_and_fall(cols, &clear))
    }
}

/// 4個消しモード用の結果
pub enum StepExact {
    NoClear,
    Cleared([[u16; W]; 4]),
    Illegal,
}

/// 4個消しモード用の消去・落下
#[inline(always)]
pub fn apply_erase_and_fall_exact4(cols: &[[u16; W]; 4]) -> StepExact {
    if !any_color_has_four(cols) {
        return StepExact::NoClear;
    }
    let (clear, had_ge5, had_four) = compute_erase_mask_and_flags(cols);
    if had_ge5 {
        return StepExact::Illegal;
    }
    let any = (0..W).any(|x| clear[x] != 0);
    if !any || !had_four {
        StepExact::NoClear
    } else {
        StepExact::Cleared(apply_given_clear_and_fall(cols, &clear))
    }
}

/// 列への一括代入
#[inline(always)]
pub fn assign_col_unrolled(cols: &mut [[u16; W]; 4], x: usize, masks: &[u16; 4]) {
    debug_assert!(x < W);
    cols[0][x] = masks[0];
    cols[1][x] = masks[1];
    cols[2][x] = masks[2];
    cols[3][x] = masks[3];
}

/// 列のゼロクリア
#[inline(always)]
pub fn clear_col_unrolled(cols: &mut [[u16; W]; 4], x: usize) {
    debug_assert!(x < W);
    cols[0][x] = 0;
    cols[1][x] = 0;
    cols[2][x] = 0;
    cols[3][x] = 0;
}
