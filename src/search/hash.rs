// ハッシュとシリアライズ関数

use crate::constants::{BOARD_MASK, H, MASK14, W};
use crate::search::board::BB;
use std::collections::HashMap;

/// LUT風：色マッピングの更新
#[inline(always)]
fn map_code_lut(entry: &mut u8, next: &mut u8) -> u64 {
    if *entry == u8::MAX {
        *entry = *next;
        *next = next.wrapping_add(1);
    }
    *entry as u64
}

/// 占有パターン比較でミラー判定
#[inline(always)]
pub fn choose_mirror_by_occupancy(cols: &[[u16; W]; 4]) -> Option<bool> {
    let mut packed: u128 = 0;
    let mut rev: u128 = 0;
    for x in 0..W {
        let occ = (cols[0][x] | cols[1][x] | cols[2][x] | cols[3][x]) as u128;
        packed |= occ << (x * 16);
        rev |= occ << ((W - 1 - x) * 16);
    }
    if packed < rev {
        Some(false)
    } else if packed > rev {
        Some(true)
    } else {
        None
    }
}

/// FNV-1a 64bit ハッシュ（占有ビット走査版）
#[inline(always)]
fn canonical_hash64_oriented_bits(cols: &[[u16; W]; 4], mirror: bool) -> u64 {
    const P: u64 = 1099511628211;
    const O: u64 = 14695981039346656037;

    let mut p_pow = [1u64; 15];
    for i in 1..15 {
        p_pow[i] = p_pow[i - 1].wrapping_mul(P);
    }
    #[inline(always)]
    fn mul_pow(h: u64, pp: &[u64; 15], k: usize) -> u64 {
        debug_assert!(k < 15);
        h.wrapping_mul(pp[k])
    }

    let mut h = O;
    let mut map: [u8; 4] = [u8::MAX; 4];
    let mut next: u8 = 1;

    if !mirror {
        for xi in 0..W {
            let c0 = cols[0][xi];
            let c1 = cols[1][xi];
            let c2 = cols[2][xi];
            let c3 = cols[3][xi];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            let mut prev_y: i32 = -1;

            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }

            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    } else {
        for xr in (0..W).rev() {
            let c0 = cols[0][xr];
            let c1 = cols[1][xr];
            let c2 = cols[2][xr];
            let c3 = cols[3][xr];
            let mut occ: u16 = (c0 | c1 | c2 | c3) as u16;

            if occ == 0 {
                h = mul_pow(h, &p_pow, 14);
                continue;
            }

            let mut prev_y: i32 = -1;

            while occ != 0 {
                let bit = occ & occ.wrapping_neg();
                let y = bit.trailing_zeros() as i32;

                let gap = (y - prev_y - 1) as usize;
                h = mul_pow(h, &p_pow, gap);

                let code = if (c0 & bit) != 0 {
                    map_code_lut(&mut map[0], &mut next)
                } else if (c1 & bit) != 0 {
                    map_code_lut(&mut map[1], &mut next)
                } else if (c2 & bit) != 0 {
                    map_code_lut(&mut map[2], &mut next)
                } else {
                    map_code_lut(&mut map[3], &mut next)
                };

                h ^= code;
                h = h.wrapping_mul(P);

                prev_y = y;
                occ &= occ - 1;
            }

            let tail = (14 - (prev_y as usize) - 1) as usize;
            h = mul_pow(h, &p_pow, tail);
        }
    }

    h
}

/// 正規化ハッシュ
#[inline(always)]
pub fn canonical_hash64_fast(cols: &[[u16; W]; 4]) -> (u64, bool) {
    if let Some(mirror) = choose_mirror_by_occupancy(cols) {
        let h = canonical_hash64_oriented_bits(cols, mirror);
        (h, mirror)
    } else {
        let h0 = canonical_hash64_oriented_bits(cols, false);
        let h1 = canonical_hash64_oriented_bits(cols, true);
        if h0 <= h1 {
            (h0, false)
        } else {
            (h1, true)
        }
    }
}

/// 正規化文字列エンコード
pub fn encode_canonical_string(cols: &[[u16; W]; 4], mirror: bool) -> String {
    let mut map: [u8; 4] = [0; 4];
    let mut next: u8 = b'A';
    let mut s = String::with_capacity(W * H);
    if !mirror {
        for x in 0..W {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    } else {
        for x in (0..W).rev() {
            for y in 0..H {
                let bit = 1u16 << y;
                let color = if cols[0][x] & bit != 0 {
                    0
                } else if cols[1][x] & bit != 0 {
                    1
                } else if cols[2][x] & bit != 0 {
                    2
                } else if cols[3][x] & bit != 0 {
                    3
                } else {
                    4
                };
                if color == 4 {
                    s.push('.');
                } else {
                    if map[color] == 0 {
                        map[color] = next;
                        next = next.wrapping_add(1);
                    }
                    s.push(map[color] as char);
                }
            }
        }
    }
    s
}

/// ボードをシリアライズ
pub fn serialize_board_from_cols(cols: &[[u16; W]; 4]) -> Vec<String> {
    let mut rows = Vec::with_capacity(H);
    for y in 0..H {
        let mut line = String::with_capacity(W);
        for x in 0..W {
            let bit = 1u16 << y;
            let ch = if cols[0][x] & bit != 0 {
                '0'
            } else if cols[1][x] & bit != 0 {
                '1'
            } else if cols[2][x] & bit != 0 {
                '2'
            } else if cols[3][x] & bit != 0 {
                '3'
            } else {
                '.'
            };
            line.push(ch);
        }
        rows.push(line);
    }
    rows
}

/// FNV-1a 32bit
pub fn fnv1a32(s: &str) -> u32 {
    let mut h: u32 = 0x811c9dc5;
    for &b in s.as_bytes() {
        h ^= b as u32;
        h = h
            .wrapping_add(h << 1)
            .wrapping_add(h << 4)
            .wrapping_add(h << 7)
            .wrapping_add(h << 8)
            .wrapping_add(h << 24);
    }
    h
}

/// JSON文字列生成
#[inline(always)]
pub fn make_json_line_str(
    key: &str,
    hash: u32,
    chains: u32,
    rows: &[String],
    mapping: &HashMap<char, u8>,
    mirror: bool,
) -> String {
    let mut s = String::with_capacity(256);
    s.push('{');
    s.push_str(r#""key":"#);
    s.push('"');
    for ch in key.chars() {
        if ch == '"' {
            s.push('\\');
        }
        s.push(ch);
    }
    s.push('"');
    s.push(',');

    s.push_str(r#""hash":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", hash));
    s.push(',');
    s.push_str(r#""chains":"#);
    let _ = std::fmt::write(&mut s, format_args!("{}", chains));
    s.push(',');

    s.push_str(r#""pre_chain_board":["#);
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('"');
        for ch in row.chars() {
            if ch == '"' {
                s.push('\\');
            }
            s.push(ch);
        }
        s.push('"');
    }
    s.push_str("],");

    s.push_str(r#""example_mapping":{"#);
    let mut keys: Vec<_> = mapping.keys().copied().collect();
    keys.sort_unstable();
    let mut first = true;
    for k in keys {
        if !first {
            s.push(',');
        }
        first = false;
        s.push('"');
        s.push(k);
        s.push_str(r#"":"#);
        let _ = std::fmt::write(&mut s, format_args!("{}", mapping[&k]));
    }
    s.push_str("},");

    s.push_str(r#""mirror":"#);
    s.push_str(if mirror { "true" } else { "false" });
    s.push('}');
    s
}
