use std::collections::{HashMap, HashSet};

use crate::constants::{W, H};
use crate::vlog;

// デバッグログ制御フラグ
const LOG_CANDIDATE_DETAIL: bool = false;  // 各候補の詳細ログ（盤面表示など）
const LOG_CANDIDATE_SUMMARY: bool = false; // 各候補の個別ログ（採用/棄却）
const LOG_ONLY_ACCEPTED: bool = true;      // 採用された候補のみログ出力

// 色番号を漢字に変換
fn color_name(color: u8) -> &'static str {
    match color {
        0 => "赤",
        1 => "緑",
        2 => "青",
        3 => "黄",
        _ => "?",
    }
}
use super::grid::{Board, CellData, IterId, empty_board, apply_gravity, find_bottom_empty, get_connected_cells, cols_to_board, board_to_cols, in_range};
use super::detector::{Detector, RejectKind};
use super::generator::Generator;

#[derive(Clone)]
struct BeamState {
    field: Board,
    acc_chain: i32,
    acc_adds: Vec<Vec<CellData>>, // per column added cells (top-first)
    blocked_cols: HashSet<usize>,
    prev_adds: HashMap<(usize,usize), CellData>,
    iteration: u8,
}

fn extract_additions(merged: &Board, base: &Board) -> Vec<Vec<CellData>> {
    let mut adds: Vec<Vec<CellData>> = vec![Vec::new(); W];
    for x in 0..W {
        let mut col_merged: Vec<CellData> = (0..H).filter_map(|y| merged[y][x]).collect(); // bottom->top
        let mut col_base: Vec<CellData> = (0..H).filter_map(|y| base[y][x]).collect();   // bottom->top
        let lm = col_merged.len();
        let lb = col_base.len();
        if lm > lb {
            let diff = lm - lb;
            let start = lm - diff;
            adds[x].extend_from_slice(&col_merged[start..]); // take last diff (top-most in bottom->top order)
        }
    }
    adds
}

fn add_accumulated(accum: &Vec<Vec<CellData>>, new_add: &Vec<Vec<CellData>>) -> Vec<Vec<CellData>> {
    let mut out: Vec<Vec<CellData>> = vec![Vec::new(); W];
    for x in 0..W {
        let mut v = accum[x].clone();
        v.extend_from_slice(&new_add[x]); // append new additions at top (end)
        out[x] = v;
    }
    out
}

fn build_field_from_accum(original: &Board, accum: &Vec<Vec<CellData>>) -> Board {
    let mut merged = vec![vec![None; W]; H];
    for x in 0..W {
        let mut col: Vec<CellData> = (0..H).filter_map(|y| original[y][x]).collect(); // bottom->top base
        col.extend_from_slice(&accum[x]); // push additions to top
        for y in 0..H {
            merged[y][x] = if y < col.len() { Some(col[y]) } else { None };
        }
    }
    merged
}

fn simulate_chain_without_mapping(field: &Board) -> Board {
    let mut det = Detector::new(field.clone());
    let _ = det.simulate_chain_physical();
    det.field
}

fn try_extended_cleanup_arrangement(
    pre_chain_field: &Board,
    original_field: &Board,
    baseline_chain: i32,
    iteration: u8,
    blocked_columns: Option<&HashSet<usize>>,
    previous_additions: Option<&HashMap<(usize,usize), CellData>>,
) -> Vec<(i32, Board, u8)> {
    const LOG_BEAM_VERBOSE: bool = false;
    vlog!("  [拡張] 追撃探索開始: iter={}", iteration);
    let leftover_field = simulate_chain_without_mapping(pre_chain_field);

    let generator = Generator::new(leftover_field.clone()).with_full_column(true);

    // // 各列の最下段空セルに対する隣接色サマリ（1行のみ）
    // {
    //     let mut parts: Vec<String> = Vec::with_capacity(W);
    //     for col in 0..W {
    //         if let Some(ey) = find_bottom_empty(&leftover_field, col) {
    //             let dirs = [(1isize,0isize),(-1,0),(0,1),(0,-1)];
    //             let mut colors: Vec<u8> = Vec::new();
    //             for (dx,dy) in dirs {
    //                 let nx = col as isize + dx; let ny = ey as isize + dy;
    //                 if !in_range(nx, ny) { continue; }
    //                 let nxu = nx as usize; let nyu = ny as usize;
    //                 if let Some(c) = leftover_field[nyu][nxu] { colors.push(c.color); }
    //             }
    //             colors.sort_unstable();
    //             colors.dedup();
    //             if colors.is_empty() {
    //                 parts.push(format!("c{}:-", col));
    //             } else {
    //                 let list = colors.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    //                 parts.push(format!("c{}:{{{}}}", col, list));
    //             }
    //         } else {
    //             parts.push(format!("c{}:full", col));
    //         }
    //     }
    //     vlog!("  [拡張] 隣接色: {}", parts.join(" "));
    // }

    // Aグループ探索は常に「直前の連鎖が終わった後の盤面（leftover_field）」を用いる
    let base_for_seed = &leftover_field;

    // ぷよA候補の探索（隣接色からグループ取得）
    let mut candidate_groups: Vec<(i32, HashSet<(usize,usize)>, usize, u8, (usize,usize))> = Vec::new();
    let dirs = [(1isize,0isize),(-1,0),(0,1),(0,-1)];
    for col in 0..W {
        let Some(empty_y) = find_bottom_empty(base_for_seed, col) else { continue; };
        let mut neighbor_pos = Vec::new();
        for (dx,dy) in dirs {
            let nx = col as isize + dx; let ny = empty_y as isize + dy;
            if !in_range(nx, ny) { continue; }
            let nxu = nx as usize; let nyu = ny as usize;
            if base_for_seed[nyu][nxu].is_some() { neighbor_pos.push((nxu, nyu)); }
        }
        for (nx, ny) in neighbor_pos {
            let candidate_color = base_for_seed[ny][nx].unwrap().color;
            let mut group: HashSet<(usize,usize)> = get_connected_cells(base_for_seed, nx, ny).into_iter().collect();
            for (dx,dy) in dirs {
                let ax = col as isize + dx; let ay = empty_y as isize + dy;
                if !in_range(ax, ay) { continue; }
                let axu = ax as usize; let ayu = ay as usize;
                if let Some(c) = base_for_seed[ayu][axu] { 
                    if c.color == candidate_color {
                        for p in get_connected_cells(base_for_seed, axu, ayu) { group.insert(p); }
                    }
                }
            }
            let mut candidate_field = base_for_seed.clone();
            candidate_field[empty_y][col] = Some(CellData{ color: candidate_color, iter: IterId(iteration), original_pos: None });
            apply_gravity(&mut candidate_field);
            let mut det = Detector::new(candidate_field.clone());
            let chain_count = det.simulate_chain();
            candidate_groups.push((chain_count, group, col, candidate_color, (col, empty_y)));
        }
    }

    // vlog!("  [拡張] A候補数={}", candidate_groups.len());
    // A候補を連鎖スコア降順で試す
    let mut groups_sorted = candidate_groups;
    groups_sorted.sort_by(|a,b| b.0.cmp(&a.0));

    let mut successful_candidates: Vec<(i32, Board, u8)> = Vec::new();

    for (seed_chain, puyo_a_group, _seed_col, target_color, _seed_pos) in groups_sorted.into_iter() {
        vlog!("    [拡張] A対象色={} を試行", color_name(target_color));

        let candidates = generator.find_best_arrangement(
            Some(target_color),
            blocked_columns,
            None,
            previous_additions,
            iteration,
        );
        vlog!("    [拡張] 生成候補数={}", candidates.len());

        // 理由別の棄却カウンタ
        let mut cnt_zero_add: usize = 0;
        let mut cnt_c_collision: usize = 0;
        let mut cnt_first_multi: usize = 0;
        let mut cnt_no_latest: usize = 0;
        let mut cnt_mixed_iter: usize = 0;
        let mut cnt_other_reject: usize = 0;

        // 1連鎖目の禁止ルールを Leftover/PreChain で内訳
        let mut cnt_first_multi_L: usize = 0; // leftover+追加 側で検出
        let mut cnt_first_multi_P: usize = 0; // pre_chain+追加 側で検出
        let mut cnt_no_latest_L: usize = 0;   // leftover+追加 側で検出
        let mut cnt_no_latest_P: usize = 0;   // pre_chain+追加 側で検出

        let total_candidates = candidates.len();
        let mut color_accepted = false;
        'candidate_loop: for (cand_idx, (_candidate_chain, candidate_field, coords)) in candidates.into_iter().enumerate() {
            if LOG_CANDIDATE_DETAIL { vlog!("\n========== 候補 {}/{} の評価開始 ==========", cand_idx + 1, total_candidates); }
            // (1) leftover+追加 で連鎖を評価し、1連鎖目の禁止ルールをここで適用
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ1: leftover+追加 で評価", cand_idx + 1, total_candidates); }
            let mut det2 = Detector::new(candidate_field.clone());
            let new_chain_candidate = det2.simulate_chain();
            if new_chain_candidate == -1 {
                if let Some(r) = det2.take_last_reject() {
                    match r {
                        RejectKind::FirstChainMultipleGroups => { 
                            cnt_first_multi += 1; cnt_first_multi_L += 1; 
                            continue 'candidate_loop;
                        }
                        RejectKind::FirstChainNoLatestAdd => { 
                            cnt_no_latest += 1; cnt_no_latest_L += 1; 
                            continue 'candidate_loop;
                        }
                        RejectKind::MixedIterationInGroup => { 
                            cnt_mixed_iter += 1; 
                            continue 'candidate_loop;
                        }
                    }
                } else {
                    cnt_other_reject += 1;
                    continue 'candidate_loop;
                }
            }
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ1: 通過 (chain={})", cand_idx + 1, total_candidates, new_chain_candidate); }

            // 追加数カウント
            let mut additional_counts = vec![0usize; W];
            for x in 0..W {
                let cnt_left = (0..H).filter(|&y| leftover_field[y][x].is_some()).count();
                let cnt_cand = (0..H).filter(|&y| candidate_field[y][x].is_some()).count();
                additional_counts[x] = cnt_cand.saturating_sub(cnt_left);
            }
            let add_sum: usize = additional_counts.iter().sum();
            if add_sum == 0 {
                cnt_zero_add += 1;
                continue 'candidate_loop;
            }
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ2: 追加数チェック通過 (追加={}個)", cand_idx + 1, total_candidates, add_sum); }

            // pre_chain_field に candidate の追加分だけを合成
            let mut merged_field_candidate = pre_chain_field.clone();
            for x in 0..W {
                let mut base_col: Vec<CellData> = (0..H).filter_map(|y| merged_field_candidate[y][x]).collect(); // bottom->top
                let cand_col: Vec<CellData> = (0..H).filter_map(|y| candidate_field[y][x]).collect(); // bottom->top
                let take_n = additional_counts[x].min(cand_col.len());
                if take_n > 0 {
                    let start = cand_col.len() - take_n;
                    base_col.extend_from_slice(&cand_col[start..]); // append top-most additions
                }
                for y in 0..H {
                    merged_field_candidate[y][x] = if y < base_col.len() { Some(base_col[y]) } else { None };
                }
            }
            apply_gravity(&mut merged_field_candidate);

            // (2) pre_chain+追加 でも 1連鎖目の禁止ルール（複数同時消し/最新追加不関与）と
            //     iteration 整合（同一グループ内 異iteration混在）を確認する。
            
            if LOG_CANDIDATE_DETAIL { 
                vlog!("[候補{}/{}] ステップ3: pre_chain+追加 で評価", cand_idx + 1, total_candidates);
                // デバッグ: マージ後の盤面を出力
                vlog!("    [候補/DEBUG] マージ後の盤面 (pre_chain+追加):");
                for y in (0..H).rev() {
                    let mut line = format!("      y={}: ", y);
                    for x in 0..W {
                        if let Some(c) = merged_field_candidate[y][x] {
                            line.push_str(&format!("{}(i{})", color_name(c.color), c.iter.0));
                        } else {
                            line.push_str("  .  ");
                        }
                        line.push(' ');
                    }
                    vlog!("{}", line);
                }
            }
            
            let mut md = Detector::new(merged_field_candidate.clone());
            let merged_chain = md.simulate_chain();
            if merged_chain == -1 {
                if let Some(r) = md.take_last_reject() {
                    match r {
                        RejectKind::MixedIterationInGroup => { 
                            cnt_mixed_iter += 1; 
                            continue 'candidate_loop; 
                        }
                        RejectKind::FirstChainMultipleGroups => { 
                            cnt_first_multi += 1; cnt_first_multi_P += 1; 
                            continue 'candidate_loop; 
                        }
                        RejectKind::FirstChainNoLatestAdd => { 
                            cnt_no_latest += 1; cnt_no_latest_P += 1; 
                            continue 'candidate_loop; 
                        }
                    }
                } else {
                    // 未特定は安全側でスキップ
                    cnt_other_reject += 1;
                    continue 'candidate_loop;
                }
            }
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ3: 通過 (merged_chain={})", cand_idx + 1, total_candidates, merged_chain); }

            // current_new_additions_candidate の抽出と色統一
            let mut current_new: HashSet<(usize,usize)> = HashSet::new();
            for y in 0..H { for x in 0..W {
                if merged_field_candidate[y][x].is_some() && pre_chain_field[y][x].is_none() {
                    merged_field_candidate[y][x] = Some(CellData{ color: target_color, iter: IterId(iteration), original_pos: None });
                    current_new.insert((x,y));
                }
            }}

            let mut puyo_c: HashSet<(usize,usize)> = HashSet::new();
            for y in 0..H { for x in 0..W {
                if let Some(c) = merged_field_candidate[y][x] {
                    if c.color == target_color && !puyo_a_group.contains(&(x,y)) {
                        puyo_c.insert((x,y));
                    }
                }
            }}
            for p in &current_new { puyo_c.remove(p); }

            // 衝突判定（新規追加がCと隣接するか）
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ4: C隣接衝突チェック", cand_idx + 1, total_candidates); }
            for &(bx, by) in &current_new {
                const DIRS: [(isize,isize);4] = [(1,0),(-1,0),(0,1),(0,-1)];
                for (dx,dy) in DIRS {
                    let nx = bx as isize + dx; let ny = by as isize + dy;
                    if nx < 0 || ny < 0 { continue; }
                    let nxu = nx as usize; let nyu = ny as usize;
                    if nxu < W && nyu < H && puyo_c.contains(&(nxu,nyu)) {
                        cnt_c_collision += 1;
                        continue 'candidate_loop; // 次の候補へ
                    }
                }
            }
            if LOG_CANDIDATE_DETAIL { vlog!("[候補{}/{}] ステップ4: 通過", cand_idx + 1, total_candidates); }

            if LOG_ONLY_ACCEPTED || LOG_CANDIDATE_SUMMARY {
                vlog!("      ✅ 色={} 基点:({},{}) new_chain={}", 
                         color_name(target_color), coords.0, coords.1, new_chain_candidate);
            }
            if LOG_CANDIDATE_DETAIL { vlog!("========== 候補 {}/{} の評価終了（✅採用） ==========\n", cand_idx + 1, total_candidates); }
            successful_candidates.push((new_chain_candidate, merged_field_candidate, target_color));
            color_accepted = true;
            break 'candidate_loop; // この色では最初の成功候補のみ採用
        }

        // このA対象色では採用なし: サマリを出力
        if total_candidates > 0 && !color_accepted {
            vlog!(
                "      ❌ 色={}: 全{}件棄却 (複数同時消し={} / C隣接={} / 追加0={} / その他={})",
                color_name(target_color), total_candidates,
                cnt_first_multi, cnt_c_collision, cnt_zero_add,
                cnt_no_latest + cnt_mixed_iter + cnt_other_reject
            );
        }
    }

    if successful_candidates.is_empty() {
        vlog!("  [拡張] 採用候補なし");
    } else {
        vlog!("  [拡張] 採用候補数={}", successful_candidates.len());
    }
    successful_candidates
}

pub fn iterative_chain_clearing(
    original_field: &Board,
    base_chain: Option<i32>,
    beam_width: usize,
    max_depth: u8,
) -> (i32, Board) {
    let mut det0 = Detector::new(original_field.clone());
    let baseline_chain = base_chain.unwrap_or_else(|| det0.simulate_chain());
    vlog!(
        "[ビーム] 開始: baseline_chain={} / beam_width={} / max_depth={}",
        baseline_chain, beam_width, max_depth
    );

    let initial = BeamState {
        field: original_field.clone(),
        acc_chain: baseline_chain,
        acc_adds: vec![Vec::new(); W],
        blocked_cols: HashSet::new(),
        prev_adds: HashMap::new(),
        iteration: 0,
    };
    let mut beam = vec![initial.clone()];
    let mut best_state = initial;

    while !beam.is_empty() {
        let cur_depth = if beam.is_empty() { 0 } else { beam[0].iteration };
        vlog!("[ビーム] 深さ={} / ビーム幅={}", cur_depth, beam.len());
        if beam.iter().all(|s| s.iteration >= max_depth) { break; }
        let mut next_beam: Vec<BeamState> = Vec::new();
        let mut any_extended = false;
        for state in beam.into_iter() {
            vlog!(
                "  [展開] it={} → it+1={}, acc_chain={}, blocked_cols={}",
                state.iteration,
                state.iteration + 1,
                state.acc_chain,
                state.blocked_cols.len()
            );
            if state.iteration >= max_depth { next_beam.push(state); continue; }
            let it = state.iteration + 1;
            let candidates = try_extended_cleanup_arrangement(
                &state.field,
                original_field,
                baseline_chain,
                it,
                if it > 1 { Some(&state.blocked_cols) } else { None },
                Some(&state.prev_adds),
            );
            if candidates.is_empty() {
                vlog!("    [展開結果] it={} 候補なし（拡張失敗）", it);
                next_beam.push(state);
            } else {
                any_extended = true;
                for (new_chain, candidate, used_color) in candidates {
                    let new_add = extract_additions(&candidate, &state.field);
                    let accum = add_accumulated(&state.acc_adds, &new_add);
                    let add_sum: usize = new_add.iter().map(|v| v.len()).sum();
                    let add_cols: Vec<_> = new_add.iter().enumerate().filter(|(_, v)| !v.is_empty()).map(|(i, _)| i).collect();
                    vlog!("    [展開結果] it={} 追加={}個 / 列={:?} / 色={} / new_chain={}", 
                             it, add_sum, add_cols, color_name(used_color), new_chain);
                    let mut blk = state.blocked_cols.clone();
                    if it == 1 {
                        for (col_idx, adds) in new_add.iter().enumerate() { if !adds.is_empty() { blk.insert(col_idx); } }
                    }
                    let mut new_prev = HashMap::new();
                    for y in 0..H { for x in 0..W {
                        if candidate[y][x].is_some() && state.field[y][x].is_none() {
                            new_prev.insert((x,y), candidate[y][x].unwrap());
                        }
                    }}
                    let mut merged_prev = state.prev_adds.clone();
                    merged_prev.extend(new_prev.into_iter());
                    let next_field = build_field_from_accum(original_field, &accum);
                    next_beam.push(BeamState{
                        field: next_field,
                        acc_chain: state.acc_chain + new_chain,
                        acc_adds: accum,
                        blocked_cols: blk,
                        prev_adds: merged_prev,
                        iteration: it,
                    });
                }
            }
        }
        if !any_extended {
            vlog!("[ビーム] 追加配置が一度も発生せず、探索終了");
            break;
        }
        next_beam.sort_by(|a,b| b.acc_chain.cmp(&a.acc_chain));
        let mut chains: Vec<i32> = next_beam.iter().map(|s| s.acc_chain).collect();
        chains.sort_unstable_by(|a,b| b.cmp(a));
        vlog!("[ビーム] 次ビーム候補（上位）: {:?}", &chains[..chains.len().min(5)]);
        beam = next_beam.into_iter().take(beam_width.max(1)).collect();
        for s in &beam {
            if s.acc_chain > best_state.acc_chain { best_state = s.clone(); }
        }
    }

    (best_state.acc_chain, best_state.field)
}

/// 実盤面(cols)から、Python右盤面相当の最良配置を求め、目標盤面として返す
pub fn compute_target_from_actual_with_params(
    cols: &[[u16; W]; 4],
    beam_width: usize,
    max_depth: u8,
) -> ([[u16; W]; 4], i32) {
    let board = cols_to_board(cols);
    let (chain, result) = iterative_chain_clearing(&board, None, beam_width, max_depth);
    let target = board_to_cols(&result);
    (target, chain)
}

pub fn compute_target_from_actual(cols: &[[u16; W]; 4]) -> ([[u16; W]; 4], i32) {
    compute_target_from_actual_with_params(cols, 3, 3)
}
