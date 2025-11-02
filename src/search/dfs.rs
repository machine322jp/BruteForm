// DFS探索ロジック

use crate::app::{Message, StatDelta};
use crate::constants::{DU64Map, DU64Set, U64Set, H, W};
use crate::prof;
use crate::profiling::{time_delta_has_any, TimeDelta};
use crate::search::board::{any_color_has_four, assign_col_unrolled, clear_col_unrolled};
use crate::search::coloring::{stream_column_candidates, stream_column_candidates_timed, ColGen};
use crate::search::hash::{
    canonical_hash64_fast, encode_canonical_string, fnv1a32, make_json_line_str,
    serialize_board_from_cols,
};
use crate::search::lru::ApproxLru;
use crate::search::pruning::reaches_t_from_pre_single_e1;
use crossbeam_channel::Sender;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::{Duration, Instant};

#[allow(clippy::too_many_arguments)]
pub fn dfs_combine_parallel(
    depth: usize,
    cols0: &mut [[u16; W]; 4],
    gens: &[ColGen; W],
    order: &[usize],
    threshold: u32,
    exact_four_only: bool,
    _memo: &mut ApproxLru,
    local_output_once: &mut U64Set,
    global_output_once: &Arc<DU64Set>,
    _global_memo: &Arc<DU64Map<bool>>,
    map_label_to_color: &HashMap<char, u8>,
    batch: &mut Vec<String>,
    batch_sender: &Sender<Vec<String>>,
    stat_sender: &Sender<StatDelta>,
    profile_enabled: bool,
    time_batch: &mut TimeDelta,
    nodes_batch: &mut u64,
    leaves_batch: &mut u64,
    outputs_batch: &mut u64,
    pruned_batch: &mut u64,
    lhit_batch: &mut u64,
    ghit_batch: &mut u64,
    mmiss_batch: &mut u64,
    preview_ok: bool,
    preview_tx: &Sender<Message>,
    last_preview: &mut Instant,
    _lru_limit: usize,
    t0: Instant,
    abort: &AtomicBool,
    placed_total: u32,
    remain_suffix: &[u16],
) -> anyhow::Result<()> {
    if abort.load(Ordering::Relaxed) {
        return Ok(());
    }
    *nodes_batch += 1;
    if profile_enabled {
        time_batch.dfs_counts[depth].nodes += 1;
    }

    // 葉処理
    if depth == W {
        *leaves_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].leaves += 1;
        }

        if placed_total < 4 * threshold {
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_tshort += 1;
            }
            return Ok(());
        }
        if !any_color_has_four(cols0) {
            if profile_enabled {
                time_batch.dfs_counts[depth].leaf_pre_e1_impossible += 1;
            }
            return Ok(());
        }

        let pre = *cols0;

        if profile_enabled {
            time_batch.dfs_counts[depth].memo_miss += 1;
        }
        *mmiss_batch += 1;
        let reached = prof!(
            profile_enabled,
            time_batch.dfs_times[depth].leaf_memo_miss_compute,
            { reaches_t_from_pre_single_e1(&pre, threshold, exact_four_only) }
        );
        if !reached {
            if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
                && (*nodes_batch > 0
                    || *leaves_batch > 0
                    || *outputs_batch > 0
                    || *pruned_batch > 0
                    || *lhit_batch > 0
                    || *ghit_batch > 0
                    || *mmiss_batch > 0)
            {
                let _ = stat_sender.send(StatDelta {
                    nodes: *nodes_batch,
                    leaves: *leaves_batch,
                    outputs: *outputs_batch,
                    pruned: *pruned_batch,
                    lhit: *lhit_batch,
                    ghit: *ghit_batch,
                    mmiss: *mmiss_batch,
                });
                *nodes_batch = 0;
                *leaves_batch = 0;
                *outputs_batch = 0;
                *pruned_batch = 0;
                *lhit_batch = 0;
                *ghit_batch = 0;
                *mmiss_batch = 0;

                if profile_enabled && time_delta_has_any(time_batch) {
                    let td = time_batch.clone();
                    *time_batch = TimeDelta::default();
                    let _ = preview_tx.send(Message::TimeDelta(td));
                }
            }
            return Ok(());
        }

        let (key64, mirror) = prof!(profile_enabled, time_batch.dfs_times[depth].leaf_hash, {
            canonical_hash64_fast(&pre)
        });

        if !local_output_once.contains(&key64) {
            const OUTPUT_SET_CAP: usize = 2_000_000;
            const OUTPUT_SAMPLE_MASK: u64 = 0x7;
            const OUTPUT_SAMPLE_MATCH: u64 = 0;

            let under_cap = global_output_once.len() < OUTPUT_SET_CAP;
            let sampled = (key64 & OUTPUT_SAMPLE_MASK) == OUTPUT_SAMPLE_MATCH;
            let should_insert_global = preview_ok && under_cap && sampled;

            let is_new = if should_insert_global {
                global_output_once.insert(key64)
            } else {
                !global_output_once.contains(&key64)
            };

            if is_new {
                local_output_once.insert(key64);

                if preview_ok && last_preview.elapsed() >= Duration::from_millis(3000) {
                    let _ = preview_tx.send(Message::Preview(pre));
                    *last_preview = Instant::now();
                }

                let line = prof!(
                    profile_enabled,
                    time_batch.dfs_times[depth].out_serialize,
                    {
                        let key_str = encode_canonical_string(&pre, mirror);
                        let hash = fnv1a32(&key_str);
                        let rows = serialize_board_from_cols(&pre);
                        make_json_line_str(
                            &key_str,
                            hash,
                            threshold,
                            &rows,
                            map_label_to_color,
                            mirror,
                        )
                    }
                );
                batch.push(line);
                *outputs_batch += 1;

                if batch.len() >= 2048 {
                    let out = std::mem::take(batch);
                    let _ = batch_sender.send(out);
                }
            }
        }

        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    // 上界枝刈り
    let pruned_now = prof!(profile_enabled, time_batch.dfs_times[depth].upper_bound, {
        let placed = placed_total;
        let remain_cap = *remain_suffix.get(depth).unwrap_or(&0) as u32;
        placed + remain_cap < 4 * threshold
    });
    if pruned_now {
        *pruned_batch += 1;
        if profile_enabled {
            time_batch.dfs_counts[depth].pruned_upper += 1;
        }
        if (*nodes_batch >= 4096 || t0.elapsed().as_millis() % 500 == 0)
            && (*nodes_batch > 0
                || *leaves_batch > 0
                || *outputs_batch > 0
                || *pruned_batch > 0
                || *lhit_batch > 0
                || *ghit_batch > 0
                || *mmiss_batch > 0)
        {
            let _ = stat_sender.send(StatDelta {
                nodes: *nodes_batch,
                leaves: *leaves_batch,
                outputs: *outputs_batch,
                pruned: *pruned_batch,
                lhit: *lhit_batch,
                ghit: *ghit_batch,
                mmiss: *mmiss_batch,
            });
            *nodes_batch = 0;
            *leaves_batch = 0;
            *outputs_batch = 0;
            *pruned_batch = 0;
            *lhit_batch = 0;
            *ghit_batch = 0;
            *mmiss_batch = 0;

            if profile_enabled && time_delta_has_any(time_batch) {
                let td = time_batch.clone();
                *time_batch = TimeDelta::default();
                let _ = preview_tx.send(Message::TimeDelta(td));
            }
        }
        return Ok(());
    }

    let x = order[depth];
    match &gens[x] {
        ColGen::Pre(v) => {
            if profile_enabled {
                time_batch.dfs_counts[depth].cand_generated += v.len() as u64;
            }
            for &masks in v {
                if abort.load(Ordering::Relaxed) {
                    return Ok(());
                }
                prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                    assign_col_unrolled(cols0, x, &masks);
                });
                let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                let _ = dfs_combine_parallel(
                    depth + 1,
                    cols0,
                    gens,
                    order,
                    threshold,
                    exact_four_only,
                    _memo,
                    local_output_once,
                    global_output_once,
                    _global_memo,
                    map_label_to_color,
                    batch,
                    batch_sender,
                    stat_sender,
                    profile_enabled,
                    time_batch,
                    nodes_batch,
                    leaves_batch,
                    outputs_batch,
                    pruned_batch,
                    lhit_batch,
                    ghit_batch,
                    mmiss_batch,
                    preview_ok,
                    preview_tx,
                    last_preview,
                    _lru_limit,
                    t0,
                    abort,
                    placed_total + add,
                    remain_suffix,
                );
                clear_col_unrolled(cols0, x);
            }
        }
        ColGen::Stream(colv) => {
            if profile_enabled {
                let mut enum_time = Duration::ZERO;
                stream_column_candidates_timed(colv, &mut enum_time, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    time_batch.dfs_counts[depth].cand_generated += 1;

                    prof!(profile_enabled, time_batch.dfs_times[depth].assign_cols, {
                        assign_col_unrolled(cols0, x, &masks);
                    });

                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
                time_batch.dfs_times[depth].gen_candidates += enum_time;
            } else {
                stream_column_candidates(colv, |masks| {
                    if abort.load(Ordering::Relaxed) {
                        return;
                    }
                    assign_col_unrolled(cols0, x, &masks);
                    let add = (0..4).map(|c| masks[c].count_ones()).sum::<u32>();
                    let _ = dfs_combine_parallel(
                        depth + 1,
                        cols0,
                        gens,
                        order,
                        threshold,
                        exact_four_only,
                        _memo,
                        local_output_once,
                        global_output_once,
                        _global_memo,
                        map_label_to_color,
                        batch,
                        batch_sender,
                        stat_sender,
                        profile_enabled,
                        time_batch,
                        nodes_batch,
                        leaves_batch,
                        outputs_batch,
                        pruned_batch,
                        lhit_batch,
                        ghit_batch,
                        mmiss_batch,
                        preview_ok,
                        preview_tx,
                        last_preview,
                        _lru_limit,
                        t0,
                        abort,
                        placed_total + add,
                        remain_suffix,
                    );
                    clear_col_unrolled(cols0, x);
                });
            }
        }
    }

    Ok(())
}
