// 探索エンジン

use anyhow::{anyhow, Result};
use crossbeam_channel::Sender;
use nohash_hasher::BuildNoHashHasher;
use num_bigint::BigUint;
use num_traits::{One, Zero};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Instant;

use crate::application::bruteforce::event::{SearchEvent, SearchProgress, StatDelta};
use crate::constants::{DU64Map, DU64Set, U64Set, H, W};
use crate::profiling::{time_delta_has_any, TimeDelta};
use crate::domain::constraints::coloring::*;
use crate::domain::search::SearchConfig;
use crate::application::bruteforce::search::dfs::dfs_combine_parallel;
use crate::application::bruteforce::search::writer::spawn_writer_thread;
use crate::application::bruteforce::search::aggregator::spawn_aggregator_thread;
use crate::infrastructure::cache::lru::{array_init, ApproxLru};

/// 彩色メタデータ
struct ColoringMeta {
    label_to_color: HashMap<char, u8>,
    generators: [ColGen; W],
    max_fill: [u8; W],
    column_order: Vec<usize>,
}

pub fn run_search(
    base_board: Vec<char>,
    config: &SearchConfig,
    outfile: PathBuf,
    tx: Sender<SearchEvent>,
    abort: Arc<AtomicBool>,
) -> Result<()> {
    let threshold = config.threshold.get();
    let lru_limit = config.lru_limit.get();
    let stop_progress_plateau = config.stop_progress_plateau.get();
    let exact_four_only = config.exact_four_only;
    let profile_enabled = config.profile_enabled;
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        let _ = tx.send(SearchEvent::Log(
            "抽象ラベルの4色彩色が存在しないため、探索を終了します。".into(),
        ));
        let _ = tx.send(SearchEvent::Finished(SearchProgress::default()));
        return Ok(());
    }
    let _ = tx.send(SearchEvent::Log(format!(
        "抽象ラベル={} / 彩色候補={} / 4個消しモード={} / 計測={}",
        info.labels.iter().collect::<String>(),
        colorings.len(),
        if exact_four_only { "ON" } else { "OFF" },
        if profile_enabled { "ON" } else { "OFF" }
    )));

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        let bmi2 = std::is_x86_feature_detected!("bmi2");
        let popcnt = std::is_x86_feature_detected!("popcnt");
        let _ = tx.send(SearchEvent::Log(format!(
            "CPU features: bmi2={} / popcnt={}",
            bmi2, popcnt
        )));
    }

    let mut metas: Vec<ColoringMeta> = Vec::new();
    let mut total = BigUint::zero();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut cols: [Vec<crate::domain::board::TCell>; W] = array_init(|_| Vec::with_capacity(H));
        for x in 0..W {
            for y in 0..H {
                cols[x].push(templ[y * W + x]);
            }
        }
        let mut prod = BigUint::one();
        let mut impossible = false;
        let mut counts: [BigUint; W] = array_init(|_| BigUint::zero());
        let mut max_fill_arr: [u8; W] = [0; W];
        for x in 0..W {
            let cnt = count_column_candidates_dp(&cols[x]);
            if cnt.is_zero() {
                impossible = true;
                break;
            }
            counts[x] = cnt.clone();
            max_fill_arr[x] = compute_max_fill(&cols[x]);
            prod *= cnt;
        }
        if !impossible {
            total += prod;
            let mut order: Vec<usize> = (0..W).collect();
            order.sort_by(|&a, &b| counts[a].cmp(&counts[b]));
            let gens: [ColGen; W] = array_init(|x| build_colgen(&cols[x], &counts[x]));
            metas.push(ColoringMeta {
                label_to_color: map,
                generators: gens,
                max_fill: max_fill_arr,
                column_order: order,
            });
        }
    }
    let _ = tx.send(SearchEvent::Log(format!(
        "厳密な総組合せ（列制約適用）: {}",
        total
    )));

    // ライタースレッドを起動
    let (wtx, writer_handle) = spawn_writer_thread(outfile.clone(), tx.clone());

    // 共有データ構造を初期化
    let global_output_once: Arc<DU64Set> =
        Arc::new(DU64Set::with_hasher(BuildNoHashHasher::default()));
    let global_memo: Arc<DU64Map<bool>> =
        Arc::new(DU64Map::with_hasher(BuildNoHashHasher::default()));

    // 集約スレッドを起動
    let (stx, agg_handle) = spawn_aggregator_thread(
        total.clone(),
        lru_limit,
        stop_progress_plateau,
        tx.clone(),
        abort.clone(),
        global_memo.clone(),
    );

    let t0 = Instant::now();
    let num_threads = rayon::current_num_threads().max(1);
    let lru_per_thread = lru_limit / num_threads;

    // 並列探索
    metas.par_iter().enumerate().try_for_each(
        |(i, meta)| -> Result<()> {
            if abort.load(Ordering::Relaxed) {
                return Ok(());
            }

            let preview_ok = i == 0;
            let first_x = meta.column_order[0];
            let mut first_candidates: Vec<[u16; 4]> = Vec::new();
            match &meta.generators[first_x] {
                ColGen::Pre(v) => first_candidates.extend_from_slice(v),
                ColGen::Stream(colv) => {
                    stream_column_candidates(colv, |m| first_candidates.push(m));
                }
            }

            let mut remain_suffix: Vec<u16> = vec![0; W + 1];
            for d in (0..W).rev() {
                remain_suffix[d] = remain_suffix[d + 1] + meta.max_fill[meta.column_order[d]] as u16;
            }

            if first_candidates.len() >= num_threads {
                first_candidates
                    .par_iter()
                    .try_for_each(|masks| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        let mut cols0 = [[0u16; W]; 4];
                        for c in 0..4 {
                            cols0[c][first_x] = masks[c];
                        }
                        let mut memo = ApproxLru::new(lru_per_thread);
                        let mut local_output_once: U64Set = U64Set::default();
                        let mut batch: Vec<String> = Vec::with_capacity(256);

                        let mut nodes_batch: u64 = 0;
                        let mut leaves_batch: u64 = 0;
                        let mut outputs_batch: u64 = 0;
                        let mut pruned_batch: u64 = 0;
                        let mut lhit_batch: u64 = 0;
                        let mut ghit_batch: u64 = 0;
                        let mut mmiss_batch: u64 = 0;
                        let mut last_preview = Instant::now();
                        let mut time_batch = TimeDelta::default();

                        let placed_first: u32 = (0..4).map(|c| masks[c].count_ones()).sum();
                        let _ = dfs_combine_parallel(
                            1,
                            &mut cols0,
                            &meta.generators,
                            &meta.column_order,
                            threshold,
                            exact_four_only,
                            &mut memo,
                            &mut local_output_once,
                            &global_output_once,
                            &global_memo,
                            &meta.label_to_color,
                            &mut batch,
                            &wtx,
                            &stx,
                            profile_enabled,
                            &mut time_batch,
                            &mut nodes_batch,
                            &mut leaves_batch,
                            &mut outputs_batch,
                            &mut pruned_batch,
                            &mut lhit_batch,
                            &mut ghit_batch,
                            &mut mmiss_batch,
                            preview_ok,
                            &tx,
                            &mut last_preview,
                            lru_limit,
                            t0,
                            &abort,
                            placed_first,
                            &remain_suffix,
                        );

                        if !batch.is_empty() {
                            let _ = wtx.send(batch);
                        }
                        if nodes_batch > 0
                            || leaves_batch > 0
                            || outputs_batch > 0
                            || pruned_batch > 0
                            || lhit_batch > 0
                            || ghit_batch > 0
                            || mmiss_batch > 0
                        {
                            let _ = stx.send(StatDelta {
                                nodes: nodes_batch,
                                leaves: leaves_batch,
                                outputs: outputs_batch,
                                pruned: pruned_batch,
                                lhit: lhit_batch,
                                ghit: ghit_batch,
                                mmiss: mmiss_batch,
                            });
                        }
                        if profile_enabled && time_delta_has_any(&time_batch) {
                            let _ = tx.send(SearchEvent::ProfileDelta(time_batch));
                        }
                        Ok(())
                    })?;
            } else {
                let second_x = meta.column_order[1];
                let mut second_candidates: Vec<[u16; 4]> = Vec::new();
                match &meta.generators[second_x] {
                    ColGen::Pre(v) => second_candidates.extend_from_slice(v),
                    ColGen::Stream(colv) => {
                        stream_column_candidates(colv, |m| second_candidates.push(m));
                    }
                }

                second_candidates
                    .par_iter()
                    .try_for_each(|m2| -> Result<()> {
                        if abort.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                        for masks in &first_candidates {
                            if abort.load(Ordering::Relaxed) {
                                break;
                            }

                            let mut cols0 = [[0u16; W]; 4];
                            for c in 0..4 {
                                cols0[c][first_x] = masks[c];
                                cols0[c][second_x] = m2[c];
                            }
                            let mut memo = ApproxLru::new(lru_per_thread);
                            let mut local_output_once: U64Set = U64Set::default();
                            let mut batch: Vec<String> = Vec::with_capacity(256);

                            let mut nodes_batch: u64 = 0;
                            let mut leaves_batch: u64 = 0;
                            let mut outputs_batch: u64 = 0;
                            let mut pruned_batch: u64 = 0;
                            let mut lhit_batch: u64 = 0;
                            let mut ghit_batch: u64 = 0;
                            let mut mmiss_batch: u64 = 0;
                            let mut last_preview = Instant::now();
                            let mut time_batch = TimeDelta::default();

                            let placed2: u32 = (0..4)
                                .map(|c| masks[c].count_ones() + m2[c].count_ones())
                                .sum();
                            let _ = dfs_combine_parallel(
                                2,
                                &mut cols0,
                                &meta.generators,
                                &meta.column_order,
                                threshold,
                                exact_four_only,
                                &mut memo,
                                &mut local_output_once,
                                &global_output_once,
                                &global_memo,
                                &meta.label_to_color,
                                &mut batch,
                                &wtx,
                                &stx,
                                profile_enabled,
                                &mut time_batch,
                                &mut nodes_batch,
                                &mut leaves_batch,
                                &mut outputs_batch,
                                &mut pruned_batch,
                                &mut lhit_batch,
                                &mut ghit_batch,
                                &mut mmiss_batch,
                                preview_ok,
                                &tx,
                                &mut last_preview,
                                lru_limit,
                                t0,
                                &abort,
                                placed2,
                                &remain_suffix,
                            );

                            if !batch.is_empty() {
                                let _ = wtx.send(batch);
                            }
                            if nodes_batch > 0
                                || leaves_batch > 0
                                || outputs_batch > 0
                                || pruned_batch > 0
                                || lhit_batch > 0
                                || ghit_batch > 0
                                || mmiss_batch > 0
                            {
                                let _ = stx.send(StatDelta {
                                    nodes: nodes_batch,
                                    leaves: leaves_batch,
                                    outputs: outputs_batch,
                                    pruned: pruned_batch,
                                    lhit: lhit_batch,
                                    ghit: ghit_batch,
                                    mmiss: mmiss_batch,
                                });
                            }
                            if profile_enabled && time_delta_has_any(&time_batch) {
                                let _ = tx.send(SearchEvent::ProfileDelta(time_batch));
                            }
                        }
                        Ok(())
                    })?;
            }
            Ok(())
        },
    )?;

    drop(wtx);
    drop(stx);
    let writer_result = writer_handle
        .join()
        .map_err(|_| anyhow!("writer join error"))?;
    writer_result?;
    agg_handle.join().map_err(|_| anyhow!("agg join error"))?;

    Ok(())
}
