// 探索エンジン

use anyhow::{anyhow, Context, Result};
use crossbeam_channel::{unbounded, Sender};
use nohash_hasher::BuildNoHashHasher;
use num_bigint::BigUint;
use num_traits::{One, ToPrimitive, Zero};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::app::{Message, StatDelta, Stats};
use crate::constants::{DU64Map, DU64Set, U64Set, H, W};
use crate::profiling::{time_delta_has_any, ProfileTotals, TimeDelta};
use crate::search::coloring::*;
use crate::search::dfs::dfs_combine_parallel;
use crate::search::lru::{array_init, ApproxLru};

#[allow(clippy::too_many_arguments)]
pub fn run_search(
    base_board: Vec<char>,
    threshold: u32,
    lru_limit: usize,
    outfile: PathBuf,
    tx: Sender<Message>,
    abort: Arc<AtomicBool>,
    stop_progress_plateau: f32,
    exact_four_only: bool,
    profile_enabled: bool,
) -> Result<()> {
    let info = build_abstract_info(&base_board);
    let colorings = enumerate_colorings_fast(&info);
    if colorings.is_empty() {
        let _ = tx.send(Message::Log(
            "抽象ラベルの4色彩色が存在しないため、探索を終了します。".into(),
        ));
        let _ = tx.send(Message::Finished(Stats::default()));
        return Ok(());
    }
    let _ = tx.send(Message::Log(format!(
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
        let _ = tx.send(Message::Log(format!(
            "CPU features: bmi2={} / popcnt={}",
            bmi2, popcnt
        )));
    }

    type Meta = (HashMap<char, u8>, [ColGen; W], [u8; W], Vec<usize>);
    let mut metas: Vec<Meta> = Vec::new();
    let mut total = BigUint::zero();
    for assign in &colorings {
        let mut map = HashMap::<char, u8>::new();
        for (i, &lab) in info.labels.iter().enumerate() {
            map.insert(lab, assign[i]);
        }
        let templ = apply_coloring_to_template(&base_board, &map);
        let mut cols: [Vec<crate::model::TCell>; W] = array_init(|_| Vec::with_capacity(H));
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
            metas.push((map, gens, max_fill_arr, order));
        }
    }
    let _ = tx.send(Message::Log(format!(
        "厳密な総組合せ（列制約適用）: {}",
        total
    )));

    // writer
    let (wtx, wrx) = unbounded::<Vec<String>>();
    let tx_for_writer = tx.clone();
    let writer_handle = {
        let outfile = outfile.clone();
        thread::spawn(move || -> Result<()> {
            let mut io_time = Duration::ZERO;
            let file = File::create(&outfile)
                .with_context(|| format!("出力を作成できません: {}", outfile.display()))?;
            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
            while let Ok(batch) = wrx.recv() {
                let t0 = Instant::now();
                for line in batch {
                    writer.write_all(line.as_bytes())?;
                    writer.write_all(b"\n")?;
                }
                io_time += t0.elapsed();
            }
            writer.flush()?;
            if io_time != Duration::ZERO {
                let mut td = TimeDelta::default();
                td.io_write_total = io_time;
                let _ = tx_for_writer.send(Message::TimeDelta(td));
            }
            Ok(())
        })
    };

    // 集約 Progress
    let (stx, srx) = unbounded::<StatDelta>();
    let t0 = Instant::now();
    let tx_progress = tx.clone();
    let total_clone = total.clone();
    let abort_for_agg = abort.clone();
    let global_output_once: Arc<DU64Set> =
        Arc::new(DU64Set::with_hasher(BuildNoHashHasher::default()));
    let global_memo: Arc<DU64Map<bool>> =
        Arc::new(DU64Map::with_hasher(BuildNoHashHasher::default()));

    let global_memo_for_agg = global_memo.clone();
    let lru_limit_for_agg = lru_limit;
    let agg_handle = thread::spawn(move || {
        let mut nodes: u64 = 0;
        let mut outputs: u64 = 0;
        let mut done = BigUint::zero();
        let mut pruned: u64 = 0;
        let mut lhit: u64 = 0;
        let mut ghit: u64 = 0;
        let mut mmiss: u64 = 0;
        let mut last_send = Instant::now();

        let mut last_output: u64 = 0;
        let mut progress_at_last_output: f64 = 0.0;
        let plateau: f64 = stop_progress_plateau as f64;

        loop {
            match srx.recv_timeout(Duration::from_millis(500)) {
                Ok(d) => {
                    nodes += d.nodes;
                    outputs += d.outputs;
                    if d.leaves > 0 {
                        done += BigUint::from(d.leaves);
                    }
                    pruned += d.pruned;
                    lhit += d.lhit;
                    ghit += d.ghit;
                    mmiss += d.mmiss;

                    if outputs > last_output {
                        last_output = outputs;
                        if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                            if tt > 0.0 {
                                progress_at_last_output = (td / tt).clamp(0.0, 1.0);
                            }
                        }
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                    let dt = t0.elapsed().as_secs_f64();
                    let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                    let memo_len = global_memo_for_agg.len();
                    let st = Stats {
                        searching: false,
                        unique: outputs,
                        output: outputs,
                        nodes,
                        pruned,
                        memo_hit_local: lhit,
                        memo_hit_global: ghit,
                        memo_miss: mmiss,
                        total: total_clone.clone(),
                        done: done.clone(),
                        rate,
                        memo_len,
                        lru_limit: lru_limit_for_agg,
                        profile: ProfileTotals::default(),
                    };
                    let _ = tx_progress.send(Message::Finished(st));
                    break;
                }
            }

            if plateau > 0.0 {
                if let (Some(td), Some(tt)) = (done.to_f64(), total_clone.to_f64()) {
                    if tt > 0.0 {
                        let p = (td / tt).clamp(0.0, 1.0);
                        if p - progress_at_last_output >= plateau {
                            let msg = format!(
                                "早期終了: 進捗が {:.1}% 進む間に新規出力がありませんでした（しきい値 {:.1}%）",
                                (p - progress_at_last_output) * 100.0,
                                plateau * 100.0
                            );
                            let _ = tx_progress.send(Message::Log(msg));
                            abort_for_agg.store(true, Ordering::Relaxed);
                        }
                    }
                }
            }

            if last_send.elapsed() >= Duration::from_millis(500) {
                let dt = t0.elapsed().as_secs_f64();
                let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                let memo_len = global_memo_for_agg.len();
                let st = Stats {
                    searching: true,
                    unique: outputs,
                    output: outputs,
                    nodes,
                    pruned,
                    memo_hit_local: lhit,
                    memo_hit_global: ghit,
                    memo_miss: mmiss,
                    total: total_clone.clone(),
                    done: done.clone(),
                    rate,
                    memo_len,
                    lru_limit: lru_limit_for_agg,
                    profile: ProfileTotals::default(),
                };
                let _ = tx_progress.send(Message::Progress(st));
                last_send = Instant::now();
            }
        }
    });

    // 並列探索
    metas.par_iter().enumerate().try_for_each(
        |(i, (map_label_to_color, gens, max_fill, order))| -> Result<()> {
            if abort.load(Ordering::Relaxed) {
                return Ok(());
            }

            let preview_ok = i == 0;
            let first_x = order[0];
            let mut first_candidates: Vec<[u16; 4]> = Vec::new();
            match &gens[first_x] {
                ColGen::Pre(v) => first_candidates.extend_from_slice(v),
                ColGen::Stream(colv) => {
                    stream_column_candidates(colv, |m| first_candidates.push(m));
                }
            }

            let mut remain_suffix: Vec<u16> = vec![0; W + 1];
            for d in (0..W).rev() {
                remain_suffix[d] = remain_suffix[d + 1] + max_fill[order[d]] as u16;
            }

            let threads = rayon::current_num_threads().max(1);
            if first_candidates.len() >= threads {
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
                        let mut memo =
                            ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
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
                            gens,
                            order,
                            threshold,
                            exact_four_only,
                            &mut memo,
                            &mut local_output_once,
                            &global_output_once,
                            &global_memo,
                            map_label_to_color,
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
                            let _ = tx.send(Message::TimeDelta(time_batch));
                        }
                        Ok(())
                    })?;
            } else {
                let second_x = order[1];
                let mut second_candidates: Vec<[u16; 4]> = Vec::new();
                match &gens[second_x] {
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
                            let mut memo =
                                ApproxLru::new(lru_limit / rayon::current_num_threads().max(1));
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
                                gens,
                                order,
                                threshold,
                                exact_four_only,
                                &mut memo,
                                &mut local_output_once,
                                &global_output_once,
                                &global_memo,
                                map_label_to_color,
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
                                let _ = tx.send(Message::TimeDelta(time_batch));
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
