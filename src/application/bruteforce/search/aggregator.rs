// 進捗集約スレッド

use crossbeam_channel::{unbounded, Receiver, Sender};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::application::bruteforce::event::{SearchEvent, SearchProgress, StatDelta};
use crate::constants::DU64Map;

/// 進捗集約スレッドを起動
pub fn spawn_aggregator_thread(
    total_combinations: BigUint,
    lru_limit: usize,
    stop_progress_plateau: f32,
    event_tx: Sender<SearchEvent>,
    abort_flag: Arc<AtomicBool>,
    global_memo: Arc<DU64Map<bool>>,
) -> (Sender<StatDelta>, JoinHandle<()>) {
    let (stx, srx) = unbounded::<StatDelta>();
    
    let handle = thread::spawn(move || {
        aggregator_thread_main(
            srx,
            total_combinations,
            lru_limit,
            stop_progress_plateau,
            event_tx,
            abort_flag,
            global_memo,
        )
    });
    
    (stx, handle)
}

/// 集約スレッドのメイン処理
#[allow(clippy::too_many_arguments)]
fn aggregator_thread_main(
    srx: Receiver<StatDelta>,
    total_combinations: BigUint,
    lru_limit: usize,
    stop_progress_plateau: f32,
    tx: Sender<SearchEvent>,
    abort: Arc<AtomicBool>,
    global_memo: Arc<DU64Map<bool>>,
) {
    let t0 = Instant::now();
    let mut nodes: u64 = 0;
    let mut outputs: u64 = 0;
    let mut done = BigUint::from(0u32);
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
                    if let (Some(td), Some(tt)) = (done.to_f64(), total_combinations.to_f64()) {
                        if tt > 0.0 {
                            progress_at_last_output = (td / tt).clamp(0.0, 1.0);
                        }
                    }
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                // 全探索スレッドが終了
                let dt = t0.elapsed().as_secs_f64();
                let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
                let memo_len = global_memo.len();
                let st = SearchProgress {
                    searching: false,
                    unique_results: outputs,
                    output_count: outputs,
                    nodes_searched: nodes,
                    pruned_count: pruned,
                    memo_hit_local: lhit,
                    memo_hit_global: ghit,
                    memo_miss: mmiss,
                    total_combinations: total_combinations.clone(),
                    completed_combinations: done.clone(),
                    search_rate: rate,
                    memo_size: memo_len,
                    lru_limit,
                };
                let _ = tx.send(SearchEvent::Finished(st));
                break;
            }
        }

        // 早期終了チェック（進捗停滞）
        if plateau > 0.0 {
            if let (Some(td), Some(tt)) = (done.to_f64(), total_combinations.to_f64()) {
                if tt > 0.0 {
                    let p = (td / tt).clamp(0.0, 1.0);
                    if p - progress_at_last_output >= plateau {
                        let msg = format!(
                            "早期終了: 進捗が {:.1}% 進む間に新規出力がありませんでした（しきい値 {:.1}%）",
                            (p - progress_at_last_output) * 100.0,
                            plateau * 100.0
                        );
                        let _ = tx.send(SearchEvent::Log(msg));
                        abort.store(true, Ordering::Relaxed);
                    }
                }
            }
        }

        // 定期的な進捗通知
        if last_send.elapsed() >= Duration::from_millis(500) {
            let dt = t0.elapsed().as_secs_f64();
            let rate = if dt > 0.0 { nodes as f64 / dt } else { 0.0 };
            let memo_len = global_memo.len();
            let st = SearchProgress {
                searching: true,
                unique_results: outputs,
                output_count: outputs,
                nodes_searched: nodes,
                pruned_count: pruned,
                memo_hit_local: lhit,
                memo_hit_global: ghit,
                memo_miss: mmiss,
                total_combinations: total_combinations.clone(),
                completed_combinations: done.clone(),
                search_rate: rate,
                memo_size: memo_len,
                lru_limit,
            };
            let _ = tx.send(SearchEvent::Progress(st));
            last_send = Instant::now();
        }
    }
}
