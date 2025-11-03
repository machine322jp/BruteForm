// Application層のイベントをUI層の型に変換するアダプタ

use crate::app::{Message, Stats};
use crate::application::bruteforce::{SearchEvent, SearchProgress};
use crate::profiling::ProfileTotals;

/// SearchEvent を UI の Message に変換
/// 
/// Note: Stats の profile フィールドは UI 側で上書きされるため、
/// ここではデフォルト値を設定します。
pub fn search_event_to_message(event: SearchEvent) -> Message {
    match event {
        SearchEvent::Log(msg) => Message::Log(msg),
        SearchEvent::Preview(preview) => Message::Preview(preview),
        SearchEvent::Progress(progress) => {
            Message::Progress(search_progress_to_stats(progress, true))
        }
        SearchEvent::Finished(progress) => {
            Message::Finished(search_progress_to_stats(progress, false))
        }
        SearchEvent::Error(msg) => Message::Error(msg),
        SearchEvent::ProfileDelta(delta) => Message::TimeDelta(delta),
    }
}

/// SearchProgress を Stats に変換
/// 
/// profile フィールドはデフォルト値を設定します（UI 側で実際の値に上書きされます）
fn search_progress_to_stats(
    progress: SearchProgress,
    searching: bool,
) -> Stats {
    Stats {
        searching,
        unique: progress.unique_results,
        output: progress.output_count,
        nodes: progress.nodes_searched,
        pruned: progress.pruned_count,
        memo_hit_local: progress.memo_hit_local,
        memo_hit_global: progress.memo_hit_global,
        memo_miss: progress.memo_miss,
        total: progress.total_combinations,
        done: progress.completed_combinations,
        rate: progress.search_rate,
        memo_len: progress.memo_size,
        lru_limit: progress.lru_limit,
        profile: ProfileTotals::default(),
    }
}
