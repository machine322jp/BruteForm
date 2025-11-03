// 連鎖アニメーション制御

use crate::app::chain_play::{AnimPhase, AnimState, ChainPlay};
use crate::constants::W;
use crate::domain::board::bitboard::{
    apply_clear_no_fall, apply_given_clear_and_fall, compute_erase_mask_cols,
};

/// アニメーション制御のユーティリティ
pub struct Animation;

impl Animation {
    /// 連鎖が発生するかチェックし、あればアニメーション開始
    pub fn check_and_start_chain(cp: &mut ChainPlay) {
        let clear = compute_erase_mask_cols(&cp.cols);
        let any = (0..W).any(|x| clear[x] != 0);
        if !any {
            return;
        }
        cp.lock = true;
        let erased = apply_clear_no_fall(&cp.cols, &clear);
        let next = apply_given_clear_and_fall(&cp.cols, &clear);
        cp.erased_cols = Some(erased);
        cp.next_cols = Some(next);
        cp.cols = erased;
        cp.anim = Some(AnimState {
            phase: AnimPhase::AfterErase,
            since: std::time::Instant::now(),
        });
    }

    /// アニメーションを1ステップ進める
    pub fn step_animation(cp: &mut ChainPlay) {
        let Some(anim) = cp.anim else { return };
        let elapsed = anim.since.elapsed();
        if elapsed < std::time::Duration::from_millis(500) {
            return;
        }
        match anim.phase {
            AnimPhase::AfterErase => {
                if let Some(next) = cp.next_cols.take() {
                    cp.cols = next;
                }
                cp.anim = Some(AnimState {
                    phase: AnimPhase::AfterFall,
                    since: std::time::Instant::now(),
                });
            }
            AnimPhase::AfterFall => {
                let clear = compute_erase_mask_cols(&cp.cols);
                let any = (0..W).any(|x| clear[x] != 0);
                if !any {
                    cp.anim = None;
                    cp.erased_cols = None;
                    cp.next_cols = None;
                    cp.lock = false;
                } else {
                    let erased = apply_clear_no_fall(&cp.cols, &clear);
                    let next = apply_given_clear_and_fall(&cp.cols, &clear);
                    cp.cols = erased;
                    cp.erased_cols = Some(erased);
                    cp.next_cols = Some(next);
                    cp.anim = Some(AnimState {
                        phase: AnimPhase::AfterErase,
                        since: std::time::Instant::now(),
                    });
                }
            }
        }
    }
}
