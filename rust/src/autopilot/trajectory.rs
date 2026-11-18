//! 直前に走った軌跡（パンくず）と、その範囲に限った後退（PC の `safety/trajectory_tracker.py`）。
//!
//! 後ろにはセンサーがないので、記録のない範囲へは下がらない。前進と同じ舵角で後退すると、
//! 同じ軌跡を戻る。

use std::collections::VecDeque;

/// Pi が受け付ける最短の指令（ms）。これより短い後退の段は作らない
pub const MIN_COMMAND_MS: u64 = 50;
/// 後退の後に車首をそらす旋回の時間
pub const ESCAPE_TURN_MS: u64 = 400;
pub const MAX_STEER_DEG: i32 = 35;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Step {
    steer_deg: i32,
    duration_ms: u64,
}

/// 後退か旋回の1段。(舵角, 時間)。速度は呼ぶ側が決める
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Move {
    pub steer_deg: i32,
    pub duration_ms: u64,
}

#[derive(Debug, Clone)]
pub struct Trajectory {
    history: VecDeque<Step>,
    capacity: usize,
}

impl Trajectory {
    /// 15Hz で 100 段 ≒ 7 秒分
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// 前進した1段を記録する
    pub fn record(&mut self, steer_deg: i32, duration_ms: u64) {
        if duration_ms == 0 {
            return;
        }
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(Step {
            steer_deg,
            duration_ms,
        });
    }

    pub fn clear(&mut self) {
        self.history.clear();
    }

    pub fn recorded_ms(&self) -> u64 {
        self.history.iter().map(|s| s.duration_ms).sum()
    }

    /// 新しい方から `target_ms` 分の後退の段。記録が足りなければ短く、なければ空。
    /// 周期が速いと1段が Pi の最短の指令より短くなるので、続く短い段はまとめる（舵角は時間で平均）
    pub fn reverse_sequence(&self, target_ms: u64) -> Vec<Move> {
        let mut moves = Vec::new();
        let mut accumulated = 0;
        let (mut pending_ms, mut pending_steer) = (0u64, 0f64);
        for step in self.history.iter().rev() {
            if accumulated >= target_ms {
                break;
            }
            let ms = step.duration_ms.min(target_ms - accumulated);
            accumulated += ms;
            pending_ms += ms;
            pending_steer += f64::from(step.steer_deg) * ms as f64;
            if pending_ms >= MIN_COMMAND_MS {
                moves.push(Move {
                    steer_deg: (pending_steer / pending_ms as f64).round() as i32,
                    duration_ms: pending_ms,
                });
                (pending_ms, pending_steer) = (0, 0.0);
            }
        }
        moves
    }

    /// 実際に下がった分を履歴から取り除く。残りは次の後退に使える
    pub fn consume(&mut self, mut duration_ms: u64) {
        while duration_ms > 0 {
            let Some(newest) = self.history.back_mut() else {
                break;
            };
            if newest.duration_ms <= duration_ms {
                duration_ms -= newest.duration_ms;
                self.history.pop_back();
            } else {
                newest.duration_ms -= duration_ms;
                duration_ms = 0;
            }
        }
    }

    /// 後退の後に車首をそらす旋回。段差の側が分かればその反対、なければ直前の曲がりの反対
    pub fn escape_turn(&self, cliff: Option<[bool; 3]>) -> Move {
        if let Some([left, _, right]) = cliff {
            if left && !right {
                return Move {
                    steer_deg: MAX_STEER_DEG,
                    duration_ms: ESCAPE_TURN_MS,
                };
            }
            if right && !left {
                return Move {
                    steer_deg: -MAX_STEER_DEG,
                    duration_ms: ESCAPE_TURN_MS,
                };
            }
        }
        let recent: Vec<i32> = self
            .history
            .iter()
            .rev()
            .take(10)
            .map(|s| s.steer_deg)
            .collect();
        let average = if recent.is_empty() {
            0.0
        } else {
            f64::from(recent.iter().sum::<i32>()) / recent.len() as f64
        };
        Move {
            steer_deg: if average >= 0.0 {
                -MAX_STEER_DEG
            } else {
                MAX_STEER_DEG
            },
            duration_ms: ESCAPE_TURN_MS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_replays_the_newest_steps_up_to_the_target() {
        let mut t = Trajectory::new(10);
        t.record(0, 100);
        t.record(10, 100);
        t.record(20, 100);
        assert_eq!(
            t.reverse_sequence(250),
            vec![
                Move {
                    steer_deg: 20,
                    duration_ms: 100
                },
                Move {
                    steer_deg: 10,
                    duration_ms: 100
                },
                Move {
                    steer_deg: 0,
                    duration_ms: 50
                },
            ]
        );
        assert!(Trajectory::new(5).reverse_sequence(500).is_empty());
    }

    #[test]
    fn short_steps_are_merged_and_a_too_short_rest_is_dropped() {
        let mut t = Trajectory::new(10);
        t.record(5, 100);
        assert_eq!(t.reverse_sequence(130).len(), 1); // 残り 30ms は作らない
        let mut fast = Trajectory::new(10);
        for steer in [0, 0, 20, 20] {
            fast.record(steer, 25); // 40Hz の周期
        }
        assert_eq!(
            fast.reverse_sequence(1000),
            vec![
                Move {
                    steer_deg: 20,
                    duration_ms: 50
                },
                Move {
                    steer_deg: 0,
                    duration_ms: 50
                },
            ]
        );
    }

    #[test]
    fn consume_removes_what_was_reversed_and_capacity_drops_the_oldest() {
        let mut t = Trajectory::new(3);
        for steer in [1, 2, 3, 4] {
            t.record(steer, 100);
        }
        assert_eq!(t.recorded_ms(), 300);
        t.consume(150);
        assert_eq!(t.recorded_ms(), 150);
        assert_eq!(
            t.reverse_sequence(1000)[0],
            Move {
                steer_deg: 3,
                duration_ms: 50
            }
        );
    }

    #[test]
    fn escape_turn_moves_away_from_the_cliff_side_or_the_last_turn() {
        let mut t = Trajectory::new(10);
        assert_eq!(t.escape_turn(Some([true, false, false])).steer_deg, 35);
        assert_eq!(t.escape_turn(Some([false, false, true])).steer_deg, -35);
        t.record(-20, 100);
        assert_eq!(t.escape_turn(None).steer_deg, 35);
        t.record(30, 100);
        t.record(30, 100);
        assert_eq!(t.escape_turn(Some([true, true, true])).steer_deg, -35);
    }
}
