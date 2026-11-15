//! 障害物を避ける判断。走行モデルへ渡す意図（いつ、どちらへ曲がるか）を決める。
//!
//! PC の `python/src/e2e/avoidance.py` と同じ規則。走行モデルは「おまかせ」ではまっすぐ進み、
//! 曲がる向きは意図で受け取る（ROADMAP の M3）。
//!
//! - 曲がり始め: 超音波が近い、またはモデルの「前方が塞がっている確率」が高い状態が続いた
//! - 向き: 上の層の意図が左右ならその向き。おまかせなら右を先に試し、曲がれなかったら次は反対側
//! - 曲がり終え: 超音波もカメラも空いている状態が続いた
//! - 減速: カメラが塞がっていそうと判断している間は、確実に動く最低の速度まで落とす
//!
//! 時刻は秒（f64）で受け取り、実機なしで確かめられるようにする。

use super::Intent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvoidState {
    Cruise,
    Avoid,
}

impl AvoidState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cruise => "cruise",
            Self::Avoid => "avoid",
        }
    }
}

/// 距離は cm、時間は秒、確率は 0〜1。値は PC の AvoidanceConfig と同じ
#[derive(Debug, Clone, Copy)]
pub struct AvoidanceConfig {
    pub avoid_cm: f64,
    pub clear_cm: f64,
    pub clear_frames: u32,
    pub avoid_min_s: f64,
    pub avoid_max_s: f64,
    pub first_side: Intent,
    pub blocked_enter: f64,
    pub blocked_exit: f64,
    pub blocked_frames: u32,
    pub slow_probability: f64,
}

impl Default for AvoidanceConfig {
    fn default() -> Self {
        Self {
            avoid_cm: 50.0,
            clear_cm: 90.0,
            clear_frames: 4,
            avoid_min_s: 1.0,
            avoid_max_s: 4.0,
            first_side: Intent::Right,
            blocked_enter: 0.6,
            blocked_exit: 0.4,
            blocked_frames: 2,
            slow_probability: 0.5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AvoidEvent {
    Begin,
    Success,
    /// 曲がれず反対側へ切り替えた
    Switch,
}

impl AvoidEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Begin => "begin",
            Self::Success => "success",
            Self::Switch => "switch",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub intent: Intent,
    pub slow: bool,
    pub state: AvoidState,
    pub event: Option<AvoidEvent>,
    pub reason: &'static str,
}

#[derive(Debug, Clone)]
pub struct AvoidancePolicy {
    pub config: AvoidanceConfig,
    pub state: AvoidState,
    side: Intent,
    next_side: Option<Intent>,
    since: f64,
    clear_count: u32,
    blocked_count: u32,
    pub failures: u32,
}

impl AvoidancePolicy {
    pub fn new(config: AvoidanceConfig) -> Self {
        Self {
            config,
            state: AvoidState::Cruise,
            side: config.first_side,
            next_side: None,
            since: 0.0,
            clear_count: 0,
            blocked_count: 0,
            failures: 0,
        }
    }

    /// `distance_cm` は反射が戻らない（遠い）時と分からない時は None。
    /// `blocked` はモデルがその出力を持たない（学習していない）時は None
    pub fn step(
        &mut self,
        now: f64,
        distance_cm: Option<f64>,
        blocked: Option<f64>,
        upper: Intent,
    ) -> Decision {
        let c = self.config;
        let camera_blocked = blocked.is_some_and(|p| p >= c.blocked_enter);
        self.blocked_count = if camera_blocked {
            self.blocked_count + 1
        } else {
            0
        };
        let slow = blocked.is_some_and(|p| p >= c.slow_probability);
        let near = distance_cm.is_some_and(|d| d < c.avoid_cm);

        if self.state == AvoidState::Cruise {
            let seen_by_camera = self.blocked_count >= c.blocked_frames;
            if near || seen_by_camera {
                let side = if upper.is_side() {
                    upper
                } else {
                    self.next_side.unwrap_or(c.first_side)
                };
                self.enter_avoid(side, now);
                let reason = if near { "ultrasonic" } else { "camera" };
                return self.decision(self.side, slow, Some(AvoidEvent::Begin), reason);
            }
            return self.decision(upper, slow, None, "");
        }

        let ultrasonic_clear = distance_cm.is_none_or(|d| d > c.clear_cm);
        let camera_clear = blocked.is_none_or(|p| p < c.blocked_exit);
        self.clear_count = if ultrasonic_clear && camera_clear {
            self.clear_count + 1
        } else {
            0
        };
        let elapsed = now - self.since;
        if self.clear_count >= c.clear_frames && elapsed >= c.avoid_min_s {
            self.state = AvoidState::Cruise;
            self.failures = 0;
            self.next_side = None;
            self.clear_count = 0;
            self.blocked_count = 0;
            return self.decision(upper, slow, Some(AvoidEvent::Success), "");
        }
        if elapsed > c.avoid_max_s {
            return self.switch_side(now, slow, "timeout");
        }
        self.decision(self.side, slow, None, "")
    }

    /// Pi が障害物で前進を止めた。曲がれなかったので、次は反対側へ曲がる
    pub fn on_stopped_by_obstacle(&mut self, now: f64) -> Decision {
        if self.state == AvoidState::Cruise {
            let side = self.next_side.unwrap_or(self.config.first_side);
            self.enter_avoid(side, now);
            self.failures += 1;
            return self.decision(self.side, true, Some(AvoidEvent::Begin), "stopped");
        }
        self.switch_side(now, true, "stopped")
    }

    fn enter_avoid(&mut self, side: Intent, now: f64) {
        self.state = AvoidState::Avoid;
        self.side = side;
        self.since = now;
        self.clear_count = 0;
    }

    fn switch_side(&mut self, now: f64, slow: bool, reason: &'static str) -> Decision {
        self.failures += 1;
        let other = self.side.other_side();
        self.next_side = Some(other);
        self.enter_avoid(other, now);
        self.decision(self.side, slow, Some(AvoidEvent::Switch), reason)
    }

    fn decision(
        &self,
        intent: Intent,
        slow: bool,
        event: Option<AvoidEvent>,
        reason: &'static str,
    ) -> Decision {
        Decision {
            intent,
            slow,
            state: self.state,
            event,
            reason,
        }
    }
}

/// 避けている間の舵角の下限（度）。自動収集の手本は避ける時に30°まで切っていたが、
/// 床での最初の走行（2026-11-15）では右の意図で 8〜17°しか切らず、椅子やテーブルの脚へ
/// 寄っていった（左は 18〜30°）。避けている間は、意図の向きへ少なくともこれだけ切る
pub const AVOID_MIN_STEER_DEG: i32 = 20;

/// 避けている間だけ、モデルの舵角を意図の向きへ `min_deg` 以上にする。それ以外はそのまま
pub fn turn_floor(steer_deg: i32, decision: &Decision, min_deg: i32) -> i32 {
    if decision.state != AvoidState::Avoid {
        return steer_deg;
    }
    match decision.intent {
        Intent::Right => steer_deg.max(min_deg),
        Intent::Left => steer_deg.min(-min_deg),
        Intent::Any => steer_deg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> AvoidancePolicy {
        AvoidancePolicy::new(AvoidanceConfig::default())
    }

    #[test]
    fn cruise_passes_the_upper_intent_through() {
        let mut p = policy();
        let d = p.step(0.0, Some(120.0), None, Intent::Any);
        assert_eq!(
            (d.intent, d.state, d.event),
            (Intent::Any, AvoidState::Cruise, None)
        );
        assert_eq!(p.step(0.1, None, None, Intent::Left).intent, Intent::Left);
    }

    #[test]
    fn near_obstacle_turns_right_first_then_succeeds_when_clear() {
        let mut p = policy();
        let d = p.step(0.0, Some(40.0), None, Intent::Any);
        assert_eq!(d.event, Some(AvoidEvent::Begin));
        assert_eq!((d.intent, d.reason), (Intent::Right, "ultrasonic"));
        // 空いていても、曲がり始めて1秒は続ける
        for i in 1..=4 {
            let d = p.step(0.1 * f64::from(i), None, None, Intent::Any);
            assert_eq!(d.intent, Intent::Right);
        }
        let d = p.step(1.1, Some(120.0), None, Intent::Any);
        assert_eq!(d.event, Some(AvoidEvent::Success));
        assert_eq!((d.intent, d.state), (Intent::Any, AvoidState::Cruise));
    }

    #[test]
    fn the_upper_layer_chooses_the_side() {
        let mut p = policy();
        assert_eq!(
            p.step(0.0, Some(30.0), None, Intent::Left).intent,
            Intent::Left
        );
    }

    #[test]
    fn timeout_switches_to_the_other_side_and_remembers_it() {
        let mut p = policy();
        p.step(0.0, Some(30.0), None, Intent::Any);
        let d = p.step(4.5, Some(30.0), None, Intent::Any);
        assert_eq!(d.event, Some(AvoidEvent::Switch));
        assert_eq!((d.intent, d.reason), (Intent::Left, "timeout"));
        assert_eq!(p.failures, 1);
    }

    #[test]
    fn stopped_by_obstacle_turns_and_then_switches_side() {
        let mut p = policy();
        let d = p.on_stopped_by_obstacle(0.0);
        assert_eq!(
            (d.intent, d.event, d.slow),
            (Intent::Right, Some(AvoidEvent::Begin), true)
        );
        let d = p.on_stopped_by_obstacle(1.0);
        assert_eq!(
            (d.intent, d.event),
            (Intent::Left, Some(AvoidEvent::Switch))
        );
        // 成功するまで、おまかせの次の向きは反対側のまま
        for i in 0..5 {
            p.step(2.0 + f64::from(i) * 0.3, None, None, Intent::Any);
        }
        assert_eq!(p.state, AvoidState::Cruise);
        assert_eq!(p.failures, 0);
    }

    #[test]
    fn camera_blocked_needs_two_frames_and_slows_down() {
        let mut p = policy();
        let d = p.step(0.0, None, Some(0.7), Intent::Any);
        assert_eq!((d.state, d.slow), (AvoidState::Cruise, true));
        let d = p.step(0.1, None, Some(0.7), Intent::Any);
        assert_eq!((d.event, d.reason), (Some(AvoidEvent::Begin), "camera"));
        // 曲がっている間は、確率が出口の値より下がるまで空いたとみなさない
        for i in 0..12 {
            let d = p.step(0.2 + f64::from(i) * 0.1, None, Some(0.5), Intent::Any);
            assert_eq!(d.state, AvoidState::Avoid);
        }
    }

    #[test]
    fn avoiding_turns_at_least_the_floor_toward_the_intent() {
        let decision = |intent, state| Decision {
            intent,
            slow: false,
            state,
            event: None,
            reason: "",
        };
        let right = decision(Intent::Right, AvoidState::Avoid);
        assert_eq!(turn_floor(8, &right, 20), 20);
        assert_eq!(turn_floor(-5, &right, 20), 20);
        assert_eq!(turn_floor(30, &right, 20), 30); // 強く切っていればそのまま
        let left = decision(Intent::Left, AvoidState::Avoid);
        assert_eq!(turn_floor(-8, &left, 20), -20);
        assert_eq!(turn_floor(-30, &left, 20), -30);
        // 避けていない時（上の層が意図を与えただけの時も含む）はモデルのまま
        assert_eq!(
            turn_floor(8, &decision(Intent::Right, AvoidState::Cruise), 20),
            8
        );
        assert_eq!(
            turn_floor(3, &decision(Intent::Any, AvoidState::Avoid), 20),
            3
        );
    }
}
