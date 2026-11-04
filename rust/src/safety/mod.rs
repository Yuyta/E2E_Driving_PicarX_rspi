//! Pi 内の安全判断。時刻を引数で受け取り、実機なしで検証できるようにしている。

pub mod reflex;

use crate::api::models::DriveDirection;
use crate::picarx::ultrasonic::FAR_DISTANCE_CM;
use crate::sensor_monitor::SensorSnapshot;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// これより近い障害物があれば前進しない
pub const MIN_SAFE_DISTANCE_CM: f64 = 15.0;
/// 超音波の値をこの時間だけ有効とする。外れ値フィルターと、測定失敗の数回分を許容する。
pub const DISTANCE_MAX_AGE: Duration = Duration::from_millis(300);
/// 床センサーの値をこの時間だけ有効とする
pub const FLOOR_MAX_AGE: Duration = Duration::from_millis(200);
/// 反射ループの更新がこの時間途絶えたら走行を拒否する
/// 反射が戻らない間の速度の上限。音を吸う物にぶつかる場合に備える。実機で調整する
pub const FAR_MAX_SPEED: i32 = 60;
const REFLEX_MAX_GAP: Duration = Duration::from_millis(250);
const EVENT_CAPACITY: usize = 256;

// 段差閾値の校正値。実機では、白い床が 1000 以上、机の縁へ近づくと 183 → 91 → 46 と段階的に
// 下がり、縁の外は 7〜26 だった。低い閾値では検知が遅れ、速度20で前輪と縁の間が 2cm だった。
/// 校正に必要なサンプル数
pub const CALIBRATION_SAMPLES: usize = 15;
/// 床がこの値より暗いと段差と区別できないので校正を拒否する
const MIN_FLOOR_ADC: u16 = 60;
const CLIFF_THRESHOLD_PERCENT: u32 = 35;
/// 明るい床で校正しても、暗い敷物を段差と誤認しないための上限。暗い床では実機で要確認
const MAX_CLIFF_THRESHOLD: u16 = 300;

/// 停止または走行拒否の理由
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SafetyReason {
    EmergencyStop,
    Obstacle,
    Cliff,
    DistanceUnknown,
    FloorUnknown,
    CliffUncalibrated,
    ReflexStalled,
    TtlExpired,
}

impl SafetyReason {
    pub fn message(self) -> &'static str {
        match self {
            Self::EmergencyStop => "緊急停止中です。解除するまで走行できません",
            Self::Obstacle => "安全ガード作動: 障害物が近すぎます",
            Self::Cliff => "安全ガード作動: 段差を検知しました",
            Self::DistanceUnknown => "距離が不明なため前進できません",
            Self::FloorUnknown => "床センサーの値が不明なため走行できません",
            Self::CliffUncalibrated => "段差の校正が済んでいません。床に置いて校正してください",
            Self::ReflexStalled => "反射ループが動作していないため走行できません",
            Self::TtlExpired => "走行コマンドの期限が切れました",
        }
    }
}

/// `/api/events` で返す事象の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    /// 反射ループが走行を止めた
    ReflexStop(SafetyReason),
    EmergencyStop,
    EmergencyStopReset,
    /// モーターへの停止指令が失敗した。成功するまで再送する
    StopFailed,
    CliffCalibrated,
    CliffCalibrationFailed,
}

#[derive(Debug, Clone, Serialize)]
pub struct SafetyEvent {
    pub id: u64,
    pub at_unix_ms: i64,
    #[serde(flatten)]
    pub kind: EventKindField,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub direction: Option<DriveDirection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance_cm: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grayscale: Option<[u16; 3]>,
    /// 危険を示す値を取得してから停止指令が完了するまでの時間
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_latency_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// JSON では `{"kind": "reflex_stop", "reason": "obstacle"}` の形にする
#[derive(Debug, Clone, Serialize)]
pub struct EventKindField {
    pub kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<SafetyReason>,
}

impl From<EventKind> for EventKindField {
    fn from(kind: EventKind) -> Self {
        let (kind, reason) = match kind {
            EventKind::ReflexStop(reason) => ("reflex_stop", Some(reason)),
            EventKind::EmergencyStop => ("emergency_stop", None),
            EventKind::EmergencyStopReset => ("emergency_stop_reset", None),
            EventKind::StopFailed => ("stop_failed", None),
            EventKind::CliffCalibrated => ("cliff_calibrated", None),
            EventKind::CliffCalibrationFailed => ("cliff_calibration_failed", None),
        };
        Self { kind, reason }
    }
}

/// 反射ループが検知した危険
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hazard {
    pub reason: SafetyReason,
    /// 危険が確定した時刻。停止までの時間の起点。
    pub since: Instant,
    pub direction: DriveDirection,
    pub distance_cm: Option<f64>,
    pub grayscale: Option<[u16; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveMotion {
    pub direction: DriveDirection,
    pub speed: i32,
    pub deadline: Instant,
}

/// 校正結果。閾値はセンサーごとに持つ。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CliffCalibration {
    pub baseline: [u16; 3],
    pub thresholds: [u16; 3],
}

impl CliffCalibration {
    /// 床で測った値から閾値を決める。段差と区別できない値なら拒否する。
    pub fn from_samples(samples: &[[u16; 3]]) -> Result<Self, String> {
        if samples.len() < CALIBRATION_SAMPLES {
            return Err(format!(
                "床センサーのサンプルが足りません ({}/{CALIBRATION_SAMPLES})",
                samples.len()
            ));
        }

        let mut baseline = [0u16; 3];
        let mut thresholds = [0u16; 3];
        for sensor in 0..3 {
            let mut values: Vec<u16> = samples.iter().map(|s| s[sensor]).collect();
            values.sort_unstable();
            let median = values[values.len() / 2];
            if median < MIN_FLOOR_ADC {
                return Err(format!(
                    "床センサー{sensor}の値が低すぎます ({median} < {MIN_FLOOR_ADC})。\
                     車体を床に置いて、段差から離してください"
                ));
            }
            baseline[sensor] = median;
            let scaled = u32::from(median) * CLIFF_THRESHOLD_PERCENT / 100;
            thresholds[sensor] = scaled.min(u32::from(MAX_CLIFF_THRESHOLD)) as u16;
        }
        Ok(Self {
            baseline,
            thresholds,
        })
    }

    pub fn is_cliff(&self, floor: [u16; 3]) -> [bool; 3] {
        [
            floor[0] < self.thresholds[0],
            floor[1] < self.thresholds[1],
            floor[2] < self.thresholds[2],
        ]
    }
}

/// 反射が戻らず「遠い」と判断した値かどうか
pub fn is_far(distance_cm: f64) -> bool {
    distance_cm >= FAR_DISTANCE_CM
}

/// 反射が戻らない間は、前進の速度を抑える
pub fn limit_speed(direction: DriveDirection, speed: i32, distance_cm: Option<f64>) -> i32 {
    if direction == DriveDirection::Forward && distance_cm.is_some_and(is_far) {
        speed.min(FAR_MAX_SPEED)
    } else {
        speed
    }
}

/// 反射ループの動作状況
#[derive(Debug, Clone, Copy, Default)]
pub struct ReflexStats {
    pub last_tick: Option<Instant>,
    pub ticks: u64,
    pub interval_ms_ema: f64,
    pub max_interval_ms: f64,
}

/// 安全状態。ハンドラと反射ループで共有し、モーター指令と同じロックの中で更新する。
pub struct SafetyController {
    min_distance_cm: f64,
    cliff_guard_enabled: bool,
    cliff: Option<CliffCalibration>,
    motion: Option<ActiveMotion>,
    emergency_stopped: bool,
    stop_pending: bool,
    steer_angle: i32,
    reflex: ReflexStats,
    events: VecDeque<SafetyEvent>,
    next_event_id: u64,
}

impl SafetyController {
    pub fn new(cliff_guard_enabled: bool) -> Self {
        Self {
            min_distance_cm: MIN_SAFE_DISTANCE_CM,
            cliff_guard_enabled,
            cliff: None,
            motion: None,
            emergency_stopped: false,
            stop_pending: false,
            steer_angle: 0,
            reflex: ReflexStats::default(),
            events: VecDeque::with_capacity(EVENT_CAPACITY),
            next_event_id: 1,
        }
    }

    /// `direction` へ進むと危険かどうかを判定する。
    ///
    /// 超音波と段差センサーは前方にしかないので、前方の危険は後退を妨げない。
    /// 床センサーが読めない時は I2C 自体の異常が疑われるため、後退も止める。
    fn hazard_for(
        &self,
        direction: DriveDirection,
        sensors: &SensorSnapshot,
        now: Instant,
    ) -> Option<Hazard> {
        let distance = sensors
            .distance_cm
            .and_then(|r| r.fresh(now, DISTANCE_MAX_AGE));
        let floor = sensors.grayscale.and_then(|r| r.fresh(now, FLOOR_MAX_AGE));
        let hazard = |reason, since| {
            Some(Hazard {
                reason,
                since,
                direction,
                distance_cm: distance,
                grayscale: floor,
            })
        };

        if self.cliff_guard_enabled && floor.is_none() {
            let since = sensors
                .grayscale
                .map_or(now, |r| r.at + FLOOR_MAX_AGE)
                .min(now);
            return hazard(SafetyReason::FloorUnknown, since);
        }
        if direction != DriveDirection::Forward {
            return None;
        }

        if self.cliff_guard_enabled {
            let Some(cliff) = self.cliff else {
                return hazard(SafetyReason::CliffUncalibrated, now);
            };
            if let (Some(reading), Some(values)) = (sensors.grayscale, floor)
                && cliff.is_cliff(values).contains(&true)
            {
                return hazard(SafetyReason::Cliff, reading.at);
            }
        }

        let Some(reading) = sensors.distance_cm else {
            return hazard(SafetyReason::DistanceUnknown, now);
        };
        match distance {
            None => hazard(
                SafetyReason::DistanceUnknown,
                (reading.at + DISTANCE_MAX_AGE).min(now),
            ),
            Some(cm) if cm < self.min_distance_cm => hazard(SafetyReason::Obstacle, reading.at),
            Some(_) => None,
        }
    }

    /// 走行を始めてよいか判定する。`Stop` は常に許可する。
    pub fn check_start(
        &self,
        direction: DriveDirection,
        sensors: &SensorSnapshot,
        now: Instant,
    ) -> Result<(), SafetyReason> {
        if direction == DriveDirection::Stop {
            return Ok(());
        }
        if self.emergency_stopped {
            return Err(SafetyReason::EmergencyStop);
        }
        if self.reflex_stalled(now) {
            return Err(SafetyReason::ReflexStalled);
        }
        match self.hazard_for(direction, sensors, now) {
            Some(hazard) => Err(hazard.reason),
            None => Ok(()),
        }
    }

    pub fn begin_motion(
        &mut self,
        direction: DriveDirection,
        speed: i32,
        ttl: Duration,
        now: Instant,
    ) {
        self.motion = Some(ActiveMotion {
            direction,
            speed,
            deadline: now + ttl,
        });
    }

    pub fn clear_motion(&mut self) {
        self.motion = None;
    }

    pub fn motion(&self) -> Option<ActiveMotion> {
        self.motion
    }

    /// 反射ループの1周分の判定。止めるべきなら危険の内容を返す。
    pub fn evaluate(&mut self, sensors: &SensorSnapshot, now: Instant) -> Option<Hazard> {
        self.record_tick(now);
        let motion = self.motion?;
        if let Some(hazard) = self.hazard_for(motion.direction, sensors, now) {
            return Some(hazard);
        }
        (now >= motion.deadline).then(|| Hazard {
            reason: SafetyReason::TtlExpired,
            since: motion.deadline,
            direction: motion.direction,
            distance_cm: sensors.distance_cm.map(|r| r.value),
            grayscale: sensors.grayscale.map(|r| r.value),
        })
    }

    fn record_tick(&mut self, now: Instant) {
        if let Some(last) = self.reflex.last_tick {
            let interval = now.saturating_duration_since(last).as_secs_f64() * 1000.0;
            self.reflex.interval_ms_ema = if self.reflex.ticks <= 1 {
                interval
            } else {
                self.reflex.interval_ms_ema * 0.95 + interval * 0.05
            };
            self.reflex.max_interval_ms = self.reflex.max_interval_ms.max(interval);
        }
        self.reflex.last_tick = Some(now);
        self.reflex.ticks += 1;
    }

    fn reflex_stalled(&self, now: Instant) -> bool {
        self.reflex
            .last_tick
            .is_none_or(|last| now.saturating_duration_since(last) > REFLEX_MAX_GAP)
    }

    pub fn reflex_stats(&self) -> ReflexStats {
        self.reflex
    }

    pub fn is_emergency_stopped(&self) -> bool {
        self.emergency_stopped
    }

    pub fn set_emergency_stopped(&mut self, stopped: bool) {
        self.emergency_stopped = stopped;
        if stopped {
            self.motion = None;
        }
        let kind = if stopped {
            EventKind::EmergencyStop
        } else {
            EventKind::EmergencyStopReset
        };
        self.push_event(kind, None, None, None, None, None);
    }

    /// 停止指令の結果を記録する。失敗したら反射ループが再送する。
    pub fn note_stop_result(&mut self, result: &Result<(), String>) {
        match result {
            Ok(()) => self.stop_pending = false,
            Err(error) => {
                if !self.stop_pending {
                    self.push_event(
                        EventKind::StopFailed,
                        None,
                        None,
                        None,
                        None,
                        Some(error.clone()),
                    );
                }
                self.stop_pending = true;
            }
        }
    }

    pub fn stop_pending(&self) -> bool {
        self.stop_pending
    }

    pub fn record_hazard_stop(&mut self, hazard: &Hazard, stopped_at: Instant) {
        // 軌跡に沿った後退は1歩ごとに期限切れになる。記録すると上限を超えて、
        // 障害物や段差の記録を押し出してしまう。
        if hazard.reason == SafetyReason::TtlExpired && hazard.direction == DriveDirection::Backward
        {
            return;
        }
        let latency = stopped_at
            .saturating_duration_since(hazard.since)
            .as_secs_f64()
            * 1000.0;
        self.push_event(
            EventKind::ReflexStop(hazard.reason),
            Some(hazard.direction),
            hazard.distance_cm,
            hazard.grayscale,
            Some(latency),
            None,
        );
    }

    pub fn cliff_guard_enabled(&self) -> bool {
        self.cliff_guard_enabled
    }

    pub fn cliff(&self) -> Option<CliffCalibration> {
        self.cliff
    }

    pub fn set_cliff(&mut self, calibration: CliffCalibration) {
        self.cliff = Some(calibration);
        self.push_event(
            EventKind::CliffCalibrated,
            None,
            None,
            Some(calibration.baseline),
            None,
            None,
        );
    }

    pub fn record_calibration_failure(&mut self, error: &str) {
        self.push_event(
            EventKind::CliffCalibrationFailed,
            None,
            None,
            None,
            None,
            Some(error.to_string()),
        );
    }

    pub fn steer_angle(&self) -> i32 {
        self.steer_angle
    }

    pub fn set_steer_angle(&mut self, angle: i32) {
        self.steer_angle = angle;
    }

    fn push_event(
        &mut self,
        kind: EventKind,
        direction: Option<DriveDirection>,
        distance_cm: Option<f64>,
        grayscale: Option<[u16; 3]>,
        stop_latency_ms: Option<f64>,
        detail: Option<String>,
    ) {
        if self.events.len() == EVENT_CAPACITY {
            self.events.pop_front();
        }
        self.events.push_back(SafetyEvent {
            id: self.next_event_id,
            at_unix_ms: chrono::Utc::now().timestamp_millis(),
            kind: kind.into(),
            direction,
            distance_cm,
            grayscale,
            stop_latency_ms,
            detail,
        });
        self.next_event_id += 1;
    }

    pub fn latest_event_id(&self) -> u64 {
        self.next_event_id - 1
    }

    /// `after` より新しい事象を返す。再起動で ID が戻った場合は全件を返す。
    pub fn events_after(&self, after: u64) -> Vec<SafetyEvent> {
        let after = if after > self.latest_event_id() {
            0
        } else {
            after
        };
        self.events
            .iter()
            .filter(|event| event.id > after)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sensor_monitor::Timed;

    const FLOOR: [u16; 3] = [500, 500, 500];

    fn sensors(now: Instant, distance: f64, floor: [u16; 3]) -> SensorSnapshot {
        SensorSnapshot {
            distance_cm: Some(Timed::new(distance, now)),
            grayscale: Some(Timed::new(floor, now)),
            battery: None,
        }
    }

    fn calibrated(now: Instant) -> SafetyController {
        let mut safety = SafetyController::new(true);
        safety.set_cliff(CliffCalibration::from_samples(&[FLOOR; CALIBRATION_SAMPLES]).unwrap());
        safety.evaluate(&SensorSnapshot::default(), now);
        safety
    }

    #[test]
    fn no_echo_allows_forward_at_a_limited_speed() {
        let now = Instant::now();
        let safety = calibrated(now);
        assert_eq!(
            safety.check_start(
                DriveDirection::Forward,
                &sensors(now, FAR_DISTANCE_CM, FLOOR),
                now
            ),
            Ok(())
        );
        let far = Some(FAR_DISTANCE_CM);
        assert_eq!(
            limit_speed(DriveDirection::Forward, 100, far),
            FAR_MAX_SPEED
        );
        assert_eq!(limit_speed(DriveDirection::Forward, 40, far), 40);
        assert_eq!(limit_speed(DriveDirection::Backward, 100, far), 100);
        assert_eq!(limit_speed(DriveDirection::Forward, 100, Some(100.0)), 100);
        assert_eq!(limit_speed(DriveDirection::Forward, 100, None), 100);
    }

    #[test]
    fn forward_is_allowed_when_all_sensors_are_good() {
        let now = Instant::now();
        let safety = calibrated(now);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &sensors(now, 40.0, FLOOR), now),
            Ok(())
        );
    }

    #[test]
    fn obstacle_blocks_forward_but_not_backward() {
        let now = Instant::now();
        let safety = calibrated(now);
        let snap = sensors(now, 10.0, FLOOR);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &snap, now),
            Err(SafetyReason::Obstacle)
        );
        assert_eq!(
            safety.check_start(DriveDirection::Backward, &snap, now),
            Ok(())
        );
    }

    #[test]
    fn cliff_blocks_forward_but_allows_backing_away() {
        let now = Instant::now();
        let safety = calibrated(now);
        let snap = sensors(now, 40.0, [500, 8, 500]);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &snap, now),
            Err(SafetyReason::Cliff)
        );
        assert_eq!(
            safety.check_start(DriveDirection::Backward, &snap, now),
            Ok(())
        );
    }

    #[test]
    fn unknown_sensors_refuse_motion_instead_of_assuming_safe() {
        let now = Instant::now();
        let safety = calibrated(now);

        let no_distance = SensorSnapshot {
            distance_cm: None,
            ..sensors(now, 0.0, FLOOR)
        };
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &no_distance, now),
            Err(SafetyReason::DistanceUnknown)
        );

        let no_floor = SensorSnapshot {
            grayscale: None,
            ..sensors(now, 40.0, FLOOR)
        };
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &no_floor, now),
            Err(SafetyReason::FloorUnknown)
        );
        assert_eq!(
            safety.check_start(DriveDirection::Backward, &no_floor, now),
            Err(SafetyReason::FloorUnknown)
        );
    }

    #[test]
    fn stale_readings_count_as_unknown() {
        let t0 = Instant::now();
        let later = t0 + DISTANCE_MAX_AGE + Duration::from_millis(1);
        let mut safety = calibrated(t0);
        safety.evaluate(&SensorSnapshot::default(), later);
        let snap = SensorSnapshot {
            distance_cm: Some(Timed::new(40.0, t0)),
            grayscale: Some(Timed::new(FLOOR, later)),
            battery: None,
        };
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &snap, later),
            Err(SafetyReason::DistanceUnknown)
        );
    }

    #[test]
    fn uncalibrated_cliff_guard_blocks_forward_only() {
        let now = Instant::now();
        let mut safety = SafetyController::new(true);
        safety.evaluate(&SensorSnapshot::default(), now);
        let snap = sensors(now, 40.0, FLOOR);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &snap, now),
            Err(SafetyReason::CliffUncalibrated)
        );
        assert_eq!(
            safety.check_start(DriveDirection::Backward, &snap, now),
            Ok(())
        );
    }

    #[test]
    fn disabled_cliff_guard_ignores_the_floor_sensors() {
        let now = Instant::now();
        let mut safety = SafetyController::new(false);
        safety.evaluate(&SensorSnapshot::default(), now);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &sensors(now, 40.0, [0, 0, 0]), now),
            Ok(())
        );
    }

    #[test]
    fn evaluate_stops_when_ttl_expires() {
        let t0 = Instant::now();
        let mut safety = calibrated(t0);
        let ttl = Duration::from_millis(300);
        safety.begin_motion(DriveDirection::Forward, 30, ttl, t0);

        let before = t0 + Duration::from_millis(299);
        assert_eq!(safety.evaluate(&sensors(before, 40.0, FLOOR), before), None);

        let after = t0 + ttl;
        let hazard = safety
            .evaluate(&sensors(after, 40.0, FLOOR), after)
            .unwrap();
        assert_eq!(hazard.reason, SafetyReason::TtlExpired);
        assert_eq!(hazard.since, after);
    }

    #[test]
    fn renewing_a_command_extends_the_deadline() {
        let t0 = Instant::now();
        let mut safety = calibrated(t0);
        let ttl = Duration::from_millis(300);
        safety.begin_motion(DriveDirection::Forward, 30, ttl, t0);
        let renew = t0 + Duration::from_millis(200);
        safety.begin_motion(DriveDirection::Forward, 30, ttl, renew);

        let check = t0 + Duration::from_millis(400);
        assert_eq!(safety.evaluate(&sensors(check, 40.0, FLOOR), check), None);
    }

    #[test]
    fn evaluate_stops_for_hazards_while_moving() {
        let t0 = Instant::now();
        let ttl = Duration::from_secs(1);
        for (snap, reason) in [
            (sensors(t0, 10.0, FLOOR), SafetyReason::Obstacle),
            (sensors(t0, 40.0, [5, 5, 5]), SafetyReason::Cliff),
            (SensorSnapshot::default(), SafetyReason::FloorUnknown),
        ] {
            let mut safety = calibrated(t0);
            safety.begin_motion(DriveDirection::Forward, 30, ttl, t0);
            assert_eq!(safety.evaluate(&snap, t0).unwrap().reason, reason);
        }
    }

    #[test]
    fn evaluate_does_nothing_while_idle() {
        let now = Instant::now();
        let mut safety = calibrated(now);
        assert_eq!(safety.evaluate(&sensors(now, 1.0, [0, 0, 0]), now), None);
    }

    #[test]
    fn hazard_records_the_reading_time_as_latency_origin() {
        let t0 = Instant::now();
        let check = t0 + Duration::from_millis(30);
        let mut safety = calibrated(t0);
        safety.begin_motion(DriveDirection::Forward, 30, Duration::from_secs(1), t0);
        let snap = SensorSnapshot {
            distance_cm: Some(Timed::new(10.0, t0)),
            grayscale: Some(Timed::new(FLOOR, check)),
            battery: None,
        };
        let hazard = safety.evaluate(&snap, check).unwrap();
        assert_eq!(hazard.since, t0);

        safety.record_hazard_stop(&hazard, check);
        let event = safety.events_after(1).pop().unwrap();
        assert_eq!(event.kind.reason, Some(SafetyReason::Obstacle));
        assert!((event.stop_latency_ms.unwrap() - 30.0).abs() < 0.001);
    }

    #[test]
    fn emergency_stop_latches_until_cleared_and_cancels_motion() {
        let now = Instant::now();
        let mut safety = calibrated(now);
        safety.begin_motion(DriveDirection::Forward, 30, Duration::from_secs(1), now);
        safety.set_emergency_stopped(true);
        assert_eq!(safety.motion(), None);

        let snap = sensors(now, 40.0, FLOOR);
        assert_eq!(
            safety.check_start(DriveDirection::Backward, &snap, now),
            Err(SafetyReason::EmergencyStop)
        );
        assert_eq!(safety.check_start(DriveDirection::Stop, &snap, now), Ok(()));

        safety.set_emergency_stopped(false);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &snap, now),
            Ok(())
        );
    }

    #[test]
    fn motion_is_refused_when_the_reflex_loop_stops_ticking() {
        let t0 = Instant::now();
        let safety = calibrated(t0);
        let later = t0 + REFLEX_MAX_GAP + Duration::from_millis(1);
        assert_eq!(
            safety.check_start(DriveDirection::Forward, &sensors(later, 40.0, FLOOR), later),
            Err(SafetyReason::ReflexStalled)
        );
    }

    #[test]
    fn calibration_scales_per_sensor_and_caps_the_threshold() {
        let cliff = CliffCalibration::from_samples(&[[100, 120, 1000]; CALIBRATION_SAMPLES])
            .expect("floor values are valid");
        assert_eq!(cliff.baseline, [100, 120, 1000]);
        assert_eq!(cliff.thresholds, [35, 42, MAX_CLIFF_THRESHOLD]);
    }

    #[test]
    fn calibration_uses_the_median_to_ignore_a_noisy_sample() {
        let mut samples = [FLOOR; CALIBRATION_SAMPLES];
        samples[0] = [0, 4095, 0];
        let cliff = CliffCalibration::from_samples(&samples).unwrap();
        assert_eq!(cliff.baseline, FLOOR);
    }

    #[test]
    fn calibration_is_rejected_when_lifted_or_over_an_edge() {
        // 記録にある段差時の値。空中や段差の上で校正してしまうのを防ぐ
        assert!(CliffCalibration::from_samples(&[[8, 13, 11]; CALIBRATION_SAMPLES]).is_err());
        assert!(CliffCalibration::from_samples(&[[500, 500, 11]; CALIBRATION_SAMPLES]).is_err());
        assert!(CliffCalibration::from_samples(&[FLOOR; 3]).is_err());
    }

    #[test]
    fn calibration_detects_the_approach_to_a_real_edge_on_a_bright_floor() {
        // 実機の白い床と机の縁で記録した値
        let cliff =
            CliffCalibration::from_samples(&[[1178, 1175, 1240]; CALIBRATION_SAMPLES]).unwrap();
        assert_eq!(cliff.thresholds, [MAX_CLIFF_THRESHOLD; 3]);
        assert_eq!(cliff.is_cliff([1038, 1050, 1081]), [false, false, false]);
        assert_eq!(cliff.is_cliff([989, 1009, 1020]), [false, false, false]);
        assert_eq!(cliff.is_cliff([183, 136, 83]), [true, true, true]);
    }

    #[test]
    fn calibration_on_a_dark_floor_scales_below_the_cap() {
        let cliff =
            CliffCalibration::from_samples(&[[245, 239, 240]; CALIBRATION_SAMPLES]).unwrap();
        assert_eq!(cliff.thresholds, [85, 83, 84]);
        assert_eq!(cliff.is_cliff([200, 200, 200]), [false, false, false]);
        assert_eq!(cliff.is_cliff([8, 13, 11]), [true, true, true]);
    }

    #[test]
    fn backward_ttl_stops_are_not_logged_but_hazards_are() {
        let now = Instant::now();
        let mut safety = SafetyController::new(true);
        let latest = safety.latest_event_id();
        let stop = |reason, direction| Hazard {
            reason,
            since: now,
            direction,
            distance_cm: None,
            grayscale: None,
        };
        safety.record_hazard_stop(
            &stop(SafetyReason::TtlExpired, DriveDirection::Backward),
            now,
        );
        assert_eq!(safety.latest_event_id(), latest);

        safety.record_hazard_stop(
            &stop(SafetyReason::TtlExpired, DriveDirection::Forward),
            now,
        );
        safety.record_hazard_stop(
            &stop(SafetyReason::FloorUnknown, DriveDirection::Backward),
            now,
        );
        assert_eq!(safety.latest_event_id(), latest + 2);
    }

    #[test]
    fn event_log_is_bounded_and_filters_by_id() {
        let mut safety = SafetyController::new(true);
        for _ in 0..EVENT_CAPACITY + 5 {
            safety.set_emergency_stopped(true);
        }
        let latest = safety.latest_event_id();
        assert_eq!(latest, (EVENT_CAPACITY + 5) as u64);
        assert_eq!(safety.events_after(0).len(), EVENT_CAPACITY);
        assert_eq!(safety.events_after(latest - 2).len(), 2);
        // 再起動後の古い ID には全件を返す
        assert_eq!(safety.events_after(latest + 100).len(), EVENT_CAPACITY);
    }

    #[test]
    fn failed_stop_is_reported_once_and_cleared_on_success() {
        let mut safety = SafetyController::new(true);
        let failure = Err("i2c".to_string());
        safety.note_stop_result(&failure);
        safety.note_stop_result(&failure);
        assert!(safety.stop_pending());
        assert_eq!(safety.events_after(0).len(), 1);
        safety.note_stop_result(&Ok(()));
        assert!(!safety.stop_pending());
    }
}
