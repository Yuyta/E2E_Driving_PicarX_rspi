//! 自律走行のループ（走行のスレッドで動く）。PC の `autonomous_driver.run` を Pi へ移したもの。
//!
//! 1周の流れ: 新しいフレームを待つ → 人の指令が来ていれば何もしない → 前へ進めるかを安全の
//! 判定で確かめる（障害物と段差なら後退して向きを変える。後退を繰り返す時と、段差から
//! 抜けられない時は、動き回らずに走行を終えて人を待つ） → 前処理 → 引っかかりの確認 →
//! 推論と避ける判断 → 速度を決めて指令する。指令の期限（TTL）が切れれば車両は止まるので、
//! このスレッドが止まっても走り続けない。

use super::avoidance::{AVOID_MIN_STEER_DEG, AvoidanceConfig, AvoidancePolicy, turn_floor};
use super::model::Steering;
use super::preprocess::{decode_jpeg, resize, to_tensor};
use super::runlog::{RunLog, unix_now};
use super::speed::SpeedLimiter;
use super::stuck::{StuckConfig, StuckDetector, thumbnail};
use super::trajectory::{MAX_STEER_DEG, Trajectory};
use super::{Intent, RunParams, Shared, StopRequest};
use crate::api::handlers::{AppState, MoveError};
use crate::api::models::DriveDirection;
use crate::rpi_camera::Frame;
use crate::safety::{DISTANCE_MAX_AGE, SafetyReason, is_far};
use serde_json::json;
use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// 新しいフレームを待つ時間
const FRAME_WAIT: Duration = Duration::from_millis(500);
/// これより古いフレームでは走らない
const FRAME_MAX_AGE: Duration = Duration::from_millis(500);
/// フレームがこの時間届かなければ走行を終える
const CAMERA_LOST: Duration = Duration::from_secs(3);
/// 前へ進めない理由が分からない時（センサーの値が不明など）に待つ時間
const WAIT_SLEEP: Duration = Duration::from_millis(100);
/// 障害物の前でこの周回数（約1秒）止められたままなら、後退して抜ける
const OBSTACLE_ESCAPE_CYCLES: u32 = 10;
/// 後退する時間
const REVERSE_MS: u64 = 1200;
const STUCK_REVERSE_MS: u64 = 1000;
/// 前進を始めてからこの時間は、画像が変わらなくても引っかかりとみなさない（加速の分）
const STUCK_SETTLE: Duration = Duration::from_millis(500);
/// この時間のうちにこの回数引っかかったら、人の助けが要るとして走行を終える
const STUCK_WINDOW_S: f64 = 30.0;
const STUCK_LIMIT: usize = 3;
/// 後退して抜ける動作（障害物、段差、引っかかり）を、この時間のうちにこの回数まで行う。
/// 超えたら後退と前進を繰り返さず、人の助けが要るとして走行を終える
const ESCAPE_WINDOW_S: f64 = 30.0;
const ESCAPE_LIMIT: usize = 4;
/// 段差は落ちれば壊れるので、この時間のうちに2回目を見つけたら走行を終える
const CLIFF_WINDOW_S: f64 = 60.0;
const CLIFF_LIMIT: usize = 2;
/// 人の助けをこれだけ待っても来なければ、走行を終える
const HELP_TIMEOUT: Duration = Duration::from_secs(180);

/// 走行を終えた理由
struct End {
    trigger: Option<String>,
    note: String,
}

impl End {
    fn safety(trigger: &str, note: impl Into<String>) -> Self {
        Self {
            trigger: Some(trigger.to_string()),
            note: note.into(),
        }
    }

    fn quiet(note: impl Into<String>) -> Self {
        Self {
            trigger: None,
            note: note.into(),
        }
    }
}

impl From<StopRequest> for End {
    fn from(r: StopRequest) -> Self {
        Self {
            trigger: r.trigger,
            note: r.note,
        }
    }
}

/// 時刻を足し、`window_s` より古いものを捨てて、残りの数を返す
fn push_within(times: &mut VecDeque<f64>, at: f64, window_s: f64) -> usize {
    times.push_back(at);
    while times.front().is_some_and(|&t| at - t > window_s) {
        times.pop_front();
    }
    times.len()
}

fn round1(v: f64) -> f64 {
    (v * 10.0).round() / 10.0
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

fn ema(previous: Option<f64>, value: f64) -> Option<f64> {
    Some(previous.map_or(value, |p| p * 0.9 + value * 0.1))
}

pub struct Runner {
    state: Arc<AppState>,
    shared: Arc<Shared>,
    model: Box<dyn Steering>,
    log: RunLog,
    params: RunParams,
    avoidance: AvoidancePolicy,
    limiter: SpeedLimiter,
    trajectory: Trajectory,
    stuck: StuckDetector,
    t0: Instant,
    last_cycle: Instant,
    /// 前の周に推論した意図（塞がっているかは意図によらないので、まずこれで推論する）
    last_intent: Option<Intent>,
    human: bool,
    takeovers: u32,
    obstacle_cycles: u32,
    obstacle_logged: bool,
    /// 今の段差の検出で、もう後退したか（後退しても段差のままなら走行を終える）
    cliff_escaped: bool,
    stuck_times: VecDeque<f64>,
    escape_times: VecDeque<f64>,
    cliff_times: VecDeque<f64>,
    cycle_log_failed: bool,
    /// 人の助けを待ち始めた時刻と理由
    help: Option<(Instant, String)>,
    help_requests: u32,
    /// 前進の指令が通り続けている始まり
    forward_since: Option<Instant>,
    frames: u64,
}

impl Runner {
    pub fn new(
        state: Arc<AppState>,
        shared: Arc<Shared>,
        model: Box<dyn Steering>,
        log: RunLog,
        params: RunParams,
    ) -> Result<Self, String> {
        let now = Instant::now();
        Ok(Self {
            state,
            shared,
            model,
            log,
            params,
            avoidance: AvoidancePolicy::new(AvoidanceConfig::default()),
            limiter: SpeedLimiter::new(params.speed, params.min_speed)?,
            trajectory: Trajectory::new(100),
            stuck: StuckDetector::new(StuckConfig::default()),
            t0: now,
            last_cycle: now,
            last_intent: None,
            human: false,
            takeovers: 0,
            obstacle_cycles: 0,
            obstacle_logged: false,
            cliff_escaped: false,
            stuck_times: VecDeque::new(),
            escape_times: VecDeque::new(),
            cliff_times: VecDeque::new(),
            cycle_log_failed: false,
            help: None,
            help_requests: 0,
            forward_since: None,
            frames: 0,
        })
    }

    fn secs(&self, at: Instant) -> f64 {
        at.duration_since(self.t0).as_secs_f64()
    }

    fn set_phase(&self, phase: &str, detail: Option<String>) {
        let mut status = self.shared.status();
        status.phase = phase.to_string();
        status.detail = detail;
    }

    fn note_event(&self, event: String) {
        self.shared.status().last_event = Some(event);
    }

    fn intervention(&mut self, trigger: &str, note: &str, extra: serde_json::Value) {
        if let Err(error) = self.log.intervention(trigger, unix_now(), note, extra) {
            eprintln!("⚠ 走行の記録に書けません: {error}");
        }
        self.note_event(format!("{trigger}: {note}"));
        let mut status = self.shared.status();
        status.interventions = self.log.counts;
    }

    /// 1周分の判断を `cycles.jsonl` に残す。書けなくなっても走行は続け、警告は1回だけ出す
    fn log_cycle(&mut self, record: serde_json::Value) {
        if let Err(error) = self.log.cycle(unix_now(), record)
            && !self.cycle_log_failed
        {
            self.cycle_log_failed = true;
            eprintln!("⚠ 1周ごとの記録に書けません: {error}");
        }
    }

    pub fn run(mut self) {
        let end = self.drive_loop();
        if let Err(error) = self.state.stop_motors() {
            eprintln!("⚠ 自律走行の終了時に止められませんでした: {error}");
        }
        let _ = self.state.command_steer(0);
        if let Some(trigger) = end.trigger.clone() {
            self.intervention(&trigger, &end.note, json!({"ends_run": true}));
        }
        if let Err(error) = self.log.close(unix_now(), &end.note) {
            eprintln!("⚠ 走行の記録を閉じられません: {error}");
        }
        let mut status = self.shared.status();
        status.running = false;
        status.phase = "stopped".to_string();
        status.detail = None;
        status.stopped_reason = Some(end.note);
        status.autonomous_s = self.log.autonomous_s;
        status.human_s = self.log.human_s;
        status.interventions = self.log.counts;
    }

    /// 新しいフレームを待つ。止めるよう言われたら None
    fn wait_frame(&self, last_seq: u64) -> Option<Arc<Frame>> {
        let rx = self.state.camera.subscribe();
        let deadline = Instant::now() + FRAME_WAIT;
        loop {
            if let Some(frame) = rx.borrow().as_ref()
                && frame.seq > last_seq
                && frame.captured_at.elapsed() <= FRAME_MAX_AGE
            {
                return Some(Arc::clone(frame));
            }
            if Instant::now() >= deadline || self.shared.stop_requested() {
                return None;
            }
            thread::sleep(Duration::from_millis(5));
        }
    }

    /// 止めるよう言われるか人が運転し始めたら false を返して、途中でやめる
    fn pause(&self, duration: Duration) -> bool {
        let end = Instant::now() + duration;
        while Instant::now() < end {
            if self.shared.stop_requested() || self.shared.manual_active(Instant::now()) {
                return false;
            }
            thread::sleep(Duration::from_millis(10).min(end - Instant::now()));
        }
        true
    }

    fn drive_loop(&mut self) -> End {
        let mut last_seq = 0;
        let mut last_frame_at = Instant::now();
        loop {
            if let Some(request) = self.shared.take_stop() {
                return request.into();
            }
            self.state.camera.note_request();
            let Some(frame) = self.wait_frame(last_seq) else {
                if let Some(request) = self.shared.take_stop() {
                    return request.into();
                }
                if last_frame_at.elapsed() > CAMERA_LOST {
                    return End::safety("camera_lost", "カメラのフレームが届きません");
                }
                self.set_phase("waiting", Some("camera".to_string()));
                continue;
            };
            last_seq = frame.seq;
            last_frame_at = Instant::now();
            self.frames += 1;
            if let Some(end) = self.cycle(&frame) {
                return end;
            }
        }
    }

    /// 1周分。走行を終えるなら理由を返す
    fn cycle(&mut self, frame: &Frame) -> Option<End> {
        let now = Instant::now();
        let dt = now.duration_since(self.last_cycle);
        self.last_cycle = now;
        let dt_s = dt
            .min(Duration::from_millis(self.params.ttl_ms))
            .as_secs_f64();
        {
            let mut status = self.shared.status();
            status.frames = self.frames;
            status.cycle_ms = ema(status.cycle_ms, dt.as_secs_f64() * 1000.0);
        }

        // 人の指令が来ている間は、自律の指令を出さない
        let human = self.shared.manual_active(now);
        if human != self.human {
            self.human = human;
            if human {
                self.takeovers += 1;
                let intent = self.last_intent.unwrap_or(self.shared.upper_intent());
                self.intervention(
                    "human_takeover",
                    "manual drive command",
                    json!({"takeover_id": self.takeovers, "policy_intent": intent.as_str()}),
                );
                self.shared.status().takeovers = self.takeovers;
            } else {
                // 自律に戻る時は、曲がりの途中の判断を捨てる
                self.avoidance = AvoidancePolicy::new(self.avoidance.config);
                self.last_intent = None;
                if self.help.take().is_some() {
                    // 人が抜け出させたので、回数の数えをやり直す
                    self.shared.status().help_reason = None;
                    self.stuck_times.clear();
                    self.escape_times.clear();
                    self.cliff_times.clear();
                    self.cliff_escaped = false;
                    self.obstacle_cycles = 0;
                    self.obstacle_logged = false;
                    // 人が走らせた分は記録した軌跡と合わないので、後退に使わない
                    self.trajectory = Trajectory::new(100);
                }
                self.note_event("autonomous again".to_string());
            }
            self.forward_since = None;
            self.stuck.reset();
        }
        if human {
            self.log.add_time(dt_s, true);
            self.shared.status().human_s = self.log.human_s;
            self.set_phase("human", None);
            return None;
        }
        // 人の助けを待っている間は動かない（時間は自律にも人にも数えない）
        if let Some((since, reason)) = &self.help {
            if now.duration_since(*since) >= HELP_TIMEOUT {
                return Some(End::quiet(format!(
                    "no help within {}s ({reason})",
                    HELP_TIMEOUT.as_secs()
                )));
            }
            self.set_phase("needs_help", Some(reason.clone()));
            thread::sleep(WAIT_SLEEP);
            return None;
        }
        self.log.add_time(dt_s, false);
        self.shared.status().autonomous_s = self.log.autonomous_s;

        let sensors = self.state.hardware.sensors();
        let distance = sensors
            .distance_cm
            .and_then(|r| r.fresh(now, DISTANCE_MAX_AGE));
        // 反射が戻らない時は遠い（避ける判断と減速には距離なしとして渡す）
        let seen = distance.filter(|&d| !is_far(d));
        self.shared.status().distance_cm = distance;

        let verdict = self
            .state
            .safety()
            .check_start(DriveDirection::Forward, &sensors, now);
        match verdict {
            Ok(()) => {
                self.obstacle_cycles = 0;
                self.obstacle_logged = false;
                self.cliff_escaped = false;
            }
            Err(SafetyReason::Obstacle) => {
                self.log_cycle(json!({"phase": "obstacle", "distance_cm": distance.map(round1)}));
                return self.on_obstacle(now, distance);
            }
            Err(SafetyReason::Cliff) => {
                let floor = sensors.grayscale.map(|g| g.value);
                self.log_cycle(json!({"phase": "cliff", "floor": floor}));
                return self.on_cliff(now, floor);
            }
            Err(SafetyReason::EmergencyStop) => {
                return Some(End::safety("manual_intervention", "emergency stop"));
            }
            Err(reason) => {
                self.forward_since = None;
                self.stuck.reset();
                self.set_phase("waiting", Some(format!("{reason:?}")));
                thread::sleep(WAIT_SLEEP);
                return None;
            }
        }

        // 前処理
        let started = Instant::now();
        let input = &self.model.manifest().input;
        let (w, h) = (input.width, input.height);
        let image = match decode_jpeg(&frame.jpeg) {
            Ok(image) => resize(&image, w, h),
            Err(error) => {
                self.set_phase("waiting", Some(error));
                return None;
            }
        };
        let tensor = to_tensor(&image, &self.model.manifest().normalization());

        // 引っかかりの確認（前進の指令が通り続けている間だけ）
        if self
            .forward_since
            .is_some_and(|since| now.duration_since(since) >= STUCK_SETTLE)
        {
            let stuck = self
                .stuck
                .update(self.secs(now), thumbnail(&image.data, w, h));
            self.shared.status().image_change = self.stuck.last_diff;
            if stuck {
                return self.on_stuck(now);
            }
        } else {
            self.stuck.reset();
        }

        // 推論と避ける判断
        let upper = self.shared.upper_intent();
        let first = self.last_intent.unwrap_or(upper);
        let output = match self.model.infer(&tensor, first) {
            Ok(output) => output,
            Err(error) => return Some(End::safety("model_error", error)),
        };
        let decision = self
            .avoidance
            .step(self.secs(now), seen, output.blocked, upper);
        let output = if decision.intent != first {
            match self.model.infer(&tensor, decision.intent) {
                Ok(output) => output,
                Err(error) => return Some(End::safety("model_error", error)),
            }
        } else {
            output
        };
        self.last_intent = Some(decision.intent);
        if let Some(event) = decision.event {
            self.note_event(format!(
                "avoid {} -> {} ({})",
                event.as_str(),
                decision.intent.as_str(),
                decision.reason
            ));
        }
        let inference_ms = started.elapsed().as_secs_f64() * 1000.0;

        // 速度と舵角
        let scale = self.model.manifest().steering_scale_deg;
        let model_deg =
            ((output.steering * scale).round() as i32).clamp(-MAX_STEER_DEG, MAX_STEER_DEG);
        let steer_deg = turn_floor(model_deg, &decision, AVOID_MIN_STEER_DEG);
        // 速度は実際に切る舵角で決める（下限で強く切った分も落とす）
        let mut speed = self.limiter.calculate(f64::from(steer_deg) / scale, seen);
        if decision.slow {
            speed = speed.min(self.params.min_speed);
        }

        if let Err(error) = self.state.command_steer(steer_deg) {
            self.set_phase("waiting", Some(format!("steer: {error}")));
        }
        let ttl = Duration::from_millis(self.params.ttl_ms);
        let applied = match self.state.command_move(DriveDirection::Forward, speed, ttl) {
            Ok(applied) => {
                self.forward_since.get_or_insert(now);
                self.trajectory.record(steer_deg, (dt_s * 1000.0) as u64);
                Some(applied)
            }
            Err(error) => {
                self.forward_since = None;
                self.set_phase("waiting", Some(format!("{error:?}")));
                None
            }
        };

        self.log_cycle(json!({
            "phase": if applied.is_some() { "driving" } else { "refused" },
            "distance_cm": distance.map(round1),
            "steering": round3(output.steering),
            "model_deg": model_deg,
            "steer_deg": steer_deg,
            "speed": applied,
            "slow": decision.slow,
            "intent": decision.intent.as_str(),
            "avoid": decision.state.as_str(),
            "blocked": output.blocked.map(round3),
            "image_change": self.stuck.last_diff.map(round3),
            "infer_ms": round1(inference_ms),
        }));

        let mut status = self.shared.status();
        if applied.is_some() {
            status.phase = "driving".to_string();
            status.detail = None;
        }
        status.intent = Some(decision.intent);
        status.avoid_state = Some(decision.state.as_str().to_string());
        status.steering = Some(output.steering);
        status.steer_deg = Some(steer_deg);
        status.speed = applied;
        status.blocked = output.blocked;
        status.inference_ms = ema(status.inference_ms, inference_ms);
        None
    }

    fn on_obstacle(&mut self, now: Instant, distance: Option<f64>) -> Option<End> {
        self.forward_since = None;
        self.stuck.reset();
        self.obstacle_cycles += 1;
        self.set_phase("obstacle", distance.map(|d| format!("{d:.0}cm")));
        if !self.obstacle_logged {
            self.obstacle_logged = true;
            // 曲がれずに止められた。動けるようになったら反対側へ曲がる
            let decision = self.avoidance.on_stopped_by_obstacle(self.secs(now));
            self.last_intent = Some(decision.intent);
            let note = match distance {
                Some(d) => format!("obstacle at {d:.0}cm"),
                None => "obstacle".to_string(),
            };
            self.intervention(
                "obstacle_stop",
                &note,
                json!({"distance_cm": distance, "intent_after": decision.intent.as_str()}),
            );
        }
        if self.obstacle_cycles >= OBSTACLE_ESCAPE_CYCLES {
            self.obstacle_cycles = 0;
            if let Some(end) = self.count_escape(now, "obstacle") {
                return Some(end);
            }
            self.escape(REVERSE_MS, None);
        }
        None
    }

    /// 段差（机の縁など）。軌跡に沿って1回だけ後退して向きを変える。後退しても段差のまま、
    /// 下がる軌跡がない、短い時間に2回目、のどれかなら、動き回らずに走行を終えて人を待つ
    fn on_cliff(&mut self, now: Instant, floor: Option<[u16; 3]>) -> Option<End> {
        self.forward_since = None;
        self.stuck.reset();
        self.set_phase("escaping", Some("cliff".to_string()));
        let calibration = self.state.safety().cliff();
        let cliff = match (calibration, floor) {
            (Some(calibration), Some(values)) => Some(calibration.is_cliff(values)),
            _ => None,
        };
        if self.cliff_escaped {
            return self.need_help(
                now,
                format!(
                    "cliff still detected after backing out (floor sensors {floor:?}); needs help"
                ),
            );
        }
        self.intervention(
            "cliff_detected",
            &format!("floor sensors {floor:?}"),
            json!({
                "floor": floor,
                "cliff_sensors": cliff,
                "thresholds": calibration.map(|c| c.thresholds),
                "baseline": calibration.map(|c| c.baseline),
            }),
        );
        let at = self.secs(now);
        if push_within(&mut self.cliff_times, at, CLIFF_WINDOW_S) >= CLIFF_LIMIT {
            return self.need_help(
                now,
                format!("cliff {CLIFF_LIMIT} times within {CLIFF_WINDOW_S:.0}s; needs help"),
            );
        }
        if let Some(end) = self.count_escape(now, "cliff") {
            return Some(end);
        }
        self.cliff_escaped = true;
        if !self.escape(REVERSE_MS, cliff) {
            return self.need_help(
                now,
                "cliff detected with no recorded path to back out along; needs help",
            );
        }
        None
    }

    /// 自分では抜け出せない。見守る人がいれば止まって助けを待ち、いなければ走行を終える
    fn need_help(&mut self, now: Instant, reason: impl Into<String>) -> Option<End> {
        let reason = reason.into();
        if !self.params.wait_for_help {
            return Some(End::quiet(reason));
        }
        let _ = self.state.stop_motors();
        self.forward_since = None;
        self.stuck.reset();
        self.help_requests += 1;
        self.note_event(format!("help requested: {reason}"));
        {
            let mut status = self.shared.status();
            status.help_reason = Some(reason.clone());
            status.help_requests = self.help_requests;
        }
        self.set_phase("needs_help", Some(reason.clone()));
        self.log_cycle(json!({"phase": "needs_help", "reason": reason}));
        self.help = Some((now, reason));
        None
    }

    /// 後退して抜ける回数を数え、多すぎれば走行を終える理由を返す
    fn count_escape(&mut self, now: Instant, kind: &str) -> Option<End> {
        let at = self.secs(now);
        if push_within(&mut self.escape_times, at, ESCAPE_WINDOW_S) > ESCAPE_LIMIT {
            return self.need_help(
                now,
                format!(
                    "more than {ESCAPE_LIMIT} escapes within {ESCAPE_WINDOW_S:.0}s (last: {kind}); \
                 needs help"
                ),
            );
        }
        None
    }

    fn on_stuck(&mut self, now: Instant) -> Option<End> {
        let at = self.secs(now);
        let diff = self.stuck.last_diff.unwrap_or(0.0);
        self.intervention(
            "motion_mismatch",
            &format!(
                "no_motion: the image did not change while driving forward (change {diff:.2})"
            ),
            json!({"flag": "no_motion"}),
        );
        if push_within(&mut self.stuck_times, at, STUCK_WINDOW_S) >= STUCK_LIMIT {
            return self.need_help(
                now,
                format!("stuck {STUCK_LIMIT} times within {STUCK_WINDOW_S:.0}s; needs help"),
            );
        }
        if let Some(end) = self.count_escape(now, "stuck") {
            return Some(end);
        }
        let decision = self.avoidance.on_stopped_by_obstacle(at);
        self.last_intent = Some(decision.intent);
        self.escape(STUCK_REVERSE_MS, None);
        None
    }

    /// 記録した軌跡の範囲で後退し、車首をそらす。1歩も下がれなければ false
    fn escape(&mut self, target_ms: u64, cliff: Option<[bool; 3]>) -> bool {
        let _ = self.state.stop_motors();
        self.forward_since = None;
        self.stuck.reset();
        let sequence = self.trajectory.reverse_sequence(target_ms);
        if sequence.is_empty() {
            self.set_phase(
                "waiting",
                Some("no recorded path behind the car".to_string()),
            );
            return false;
        }
        self.set_phase("escaping", None);
        let speed = self.params.speed;
        let mut reversed = 0;
        for step in sequence {
            if self.state.command_steer(step.steer_deg).is_err() {
                break;
            }
            let ttl = Duration::from_millis(step.duration_ms);
            match self
                .state
                .command_move(DriveDirection::Backward, speed, ttl)
            {
                Ok(_) => {}
                Err(MoveError::Refused(_)) | Err(MoveError::Hardware(_)) => break,
            }
            let completed = self.pause(ttl);
            reversed += step.duration_ms;
            if !completed {
                break;
            }
        }
        self.trajectory.consume(reversed);
        let _ = self.state.stop_motors();
        if reversed == 0 {
            return false;
        }
        let turn = self.trajectory.escape_turn(cliff);
        if !self.shared.stop_requested()
            && self.state.command_steer(turn.steer_deg).is_ok()
            && self
                .state
                .command_move(
                    DriveDirection::Forward,
                    speed,
                    Duration::from_millis(turn.duration_ms),
                )
                .is_ok()
        {
            self.pause(Duration::from_millis(turn.duration_ms));
            self.trajectory.record(turn.steer_deg, turn.duration_ms);
        }
        let _ = self.state.stop_motors();
        self.last_cycle = Instant::now();
        self.note_event(format!("escaped: reversed {reversed}ms"));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::avoidance::AVOID_MIN_STEER_DEG;
    use super::super::model::tests::v006_bundle;
    use super::super::model::{Manifest, Output, read_manifest};
    use super::super::{Autopilot, Intent, RunParams};
    use crate::api::handlers::{AppState, RobotHardware};
    use crate::api::models::DriveDirection;
    use crate::rpi_camera::CameraHub;
    use crate::safety::reflex::reflex_tick;
    use crate::safety::{CALIBRATION_SAMPLES, CliffCalibration, SafetyController};
    use crate::sensor_monitor::{SensorSnapshot, Timed};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    const JPEG: &[u8] = include_bytes!("../../tests/fixtures/pattern_320x240.jpg");
    const INVERTED: &[u8] = include_bytes!("../../tests/fixtures/pattern_inverted_320x240.jpg");
    const FLOOR: [u16; 3] = [500, 500, 500];

    #[derive(Default)]
    struct Log {
        commands: Vec<(DriveDirection, i32)>,
        steers: Vec<i32>,
        stops: usize,
    }

    struct Hardware {
        distance: Arc<Mutex<Option<f64>>>,
        floor: Arc<Mutex<[u16; 3]>>,
        log: Arc<Mutex<Log>>,
    }

    impl RobotHardware for Hardware {
        fn drive(&self, direction: DriveDirection, speed: i32) -> Result<(), String> {
            self.log.lock().unwrap().commands.push((direction, speed));
            Ok(())
        }
        fn steer(&self, angle: i32) -> Result<(), String> {
            self.log.lock().unwrap().steers.push(angle);
            Ok(())
        }
        fn stop(&self) -> Result<(), String> {
            self.log.lock().unwrap().stops += 1;
            Ok(())
        }
        fn sensors(&self) -> SensorSnapshot {
            let now = Instant::now();
            SensorSnapshot {
                distance_cm: self.distance.lock().unwrap().map(|d| Timed::new(d, now)),
                grayscale: Some(Timed::new(*self.floor.lock().unwrap(), now)),
                battery: Some(Timed::new((7.6, 50), now)),
            }
        }
        fn speak(&self, _: &str) -> Result<(), String> {
            Ok(())
        }
        fn stop_speech(&self) -> Result<(), String> {
            Ok(())
        }
        fn is_speaking(&self) -> bool {
            false
        }
        fn listen(&self) -> Result<String, String> {
            Ok(String::new())
        }
    }

    /// 決まった舵角を返すモデル
    struct FixedModel {
        manifest: Manifest,
        steering: f64,
        intents: Arc<Mutex<Vec<Intent>>>,
    }

    impl Steering for FixedModel {
        fn manifest(&self) -> &Manifest {
            &self.manifest
        }
        fn infer(&mut self, input: &[f32], intent: Intent) -> Result<Output, String> {
            assert_eq!(input.len(), 3 * 120 * 160);
            self.intents.lock().unwrap().push(intent);
            Ok(Output {
                steering: self.steering,
                blocked: None,
            })
        }
    }

    struct Rig {
        state: Arc<AppState>,
        distance: Arc<Mutex<Option<f64>>>,
        floor: Arc<Mutex<[u16; 3]>>,
        log: Arc<Mutex<Log>>,
        intents: Arc<Mutex<Vec<Intent>>>,
        running: Arc<AtomicBool>,
        tmp: tempfile::TempDir,
    }

    impl Rig {
        /// カメラは2枚の画像を交互に出す（前進すると景色が変わる）
        fn new(distance: f64) -> Self {
            Self::with_camera(distance, true)
        }

        /// `changing` が false なら同じ画像を出し続ける（引っかかった時の景色）
        fn with_camera(distance: f64, changing: bool) -> Self {
            let tmp = tempfile::tempdir().unwrap();
            let distance = Arc::new(Mutex::new(Some(distance)));
            let floor = Arc::new(Mutex::new(FLOOR));
            let log = Arc::new(Mutex::new(Log::default()));
            let mut safety = SafetyController::new(true);
            safety
                .set_cliff(CliffCalibration::from_samples(&[FLOOR; CALIBRATION_SAMPLES]).unwrap());
            let state = Arc::new(AppState {
                hardware: Box::new(Hardware {
                    distance: Arc::clone(&distance),
                    floor: Arc::clone(&floor),
                    log: Arc::clone(&log),
                }),
                safety: Mutex::new(safety),
                api_token: "t".to_string(),
                camera: Arc::new(CameraHub::new()),
                autopilot: Autopilot::new(tmp.path().join("models"), tmp.path().join("runs")),
            });
            let running = Arc::new(AtomicBool::new(true));
            // 反射ループとカメラの代わり
            {
                let state = Arc::clone(&state);
                let running = Arc::clone(&running);
                thread::spawn(move || {
                    let mut n = 0u64;
                    while running.load(Ordering::SeqCst) {
                        reflex_tick(&state);
                        let jpeg = if changing && n % 2 == 1 {
                            INVERTED
                        } else {
                            JPEG
                        };
                        state.camera.publish(jpeg.to_vec());
                        n += 1;
                        thread::sleep(Duration::from_millis(25));
                    }
                });
            }
            Self {
                state,
                distance,
                floor,
                log,
                intents: Arc::new(Mutex::new(Vec::new())),
                running,
                tmp,
            }
        }

        fn start(&self, steering: f64) {
            self.start_with(steering, RunParams::default());
        }

        fn start_with(&self, steering: f64, params: RunParams) {
            let manifest = read_manifest(&v006_bundle(self.tmp.path())).unwrap();
            let model = FixedModel {
                manifest,
                steering,
                intents: Arc::clone(&self.intents),
            };
            self.state
                .autopilot
                .start_with_model(&self.state, Box::new(model), params)
                .unwrap();
        }

        fn wait_until(&self, what: &str, check: impl Fn(&Self) -> bool) {
            self.wait_for(Duration::from_secs(5), what, check);
        }

        fn wait_for(&self, timeout: Duration, what: &str, check: impl Fn(&Self) -> bool) {
            let deadline = Instant::now() + timeout;
            while !check(self) {
                assert!(Instant::now() < deadline, "timed out waiting for {what}");
                thread::sleep(Duration::from_millis(10));
            }
        }

        fn forward_commands(&self) -> usize {
            self.log
                .lock()
                .unwrap()
                .commands
                .iter()
                .filter(|(d, _)| *d == DriveDirection::Forward)
                .count()
        }

        fn backward_commands(&self) -> usize {
            self.log
                .lock()
                .unwrap()
                .commands
                .iter()
                .filter(|(d, _)| *d == DriveDirection::Backward)
                .count()
        }

        fn run_dir(&self) -> std::path::PathBuf {
            std::fs::read_dir(self.tmp.path().join("runs"))
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path()
        }

        fn interventions(&self) -> Vec<serde_json::Value> {
            std::fs::read_to_string(self.run_dir().join("interventions.jsonl"))
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }

        fn wait_stopped(&self, timeout: Duration) -> String {
            self.wait_for(timeout, "the run to end", |r| {
                !r.state.autopilot.is_running()
            });
            self.state.autopilot.status().stopped_reason.unwrap()
        }

        fn stop_and_read_run(&self) -> serde_json::Value {
            self.state
                .autopilot
                .request_stop(Some("manual_intervention"), "test done");
            self.state.autopilot.join();
            let run_dir = std::fs::read_dir(self.tmp.path().join("runs"))
                .unwrap()
                .next()
                .unwrap()
                .unwrap()
                .path();
            serde_json::from_str(&std::fs::read_to_string(run_dir.join("run.json")).unwrap())
                .unwrap()
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            self.state.autopilot.request_stop(None, "test cleanup");
            self.state.autopilot.join();
            self.running.store(false, Ordering::SeqCst);
        }
    }

    use super::super::model::Steering;

    #[test]
    fn drives_forward_with_the_model_steering_and_writes_a_run_log() {
        let rig = Rig::new(150.0);
        rig.start(0.5);
        rig.wait_until("forward commands", |r| r.forward_commands() >= 3);
        let status = rig.state.autopilot.status();
        assert!(status.running);
        assert_eq!(status.phase, "driving");
        assert_eq!(status.steer_deg, Some(18)); // 0.5 x 35 を丸めた
        assert_eq!(status.speed, Some(45)); // 舵角の分だけ落とす
        assert_eq!(rig.log.lock().unwrap().steers.last(), Some(&18));

        let run = rig.stop_and_read_run();
        assert_eq!(run["model_version"], "v006");
        assert_eq!(run["ended"], true);
        assert_eq!(run["interventions"]["human"], 1); // 止めたのは人
        assert!(run["autonomous_s"].as_f64().unwrap() > 0.0);
        let status = rig.state.autopilot.status();
        assert!(!status.running);
        assert_eq!(status.phase, "stopped");
        assert!(rig.log.lock().unwrap().stops > 0);
    }

    #[test]
    fn a_near_obstacle_turns_to_the_right_first() {
        let rig = Rig::new(45.0);
        rig.start(0.0);
        rig.wait_until("a right intent", |r| {
            r.intents.lock().unwrap().contains(&Intent::Right)
        });
        // 推論の後に状態を書くので、意図が見えてから少し待つことがある
        rig.wait_until("the avoid state", |r| {
            r.state.autopilot.status().avoid_state.as_deref() == Some("avoid")
        });
        // モデルが 0°でも、避けている間は右へ下限まで切る
        rig.wait_until("the steering floor", |r| {
            r.state.autopilot.status().steer_deg == Some(AVOID_MIN_STEER_DEG)
        });
    }

    #[test]
    fn manual_commands_pause_the_autopilot_and_it_resumes_after_quiet_time() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        rig.wait_until("driving", |r| r.forward_commands() >= 2);
        rig.state.autopilot.note_manual(Instant::now());
        rig.wait_until("human phase", |r| {
            r.state.autopilot.status().phase == "human"
        });
        let before = rig.forward_commands();
        thread::sleep(Duration::from_millis(500));
        assert_eq!(
            rig.forward_commands(),
            before,
            "no autonomous commands while a human drives"
        );
        rig.wait_until("autonomous again", |r| r.forward_commands() > before);
        let status = rig.state.autopilot.status();
        assert_eq!(status.takeovers, 1);
        assert_eq!(status.interventions.human, 1);
        assert!(status.human_s > 0.0);
    }

    #[test]
    fn an_obstacle_stop_is_logged_once_and_the_car_backs_out_along_its_path() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        rig.wait_until("some recorded path", |r| r.forward_commands() >= 10);
        *rig.distance.lock().unwrap() = Some(10.0);
        rig.wait_until("reverse", |r| {
            r.log
                .lock()
                .unwrap()
                .commands
                .iter()
                .any(|(d, _)| *d == DriveDirection::Backward)
        });
        *rig.distance.lock().unwrap() = Some(150.0);
        let status = rig.state.autopilot.status();
        assert_eq!(status.interventions.safety, 1, "{:?}", status.last_event);
        let run = rig.stop_and_read_run();
        assert_eq!(run["interventions"]["safety"], 1);
    }

    #[test]
    fn the_same_image_while_driving_forward_means_stuck() {
        let rig = Rig::with_camera(150.0, false);
        rig.start(0.0);
        rig.wait_until("a stuck event", |r| {
            r.state.autopilot.status().interventions.safety >= 1
        });
        let status = rig.state.autopilot.status();
        assert!(
            status.last_event.unwrap_or_default().contains("no_motion")
                || status.phase == "escaping"
        );
        // 30 秒のうちに 3 回引っかかったら、人の助けが要るとして終える
        // 1回あたり、見つけるまで約 1.3 秒と後退で約 1.4 秒
        rig.wait_for(Duration::from_secs(15), "stop after repeated stucks", |r| {
            !r.state.autopilot.is_running()
        });
        let status = rig.state.autopilot.status();
        assert!(status.stopped_reason.unwrap().contains("needs help"));
        // 終えた理由と、1周ごとの判断がファイルに残る
        let dir = rig.run_dir();
        let run: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("run.json")).unwrap()).unwrap();
        assert!(run["end_reason"].as_str().unwrap().contains("needs help"));
        let cycles = std::fs::read_to_string(dir.join("cycles.jsonl")).unwrap();
        let driving: Vec<serde_json::Value> = cycles
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .filter(|c: &serde_json::Value| c["phase"] == "driving")
            .collect();
        assert!(!driving.is_empty());
        assert_eq!(driving[0]["distance_cm"], 150.0);
        assert_eq!(driving[0]["intent"], "any");
    }

    /// 中央の床センサーだけが段差（机の縁を正面に見た時）
    const EDGE: [u16; 3] = [500, 100, 500];

    #[test]
    fn push_within_drops_old_times() {
        let mut times = std::collections::VecDeque::new();
        assert_eq!(super::push_within(&mut times, 0.0, 30.0), 1);
        assert_eq!(super::push_within(&mut times, 10.0, 30.0), 2);
        assert_eq!(super::push_within(&mut times, 35.0, 30.0), 2); // 0.0 は捨てる
    }

    #[test]
    fn a_cliff_backs_out_once_and_ends_the_run_if_the_edge_is_still_there() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        rig.wait_until("some recorded path", |r| r.forward_commands() >= 10);
        *rig.floor.lock().unwrap() = EDGE;
        let reason = rig.wait_stopped(Duration::from_secs(8));
        assert!(reason.contains("still detected"), "{reason}");
        assert!(
            rig.backward_commands() > 0,
            "backed out along the path once"
        );
        let status = rig.state.autopilot.status();
        assert_eq!(status.interventions.safety, 1);
        // 電源を切られても調べられるよう、段差の値と閾値を残す
        let records = rig.interventions();
        let cliff = records
            .iter()
            .find(|r| r["trigger"] == "cliff_detected")
            .unwrap();
        assert_eq!(cliff["floor"], serde_json::json!(EDGE));
        assert_eq!(
            cliff["cliff_sensors"],
            serde_json::json!([false, true, false])
        );
        assert_eq!(cliff["thresholds"], serde_json::json!([175, 175, 175]));
    }

    #[test]
    fn a_cliff_with_no_path_behind_ends_without_reversing() {
        let rig = Rig::new(150.0);
        *rig.floor.lock().unwrap() = EDGE;
        rig.start(0.0);
        let reason = rig.wait_stopped(Duration::from_secs(5));
        assert!(reason.contains("no recorded path"), "{reason}");
        assert_eq!(rig.backward_commands(), 0);
    }

    #[test]
    fn a_second_cliff_soon_after_ends_the_run() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        rig.wait_until("some recorded path", |r| r.forward_commands() >= 10);
        *rig.floor.lock().unwrap() = EDGE;
        rig.wait_until("reverse", |r| r.backward_commands() > 0);
        *rig.floor.lock().unwrap() = FLOOR;
        let before = rig.forward_commands();
        rig.wait_until("driving again", |r| r.forward_commands() >= before + 5);
        assert!(rig.state.autopilot.is_running());
        *rig.floor.lock().unwrap() = EDGE;
        let reason = rig.wait_stopped(Duration::from_secs(5));
        assert!(reason.contains("cliff 2 times"), "{reason}");
        assert_eq!(rig.state.autopilot.status().interventions.safety, 2);
    }

    #[test]
    fn an_obstacle_that_never_clears_ends_the_run_after_repeated_escapes() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        rig.wait_until("some recorded path", |r| r.forward_commands() >= 10);
        *rig.distance.lock().unwrap() = Some(10.0);
        let reason = rig.wait_stopped(Duration::from_secs(15));
        assert!(reason.contains("escapes within"), "{reason}");
        // 障害物の停止は止められている間に1回だけ数える
        assert_eq!(rig.state.autopilot.status().interventions.safety, 1);
    }

    #[test]
    fn with_someone_watching_it_waits_for_help_and_resumes_after_the_human_drives() {
        let rig = Rig::with_camera(150.0, false);
        let params = RunParams {
            wait_for_help: true,
            ..RunParams::default()
        };
        rig.start_with(0.0, params);
        rig.wait_for(Duration::from_secs(15), "a help request", |r| {
            r.state.autopilot.status().phase == "needs_help"
        });
        let status = rig.state.autopilot.status();
        assert!(status.running, "waiting, not ended");
        assert!(status.help_reason.unwrap().contains("needs help"));
        assert_eq!(status.help_requests, 1);
        // 待っている間は動かない
        let before = rig.log.lock().unwrap().commands.len();
        thread::sleep(Duration::from_millis(400));
        assert_eq!(rig.log.lock().unwrap().commands.len(), before);
        // 人がパッドで動かし、手を離すと自律に戻る
        rig.state.autopilot.note_manual(Instant::now());
        rig.wait_until("human phase", |r| {
            r.state.autopilot.status().phase == "human"
        });
        rig.wait_until("autonomous again", |r| {
            r.forward_commands() > 0 && r.state.autopilot.status().phase == "driving"
        });
        let status = rig.state.autopilot.status();
        assert!(status.help_reason.is_none());
        assert_eq!(status.takeovers, 1);
        assert_eq!(status.interventions.human, 1);
    }

    #[test]
    fn camera_loss_ends_the_run() {
        let rig = Rig::new(150.0);
        rig.running.store(false, Ordering::SeqCst); // カメラも反射ループも止める
        rig.start(0.0);
        let deadline = Instant::now() + Duration::from_secs(6);
        while rig.state.autopilot.is_running() {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(50));
        }
        let status = rig.state.autopilot.status();
        assert_eq!(status.interventions.safety, 1);
        assert!(status.stopped_reason.unwrap().contains("カメラ"));
    }

    #[test]
    fn a_second_start_is_refused_while_running() {
        let rig = Rig::new(150.0);
        rig.start(0.0);
        let manifest = read_manifest(&v006_bundle(rig.tmp.path())).unwrap();
        let model = FixedModel {
            manifest,
            steering: 0.0,
            intents: Arc::new(Mutex::new(Vec::new())),
        };
        let error = rig
            .state
            .autopilot
            .start_with_model(&rig.state, Box::new(model), RunParams::default())
            .unwrap_err();
        assert!(error.contains("すでに"));
    }
}
