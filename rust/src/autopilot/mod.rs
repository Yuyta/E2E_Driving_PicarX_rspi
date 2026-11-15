//! Pi 上の自律走行（ROADMAP の M3「推論を Pi 5 へ移行」、2026-11-14 に走行のループごと Pi に
//! 置くと決めた）。
//!
//! カメラの最新フレームを前処理し、走行モデル（`tract`）で舵角を推論して、障害物を避ける判断で
//! 意図を決め、速度を決めて走らせる。指令は HTTP の走行指令と同じ安全の判定（反射ループ、TTL）を
//! 通る。PC がなくても走れ、Wi-Fi の遅れの影響も受けない。
//!
//! - 走行中に HTTP で走行や舵の指令が来たら、人が運転を引き継いだとみなして指令を出すのをやめ、
//!   指令が 1.5 秒途絶えたら自律に戻る。停止の指令と緊急停止では走行を終える
//! - 障害物や段差で止められたら、記録した軌跡の範囲で後退して向きを変える
//! - 画像が変わらないまま前進の指令を続けていたら、引っかかったとみなして同じく後退する
//! - 走行ごとに、PC と同じ形式の記録（run.json、interventions.jsonl）を残す

pub mod avoidance;
pub mod model;
pub mod preprocess;
pub mod runlog;
mod runner;
pub mod speed;
pub mod store;
pub mod stuck;
pub mod trajectory;

use crate::api::handlers::AppState;
use model::{Steering, TractModel};
use runlog::{Counts, RunLog, unix_now};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use store::ModelStore;

/// 行きたい向き。順番は走行モデルの意図の入力（one-hot）と同じ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Intent {
    #[default]
    Any,
    Left,
    Right,
}

impl Intent {
    pub const ALL: [Intent; 3] = [Intent::Any, Intent::Left, Intent::Right];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|i| i.as_str() == text)
    }

    pub fn one_hot(self) -> [f32; 3] {
        let mut v = [0.0; 3];
        v[self as usize] = 1.0;
        v
    }

    pub fn is_side(self) -> bool {
        self != Self::Any
    }

    /// 左右の反対。おまかせはおまかせのまま
    pub fn other_side(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Any => Self::Any,
        }
    }
}

/// モデルのフォルダ（版の管理は `store`）を指定する環境変数
pub const MODELS_DIR_ENV: &str = "PICARX_MODELS_DIR";
/// 走行の記録のフォルダを指定する環境変数
pub const RUNS_DIR_ENV: &str = "PICARX_RUNS_DIR";

fn home_dir(sub: &str) -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("picarx")
        .join(sub)
}

pub fn models_dir_from_env() -> PathBuf {
    std::env::var(MODELS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir("models"))
}

pub fn runs_dir_from_env() -> PathBuf {
    std::env::var(RUNS_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|_| home_dir("runs"))
}

/// 走行の設定（`POST /api/autopilot/start` の本文）
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct RunParams {
    pub speed: i32,
    pub min_speed: i32,
    pub intent: Intent,
    /// 1回の指令の期限。指令が途絶えたら、この時間で止まる
    pub ttl_ms: u64,
    /// 自分で抜け出せない時（引っかかりや段差が続く、後退を繰り返す）に、走行を終えずに
    /// 止まって人の助けを待つ。人がパッドで動かして手を離すと自律に戻る。
    /// 見守る人がいる時（PC の `pi_autopilot --gamepad`）だけ true にする
    pub wait_for_help: bool,
}

impl Default for RunParams {
    fn default() -> Self {
        Self {
            speed: speed::DRIVE_SPEED,
            min_speed: speed::MIN_MOVING_SPEED,
            intent: Intent::Any,
            ttl_ms: 400,
            wait_for_help: false,
        }
    }
}

impl RunParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(speed::MIN_MOVING_SPEED..=crate::api::handlers::MAX_SPEED).contains(&self.speed) {
            return Err(format!(
                "speed は {} から {} の間にしてください",
                speed::MIN_MOVING_SPEED,
                crate::api::handlers::MAX_SPEED
            ));
        }
        if self.min_speed < speed::MIN_MOVING_SPEED || self.min_speed > self.speed {
            return Err(format!(
                "min_speed は {} 以上、speed 以下にしてください（それより遅いと車輪が回りません）",
                speed::MIN_MOVING_SPEED
            ));
        }
        if !(100..=1000).contains(&self.ttl_ms) {
            return Err("ttl_ms は 100 から 1000 の間にしてください".to_string());
        }
        Ok(())
    }
}

/// `GET /api/autopilot/status` で返す状態
#[derive(Debug, Clone, Serialize, Default)]
pub struct AutopilotStatus {
    pub running: bool,
    /// starting, driving, human, obstacle, escaping, waiting, needs_help, stopped
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub run_id: Option<String>,
    pub model_version: Option<String>,
    pub blocked_output: bool,
    pub upper_intent: Intent,
    pub intent: Option<Intent>,
    pub avoid_state: Option<String>,
    pub steering: Option<f64>,
    pub steer_deg: Option<i32>,
    pub speed: Option<i32>,
    pub blocked: Option<f64>,
    pub distance_cm: Option<f64>,
    pub image_change: Option<f64>,
    pub cycle_ms: Option<f64>,
    pub inference_ms: Option<f64>,
    pub frames: u64,
    pub autonomous_s: f64,
    pub human_s: f64,
    pub interventions: Counts,
    pub takeovers: u32,
    pub last_event: Option<String>,
    /// 人の助けを待っている理由（待っていなければ None）
    pub help_reason: Option<String>,
    /// これまでに人の助けを求めた回数
    pub help_requests: u32,
    pub stopped_reason: Option<String>,
}

/// 走行を終える理由。`trigger` があれば介入として記録する
#[derive(Debug, Clone)]
pub struct StopRequest {
    pub trigger: Option<String>,
    pub note: String,
}

/// 走行の外（HTTP のハンドラ）と、走行のスレッドで共有する状態
#[derive(Default)]
pub struct Shared {
    stop: Mutex<Option<StopRequest>>,
    manual_at: Mutex<Option<Instant>>,
    upper_intent: Mutex<Intent>,
    status: Mutex<AutopilotStatus>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// 人の指令がこの時間途絶えたら自律に戻る
pub const TAKEOVER_RELEASE: Duration = Duration::from_millis(1500);

impl Shared {
    fn stop_requested(&self) -> bool {
        lock(&self.stop).is_some()
    }

    fn take_stop(&self) -> Option<StopRequest> {
        lock(&self.stop).take()
    }

    fn manual_active(&self, now: Instant) -> bool {
        lock(&self.manual_at).is_some_and(|t| now.saturating_duration_since(t) < TAKEOVER_RELEASE)
    }

    fn upper_intent(&self) -> Intent {
        *lock(&self.upper_intent)
    }

    fn status(&self) -> MutexGuard<'_, AutopilotStatus> {
        lock(&self.status)
    }
}

/// 走行の開始時に返す情報
#[derive(Debug, Clone, Serialize)]
pub struct StartInfo {
    pub run_id: String,
    /// 動いている走行サーバーのビルド（どのバイナリが動いているかを PC から確かめる）
    pub server_build: String,
    pub model_version: String,
    pub blocked_output: bool,
    pub run_dir: String,
}

/// 使用中のモデルの確認結果（`GET /api/autopilot/model`）
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub models_dir: String,
    pub current: Option<String>,
    pub versions: Vec<store::VersionInfo>,
    pub verified: bool,
    pub error: Option<String>,
    pub manifest: Option<model::Manifest>,
}

pub struct Autopilot {
    pub models_dir: PathBuf,
    pub runs_dir: PathBuf,
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl Autopilot {
    pub fn new(models_dir: PathBuf, runs_dir: PathBuf) -> Self {
        Self {
            models_dir,
            runs_dir,
            shared: Arc::new(Shared::default()),
            thread: Mutex::new(None),
        }
    }

    pub fn from_env() -> Self {
        Self::new(models_dir_from_env(), runs_dir_from_env())
    }

    pub fn is_running(&self) -> bool {
        lock(&self.thread)
            .as_ref()
            .is_some_and(|h| !h.is_finished())
    }

    pub fn status(&self) -> AutopilotStatus {
        let mut status = self.shared.status().clone();
        status.running = self.is_running();
        status.upper_intent = self.shared.upper_intent();
        status
    }

    pub fn model_info(&self) -> ModelInfo {
        let store = ModelStore::new(&self.models_dir);
        let current = store.current();
        let (verified, error, manifest) = match &current {
            Some((_, dir)) => match model::verify(dir) {
                Ok(m) => (true, None, Some(m)),
                Err(e) => (false, Some(e), None),
            },
            None => (false, Some("使用中のモデルがありません".to_string()), None),
        };
        ModelInfo {
            models_dir: self.models_dir.display().to_string(),
            current: current.map(|(v, _)| v),
            versions: store.list(),
            verified,
            error,
            manifest,
        }
    }

    /// 使用中のモデルを確かめて読み込み、走行を始める（読み込みに時間がかかるので同期で呼ぶ）
    pub fn start(&self, state: &Arc<AppState>, params: RunParams) -> Result<StartInfo, String> {
        if self.is_running() {
            return Err("自律走行はすでに動いています".to_string());
        }
        params.validate()?;
        let (_, dir) = ModelStore::new(&self.models_dir).current().ok_or_else(|| {
            format!(
                "使用中のモデルがありません（{}）。PC から送って `picar-x-rust model install` してください",
                self.models_dir.display()
            )
        })?;
        let model = TractModel::load(&dir)?;
        self.start_with_model(state, Box::new(model), params)
    }

    /// 読み込んだモデルで走行を始める（テストではモデルを差し替える）
    pub fn start_with_model(
        &self,
        state: &Arc<AppState>,
        model: Box<dyn Steering>,
        params: RunParams,
    ) -> Result<StartInfo, String> {
        let mut thread = lock(&self.thread);
        if thread.as_ref().is_some_and(|h| !h.is_finished()) {
            return Err("自律走行はすでに動いています".to_string());
        }
        params.validate()?;
        let manifest = model.manifest().clone();
        let settings = serde_json::json!({
            "speed": params.speed,
            "min_speed": params.min_speed,
            "intent": params.intent.as_str(),
            "ttl_ms": params.ttl_ms,
            "wait_for_help": params.wait_for_help,
            "avoidance": true,
            "blocked_output": manifest.blocked_trained,
            "motion_check": "image_change",
        });
        let log = RunLog::create(&self.runs_dir, &manifest, settings, unix_now())?;
        let info = StartInfo {
            run_id: log.run_id.clone(),
            server_build: crate::server_build(),
            model_version: manifest.model_version.clone(),
            blocked_output: manifest.blocked_trained,
            run_dir: log.dir().display().to_string(),
        };
        *lock(&self.shared.stop) = None;
        *lock(&self.shared.manual_at) = None;
        *lock(&self.shared.upper_intent) = params.intent;
        *self.shared.status() = AutopilotStatus {
            running: true,
            phase: "starting".to_string(),
            run_id: Some(info.run_id.clone()),
            model_version: Some(info.model_version.clone()),
            blocked_output: info.blocked_output,
            upper_intent: params.intent,
            ..Default::default()
        };
        let runner = runner::Runner::new(
            Arc::clone(state),
            Arc::clone(&self.shared),
            model,
            log,
            params,
        )?;
        let handle = std::thread::Builder::new()
            .name("autopilot".to_string())
            .spawn(move || runner.run())
            .map_err(|e| format!("走行のスレッドを起動できません: {e}"))?;
        *thread = Some(handle);
        Ok(info)
    }

    /// 走行を終えるよう伝える。動いていなければ false
    pub fn request_stop(&self, trigger: Option<&str>, note: &str) -> bool {
        if !self.is_running() {
            return false;
        }
        *lock(&self.shared.stop) = Some(StopRequest {
            trigger: trigger.map(str::to_string),
            note: note.to_string(),
        });
        true
    }

    /// 走行のスレッドが終わるのを待つ
    pub fn join(&self) {
        let handle = lock(&self.thread).take();
        if let Some(handle) = handle {
            let _ = handle.join();
        }
    }

    /// HTTP で走行か舵の指令が来た（人が運転している）
    pub fn note_manual(&self, now: Instant) {
        if self.is_running() {
            *lock(&self.shared.manual_at) = Some(now);
        }
    }

    pub fn set_intent(&self, intent: Intent) {
        *lock(&self.shared.upper_intent) = intent;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_one_hot_matches_the_model_input_order() {
        assert_eq!(Intent::Any.one_hot(), [1.0, 0.0, 0.0]);
        assert_eq!(Intent::Right.one_hot(), [0.0, 0.0, 1.0]);
        assert_eq!(Intent::parse("left"), Some(Intent::Left));
        assert_eq!(Intent::parse("up"), None);
        assert_eq!(Intent::Left.other_side(), Intent::Right);
    }

    #[test]
    fn params_reject_speeds_that_do_not_move_the_wheels() {
        assert!(RunParams::default().validate().is_ok());
        let slow = RunParams {
            min_speed: 30,
            ..Default::default()
        };
        assert!(slow.validate().is_err());
        let fast = RunParams {
            speed: 200,
            ..Default::default()
        };
        assert!(fast.validate().is_err());
    }
}
