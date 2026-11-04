use crate::safety::{CliffCalibration, SafetyEvent, SafetyReason};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriveDirection {
    Forward,
    Backward,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveCommand {
    pub direction: DriveDirection,
    #[serde(default = "default_speed")]
    pub speed: i32,
    /// この時間内に次のコマンドが届かなければ Pi が停止する
    #[serde(default = "default_ttl")]
    pub ttl_ms: u64,
}

fn default_speed() -> i32 {
    30
}

fn default_ttl() -> u64 {
    300
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteerCommand {
    pub angle: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasonicResponse {
    pub distance_cm: f32,
    pub age_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayscaleResponse {
    pub left: i32,
    pub middle: i32,
    pub right: i32,
    pub age_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakCommand {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListenResponse {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSnapshotResponse {
    pub image_base64: String,
    pub width: u32,
    pub height: u32,
    pub seq: u64,
    pub age_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// 走行を拒否した理由
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<SafetyReason>,
}

impl ApiResponse {
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            message: None,
            reason: None,
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            message: Some(msg.into()),
            reason: None,
        }
    }

    pub fn refused(reason: SafetyReason) -> Self {
        Self {
            status: "refused".to_string(),
            message: Some(reason.message().to_string()),
            reason: Some(reason),
        }
    }
}

/// センサー値が古い、または読めていない時は値が `null` になる
#[derive(Debug, Clone, Serialize)]
pub struct DistanceStatus {
    pub cm: Option<f64>,
    pub age_ms: Option<u64>,
    /// 反射が戻らず、測定範囲より遠いと判断している。音を吸う物の可能性もある
    pub far: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct FloorStatus {
    pub values: Option<[u16; 3]>,
    pub age_ms: Option<u64>,
    /// 校正値と比べて段差と判定したセンサー（左、中央、右）
    pub cliff: Option<[bool; 3]>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BatteryStatus {
    pub voltage: Option<f64>,
    pub percent: Option<i32>,
    pub age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SensorsStatus {
    pub distance: DistanceStatus,
    pub floor: FloorStatus,
    pub battery: BatteryStatus,
}

#[derive(Debug, Clone, Serialize)]
pub struct MotionStatus {
    pub direction: DriveDirection,
    pub speed: i32,
    pub ttl_remaining_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct CliffStatus {
    pub guard_enabled: bool,
    pub calibration: Option<CliffCalibration>,
}

/// 各方向へ今すぐ走れない理由。`null` なら走行できる。
#[derive(Debug, Clone, Serialize)]
pub struct BlockedStatus {
    pub forward: Option<SafetyReason>,
    pub backward: Option<SafetyReason>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReflexStatus {
    pub hz: f64,
    pub max_interval_ms: f64,
    pub last_tick_age_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CameraStatus {
    pub running: bool,
    pub fps: f64,
    pub frames: u64,
    pub frame_age_ms: Option<u64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StatusResponse {
    /// `idle` / `forward` / `backward` / `emergency_stop`
    pub mode: &'static str,
    pub emergency_stop: bool,
    pub motion: Option<MotionStatus>,
    pub steer_angle: i32,
    pub blocked: BlockedStatus,
    pub sensors: SensorsStatus,
    pub cliff: CliffStatus,
    pub reflex: ReflexStatus,
    pub camera: CameraStatus,
    pub latest_event_id: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventsResponse {
    pub events: Vec<SafetyEvent>,
    pub latest_event_id: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventsQuery {
    #[serde(default)]
    pub after: u64,
}
