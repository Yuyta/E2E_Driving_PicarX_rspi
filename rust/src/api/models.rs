use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    #[serde(default = "default_duration")]
    pub duration_ms: u64,
}

fn default_speed() -> i32 {
    30
}

fn default_duration() -> u64 {
    500
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteerCommand {
    pub angle: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltrasonicResponse {
    pub distance_cm: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayscaleResponse {
    pub left: i32,
    pub middle: i32,
    pub right: i32,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ApiResponse {
    pub fn ok() -> Self {
        Self {
            status: "ok".to_string(),
            message: None,
        }
    }

    pub fn err(msg: impl Into<String>) -> Self {
        Self {
            status: "error".to_string(),
            message: Some(msg.into()),
        }
    }
}
