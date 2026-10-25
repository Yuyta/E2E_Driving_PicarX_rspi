use crate::api::models::{
    ApiResponse, CameraSnapshotResponse, DriveDirection, GrayscaleResponse, ListenResponse,
    MoveCommand, SpeakCommand, SteerCommand, UltrasonicResponse,
};
use crate::safety::SafetyGuard;
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use std::sync::Arc;

/// ハードウェア抽象化インターフェース
pub trait RobotHardware: Send + Sync {
    fn drive(&self, direction: DriveDirection, speed: i32, duration_ms: u64) -> Result<(), String>;
    fn steer(&self, angle: i32) -> Result<(), String>;
    fn stop(&self) -> Result<(), String>;
    fn get_distance(&self) -> Result<f32, String>;
    fn get_grayscale(&self) -> Result<(i32, i32, i32), String>;
    fn speak(&self, text: &str) -> Result<(), String>;
    fn listen(&self) -> Result<String, String>;
    fn take_snapshot(&self, width: u32, height: u32) -> Result<String, String>;
}

/// 共有アプリケーション状態
pub struct AppState {
    pub hardware: Box<dyn RobotHardware>,
    pub safety: Arc<SafetyGuard>,
}

pub async fn move_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<MoveCommand>,
) -> impl IntoResponse {
    state.safety.record_command().await;

    // 前進時の安全チェック
    if cmd.direction == DriveDirection::Forward {
        let dist = state.hardware.get_distance().unwrap_or(0.0);
        if !state.safety.can_move_forward(dist) {
            return (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse::err(format!(
                    "安全ガード作動: 障害物が近すぎます (距離: {:.1}cm)",
                    dist
                ))),
            );
        }
    }

    match state
        .hardware
        .drive(cmd.direction, cmd.speed, cmd.duration_ms)
    {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn steer_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<SteerCommand>,
) -> impl IntoResponse {
    state.safety.record_command().await;

    if !(-40..=40).contains(&cmd.angle) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::err(
                "ステアリング角度は-40から40の間である必要があります",
            )),
        );
    }

    match state.hardware.steer(cmd.angle) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn stop_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.safety.trigger_emergency_stop();
    match state.hardware.stop() {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn ultrasonic_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.hardware.get_distance() {
        Ok(dist) => (
            StatusCode::OK,
            Json(serde_json::to_value(UltrasonicResponse { distance_cm: dist }).unwrap()),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(ApiResponse::err(e)).unwrap()),
        ),
    }
}

pub async fn grayscale_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.hardware.get_grayscale() {
        Ok((l, m, r)) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(GrayscaleResponse {
                    left: l,
                    middle: m,
                    right: r,
                })
                .unwrap(),
            ),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(ApiResponse::err(e)).unwrap()),
        ),
    }
}

pub async fn speak_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<SpeakCommand>,
) -> impl IntoResponse {
    // 実機側ローカルで音声合成(TTS)＆再生
    match state.hardware.speak(&cmd.text) {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn listen_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // 実機側ローカルで録音＆文字起こし(STT)
    match state.hardware.listen() {
        Ok(text) => (
            StatusCode::OK,
            Json(serde_json::to_value(ListenResponse { text }).unwrap()),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(ApiResponse::err(e)).unwrap()),
        ),
    }
}

pub async fn snapshot_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // オンデマンドでカメラからBase64 JPEG画像を取得
    match state.hardware.take_snapshot(320, 240) {
        Ok(base64_str) => (
            StatusCode::OK,
            Json(
                serde_json::to_value(CameraSnapshotResponse {
                    image_base64: base64_str,
                    width: 320,
                    height: 240,
                })
                .unwrap(),
            ),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::to_value(ApiResponse::err(e)).unwrap()),
        ),
    }
}
