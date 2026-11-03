use crate::api::models::{
    ApiResponse, CameraSnapshotResponse, DriveDirection, GrayscaleResponse, ListenResponse,
    MoveCommand, SpeakCommand, SteerCommand, UltrasonicResponse,
};
use crate::safety::SafetyGuard;
use axum::{
    Json,
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::{Arc, Mutex};

const MAX_SPEED: i32 = 100;
const MIN_DURATION_MS: u64 = 50;
const MAX_DURATION_MS: u64 = 5000;

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
    /// 全エンドポイントで要求する Bearer トークン
    pub api_token: String,
    /// 「緊急停止の確認からモーター指令まで」と「緊急停止の発動」を直列化する
    pub motion_gate: Mutex<()>,
}

/// トークンを定数時間で比較する。期待値が空の場合は常に不一致とする。
pub(crate) fn token_matches(expected: &str, provided: &str) -> bool {
    let (expected, provided) = (expected.as_bytes(), provided.as_bytes());
    if expected.is_empty() || expected.len() != provided.len() {
        return false;
    }
    expected
        .iter()
        .zip(provided)
        .fold(0u8, |diff, (a, b)| diff | (a ^ b))
        == 0
}

/// `Authorization: Bearer <token>` を検証するミドルウェア
pub async fn require_token(
    State(state): State<Arc<AppState>>,
    request: Request,
    next: Next,
) -> Response {
    let authorized = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .is_some_and(|token| token_matches(&state.api_token, token));

    if authorized {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(ApiResponse::err("APIトークンが無効です")),
        )
            .into_response()
    }
}

fn validate_move(cmd: &MoveCommand) -> Result<(), String> {
    if cmd.direction == DriveDirection::Stop {
        return Ok(());
    }
    if !(0..=MAX_SPEED).contains(&cmd.speed) {
        return Err(format!("speedは0から{MAX_SPEED}の間である必要があります"));
    }
    if !(MIN_DURATION_MS..=MAX_DURATION_MS).contains(&cmd.duration_ms) {
        return Err(format!(
            "duration_msは{MIN_DURATION_MS}から{MAX_DURATION_MS}の間である必要があります"
        ));
    }
    Ok(())
}

pub async fn move_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<MoveCommand>,
) -> impl IntoResponse {
    state.safety.record_command().await;

    if let Err(message) = validate_move(&cmd) {
        return (StatusCode::BAD_REQUEST, Json(ApiResponse::err(message)));
    }

    let _gate = state
        .motion_gate
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    if cmd.direction != DriveDirection::Stop && state.safety.is_emergency_stopped() {
        return (
            StatusCode::CONFLICT,
            Json(ApiResponse::err(
                "緊急停止中です。解除するまで走行できません",
            )),
        );
    }

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

/// 通常停止。モーターを止めるだけで、以後の走行は拒否しない。
pub async fn stop_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.hardware.stop() {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

/// 緊急停止。解除されるまで走行を拒否する。
pub async fn emergency_stop_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let _gate = state
        .motion_gate
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    state.safety.trigger_emergency_stop();
    match state.hardware.stop() {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

/// 緊急停止を解除する。車両は動かさない。
pub async fn emergency_stop_reset_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.safety.clear_emergency_stop();
    (StatusCode::OK, Json(ApiResponse::ok()))
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
