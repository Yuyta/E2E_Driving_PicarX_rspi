//! 自律走行の HTTP の窓口。
//!
//! - `POST /api/autopilot/start` 本文は省略可: `{"speed": 50, "min_speed": 40, "intent": "any"}`
//! - `POST /api/autopilot/stop` 本文は省略可: `{"intervention": true}`（人の介入として記録する）
//! - `POST /api/autopilot/intent` `{"intent": "left"}`（上の層が分かれ道で向きを変える）
//! - `GET /api/autopilot/status`
//! - `GET /api/autopilot/model` 使用中のモデルの版と、ハッシュの確認結果

use crate::api::handlers::AppState;
use crate::api::models::ApiResponse;
use crate::autopilot::{Intent, RunParams};
use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use std::sync::Arc;

fn parse_body<T: for<'de> Deserialize<'de> + Default>(body: &Bytes) -> Result<T, String> {
    if body.iter().all(u8::is_ascii_whitespace) {
        return Ok(T::default());
    }
    serde_json::from_slice(body).map_err(|e| format!("本文の形式が違います: {e}"))
}

fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, Json(ApiResponse::err(message))).into_response()
}

pub async fn start_handler(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let params: RunParams = match parse_body(&body) {
        Ok(params) => params,
        Err(message) => return bad_request(message),
    };
    // モデルの確認と読み込みに時間がかかるので、別のスレッドで行う
    let worker = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || worker.autopilot.start(&worker, params)).await;
    match result {
        Ok(Ok(info)) => (StatusCode::OK, Json(info)).into_response(),
        Ok(Err(error)) => {
            let status = if state.autopilot.is_running() {
                StatusCode::CONFLICT
            } else {
                StatusCode::UNPROCESSABLE_ENTITY
            };
            (status, Json(ApiResponse::err(error))).into_response()
        }
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(error.to_string())),
        )
            .into_response(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct StopBody {
    intervention: bool,
}

impl Default for StopBody {
    fn default() -> Self {
        Self { intervention: true }
    }
}

pub async fn stop_handler(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let body: StopBody = match parse_body(&body) {
        Ok(body) => body,
        Err(message) => return bad_request(message),
    };
    let trigger = body.intervention.then_some("manual_intervention");
    if state
        .autopilot
        .request_stop(trigger, "autopilot stop command")
    {
        let worker = Arc::clone(&state);
        let _ = tokio::task::spawn_blocking(move || worker.autopilot.join()).await;
    }
    (StatusCode::OK, Json(state.autopilot.status())).into_response()
}

#[derive(Debug, Deserialize)]
pub struct IntentBody {
    pub intent: Intent,
}

pub async fn intent_handler(
    State(state): State<Arc<AppState>>,
    Json(body): Json<IntentBody>,
) -> Response {
    state.autopilot.set_intent(body.intent);
    (StatusCode::OK, Json(ApiResponse::ok())).into_response()
}

pub async fn status_handler(State(state): State<Arc<AppState>>) -> Response {
    (StatusCode::OK, Json(state.autopilot.status())).into_response()
}

pub async fn model_handler(State(state): State<Arc<AppState>>) -> Response {
    let worker = Arc::clone(&state);
    match tokio::task::spawn_blocking(move || worker.autopilot.model_info()).await {
        Ok(info) => (StatusCode::OK, Json(info)).into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ApiResponse::err(error.to_string())),
        )
            .into_response(),
    }
}
