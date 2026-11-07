use crate::api::models::{
    ApiResponse, AudioStatus, BatteryStatus, BlockedStatus, CameraSnapshotResponse, CameraStatus,
    CliffStatus, DistanceStatus, DriveDirection, EventsQuery, EventsResponse, FloorStatus,
    GrayscaleResponse, ListenResponse, MotionStatus, MoveCommand, ReflexStatus, SensorsStatus,
    SpeakCommand, StatusResponse, SteerCommand, UltrasonicResponse,
};
use crate::rpi_camera::{CameraHub, FRAME_HEIGHT, FRAME_WIDTH, Frame};
#[cfg(test)]
use crate::safety::SafetyReason;
use crate::safety::{
    CALIBRATION_SAMPLES, CliffCalibration, DISTANCE_MAX_AGE, FLOOR_MAX_AGE, SafetyController,
    is_far, limit_speed,
};
use crate::sensor_monitor::SensorSnapshot;
use axum::{
    Json,
    body::Body,
    extract::{Query, Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use base64::Engine as _;
use std::convert::Infallible;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

const MAX_SPEED: i32 = 100;
const MIN_TTL_MS: u64 = 50;
/// PC が切断された時に走り続ける時間の上限
const MAX_TTL_MS: u64 = 1000;
const BATTERY_MAX_AGE: Duration = Duration::from_secs(5);
/// 単発撮影で古いフレームを返さないための上限
const SNAPSHOT_MAX_AGE: Duration = Duration::from_millis(500);
/// カメラの起動を含めた単発撮影の待ち時間
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(4);
const CALIBRATION_TIMEOUT: Duration = Duration::from_secs(2);
const CALIBRATION_POLL: Duration = Duration::from_millis(20);
const STREAM_BOUNDARY: &str = "frame";

/// ハードウェア抽象化インターフェース
pub trait RobotHardware: Send + Sync {
    /// モーターを動かすだけで、止める時刻は反射ループが管理する
    fn drive(&self, direction: DriveDirection, speed: i32) -> Result<(), String>;
    fn steer(&self, angle: i32) -> Result<(), String>;
    fn stop(&self) -> Result<(), String>;
    fn sensors(&self) -> SensorSnapshot;
    /// 音声を生成して再生キューへ追加する。再生の完了は待たない
    fn speak(&self, text: &str) -> Result<(), String>;
    /// 再生中と生成中の発話をすべて止める
    fn stop_speech(&self) -> Result<(), String>;
    fn is_speaking(&self) -> bool;
    fn listen(&self) -> Result<String, String>;
}

/// 共有アプリケーション状態
pub struct AppState {
    pub hardware: Box<dyn RobotHardware>,
    /// 判定とモーター指令を直列化するため、モーターを動かす間はこのロックを持つ
    pub safety: Mutex<SafetyController>,
    /// 全エンドポイントで要求する Bearer トークン
    pub api_token: String,
    pub camera: Arc<CameraHub>,
}

impl AppState {
    /// 安全系のロックを取得する。パニック後も停止処理を続けられるようにする。
    pub fn safety(&self) -> MutexGuard<'_, SafetyController> {
        self.safety
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// モーターを止める。失敗した場合は反射ループが再送する。
    pub fn stop_motors(&self) -> Result<(), String> {
        let mut safety = self.safety();
        safety.clear_motion();
        let result = self.hardware.stop();
        safety.note_stop_result(&result);
        result
    }
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
    if !(MIN_TTL_MS..=MAX_TTL_MS).contains(&cmd.ttl_ms) {
        return Err(format!(
            "ttl_msは{MIN_TTL_MS}から{MAX_TTL_MS}の間である必要があります"
        ));
    }
    Ok(())
}

pub async fn move_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<MoveCommand>,
) -> impl IntoResponse {
    if let Err(message) = validate_move(&cmd) {
        return (StatusCode::BAD_REQUEST, Json(ApiResponse::err(message)));
    }

    if cmd.direction == DriveDirection::Stop {
        return stop_response(state.stop_motors());
    }

    let sensors = state.hardware.sensors();
    let now = Instant::now();
    let mut safety = state.safety();

    if let Err(reason) = safety.check_start(cmd.direction, &sensors, now) {
        // 走行中に別の方向の指令が拒否されても、今の走行の判定は反射ループが続ける
        return (StatusCode::CONFLICT, Json(ApiResponse::refused(reason)));
    }

    let distance = sensors
        .distance_cm
        .and_then(|r| r.fresh(now, DISTANCE_MAX_AGE));
    let speed = limit_speed(cmd.direction, cmd.speed, distance);

    match state.hardware.drive(cmd.direction, speed) {
        Ok(()) => {
            let ttl = Duration::from_millis(cmd.ttl_ms);
            safety.begin_motion(cmd.direction, speed, ttl, Instant::now());
            (StatusCode::OK, Json(ApiResponse::ok()))
        }
        Err(error) => {
            // 片方のモーターだけ動いている可能性があるので止める
            safety.clear_motion();
            let stop = state.hardware.stop();
            safety.note_stop_result(&stop);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ApiResponse::err(error)),
            )
        }
    }
}

fn stop_response(result: Result<(), String>) -> (StatusCode, Json<ApiResponse>) {
    match result {
        Ok(()) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn steer_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<SteerCommand>,
) -> impl IntoResponse {
    if !(-40..=40).contains(&cmd.angle) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ApiResponse::err(
                "ステアリング角度は-40から40の間である必要があります",
            )),
        );
    }

    match state.hardware.steer(cmd.angle) {
        Ok(_) => {
            state.safety().set_steer_angle(cmd.angle);
            (StatusCode::OK, Json(ApiResponse::ok()))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

/// 通常停止。モーターを止めるだけで、以後の走行は拒否しない。
pub async fn stop_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    stop_response(state.stop_motors())
}

/// 緊急停止。解除されるまで走行を拒否する。
pub async fn emergency_stop_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut safety = state.safety();
    safety.set_emergency_stopped(true);
    let result = state.hardware.stop();
    safety.note_stop_result(&result);
    stop_response(result)
}

/// 緊急停止を解除する。車両は動かさない。
pub async fn emergency_stop_reset_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.safety().set_emergency_stopped(false);
    (StatusCode::OK, Json(ApiResponse::ok()))
}

fn millis(duration: Duration) -> u64 {
    duration.as_millis().try_into().unwrap_or(u64::MAX)
}

pub async fn ultrasonic_handler(State(state): State<Arc<AppState>>) -> Response {
    let now = Instant::now();
    let reading = state.hardware.sensors().distance_cm;
    match reading.and_then(|r| r.fresh(now, DISTANCE_MAX_AGE).map(|cm| (cm, r.age(now)))) {
        Some((cm, age)) => Json(UltrasonicResponse {
            distance_cm: cm as f32,
            age_ms: millis(age),
        })
        .into_response(),
        None => unknown_sensor("距離が不明です"),
    }
}

pub async fn grayscale_handler(State(state): State<Arc<AppState>>) -> Response {
    let now = Instant::now();
    let reading = state.hardware.sensors().grayscale;
    match reading.and_then(|r| r.fresh(now, FLOOR_MAX_AGE).map(|v| (v, r.age(now)))) {
        Some(([left, middle, right], age)) => Json(GrayscaleResponse {
            left: left.into(),
            middle: middle.into(),
            right: right.into(),
            age_ms: millis(age),
        })
        .into_response(),
        None => unknown_sensor("床センサーの値が不明です"),
    }
}

fn unknown_sensor(message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ApiResponse::err(message)),
    )
        .into_response()
}

fn build_status(state: &AppState, now: Instant) -> StatusResponse {
    let sensors = state.hardware.sensors();
    let speaking = state.hardware.is_speaking();
    let safety = state.safety();
    let cliff = safety.cliff();

    let floor_values = sensors.grayscale.and_then(|r| r.fresh(now, FLOOR_MAX_AGE));
    let battery = sensors.battery.and_then(|r| r.fresh(now, BATTERY_MAX_AGE));
    let blocked = |direction| safety.check_start(direction, &sensors, now).err();
    let reflex = safety.reflex_stats();
    let camera = state.camera.stats(now);
    let emergency_stop = safety.is_emergency_stopped();
    let motion = safety.motion();

    StatusResponse {
        mode: match motion.map(|m| m.direction) {
            _ if emergency_stop => "emergency_stop",
            Some(DriveDirection::Forward) => "forward",
            Some(DriveDirection::Backward) => "backward",
            _ => "idle",
        },
        emergency_stop,
        motion: motion.map(|m| MotionStatus {
            direction: m.direction,
            speed: m.speed,
            ttl_remaining_ms: millis(m.deadline.saturating_duration_since(now)),
        }),
        steer_angle: safety.steer_angle(),
        blocked: BlockedStatus {
            forward: blocked(DriveDirection::Forward),
            backward: blocked(DriveDirection::Backward),
        },
        sensors: SensorsStatus {
            distance: DistanceStatus {
                cm: sensors
                    .distance_cm
                    .and_then(|r| r.fresh(now, DISTANCE_MAX_AGE)),
                age_ms: sensors.distance_cm.map(|r| millis(r.age(now))),
                far: sensors
                    .distance_cm
                    .and_then(|r| r.fresh(now, DISTANCE_MAX_AGE))
                    .is_some_and(is_far),
            },
            floor: FloorStatus {
                values: floor_values,
                age_ms: sensors.grayscale.map(|r| millis(r.age(now))),
                cliff: floor_values.zip(cliff).map(|(v, c)| c.is_cliff(v)),
            },
            battery: BatteryStatus {
                voltage: battery.map(|(voltage, _)| voltage),
                percent: battery.map(|(_, percent)| percent),
                age_ms: sensors.battery.map(|r| millis(r.age(now))),
            },
        },
        cliff: CliffStatus {
            guard_enabled: safety.cliff_guard_enabled(),
            calibration: cliff,
        },
        reflex: ReflexStatus {
            hz: if reflex.interval_ms_ema > 0.0 {
                1000.0 / reflex.interval_ms_ema
            } else {
                0.0
            },
            max_interval_ms: reflex.max_interval_ms,
            last_tick_age_ms: reflex
                .last_tick
                .map(|at| millis(now.saturating_duration_since(at))),
        },
        camera: CameraStatus {
            running: camera.running,
            fps: camera.fps,
            frames: camera.frames,
            frame_age_ms: camera.frame_age.map(millis),
            error: state.camera.last_error(),
        },
        audio: AudioStatus { speaking },
        latest_event_id: safety.latest_event_id(),
    }
}

pub async fn status_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(build_status(&state, Instant::now()))
}

pub async fn events_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EventsQuery>,
) -> impl IntoResponse {
    let safety = state.safety();
    Json(EventsResponse {
        events: safety.events_after(query.after),
        latest_event_id: safety.latest_event_id(),
    })
}

/// 停止中の床センサーから段差の閾値を校正する。結果は Pi 側で保持する。
pub async fn calibrate_floor(state: &AppState) -> Result<CliffCalibration, (StatusCode, String)> {
    let moving = |state: &AppState| state.safety().motion().is_some();
    let busy = || {
        (
            StatusCode::CONFLICT,
            "走行中は校正できません。停止してから実行してください".to_string(),
        )
    };
    if moving(state) {
        return Err(busy());
    }

    let mut samples = Vec::with_capacity(CALIBRATION_SAMPLES);
    let mut last_at = None;
    let deadline = Instant::now() + CALIBRATION_TIMEOUT;
    while samples.len() < CALIBRATION_SAMPLES && Instant::now() < deadline {
        if let Some(reading) = state.hardware.sensors().grayscale
            && last_at != Some(reading.at)
            && reading.fresh(Instant::now(), FLOOR_MAX_AGE).is_some()
        {
            samples.push(reading.value);
            last_at = Some(reading.at);
        }
        tokio::time::sleep(CALIBRATION_POLL).await;
    }

    let mut safety = state.safety();
    if safety.motion().is_some() {
        return Err(busy());
    }
    match CliffCalibration::from_samples(&samples) {
        Ok(calibration) => {
            safety.set_cliff(calibration);
            Ok(calibration)
        }
        Err(error) => {
            safety.record_calibration_failure(&error);
            Err((StatusCode::UNPROCESSABLE_ENTITY, error))
        }
    }
}

pub async fn calibrate_handler(State(state): State<Arc<AppState>>) -> Response {
    match calibrate_floor(&state).await {
        Ok(calibration) => Json(calibration).into_response(),
        Err((status, message)) => (status, Json(ApiResponse::err(message))).into_response(),
    }
}

pub async fn speak_handler(
    State(state): State<Arc<AppState>>,
    Json(cmd): Json<SpeakCommand>,
) -> impl IntoResponse {
    // 音声合成は数百ミリ秒かかるので、非同期の処理を止めないよう別スレッドで行う
    let result = tokio::task::spawn_blocking(move || state.hardware.speak(&cmd.text))
        .await
        .unwrap_or_else(|error| Err(format!("音声合成の処理が異常終了しました: {error}")));
    match result {
        Ok(_) => (StatusCode::OK, Json(ApiResponse::ok())),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ApiResponse::err(e))),
    }
}

pub async fn stop_speech_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.hardware.stop_speech() {
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

/// カメラの最新フレームを返す。必要ならカメラを起動して待つ。
pub async fn snapshot_handler(State(state): State<Arc<AppState>>) -> Response {
    state.camera.note_request();
    let mut frames = state.camera.subscribe();
    let wait = frames.wait_for(|frame| {
        frame
            .as_ref()
            .is_some_and(|f| f.captured_at.elapsed() <= SNAPSHOT_MAX_AGE)
    });
    let frame = match tokio::time::timeout(SNAPSHOT_TIMEOUT, wait).await {
        Ok(Ok(frame)) => frame.clone(),
        _ => None,
    };
    let Some(frame) = frame else {
        let detail = state
            .camera
            .last_error()
            .unwrap_or_else(|| "カメラからフレームを取得できません".to_string());
        return unknown_sensor(&detail);
    };

    Json(CameraSnapshotResponse {
        image_base64: base64::engine::general_purpose::STANDARD.encode(&frame.jpeg),
        width: FRAME_WIDTH,
        height: FRAME_HEIGHT,
        seq: frame.seq,
        age_ms: millis(frame.captured_at.elapsed()),
    })
    .into_response()
}

fn multipart_chunk(frame: &Frame) -> Vec<u8> {
    let header = format!(
        "--{STREAM_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nX-Frame-Seq: {}\r\n\r\n",
        frame.jpeg.len(),
        frame.seq
    );
    let mut chunk = Vec::with_capacity(header.len() + frame.jpeg.len() + 2);
    chunk.extend_from_slice(header.as_bytes());
    chunk.extend_from_slice(&frame.jpeg);
    chunk.extend_from_slice(b"\r\n");
    chunk
}

/// MJPEG の常時配信。受信者が遅い時は途中のフレームを飛ばし、常に最新を送る。
pub async fn stream_handler(State(state): State<Arc<AppState>>) -> Response {
    // 再接続の間にカメラを止めると、起動待ちと切断を繰り返す恐れがある
    state.camera.note_request();
    let frames = state.camera.subscribe();
    let stream = futures_util::stream::unfold(frames, |mut frames| async move {
        frames.changed().await.ok()?;
        let frame = frames.borrow_and_update().clone()?;
        Some((Ok::<_, Infallible>(multipart_chunk(&frame)), frames))
    });

    (
        [
            (
                header::CONTENT_TYPE,
                format!("multipart/x-mixed-replace; boundary={STREAM_BOUNDARY}"),
            ),
            (header::CACHE_CONTROL, "no-store".to_string()),
        ],
        Body::from_stream(stream),
    )
        .into_response()
}

/// 拒否応答の `reason` を読み取る
#[cfg(test)]
pub(crate) fn refusal_reason(body: &serde_json::Value) -> Option<SafetyReason> {
    serde_json::from_value(body.get("reason")?.clone()).ok()
}
