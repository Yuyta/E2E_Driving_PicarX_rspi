pub mod handlers;
pub mod hardware;
pub mod models;

use crate::api::handlers::{
    AppState, RobotHardware, calibrate_floor, calibrate_handler, emergency_stop_handler,
    emergency_stop_reset_handler, events_handler, grayscale_handler, listen_handler, move_handler,
    require_token, snapshot_handler, speak_handler, status_handler, steer_handler, stop_handler,
    stream_handler, ultrasonic_handler,
};
use crate::api::models::DriveDirection;
use crate::rpi_camera::{CameraHub, spawn_camera_supervisor};
use crate::safety::SafetyController;
use crate::safety::reflex::spawn_reflex_loop;
use crate::sensor_monitor::{SensorSnapshot, Timed};
use axum::{
    Router, middleware,
    routing::{get, post},
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// APIトークンを指定する環境変数
const API_TOKEN_ENV: &str = "PICARX_API_TOKEN";
/// ハードウェア初期化失敗時のモック起動を許可する環境変数
const ALLOW_MOCK_ENV: &str = "PICARX_ALLOW_MOCK";
/// 段差検知を無効にする環境変数。車輪を浮かせて試験する時など、落下の恐れがない時だけ使う
const DISABLE_CLIFF_GUARD_ENV: &str = "PICARX_DISABLE_CLIFF_GUARD";

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/drive/move", post(move_handler))
        .route("/api/drive/steer", post(steer_handler))
        .route("/api/drive/stop", post(stop_handler))
        .route("/api/drive/emergency_stop", post(emergency_stop_handler))
        .route(
            "/api/drive/emergency_stop/reset",
            post(emergency_stop_reset_handler),
        )
        .route("/api/safety/calibrate", post(calibrate_handler))
        .route("/api/status", get(status_handler))
        .route("/api/events", get(events_handler))
        .route("/api/sensor/ultrasonic", get(ultrasonic_handler))
        .route("/api/sensor/grayscale", get(grayscale_handler))
        .route("/api/audio/speak", post(speak_handler))
        .route("/api/audio/listen", get(listen_handler))
        .route("/api/camera/snapshot", get(snapshot_handler))
        .route("/api/camera/stream", get(stream_handler))
        .layer(middleware::from_fn_with_state(state.clone(), require_token))
        .with_state(state)
}

/// 実機初期化失敗時のフォールバックモック
struct MockFallbackHardware;

impl RobotHardware for MockFallbackHardware {
    fn drive(&self, _direction: DriveDirection, _speed: i32) -> Result<(), String> {
        Ok(())
    }
    fn steer(&self, _angle: i32) -> Result<(), String> {
        Ok(())
    }
    fn stop(&self) -> Result<(), String> {
        Ok(())
    }
    fn sensors(&self) -> SensorSnapshot {
        let now = Instant::now();
        SensorSnapshot {
            distance_cm: Some(Timed::new(100.0, now)),
            grayscale: Some(Timed::new([500, 500, 500], now)),
            battery: Some(Timed::new((7.4, 50), now)),
        }
    }
    fn speak(&self, _text: &str) -> Result<(), String> {
        Ok(())
    }
    fn listen(&self) -> Result<String, String> {
        Ok("Mock Audio Listen".to_string())
    }
}

/// 終了シグナルを待つ
async fn shutdown_signal() -> std::io::Result<()> {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate())?;
    tokio::select! {
        result = tokio::signal::ctrl_c() => result,
        _ = terminate.recv() => Ok(()),
    }
}

/// API サーバーを起動
pub async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let api_token = std::env::var(API_TOKEN_ENV).unwrap_or_default();
    if api_token.is_empty() {
        return Err(format!("{API_TOKEN_ENV} を設定してください").into());
    }

    let allow_mock = std::env::var(ALLOW_MOCK_ENV).is_ok_and(|value| value == "1");
    let (hardware, is_real): (Box<dyn RobotHardware>, bool) = match hardware::PiCarHardware::new() {
        Ok(hw) => {
            println!("✓ PiCar-X 実機ハードウェアを初期化しました");
            (Box::new(hw), true)
        }
        Err(e) if allow_mock => {
            eprintln!("⚠ 実機ハードウェア初期化失敗 ({ALLOW_MOCK_ENV}=1 のためモックで起動): {e}");
            (Box::new(MockFallbackHardware), false)
        }
        Err(e) => {
            return Err(format!("実機ハードウェアの初期化に失敗しました: {e}").into());
        }
    };

    let cliff_guard_enabled = !std::env::var(DISABLE_CLIFF_GUARD_ENV).is_ok_and(|v| v == "1");
    if !cliff_guard_enabled {
        eprintln!("⚠ {DISABLE_CLIFF_GUARD_ENV}=1: 段差検知は無効です。落下に注意してください");
    }

    let camera = Arc::new(CameraHub::new());
    if is_real {
        spawn_camera_supervisor(Arc::clone(&camera))?;
    }

    let state = Arc::new(AppState {
        hardware,
        safety: Mutex::new(SafetyController::new(cliff_guard_enabled)),
        api_token,
        camera,
    });
    spawn_reflex_loop(Arc::clone(&state))?;

    if cliff_guard_enabled {
        match calibrate_floor(&state).await {
            Ok(calibration) => println!("✓ 段差の閾値を校正しました: {calibration:?}"),
            Err((_, error)) => eprintln!(
                "⚠ 段差の校正に失敗しました。POST /api/safety/calibrate まで前進できません: {error}"
            ),
        }
    }

    let app = create_router(Arc::clone(&state));
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!(
        "🚀 PiCar-X API サーバーが起動しました: http://0.0.0.0:{}",
        port
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    let serve = async { axum::serve(listener, app).await };
    tokio::select! {
        result = serve => result?,
        result = shutdown_signal() => {
            result?;
            println!("終了シグナルを受信しました。モーターを停止します");
            if let Err(error) = state.stop_motors() {
                eprintln!("⚠ 終了時の停止に失敗しました: {error}");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::handlers::refusal_reason;
    use crate::picarx::ultrasonic::FAR_DISTANCE_CM;
    use crate::safety::reflex::reflex_tick;
    use crate::safety::{CALIBRATION_SAMPLES, CliffCalibration, FAR_MAX_SPEED, SafetyReason};
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;
    use tower::util::ServiceExt; // for oneshot

    const TOKEN: &str = "test-token";
    const FORWARD: &str = r#"{"direction":"forward","speed":30,"ttl_ms":500}"#;
    const BACKWARD: &str = r#"{"direction":"backward","speed":30,"ttl_ms":500}"#;
    const SHORT_TTL: &str = r#"{"direction":"forward","speed":30,"ttl_ms":50}"#;
    const FLOOR: [u16; 3] = [500, 500, 500];

    /// `None` の項目は読み取り失敗を表す
    #[derive(Clone, Copy)]
    struct MockSensors {
        distance: Option<f64>,
        floor: Option<[u16; 3]>,
    }

    #[derive(Default)]
    struct Counters {
        drive: AtomicUsize,
        last_speed: AtomicUsize,
        stop: AtomicUsize,
        fail_stop: AtomicBool,
    }

    struct MockHardware {
        sensors: Arc<Mutex<MockSensors>>,
        counters: Arc<Counters>,
    }

    impl RobotHardware for MockHardware {
        fn drive(&self, _direction: DriveDirection, speed: i32) -> Result<(), String> {
            self.counters.drive.fetch_add(1, Ordering::SeqCst);
            self.counters
                .last_speed
                .store(speed as usize, Ordering::SeqCst);
            Ok(())
        }
        fn steer(&self, _angle: i32) -> Result<(), String> {
            Ok(())
        }
        fn stop(&self) -> Result<(), String> {
            self.counters.stop.fetch_add(1, Ordering::SeqCst);
            if self.counters.fail_stop.load(Ordering::SeqCst) {
                return Err("i2c error".to_string());
            }
            Ok(())
        }
        fn sensors(&self) -> SensorSnapshot {
            let now = Instant::now();
            let sensors = *self.sensors.lock().unwrap();
            SensorSnapshot {
                distance_cm: sensors.distance.map(|d| Timed::new(d, now)),
                grayscale: sensors.floor.map(|f| Timed::new(f, now)),
                battery: Some(Timed::new((7.6, 50), now)),
            }
        }
        fn speak(&self, _text: &str) -> Result<(), String> {
            Ok(())
        }
        fn listen(&self) -> Result<String, String> {
            Ok("テスト音声認識".to_string())
        }
    }

    struct TestApp {
        state: Arc<AppState>,
        router: Router,
        sensors: Arc<Mutex<MockSensors>>,
        counters: Arc<Counters>,
    }

    impl TestApp {
        fn new(distance: f64) -> Self {
            let sensors = Arc::new(Mutex::new(MockSensors {
                distance: Some(distance),
                floor: Some(FLOOR),
            }));
            let counters = Arc::new(Counters::default());
            let mut safety = SafetyController::new(true);
            safety
                .set_cliff(CliffCalibration::from_samples(&[FLOOR; CALIBRATION_SAMPLES]).unwrap());
            let state = Arc::new(AppState {
                hardware: Box::new(MockHardware {
                    sensors: Arc::clone(&sensors),
                    counters: Arc::clone(&counters),
                }),
                safety: Mutex::new(safety),
                api_token: TOKEN.to_string(),
                camera: Arc::new(CameraHub::new()),
            });
            Self {
                router: create_router(Arc::clone(&state)),
                state,
                sensors,
                counters,
            }
        }

        fn set_distance(&self, distance: Option<f64>) {
            self.sensors.lock().unwrap().distance = distance;
        }

        fn set_floor(&self, floor: Option<[u16; 3]>) {
            self.sensors.lock().unwrap().floor = floor;
        }

        fn drives(&self) -> usize {
            self.counters.drive.load(Ordering::SeqCst)
        }

        fn stops(&self) -> usize {
            self.counters.stop.load(Ordering::SeqCst)
        }

        /// 反射ループを進めずに要求を送る
        async fn request(
            &self,
            method: &str,
            uri: &str,
            body: Option<&str>,
        ) -> (StatusCode, serde_json::Value) {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", format!("Bearer {TOKEN}"))
                .header("content-type", "application/json")
                .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
                .unwrap();
            let response = self.router.clone().oneshot(request).await.unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            let value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
            (status, value)
        }

        /// 走行要求を送る。実機では反射ループが常に動いているので、直前に1周進める。
        async fn send_move_json(&self, body: &str) -> (StatusCode, serde_json::Value) {
            reflex_tick(&self.state);
            self.request("POST", "/api/drive/move", Some(body)).await
        }

        async fn send_move(&self, body: &str) -> StatusCode {
            self.send_move_json(body).await.0
        }

        async fn send(&self, method: &str, uri: &str, body: Option<&str>) -> StatusCode {
            self.request(method, uri, body).await.0
        }

        async fn status(&self) -> serde_json::Value {
            self.request("GET", "/api/status", None).await.1
        }

        async fn last_event(&self) -> serde_json::Value {
            let (_, body) = self.request("GET", "/api/events", None).await;
            body["events"].as_array().unwrap().last().unwrap().clone()
        }
    }

    #[tokio::test]
    async fn test_drive_move_safe() {
        let app = TestApp::new(30.0); // 障害物まで30cm (安全)
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);
        assert_eq!(app.drives(), 1);
    }

    #[tokio::test]
    async fn test_drive_move_unsafe_obstacle() {
        let app = TestApp::new(10.0); // 障害物まで10cm (危険)
        let (status, body) = app.send_move_json(FORWARD).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(refusal_reason(&body), Some(SafetyReason::Obstacle));
        assert_eq!(app.drives(), 0);
    }

    #[tokio::test]
    async fn motion_is_refused_before_the_reflex_loop_has_started() {
        let app = TestApp::new(30.0);
        let (status, body) = app.request("POST", "/api/drive/move", Some(FORWARD)).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(refusal_reason(&body), Some(SafetyReason::ReflexStalled));
    }

    #[tokio::test]
    async fn no_echo_drives_at_a_limited_speed_and_is_reported() {
        let app = TestApp::new(FAR_DISTANCE_CM);
        let fast = r#"{"direction":"forward","speed":100,"ttl_ms":500}"#;
        assert_eq!(app.send_move(fast).await, StatusCode::OK);
        assert_eq!(
            app.counters.last_speed.load(Ordering::SeqCst),
            FAR_MAX_SPEED as usize
        );
        let status = app.status().await;
        assert_eq!(status["sensors"]["distance"]["far"], true);
        assert_eq!(status["motion"]["speed"], FAR_MAX_SPEED);

        app.set_distance(Some(40.0));
        assert_eq!(app.send_move(fast).await, StatusCode::OK);
        assert_eq!(app.counters.last_speed.load(Ordering::SeqCst), 100);
        assert_eq!(app.status().await["sensors"]["distance"]["far"], false);
    }

    #[tokio::test]
    async fn unknown_distance_is_not_treated_as_safe() {
        let app = TestApp::new(30.0);
        app.set_distance(None);
        let (status, body) = app.send_move_json(FORWARD).await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(refusal_reason(&body), Some(SafetyReason::DistanceUnknown));
        assert_eq!(
            app.send("GET", "/api/sensor/ultrasonic", None).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn unknown_floor_is_not_treated_as_safe() {
        let app = TestApp::new(30.0);
        app.set_floor(None);
        assert_eq!(app.send_move(BACKWARD).await, StatusCode::CONFLICT);
        assert_eq!(
            app.send("GET", "/api/sensor/grayscale", None).await,
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn reflex_stops_a_running_motor_when_an_obstacle_appears() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);

        app.set_distance(Some(10.0));
        reflex_tick(&app.state);

        assert_eq!(app.stops(), 1);
        assert_eq!(app.status().await["mode"], "idle");
        let event = app.last_event().await;
        assert_eq!(event["kind"], "reflex_stop");
        assert_eq!(event["reason"], "obstacle");
        assert!(event["stop_latency_ms"].is_number());
    }

    #[tokio::test]
    async fn reflex_stops_at_a_cliff_and_allows_backing_away() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);

        app.set_floor(Some([500, 8, 500]));
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 1);
        assert_eq!(app.last_event().await["reason"], "cliff");

        assert_eq!(app.send_move(FORWARD).await, StatusCode::CONFLICT);
        assert_eq!(app.send_move(BACKWARD).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn reflex_stops_when_the_ttl_expires_without_a_new_command() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(SHORT_TTL).await, StatusCode::OK);

        reflex_tick(&app.state);
        assert_eq!(app.stops(), 0);

        tokio::time::sleep(Duration::from_millis(60)).await;
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 1);
        assert_eq!(app.last_event().await["reason"], "ttl_expired");

        // 停止後は走行中ではないので、繰り返し止めない
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 1);
    }

    #[tokio::test]
    async fn a_new_command_does_not_get_stopped_by_an_older_deadline() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(SHORT_TTL).await, StatusCode::OK);
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);

        tokio::time::sleep(Duration::from_millis(60)).await;
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_reflex_thread_stops_the_motor_without_any_client_traffic() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(SHORT_TTL).await, StatusCode::OK);
        spawn_reflex_loop(Arc::clone(&app.state)).unwrap();

        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(app.stops(), 1);
    }

    #[tokio::test]
    async fn failed_stop_is_retried_until_it_succeeds() {
        let app = TestApp::new(30.0);
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);

        app.counters.fail_stop.store(true, Ordering::SeqCst);
        app.set_distance(Some(10.0));
        reflex_tick(&app.state);
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 2);

        app.counters.fail_stop.store(false, Ordering::SeqCst);
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 3);
        reflex_tick(&app.state);
        assert_eq!(app.stops(), 3);
    }

    #[tokio::test]
    async fn status_reports_sensors_mode_and_blocks() {
        let app = TestApp::new(30.0);
        reflex_tick(&app.state);
        let status = app.status().await;
        assert_eq!(status["mode"], "idle");
        assert_eq!(status["sensors"]["distance"]["cm"], 30.0);
        assert_eq!(status["sensors"]["battery"]["percent"], 50);
        assert_eq!(status["sensors"]["floor"]["values"][0], 500);
        assert!(status["blocked"]["forward"].is_null());

        app.send_move(FORWARD).await;
        let status = app.status().await;
        assert_eq!(status["mode"], "forward");
        assert!(status["motion"]["ttl_remaining_ms"].as_u64().unwrap() <= 500);

        app.set_distance(None);
        let status = app.status().await;
        assert!(status["sensors"]["distance"]["cm"].is_null());
        assert_eq!(status["blocked"]["forward"], "distance_unknown");
        assert!(status["blocked"]["backward"].is_null());
    }

    #[tokio::test]
    async fn events_can_be_read_incrementally() {
        let app = TestApp::new(30.0);
        app.send("POST", "/api/drive/emergency_stop", None).await;
        let (_, first) = app.request("GET", "/api/events", None).await;
        let latest = first["latest_event_id"].as_u64().unwrap();
        assert_eq!(
            first["events"].as_array().unwrap().last().unwrap()["kind"],
            "emergency_stop"
        );

        let uri = format!("/api/events?after={latest}");
        let (_, empty) = app.request("GET", &uri, None).await;
        assert!(empty["events"].as_array().unwrap().is_empty());

        app.send("POST", "/api/drive/emergency_stop/reset", None)
            .await;
        let (_, next) = app.request("GET", &uri, None).await;
        assert_eq!(next["events"][0]["kind"], "emergency_stop_reset");
    }

    #[tokio::test]
    async fn calibration_is_refused_while_moving() {
        let app = TestApp::new(30.0);
        app.send_move(FORWARD).await;
        assert_eq!(
            app.send("POST", "/api/safety/calibrate", None).await,
            StatusCode::CONFLICT
        );
    }

    #[tokio::test]
    async fn calibration_reads_the_floor_and_rejects_a_lifted_car() {
        let app = TestApp::new(30.0);
        let (status, body) = app.request("POST", "/api/safety/calibrate", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["baseline"][0], 500);

        app.set_floor(Some([8, 13, 11]));
        assert_eq!(
            app.send("POST", "/api/safety/calibrate", None).await,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        // 失敗しても以前の校正値は残る
        assert!(app.status().await["cliff"]["calibration"].is_object());
    }

    #[tokio::test]
    async fn snapshot_returns_the_latest_frame() {
        let app = TestApp::new(30.0);
        app.state.camera.publish(vec![0xFF, 0xD8, 0xFF, 0xD9]);
        let (status, body) = app.request("GET", "/api/camera/snapshot", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["image_base64"], "/9j/2Q==");
        assert_eq!(body["seq"], 1);
    }

    #[tokio::test]
    async fn stream_sends_frames_as_multipart_parts() {
        let app = TestApp::new(30.0);
        let request = Request::builder()
            .uri("/api/camera/stream")
            .header("authorization", format!("Bearer {TOKEN}"))
            .body(Body::empty())
            .unwrap();
        let response = app.router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()["content-type"],
            "multipart/x-mixed-replace; boundary=frame"
        );

        app.state.camera.publish(vec![0xFF, 0xD8, 0xFF, 0xD9]);
        let mut body = response.into_body().into_data_stream();
        let chunk = tokio::time::timeout(
            Duration::from_secs(1),
            futures_util::StreamExt::next(&mut body),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        let text = String::from_utf8_lossy(&chunk);
        assert!(text.starts_with("--frame\r\nContent-Type: image/jpeg\r\n"));
        assert!(text.contains("Content-Length: 4\r\n"));
        assert!(chunk.ends_with(&[0xFF, 0xD8, 0xFF, 0xD9, b'\r', b'\n']));
    }

    #[tokio::test]
    async fn stream_requires_the_token() {
        let app = TestApp::new(30.0);
        let request = Request::builder()
            .uri("/api/camera/stream")
            .body(Body::empty())
            .unwrap();
        let response = app.router.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_sensor_ultrasonic() {
        let app = TestApp::new(25.5);
        let (status, body) = app.request("GET", "/api/sensor/ultrasonic", None).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["distance_cm"], 25.5);
    }

    #[tokio::test]
    async fn test_audio_speak_and_listen() {
        let app = TestApp::new(50.0);
        assert_eq!(
            app.send("POST", "/api/audio/speak", Some(r#"{"text":"こんにちは"}"#))
                .await,
            StatusCode::OK
        );
        assert_eq!(
            app.send("GET", "/api/audio/listen", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn emergency_stop_blocks_motion_until_reset() {
        let app = TestApp::new(30.0);

        assert_eq!(
            app.send("POST", "/api/drive/emergency_stop", None).await,
            StatusCode::OK
        );
        assert_eq!(app.stops(), 1);

        assert_eq!(app.send_move(FORWARD).await, StatusCode::CONFLICT);
        assert_eq!(app.send_move(BACKWARD).await, StatusCode::CONFLICT);
        // 拒否された要求の後も保持され続ける
        assert_eq!(app.send_move(FORWARD).await, StatusCode::CONFLICT);
        assert_eq!(app.drives(), 0);
        assert_eq!(app.status().await["mode"], "emergency_stop");

        assert_eq!(
            app.send("POST", "/api/drive/emergency_stop/reset", None)
                .await,
            StatusCode::OK
        );
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);
        assert_eq!(app.drives(), 1);
    }

    #[tokio::test]
    async fn plain_stop_does_not_block_following_moves() {
        let app = TestApp::new(30.0);
        assert_eq!(
            app.send("POST", "/api/drive/stop", None).await,
            StatusCode::OK
        );
        assert_eq!(app.stops(), 1);
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);
    }

    #[tokio::test]
    async fn stop_direction_is_allowed_during_emergency_stop() {
        let app = TestApp::new(30.0);
        app.send("POST", "/api/drive/emergency_stop", None).await;
        assert_eq!(
            app.send_move(r#"{"direction":"stop"}"#).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn invalid_move_values_are_rejected() {
        let app = TestApp::new(30.0);
        for body in [
            r#"{"direction":"forward","speed":30,"ttl_ms":0}"#,
            r#"{"direction":"forward","speed":30,"ttl_ms":49}"#,
            r#"{"direction":"forward","speed":30,"ttl_ms":1001}"#,
            r#"{"direction":"backward","speed":101,"ttl_ms":500}"#,
            r#"{"direction":"backward","speed":-1,"ttl_ms":500}"#,
        ] {
            assert_eq!(app.send_move(body).await, StatusCode::BAD_REQUEST, "{body}");
        }
        assert_eq!(app.drives(), 0);
    }

    #[tokio::test]
    async fn boundary_move_values_are_accepted() {
        let app = TestApp::new(30.0);
        for body in [
            r#"{"direction":"forward","speed":100,"ttl_ms":50}"#,
            r#"{"direction":"forward","speed":0,"ttl_ms":1000}"#,
        ] {
            assert_eq!(app.send_move(body).await, StatusCode::OK, "{body}");
        }
    }

    #[tokio::test]
    async fn requests_without_valid_token_are_rejected() {
        let app = TestApp::new(30.0);
        for auth in [
            None,
            Some("Bearer wrong-token"),
            Some(TOKEN),
            Some("Bearer "),
        ] {
            let mut builder = Request::builder()
                .method("POST")
                .uri("/api/drive/move")
                .header("content-type", "application/json");
            if let Some(value) = auth {
                builder = builder.header("authorization", value);
            }
            let request = builder.body(Body::from(FORWARD)).unwrap();
            let response = app.router.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED, "{auth:?}");
        }
        assert_eq!(app.drives(), 0);
    }

    #[test]
    fn token_comparison_rejects_empty_expected_token() {
        assert!(!crate::api::handlers::token_matches("", ""));
        assert!(crate::api::handlers::token_matches("abc", "abc"));
        assert!(!crate::api::handlers::token_matches("abc", "abd"));
    }
}
