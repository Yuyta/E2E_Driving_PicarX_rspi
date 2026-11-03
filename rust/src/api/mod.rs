pub mod handlers;
pub mod hardware;
pub mod models;

use crate::api::handlers::{
    AppState, RobotHardware, emergency_stop_handler, emergency_stop_reset_handler,
    grayscale_handler, listen_handler, move_handler, require_token, snapshot_handler,
    speak_handler, steer_handler, stop_handler, ultrasonic_handler,
};
use crate::api::models::DriveDirection;
use crate::safety::SafetyGuard;
use axum::{
    Router, middleware,
    routing::{get, post},
};
use std::sync::{Arc, Mutex};

/// APIトークンを指定する環境変数
const API_TOKEN_ENV: &str = "PICARX_API_TOKEN";
/// ハードウェア初期化失敗時のモック起動を許可する環境変数
const ALLOW_MOCK_ENV: &str = "PICARX_ALLOW_MOCK";

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
        .route("/api/sensor/ultrasonic", get(ultrasonic_handler))
        .route("/api/sensor/grayscale", get(grayscale_handler))
        .route("/api/audio/speak", post(speak_handler))
        .route("/api/audio/listen", get(listen_handler))
        .route("/api/camera/snapshot", get(snapshot_handler))
        .layer(middleware::from_fn_with_state(state.clone(), require_token))
        .with_state(state)
}

/// 実機初期化失敗時のフォールバックモック
struct MockFallbackHardware;

impl RobotHardware for MockFallbackHardware {
    fn drive(
        &self,
        _direction: DriveDirection,
        _speed: i32,
        _duration_ms: u64,
    ) -> Result<(), String> {
        Ok(())
    }
    fn steer(&self, _angle: i32) -> Result<(), String> {
        Ok(())
    }
    fn stop(&self) -> Result<(), String> {
        Ok(())
    }
    fn get_distance(&self) -> Result<f32, String> {
        Ok(0.0)
    }
    fn get_grayscale(&self) -> Result<(i32, i32, i32), String> {
        Ok((0, 0, 0))
    }
    fn speak(&self, _text: &str) -> Result<(), String> {
        Ok(())
    }
    fn listen(&self) -> Result<String, String> {
        Ok("Mock Audio Listen".to_string())
    }
    fn take_snapshot(&self, _width: u32, _height: u32) -> Result<String, String> {
        Ok("".to_string())
    }
}

/// API サーバーを起動
pub async fn run_server(port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let api_token = std::env::var(API_TOKEN_ENV).unwrap_or_default();
    if api_token.is_empty() {
        return Err(format!("{API_TOKEN_ENV} を設定してください").into());
    }

    let allow_mock = std::env::var(ALLOW_MOCK_ENV).is_ok_and(|value| value == "1");
    let hardware: Box<dyn RobotHardware> = match hardware::PiCarHardware::new() {
        Ok(hw) => {
            println!("✓ PiCar-X 実機ハードウェアを初期化しました");
            Box::new(hw)
        }
        Err(e) if allow_mock => {
            eprintln!("⚠ 実機ハードウェア初期化失敗 ({ALLOW_MOCK_ENV}=1 のためモックで起動): {e}");
            Box::new(MockFallbackHardware)
        }
        Err(e) => {
            return Err(format!("実機ハードウェアの初期化に失敗しました: {e}").into());
        }
    };

    let safety = Arc::new(SafetyGuard::new(15.0, 3));
    let state = Arc::new(AppState {
        hardware,
        safety,
        api_token,
        motion_gate: Mutex::new(()),
    });
    let app = create_router(state);

    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    println!(
        "🚀 PiCar-X API サーバーが起動しました: http://0.0.0.0:{}",
        port
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::util::ServiceExt; // for oneshot

    const TOKEN: &str = "test-token";
    const FORWARD: &str = r#"{"direction":"forward","speed":30,"duration_ms":500}"#;
    const BACKWARD: &str = r#"{"direction":"backward","speed":30,"duration_ms":500}"#;

    struct MockHardware {
        mock_distance: f32,
        drive_calls: Arc<AtomicUsize>,
        stop_calls: Arc<AtomicUsize>,
    }

    impl RobotHardware for MockHardware {
        fn drive(
            &self,
            _direction: DriveDirection,
            _speed: i32,
            _duration_ms: u64,
        ) -> Result<(), String> {
            self.drive_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn steer(&self, _angle: i32) -> Result<(), String> {
            Ok(())
        }
        fn stop(&self) -> Result<(), String> {
            self.stop_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn get_distance(&self) -> Result<f32, String> {
            Ok(self.mock_distance)
        }
        fn get_grayscale(&self) -> Result<(i32, i32, i32), String> {
            Ok((100, 100, 100))
        }
        fn speak(&self, _text: &str) -> Result<(), String> {
            Ok(())
        }
        fn listen(&self) -> Result<String, String> {
            Ok("テスト音声認識".to_string())
        }
        fn take_snapshot(&self, _width: u32, _height: u32) -> Result<String, String> {
            Ok("dGVzdA==".to_string()) // "test" base64
        }
    }

    struct TestApp {
        router: Router,
        drive_calls: Arc<AtomicUsize>,
        stop_calls: Arc<AtomicUsize>,
    }

    impl TestApp {
        fn new(distance: f32) -> Self {
            let drive_calls = Arc::new(AtomicUsize::new(0));
            let stop_calls = Arc::new(AtomicUsize::new(0));
            let state = Arc::new(AppState {
                hardware: Box::new(MockHardware {
                    mock_distance: distance,
                    drive_calls: Arc::clone(&drive_calls),
                    stop_calls: Arc::clone(&stop_calls),
                }),
                safety: Arc::new(SafetyGuard::new(15.0, 3)),
                api_token: TOKEN.to_string(),
                motion_gate: Mutex::new(()),
            });
            Self {
                router: create_router(state),
                drive_calls,
                stop_calls,
            }
        }

        async fn send(&self, method: &str, uri: &str, body: Option<&str>) -> StatusCode {
            let request = Request::builder()
                .method(method)
                .uri(uri)
                .header("authorization", format!("Bearer {TOKEN}"))
                .header("content-type", "application/json")
                .body(body.map_or_else(Body::empty, |b| Body::from(b.to_string())))
                .unwrap();
            self.router.clone().oneshot(request).await.unwrap().status()
        }

        async fn send_move(&self, body: &str) -> StatusCode {
            self.send("POST", "/api/drive/move", Some(body)).await
        }
    }

    #[tokio::test]
    async fn test_drive_move_safe() {
        let app = TestApp::new(30.0); // 障害物まで30cm (安全)
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_drive_move_unsafe_obstacle() {
        let app = TestApp::new(10.0); // 障害物まで10cm (危険)
        assert_eq!(app.send_move(FORWARD).await, StatusCode::BAD_REQUEST);
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_sensor_ultrasonic() {
        let app = TestApp::new(25.5);
        assert_eq!(
            app.send("GET", "/api/sensor/ultrasonic", None).await,
            StatusCode::OK
        );
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
        assert_eq!(app.stop_calls.load(Ordering::SeqCst), 1);

        assert_eq!(app.send_move(FORWARD).await, StatusCode::CONFLICT);
        assert_eq!(app.send_move(BACKWARD).await, StatusCode::CONFLICT);
        // 拒否された要求の後も保持され続ける
        assert_eq!(app.send_move(FORWARD).await, StatusCode::CONFLICT);
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 0);

        assert_eq!(
            app.send("POST", "/api/drive/emergency_stop/reset", None)
                .await,
            StatusCode::OK
        );
        assert_eq!(app.send_move(FORWARD).await, StatusCode::OK);
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn plain_stop_does_not_block_following_moves() {
        let app = TestApp::new(30.0);
        assert_eq!(
            app.send("POST", "/api/drive/stop", None).await,
            StatusCode::OK
        );
        assert_eq!(app.stop_calls.load(Ordering::SeqCst), 1);
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
            r#"{"direction":"forward","speed":30,"duration_ms":0}"#,
            r#"{"direction":"forward","speed":30,"duration_ms":49}"#,
            r#"{"direction":"forward","speed":30,"duration_ms":5001}"#,
            r#"{"direction":"backward","speed":101,"duration_ms":500}"#,
            r#"{"direction":"backward","speed":-1,"duration_ms":500}"#,
        ] {
            assert_eq!(app.send_move(body).await, StatusCode::BAD_REQUEST, "{body}");
        }
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn boundary_move_values_are_accepted() {
        let app = TestApp::new(30.0);
        for body in [
            r#"{"direction":"forward","speed":100,"duration_ms":50}"#,
            r#"{"direction":"forward","speed":0,"duration_ms":5000}"#,
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
        assert_eq!(app.drive_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn token_comparison_rejects_empty_expected_token() {
        assert!(!crate::api::handlers::token_matches("", ""));
        assert!(crate::api::handlers::token_matches("abc", "abc"));
        assert!(!crate::api::handlers::token_matches("abc", "abd"));
    }
}
