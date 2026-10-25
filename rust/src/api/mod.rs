pub mod handlers;
pub mod hardware;
pub mod models;

use crate::api::handlers::{
    AppState, RobotHardware, grayscale_handler, listen_handler, move_handler, snapshot_handler,
    speak_handler, steer_handler, stop_handler, ultrasonic_handler,
};
use crate::api::models::DriveDirection;
use crate::safety::SafetyGuard;
use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub fn create_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/api/drive/move", post(move_handler))
        .route("/api/drive/steer", post(steer_handler))
        .route("/api/drive/stop", post(stop_handler))
        .route("/api/sensor/ultrasonic", get(ultrasonic_handler))
        .route("/api/sensor/grayscale", get(grayscale_handler))
        .route("/api/audio/speak", post(speak_handler))
        .route("/api/audio/listen", get(listen_handler))
        .route("/api/camera/snapshot", get(snapshot_handler))
        .layer(CorsLayer::permissive())
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
    let hardware: Box<dyn RobotHardware> = match hardware::PiCarHardware::new() {
        Ok(hw) => {
            println!("✓ PiCar-X 実機ハードウェアを初期化しました");
            Box::new(hw)
        }
        Err(e) => {
            eprintln!("⚠ 実機ハードウェア初期化失敗 (モックモードで起動): {}", e);
            Box::new(MockFallbackHardware)
        }
    };

    let safety = Arc::new(SafetyGuard::new(15.0, 3));
    let state = Arc::new(AppState { hardware, safety });
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
    use crate::api::handlers::RobotHardware;
    use crate::api::models::DriveDirection;
    use crate::safety::SafetyGuard;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::util::ServiceExt; // for oneshot

    struct MockHardware {
        pub mock_distance: f32,
    }

    impl RobotHardware for MockHardware {
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

    fn create_test_app(distance: f32) -> Router {
        let state = Arc::new(AppState {
            hardware: Box::new(MockHardware {
                mock_distance: distance,
            }),
            safety: Arc::new(SafetyGuard::new(15.0, 3)),
        });
        create_router(state)
    }

    #[tokio::test]
    async fn test_drive_move_safe() {
        let app = create_test_app(30.0); // 障害物まで30cm (安全)
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/drive/move")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"direction":"forward","speed":30,"duration_ms":500}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_drive_move_unsafe_obstacle() {
        let app = create_test_app(10.0); // 障害物まで10cm (危険)
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/drive/move")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"direction":"forward","speed":30,"duration_ms":500}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_sensor_ultrasonic() {
        let app = create_test_app(25.5);
        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/sensor/ultrasonic")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_audio_speak_and_listen() {
        let app = create_test_app(50.0);

        // Speak
        let resp_speak = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/audio/speak")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"text":"こんにちは"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_speak.status(), StatusCode::OK);

        // Listen
        let resp_listen = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/audio/listen")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp_listen.status(), StatusCode::OK);
    }
}
