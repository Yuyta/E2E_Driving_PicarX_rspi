use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::api::handlers::RobotHardware;
use crate::api::models::DriveDirection;
use crate::picarx::Picarx;
use crate::sensor_monitor::SensorMonitor;
use crate::tts::{TtsEngine, detect_language, generate_tts_file};

/// 実機 PiCar-X ハードウェア制御実装
pub struct PiCarHardware {
    px: Arc<Mutex<Picarx>>,
    sensor_monitor: SensorMonitor,
}

impl PiCarHardware {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let px = Picarx::new()?;
        let px = Arc::new(Mutex::new(px));
        let sensor_monitor = SensorMonitor::spawn(Arc::clone(&px));

        Ok(Self { px, sensor_monitor })
    }
}

impl RobotHardware for PiCarHardware {
    fn drive(&self, direction: DriveDirection, speed: i32, duration_ms: u64) -> Result<(), String> {
        let px = Arc::clone(&self.px);
        thread::spawn(move || {
            let mut car = match px.lock() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("Failed to lock car: {}", e);
                    return;
                }
            };
            match direction {
                DriveDirection::Forward => car.forward(speed),
                DriveDirection::Backward => car.backward(speed),
                DriveDirection::Stop => car.stop(),
            }
            drop(car);

            if duration_ms > 0 && direction != DriveDirection::Stop {
                thread::sleep(Duration::from_millis(duration_ms));
                if let Ok(mut car) = px.lock() {
                    car.stop();
                }
            }
        });

        Ok(())
    }

    fn steer(&self, angle: i32) -> Result<(), String> {
        let mut car = self.px.lock().map_err(|e| e.to_string())?;
        car.set_dir_servo_angle(angle);
        Ok(())
    }

    fn stop(&self) -> Result<(), String> {
        let mut car = self.px.lock().map_err(|e| e.to_string())?;
        car.stop();
        Ok(())
    }

    fn get_distance(&self) -> Result<f32, String> {
        let snap = self.sensor_monitor.snapshot();
        Ok(snap.distance_cm.unwrap_or(0.0) as f32)
    }

    fn get_grayscale(&self) -> Result<(i32, i32, i32), String> {
        let snap = self.sensor_monitor.snapshot();
        match snap.grayscale {
            Some([l, m, r]) => Ok((l as i32, m as i32, r as i32)),
            None => Ok((0, 0, 0)),
        }
    }

    fn speak(&self, text: &str) -> Result<(), String> {
        let lang = detect_language(text);
        let wav_path = generate_tts_file(text, lang, TtsEngine::Default)
            .map_err(|e| format!("TTS生成エラー: {}", e))?;

        // 旧Rust GUIと同じPiCar-X固有の音声出力経路を使用する。
        let mut car = self.px.lock().map_err(|error| error.to_string())?;
        car.play_file(&wav_path, 1.0)
            .map_err(|error| format!("音声ファイル再生エラー: {error}"))?;
        drop(car);
        if let Err(error) = std::fs::remove_file(&wav_path) {
            eprintln!("音声一時ファイル削除エラー: {error}");
        }

        Ok(())
    }

    fn listen(&self) -> Result<String, String> {
        // 音声認識結果（将来のVosk/Whisper統合用、現在は受信待機）
        Ok("音声認識スタンバイ".to_string())
    }

    fn take_snapshot(&self, _width: u32, _height: u32) -> Result<String, String> {
        let args = [
            "--nopreview",
            "-t",
            "100",
            "--width",
            "320",
            "--height",
            "240",
            "--quality",
            "70",
            "--encoding",
            "jpg",
            "-o",
            "-",
        ];
        let output = std::process::Command::new("rpicam-still")
            .args(args)
            .output()
            .or_else(|_| {
                std::process::Command::new("libcamera-still")
                    .args(args)
                    .output()
            })
            .map_err(|error| format!("カメラコマンドを起動できません: {error}"))?;

        if !output.status.success() || output.stdout.is_empty() {
            let detail = String::from_utf8_lossy(&output.stderr);
            return Err(format!("カメラ撮影に失敗しました: {detail}"));
        }

        use base64::Engine as _;
        Ok(base64::engine::general_purpose::STANDARD.encode(output.stdout))
    }
}
