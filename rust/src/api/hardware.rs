use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::api::handlers::RobotHardware;
use crate::api::models::DriveDirection;
use crate::picarx::Picarx;
use crate::sensor_monitor::{SensorMonitor, SensorSnapshot};
use crate::tts::{detect_language, generate_tts_file};

/// 実機 PiCar-X ハードウェア制御実装
pub struct PiCarHardware {
    px: Arc<Mutex<Picarx>>,
    sensor_monitor: SensorMonitor,
    /// 発話を止めるたびに増やす。生成中だった音声を再生しないために使う
    speech_epoch: AtomicU64,
}

impl PiCarHardware {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let px = Picarx::new()?;
        let px = Arc::new(Mutex::new(px));
        let sensor_monitor = SensorMonitor::spawn(Arc::clone(&px));

        Ok(Self {
            px,
            sensor_monitor,
            speech_epoch: AtomicU64::new(0),
        })
    }

    fn car(&self) -> std::sync::MutexGuard<'_, Picarx> {
        // 停止指令を確実に通すため、別スレッドのパニックでも操作を続ける
        self.px
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl RobotHardware for PiCarHardware {
    fn drive(&self, direction: DriveDirection, speed: i32) -> Result<(), String> {
        let mut car = self.car();
        let result = match direction {
            DriveDirection::Forward => car.try_drive(speed),
            DriveDirection::Backward => car.try_drive(-speed),
            DriveDirection::Stop => car.try_stop(),
        };
        result.map_err(|error| format!("モーター指令に失敗しました: {error}"))
    }

    fn steer(&self, angle: i32) -> Result<(), String> {
        self.car().set_dir_servo_angle(angle);
        Ok(())
    }

    fn stop(&self) -> Result<(), String> {
        self.car()
            .try_stop()
            .map_err(|error| format!("モーター停止に失敗しました: {error}"))
    }

    fn sensors(&self) -> SensorSnapshot {
        self.sensor_monitor.snapshot()
    }

    fn speak(&self, text: &str) -> Result<(), String> {
        let epoch = self.speech_epoch.load(Ordering::SeqCst);
        let lang = detect_language(text);
        let wav_path =
            generate_tts_file(text, lang).map_err(|e| format!("TTS生成エラー: {}", e))?;

        // 生成中に発話の停止が来ていたら再生しない。停止と同じロックの中で確かめる
        let result = {
            let mut car = self.car();
            if self.speech_epoch.load(Ordering::SeqCst) == epoch {
                car.play_file(&wav_path, 1.0)
                    .map_err(|error| format!("音声ファイル再生エラー: {error}"))
            } else {
                Ok(())
            }
        };
        // 再生はファイルを開いた後なので、ここで消しても続く
        if let Err(error) = std::fs::remove_file(&wav_path) {
            eprintln!("音声一時ファイル削除エラー: {error}");
        }
        result
    }

    fn stop_speech(&self) -> Result<(), String> {
        self.speech_epoch.fetch_add(1, Ordering::SeqCst);
        self.car().stop_audio();
        Ok(())
    }

    fn is_speaking(&self) -> bool {
        self.car().is_playing_audio()
    }

    fn listen(&self) -> Result<String, String> {
        // 音声認識結果（将来のVosk/Whisper統合用、現在は受信待機）
        Ok("音声認識スタンバイ".to_string())
    }
}
