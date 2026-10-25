use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// 安全性監視リミッター
pub struct SafetyGuard {
    last_command_time: Arc<Mutex<Instant>>,
    is_emergency_stopped: AtomicBool,
    min_safe_distance_cm: f32,
    timeout_duration: Duration,
}

impl SafetyGuard {
    pub fn new(min_safe_distance_cm: f32, timeout_secs: u64) -> Self {
        Self {
            last_command_time: Arc::new(Mutex::new(Instant::now())),
            is_emergency_stopped: AtomicBool::new(false),
            min_safe_distance_cm,
            timeout_duration: Duration::from_secs(timeout_secs),
        }
    }

    /// コマンド受信時のタイムスタンプを更新
    pub async fn record_command(&self) {
        let mut t = self.last_command_time.lock().await;
        *t = Instant::now();
        self.is_emergency_stopped.store(false, Ordering::SeqCst);
    }

    /// 前進が可能かどうか（障害物距離および緊急停止フラグ）を判定
    pub fn can_move_forward(&self, current_distance_cm: f32) -> bool {
        if self.is_emergency_stopped.load(Ordering::SeqCst) {
            return false;
        }
        current_distance_cm >= self.min_safe_distance_cm
    }

    /// 緊急停止を発動
    pub fn trigger_emergency_stop(&self) {
        self.is_emergency_stopped.store(true, Ordering::SeqCst);
    }

    /// 通信タイムアウトが発生しているか判定
    pub async fn is_timed_out(&self) -> bool {
        let t = self.last_command_time.lock().await;
        t.elapsed() > self.timeout_duration
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safety_guard_obstacle() {
        let guard = SafetyGuard::new(15.0, 3);
        // 20cm離れていれば前進可能
        assert!(guard.can_move_forward(20.0));
        // 10cmは近すぎるため拒否
        assert!(!guard.can_move_forward(10.0));

        // 緊急停止時
        guard.trigger_emergency_stop();
        assert!(!guard.can_move_forward(20.0));
    }
}
