//! 走行中の危険を監視して止める反射ループ。PC との通信に依存しない。

use crate::api::handlers::AppState;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

/// 目標は 20Hz 以上。センサー更新と合わせても検知から停止まで 100ms に収まる周期にする。
const TICK_PERIOD: Duration = Duration::from_millis(25);

/// 1周分の処理。モーターの停止までを `safety` のロックの中で行い、
/// 判定と停止の間に新しい走行が割り込まないようにする。
pub fn reflex_tick(state: &AppState) {
    let sensors = state.hardware.sensors();
    let now = Instant::now();
    let mut safety = state.safety();

    if let Some(hazard) = safety.evaluate(&sensors, now) {
        safety.clear_motion();
        let result = state.hardware.stop();
        let stopped_at = Instant::now();
        safety.note_stop_result(&result);
        safety.record_hazard_stop(&hazard, stopped_at);
    } else if safety.stop_pending() {
        let result = state.hardware.stop();
        safety.note_stop_result(&result);
    }
}

/// スレッドが異常終了する時に、モーターを止める
struct StopOnPanic(Arc<AppState>);

impl Drop for StopOnPanic {
    fn drop(&mut self) {
        if thread::panicking() {
            eprintln!("反射ループが異常終了しました。モーターを停止します");
            self.0.safety().clear_motion();
            let _ = self.0.hardware.stop();
        }
    }
}

pub fn spawn_reflex_loop(state: Arc<AppState>) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("reflex".to_string())
        .spawn(move || {
            let guard = StopOnPanic(state);
            loop {
                reflex_tick(&guard.0);
                thread::sleep(TICK_PERIOD);
            }
        })
}
