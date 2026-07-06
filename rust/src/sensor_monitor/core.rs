use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use crate::picarx::Picarx;

// センサーのポーリング間隔。距離計測は1回あたり最大でも数十ms程度に収まるよう
// try_get_distance_once()を使うため、この間隔なら手動運転側のロック待ちも問題にならない。
const POLL_INTERVAL: Duration = Duration::from_millis(300);

// 超音波センサーはモーター動作中の電気的ノイズ等で単発の外れ値(スパイク)が
// 出やすいため、複数回サンプリングして中央値を採用することでノイズを除去する。
const DISTANCE_SAMPLE_COUNT: usize = 3;
const DISTANCE_SAMPLE_GAP_MS: u64 = 10;

#[derive(Debug, Clone, Copy, Default)]
pub struct SensorSnapshot {
    pub distance_cm: Option<f64>,
    pub grayscale: Option<[u16; 3]>,
    pub battery: Option<(f64, i32)>,
}

/// `Picarx`を共有するバックグラウンドスレッドで距離・グレイスケール・バッテリーを
/// 定期的に読み取り、GUIスレッドからは`snapshot()`で最新値を取得するだけにする。
pub struct SensorMonitor {
    snapshot: Arc<Mutex<SensorSnapshot>>,
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl SensorMonitor {
    pub fn spawn(px: Arc<Mutex<Picarx>>) -> Self {
        let snapshot = Arc::new(Mutex::new(SensorSnapshot::default()));
        let shutdown = Arc::new(AtomicBool::new(false));

        let snapshot_clone = Arc::clone(&snapshot);
        let shutdown_clone = Arc::clone(&shutdown);

        let handle = thread::spawn(move || {
            while !shutdown_clone.load(Ordering::Relaxed) {
                {
                    let mut px = px.lock().unwrap();
                    let distance = Self::read_distance_median(&mut px);
                    let grayscale = px.get_grayscale_data().ok();
                    let battery = px.get_battery_status().ok();
                    drop(px);

                    // 読み取りに失敗した項目は直前の値を保持する(表示のちらつき防止)
                    let mut snap = snapshot_clone.lock().unwrap();
                    if let Some(d) = distance { snap.distance_cm = Some(d); }
                    if let Some(g) = grayscale { snap.grayscale = Some(g); }
                    if let Some(b) = battery { snap.battery = Some(b); }
                }
                thread::sleep(POLL_INTERVAL);
            }
        });

        Self { snapshot, shutdown, handle: Some(handle) }
    }

    /// 複数回計測して中央値を採用することで、モーター動作時の電気的ノイズ等による
    /// 単発の外れ値(実際より大きく外れた距離)の影響を軽減する。
    fn read_distance_median(px: &mut Picarx) -> Option<f64> {
        let mut samples = Vec::with_capacity(DISTANCE_SAMPLE_COUNT);
        for i in 0..DISTANCE_SAMPLE_COUNT {
            if let Some(d) = px.try_get_distance_once() {
                samples.push(d);
            }
            if i + 1 < DISTANCE_SAMPLE_COUNT {
                thread::sleep(Duration::from_millis(DISTANCE_SAMPLE_GAP_MS));
            }
        }
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(samples[samples.len() / 2])
    }

    /// 最新のセンサー値を取得する(軽量・非ブロッキング)
    pub fn snapshot(&self) -> SensorSnapshot {
        *self.snapshot.lock().unwrap()
    }
}

impl Drop for SensorMonitor {
    fn drop(&mut self) {
        // ポーリングスレッドを止めてjoinすることで、Arc<Mutex<Picarx>>の最後の
        // 参照がここで確実に解放され、Picarxのdrop(モーター停止・サーボセンタリング等)
        // が呼び出し元の終了時に確実に実行されることを保証する。
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}
