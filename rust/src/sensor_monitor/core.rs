use crate::picarx::Picarx;
use crate::picarx::ultrasonic::Ultrasonic;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// 超音波は1回の測定が最大で約20ms。残響の混信を避けるため少し間隔を空ける。
const ULTRASONIC_PERIOD: Duration = Duration::from_millis(40);
// I2C読み取りは1周あたり数ms。反射ループ(25ms周期)より短い間隔で更新する。
const BUS_PERIOD: Duration = Duration::from_millis(20);
const BATTERY_PERIOD: Duration = Duration::from_secs(1);

// 超音波センサーはモーター動作中の電気的ノイズ等で単発の外れ値(スパイク)が
// 出やすいため、直近の測定の中央値を採用する。古い測定は混ぜない。
const DISTANCE_WINDOW: usize = 3;
const DISTANCE_WINDOW_MAX_AGE: Duration = Duration::from_millis(200);

/// 起動直後の床センサーで観測された、測定値ではない連番のデータ（A0 から順に 0A 0B 0C 0D 0E 0F）。
/// 原因は未特定。高い値なので、段差を見逃さないよう捨てて「読めなかった」として扱う。
const INVALID_FLOOR_READING: [u16; 3] = [0x0E0F, 0x0C0D, 0x0A0B];

/// 値と、その値を取得した時刻の組
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Timed<T> {
    pub value: T,
    pub at: Instant,
}

impl<T: Copy> Timed<T> {
    pub fn new(value: T, at: Instant) -> Self {
        Self { value, at }
    }

    #[cfg(test)]
    pub fn now(value: T) -> Self {
        Self::new(value, Instant::now())
    }

    pub fn age(&self, now: Instant) -> Duration {
        now.saturating_duration_since(self.at)
    }

    /// `max_age` 以内に取得した値だけを返す。古い値は「不明」として扱う。
    pub fn fresh(&self, now: Instant, max_age: Duration) -> Option<T> {
        (self.age(now) <= max_age).then_some(self.value)
    }
}

/// 各センサーの最新値。一度も読めていない項目は `None`。
///
/// 読み取りに失敗しても値は残るが時刻は更新されない。安全判断では必ず
/// `Timed::fresh` を通すこと。
#[derive(Debug, Clone, Copy, Default)]
pub struct SensorSnapshot {
    pub distance_cm: Option<Timed<f64>>,
    pub grayscale: Option<Timed<[u16; 3]>>,
    pub battery: Option<Timed<(f64, i32)>>,
}

/// 超音波とI2Cを別スレッドで読み取り、`snapshot()` で最新値を渡す。
pub struct SensorMonitor {
    snapshot: Arc<Mutex<SensorSnapshot>>,
    shutdown: Arc<AtomicBool>,
    handles: Vec<JoinHandle<()>>,
}

fn lock_ignoring_poison<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl SensorMonitor {
    pub fn spawn(px: Arc<Mutex<Picarx>>) -> Self {
        let snapshot = Arc::new(Mutex::new(SensorSnapshot::default()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let ultrasonic = lock_ignoring_poison(&px).ultrasonic_handle();

        let handles = vec![
            Self::spawn_ultrasonic(ultrasonic, Arc::clone(&snapshot), Arc::clone(&shutdown)),
            Self::spawn_bus(px, Arc::clone(&snapshot), Arc::clone(&shutdown)),
        ];

        Self {
            snapshot,
            shutdown,
            handles,
        }
    }

    /// 距離の測定中は `Picarx` のロックを持たない。モーター指令を待たせないため。
    fn spawn_ultrasonic(
        ultrasonic: Arc<Mutex<Ultrasonic>>,
        snapshot: Arc<Mutex<SensorSnapshot>>,
        shutdown: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut filter = DistanceFilter::default();
            while !shutdown.load(Ordering::Relaxed) {
                let measured = lock_ignoring_poison(&ultrasonic).try_read_distance_once();
                let now = Instant::now();
                if let Some(distance) = filter.update(measured, now) {
                    lock_ignoring_poison(&snapshot).distance_cm = Some(Timed::new(distance, now));
                }
                thread::sleep(ULTRASONIC_PERIOD);
            }
        })
    }

    fn spawn_bus(
        px: Arc<Mutex<Picarx>>,
        snapshot: Arc<Mutex<SensorSnapshot>>,
        shutdown: Arc<AtomicBool>,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            let mut last_battery: Option<Instant> = None;
            while !shutdown.load(Ordering::Relaxed) {
                let read_battery = last_battery.is_none_or(|at| at.elapsed() >= BATTERY_PERIOD);
                let (grayscale, battery) = {
                    let mut px = lock_ignoring_poison(&px);
                    let grayscale = px
                        .get_grayscale_data()
                        .ok()
                        .filter(|values| is_valid_floor_reading(*values));
                    let battery = if read_battery {
                        px.get_battery_status().ok()
                    } else {
                        None
                    };
                    (grayscale, battery)
                };

                let now = Instant::now();
                if read_battery {
                    last_battery = Some(now);
                }
                let mut snap = lock_ignoring_poison(&snapshot);
                if let Some(g) = grayscale {
                    snap.grayscale = Some(Timed::new(g, now));
                }
                if let Some(b) = battery {
                    snap.battery = Some(Timed::new(b, now));
                }
                drop(snap);

                thread::sleep(BUS_PERIOD);
            }
        })
    }

    /// 最新のセンサー値を取得する(軽量・非ブロッキング)
    pub fn snapshot(&self) -> SensorSnapshot {
        *lock_ignoring_poison(&self.snapshot)
    }
}

fn is_valid_floor_reading(values: [u16; 3]) -> bool {
    values != INVALID_FLOOR_READING
}

/// 直近の測定から距離を決める。有効な測定がなければ `None` を返す。
#[derive(Debug, Default)]
struct DistanceFilter {
    samples: VecDeque<(Instant, f64)>,
}

impl DistanceFilter {
    fn update(&mut self, measured: Option<f64>, now: Instant) -> Option<f64> {
        if let Some(distance) = measured {
            self.samples.push_back((now, distance));
            if self.samples.len() > DISTANCE_WINDOW {
                self.samples.pop_front();
            }
        }
        self.samples
            .retain(|(at, _)| now.saturating_duration_since(*at) <= DISTANCE_WINDOW_MAX_AGE);
        // 今回の測定に失敗した時は時刻を更新しない。古い値が新しく見えるのを防ぐ。
        measured?;
        let mut values: Vec<f64> = self.samples.iter().map(|(_, d)| *d).collect();
        median_distance(&mut values)
    }
}

/// 偶数個の時は小さい方を選ぶ。安全側(より近い距離)に倒すため。
fn median_distance(samples: &mut [f64]) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_by(f64::total_cmp);
    Some(samples[(samples.len() - 1) / 2])
}

impl Drop for SensorMonitor {
    fn drop(&mut self) {
        // ポーリングスレッドを止めてjoinすることで、Arc<Mutex<Picarx>>の最後の
        // 参照がここで確実に解放され、Picarxのdrop(モーター停止・サーボセンタリング等)
        // が呼び出し元の終了時に確実に実行されることを保証する。
        self.shutdown.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DISTANCE_WINDOW_MAX_AGE, DistanceFilter, Timed, is_valid_floor_reading, median_distance,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn floor_reading_rejects_the_startup_byte_pattern() {
        // 実機の起動時の校正で記録された値
        assert!(!is_valid_floor_reading([3599, 3085, 2571]));
        assert!(is_valid_floor_reading([1176, 1158, 1212]));
    }

    #[test]
    fn median_distance_rejects_empty_samples() {
        assert_eq!(median_distance(&mut []), None);
    }

    #[test]
    fn median_distance_sorts_samples_before_selecting_middle() {
        assert_eq!(median_distance(&mut [120.0, 10.0, 30.0]), Some(30.0));
    }

    #[test]
    fn median_distance_prefers_the_nearer_value_when_even() {
        assert_eq!(median_distance(&mut [40.0, 10.0, 30.0, 20.0]), Some(20.0));
    }

    #[test]
    fn distance_filter_ignores_a_single_spike() {
        let t0 = Instant::now();
        let mut filter = DistanceFilter::default();
        filter.update(Some(50.0), t0);
        filter.update(Some(51.0), t0 + Duration::from_millis(40));
        let result = filter.update(Some(300.0), t0 + Duration::from_millis(80));
        assert_eq!(result, Some(51.0));
    }

    #[test]
    fn distance_filter_reports_nothing_when_the_measurement_fails() {
        let t0 = Instant::now();
        let mut filter = DistanceFilter::default();
        filter.update(Some(50.0), t0);
        assert_eq!(filter.update(None, t0 + Duration::from_millis(40)), None);
    }

    #[test]
    fn distance_filter_does_not_mix_in_old_samples() {
        let t0 = Instant::now();
        let mut filter = DistanceFilter::default();
        filter.update(Some(10.0), t0);
        filter.update(Some(10.0), t0 + Duration::from_millis(10));
        let later = t0 + DISTANCE_WINDOW_MAX_AGE + Duration::from_millis(50);
        assert_eq!(filter.update(Some(80.0), later), Some(80.0));
    }

    #[test]
    fn timed_value_expires_after_max_age() {
        let t0 = Instant::now();
        let reading = Timed::new(1.0, t0);
        let max_age = Duration::from_millis(100);
        assert_eq!(reading.fresh(t0 + max_age, max_age), Some(1.0));
        assert_eq!(reading.fresh(t0 + max_age * 2, max_age), None);
    }
}
