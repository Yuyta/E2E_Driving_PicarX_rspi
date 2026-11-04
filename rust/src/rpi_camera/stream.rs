//! `rpicam-vid` の MJPEG 出力を読み、最新フレームを配信する。
//!
//! カメラは要求がある間だけ動かす。室内を不要に撮影せず、電池も節約するため。

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub const FRAME_WIDTH: u32 = 320;
pub const FRAME_HEIGHT: u32 = 240;
const FRAMERATE: u32 = 15;
const JPEG_QUALITY: u32 = 70;

/// 最後の要求からこの時間が過ぎたらカメラを止める
const IDLE_SHUTDOWN: Duration = Duration::from_secs(10);
/// フレームがこの時間届かなければカメラを再起動する
const STALL_RESTART: Duration = Duration::from_secs(5);
const RESTART_BACKOFF: Duration = Duration::from_secs(1);
const SUPERVISOR_POLL: Duration = Duration::from_millis(50);
const MAX_JPEG_BYTES: usize = 2 * 1024 * 1024;

/// 新しい OS の名前を優先し、なければ旧名を使う
const CAMERA_PROGRAMS: [&str; 2] = ["rpicam-vid", "libcamera-vid"];

const SOI: [u8; 2] = [0xFF, 0xD8];
const EOI: [u8; 2] = [0xFF, 0xD9];

#[derive(Debug)]
pub struct Frame {
    pub seq: u64,
    pub jpeg: Vec<u8>,
    pub captured_at: Instant,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraStats {
    pub running: bool,
    pub frames: u64,
    pub fps: f64,
    pub frame_age: Option<Duration>,
}

#[derive(Debug, Default)]
struct RateTracker {
    last_at: Option<Instant>,
    interval_ema: f64,
}

/// 最新フレームの受け渡し口。配信先が遅くても古いフレームは溜まらない。
pub struct CameraHub {
    tx: watch::Sender<Option<Arc<Frame>>>,
    last_request: Mutex<Option<Instant>>,
    rate: Mutex<RateTracker>,
    running: AtomicBool,
    last_error: Mutex<Option<String>>,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

impl Default for CameraHub {
    fn default() -> Self {
        Self::new()
    }
}

impl CameraHub {
    pub fn new() -> Self {
        Self {
            tx: watch::Sender::new(None),
            last_request: Mutex::new(None),
            rate: Mutex::new(RateTracker::default()),
            running: AtomicBool::new(false),
            last_error: Mutex::new(None),
        }
    }

    /// 配信の受信者を増やす。受信者がいる間はカメラが動き続ける。
    pub fn subscribe(&self) -> watch::Receiver<Option<Arc<Frame>>> {
        self.tx.subscribe()
    }

    /// 要求を記録する。受信者がいなくなっても一定時間はカメラを動かし続ける。
    pub fn note_request(&self) {
        *lock(&self.last_request) = Some(Instant::now());
    }

    fn wants_frames(&self, now: Instant) -> bool {
        self.tx.receiver_count() > 0
            || lock(&self.last_request)
                .is_some_and(|at| now.saturating_duration_since(at) <= IDLE_SHUTDOWN)
    }

    pub fn publish(&self, jpeg: Vec<u8>) {
        let now = Instant::now();
        {
            let mut rate = lock(&self.rate);
            if let Some(last) = rate.last_at {
                let interval = now.saturating_duration_since(last).as_secs_f64();
                rate.interval_ema = if rate.interval_ema == 0.0 {
                    interval
                } else {
                    rate.interval_ema * 0.9 + interval * 0.1
                };
            }
            rate.last_at = Some(now);
        }
        self.tx.send_modify(|slot| {
            let seq = slot.as_ref().map_or(1, |frame| frame.seq + 1);
            *slot = Some(Arc::new(Frame {
                seq,
                jpeg,
                captured_at: now,
            }));
        });
    }

    fn last_frame_at(&self) -> Option<Instant> {
        self.tx.borrow().as_ref().map(|frame| frame.captured_at)
    }

    pub fn stats(&self, now: Instant) -> CameraStats {
        let (frames, frame_age) = {
            let frame = self.tx.borrow();
            frame.as_ref().map_or((0, None), |frame| {
                (
                    frame.seq,
                    Some(now.saturating_duration_since(frame.captured_at)),
                )
            })
        };
        let running = self.running.load(Ordering::Relaxed);
        let interval = lock(&self.rate).interval_ema;
        CameraStats {
            running,
            frames,
            fps: if running && interval > 0.0 {
                1.0 / interval
            } else {
                0.0
            },
            frame_age,
        }
    }

    pub fn last_error(&self) -> Option<String> {
        lock(&self.last_error).clone()
    }

    fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
        // 再起動の間隔を次回の fps に混ぜない
        *lock(&self.rate) = RateTracker::default();
    }

    fn set_error(&self, error: Option<String>) {
        *lock(&self.last_error) = error;
    }
}

/// バイト列から JPEG を1枚ずつ切り出す。
/// JPEG の圧縮データ内で 0xFF はエスケープされるので、EOI は画像の終端にしか現れない。
#[derive(Debug, Default)]
pub struct MjpegSplitter {
    buffer: Vec<u8>,
}

fn find_marker(data: &[u8], marker: &[u8; 2]) -> Option<usize> {
    data.windows(2).position(|window| window == marker)
}

impl MjpegSplitter {
    pub fn push(&mut self, data: &[u8]) -> Vec<Vec<u8>> {
        self.buffer.extend_from_slice(data);
        let mut frames = Vec::new();
        loop {
            let Some(start) = find_marker(&self.buffer, &SOI) else {
                // 末尾の 0xFF は次の読み込みで SOI になる可能性がある
                let keep = usize::from(self.buffer.last() == Some(&SOI[0]));
                self.buffer.drain(..self.buffer.len() - keep);
                break;
            };
            self.buffer.drain(..start);

            let Some(end) = find_marker(&self.buffer[SOI.len()..], &EOI) else {
                if self.buffer.len() > MAX_JPEG_BYTES {
                    self.buffer.clear();
                }
                break;
            };
            let frame_len = SOI.len() + end + EOI.len();
            frames.push(self.buffer.drain(..frame_len).collect());
        }
        frames
    }
}

fn spawn_camera_process(program: &str, fallback: &str) -> std::io::Result<Child> {
    let args = [
        "--nopreview".to_string(),
        "-t".to_string(),
        "0".to_string(),
        "--codec".to_string(),
        "mjpeg".to_string(),
        "--width".to_string(),
        FRAME_WIDTH.to_string(),
        "--height".to_string(),
        FRAME_HEIGHT.to_string(),
        "--framerate".to_string(),
        FRAMERATE.to_string(),
        "--quality".to_string(),
        JPEG_QUALITY.to_string(),
        "--flush".to_string(),
        "-o".to_string(),
        "-".to_string(),
    ];
    let spawn = |program: &str| {
        Command::new(program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
    };
    spawn(program).or_else(|_| spawn(fallback))
}

fn read_frames(mut stdout: impl Read, hub: &CameraHub) {
    let mut splitter = MjpegSplitter::default();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        match stdout.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                for jpeg in splitter.push(&chunk[..n]) {
                    hub.publish(jpeg);
                }
            }
        }
    }
}

/// カメラプロセスを1回分動かす。正常に止めた時は `true` を返す。
fn run_camera_once(hub: &Arc<CameraHub>, programs: [&str; 2]) -> bool {
    let mut child = match spawn_camera_process(programs[0], programs[1]) {
        Ok(child) => child,
        Err(error) => {
            hub.set_error(Some(format!("カメラコマンドを起動できません: {error}")));
            return false;
        }
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return false;
    };

    let reader_hub = Arc::clone(hub);
    let reader = thread::spawn(move || read_frames(stdout, &reader_hub));
    hub.set_running(true);
    let started = Instant::now();
    let mut stopped_on_purpose = false;

    loop {
        let now = Instant::now();
        if !hub.wants_frames(now) {
            stopped_on_purpose = true;
            break;
        }
        match child.try_wait() {
            Ok(None) => {}
            Ok(Some(status)) => {
                hub.set_error(Some(format!("カメラが終了しました: {status}")));
                break;
            }
            Err(error) => {
                hub.set_error(Some(format!("カメラの状態を確認できません: {error}")));
                break;
            }
        }
        let last_activity = hub.last_frame_at().map_or(started, |at| at.max(started));
        if now.saturating_duration_since(last_activity) > STALL_RESTART {
            hub.set_error(Some("カメラからフレームが届きません".to_string()));
            break;
        }
        if last_activity > started {
            hub.set_error(None);
        }
        thread::sleep(SUPERVISOR_POLL);
    }

    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    hub.set_running(false);
    stopped_on_purpose
}

/// 要求に応じてカメラを起動、停止、再起動する
pub fn spawn_camera_supervisor(hub: Arc<CameraHub>) -> std::io::Result<thread::JoinHandle<()>> {
    thread::Builder::new()
        .name("camera".to_string())
        .spawn(move || {
            loop {
                if !hub.wants_frames(Instant::now()) {
                    thread::sleep(SUPERVISOR_POLL);
                    continue;
                }
                if !run_camera_once(&hub, CAMERA_PROGRAMS) {
                    thread::sleep(RESTART_BACKOFF);
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn jpeg(body: &[u8]) -> Vec<u8> {
        [&SOI[..], body, &EOI[..]].concat()
    }

    #[test]
    fn splitter_extracts_consecutive_frames() {
        let (a, b) = (jpeg(&[1, 2, 3]), jpeg(&[4, 5]));
        let frames = MjpegSplitter::default().push(&[a.clone(), b.clone()].concat());
        assert_eq!(frames, vec![a, b]);
    }

    #[test]
    fn splitter_handles_markers_split_across_reads() {
        let frame = jpeg(&[1, 2, 3]);
        let mut splitter = MjpegSplitter::default();
        let mut collected = Vec::new();
        for byte in &frame {
            collected.extend(splitter.push(std::slice::from_ref(byte)));
        }
        assert_eq!(collected, vec![frame]);
    }

    #[test]
    fn splitter_skips_garbage_before_a_frame_and_keeps_stuffed_bytes() {
        let frame = jpeg(&[0xFF, 0x00, 7]);
        let mut input = vec![9, 9, 0xFF];
        input.extend_from_slice(&frame);
        assert_eq!(MjpegSplitter::default().push(&input), vec![frame]);
    }

    #[test]
    fn splitter_discards_an_endless_frame() {
        let mut splitter = MjpegSplitter::default();
        splitter.push(&SOI);
        splitter.push(&vec![0u8; MAX_JPEG_BYTES + 1]);
        let frame = jpeg(&[1]);
        assert_eq!(splitter.push(&frame), vec![frame]);
    }

    #[test]
    fn hub_numbers_frames_and_reports_rate() {
        let hub = CameraHub::new();
        hub.set_running(true);
        hub.publish(vec![1]);
        thread::sleep(Duration::from_millis(10));
        hub.publish(vec![2]);
        let stats = hub.stats(Instant::now());
        assert_eq!(stats.frames, 2);
        assert!(stats.fps > 0.0);
        assert_eq!(hub.subscribe().borrow().as_ref().unwrap().jpeg, vec![2]);
    }

    #[test]
    fn hub_wants_frames_only_while_requested() {
        let hub = CameraHub::new();
        let now = Instant::now();
        assert!(!hub.wants_frames(now));

        let receiver = hub.subscribe();
        assert!(hub.wants_frames(now));
        drop(receiver);
        assert!(!hub.wants_frames(now));

        hub.note_request();
        let now = Instant::now();
        assert!(hub.wants_frames(now));
        assert!(!hub.wants_frames(now + IDLE_SHUTDOWN + Duration::from_secs(1)));
    }

    /// `rpicam-vid` の代わりに JPEG を出し続けるスクリプト
    #[cfg(unix)]
    fn fake_camera(name: &str, body: &str) -> std::path::PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("picarx-camera-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

    #[cfg(unix)]
    #[test]
    fn camera_runs_while_watched_and_stops_when_no_one_is_watching() {
        let script = fake_camera(
            "streaming",
            r"while true; do printf '\377\330x\377\331'; sleep 0.02; done",
        );
        let program = script.to_str().unwrap().to_string();
        let hub = Arc::new(CameraHub::new());
        let mut frames = hub.subscribe();

        let runner_hub = Arc::clone(&hub);
        let runner = thread::spawn(move || run_camera_once(&runner_hub, [&program, &program]));

        let deadline = Instant::now() + Duration::from_secs(5);
        while frames.borrow_and_update().is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        let frame = frames
            .borrow()
            .clone()
            .expect("frame from the camera process");
        assert_eq!(frame.jpeg, vec![0xFF, 0xD8, b'x', 0xFF, 0xD9]);
        assert!(hub.stats(Instant::now()).running);

        drop(frames);
        assert!(
            runner.join().unwrap(),
            "stopping for lack of demand is not a failure"
        );
        assert!(!hub.stats(Instant::now()).running);
    }

    #[cfg(unix)]
    #[test]
    fn camera_exit_is_reported_as_a_failure() {
        let script = fake_camera("failing", "exit 3");
        let program = script.to_str().unwrap();
        let hub = Arc::new(CameraHub::new());
        let _watcher = hub.subscribe();

        assert!(!run_camera_once(&hub, [program, program]));
        assert!(hub.last_error().unwrap().contains("終了"));
    }

    #[test]
    fn missing_camera_command_is_reported() {
        let hub = Arc::new(CameraHub::new());
        let _watcher = hub.subscribe();
        assert!(!run_camera_once(
            &hub,
            ["no-such-camera-a", "no-such-camera-b"]
        ));
        assert!(hub.last_error().unwrap().contains("起動できません"));
    }
}
