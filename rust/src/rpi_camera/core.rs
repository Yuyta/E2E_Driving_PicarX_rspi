use std::io::Read;
use std::process::{Command, Child, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use ndarray::{Array1, Array3, ArrayView2};

pub struct RpiCamera {
    child: Child,
    yuv_buffer_shared: Arc<Mutex<Array1<u8>>>,
    yuv_buffer_local: Array1<u8>,
    width: usize,
    height: usize,
}

impl RpiCamera {
    pub fn new(width: usize, height: usize, framerate: u32) -> Result<Self, Box<dyn std::error::Error>> {
        let yuv_buffer_size = (width * height * 3) / 2;
        let yuv_buffer_shared = Arc::new(Mutex::new(Array1::<u8>::zeros(yuv_buffer_size)));
        let yuv_buffer_local = Array1::<u8>::zeros(yuv_buffer_size);

        // 1. カメラプロセスの起動 (rpicam-vid 優先、ダメなら libcamera-vid)
        let mut child = Command::new("rpicam-vid")
            .args([
                "--nopreview", "-t", "0", "--codec", "yuv420",
                "--width", &width.to_string(),
                "--height", &height.to_string(),
                "--framerate", &framerate.to_string(),
                "-o", "-"
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .or_else(|_| {
                Command::new("libcamera-vid")
                    .args([
                        "--nopreview", "-t", "0", "--codec", "yuv420",
                        "--width", &width.to_string(),
                        "--height", &height.to_string(),
                        "--framerate", &framerate.to_string(),
                        "-o", "-"
                    ])
                    .stdout(Stdio::piped())
                    .stderr(Stdio::null())
                    .spawn()
            })?;

        let mut stdout = child.stdout.take().ok_or("Failed to open stdout")?;
        let yuv_buffer_shared_clone = Arc::clone(&yuv_buffer_shared);

        // capture_loop
        thread::spawn(move || {
            let mut local_buf = Array1::<u8>::zeros(yuv_buffer_size);
            loop {
                if stdout.read_exact(local_buf.as_slice_mut().unwrap()).is_err() {
                    break;
                }
                if let Ok(mut guard) = yuv_buffer_shared_clone.lock() {
                    guard.assign(&local_buf);
                }
            }
        });

        Ok(Self {
            child,
            yuv_buffer_shared,
            yuv_buffer_local,
            width,
            height,
        })
    }

    /// 最新のフレームを取得し、外部から渡された rgb_array を直接書き換える (アロケーションフリー)
    pub fn fetch_latest_frame_to_rgb(&mut self, rgb_array: &mut Array3<u8>) -> bool {
        // 共有エリアから最新YUVを手元にコピー
        if let Ok(guard) = self.yuv_buffer_shared.lock() {
            self.yuv_buffer_local.assign(&*guard);
        } else {
            return false;
        }

        // RGBへ変換
        Self::convert_yuv420_to_rgb(&self.yuv_buffer_local, rgb_array, self.width, self.height);
        true
    }

    /// YUV420からRGBへの変換内部ロジック (staticメソッド)
    fn convert_yuv420_to_rgb(yuv_array: &Array1<u8>, rgb_array: &mut Array3<u8>, width: usize, height: usize) {
        let uv_width = width / 2;
        let uv_height = height / 2;
        let y_size = width * height;
        let uv_size = uv_width * uv_height;
        let yuv_slice = yuv_array.as_slice().unwrap();

        let y_plane = ArrayView2::from_shape((height, width), &yuv_slice[0..y_size]).unwrap();
        let u_plane = ArrayView2::from_shape((uv_height, uv_width), &yuv_slice[y_size..(y_size + uv_size)]).unwrap();
        let v_plane = ArrayView2::from_shape((uv_height, uv_width), &yuv_slice[(y_size + uv_size)..]).unwrap();

        for y in 0..height {
            let uv_y = y / 2;
            for x in 0..width {
                let uv_x = x / 2;
                let yp = y_plane[[y, x]] as f32;
                let up = u_plane[[uv_y, uv_x]] as f32 - 128.0;
                let vp = v_plane[[uv_y, uv_x]] as f32 - 128.0;

                rgb_array[[y, x, 0]] = (yp + 1.402 * vp).clamp(0.0, 255.0) as u8;
                rgb_array[[y, x, 1]] = (yp - 0.344136 * up - 0.714136 * vp).clamp(0.0, 255.0) as u8;
                rgb_array[[y, x, 2]] = (yp + 1.772 * up).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

impl Drop for RpiCamera {
    fn drop(&mut self) {
        println!("Shutting down camera process cleanly...");
        let _ = self.child.kill();
    }
}

