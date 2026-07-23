use ndarray::{Array1, Array3, ArrayView2, s};
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

pub struct RpiCamera {
    child: Child,
    yuv_buffer_shared: Arc<Mutex<Array1<u8>>>,
    yuv_buffer_local: Array1<u8>,
    width: usize,
    height: usize,
}

impl RpiCamera {
    pub fn new(
        width: usize,
        height: usize,
        framerate: u32,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let yuv_buffer_size = (width * height * 3) / 2;
        let yuv_buffer_shared = Arc::new(Mutex::new(Array1::<u8>::zeros(yuv_buffer_size)));
        let yuv_buffer_local = Array1::<u8>::zeros(yuv_buffer_size);

        // 1. カメラプロセスの起動 (rpicam-vid 優先、ダメなら libcamera-vid)
        let mut child = Command::new("rpicam-vid")
            .args([
                "--nopreview",
                "-t",
                "0",
                "--codec",
                "yuv420",
                "--width",
                &width.to_string(),
                "--height",
                &height.to_string(),
                "--framerate",
                &framerate.to_string(),
                "-o",
                "-",
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .or_else(|_| {
                Command::new("libcamera-vid")
                    .args([
                        "--nopreview",
                        "-t",
                        "0",
                        "--codec",
                        "yuv420",
                        "--width",
                        &width.to_string(),
                        "--height",
                        &height.to_string(),
                        "--framerate",
                        &framerate.to_string(),
                        "-o",
                        "-",
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
                if stdout
                    .read_exact(local_buf.as_slice_mut().unwrap())
                    .is_err()
                {
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

    /// YUV420からRGBへの変換内部ロジック
    fn convert_yuv420_to_rgb(
        yuv_array: &Array1<u8>,
        rgb_array: &mut Array3<u8>,
        width: usize,
        height: usize,
    ) {
        let y_size = width * height;
        let uv_width = width / 2;
        let uv_height = height / 2;
        let uv_size = uv_width * uv_height;
        let yuv_slice = yuv_array.as_slice().unwrap();

        // 各プレーンを2次元のビュー(View)として切り出す
        let y_plane = ArrayView2::from_shape((height, width), &yuv_slice[0..y_size]).unwrap();
        let u_plane_raw = ArrayView2::from_shape(
            (uv_height, uv_width),
            &yuv_slice[y_size..(y_size + uv_size)],
        )
        .unwrap();
        let v_plane_raw =
            ArrayView2::from_shape((uv_height, uv_width), &yuv_slice[(y_size + uv_size)..])
                .unwrap();

        let u_shaped = u_plane_raw.to_shape((uv_height, 1, uv_width, 1)).unwrap();
        let u_broadcasted = u_shaped.broadcast((uv_height, 2, uv_width, 2)).unwrap();
        let u_plane = u_broadcasted.to_shape((height, width)).unwrap();

        let v_shaped = v_plane_raw.to_shape((uv_height, 1, uv_width, 1)).unwrap();
        let v_broadcasted = v_shaped.broadcast((uv_height, 2, uv_width, 2)).unwrap();
        let v_plane = v_broadcasted.to_shape((height, width)).unwrap();

        // 一括浮動小数点演算
        let y_f32 = y_plane.mapv(|x| x as f32);
        let u_f32 = u_plane.mapv(|x| x as f32 - 128.0);
        let v_f32 = v_plane.mapv(|x| x as f32 - 128.0);

        // 各チャンネル(R, G, B)の全ピクセルを一括計算
        let r_channel = (&y_f32 + &v_f32 * 1.402).mapv(|x| x.clamp(0.0, 255.0) as u8);
        let g_channel =
            (&y_f32 - &u_f32 * 0.344136 - &v_f32 * 0.714136).mapv(|x| x.clamp(0.0, 255.0) as u8);
        let b_channel = (&y_f32 + &u_f32 * 1.772).mapv(|x| x.clamp(0.0, 255.0) as u8);

        // 最終出力用の rgb_array (Array3) の各スライスへ一括代入
        rgb_array.slice_mut(s![.., .., 0]).assign(&r_channel);
        rgb_array.slice_mut(s![.., .., 1]).assign(&g_channel);
        rgb_array.slice_mut(s![.., .., 2]).assign(&b_channel);
    }
}

impl Drop for RpiCamera {
    fn drop(&mut self) {
        println!("Shutting down camera process cleanly...");
        let _ = self.child.kill();
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array3};

    use super::RpiCamera;

    #[test]
    fn neutral_yuv_frame_converts_to_neutral_rgb() {
        let yuv = Array1::from_vec(vec![128; 6]);
        let mut rgb = Array3::<u8>::zeros((2, 2, 3));

        RpiCamera::convert_yuv420_to_rgb(&yuv, &mut rgb, 2, 2);

        assert!(rgb.iter().all(|channel| *channel == 128));
    }
}
