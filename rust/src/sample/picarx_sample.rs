use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::Array3;
use show_image::{create_window, WindowOptions, ImageInfo, PixelFormat, ImageView};
use show_image::event::VirtualKeyCode;
use crate::rpi_camera::RpiCamera;
use crate::picarx::Picarx;

use super::rc_controller::{RcController, InputAction};

#[show_image::main]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================");
    println!(" Starting PiCar-X Key-Control Mode (with Camera)");
    println!("=================================================");
    println!("[操作方法] ※cameraウィンドウにfocusを当ててください");
    println!("  ・W / S : 前進 / 後退");
    println!("  ・A / D : ステアリング (左旋回 / 右旋回)");
    println!("  ・I / K : カメラ首振り Tilt (上下)");
    println!("  ・J / L : カメラ首振り Pan  (左右)");
    println!("  ・C     : カメラを正面に戻す (センタリング)");
    println!("  · G     : グレイスケールセンサーの値を表示");
    println!("  ・U     : 超音波センサーの距離 (cm) を表示");
    println!("  ・V     : バッテリーステータス(電圧)を表示");
    println!("  ・B     : スピーカーからBeep音を鳴らす");
    println!("  ・Q / ESC : プログラム終了");
    println!("-------------------------------------------------");

    // キーマップ設定
    let mut key_map = HashMap::new();
    key_map.insert(VirtualKeyCode::W, InputAction::MoveForward);
    key_map.insert(VirtualKeyCode::S, InputAction::MoveBackward);
    key_map.insert(VirtualKeyCode::A, InputAction::SteerLeft);
    key_map.insert(VirtualKeyCode::D, InputAction::SteerRight);
    key_map.insert(VirtualKeyCode::I, InputAction::CameraUp);
    key_map.insert(VirtualKeyCode::K, InputAction::CameraDown);
    key_map.insert(VirtualKeyCode::J, InputAction::CameraLeft);
    key_map.insert(VirtualKeyCode::L, InputAction::CameraRight);
    key_map.insert(VirtualKeyCode::C, InputAction::CameraCenter);
    key_map.insert(VirtualKeyCode::G, InputAction::GetGrayscale);
    key_map.insert(VirtualKeyCode::U, InputAction::GetDistance);
    key_map.insert(VirtualKeyCode::V, InputAction::GetBatteryStatus);
    key_map.insert(VirtualKeyCode::B, InputAction::BeepBuzzer);

    let width = 640;
    let height = 480;
    let framerate = 30;
    let frame_interval = Duration::from_secs_f64(1.0 / framerate as f64);
    let mut rgb_buffer = Array3::<u8>::zeros((height, width, 3));

    let mut camera = RpiCamera::new(width, height, framerate)?;
    let mut px = Picarx::new()?;
    let mut controller = RcController::new(key_map);

    let window = create_window(
        "PiCar-X Rust Camera Live",
        WindowOptions { preserve_aspect_ratio: true, ..Default::default() },
    )?;
    let event_channel = window.event_channel()?;

    'main_loop: loop {
        let loop_start = Instant::now();

        // 映像の取得と描画
        if !camera.fetch_latest_frame_to_rgb(&mut rgb_buffer) { break; }
        let image_view = ImageView::new(
            ImageInfo::new(PixelFormat::Rgb8, width as u32, height as u32),
            rgb_buffer.as_slice().unwrap()
        );
        if window.set_image("live_frame", image_view).is_err() { break; }

        // キー入力イベントの処理
        while let Ok(event) = event_channel.try_recv() {
            if !controller.process_event(&event) {
                println!("Exit signal received.");
                break 'main_loop;
            }
        }

        // コントローラーに px を渡して、ハードウェアへの命令送信を一任する
        controller.update_hardware(&mut px);

        // カメラのフレームレートに合わせてループを間引き、CPU占有とI2C書き込みの過多を防ぐ
        let elapsed = loop_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    println!("Stopping camera and cleaning up PiCar-X...");
    Ok(())
}

