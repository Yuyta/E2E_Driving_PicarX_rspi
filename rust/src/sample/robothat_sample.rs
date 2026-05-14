use std::collections::HashMap;
use show_image::{create_window, WindowOptions, ImageInfo, PixelFormat, ImageView};
use show_image::event::VirtualKeyCode;
use crate::picarx::Picarx;

use super::rc_controller::{RcController, InputAction};

#[show_image::main]
pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=================================================");
    println!("🤖 Starting Robot-Hat Key-Control Mode (No Camera)");
    println!("=================================================");
    println!("[操作方法] ※フォーカスを操作ウィンドウに当ててください");
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

    // 1. キーマップ設定
    let mut key_map = HashMap::new();
    key_map.insert(VirtualKeyCode::W, InputAction::MoveForward);
    key_map.insert(VirtualKeyCode::S, InputAction::MoveBackward);
    key_map.insert(VirtualKeyCode::A, InputAction::SteerLeft);
    key_map.insert(VirtualKeyCode::D, InputAction::SteerRight);

    // カメラ映像は出ませんが、サーボモーター単体の動作テストができるようにマッピングを維持
    key_map.insert(VirtualKeyCode::I, InputAction::CameraUp);
    key_map.insert(VirtualKeyCode::K, InputAction::CameraDown);
    key_map.insert(VirtualKeyCode::J, InputAction::CameraLeft);
    key_map.insert(VirtualKeyCode::L, InputAction::CameraRight);
    key_map.insert(VirtualKeyCode::C, InputAction::CameraCenter);

    key_map.insert(VirtualKeyCode::G, InputAction::GetGrayscale);
    key_map.insert(VirtualKeyCode::U, InputAction::GetDistance);
    key_map.insert(VirtualKeyCode::V, InputAction::GetBatteryStatus);
    key_map.insert(VirtualKeyCode::B, InputAction::BeepBuzzer);

    // 2. ハードウェアとコントローラー（既存クラス）の初期化
    let mut px = Picarx::new()?;
    let mut controller = RcController::new(key_map);

    // 3. ウィンドウを維持するための軽量なダミー画像（200x100の真っ黒なRGBデータ）を用意
    let win_width = 200;
    let win_height = 100;
    let dummy_black_image = vec![0u8; win_width * win_height * 3];

    // 4. 操作用の受付ウィンドウを作成
    let window = create_window(
        "Robot-Hat Control Pad [Focus Here]",
        WindowOptions { preserve_aspect_ratio: true, ..Default::default() },
    )?;
    let event_channel = window.event_channel()?;

    println!("★ Control Window Opened! Focus the window and use W/A/S/D to drive!");
    println!("★ Press ESC or Q in the window to Exit.");

    'main_loop: loop {
        // 5. 描画処理（show_imageの仕様上、ウィンドウの維持とキーイベント取得のために必要）
        let image_view = ImageView::new(
            ImageInfo::new(PixelFormat::Rgb8, win_width as u32, win_height as u32),
            &dummy_black_image
        );
        if window.set_image("control_pad", image_view).is_err() {
            break;
        }

        // 6. 既存の RcController にキー入力を渡して処理
        while let Ok(event) = event_channel.try_recv() {
            if !controller.process_event(&event) {
                println!("Exit signal received.");
                break 'main_loop;
            }
        }

        // 7. 既存の RcController を使って、実機（px）に命令送信を一任
        controller.update_hardware(&mut px);

        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    println!("Cleaning up and stopping Robot-Hat...");
    Ok(())
}

