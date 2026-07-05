use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::Array3;
use show_image::{create_window, WindowOptions, ImageInfo, PixelFormat, ImageView};
use show_image::event::{VirtualKeyCode, WindowEvent, ElementState};
use crate::rpi_camera::RpiCamera;
use crate::picarx::Picarx;
use crate::llm_agent::{LlmAgent, LlmResponse, CommandQueue};

use super::rc_controller::{RcController, InputAction};

/// アプリの動作モード。Manual以外ではRcControllerへイベントを渡さないことで、
/// プロンプト入力中のキー入力(例: "w"を含む文章)が誤って車を動かすのを防ぐ。
enum AppMode {
    Manual,
    PromptInput { buffer: String },
    WaitingForLlm,
    ExecutingLlmCommands { queue: CommandQueue },
}

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
    println!("  ・Enter : 自然言語で指示を入力 (LLM経由でコマンド実行)");
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
    let llm_agent = LlmAgent::new();
    let mut app_mode = AppMode::Manual;

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

        // キー入力イベントの処理 (モードごとに分岐)
        while let Ok(event) = event_channel.try_recv() {
            match &mut app_mode {
                AppMode::Manual => {
                    if let WindowEvent::KeyboardInput(k) = &event {
                        if k.input.state == ElementState::Pressed && k.input.key_code == Some(VirtualKeyCode::Return) {
                            px.stop();
                            app_mode = AppMode::PromptInput { buffer: String::new() };
                            println!("[LLM] プロンプト入力モード: 文章を入力しEnterで送信、Escでキャンセル");
                            continue;
                        }
                    }

                    if !controller.process_event(&event) {
                        println!("Exit signal received.");
                        break 'main_loop;
                    }
                }
                AppMode::PromptInput { buffer } => {
                    match &event {
                        WindowEvent::TextInput(text_event) => {
                            if !text_event.character.is_control() {
                                buffer.push(text_event.character);
                            }
                        }
                        WindowEvent::KeyboardInput(k) if k.input.state == ElementState::Pressed => {
                            match k.input.key_code {
                                Some(VirtualKeyCode::Return) => {
                                    let prompt = buffer.clone();
                                    println!("[LLM] 送信: {}", prompt);
                                    llm_agent.submit_prompt(prompt);
                                    app_mode = AppMode::WaitingForLlm;
                                }
                                Some(VirtualKeyCode::Back) => {
                                    buffer.pop();
                                }
                                Some(VirtualKeyCode::Escape) => {
                                    println!("[LLM] 入力をキャンセルしました");
                                    app_mode = AppMode::Manual;
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    }
                    // PromptInput中はcontroller.process_event()を呼ばない
                    // (打鍵中の文字が誤ってWASD等のアクションとして解釈されるのを防ぐ)
                }
                AppMode::WaitingForLlm | AppMode::ExecutingLlmCommands { .. } => {
                    // LLM応答待ち/コマンド実行中はEscapeで緊急停止のみ受け付ける
                    if let WindowEvent::KeyboardInput(k) = &event {
                        if k.input.state == ElementState::Pressed && k.input.key_code == Some(VirtualKeyCode::Escape) {
                            println!("[LLM] 緊急停止");
                            px.stop();
                            app_mode = AppMode::Manual;
                        }
                    }
                }
            }
        }

        // モードごとのハードウェア更新
        match &mut app_mode {
            AppMode::Manual => {
                controller.update_hardware(&mut px);
            }
            AppMode::PromptInput { .. } => {
                // 入力中はハードウェアを更新しない
            }
            AppMode::WaitingForLlm => {
                if let Some(response) = llm_agent.try_recv_response() {
                    match response {
                        LlmResponse::Ok(commands) => {
                            println!("[LLM] {}件のコマンドを受信、実行します", commands.len());
                            app_mode = AppMode::ExecutingLlmCommands { queue: CommandQueue::new(commands) };
                        }
                        LlmResponse::Err(e) => {
                            println!("[LLM] エラー: {}", e);
                            app_mode = AppMode::Manual;
                        }
                    }
                }
            }
            AppMode::ExecutingLlmCommands { queue } => {
                if !queue.tick(&mut px) {
                    println!("[LLM] コマンド実行完了");
                    app_mode = AppMode::Manual;
                }
            }
        }

        // カメラのフレームレートに合わせてループを間引き、CPU占有とI2C書き込みの過多を防ぐ
        let elapsed = loop_start.elapsed();
        if elapsed < frame_interval {
            std::thread::sleep(frame_interval - elapsed);
        }
    }

    println!("Stopping camera and cleaning up PiCar-X...");
    Ok(())
}
