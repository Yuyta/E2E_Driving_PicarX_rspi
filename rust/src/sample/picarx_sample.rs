use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use ndarray::Array3;
use eframe::egui;
use crate::rpi_camera::RpiCamera;
use crate::picarx::Picarx;
use crate::llm_agent::{LlmAgent, LlmResponse, CommandQueue};
use crate::sensor_monitor::SensorMonitor;

use super::rc_controller::{RcController, InputAction};

// バッテリー低下警告の閾値(V)
const BATTERY_LOW_VOLTAGE: f64 = 7.0;

/// 日本語グリフ(egui標準フォントには含まれない)を表示できるようフォントを追加する。
/// Sawarabi Gothic (OFLライセンス, rust/assets/に同梱)
fn setup_japanese_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "sawarabi_gothic".to_owned(),
        egui::FontData::from_static(include_bytes!("../../assets/SawarabiGothic-Regular.ttf")).into(),
    );
    fonts.families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "sawarabi_gothic".to_owned());
    fonts.families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("sawarabi_gothic".to_owned());
    ctx.set_fonts(fonts);
}

/// アプリの動作モード。Manual以外ではRcControllerを駆動しないことで、
/// LLMコマンド実行中に手動操作が割り込むのを防ぐ。
enum AppMode {
    Manual,
    WaitingForLlm,
    ExecutingLlmCommands { queue: CommandQueue },
}

struct PiCarApp {
    camera: RpiCamera,
    px: Arc<Mutex<Picarx>>,
    controller: RcController,
    llm_agent: LlmAgent,
    sensor_monitor: SensorMonitor,
    app_mode: AppMode,
    prompt_buffer: String,
    prompt_field_id: egui::Id,
    texture: Option<egui::TextureHandle>,
    rgb_buffer: Array3<u8>,
    width: usize,
    height: usize,
    frame_interval: Duration,
}

impl PiCarApp {
    fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let width = 640;
        let height = 480;
        let framerate = 30;

        let mut key_map = HashMap::new();
        key_map.insert(egui::Key::W, InputAction::MoveForward);
        key_map.insert(egui::Key::S, InputAction::MoveBackward);
        key_map.insert(egui::Key::A, InputAction::SteerLeft);
        key_map.insert(egui::Key::D, InputAction::SteerRight);
        key_map.insert(egui::Key::I, InputAction::CameraUp);
        key_map.insert(egui::Key::K, InputAction::CameraDown);
        key_map.insert(egui::Key::J, InputAction::CameraLeft);
        key_map.insert(egui::Key::L, InputAction::CameraRight);
        key_map.insert(egui::Key::C, InputAction::CameraCenter);
        key_map.insert(egui::Key::B, InputAction::BeepBuzzer);

        let camera = RpiCamera::new(width, height, framerate)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
        let px = Picarx::new()
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.to_string().into() })?;
        let px = Arc::new(Mutex::new(px));
        let sensor_monitor = SensorMonitor::spawn(Arc::clone(&px));

        Ok(Self {
            camera,
            px,
            controller: RcController::new(key_map),
            llm_agent: LlmAgent::new(),
            sensor_monitor,
            app_mode: AppMode::Manual,
            prompt_buffer: String::new(),
            prompt_field_id: egui::Id::new("prompt_input"),
            texture: None,
            rgb_buffer: Array3::<u8>::zeros((height, width, 3)),
            width,
            height,
            frame_interval: Duration::from_secs_f64(1.0 / framerate as f64),
        })
    }
}

impl eframe::App for PiCarApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // カメラのフレームレートに合わせて再描画し続ける(eguiは既定では入力がないと再描画しない)
        ctx.request_repaint_after(self.frame_interval);

        // 映像の取得とテクスチャ更新 (GPUテクスチャは使い回す)
        if self.camera.fetch_latest_frame_to_rgb(&mut self.rgb_buffer) {
            let color_image = egui::ColorImage::from_rgb(
                [self.width, self.height],
                self.rgb_buffer.as_slice().unwrap(),
            );
            match &mut self.texture {
                Some(tex) => tex.set(color_image, egui::TextureOptions::LINEAR),
                None => {
                    self.texture = Some(ctx.load_texture(
                        "live_frame",
                        color_image,
                        egui::TextureOptions::LINEAR,
                    ));
                }
            }
        }

        // レイアウトより先にフォーカス状態を確定させる
        // (プロンプト入力中はWASD等の直接操作もEscape終了も無効にするため)
        let prompt_has_focus = ctx.memory(|m| m.has_focus(self.prompt_field_id));

        if !prompt_has_focus
            && ctx.input(|i| i.key_pressed(egui::Key::Escape) || i.key_pressed(egui::Key::Q))
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            // 1. センサー計器類 (最上部、バックグラウンドスレッドの最新値を常時表示)
            let snapshot = self.sensor_monitor.snapshot();
            ui.horizontal(|ui| {
                match snapshot.distance_cm {
                    Some(d) if d >= 0.0 => {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, format!("距離: {:.1} cm", d));
                    }
                    _ => { ui.colored_label(egui::Color32::LIGHT_BLUE, "距離: -- cm"); }
                }
                if let Some(g) = snapshot.grayscale {
                    ui.colored_label(
                        egui::Color32::LIGHT_YELLOW,
                        format!("グレイスケール: L{:4} C{:4} R{:4}", g[0], g[1], g[2]),
                    );
                }
                match snapshot.battery {
                    Some((voltage, percent)) if voltage < BATTERY_LOW_VOLTAGE => {
                        ui.colored_label(egui::Color32::RED, format!("バッテリー: {:.2}V ({}%) 低下!", voltage, percent));
                    }
                    Some((voltage, percent)) => {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, format!("バッテリー: {:.2}V ({}%)", voltage, percent));
                    }
                    None => { ui.colored_label(egui::Color32::LIGHT_GREEN, "バッテリー: --"); }
                }
            });

            ui.separator();

            // 2. カメラ映像 (中央)
            if let Some(tex) = &self.texture {
                ui.add(egui::Image::new(tex).max_width(self.width as f32));
            }

            ui.separator();

            // 3. プロンプト入力 (最下部)
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.prompt_buffer)
                    .id(self.prompt_field_id)
                    .hint_text("自然言語で指示を入力 (Enterで送信)")
                    .desired_width(f32::INFINITY),
            );

            match &self.app_mode {
                AppMode::WaitingForLlm => { ui.label("[LLM] 応答待ち..."); }
                AppMode::ExecutingLlmCommands { .. } => { ui.label("[LLM] コマンド実行中 (Escで緊急停止)"); }
                AppMode::Manual => {}
            }

            let submitted = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if submitted && matches!(self.app_mode, AppMode::Manual) && !self.prompt_buffer.trim().is_empty() {
                let prompt = std::mem::take(&mut self.prompt_buffer);
                println!("[LLM] 送信: {}", prompt);
                self.px.lock().unwrap().stop();
                self.llm_agent.submit_prompt(prompt);
                self.app_mode = AppMode::WaitingForLlm;
            }

            // 手動運転はManualモード、かつプロンプト欄が未フォーカスのときだけ反映する
            if matches!(self.app_mode, AppMode::Manual) && !prompt_has_focus {
                self.controller.sync_from_egui(ctx);
                self.controller.update_hardware(&mut self.px.lock().unwrap());
            }
        });

        match &mut self.app_mode {
            AppMode::Manual => {}
            AppMode::WaitingForLlm => {
                if !prompt_has_focus && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    println!("[LLM] 緊急停止");
                    self.px.lock().unwrap().stop();
                    self.app_mode = AppMode::Manual;
                    return;
                }
                if let Some(response) = self.llm_agent.try_recv_response() {
                    match response {
                        LlmResponse::Ok(commands) => {
                            println!("[LLM] {}件のコマンドを受信、実行します", commands.len());
                            self.app_mode = AppMode::ExecutingLlmCommands { queue: CommandQueue::new(commands) };
                        }
                        LlmResponse::Err(e) => {
                            println!("[LLM] エラー: {}", e);
                            self.app_mode = AppMode::Manual;
                        }
                    }
                }
            }
            AppMode::ExecutingLlmCommands { queue } => {
                if !prompt_has_focus && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    println!("[LLM] 緊急停止");
                    self.px.lock().unwrap().stop();
                    self.app_mode = AppMode::Manual;
                    return;
                }
                if !queue.tick(&mut self.px.lock().unwrap(), self.sensor_monitor.snapshot()) {
                    println!("[LLM] コマンド実行完了");
                    self.app_mode = AppMode::Manual;
                }
            }
        }
    }
}

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
    println!("  ・B     : スピーカーからBeep音を鳴らす");
    println!("  ・距離/グレイスケール/バッテリーはウィンドウ上部に常時表示されます");
    println!("  ・テキスト欄をクリックして自然言語で指示を入力、Enterで送信 (LLM経由でコマンド実行)");
    println!("  ・Q / ESC : プログラム終了 (テキスト欄が未フォーカスのとき)");
    println!("-------------------------------------------------");

    // Raspberry Pi 3B+ (VideoCore IV, OpenGL ES 2.0)ではglowバックエンドで
    // GL_INVALID_ENUM等の既知の不具合報告があるため、明示的にwgpuを指定する
    // (emilk/egui#4302)
    let native_options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "PiCar-X Rust Camera Live",
        native_options,
        Box::new(|cc| {
            setup_japanese_font(&cc.egui_ctx);
            Ok(Box::new(PiCarApp::new()?))
        }),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { e.to_string().into() })?;

    println!("Stopping camera and cleaning up PiCar-X...");
    Ok(())
}
