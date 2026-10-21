use crate::rpi_camera::RpiCamera;
use eframe::egui;
use ndarray::Array3;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;
// 自作モジュール、PiCar-X制御用
use crate::llm_agent::{
    CommandQueue, GeminiAgent, GeminiProvider, LlmProvider, LlmResponse, LlmType, OllamaAgent,
};
use crate::picarx::Picarx;
use crate::sensor_monitor::SensorMonitor;

use super::rc_controller::{InputAction, RcController};

// バッテリー低下警告の閾値(V)
const BATTERY_LOW_VOLTAGE: f64 = 7.0;

/// 日本語グリフ(egui標準フォントには含まれない)を表示できるようフォントを追加する。
/// Sawarabi Gothic (OFLライセンス, rust/assets/に同梱)
fn setup_japanese_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "sawarabi_gothic".to_owned(),
        egui::FontData::from_static(include_bytes!("../../assets/SawarabiGothic-Regular.ttf"))
            .into(),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "sawarabi_gothic".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .push("sawarabi_gothic".to_owned());
    ctx.set_fonts(fonts);
}

/// アプリの動作モード。Manual以外ではRcControllerを駆動しないことで、
/// LLMコマンド実行中に手動操作が割り込むのを防ぐ。
enum AppMode {
    Manual,
    WaitingForLlm { llm: LlmType, request_id: u64 },
    ExecutingLlmCommands { queue: CommandQueue },
}

struct PiCarApp {
    camera: RpiCamera,
    px: Arc<Mutex<Picarx>>,
    controller: RcController,
    ollama_agent: OllamaAgent,
    gemini_agent: GeminiAgent,
    selected_llm: LlmType,
    selected_tts: crate::tts::TtsEngine,
    sensor_monitor: SensorMonitor,
    app_mode: AppMode,
    prompt_buffer: String,
    prompt_field_id: egui::Id,
    texture: Option<egui::TextureHandle>,
    rgb_buffer: Array3<u8>,
    width: usize,
    height: usize,
    frame_interval: Duration,
    last_reply: String,
    chat_messages: Vec<(String, String)>,
    audio_recorder: crate::audio_input::AudioRecorder,
    is_voice_recording: bool,
    memory_repo: Arc<crate::memory::repository::MemoryRepository>,
    conversation_id: String,
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

        // 会話メモリリポジトリの初期化
        let db_path = "picarx_memory.db";
        let memory_repo = crate::memory::repository::MemoryRepository::new(db_path)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e.into() })?;
        let memory_repo = Arc::new(memory_repo);

        let conversation_id = uuid::Uuid::new_v4().to_string();
        if let Err(e) = memory_repo.create_conversation(&conversation_id) {
            eprintln!("[Memory] 会話の作成に失敗: {}", e);
        } else {
            println!(
                "[Memory] 新規セッション開始 conversation_id: {}",
                conversation_id
            );
        }

        Ok(Self {
            camera,
            px,
            controller: RcController::new(key_map),
            ollama_agent: OllamaAgent::new(),
            gemini_agent: GeminiAgent::new(),
            selected_llm: LlmType::Gemini,
            selected_tts: crate::tts::TtsEngine::Default,
            sensor_monitor,
            app_mode: AppMode::Manual,
            prompt_buffer: String::new(),
            prompt_field_id: egui::Id::new("prompt_input"),
            texture: None,
            rgb_buffer: Array3::<u8>::zeros((height, width, 3)),
            width,
            height,
            frame_interval: Duration::from_secs_f64(1.0 / framerate as f64),
            last_reply: String::new(),
            chat_messages: Vec::new(),
            audio_recorder: crate::audio_input::AudioRecorder::new(),
            is_voice_recording: false,
            memory_repo,
            conversation_id,
        })
    }
}

fn show_chat_bubble(ui: &mut egui::Ui, role: &str, message: &str) {
    let available_width = ui.available_width();
    let bubble_width = (available_width * 0.82).max(120.0);
    let is_user = role == "user";

    // ユーザーは右寄せ、アシスタントは左寄せで表示する。
    let layout = if is_user {
        egui::Layout::right_to_left(egui::Align::Min)
    } else {
        egui::Layout::left_to_right(egui::Align::Min)
    };

    ui.with_layout(layout, |ui| {
        egui::Frame::group(ui.style())
            .inner_margin(egui::Margin::symmetric(10, 7))
            .show(ui, |ui| {
                ui.set_max_width(bubble_width);
                ui.label(message);
            });
    });

    ui.add_space(6.0);
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

        egui::SidePanel::right("chat_panel")
            .resizable(true)
            .default_width(320.0)
            .min_width(240.0)
            .show(ctx, |ui| {
                ui.heading("チャット");
                ui.separator();

                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if self.chat_messages.is_empty() {
                            ui.centered_and_justified(|ui| {
                                ui.label("会話履歴はありません");
                            });
                        } else {
                            for (role, message) in &self.chat_messages {
                                show_chat_bubble(ui, role, message);
                            }
                        }
                    });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // 1. センサー計器類 (最上部、バックグラウンドスレッドの最新値を常時表示)
            let snapshot = self.sensor_monitor.snapshot();
            ui.horizontal(|ui| {
                match snapshot.distance_cm {
                    Some(d) if d >= 0.0 => {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, format!("距離: {:.1} cm", d));
                    }
                    _ => {
                        ui.colored_label(egui::Color32::LIGHT_BLUE, "距離: -- cm");
                    }
                }
                if let Some(g) = snapshot.grayscale {
                    ui.colored_label(
                        egui::Color32::LIGHT_YELLOW,
                        format!("グレイスケール: L{:4} C{:4} R{:4}", g[0], g[1], g[2]),
                    );
                }
                match snapshot.battery {
                    Some((voltage, percent)) if voltage < BATTERY_LOW_VOLTAGE => {
                        ui.colored_label(
                            egui::Color32::RED,
                            format!("バッテリー: {:.2}V ({}%) 低下!", voltage, percent),
                        );
                    }
                    Some((voltage, percent)) => {
                        ui.colored_label(
                            egui::Color32::LIGHT_GREEN,
                            format!("バッテリー: {:.2}V ({}%)", voltage, percent),
                        );
                    }
                    None => {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, "バッテリー: --");
                    }
                }
            });

            ui.separator();

            // 2. カメラ映像 (中央)
            if let Some(tex) = &self.texture {
                ui.add(egui::Image::new(tex).max_width(self.width as f32));
            }

            ui.separator();

            // 3. プロンプト入力 (最下部)
            ui.horizontal(|ui| {
                ui.label("使用するLLM:");
                let can_change_llm = matches!(&self.app_mode, AppMode::Manual);
                ui.add_enabled_ui(can_change_llm, |ui| {
                    ui.radio_value(&mut self.selected_llm, LlmType::Gemini, "Gemini API");
                    ui.radio_value(&mut self.selected_llm, LlmType::Ollama, "Ollama (ローカル)");
                });
                ui.separator();
                ui.label("TTSエンジン:");
                ui.radio_value(
                    &mut self.selected_tts,
                    crate::tts::TtsEngine::Default,
                    "標準 (JTalk/eSpeak)",
                );
                ui.radio_value(
                    &mut self.selected_tts,
                    crate::tts::TtsEngine::PiperPlus,
                    "高品質 (Piper Plus)",
                );
            });

            if !self.last_reply.is_empty() {
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::LIGHT_GREEN, &self.last_reply);
                });
                ui.add_space(4.0);
            }

            let response = ui.add(
                egui::TextEdit::singleline(&mut self.prompt_buffer)
                    .id(self.prompt_field_id)
                    .hint_text("自然言語で指示を入力 (Enterで送信)")
                    .desired_width(f32::INFINITY),
            );

            // 4. 音声入力 (Push-to-Talk)
            let space_down = !prompt_has_focus && ctx.input(|i| i.key_down(egui::Key::Space));
            let mut record_btn_pressed = false;

            ui.horizontal(|ui| {
                let btn_text = if self.is_voice_recording {
                    "🔴 録音中... (離すと送信)"
                } else {
                    "🎙 音声指示 (Spaceキー長押し)"
                };
                let btn_response = ui.add_sized([220.0, 24.0], egui::Button::new(btn_text));
                record_btn_pressed = btn_response.is_pointer_button_down_on();

                if self.is_voice_recording {
                    ui.colored_label(egui::Color32::RED, "マイクに向かって話しかけてください...");
                }
            });

            let wants_record =
                (space_down || record_btn_pressed) && matches!(&self.app_mode, AppMode::Manual);

            if wants_record && !self.is_voice_recording {
                self.px.lock().unwrap().stop();
                match self.audio_recorder.start_recording() {
                    Ok(()) => {
                        println!("[Audio] 録音開始 (Push-to-Talk)");
                        self.is_voice_recording = true;
                    }
                    Err(e) => {
                        eprintln!("[Audio] 録音開始失敗: {}", e);
                    }
                }
            } else if !wants_record && self.is_voice_recording {
                self.is_voice_recording = false;
                println!("[Audio] 録音終了、WAV生成中...");
                match self.audio_recorder.stop_recording() {
                    Ok(wav_bytes) => {
                        println!(
                            "[Audio] 録音完了 ({} bytes)、Gemini APIに送信します",
                            wav_bytes.len()
                        );

                        self.px.lock().unwrap().stop();
                        let context = self
                            .memory_repo
                            .build_context(&self.conversation_id)
                            .unwrap_or_default();
                        match self
                            .gemini_agent
                            .submit_audio_with_context(wav_bytes, context)
                        {
                            Ok(request_id) => {
                                println!(
                                    "[LLM (Gemini, request_id={})] 音声要求を登録しました",
                                    request_id
                                );
                                self.app_mode = AppMode::WaitingForLlm {
                                    llm: LlmType::Gemini,
                                    request_id,
                                };
                            }
                            Err(error) => {
                                eprintln!("[LLM (Gemini)] 音声要求の登録に失敗しました: {}", error);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[Audio] 録音データの取得に失敗: {}", e);
                    }
                }
            }

            match &self.app_mode {
                AppMode::WaitingForLlm { .. } => {
                    ui.label("[LLM] 応答待ち...");
                }
                AppMode::ExecutingLlmCommands { .. } => {
                    ui.label("[LLM] コマンド実行中 (Escで緊急停止)");
                }
                AppMode::Manual => {}
            }

            // TextEditからフォーカスが外れたフレームでEnterが押された場合に送信する。
            let submitted = response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if submitted
                && matches!(&self.app_mode, AppMode::Manual)
                && !self.prompt_buffer.trim().is_empty()
            {
                let prompt = std::mem::take(&mut self.prompt_buffer);
                self.chat_messages.push(("user".to_owned(), prompt.clone()));
                let active_llm = self.selected_llm;
                self.px.lock().unwrap().stop();

                // ユーザー発話をDBに保存
                if let Err(e) =
                    self.memory_repo
                        .save_message(&self.conversation_id, "user", &prompt)
                {
                    eprintln!("[Memory] ユーザーメッセージの保存に失敗: {}", e);
                }

                println!("[LLM ({:?})] 送信: {}", active_llm, prompt);
                let request_id = match active_llm {
                    LlmType::Gemini => {
                        let context = self
                            .memory_repo
                            .build_context(&self.conversation_id)
                            .unwrap_or_default();
                        self.gemini_agent
                            .submit_prompt_with_context(prompt, context)
                    }
                    LlmType::Ollama => self.ollama_agent.submit_prompt(prompt),
                };
                let request_id = match request_id {
                    Ok(request_id) => request_id,
                    Err(error) => {
                        println!(
                            "[LLM ({:?})] 要求の登録に失敗しました: {}",
                            active_llm, error
                        );
                        return;
                    }
                };
                println!(
                    "[LLM ({:?}, request_id={})] 要求を登録しました",
                    active_llm, request_id
                );
                self.app_mode = AppMode::WaitingForLlm {
                    llm: active_llm,
                    request_id,
                };
            }

            // 手動運転はManualモード、録音中でなく、プロンプト欄が未フォーカスのときだけ反映する
            if matches!(&self.app_mode, AppMode::Manual)
                && !prompt_has_focus
                && !self.is_voice_recording
            {
                self.controller.sync_from_egui(ctx);
                self.controller
                    .update_hardware(&mut self.px.lock().unwrap());
            }
        });

        match &mut self.app_mode {
            AppMode::Manual => {}
            AppMode::WaitingForLlm { llm, request_id } => {
                let active_llm = *llm;
                let active_request_id = *request_id;
                if !prompt_has_focus && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
                    println!("[LLM] 緊急停止");
                    match active_llm {
                        LlmType::Gemini => self.gemini_agent.cancel(active_request_id),
                        LlmType::Ollama => self.ollama_agent.cancel(active_request_id),
                    }
                    self.px.lock().unwrap().stop();
                    self.app_mode = AppMode::Manual;
                    return;
                }
                let response = match active_llm {
                    LlmType::Gemini => self.gemini_agent.try_recv_response(),
                    LlmType::Ollama => self.ollama_agent.try_recv_response(),
                };
                match response {
                    Ok(Some(res)) if res.request_id() != active_request_id => {
                        println!(
                            "[LLM ({:?}, request_id={})] 古いレスポンスを破棄します",
                            active_llm,
                            res.request_id()
                        );
                    }
                    Ok(Some(res)) => match res {
                        LlmResponse::Ok {
                            user_transcription,
                            commands,
                            reply,
                            memory_updates,
                            ..
                        } => {
                            println!(
                                "[LLM ({:?}, request_id={})] {}件のコマンドを受信、実行します。返答: {}",
                                active_llm,
                                active_request_id,
                                commands.len(),
                                reply
                            );
                            self.last_reply = format!("[PiCar-X]: {}", reply);

                            // 音声入力の場合は、Geminiが返した文字起こしをチャットに追加する。
                            // テキスト入力の場合は送信時点ですでに追加されているため、
                            // 同じ内容を二重表示しない。
                            if !user_transcription.trim().is_empty() {
                                let already_shown = self
                                    .chat_messages
                                    .last()
                                    .map(|(role, message)| {
                                        role == "user" && message == &user_transcription
                                    })
                                    .unwrap_or(false);

                                if !already_shown {
                                    self.chat_messages
                                        .push(("user".to_owned(), user_transcription.clone()));

                                    // 音声入力の文字起こしをユーザー発話としてDBに保存
                                    if let Err(e) = self.memory_repo.save_message(
                                        &self.conversation_id,
                                        "user",
                                        &user_transcription,
                                    ) {
                                        eprintln!(
                                            "[Memory] 音声ユーザーメッセージの保存に失敗: {}",
                                            e
                                        );
                                    }
                                }
                            }

                            self.chat_messages
                                .push(("assistant".to_owned(), reply.clone()));

                            // アシスタント発話をDBに保存
                            if let Err(e) = self.memory_repo.save_message(
                                &self.conversation_id,
                                "assistant",
                                &reply,
                            ) {
                                eprintln!("[Memory] 応答メッセージの保存に失敗: {}", e);
                            }

                            // memory_updates のバリデーション＆反映
                            if let Some(updates) = memory_updates
                                && let Err(e) =
                                    crate::memory::summarizer::validate_and_apply_memory_updates(
                                        &self.memory_repo,
                                        &updates,
                                    )
                            {
                                eprintln!("[Memory] 長期記憶の反映に失敗: {}", e);
                            }

                            // 未要約メッセージの確認＆自動要約（Geminiの場合のみ）
                            if active_llm == LlmType::Gemini {
                                let repo_clone = Arc::clone(&self.memory_repo);
                                let conv_id = self.conversation_id.clone();
                                let gemini_provider = GeminiProvider::default();
                                std::thread::spawn(move || {
                                    if let Err(e) = crate::memory::summarizer::check_and_summarize(
                                        &repo_clone,
                                        &conv_id,
                                        |prompt| gemini_provider.request_summarize(prompt),
                                    ) {
                                        eprintln!("[Memory] 要約処理エラー: {}", e);
                                    }
                                });
                            }

                            let reply_clone = reply.clone();
                            let px_clone = Arc::clone(&self.px);
                            let tts_engine = self.selected_tts;
                            std::thread::spawn(move || {
                                let lang = crate::tts::detect_language(&reply_clone);
                                match crate::tts::generate_tts_file(&reply_clone, lang, tts_engine)
                                {
                                    Ok(path) => {
                                        {
                                            let mut px = px_clone.lock().unwrap();
                                            if let Err(e) = px.play_file(&path, 1.0) {
                                                eprintln!("⚠ 音声ファイルの再生に失敗: {}", e);
                                            }
                                        }
                                        if let Err(e) = std::fs::remove_file(&path) {
                                            eprintln!("⚠ 一時音声ファイルの削除に失敗: {}", e);
                                        }
                                    }
                                    Err(e) => eprintln!("⚠ 音声合成に失敗: {}", e),
                                }
                            });

                            self.app_mode = AppMode::ExecutingLlmCommands {
                                queue: CommandQueue::new(commands),
                            };
                        }
                        LlmResponse::Err { message, .. } => {
                            println!(
                                "[LLM ({:?}, request_id={})] エラー: {}",
                                active_llm, active_request_id, message
                            );
                            self.app_mode = AppMode::Manual;
                        }
                    },
                    Ok(None) => {}
                    Err(error) => {
                        println!("[LLM ({:?})] 応答ワーカーエラー: {}", active_llm, error);
                        self.px.lock().unwrap().stop();
                        self.app_mode = AppMode::Manual;
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
    println!(
        "  ・テキスト欄をクリックして自然言語で指示を入力、Enterで送信 (LLM経由でコマンド実行)"
    );
    println!("  ・Spaceキー長押し (またはボタン長押し) : 音声で指示を入力 (離すと送信)");
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
