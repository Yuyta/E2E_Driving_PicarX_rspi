use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use serde::Deserialize;
use super::command::Command;
use super::LlmResponse;
use std::time::Duration;

const SYSTEM_PROMPT: &str = r#"You are a command interpreter for a PiCar-X robot car.

Respond ONLY with a JSON object matching the provided schema.
Do not output markdown or explanations.

For every command, include ALL parameters required by that command.

Required parameters:
- forward: action, speed, duration_ms
- backward: action, speed, duration_ms
- steer: action, angle, duration_ms
- stop: action only
- set_cam_pan: action, angle
- set_cam_tilt: action, angle
- get_distance: action only
- get_grayscale: action only
- get_battery_status: action only

If the user specifies a duration (e.g., "10s", "5 seconds", "2秒"), you MUST convert it to milliseconds and set it in "duration_ms" (e.g., "10s" -> 10000).

If the user does not specify:
- speed, use 30
- duration_ms, use 1000
- steering angle, use ±30 degrees as appropriate.

Safety limits:
- speed: 0–60
- duration_ms: 100–30000
- angle: -90–90

Never omit a required field."#;

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Vec<GeminiResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponseCandidate {
    content: Option<GeminiResponseContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePayload {
    candidates: Vec<GeminiResponseCandidate>,
}

#[derive(Debug, Deserialize)]
struct CommandListPayload {
    commands: Vec<Command>,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorBody {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    code: i32,
    message: String,
    status: String,
}

/// LLM(Gemini API)との通信を専用スレッドで行い、メインループをブロックしない。
pub struct GeminiAgent {
    prompt_tx: Sender<String>,
    response_rx: Receiver<LlmResponse>,
}

impl GeminiAgent {
    /// `.env`ファイルまたは環境変数から `GEMINI_API_KEY` を読み込み、ワーカースレッドを起動する。
    pub fn new() -> Self {
        // 1. 通常の dotenv 読み込み (カレントディレクトリから親へ探索)
        let _ = dotenvy::dotenv();

        // 2. 実行ファイルのあるディレクトリの .env 読み込みを試みる
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let _ = dotenvy::from_path(exe_dir.join(".env"));
            }
        }

        // 3. 開発環境向け (プロジェクトルートから起動された場合)
        let _ = dotenvy::from_path("rust/.env");

        let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();

        let (prompt_tx, prompt_rx) = mpsc::channel::<String>();
        let (response_tx, response_rx) = mpsc::channel::<LlmResponse>();

        thread::spawn(move || {
            for prompt in prompt_rx {
                let result = Self::call_gemini(&api_key, &prompt);
                if response_tx.send(result).is_err() {
                    break;
                }
            }
        });

        Self { prompt_tx, response_rx }
    }

    /// プロンプトをワーカースレッドへ送るだけの非ブロッキング呼び出し。
    pub fn submit_prompt(&self, prompt: String) {
        let _ = self.prompt_tx.send(prompt);
    }

    /// レスポンスが届いていれば取り出す。届いていなければ`None`(非ブロッキング)。
    pub fn try_recv_response(&self) -> Option<LlmResponse> {
        self.response_rx.try_recv().ok()
    }

    fn call_gemini(api_key: &str, prompt: &str) -> LlmResponse {
        if api_key.is_empty() || api_key == "YOUR_GEMINI_API_KEY_HERE" {
            return LlmResponse::Err(
                "GEMINI_API_KEY is not set. Please set your API key in the .env file (same directory as the executable) or set it as a GEMINI_API_KEY environment variable."
                    .to_string(),
            );
        }

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={}",
            api_key
        );

        // Structured Outputs 用の JSON Schema 定義
        let response_schema = serde_json::json!({
            "type": "OBJECT",
            "properties": {
                "commands": {
                    "type": "ARRAY",
                    "description": "List of robot commands to execute in sequence.",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "action": {
                                "type": "STRING",
                                "enum": [
                                    "forward",
                                    "backward",
                                    "steer",
                                    "stop",
                                    "set_cam_pan",
                                    "set_cam_tilt",
                                    "get_distance",
                                    "get_grayscale",
                                    "get_battery_status"
                                ]
                            },
                            "speed": {
                                "type": "INTEGER",
                                "minimum": 0,
                                "maximum": 60
                            },
                            "duration_ms": {
                                "type": "INTEGER",
                                "minimum": 100,
                                "maximum": 30000
                            },
                            "angle": {
                                "type": "INTEGER",
                                "minimum": -90,
                                "maximum": 90
                            }
                        },
                        "required": ["action", "duration_ms"]
                    }
                }
            },
            "required": ["commands"]
        });

        let request_body = serde_json::json!({
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "systemInstruction": {
                "parts": [{
                    "text": SYSTEM_PROMPT
                }]
            },
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        });

        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(10))
            .build();

        let result = agent
            .post(&url)
            .send_json(&request_body)
            .map_err(|e| {
                match e {
                    ureq::Error::Status(_, resp) => {
                        if let Ok(err) = resp.into_json::<GeminiErrorBody>() {
                            format!(
                                "Gemini API Error {} ({}): {}",
                                err.error.code,
                                err.error.status,
                                err.error.message
                            )
                        } else {
                            "Gemini APIがHTTPエラーを返しました。".to_string()
                        }
                    }
                    _ => format!("Gemini APIへの通信に失敗しました: {}", e),
                }
            })
            .and_then(|resp| {
                resp.into_json::<GeminiResponsePayload>()
                    .map_err(|e| format!("Gemini APIのレスポンス解析に失敗: {}", e))
            })
            .and_then(|gemini_resp| {
                // candidateが存在するか確認
                let candidate = gemini_resp
                    .candidates
                    .first()
                    .ok_or_else(|| {
                        "Gemini APIから回答（candidates）が得られませんでした。".to_string()
                    })?;

                // finishReasonを確認
                if let Some(reason) = candidate.finish_reason.as_deref() {
                    if reason != "STOP" {
                        return Err(format!(
                            "Geminiが正常終了しませんでした。finishReason={}",
                            reason
                        ));
                    }
                }

                // contentが存在するか確認
                let content_text = candidate.content.as_ref()
                    .ok_or_else(|| {
                        "Gemini APIのレスポンスがありません。".to_string()
                    })?;

                // partsが存在するか確認
                let part_text = content_text.parts.first()
                    .ok_or_else(|| {
                        "Gemini APIのレスポンス（parts）が空です。".to_string()
                    })?;

                // Debug用に受信内容をログ出力
                println!("[DEBUG] Gemini API Response: {}", part_text.text);

                // まずJSONをValueとして読み込む
                let mut value: serde_json::Value =
                    serde_json::from_str(&part_text.text)
                        .map_err(|e| {
                            format!(
                                "LLMが返したJSONの解析に失敗: {} (受信内容: {})",
                                e, part_text.text
                            )
                        })?;

                // 必須項目を補完
                if let Some(commands) = value["commands"].as_array_mut() {
                    for cmd in commands {
                        let action = cmd
                            .get("action")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");

                        match action {
                            "forward" | "backward" | "steer" => {
                                if cmd.get("duration_ms").is_none() {
                                    cmd["duration_ms"] = serde_json::json!(1000);
                                }
                            }
                            _ => {}
                        }
                    }
                }

                // CommandListPayloadへ変換
                serde_json::from_value::<CommandListPayload>(value)
                    .map_err(|e| {
                        format!(
                            "LLMが返したコマンドJSONの解析に失敗: {} (受信内容: {})",
                            e, part_text.text
                        )
                    })


            });

        match result {
            Ok(payload) => LlmResponse::Ok(payload.commands),
            Err(e) => LlmResponse::Err(e),
        }
    }
}
