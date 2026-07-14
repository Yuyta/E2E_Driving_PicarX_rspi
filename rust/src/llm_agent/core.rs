use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use serde::{Deserialize, Serialize};
use super::command::Command;
use super::LlmResponse;

const DEFAULT_OLLAMA_URL: &str = "http://127.0.0.1:11434/api/chat";
const DEFAULT_OLLAMA_MODEL: &str = "llama3.1";

const SYSTEM_PROMPT: &str = r#"You are a command interpreter for a PiCar-X robot car.
Given a natural-language instruction (which may be in Japanese or English), respond with ONLY a JSON object, no prose, no markdown fences.
The object has a single field "commands": an array of command objects. Each command object has an "action" field and action-specific fields:
- {"action": "forward", "speed": <0-100>, "duration_ms": <milliseconds>}
- {"action": "backward", "speed": <0-100>, "duration_ms": <milliseconds>}
- {"action": "steer", "angle": <-90 to 90, negative=left, positive=right>, "duration_ms": <milliseconds>}
- {"action": "stop"}
- {"action": "set_cam_pan", "angle": <-90 to 90>}
- {"action": "set_cam_tilt", "angle": <-90 to 90>}
- {"action": "get_distance"}
- {"action": "get_grayscale"}
- {"action": "get_battery_status"}

Example instruction: "2秒前に進んで止まる" (go forward for 2 seconds then stop)
Example response: {"commands": [{"action": "forward", "speed": 50, "duration_ms": 2000}, {"action": "stop"}]}"#;

#[derive(Serialize)]
struct OllamaMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: Vec<OllamaMessage<'a>>,
    format: &'a str,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaChatResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaChatResponseMessage,
}

#[derive(Deserialize)]
struct CommandListPayload {
    commands: Vec<Command>,
}



/// LLM(Ollama)との通信を専用スレッドで行い、メインループ(カメラ描画)をブロックしない。
pub struct LlmAgent {
    prompt_tx: Sender<String>,
    response_rx: Receiver<LlmResponse>,
}

impl LlmAgent {
    /// `OLLAMA_URL` / `OLLAMA_MODEL` 環境変数からサーバー設定を読み込み、ワーカースレッドを起動する。
    pub fn new() -> Self {
        let ollama_url = std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_URL.to_string());
        let ollama_model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string());

        let (prompt_tx, prompt_rx) = mpsc::channel::<String>();
        let (response_tx, response_rx) = mpsc::channel::<LlmResponse>();

        thread::spawn(move || {
            for prompt in prompt_rx {
                let result = Self::call_ollama(&ollama_url, &ollama_model, &prompt);
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

    fn call_ollama(url: &str, model: &str, prompt: &str) -> LlmResponse {
        let request_body = OllamaChatRequest {
            model,
            messages: vec![
                OllamaMessage { role: "system", content: SYSTEM_PROMPT },
                OllamaMessage { role: "user", content: prompt },
            ],
            format: "json",
            stream: false,
        };

        let result = ureq::post(url)
            .send_json(&request_body)
            .map_err(|e| format!("Ollamaへのリクエストに失敗: {}", e))
            .and_then(|resp| {
                resp.into_json::<OllamaChatResponse>()
                    .map_err(|e| format!("Ollamaのレスポンス解析に失敗: {}", e))
            })
            .and_then(|chat_resp| {
                serde_json::from_str::<CommandListPayload>(&chat_resp.message.content).map_err(|e| {
                    format!(
                        "LLMが返したコマンドJSONの解析に失敗: {} (受信内容: {})",
                        e, chat_resp.message.content
                    )
                })
            });

        match result {
            Ok(payload) => LlmResponse::Ok(payload.commands),
            Err(e) => LlmResponse::Err(e),
        }
    }
}
