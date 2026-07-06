use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use serde::{Deserialize, Serialize};
use super::command::Command;

const DEFAULT_OLLAMA_HOST: &str = "127.0.0.1";
const DEFAULT_OLLAMA_PORT: &str = "11434";
const DEFAULT_OLLAMA_MODEL: &str = "llama3.1";

const SYSTEM_PROMPT: &str = r#"You are a command interpreter for a PiCar-X robot car.
Given a natural-language instruction (which may be in Japanese or English), respond with ONLY a JSON object, no prose, no markdown fences.
The object has a single field "commands": an array of command objects. Each command object has an "action" field and action-specific fields:
- {"action": "forward", "speed": <0-100>, "duration_ms": <milliseconds>, "until": <optional condition, see below>}
- {"action": "backward", "speed": <0-100>, "duration_ms": <milliseconds>, "until": <optional condition>}
- {"action": "steer", "angle": <-90 to 90, negative=left, positive=right>, "duration_ms": <milliseconds>, "until": <optional condition>}
- {"action": "stop"}
- {"action": "set_cam_pan", "angle": <-90 to 90>}
- {"action": "set_cam_tilt", "angle": <-90 to 90>}
- {"action": "get_distance"}
- {"action": "get_grayscale"}
- {"action": "get_battery_status"}

An "until" condition is optional and can be attached to forward/backward/steer. While the action is running,
the sensor is checked continuously; the action stops immediately once the condition becomes true, even before
duration_ms elapses. duration_ms is still required as a safety-bound maximum run time in case the condition
never becomes true (choose a generous value, e.g. 10000-30000ms, when using "until").
A condition is either a single sensor comparison (a "leaf"), or a combination of conditions using a logical operator.
These can be nested arbitrarily.
- Leaf: {"sensor": "distance"|"grayscale_left"|"grayscale_center"|"grayscale_right"|"battery_voltage"|"battery_percent", "op": "eq"|"ne"|"gte"|"lte"|"gt"|"lt", "value": <number>}
- AND (all must be true): {"and": [<condition>, <condition>, ...]}
- OR (at least one must be true): {"or": [<condition>, <condition>, ...]}
- XOR (exactly one must be true): {"xor": [<condition>, <condition>, ...]}
- NOT (inverts one condition): {"not": <condition>}
Vague comparative words like "near"/"close" or "far" in the instruction should be translated into a concrete
"lt"/"gt" (or "lte"/"gte") operator with a reasonable numeric threshold you choose (e.g. distance "lt" 15 for "close").

Example instruction: "2秒前に進んで止まる" (go forward for 2 seconds then stop)
Example response: {"commands": [{"action": "forward", "speed": 50, "duration_ms": 2000}, {"action": "stop"}]}

Example instruction: "前進して、距離が10cm未満になったら止まる" (go forward, and stop as soon as the distance is under 10cm)
Example response: {"commands": [{"action": "forward", "speed": 40, "duration_ms": 15000, "until": {"sensor": "distance", "op": "lt", "value": 10}}]}

Example instruction: "前進して、正面が近いか、左右どちらかのグレイスケールが白線を検知したら止まる" (go forward, and stop once something is close in front, or either the left/right grayscale sensor detects a white line)
Example response: {"commands": [{"action": "forward", "speed": 40, "duration_ms": 15000, "until": {"or": [{"sensor": "distance", "op": "lt", "value": 10}, {"sensor": "grayscale_left", "op": "gt", "value": 3000}, {"sensor": "grayscale_right", "op": "gt", "value": 3000}]}}]}"#;

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

pub enum LlmResponse {
    Ok(Vec<Command>),
    Err(String),
}

/// LLM(Ollama)との通信を専用スレッドで行い、メインループ(カメラ描画)をブロックしない。
pub struct LlmAgent {
    prompt_tx: Sender<String>,
    response_rx: Receiver<LlmResponse>,
}

impl LlmAgent {
    /// `OLLAMA_URL` / `OLLAMA_MODEL` 環境変数からサーバー設定を読み込み、ワーカースレッドを起動する。
    /// `OLLAMA_URL`は `192.168.0.129` のようなホスト(ポート・パス省略可)でも、
    /// `http://192.168.0.129:11434/api/chat` のような完全なURLでも受け付ける。
    pub fn new() -> Self {
        let ollama_host = std::env::var("OLLAMA_URL").unwrap_or_else(|_| DEFAULT_OLLAMA_HOST.to_string());
        let ollama_url = Self::build_chat_url(&ollama_host);
        let ollama_model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| DEFAULT_OLLAMA_MODEL.to_string());
        println!("[LLM] Ollama接続先: {} (model={})", ollama_url, ollama_model);

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

    /// ホスト名/IPだけ("192.168.0.129")でも、ポート付き("192.168.0.129:11434")でも、
    /// 完全なURL("http://192.168.0.129:11434/api/chat")でも受け付け、
    /// Ollamaのchatエンドポイントの完全なURLへ正規化する。
    fn build_chat_url(input: &str) -> String {
        let mut url = input.trim().to_string();
        if url.is_empty() {
            url = DEFAULT_OLLAMA_HOST.to_string();
        }
        if !url.contains("://") {
            url = format!("http://{}", url);
        }

        // スキームより後ろに"/"があれば、既にパスまで指定された完全なURLとみなしそのまま使う
        let after_scheme = url.splitn(2, "://").nth(1).unwrap_or("");
        if after_scheme.contains('/') {
            return url;
        }

        // ポートが無ければデフォルトポートを補う
        if !after_scheme.contains(':') {
            url = format!("{}:{}", url, DEFAULT_OLLAMA_PORT);
        }

        format!("{}/api/chat", url)
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
