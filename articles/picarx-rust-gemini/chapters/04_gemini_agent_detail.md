# 第4章：実装詳細：`gemini_agent.rs` の詳細解説


## 構造化出力のためのリクエスト構造

本プロジェクトでGemini APIと通信する心臓部が [gemini_agent.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/gemini_agent.rs) です。
Geminiに対して構造化出力を指示するには、リクエストの送信時に「JSONを要求すること」と「JSONの具体的なスキーマ」の2つを設定します。

具体的には、以下のようにリクエストボディを構築します。

```rust
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
                        "description": "Streering angle in degrees. Can be used with forward/backward commands to turn while moving.",
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
```

#### ポイント解説
- **JSON Schemaの定義**: `response_schema` にて、最上位が `commands` という配列を持つオブジェクトであることを宣言しています。配列の各要素は `action` (列挙型), `speed`, `duration_ms`, `angle` を持つオブジェクトです。
- **値の範囲指定**: `minimum` / `maximum` を用いて、ロボットのモーターに悪影響を与えないスピード範囲（0〜60）や、サーボモーターの回転角（-90〜90）をAPIレベルで制限しています。
- **`generationConfig` の指定**: `responseMimeType` に `"application/json"` を指定し、`responseSchema` に先ほど定義した `response_schema` を渡すことで、Geminiにこのスキーマ通りのJSONを出力させます。

---

## Rustの `serde` による型マッピング

Geminiから返ってきたJSONを受け取る側の、Rustの構造体（型）定義を見てみましょう。
[command.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/command.rs) にて、以下のように型が定義されています。

```rust
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Command {
    Forward {
        speed: i32,
        duration_ms: u64,
        angle: Option<i32>,
        #[serde(default)]
        until: Option<ConditionExpr>,
    },
    Backward {
        speed: i32,
        duration_ms: u64,
        angle: Option<i32>,
        #[serde(default)]
        until: Option<ConditionExpr>,
    },
    Steer {
        angle: i32,
        duration_ms: u64,
        #[serde(default)]
        until: Option<ConditionExpr>,
    },
    Stop,
    SetCamPan { angle: i32 },
    SetCamTilt { angle: i32 },
    GetDistance,
    GetGrayscale,
    GetBatteryStatus,
}
```

#### ポイント解説
- **`serde(tag = "action")`**: JSONの `action` キーの値（`forward` や `backward` など）を見て、自動的に対応する enum のバリアントにマッピングします。
- **`rename_all = "snake_case"`**: JSONのキー名とRustのフィールド名を自動で変換してマッピングします。
- **堅牢なデシリアライズ**: Geminiの構造化出力によって `action` と `duration_ms` が必ず含まれることが保証されているため、パース時に項目不足エラーでプログラムが落ちる（クラッシュする）のをほぼ完全に防ぐことができます。

---

## メインループをブロックしない非同期アーキテクチャ

ラズパイカーは、走っている間も常にカメラの映像を描画したり、周囲のセンサー値をスキャンし続けたりしなければなりません。
もし、LLMのAPIリクエストをメインスレッドで実行してしまうと、AIが考えている数秒間、ロボットの画面描画やセンサー監視が完全に停止（フリーズ）してしまい、誤動作の原因になります。

そのため、[gemini_agent.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/gemini_agent.rs) ではスレッドとチャネルを用いた非同期処理を採用しています。

```rust
pub struct GeminiAgent {
    prompt_tx: Sender<String>,
    response_rx: Receiver<LlmResponse>,
}

impl GeminiAgent {
    pub fn new() -> Self {
        let api_key = std::env::var("GEMINI_API_KEY").unwrap_or_default();

        let (prompt_tx, prompt_rx) = mpsc::channel::<String>();
        let (response_tx, response_rx) = mpsc::channel::<LlmResponse>();

        // HTTP通信を行うバックグラウンド（専用）スレッドを起動
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

    /// メインスレッドからプロンプトを送る（非ブロッキング）
    pub fn submit_prompt(&self, prompt: String) {
        let _ = self.prompt_tx.send(prompt);
    }

    /// メインスレッドから返事を取り出す（非ブロッキング）
    pub fn try_recv_response(&self) -> Option<LlmResponse> {
        self.response_rx.try_recv().ok()
    }
}
```

#### ポイント解説
- **`thread::spawn`**: Gemini APIとの重たいHTTP通信は、すべてバックグラウンドの別スレッド内で処理されます。
- **`mpsc::channel`**: メインスレッドと通信用スレッドの間で、プロンプトの「送信（`prompt_tx`）」とレスポンスの「受信（`response_rx`）」をスレッド安全に行います。
- **`try_recv()`**: メインループの中で「AIからの応答が届いているか？」を判定する際、`try_recv`（非ブロッキング呼び出し）を使うことで、応答がまだ届いていなくても処理を止めずに、そのままループを回し続けることができます。
