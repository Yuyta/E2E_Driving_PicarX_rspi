# Gemini API 統合実装方針

既存の `llm_agent` モジュールのコード（Ollama用の `core.rs` 等）に変更を加えず、新しく `rust/src/llm_agent/gemini_agent.rs` を配置して Gemini API の Structured Outputs（型安全JSON出力）を統合するための実装方針です。

---

## 1. 基本方針

- **モジュール内同居による整理**:
  - `llm_agent` フォルダ内に `gemini_agent.rs` を追加し、LLM関連 of 処理を1箇所にまとめます。
  - Ollama用（`core.rs`）の既存コードには一切変更を加えません。
- **データ構造の再利用**:
  - 同一フォルダ内の `Command`（コマンド列挙型）と `CommandQueue`（非ブロッキング実行キュー）をインポートしてそのまま再利用し、重複コードを最小限に抑えます。
- **安全な画面処理の維持**:
  - 画面をフリーズさせない「非ブロッキング方式（`CommandQueue::tick`）」で動作させます。

---

## 2. Structured Outputs による型安全の確保

Gemini API の `generationConfig.responseSchema` に Rust 側の `Command` と同一の JSON Schema を指定し、型安全な出力を強制します。

### スキーマ定義の概要

```json
{
  "type": "OBJECT",
  "properties": {
    "commands": {
      "type": "ARRAY",
      "items": {
        "type": "OBJECT",
        "properties": {
          "action": {
            "type": "STRING",
            "enum": [
              "forward", "backward", "steer", "stop", 
              "set_cam_pan", "set_cam_tilt", 
              "get_distance", "get_grayscale", "get_battery_status"
            ]
          },
          "speed": { "type": "INTEGER" },
          "duration_ms": { "type": "INTEGER" },
          "angle": { "type": "INTEGER" }
        },
        "required": ["action", "duration_ms"]
      }
    }
  },
  "required": ["commands"]
}
```

### 実装・運用上の注意点 (エラー回避の知見)

1. **`duration_ms` の必須化（requiredへの追加）**:
   * Geminiはデフォルトでパラメータを省略する傾向があります。「`10s`」などの時間指示があっても省略されるのを防ぐため、`required` に `"duration_ms"` を指定して出力を強制します。
   * 移動以外のコマンド（`stop`など）で `duration_ms` が出力された場合でも、Rustのデシリアライズ側で安全に無視されるため動作上の問題はありません。
2. **複雑なスキーマ制限 (`anyOf` の回避)**:
   * Gemini APIのJSON Schemaパーサーでは、`anyOf` を使用した複雑なオブジェクト型定義や `"const"` キーワードの指定がエラー（`INVALID_ARGUMENT`）になる制限があります。スキーマ構造は極力シンプルに保つ必要があります。
3. **`maxOutputTokens` の設定値（1024以上）**:
   * 複数の連続した指示（例：「10秒進んで、左に曲がって、1秒進んで、止まる」）を送ると生成されるJSONが長くなります。
   * トークン上限が小さい（例: 256）と途中で出力が切れてエラー（`finishReason=MAX_TOKENS`）になるため、`maxOutputTokens` は `1024` 以上に設定します。
4. **システムプロンプトによる指示の補強**:
   * 「`10s`」「`5秒`」などの指示をミリ秒（`10000`, `5000`）へ正確に単位変換させるため、システムプロンプト側でも変換ルールを具体例つきで明示します。

---

## 3. ディレクトリ・ファイル構成

```text
rust/
  ├── Cargo.toml                  # [MODIFY] dotenvy の依存関係を追加
  ├── .env                        # [NEW] GEMINI_API_KEY を定義
  └── src/
      └── llm_agent/
          ├── mod.rs              # [MODIFY] pub mod gemini_agent; を追加
          ├── command.rs          # 変更なし
          ├── core.rs             # 変更なし（Ollama用の既存コード）
          └── gemini_agent.rs     # [NEW] GeminiAgent構造体とAPI通信を定義
```

### `gemini_agent.rs` の役割と実装イメージ

`LlmAgent` と同様に、バックグラウンドスレッドで Gemini API と同期通信（`ureq`）を行います。

```rust
pub enum GeminiResponse {
    Ok(Vec<super::command::Command>),
    Err(String),
}

pub struct GeminiAgent {
    prompt_tx: std::sync::mpsc::Sender<String>,
    response_rx: std::sync::mpsc::Receiver<GeminiResponse>,
}
```

---

## 4. 動作検証手順

1. **依存関係追加とビルド確認**:
   `Cargo.toml` に `dotenvy` を追加した上で `cargo check` を実行。
3. **キー設定と実行**:
   `rust/.env` に `GEMINI_API_KEY` を設定し、`cargo run` を実行。
4. **接続テスト**:
   `picarx_sample.rs` のインポート先を `GeminiAgent` に切り替えることで、Gemini API による安全な非ブロッキング操作を確認します。
