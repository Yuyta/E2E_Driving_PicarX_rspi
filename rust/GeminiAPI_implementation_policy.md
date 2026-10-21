# Gemini API 統合実装方針

`llm_agent` の共通処理を `common.rs` に集約し、Ollama と Gemini のプロバイダ実装を同じ非同期エージェントとして扱うための実装方針です。

---

## 1. 基本方針

- **プロバイダと共通処理の分離**:
  - `common.rs` に共通JSON Schema、要求キュー、要求ID、レスポンス受信、JSON parser を集約します。
  - `ollama.rs` と `gemini.rs` は、それぞれのAPI通信と設定だけを担当します。
- **データ構造の再利用**:
  - 同一フォルダ内の `Command`（コマンド列挙型）と `CommandQueue`（非ブロッキング実行キュー）をインポートしてそのまま再利用し、重複コードを最小限に抑えます。
- **安全な画面処理の維持**:
  - 画面をフリーズさせない「非ブロッキング方式（`CommandQueue::tick`）」で動作させます。

---

## 2. Structured Outputs による型安全の確保

Gemini API の `generationConfig.responseSchema` に Rust 側の `Command` と同一の JSON Schema を指定し、型安全な出力を強制します。

コマンドSchema本体は標準JSON Schemaとして `common.rs` に定義します。OllamaはこのSchemaを `format` にそのまま渡し、GeminiはAPI固有の型表記へ変換して `responseSchema` に渡します。どちらのプロバイダも、最終的には共通parserで再検証します。

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
        "required": ["action"]
      }
    }
  },
  "required": ["commands"]
}
```

### 実装・運用上の注意点 (エラー回避の知見)

1. **`duration_ms` の補完と安全値検証**:
   * Geminiが移動コマンドの `duration_ms` を省略しても、共通parserが `1000ms` を補完します。
   * 共通parserで `duration_ms` を `100..30000ms`、速度を `0..100`、角度を `-90..90` に制限します。
2. **複雑なスキーマ制限 (`anyOf` の回避)**:
   * Gemini APIのJSON Schemaパーサーでは、`anyOf` を使用した複雑なオブジェクト型定義や `"const"` キーワードの指定がエラー（`INVALID_ARGUMENT`）になる制限があります。スキーマ構造は極力シンプルに保つ必要があります。
3. **`maxOutputTokens` の設定値（1024以上）**:
   * 複数の連続した指示（例：「10秒進んで、左に曲がって、1秒進んで、止まる」）を送ると生成されるJSONが長くなります。
   * トークン上限が小さい（例: 256）と途中で出力が切れてエラー（`finishReason=MAX_TOKENS`）になるため、`maxOutputTokens` は `1024` 以上に設定します。
4. **システムプロンプトによる指示の補強**:
   * 「`10s`」「`5秒`」などの指示をミリ秒（`10000`, `5000`）へ正確に単位変換させるため、システムプロンプト側でも変換ルールを具体例つきで明示します。
5. **ネストしたセンサー条件**:
   * `and` / `or` / `xor` / `not` を使った条件を許可します。
   * GeminiのresponseSchemaは`$defs`/`$ref`で再帰的なConditionを定義します。Rust側のparserでも深さと配列要素数を検証します。

---

## 3. ディレクトリ・ファイル構成

```text
rust/
  ├── Cargo.toml                  # [MODIFY] dotenvy の依存関係を追加
  ├── .env                        # [NEW] GEMINI_API_KEY を定義
  └── src/
      └── llm_agent/
          ├── mod.rs              # プロバイダと共通型を公開
          ├── command.rs          # コマンド型と実行キュー
          ├── common.rs           # 共通Schema、エージェント、parser
          ├── ollama.rs           # Ollama固有のAPI通信
          └── gemini.rs           # Gemini固有のAPI通信
```

### `gemini.rs` の役割と実装イメージ

`ProviderAgent<GeminiProvider>` として、共通のバックグラウンドスレッドから Gemini API と同期通信（`ureq`）を行います。応答には要求IDが付与され、画面側で現在の要求以外を破棄します。

```rust
pub type GeminiAgent = ProviderAgent<GeminiProvider>;
```

---

## 4. 動作検証手順

1. **依存関係追加とビルド確認**:
   `Cargo.toml` に `dotenvy` を追加した上で `cargo check` を実行。
3. **キー設定と実行**:
   `rust/.env` に `GEMINI_API_KEY` を設定し、`cargo run` を実行。
4. **接続テスト**:
   `picarx_sample.rs` のインポート先を `GeminiAgent` に切り替えることで、Gemini API による安全な非ブロッキング操作を確認します。

