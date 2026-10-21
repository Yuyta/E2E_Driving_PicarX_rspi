# 会話メモリ機能の実装計画

PiCar-XのRust + Gemini APIアプリケーションに、会話履歴・要約・長期記憶を用いたコンテキスト管理機能を追加する。

## 現状の構造

```text
src/
├── main.rs                    # エントリポイント
├── llm_agent/
│   ├── mod.rs                 # LlmType, pub use
│   ├── common.rs              # SYSTEM_PROMPT, LlmProvider trait, ProviderAgent, コマンドパース
│   ├── gemini.rs              # GeminiProvider (API呼び出し、スキーマ定義)
│   ├── command.rs             # Command enum, CommandQueue
│   └── ollama.rs              # OllamaProvider
├── sample/
│   └── picarx_sample.rs       # GUIアプリ本体 (PiCarApp, AppMode)
├── tts.rs
├── audio_input.rs
├── picarx/                    # ハードウェア制御
├── rpi_camera/
└── sensor_monitor/
```

**現在のGemini API呼び出しの特徴:**
- `SYSTEM_PROMPT` は `common.rs` にハードコード（ロボットコマンドインタプリタとしての指示）
- `gemini.rs` の `send_generate_content()` で `systemInstruction` + `contents` を送信
- レスポンスは `{ "commands": [...], "reply": "..." }` のJSON
- 会話履歴は一切保持されていない（毎回単発リクエスト）
- `LlmProvider` trait は `fn request(&self, prompt: &str) -> Result<String, String>` というシンプルなインターフェース

---

## User Review Required

> [!IMPORTANT]
> **Gemini レスポンススキーマへの `memory_updates` 追加について**
> 現在のレスポンスは `{ "commands": [...], "reply": "..." }` ですが、`memory_updates` フィールドを追加します。Gemini の `responseSchema` に `memory_updates` を追加しますが、`required` には含めず **optional** とします。これにより、既存のコマンド実行フローへの影響を最小限に抑えます。

> [!WARNING]
> **Ollama プロバイダーへのメモリ機能適用**
> 今回の実装では **Gemini プロバイダーのみ** にメモリ機能を適用します。Ollama は現在のまま（単発リクエスト）とします。将来的にOllamaにも対応する場合は `LlmProvider` trait を拡張する必要があります。

> [!IMPORTANT]
> **SYSTEM_PROMPT の変更について**
> 現在の `SYSTEM_PROMPT` に以下を追記します:
> - 「ユーザーの名前やロボットの名前が Long-term Memory に記載されている場合、それを使って応答すること」
> - 「`memory_updates` フィールドで長期記憶すべき情報を返却すること」
>
> 追記はプロンプト末尾に行い、既存のコマンド生成指示には一切手を加えません。既存テストが壊れないことを確認します。

## Open Questions

> [!IMPORTANT]
> **会話の初期化タイミング**
> アプリ起動時に新しい `conversation_id` を生成する想定ですが、前回の会話を引き継ぐ選択肢も必要ですか？
> 現在の計画: **アプリ起動ごとに新しい conversation_id を生成**（Long-term Memory は conversation を跨いで永続化）

> [!IMPORTANT]
> **SQLite DBファイルの保存場所**
> Raspberry Pi 上の `/var/lib/picarx/memory.db` または実行ファイルと同じディレクトリ (`./picarx_memory.db`) のどちらが望ましいですか？
> 現在の計画: **実行ファイルと同じディレクトリ** (`./picarx_memory.db`)

---

## Proposed Changes

### 依存関係の追加

#### [MODIFY] [Cargo.toml](file:///c:/Users/tacta/Dev/PiCarX/rust/Cargo.toml)

`rusqlite` を追加:

```toml
rusqlite = { version = "0.34", features = ["bundled"] }
```

`bundled` feature により、Raspberry Pi 上でもSQLiteのシステムライブラリに依存せずビルドできる。

---

### 新規モジュール: `memory/`

#### [NEW] [mod.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/memory/mod.rs)

```rust
pub mod model;
pub mod repository;
pub mod summarizer;
```

#### [NEW] [model.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/memory/model.rs)

データ構造定義:

```rust
/// DBに保存するメッセージ
pub struct Message {
    pub id: i64,
    pub conversation_id: String,
    pub role: String,        // "user" | "assistant"
    pub content: String,
    pub created_at: String,  // ISO 8601
    pub summarized: bool,
}

/// 会話要約
pub struct ConversationSummary {
    pub id: i64,
    pub conversation_id: String,
    pub summary: String,
    pub updated_at: String,
}

/// 長期記憶エントリ
pub struct Memory {
    pub id: i64,
    pub key: String,         // e.g. "user.name", "robot.name", "user_preferences.language"
    pub value: String,       // JSON value
    pub updated_at: String,
}
```

#### [NEW] [repository.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/memory/repository.rs)

SQLite操作:

- `MemoryRepository::new(db_path)` — DB接続 + テーブル作成 (IF NOT EXISTS)
- `save_message(conversation_id, role, content)` — メッセージ保存
- `get_recent_messages(conversation_id, limit)` — 直近N件取得
- `get_unsummarized_count(conversation_id)` — 未要約件数
- `get_unsummarized_messages(conversation_id)` — 未要約メッセージ一覧
- `mark_as_summarized(message_ids)` — 要約済みフラグ更新
- `upsert_summary(conversation_id, summary)` — 要約の保存/更新
- `get_summary(conversation_id)` — 要約取得
- `upsert_memory(key, value)` — 長期記憶の保存/更新
- `get_all_memories()` — 全長期記憶取得
- `delete_memory(key)` — 長期記憶の削除

**テーブル定義:**

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    summarized INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS conversation_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL UNIQUE,
    summary TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL UNIQUE,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
```

#### [NEW] [summarizer.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/memory/summarizer.rs)

要約ロジック:

- `check_and_summarize(repo, conversation_id, summarize_fn)` — 未要約12件以上で要約実行
- 要約には **Gemini API を直接呼び出す**（コマンドスキーマなしのテキスト生成）
- 既存Summaryがある場合は統合プロンプトで更新

```text
要約プロンプト例:
「以下の会話を簡潔に要約してください。ユーザーの名前、ロボットの名前、
 重要な設定変更、繰り返されるパターンなどを必ず含めてください。
 既存の要約: {existing_summary}
 新しい会話: {messages}」
```

---

### LLM Agent の変更

#### [MODIFY] [common.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/common.rs)

**変更点 (最小限):**

1. `SYSTEM_PROMPT` の末尾に追記:

```text
Memory and Personalization Rules:
- If a "Long-term Memory" section is provided in the context, use names and preferences stored there (e.g. user's name, robot's name, default speed).
- If the user tells you their name, the robot's name, or any persistent preference, include a "memory_updates" object in your response to save it.
- memory_updates format: { "key": "value" } where key uses dot notation (e.g. "user.name", "robot.name", "robot.default_speed").
- To delete a memory, set its value to null.
- Only include memory_updates when the user provides new persistent information.
```

2. `CommandListPayload` に `memory_updates` フィールドを追加:

```rust
#[derive(Deserialize)]
struct CommandListPayload {
    commands: Vec<Command>,
    reply: String,
    #[serde(default)]
    memory_updates: Option<serde_json::Map<String, Value>>,
}
```

3. `LlmResponse::Ok` に `memory_updates` を追加:

```rust
pub enum LlmResponse {
    Ok {
        request_id: u64,
        commands: Vec<Command>,
        reply: String,
        memory_updates: Option<serde_json::Map<String, Value>>,
    },
    ...
}
```

4. `LlmProvider` trait に新メソッド追加 (デフォルト実装あり):

```rust
pub trait LlmProvider: Send + 'static {
    fn request(&self, prompt: &str) -> Result<String, String>;
    fn request_audio(&self, _wav_data: &[u8]) -> Result<String, String> { ... }

    /// メモリコンテキスト付きリクエスト（デフォルトはコンテキスト無視でrequest呼び出し）
    fn request_with_context(&self, prompt: &str, context: &str) -> Result<String, String> {
        self.request(prompt)
    }
    fn request_audio_with_context(&self, wav_data: &[u8], context: &str) -> Result<String, String> {
        self.request_audio(wav_data)
    }
}
```

> [!NOTE]
> デフォルト実装により **Ollama プロバイダーは変更不要** です。既存の `request()` がそのまま呼ばれます。

5. `_parse_commands_for_prompt` の返り値を拡張し、`memory_updates` も返すように変更:

```rust
pub fn _parse_commands_for_prompt(
    response_text: &str,
    user_prompt: &str,
) -> Result<(Vec<Command>, String, Option<serde_json::Map<String, Value>>), String>
```

> [!WARNING]
> **既存テストへの影響:** `_parse_commands` と `_parse_commands_for_prompt` の返り値がタプル3要素になるため、既存テストの全パターンマッチを `(commands, reply, _memory_updates)` に更新する必要があります。ロジック自体は変わらないため、破壊的変更にはなりません。

#### [MODIFY] [gemini.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/gemini.rs)

**変更点:**

1. `command_response_schema()` に `memory_updates` を追加:

```json
"memory_updates": {
    "type": "OBJECT",
    "description": "Key-value pairs of long-term memory to save. Use dot notation for keys.",
    "nullable": true
}
```

`required` には追加しない（optional のまま）。

2. `request_with_context` を実装:

```rust
impl LlmProvider for GeminiProvider {
    fn request_with_context(&self, prompt: &str, context: &str) -> Result<String, String> {
        let prompt_text = gemini_prompt_with_context(prompt, context);
        self.send_generate_content(vec![serde_json::json!({ "text": prompt_text })])
    }

    fn request_audio_with_context(&self, wav_data: &[u8], context: &str) -> Result<String, String> {
        // audio_prompt_guide にコンテキストを注入
        ...
    }
}
```

3. `gemini_prompt_with_context()` — メモリ・要約・会話履歴をプロンプトに注入:

```text
{SYSTEM_PROMPT}

Gemini-specific output rules:
...

--- Context ---
{Long-term Memory (JSON)}
{Conversation Summary}
--- Recent Conversation ---
User: ...
Assistant: ...
---

User request: {prompt}
```

4. `send_generate_content` の `contents` を **multi-turn形式** に変更する選択肢もあるが、初期実装ではシステムプロンプト内にテキストとして埋め込む方式を採用（既存構造への影響を最小化）。

#### [MODIFY] [mod.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/mod.rs)

`memory_updates` を `LlmResponse` 経由で公開するため、re-export の変更のみ。

---

### アプリ本体の変更

#### [MODIFY] [picarx_sample.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/sample/picarx_sample.rs)

**変更点:**

1. `PiCarApp` に `MemoryRepository` と `conversation_id` を追加:

```rust
struct PiCarApp {
    // ... 既存フィールド ...
    memory_repo: Arc<crate::memory::repository::MemoryRepository>,
    conversation_id: String,
}
```

2. `PiCarApp::new()` で DB初期化 + conversation_id 生成

3. テキスト送信時 (`submitted` ブロック):
   - ユーザーメッセージを `memory_repo.save_message()` で保存
   - `memory_repo` からコンテキスト情報を取得
   - `submit_prompt` の代わりに `submit_prompt_with_context` を呼び出し

4. レスポンス受信時 (`LlmResponse::Ok` ブロック):
   - Assistant応答を `memory_repo.save_message()` で保存
   - `memory_updates` があれば **Rust側でバリデーション** してから `memory_repo.upsert_memory()` で保存
   - 要約チェック (`check_and_summarize`) を実行

5. `ProviderAgent` に `submit_prompt_with_context` メソッドを追加:

```rust
pub fn submit_prompt_with_context(&self, text: String, context: String) -> Result<u64, String>
```

---

### メインエントリポイント

#### [MODIFY] [main.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/main.rs)

`mod memory;` を追加するのみ。

---

## Memory Updates のバリデーション

Gemini が返す `memory_updates` をそのまま保存せず、Rust側で以下を検証する:

| チェック項目 | 詳細 |
|---|---|
| キーのホワイトリスト | `user.name`, `robot.name`, `robot.default_speed`, `robot.max_speed`, `user_preferences.*`, `behavior.*` のみ許可 |
| 値の型チェック | `speed` 系は数値かつ 0–100 の範囲内、`name` 系は文字列かつ50文字以内 |
| `null` 値は削除として処理 | `{ "user.name": null }` → `delete_memory("user.name")` |
| 不正なキーは無視 | ログ出力して警告するが処理は続行 |

---

## Context 構築の具体例

Gemini に送信されるテキストの全体像:

```text
[System Prompt] (既存 SYSTEM_PROMPT + Memory Rules 追記)

[Gemini-specific output rules]

--- Long-term Memory ---
{
  "user.name": "太郎",
  "robot.name": "ピカちゃん",
  "robot.default_speed": 30
}

--- Conversation Summary ---
ユーザー（太郎）はロボット（ピカちゃん）に前進・後退の基本操作を何度か指示した。
デフォルト速度は30に設定されている。

--- Recent Conversation ---
[User]: 右に曲がって
[Assistant]: はい、右に曲がります！
[User]: もう少し速く
[Assistant]: 速度を上げて進みます。
[User]: ありがとう、ピカちゃん
[Assistant]: どういたしまして、太郎さん！
---

User request: さっきと同じように前に進んで
```

---

## ファイル変更サマリー

| ファイル | 変更種別 | 変更の大きさ |
|---|---|---|
| `Cargo.toml` | MODIFY | 小 (1行追加) |
| `src/main.rs` | MODIFY | 小 (1行追加) |
| `src/memory/mod.rs` | NEW | 小 |
| `src/memory/model.rs` | NEW | 中 |
| `src/memory/repository.rs` | NEW | 大 (SQLite操作全般) |
| `src/memory/summarizer.rs` | NEW | 中 |
| `src/llm_agent/common.rs` | MODIFY | 中 (SYSTEM_PROMPT追記、返り値拡張) |
| `src/llm_agent/gemini.rs` | MODIFY | 中 (context対応、スキーマ追加) |
| `src/llm_agent/mod.rs` | MODIFY | 小 |
| `src/sample/picarx_sample.rs` | MODIFY | 中 (DB初期化、メモリ連携) |

---

## 既存機能への影響分析

| 影響箇所 | リスク | 対策 |
|---|---|---|
| `SYSTEM_PROMPT` 変更 | **低**: 末尾追記のみ。既存コマンド生成指示に触れない | 既存テスト全通過を確認 |
| `_parse_commands` 返り値変更 | **中**: テストのパターンマッチ修正必要 | 3要素タプルに変更、`memory_updates` は `_` で無視可能 |
| `CommandListPayload` 変更 | **低**: `#[serde(default)]` により既存JSON互換 | `memory_updates` がないJSONでも正常デシリアライズ |
| `LlmResponse` 変更 | **中**: マッチパターン修正必要 | `memory_updates` を追加のフィールドとして含める |
| `LlmProvider` trait 変更 | **低**: デフォルト実装でOllamaは変更不要 | Ollama providerのコード変更ゼロ |
| Gemini `responseSchema` 変更 | **低**: optional フィールド追加のみ | `required` に含めない |
| `picarx_sample.rs` 変更 | **中**: DB初期化とメモリ連携追加 | 既存フローの構造を変更せず、DB操作を追加するのみ |

---

## Verification Plan

### Automated Tests

```bash
# 既存テスト全通過の確認
cd rust && cargo test

# 特に以下のテストが壊れないことを確認:
# - common::tests::* (コマンドパース、バリデーション)
# - gemini::tests::* (スキーマ、レスポンス正規化)
# - command::tests::* (コマンド実行ロジック)
```

### 新規テスト

- `memory::repository` のユニットテスト (インメモリSQLite)
  - メッセージの保存・取得・要約フラグ更新
  - 長期記憶のupsert/delete/get
  - 要約の保存・更新・取得
- `memory_updates` のバリデーションテスト
  - ホワイトリスト外キーの拒否
  - 値の範囲チェック
  - null値による削除
- `_parse_commands` が `memory_updates` を正しく返すことのテスト
- `gemini_prompt_with_context` がコンテキストを正しく構築することのテスト

### Manual Verification

- Raspberry Pi上でビルド・実行し、以下を確認:
  - 名前を伝えると長期記憶に保存される
  - ロボットに名前をつけると長期記憶に保存される
  - 次の会話でも名前が反映される
  - 「さっきと同じ」が直近6メッセージで解決される
  - 12メッセージ以上の会話で要約が自動生成される
  - 既存のコマンド実行（前進、後退、カメラ操作等）が正常に動作する
