# PiCar-X Agentic AI Controller 仕様書

## 1. 文書概要

本書は、ローカルPC上のPython GUIと、Raspberry Pi上のRust APIサーバーで構成されるPiCar-X制御アプリケーションの現行仕様を定義する。

対象システムは、以下の機能を提供する。

- GeminiまたはOllamaを利用した自然言語チャット
- Raspberry Pi上のPiCar-X車体制御
- 超音波センサーによる距離測定
- カメラ画像の自動取得とGUI表示
- Raspberry Pi側のローカルTTSによる音声再生

> [!NOTE]
> - プロジェクト全体の開発ロードマップ・進捗状況は [ROADMAP.md](ROADMAP.md) を参照してください。
> - End-to-End (E2E) 自律走行および MLOps パイプラインの仕様については、[E2E_SPECIFICATION.md](E2E_SPECIFICATION.md) を参照してください。

## 2. システム構成

```mermaid
flowchart TD
    subgraph PC["ローカルPC (Python 3.13 / uv)"]
        UI["GUI (PySide6 / app.py)"]
        Graph["LangGraph Agent (graph.py)"]
        Tools["Tools<br>(Drive / Sensor / Camera / Voice)"]
        Client["PicarxClient (HTTPX)"]
        LLMFactory["LLM Factory (llm_factory.py)"]
    end

    subgraph LLM["LLM プロバイダー"]
        Gemini["Gemini API (Cloud)"]
        Ollama["Ollama (Local :11434)"]
    end

    subgraph Pi["Raspberry Pi (Rust / Axum Server :8080)"]
        API["Axum HTTP API (/api/*)"]
        Safety["安全ガード<br>(超音波リミッター / 前進拒否)"]
        TTS["TTS モジュール (tts.rs)<br>(Open JTalk / eSpeak / Piper Plus)"]
        Memory[("会話メモリ DB<br>picarx_memory.db (SQLite)")]
        Cam["カメラ (rpicam-still)"]
        HW["PiCar-X 実機ハードウェア<br>(モーター / サーボ / センサー / スピーカー)"]
    end

    UI --> Graph
    Graph --> Tools
    Tools --> Client
    Graph --> LLMFactory
    LLMFactory --> Gemini & Ollama

    Client -- "POST /api/drive/*" --> API
    Client -- "POST /api/audio/speak" --> API
    API -- "Snapshot (Base64 JPEG)" --> Client
    API -- "距離・センサー計測値" --> Client

    API --> Safety
    Safety --> HW
    API --> TTS --> HW
    API --> Cam
    API -. "永続化" .-> Memory
```

### 2.1 PC側

- 実装言語: Python 3.11以上
- GUI: PySide6
- LLM制御: LangChain / LangGraph
- HTTPクライアント: HTTPX
- 設定ファイル: `.env`
- エントリポイント: `python -m src.gui.app`

主な実装ファイル:

- `python/src/gui/app.py`
- `python/src/config.py`
- `python/src/llm_factory.py`
- `python/src/picarx_client/client.py`
- `python/src/agents/graph.py`

### 2.2 Raspberry Pi側

- 実装言語: Rust
- HTTPサーバー: Axum
- ハードウェア制御: rppal / PiCar-X実装
- APIポート: `8080`
- エントリポイント: `rust/src/main.rs`

主な実装ファイル:

- `rust/src/api/mod.rs`
- `rust/src/api/handlers.rs`
- `rust/src/api/hardware.rs`
- `rust/src/tts.rs`

## 3. GUI仕様

### 3.1 画面構成

画面は左右2カラムで構成する。

左カラム:

- LLMモデル選択
- カメラ画像表示
- 距離センサー表示
- 安全状態表示
- 手動車体制御
- 緊急停止

右カラム:

- チャット履歴
- ユーザー入力欄
- Sendボタン

### 3.2 モデル選択

現在のモデルプリセットは以下のとおり。

| 表示モデル | 内部モデルID | 用途 |
|---|---|---|
| Gemini Flash | `gemini-3.8-flash` | デフォルト。通常の対話・制御 |
| Gemini Flash-Lite | `gemini-3.5-flash-lite` | 低コスト・低遅延 |
| Gemini Pro | `gemini-3.1-pro-preview` | 高度な推論 |
| Ollama Local | `ollama/qwen2.5:7b` | ローカルLLM |

モデル変更時はLangGraphを再生成する。

Gemini APIが503 `UNAVAILABLE` となった場合は、以下の順で復旧を試みる。

1. LangChainの内部再試行
2. 3秒待機後の再試行
3. `gemini-3.5-flash-lite` へのフォールバック

429 `RESOURCE_EXHAUSTED` はAPIクォータ超過であり、アプリケーション側ではクォータを増加できない。

### 3.3 チャット処理

通常のメッセージは以下の処理を行う。

1. GUIにユーザー入力を表示
2. `HumanMessage` として会話履歴へ追加
3. LangGraphを実行
4. 最新のAIメッセージからテキストを抽出
5. GUIにAssistant返信を表示
6. Assistant返信をPi側TTS APIへ送信

Gemini 3.xの応答がテキストブロック配列の場合は、`type == "text"` の要素だけを抽出して表示する。署名やメタデータは表示しない。

### 3.4 直接発話コマンド

以下の入力はGeminiを経由せず、Pi側TTS APIを直接呼び出す。

| 入力例 | 発話内容 |
|---|---|
| `喋って` | `こんにちは。PiCar-Xです。` |
| `喋って こんにちは` | `こんにちは` |
| `喋って「おはようございます」` | `おはようございます` |
| `しゃべって...` | 指定された内容 |
| `話して...` | 指定された内容 |
| `発話して...` | 指定された内容 |

直接発話ではGemini APIを使用しないため、Geminiのクォータを消費しない。

### 3.5 チャット履歴エラー対策

LLM呼び出し失敗時は、未完了のユーザー発話を会話履歴から削除する。これにより、失敗後の再送信でGeminiのfunction call turn順序が壊れることを防ぐ。

## 4. カメラ仕様

### 4.1 現在の方式

GUI起動後、1秒間隔で `/api/camera/snapshot` を自動呼び出しする。

- 画像サイズ: `320 x 240`
- 形式: JPEG
- JPEG品質: 70
- 通信形式: HTTP JSON + Base64
- GUI表示: Base64復号後のQImage/QPixmap

現在はMJPEGやH.264による連続動画ストリームではなく、1秒ごとの自動スナップショット更新である。

### 4.2 Raspberry Pi側の撮影コマンド

Rust APIサーバーは次の順でカメラコマンドを実行する。

1. `rpicam-still`
2. 起動できない場合は `libcamera-still`

どちらも利用できない場合、APIはHTTP 500を返し、GUIには `Camera Error` を表示する。

### 4.3 カメラ表示状態

| 状態 | GUI表示 |
|---|---|
| 起動直後 | `Camera Connecting...` |
| 取得成功 | JPEG画像 |
| 取得失敗 | `Camera Error: ...` |

カメラ取得成功ログをチャット欄には表示しない。

## 5. センサー仕様

距離センサーは2秒間隔でポーリングする。

| 距離 | 安全状態 |
|---:|---|
| 15cm未満 | `ALERT: Obstacle Close!` |
| 15cm以上 | `SAFE (Limiter Active)` |

APIエラー時は距離表示を `Disconnected` に変更する。

## 6. 手動制御仕様

GUIから以下の操作を実行できる。

| 操作 | API | 初期値 |
|---|---|---|
| 前進 | `POST /api/drive/move` | speed 30, 400ms |
| 後退 | `POST /api/drive/move` | speed 30, 400ms |
| 左操舵 | `POST /api/drive/steer` | -30度 |
| 右操舵 | `POST /api/drive/steer` | 30度 |
| 中央復帰 | `POST /api/drive/steer` | 0度 |
| 緊急停止 | `POST /api/drive/stop` | 即時停止 |

前進時はRust側の安全ガードが超音波距離を確認する。危険距離の場合、前進命令を拒否する。

## 7. Raspberry Pi API仕様

ベースURLは環境変数 `ROBOT_BASE_URL` で指定する。デフォルト値は `http://raspberrypi.local:8080`。

| Method | Endpoint | 機能 |
|---|---|---|
| POST | `/api/drive/move` | 前進・後退・停止 |
| POST | `/api/drive/steer` | ステアリング角度変更 |
| POST | `/api/drive/stop` | 緊急停止 |
| GET | `/api/sensor/ultrasonic` | 超音波距離取得 |
| GET | `/api/sensor/grayscale` | グレースケール取得 |
| POST | `/api/audio/speak` | TTS生成・スピーカー再生 |
| GET | `/api/audio/listen` | 音声認識結果取得 |
| GET | `/api/camera/snapshot` | JPEG Base64画像取得 |

## 8. TTS仕様

### 8.1 エンジン種別

| エンジン | 定数 | 言語 | 品質 |
|---|---|---|---|
| Default (標準) | `TtsEngine::Default` | 日本語 → Open JTalk / 英語 → eSpeak-ng | 標準 |
| PiperPlus (高品質) | `TtsEngine::PiperPlus` | 日本語・英語を含む6言語 | 高品質 (Tsukuyomi-chan モデル) |

エンジンは Rust GUI 側のラジオボタンで切り替えられる。Python GUI 側からは現在選択不可（§12参照）。

### 8.2 処理経路

Python GUI からの経路:

```text
PC GUI
  -> POST /api/audio/speak
  -> Rust detect_language()
  -> Rust generate_tts_file()
  -> open_jtalk または espeak-ng (Default エンジン固定)
  -> PiCar-X の play_file()
  -> スピーカー
```

Rust GUI 内での非同期TTS処理（別スレッド）:

```text
LLM返答受信
  -> std::thread::spawn
    -> detect_language() で ja / en 判定
    -> generate_tts_file() で /tmp/picarx_tts_<nanoseconds>.wav 生成
       [Default]  : open_jtalk (ja) または espeak-ng (en)
       [PiperPlus]: piper-plus --model tsukuyomi --noise-scale 0.5 (stdin経由)
    -> play_file() でブロッキング再生
    -> 一時ファイル削除
```

音声生成・再生・削除は `std::thread::spawn` による別スレッドで非同期実行し、GUI描画をブロックしない。

### 8.3 実装詳細 (`rust/src/tts.rs`)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtsEngine {
    Default,    // Open JTalk (ja) / eSpeak-ng (en)
    PiperPlus,  // Piper Plus: Tsukuyomi-chan 6言語高品質モデル
}

pub fn detect_language(text: &str) -> &'static str;
// 日本語文字（ひらがな・カタカナ・漢字）が含まれれば "ja"、それ以外は "en"

pub fn generate_tts_file(text: &str, lang: &str, engine: TtsEngine)
    -> Result<PathBuf, Box<dyn Error + Send + Sync>>;
// PiperPlus の場合: /usr/local/share/piper-plus/bin/piper-plus
//   --model tsukuyomi --noise-scale 0.5 --output_file <path>
//   テキストは stdin 経由で渡す
// 一時ファイルパス: /tmp/picarx_tts_<nanoseconds>.wav
```

### 8.4 Raspberry Pi 実機セットアップ

#### デフォルトエンジン

```bash
sudo apt-get install -y \
  espeak-ng \
  open-jtalk \
  open-jtalk-mecab-naist-jdic \
  hts-voice-nitech-jp-atr503-m001
```

#### Piper Plus (Tsukuyomi-chan 高品質モデル)

> [!NOTE]
> `tsukuyomi` は日本語・英語を含む6言語対応の多言語モデルのため、1つのモデルで日英両言語に対応できる。

```bash
# Piper Plus 展開
sudo mkdir -p /usr/local/share/piper-plus
cd /usr/local/share/piper-plus
sudo curl -L -o piper.tar.gz \
  https://github.com/ayutaz/piper-plus/releases/latest/download/piper-linux-arm64.tar.gz
sudo tar xzf piper.tar.gz

# Tsukuyomi-chan モデル取得
sudo mkdir -p /usr/local/share/piper-plus/models
cd /usr/local/share/piper-plus/models
sudo curl -L -o tsukuyomi.onnx \
  https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan/resolve/main/tsukuyomi-chan-6lang-fp16.onnx
sudo curl -L -o config.json \
  https://huggingface.co/ayousanz/piper-plus-tsukuyomi-chan/resolve/main/config.json
```

`--noise-scale 0.5` は WavLM discriminator で学習された高品質モデル向けの公式推奨設定。

### 8.5 TTSエラー

- TTSファイル生成失敗: HTTP 500
- PiCar-X音声再生失敗: HTTP 500
- PC側ではチャット欄に `[TTS Error]` を表示

## 9. 設定

Python側の主な環境変数:

| 変数 | 必須 | デフォルト |
|---|---|---|
| `GEMINI_API_KEY` | Gemini使用時必須 | 空文字 |
| `ROBOT_BASE_URL` | 任意 | `http://raspberrypi.local:8080` |
| `OLLAMA_BASE_URL` | Ollama使用時 | `http://localhost:11434` |

PC側のHTTPタイムアウトは5秒。

## 10. 起動・デプロイ手順

### 10.1 Raspberry Pi 側プログラムのビルド & デプロイ

> [!IMPORTANT]
> **Raspberry Pi 実機上でのコンパイルはCPU性能・メモリ不足のため非推奨**です。
> 必ず **ローカルPC上のDocker環境でクロスコンパイルし、`scp` でRaspberry Piへバイナリを送信** してください。

#### Step 1: ローカルPCでクロスコンパイル (`rust/build/Makefile`)
```bash
cd rust/build

# お使いのラズパイのモデル (4b, zero2w, 3bplus, 5) を指定してビルド
make build MODEL=4b
```
※ 初回のみ `make image` でビルド用Dockerイメージを作成してください。

#### Step 2: Raspberry Pi へバイナリを送信
```bash
make deploy RPI_HOST=pi@<ラズパイのIP> RPI_DIR=/home/pi/
```
または手動で直接 `scp` 送信:
```bash
scp rust/target/aarch64-unknown-linux-gnu/release/picar-x-rust pi@<ラズパイのIP>:~/
```

#### Step 3: Raspberry Pi 上で起動
Raspberry Pi に SSH 接続し、送信されたバイナリを実行します：
```bash
./picar-x-rust
```
APIサーバーとして起動する場合、`--gui` は付けません。起動時に `/opt/picar-x/picar-x.conf` からサーボ校正オフセットが自動適用されます。

### 10.2 PC

```powershell
cd <project_root>\python
uv sync --extra dev
uv run python -m src.gui.app
```

### 10.3 接続確認

```bash
curl http://127.0.0.1:8080/api/sensor/ultrasonic
curl -X POST http://127.0.0.1:8080/api/audio/speak \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは"}'
```

## 11. テスト・品質保証

Python CIでは以下を実行する。

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest tests/ -v
```

Rust CIでは以下を実行する。

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --locked -- -D warnings
cargo nextest run --locked --release --profile ci
```

## 12. 既知の制約・今後の拡張

- カメラは現在、1秒間隔のJPEG自動更新であり、真の動画ストリームではない。
- カメラ更新ごとにスナップショットコマンドを起動するため、低遅延映像には向かない。
- Assistantの通常返信は毎回Pi側TTSで再生する。
- Gemini無料枠の429 `RESOURCE_EXHAUSTED` はアプリケーションでは解決できない。
- Rust APIサーバーはPi上のカメラコマンド、TTSエンジン、PiCar-Xハードウェアに依存する。
- TTSエンジン切り替えUIは旧Rust GUIには存在するが、Python GUIからは現在選択できない。
- 将来的にはPi側にMJPEGまたはH.264ストリームAPIを追加し、Python GUIで動画ストリームを表示する。

## 13. 会話メモリ機能 (フェーズ4: 長期記憶)

本機能は Gemini プロバイダーのみに適用。Ollama は単発リクエストのまま。

### 13.1 概要

| 機能 | 内容 |
|---|---|
| 会話履歴 | 直近メッセージを SQLite に保存し、コンテキストとして Gemini に渡す |
| 会話要約 | 未要約メッセージが12件以上になると Gemini で自動要約・統合 |
| 長期記憶 | ユーザー名・ロボット名・速度設定などをキーバリューで永続保存 |
| DBファイル | `./picarx_memory.db`（実行ファイルと同じディレクトリ） |

### 13.2 SQLite テーブル定義

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

### 13.3 モジュール構成 (`rust/src/memory/`)

| ファイル | 役割 |
|---|---|
| `mod.rs` | `pub mod model, repository, summarizer` |
| `model.rs` | `Message`, `ConversationSummary`, `Memory` 構造体定義 |
| `repository.rs` | SQLite CRUD 操作（`MemoryRepository`） |
| `summarizer.rs` | 自動要約ロジック（未要約12件以上でトリガー） |

依存クレート: `rusqlite = { version = "0.34", features = ["bundled"] }`
（`bundled` により Pi 上でもシステム SQLite 不要）

### 13.4 Gemini レスポンスへの `memory_updates` 追加

```rust
// CommandListPayload に追加
#[serde(default)]
memory_updates: Option<serde_json::Map<String, Value>>,
```

Gemini `responseSchema` の `memory_updates` は `required` に含めず optional。
キー形式はドット記法（例: `"user.name"`, `"robot.name"`, `"robot.default_speed"`）。

### 13.5 `memory_updates` バリデーション規則

| チェック | 詳細 |
|---|---|
| キーのホワイトリスト | `user.name`, `robot.name`, `robot.default_speed`, `robot.max_speed`, `user_preferences.*`, `behavior.*` のみ許可 |
| 値の型チェック | `speed` 系は数値かつ 0〜100 の範囲内 / `name` 系は文字列かつ50文字以内 |
| `null` 値 | 対応するキーの長期記憶を削除 |
| 不正なキー | ログ出力して無視（処理は続行） |

### 13.6 Gemini へのコンテキスト注入フォーマット

```text
[System Prompt] (既存 SYSTEM_PROMPT + Memory Rules 追記)

--- Long-term Memory ---
{
  "user.name": "太郎",
  "robot.name": "ピカちゃん",
  "robot.default_speed": 30
}

--- Conversation Summary ---
ユーザー（太郎）はロボット（ピカちゃん）に前進・後退の基本操作を指示した。

--- Recent Conversation ---
[User]: 右に曲がって
[Assistant]: はい、右に曲がります！
---

User request: さっきと同じように前に進んで
```

### 13.7 SYSTEM_PROMPT への追記内容

```text
Memory and Personalization Rules:
- If a "Long-term Memory" section is provided in the context, use names and preferences stored there.
- If the user tells you their name, robot's name, or any persistent preference, include a "memory_updates" object.
- memory_updates format: { "key": "value" } where key uses dot notation (e.g. "user.name", "robot.name").
- To delete a memory, set its value to null.
- Only include memory_updates when the user provides new persistent information.
```

追記はプロンプト末尾に行い、既存のコマンド生成指示には一切手を加えない。

### 13.8 会話初期化

- アプリ起動ごとに新しい `conversation_id` を UUID で生成。
- 長期記憶 (`memories` テーブル) は会話をまたいで永続化。

## 14. 今後の開発予定（未着手・検討中）

> [!NOTE]
> 全体ロードマップ上の「Phase 5: Agent × E2E 統合」に対応する項目です。
> 全体計画と進捗サマリーは [ROADMAP.md](ROADMAP.md) を参照してください。
> 以下は方向性のみ定義済みで、詳細仕様は今後別途検討します。

### 14.1 Raspberry Pi へのローカルLLM配置

**概要**: 現行の Gemini API 依存をなくし、Raspberry Pi 上でローカルLLMを動作させる。

- **現行**: PC から Gemini API（クラウド）へリクエスト → レスポンスを Pi へ転送
- **目標**: Pi 上でローカルLLMを直接実行し、ネットワーク不要・プライバシー保護・遅延ゼロ化
- **Gemini API オプション維持**: ローカルLLM が主系、Gemini API はオプション（設定で切替可能）とする
- **検討事項**:
  - Pi 上で動作可能なモデルサイズ・量子化方式の選定（例: `qwen2.5:1.8b`, `llama3.2:1b` など）
  - 推論エンジンの選定（例: Ollama on Pi / llama.cpp / candle）
  - Pi のメモリ・CPU 制約でのレスポンス速度の許容ライン

---

### 14.2 ローカルLLM の性能向上策

**概要**: モデルが小さいことによる性能低下を補い、Gemini API に近い品質を実現するための工夫。

- **検討事項**:
  - プロンプトエンジニアリング（システムプロンプトの最適化、Few-shot 例の挿入）
  - Function Calling / Structured Output 対応方式（モデルごとの JSON 生成能力差の吸収方法）
  - RAG（Retrieval-Augmented Generation）によるコンテキスト補強
  - モデル量子化レベル（Q4 / Q5 / Q8）とタスク精度のトレードオフ調査
  - 応答フォーマットの簡略化（小モデル向けに JSON スキーマを最小化）

---

### 14.3 GUI の Agent × E2E 統合と連携

**概要**: 現在 **別々に存在する「Agentic AI 制御 GUI (Python)」と「E2E 自律走行」を統合**し、さらに Agent と E2E が協調動作するアーキテクチャを構築する。

#### GUI 統合

- **現行**:
  - `python -m src.gui.app` — Agentic AI チャット GUI
  - `uv run python -m src.e2e.gui_collector` — E2E データ収集 GUI
  - `uv run python -m src.e2e.autonomous_driver` — E2E 自律走行（CLI）
- **目標**: 上記を1つの統合 GUI から操作できるようにする（モード切替 or タブ構成）

#### Agent × E2E 協調動作

- **検討イメージ**:
  - ONNX（E2E 推論） と ローカルLLM（または Gemini API）が連携して走行判断を行う
  - E2E モデルは「瞬時の操舵反射」、LLM は「状況理解・指示解釈・上位計画」を担当する役割分担
  - 例: LLM が「障害物を迂回して右の通路へ進め」と判断 → E2E モデルが低レイテンシで操舵実行
- **検討事項**:
  - どのレイヤーで両者が信号を交換するか（カメラ画像共有 / 操舵指令の優先度 / 安全ガードの管轄）
  - 衝突・矛盾が発生した場合のフォールバック戦略
  - データ収集フェーズと自律走行フェーズの GUI 上の状態遷移設計

> [!IMPORTANT]
> §14.3 は E2E システム（[E2E_SPECIFICATION.md](E2E_SPECIFICATION.md) Phase 5）とも密接に関連する。
