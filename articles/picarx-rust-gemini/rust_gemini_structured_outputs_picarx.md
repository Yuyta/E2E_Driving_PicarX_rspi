# 第1章：導入（自然言語でロボットを安全に動かしたい！）

## ターゲット読者
- Rustを使ってIoTデバイスやロボット（PiCar-X）を動かしたい方
- LLMの応答をプログラムでパースする際のエラーに頭を抱えている方
- Gemini APIの「Structured Outputs（構造化出力）」の具体的な活用例を知りたい方

---

## はじめに：この記事を書いたきっかけ（開発の経緯）

この記事の執筆に至ったのは、趣味のラズパイ工作がきっかけです。

「Raspberry PiとPiCar-Xのキットを使って自作ロボットを作ろう！」と思い立ち、趣味の開発を始めました。しばらく動かしているうちに、「せっかくならキーボードやコントローラーでの操作ではなく、自然言語で指示を出して自律的に動かせたら面白いのではないか？」と考えました。

しかし、いざ実装しようと調べ始めると、LLMの出力フォーマットをプログラムで正しく解析し、思い通りに安定してロボットを動かすことの難しさに突き当たりました。
そこでさらに調査を進めたところ、Gemini APIの **「Structured Outputs（構造化出力）」** を使えば、意図通りのスキーマで出力を型保証し、スムーズにプログラムに組み込めるのではないかと考え、トライしてみました。

結果として、非常に安定して思い通りのロボット制御を実現することができたため、その知見を共有したくこの記事を書きました。

---

## AIでロボットを動かすロマンと、立ちはだかる「パースエラー」の壁

「2秒前進して、障害物があったら止まって」

このような自然言語 of 指示から、ラズパイカー「PiCar-X」などのロボットを思い通りに制御できたら面白いですよね。

しかし、物理的なハードウェアをLLMで制御しようとするとき、避けて通れない大きなハードルがあります。それは **「LLMの出力の不確実性（パースエラー）」** です。

通常のLLMの用途（チャット、要約、メールの下書き作成など）であれば、出力されるテキストに多少の表記揺れや余計な挨拶、フォーマットの崩れがあっても、それを読む人間が柔軟に解釈できるため問題になりません。

しかし、今回の「ロボット制御」のように、LLMの出力をプログラムが直接読み込んで機械を動かす用途では話が変わります。1文字のパースエラーや、キー名のわずかな揺れ（例えば `steer` と `steering` の違いなど）によってエラーが発生し、**ロボットが停止できずに壁にコツンとぶつかってしまったり、思った通りに動いてくれなかったりする問題**が生じます。

不確実なLLMの出力と、思い通りに正しく制御したい物理ロボット。この2つをうまく繋ぐために、本プロジェクトでは **「Rust」** と **「Gemini APIの構造化出力（Structured Outputs）」** を組み合わせました。

---

## なぜRust言語を採用したのか？

ロボットの制御にはPythonが使われることも多いですが、本プロジェクトでは**Rust**を採用しています。その理由は大きく2つあります。

1. **「プログラムの突然死（フリーズや強制終了）」を防ぐ圧倒的な頑丈さ**
   - ロボットを動かしている最中に、プログラムがメモリのバグなどで急にクラッシュしたら、ロボットは暴走するかその場で立ち往生してしまいます。Rustは、こうした「原因不明のフリーズや強制終了」を引き起こすバグを、プログラムを組み立てる（ビルドする）段階で徹底的に見つけて弾いてくれます。そのため、実世界で動かすロボットを極めて頑丈に作ることができます。
2. **AIの「考え中」も、ロボットの目と耳を止めない器用さ**
   - ネット経由でAIに指示を送ってから返事が戻ってくるまでには、数秒〜十数秒かかります。この「待ち時間」の間、ロボットの頭脳が完全にフリーズしてしまったらどうなるでしょう？目の前に壁や人があっても気づけずにぶつかってしまいます。
   - Rustなら、「AIとネット越しにおしゃべりする担当」と「センサーで周りを監視しつつタイヤを動かす担当」を、安全に別々のスレッド（同時並行処理）として分担させることができます。AIの返事を待つ間も、ロボットの安全装置やセンサー監視を1ミリ秒も止めずに動かし続けられるのです。

---

## 本記事で解説すること

本記事では、この安全なロボット制御を実現するために工夫した以下のポイントについて順を追って解説します。

- 構造化出力がないと、LLMの応答がどのように破綻しロボットに危険を及ぼすのか
- Gemini APIの **Structured Outputs** を使い、APIレベルで出力スキーマを100%保証する方法
- 複雑な「自律停止条件（until条件）」をLLMに組み立てさせ、それをRust側で安全に実行するアーキテクチャ

<div style="page-break-before: always;"></div>

# 第2章：構造化出力がないと何が起きるか？（制御破綻のリアル）

## 「JSONで返して」というプロンプトの限界

通常のLLM（JSONモードや特別なスキーマ制御を適用していない状態）に対して、「ロボットの制御指示をJSON形式だけで出力してください」とプロンプトで指示してみたとします。

その際、AIから返ってくるレスポンスには、以下のような「おなじみの問題」が多発します。

### 問題1. 余計な文字や装飾コードの混入

```text
もちろん、以下がロボットの制御コマンドです：

```json
{
  "commands": [
    {"action": "forward", "speed": 50, "duration_ms": 2000}
  ]
}
```
以上のコマンドを実行してください。
```

#### ポイント解説
- **余計なテキストの混入**: 前後の「もちろん〜」「以上のコマンド〜」といった挨拶文が混入します。
- **マークダウン装飾の付与**: JSON自体が ```json ... ``` で囲まれてしまいます。
- これらを文字列処理（正規表現やトリム処理）で無理やり取り除くことは不可能ではありませんが、AIの応答パターンは毎回答えが変わるため、例外処理が際限なく増えてコードが汚れる原因になります。

---

## 複雑な条件分岐（ネスト）における「キー名」の揺らぎ

本プロジェクトの最大のこだわりは、単に「進め」「止まれ」だけでなく、「超音波センサーの距離が15cm未満になるまで進め」という **「センサー連動の自律停止（until条件）」** をLLMに考えてもらう点です。

これを実現するために、LLMには以下のようなネスト（入れ子）構造を持つ複雑なJSONを返すように求めます。

```json
{
  "commands": [
    {
      "action": "forward",
      "speed": 40,
      "duration_ms": 15000,
      "until": {
        "sensor": "distance",
        "op": "lt",
        "value": 15
      }
    }
  ]
}
```

#### ポイント解説
- **`until`フィールド**: ロボットがアクションを中断する条件（条件式）を記述します。
- **`sensor`**: 監視するセンサーの種類（`distance`＝超音波距離センサー）。
- **`op`**: 比較演算子（`lt`＝「未満」）。
- **`value`**: 判定の閾値（15cm）。

しかし、構造化出力がない状態でLLMにこれを生成させると、プロンプトでいくら厳密に指定しても、高確率で以下のような「揺らぎ」が発生します。

- `"sensor"` というキーが `"sensor_type"` や `"target_sensor"` に変わってしまう。
- `"op"` (演算子)の値を `"lt"` ではなく、 `"less_than"` や `"Under"` と自由に変えて出力してしまう。
- ロボット側で定義していない無効な値（例: スピード制限が最大60なのに `"speed": 120` など）を出力してしまう。

プログラム（Rustのデシリアライズ処理）は定義されたキー名やデータ型、列挙型（`enum`）と**1文字でも違うと、エラーを吐いてパースを断念**します。

---

## 物理ロボットにおける「パースエラー」による誤動作

Webブラウザ上で動作するアプリであれば、パースエラーが起きても「もう一度ボタンを押してください」と画面に表示してユーザーにやり直してもらえば済みます。

しかし、実世界で動くロボットの場合、パースエラーが起きると**「思い通りに動かない」「止まらない」といった誤動作**に繋がります。

例えば、「前方に障害物があるまで前進せよ」という指示をロボットが実行しているとします。
もし、LLMから返ってきたJSONの `until` 条件（停止条件）が1文字のタイポのせいでパースできなかった場合、プログラムはどう動くべきでしょうか？

1. **エラーとして無視して処理を中断する**
   - ロボットはその場で停止し、フリーズしたようになってしまいます。これでは指示通りに動いてくれません。
2. **パースできない部分（停止条件）を無視して、とりあえず前進コマンドだけを実行する**
   - 停止条件（障害物が近づいたら止まる）が読み取れないまま「前進」だけが走り出すため、**ロボットは止まるタイミングを失って壁や障害物にコツンとぶつかってしまったり、タイヤが回りっぱなしになってしまったり**します。

AIの「出力のあいまいさ」が、ロボットのちょっとした誤動作や、モーターなどへ余計な負荷をかける原因になってしまうのです。

<div style="page-break-before: always;"></div>

# 第3章：GeminiのStructured Outputsがもたらす「型安全」な解決

## 救世主：Structured Outputs（構造化出力）とは？

前章で紹介した「JSONが崩れてロボットが誤動作する問題」を根本から解決するのが、Gemini APIの **「Structured Outputs（構造化出力）」** という機能です。

- **公式ドキュメント**: [Structured Outputs | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/structured-outputs)

これは、プロンプトで「JSONで出力して」とお願いするのではなく、**「出力するJSONの構造（スキーマ）をあらかじめ定義してAPIに送り、モデルにそのスキーマを100%厳守させる」** 機能です。

例えば、以下のように「キー名」や「データの型」「必須項目」を明示したスキーマ（JSON Schema）をリクエストに含めて送信します。

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
            "enum": ["forward", "backward", "stop"]
          },
          "speed": {
            "type": "INTEGER"
          }
        },
        "required": ["action"]
      }
    }
  },
  "required": ["commands"]
}
```

#### ポイント解説
- **`properties` と `type`**: オブジェクトに含まれるキー名（`commands`, `action`, `speed`）とそのデータ型を定義します。
- **`enum`**: 文字列に入る値を `forward`, `backward`, `stop` のいずれかに限定します。
- **`required`**: そのキーが必ずレスポンスに含まれることを強制します。

このスキーマをAPI側に渡すことで、Geminiは**レスポンスを生成する際にスキーマ外のキーを出力したり、挨拶文を混ぜたりすることが構造上できなくなります。** これにより、返ってくるデータは常に定義した通りのクリーンなJSONオブジェクトになることが保証されます。

---

## Rustの「型システム」との相乗効果

この構造化出力は、**Rust言語と非常に強力なシナジー（相乗効果）を発揮します。**

Rustには、JSONデータをRustの構造体や列挙型（`enum`）に一発で変換（デシリアライズ）できる非常に強力なライブラリ **`serde`** があります。

通常、LLMの出力が不安定だと、`serde` は「型やキーが合わない」としてすぐにパースエラー（エラー返却）を起こしてしまいます。しかし、Geminiの構造化出力によって「スキーマ通りであること」がAPIレベルで100%保証されると、以下のようなメリットが生まれます。

1. **パースエラーによる強制終了の撲滅**
   - 届くデータが必ずスキーマに適合しているため、Rust側での `serde_json::from_str` が失敗することがほぼゼロになります。
2. **無駄な「例外処理コード」の削減**
   - 挨拶のトリム処理、マークダウン記法の除去、キーの揺らぎに対する分岐処理など、これまでパースエラー対策のために書いていた泥臭いコードがすべて不要になり、ビジネスロジックに集中できます。
3. **Rustの型安全性の恩恵をフルに受ける**
   - LLMから取得したコマンド情報を、直接Rustの型安全な `Command` enum にマッピングしてそのままロボットのモーター駆動関数などに流し込めるようになります。

<div style="page-break-before: always;"></div>

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

ラズパイカーは、走っている間も常にカメラの映像を描画したり、周囲 of センサー値をスキャンし続けたりしなければなりません。
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

<div style="page-break-before: always;"></div>

# 第5章：高度な工夫：until条件による安全な自律制御

## 時間制御だけでは、おもちゃのロボットでも不十分

通常、ロボットへの単純な命令は「スピード30で、2秒間前進する」といった時間ベースのものが基本です。
しかし、もし障害物との距離が近いときにこのコマンドが実行されてしまうと、ロボットは止まるタイミングを失って障害物にコツンとぶつかってしまいます。

これを回避するために、本プロジェクトでは **「until（〜まで）条件」** というセンサー連動の停止ルールをLLMに指示できるようにしました。

例えば、「障害物がない間は前進し、距離が15cm未満になったら停止する」という指示に対し、Geminiは以下のJSONを生成します。

```json
{
  "action": "forward",
  "speed": 40,
  "duration_ms": 15000,
  "until": {
    "sensor": "distance",
    "op": "lt",
    "value": 15
  }
}
```

#### ポイント解説
- **`duration_ms`**: 条件が満たされない場合に備え、15秒（15000ms）という長めの「安全用タイムアウト時間」を確保しています。
- **`until`**: 超音波センサー（`distance`）の値が 15（cm）未満（`lt`）になった瞬間に、`duration_ms` の経過を待たずに動作を終了します。

---

## センサー条件を支えるRustの再帰的データ構造

複雑なセンサー条件をプログラム側で安全に解釈するために、[command.rs](file:///c:/Users/tacta/Dev/PiCarX/rust/src/llm_agent/command.rs) では条件式を再帰的な構造体として定義しています。

これにより、LLMが「A0（左）とA2（右）のグレイスケールセンサーの両方が白線を検知したとき」のような、論理演算（AND / OR / NOT / XOR）をネストした複雑な条件を作成しても、Rust側で正しくパースして処理できます。

```rust
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Condition {
    pub sensor: SensorKind,
    pub op: Operator,
    pub value: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ConditionExpr {
    And { and: Vec<ConditionExpr> },
    Or { or: Vec<ConditionExpr> },
    Xor { xor: Vec<ConditionExpr> },
    Not { not: Box<ConditionExpr> },
    Leaf(Condition),
}

impl ConditionExpr {
    // センサーの現在の値と条件式を比較し、条件が成立しているか判定する
    fn is_met(&self, s: &SensorSnapshot) -> bool {
        match self {
            ConditionExpr::And { and } => and.iter().all(|c| c.is_met(s)),
            ConditionExpr::Or { or } => or.iter().any(|c| c.is_met(s)),
            ConditionExpr::Xor { xor } => xor.iter().filter(|c| c.is_met(s)).count() == 1,
            ConditionExpr::Not { not } => !not.is_met(s),
            ConditionExpr::Leaf(cond) => cond.is_met(s),
        }
    }
}
```

#### ポイント解説
- **`ConditionExpr`**: 論理演算（`And`, `Or` 等）のバリアントを再帰的に保持できる enum です。
- **`#[serde(untagged)]`**: JSON内のオブジェクトの構造（キー名が `and` か `or` か、あるいは単純な葉ノードの `sensor` か）から、serdeが自動で適切なバリアントに振り分けてくれます。
- **`is_met`**: センサーのスナップショットデータ（`SensorSnapshot`）を引数に取り、条件が満たされているか（`true`）をリアルタイムで再帰的に評価します。

---

## 毎フレーム監視する「非ブロッキング」ステートマシン

APIから受信したコマンドの配列を順番に実行する際、`std::thread::sleep` のようにスレッド全体をスリープさせてしまうと、その間センサーの監視や条件評価、カメラ映像の描画がすべて止まってしまいます。

そこで本プロジェクトでは、メインループから毎フレーム呼び出される **`CommandQueue`** という非ブロッキングなステートマシンを実装しています。

```rust
pub struct CommandQueue {
    commands: Vec<Command>,
    current_index: usize,
    current_started_at: Option<Instant>,
}

impl CommandQueue {
    /// 1フレーム分の実行を進める。まだ実行中なら true、全コマンド消化済みなら false を返す。
    pub fn tick(&mut self, px: &mut Picarx, sensors: SensorSnapshot) -> bool {
        if self.current_index >= self.commands.len() {
            return false;
        }

        // コマンドの開始時にサーボの角度変更やモーター回転を呼び出す
        if self.current_started_at.is_none() {
            self.start_current(px);
        }

        let current = &self.commands[self.current_index];
        let duration_ms = current.duration_ms();
        let until = current.until();

        // 1. 指定時間が経過したか判定
        let duration_elapsed = self
            .current_started_at
            .map(|t| t.elapsed().as_millis() as u64 >= duration_ms)
            .unwrap_or(true);

        // 2. until条件が成立したか判定
        let condition_met = until.map(|c| c.is_met(&sensors)).unwrap_or(false);

        // 時間切れ、またはセンサー条件を満たした場合に次のコマンドへ移行する
        if duration_elapsed || condition_met {
            if condition_met && !duration_elapsed {
                println!("[LLM] センサー条件が成立したため中断します");
            }
            self.finish_current(px);
            self.current_index += 1;
            self.current_started_at = None;
        }

        self.current_index < self.commands.len()
    }
}
```

#### ポイント解説
- **`tick`**: 毎フレームのループで呼び出され、時間（`duration_elapsed`）とセンサー条件（`condition_met`）の2つを評価します。
- **ノンブロッキング**: スリープ処理を使わずに経過時間を `Instant::now()` との差分で測るため、ミリ秒単位でセンサーデータをスキャンしながら、動きを即座に中断（停止）させることができます。
- **安全設計の終了処理**: `finish_current` にて、移動系のコマンドが終了した後はモーターを停止させ、曲げたタイヤのステアリング角度を自動でまっすぐ（0度）に戻す安全処理を施しています。

<div style="page-break-before: always;"></div>

# 第6章：まとめ（AIの「あいまいさ」を「確実な型」に閉じ込める）

## ロボット制御における「型」の重要性

本プロジェクトでは、不確実なLLMの出力と、厳格な制御が求められる物理ロボット（PiCar-X）を繋ぐために、Gemini APIの **Structured Outputs（構造化出力）** を採用しました。

これにより、以下のような恩恵が得られました。

1. **エラーのないパース**
   - API側が出力JSONのスキーマを100%保証してくれるため、Rustの `serde_json` でのデシリアライズ失敗による動作エラーやフリーズをほぼゼロにできました。
2. **複雑な自律ロジックの実現**
   - LLMに「障害物までの距離が15cm未満」のような動的なセンサー判定条件（`until`）をJSON Schema of 制約下で正しく組み立てさせ、それをRust側の非ブロッキングな `CommandQueue::tick` でリアルタイムに監視・実行する高度な仕組みを安全に作ることができました。
3. **無駄な泥臭いコードの撲滅**
   - マークダウン記法の除去や、キー名のわずかな揺れ（表記ブレ）などを補正するための複雑な正規表現や文字列操作のコードが一切不要になりました。

物理ロボットだけでなく、スマートホームの制御や社内システムのAPI連携など、**「AIの出力をもとにプログラムを動作させる」あらゆる用途**において、構造化出力はもはや必須の技術であると言えます。

---

## 今後の展望

現在は、ロボットの「移動（前進、後退、旋回など）」に関する指示と制御が中心になっています。

今後は、ロボットができることをさらに広げるために、カメラの向きを変える指示（首振り）や、超音波・グレイスケールセンサーからの情報取得（距離や白線の検知）、バッテリー状況の確認など、移動以外の機能についても拡張性を持たせ、より複雑で自律的な対話制御ができるように機能を広げていきたいと考えています。

皆様もぜひ、お手持ちのラズパイやロボットキットをAIで動かして遊んでみてください！

---

## 引用・参考リンク
- [Structured Outputs | Gemini API | Google AI for Developers](https://ai.google.dev/gemini-api/docs/structured-outputs)
- [PiCar-X 公式ドキュメント（SunFounder）](https://docs.sunfounder.com/projects/picar-x/en/latest/)
