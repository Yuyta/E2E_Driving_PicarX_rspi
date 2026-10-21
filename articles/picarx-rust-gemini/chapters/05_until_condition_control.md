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
