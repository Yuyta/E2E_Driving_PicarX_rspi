# PiCar-X 制御プログラム (Rust実装)
## 概要
下記のようなことができます。
- カメラから映像を取得
- 後輪ドライブモーター制御
- サーボモーター制御 (前輪ステアリング, カメラpan軸, カメラtilt軸)
- 底面Grayscaleセンサからの情報取得
- 超音波センサからの測距情報取得
- バッテリー電圧確認
- スピーカーから音を再生(Beep音, mp3等の音声ファイル)

## コンパイル方法
下記2つの方法のうち、いずれかを行う
- 実機(Raspberry Pi)上でコンパイル : CPUが貧弱で非常に時間かかるので非推奨
  ```bash
  cd <project_root>/rust
  RUSTFLAGS="-C target-cpu=native" cargo run --release
  ```
  - `RUSTFLAGS="-C target-cpu=native"` は、実機の CPU 向けに最適化するための指定
- ホストPCでクロスコンパイルし、その実行ファイルをRaspberry Piに転送 : 推奨
  ```bash
  cd <project_root>/rust/build
  make image   # Docker image作成
  make build MODEL=<3bplus/zero2w/4b/5>  # Dockerコンテナ上でクロスコンパイル
  make deploy RPI_HOST=<YOUR-RPI-HOST> RPI_DIR=<YOUR-RPI-DIR> 成果物の実行ファイルを`scp`コマンドでRaspberry Piへ転送
  ```

## CI

`.github/workflows/rust-ci.yml` は、push、Pull Request、手動実行で以下を確認します。

- `rustfmt` と Clippy (`-D warnings`)
- Raspberry Pi 5向け `aarch64-unknown-linux-gnu` リリースビルド
- UnitTest実行とJUnit XMLレポート

UnitTestの結果はGitHub Actionsのチェック、ジョブサマリー、JUnit XMLのArtifactで確認できます。ローカルでは次のMakeターゲットを利用できます。

```bash
cd <project_root>/rust/build
make fmt
make lint MODEL=5
make test MODEL=5  # クロス環境でUnitTestバイナリをコンパイル
```

## 環境変数

Pi の実行ファイルは API サーバーだけを起動します。LLM、記憶、音声入力は PC の Python 側で動きます（旧 Rust GUI と Rust 内の LLM は 2026-11-07 に削除）。

| 環境変数 | 説明 | 既定値 |
|---|---|---|
| `PICARX_API_TOKEN` | API のトークン。PC 側と同じ値にする。未設定だと起動しない | なし（必須） |
| `PORT` | API サーバーのポート | `8080` |
| `PICARX_ALLOW_MOCK` | `1` の時、ハードウェアの初期化に失敗してもモックで起動する。開発用 | 無効 |
| `PICARX_DISABLE_CLIFF_GUARD` | `1` の時、段差検知を無効にする。車輪を浮かせて試す時など、落下の恐れがない時だけ使う | 無効 |
| `DIR_SERVO_OFFSET` | 前輪の舵の中心の補正（度）。正が右 | `0`（校正ファイルがあればその値） |
| `MOTOR_BALANCE` | 左右の後輪の速さの差の補正（%、±30まで）。正なら右輪を、負なら左輪を遅くする。前輪がまっすぐでも左へ寄っていく車は正にする | `0` |
| `PICARX_MODELS_DIR` | 自律走行の走行モデルのフォルダ（`versions/<版>` と `current`） | `~/picarx/models` |
| `PICARX_RUNS_DIR` | 自律走行の記録（`run.json`、`interventions.jsonl`）のフォルダ | `~/picarx/runs` |

## 自律走行（Pi 上で走行のループを動かす）

走行モデルの推論、障害物を避ける判断、後退、介入の記録を Pi で行います（`src/autopilot/`）。指令は HTTP の走行指令と同じ安全の判定（反射ループ、TTL）を通ります。

| 窓口 | 内容 |
|---|---|
| `POST /api/autopilot/start` | 使用中のモデルを確かめて読み込み、走行を始める。本文は省略可（`{"speed": 50, "min_speed": 40, "intent": "any"}`） |
| `POST /api/autopilot/stop` | 走行を終える。`{"intervention": false}` なら人の介入として記録しない |
| `POST /api/autopilot/intent` | 行きたい向きを変える（`{"intent": "left"}`） |
| `GET /api/autopilot/status` | 状態（舵角、速度、推論時間、介入の回数など） |
| `GET /api/autopilot/model` | 使用中のモデルの版と、ハッシュの確認結果 |

走行中に `/api/drive/move` か `/api/drive/steer` が来ると、人が運転を引き継いだとみなして自律の指令を止め、1.5 秒途絶えると戻ります。`/api/drive/stop` と緊急停止では走行を終えます。

モデルの版は、走行サーバーとは別に同じ実行ファイルで管理します。PC からは `uv run python -m src.e2e.deploy.pi push --model_dir models/<版>` で送れます。

```bash
picar-x-rust model install /tmp/v006 --activate  # 確かめてから版として加え、使用中にする
picar-x-rust model list                          # * が使用中
picar-x-rust model rollback                      # 1つ前に使っていた版へ戻す
picar-x-rust model verify                        # 使用中の版のハッシュを確かめる
```

切り替えは次の自律走行の開始から使われます。使用中の版と、その前の2つの版を残します。
