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
