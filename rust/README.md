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
  - `RUSTFLAGS="-C target-cpu=native"` は、ndarray simd命令による高速化のためのおまじない
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
- Raspberry Pi 3B+向け `aarch64-unknown-linux-gnu` リリースビルド
- UnitTest実行とJUnit XMLレポート

UnitTestの結果はGitHub Actionsのチェック、ジョブサマリー、JUnit XMLのArtifactで確認できます。ローカルでは次のMakeターゲットを利用できます。

```bash
cd <project_root>/rust/build
make fmt
make lint MODEL=3bplus
make test MODEL=3bplus  # クロス環境でUnitTestバイナリをコンパイル
```

## 環境変数 (LLM連携)

自然言語プロンプトからロボットへの命令を生成するため、LAN上の別ホストPCで動かした[Ollama](https://ollama.com/)サーバーに接続します。実行前に以下の環境変数を設定してください(未設定の場合は既定値が使われます)。

| 環境変数 | 説明 | デフォルト |
|---|---|---|
| `OLLAMA_URL` | Ollamaサーバーのアドレス。ホスト名/IPのみ、`host:port`、完全なURL(`http://host:port/api/chat`)のいずれの形式でも指定可能 | `127.0.0.1` (ポート`11434`, パス`/api/chat`を自動補完) |
| `OLLAMA_MODEL` | 使用するOllamaのモデル名 | `llama3.1` |

```bash
export OLLAMA_URL=192.168.0.129        # Ollamaサーバーが動いているホストPCのIP
export OLLAMA_MODEL=mistral:7b-instruct
```

起動時に実際に接続を試みるURLが `[LLM] Ollama接続先: ...` としてログ出力されるので、設定ミスがあれば確認できます。

