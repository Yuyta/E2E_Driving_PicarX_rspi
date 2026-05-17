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
  make build   # Dockerコンテナ上でクロスコンパイル
  make deploy  # 成果物の実行ファイルを`scp`コマンドでRaspberry Piへ転送
  ```

