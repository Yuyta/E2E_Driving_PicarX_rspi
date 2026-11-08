# PiCar-X End-to-End (E2E) 自動運転 実践マニュアル

本書は、PiCar-Xにおいて **「データ収集 → モデル学習 → オフライン評価 → 実機自動走行」** の一連のE2E自律走行ループを実行するための手順書です。

> [!NOTE]
> 詳細な内部仕様、モデル構造、データスキーマ、MLOps設計については [E2E_SPECIFICATION.md](../../../E2E_SPECIFICATION.md) を参照してください。

---

## 前提条件: Raspberry Pi側の準備 (Rustサーバーのビルド & 転送)

> [!IMPORTANT]
> **Raspberry Pi 実機上でのコンパイルは非推奨**です。
> Rustプログラムの変更時は、**ローカルPCのDocker上でクロスコンパイルし、`make deploy` でRaspberry Piへ送信**してください。

```bash
# ローカルPC (Docker環境) でクロスコンパイル
cd rust/build
make build MODEL=5      # 基準機種は Pi 5

# Raspberry Pi へ実行バイナリを転送
make deploy RPI_HOST=pi@<ラズパイIP> RPI_DIR=/home/pi/

# Raspberry Pi 上で起動 (SSH接続先)
./picar-x-rust
```

サーバーの起動には API トークンが必要です。同じ値を Pi の `.env`（実行ファイルと同じディレクトリ）に `PICARX_API_TOKEN=<値>`、PC の `python/.env` に `ROBOT_API_TOKEN=<値>` として設定してください。値の生成方法は [APPLICATION_SPECIFICATION.md §9](../../../APPLICATION_SPECIFICATION.md) を参照してください。

---

## 全体アーキテクチャ

```text
[ 1. データ収集 (PC GUI) ]
    カメラ映像を確認しながらキーボード操作 (W/A/S/D) で走行
    録画(R)中にカメラ画像と操作量を PC (python/data/raw/) に自動保存
           ↓
[ 2. モデル学習 (PC: PyTorch) ]
    収集データをもとに軽量CNN (SmallDrivingCNN) を学習
    最良モデルを ONNX 形式 (model.onnx) に自動エクスポート
           ↓
[ 3. オフライン評価 (PC) ]
    検証データに対するステアリング平均絶対誤差 (MAE, ±◯度) を算出
           ↓
[ 4. 実機自動走行 (PC で推論し、Raspberry Pi へ指令) ]
    ONNX Runtimeによる自律操縦 + 安全機能 (Safety Layer)
```

---

## ステップ 1: データ収集 (PC GUI)

PC上でカメラのライブ映像を見ながらキーボード操作し、走行データを記録します。

### 起動コマンド (PowerShell)
```powershell
uv run python -m src.e2e.gui_collector
```

### 操作方法
| キー | 動作 | 説明 |
| :---: | :--- | :--- |
| **`W`** | 前進 | 設定速度で前進します |
| **`S`** | 停止 | モーターを停止します |
| **`A`** | 左旋回 | ステアリングを10度左に切ります |
| **`D`** | 右旋回 | ステアリングを10度右に切ります |
| **`X`** | 直進 | ステアリング角度を0度に戻します |
| **`R`** | 録画切替 | **● 録画開始 / ⏸ 一時停止** をトグルします |

### 収集のコツ
1. 車体を走行させたい部屋の開けたスペースに置きます。
2. **`R` キーを押して録画を開始**します（画面に「● Pause Recording」と表示され、フレーム数カウントが進みます）。
3. 壁や家具を避けながらスムーズに室内を巡回走行（1,000〜2,000フレーム程度）させます。
4. 走り終わったら **`R` キーで録画を停止**し、ウィンドウを閉じます。

### 保存されるデータ
- 保存先: `python/data/raw/<session_timestamp>/`
  - `images/`: カメラ画像 (`frame_000000.jpg` ... 160x120ピクセル)
  - `labels.jsonl`: 各フレームのタイムスタンプ、ステアリング正規化値（-1.0〜+1.0）、速度、超音波の距離、その時点の指令、推定した実際の動き、指令との食い違い

### 指令と実際の動きの確認 (Motion Check)
画面右の **Motion Check** に、カメラの画像の動きと超音波から推定した実際の動きと、指令との食い違いを表示します。
- 進む指令なのに動いていない（空転、引っかかり）、曲がり方が舵角と合わない、指令していないのに動いている、の3種類を検出します
- 食い違いが1秒続くと車を止め、記憶に残します（会話の Agent に「さっき何で止まったの？」と聞くと答えます）
- 食い違いのあるフレームは学習に使いません
- しきい値は実機で校正する前の提案値です（ROADMAP の M3）

### 収集の後に行うこと
走行データはセッション単位で追記のみとし、書き換えません。収集が終わったら、マニフェスト（ファイルのハッシュの一覧）を作り、バックアップします。
```powershell
uv run python -m src.e2e.dataset.manifest --data_dir data/raw            # 新しいセッションに manifest.json を作る
uv run python -m src.e2e.dataset.manifest --data_dir data/raw --verify   # 変わっていないか確かめる
cd ..; powershell -ExecutionPolicy Bypass -File scripts\backup.ps1       # D:\PiCarX-backup へ複製（GUI を閉じてから）
```

---

## ステップ 2: モデル学習 & ONNXエクスポート (PC)

収集した走行ログをもとに、画像からステアリング角度を予測する軽量CNNを学習します。
学習は CPU で決定的に行い、同じデータと設定なら同じ結果になります。検証はセッション単位で行うので、**2セッション以上が必要**です。

### 実行コマンド (PowerShell)

#### 過去の全走行データをまとめて学習する場合 (推奨)
```powershell
uv run python -m src.e2e.training.train --config configs/train.toml --data_dir data/raw --output_dir models/v006
```
- 設定は `configs/train.toml`（エポック数、バッチサイズ、学習率、シード、検証セッション）。`--epochs` などの引数で上書きできます
- 検証セッションの指定がなければ、名前順で最後のセッションを検証に使います。`--val_sessions <session_id>` で指定できます
- 複数フォルダを指定したい場合は `--data_dir data/raw/session1 data/raw/session2` のようにスペース区切りで複数指定できます

#### モデルや設定を比べる場合（交差検証）
セッションを1つずつ抜いて学習し、抜いたセッションで評価します。学習に使っていない場面での誤差なので、比較にはこちらを使います。
```powershell
uv run python -m src.e2e.training.cross_validate --config configs/train.toml --data_dir data/raw --output data/cv/baseline.json
```

### 生成物
- `models/v006/best_model.pth` : PyTorchモデル重み
- `models/v006/model.onnx` : 実機配信用ONNXモデル
- `models/v006/metadata.json` : 設定、分割、前処理、検証の指標に加え、使ったデータのハッシュ、git のコミット、環境（ライブラリの版、`uv.lock` のハッシュ）の記録

---

## ステップ 2.5: データクリーニング (悪いデータの自動除外)

学習済みモデル（例: `v003` など）を使って全走行フレームの予測誤差（残差）を算出し、**「人間の操作遅れ」「衝突時の混乱操舵」「外れ値」を自動ではじいた高品質データセット** を作成します。

### 実行コマンド (PowerShell)
```powershell
# 閾値 0.4 (約±14度) 以上の乖離があるフレームを自動除外
uv run python -m src.e2e.dataset.cleaner --data_dir data/raw --model models/v003/model.onnx --threshold 0.4 --output_dir data/processed/clean_v1
```

#### オプション説明:
- `--data_dir` : 対象のデータフォルダ（`data/raw` または特定セッション）
- `--model` : 誤差測定に使用するモデル（`.onnx` または `.pth`）
- `--threshold` : 許容する最大ステアリング絶対誤差（デフォルト: `0.4`）
- `--top_k_percent` : 誤差が大きい上位K%を除外する場合に指定（例: `--top_k_percent 5.0`）
- `--output_dir` : クリーン化されたデータセットの出力先

出力先には画像を複製せず、元のセッションを参照する `labels.jsonl` だけを書きます。元のデータ（`data/raw`）を移動したり消したりすると読めなくなります。
モデルと合わないフレームを除くので、クリーニング後の MAE は良く見えます。改善の判断には交差検証と実走の介入回数を使います。

### クリーニング後の再学習
```powershell
uv run python -m src.e2e.training.train --config configs/train.toml --data_dir data/processed/clean_v1 --output_dir models/v006
```

---

## ステップ 3: モデル評価 (PC オフライン検証)

テストデータに対する予測精度の誤差（MAE: 平均絶対誤差）を測定します。

### 実行コマンド (PowerShell)
```powershell
uv run python -m src.e2e.evaluation.evaluate --data_dir data/raw/<session_id> --model_path models/<version>/best_model.pth
```

### 結果表示例
```text
--- Evaluation Results ---
Total Samples : 1250
Steering MAE  : 0.0821 (Normalized -1.0 to 1.0)
Steering MAE  : ±2.87° (Degrees)
--------------------------
```
※ MAE は学習データとの一致度を示す目安で、実走の性能は保証しません。
評価結果は自動的に各モデルの `models/<version>/eval_metrics.json` に記録されます。

---

## MLOps: データセット・モデルの比較 & ベンチマーク確認

収集した全セッションの統計（直進/左折/右折の分布）や、各モデルバージョン（`v001`, `v002`...）の学習パラメータ・評価結果を一覧比較できます。

```powershell
uv run python -m src.e2e.registry.compare
```

#### 表示例:
```text
================================================================================
                     PiCar-X Dataset Registry Summary
================================================================================
Session ID         | Frames  | Duration  | FPS   | Straight  | Left   | Right 
--------------------------------------------------------------------------------
20261006_235445    | 35      | 50.9s     | 0.7   | 60.0%     | 40.0%  | 0.0%  
================================================================================

=====================================================================================
                      PiCar-X Model Registry & Benchmark
=====================================================================================
Model Version  | Epochs | Batch | Val Loss  | MAE (Norm) | MAE (Deg)  | Artifacts 
-------------------------------------------------------------------------------------
v001           | 15     | 32    | 0.0946    | 0.2586     | ±9.1°      | PTH+ONNX  
=====================================================================================
```

## ステップ 4: 自動走行 (PC から実機を操縦)

ステップ 2 で生成された `models/<version>/model.onnx` を使い、実機で自動運転を実行します。
（データ収集時と同様、PC側で推論を実行し、Raspberry PiのRust APIサーバーへ操縦指示を送信する方式です）

### 1. 事前準備
1. **Raspberry Pi の起動**:
   - Raspberry Piの電源を入れ、Rust APIサーバーを起動します（通常通りポート `8080` で待機）。
2. **Wi-Fi / 疎通の確認**:
   - PCとRaspberry Piが同一Wi-Fiネットワークに接続されていることを確認します（`.env` に記載の `ROBOT_BASE_URL`）。
3. **走行エリアへの配置**:
   - 車両をコースのスタート地点に置き、進行方向に向けておきます。

### 2. 実行コマンド (PCのPowerShell)
```powershell
# 通常起動 (2D動的オキュパンシーグリッド地図 ＋ リアルタイムGUIダッシュボード起動)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --speed 35 --min_speed 20

# GUIウィンドウを表示せずヘッドレスで走行する場合
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --no_gui

# カーブ旋回時・障害物接近時の減速チューニング (最低速度を25、減速開始距離を50cmに変更)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --speed 40 --min_speed 25 --slowdown_dist 50.0

# 通信ウォッチドッグのタイムアウト時間を調整 (デフォルト: 0.6秒)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --watchdog_timeout 1.0

# スタック検知感度を調整 (閾値を3.0、検知継続フレームを20フレームに変更)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --stuck_threshold 3.0 --stuck_frames 20
```

#### オプション説明:
- `--model` : 推論に使用するONNXモデルのパス（必須）
- `--speed` : 基本巡航スピード（`0` 〜 `100`、デフォルト: `35`）
- `--min_speed` : カーブ旋回時・障害物接近時の最低減速スピード（デフォルト: `20`）
- `--curve_factor` : 旋回減速感度係数（デフォルト: `1.0`）
- `--safe_dist` : 減速の計算に使う距離（cm、デフォルト: `15.0`）。停止は Pi が 15cm で行う
- `--slowdown_dist` : 障害物接近に伴い動的減速を開始する距離（cm、デフォルト: `40.0`）
- `--reverse_duration_ms` : 安全後退の目標実行時間（ミリ秒、デフォルト: `1200` = 1.2秒）
- `--stuck_threshold` : スタック判定の視覚変化量閾値（デフォルト: `4.0`）
- `--stuck_frames` : スタック確定までの連続フレーム数（デフォルト: `15`。さらに比較用の8フレームが必要で、10fps なら計約2秒）
- `--disable_stuck` : スタック検知・自動脱出ガードを無効化するフラグ
- `--no_gui` : 2Dマップウィンドウを表示せずヘッドレスで実行するフラグ
- `--grid_size` : グリッドマップのセル数（デフォルト: `200` = 10m x 10m）
- `--cell_size` : 1セルの解像度（メートル、デフォルト: `0.05` = 5cm）
- 段差検知は Pi が行い、走行開始時に今いる床で校正する（`--cliff_threshold` と `--disable_cliff` は廃止）。試験で無効にする時は Pi 側で `PICARX_DISABLE_CLIFF_GUARD=1` を設定する
- `--watchdog_timeout` : 通信監視タイムアウト秒数（デフォルト: `0.6` 秒）
- `--max_errors` : 通信停止とみなす連続エラー許容回数（デフォルト: `3` 回）

### 3. 多重セーフティレイヤー (Phase 3 ハイブリッド安全機構)

> [!WARNING]
> 停止に関わる機能（障害物、段差、通信断）は Pi の反射ループ（約40Hz）が行い、走行ループは約15Hz（2026-11-04 実機）で動きます。超音波は細い家具の脚や音を吸う物を見落とし、実走で脚に衝突しました（ROADMAP の M1）。段差からの落下や衝突を保証するものではないので、階段や段差の近くでは走行させず、すぐ停止できる状態で使用してください。詳細は [E2E_SPECIFICATION.md §4.5](../../../E2E_SPECIFICATION.md) を参照してください。

- **Phase 3.1 & 3.4 段差・階段落下防止 & 直前軌跡安全後退 (Breadcrumb Safety)**:
  - 底面赤外線センサーが床の途切れを検知した際、直前に通ってきた安全な走行軌跡（パンくずリスト）を逆順に再生しながら十分な距離（約1.2秒間）後退し、さらに安全な方向へ車首を振って完全脱出。
- **Phase 3.2 ダイナミック・スピードリミッター (Dynamic Speed Limiter)**:
  - 急旋回時や家具・壁への接近時に、AIの要求に関わらず安全な車速（`min_speed`）まで自動減速。スリップや旋回半径膨らみ、衝突衝撃を防止。
- **Phase 3.3 通信ウォッチドッグ (Communication Watchdog)**:
  - 通信の失敗を記録します。停止は Pi が行います（走行指令の TTL が切れると止まる）。
- **Phase 3.4 前方行き止まり自動脱出 (Dead-End Escape)**:
  - 壁や家具の前で2回連続で安全停止した場合、同様に通ってきたルートを逆走して後退・旋回し、自力で脱出。
- **Phase 3.5 動的オキュパンシーグリッド & リアルタイムUI (Dynamic Mapping & HUD)**:
  - 走行軌跡と前方超音波距離から 2D 確率格子地図（Occupancy Grid）を構築。自車位置、走ったルート、壁・家具の位置をリアルタイムにPC画面へ描画。
  - 時間経過による「忘却機能（Decay）」を備え、家具の配置変更やドアの開閉にも自動適応。
  - キー操作: `[C]` で地図クリア、`[S]` で現在の地図画像（`data/maps/map_xxx.png`）を保存、`[Q]` で安全停止して終了。
- **指令と実際の動きの食い違いの検出 (Motion Check、既定)**:
  - 空転や引っかかり（進む指令なのに景色も距離も変わらない）、滑りや舵のズレ（曲がり方が舵角と合わない）、押された（指令なしで景色全体が動いた）を、カメラの画像の動きと超音波から検出します。
  - 同じ食い違いが1秒続くと止まり、`data/disengagements/` に `motion_mismatch` として記録し、記憶にも残して走行を終えます。危険を押して動き続けないためです。
- **Phase 3.6 スタック検知 & 自動脱出ビヘイビア (Stuck & Wiggle Recovery、`--disable_motion_check` の時だけ)**:
  - 「前進しているはずなのに景色が動いていない」状態をカメラ画像の変化量から検知し、バック ＋ 左右の車体振り ＋ 迂回前進で脱出を試みます。
- **前方障害物ガード (超音波)**:
  - 前方15cm以内に家具・壁を検知すると、AIの推論に関わらず Pi の反射ループがモーターを止めます。`--safe_dist` は PC 側の減速にだけ使います。
- **Disengagement ロギング (直前30フレームの自動記録)**:
  - 急ブレーキ・段差回避・通信断・手動介入 (`Ctrl+C`) の直前30フレームを `data/disengagements/` へ自動保存します。約15Hz の走行ループでは約2秒に相当します。
- **手動緊急停止**:
  - ターミナルで **`Ctrl + C`** を押すと、即座にステアリングが0度（中立）に戻りモーターが完全停止。
