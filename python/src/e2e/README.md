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
make build MODEL=4b     # お使いの機種 (4b, zero2w, 3bplus, 5)

# Raspberry Pi へ実行バイナリを転送
make deploy RPI_HOST=pi@<ラズパイIP> RPI_DIR=/home/pi/

# Raspberry Pi 上で起動 (SSH接続先)
./picar-x-rust
```

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
[ 4. 実機自動走行 (Raspberry Pi) ]
    ONNX Runtimeによる自律操縦 + 超音波センサー安全ガード (Safety Layer)
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
  - `labels.jsonl`: 各フレームのタイムスタンプ、ステアリング正規化値（-1.0〜+1.0）、速度

---

## ステップ 2: モデル学習 & ONNXエクスポート (PC)

収集した走行ログをもとに、画像からステアリング角度を予測する軽量CNNを学習します。
**過去の全走行データ（`data/raw`）の一括統合学習**、または特定セッションの個別学習が可能です。

### 実行コマンド (PowerShell)

#### 過去の全走行データをまとめて学習する場合 (推奨)
```powershell
uv run python -m src.e2e.training.train --data_dir data/raw --output_dir models/v004 --epochs 15
```

#### 特定のセッションのみを学習する場合
```powershell
uv run python -m src.e2e.training.train --data_dir data/raw/<session_id> --output_dir models/v001 --epochs 15
```
※ 複数フォルダを指定したい場合は `--data_dir data/raw/session1 data/raw/session2` のようにスペース区切りで複数指定できます。

### 生成物
- `models/v004/best_model.pth` : PyTorchモデル重み
- `models/v004/model.onnx` : 実機配信用ONNXモデル
- `models/v004/metadata.json` : 最良検証損失、ハイパーパラメータ、読み込んだセッション一覧の記録

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

### クリーニング後の再学習
厳選された良質データセットを用いて、より高精度な新モデルを学習します：
```powershell
uv run python -m src.e2e.training.train --data_dir data/processed/clean_v1 --output_dir models/v004 --epochs 15
```

---

## ステップ 3: モデル評価 (PC オフライン検証)

テストデータに対する予測精度の誤差（MAE: 平均絶対誤差）を測定します。

### 実行コマンド (PowerShell)
```powershell
uv run python -m src.e2e.evaluation.evaluate --data_dir data/raw/<session_id> --model_path models/v001/best_model.pth
```

### 結果表示例
```text
--- Evaluation Results ---
Total Samples : 1250
Steering MAE  : 0.0821 (Normalized -1.0 to 1.0)
Steering MAE  : ±2.87° (Degrees)
--------------------------
```
※ 平均誤差が数度（±2〜3度以内）に収まっていれば良好です。
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

## ステップ 4: 自動走行 (Raspberry Pi 実機での走行)

ステップ 2 で生成された `models/v001/model.onnx` を使い、実機で自動運転を実行します。
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
# 通常起動 (多重安全層: 前方停止 + 段差ガード + 動的カーブ減速 + 通信ウォッチドッグ有効)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --speed 35 --min_speed 20

# カーブ旋回時・障害物接近時の減速チューニング (最低速度を25、減速開始距離を50cmに変更)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --speed 40 --min_speed 25 --slowdown_dist 50.0

# 段差検知の閾値を調整する場合 (床の材質に合わせて変更、デフォルト: 150)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --cliff_threshold 200

# 通信ウォッチドッグのタイムアウト時間を調整 (デフォルト: 0.6秒)
uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --watchdog_timeout 1.0
```

#### オプション説明:
- `--model` : 推論に使用するONNXモデルのパス（デフォルト: `models/v001/model.onnx`）
- `--speed` : 基本巡航スピード（`0` 〜 `100`、デフォルト: `35`）
- `--min_speed` : カーブ旋回時・障害物接近時の最低減速スピード（デフォルト: `20`）
- `--curve_factor` : 旋回減速感度係数（デフォルト: `1.0`）
- `--safe_dist` : 緊急停止する前方障害物までの限界距離（cm、デフォルト: `15.0`）
- `--slowdown_dist` : 障害物接近に伴い動的減速を開始する距離（cm、デフォルト: `40.0`）
- `--cliff_threshold` : 段差・階段落下検知のADC閾値（デフォルト: `150`。これを下回ると崖判定）
- `--disable_cliff` : 段差検知ガードを無効化するフラグ
- `--reverse_duration_ms` : 安全後退の目標実行時間（ミリ秒、デフォルト: `1200` = 1.2秒）
- `--watchdog_timeout` : 通信監視タイムアウト秒数（デフォルト: `0.6` 秒）
- `--max_errors` : 通信停止とみなす連続エラー許容回数（デフォルト: `3` 回）

### 3. 多重セーフティレイヤー (Phase 3 ハイブリッド安全機構)
- **Phase 3.1 & 3.4 段差・階段落下防止 & 直前軌跡安全後退 (Breadcrumb Safety)**:
  - 底面赤外線センサーが床の途切れを検知した際、直前に通ってきた安全な走行軌跡（パンくずリスト）を逆順に再生しながら十分な距離（約1.2秒間）後退し、さらに安全な方向へ車首を振って完全脱出。
- **Phase 3.2 ダイナミック・スピードリミッター (Dynamic Speed Limiter)**:
  - 急旋回時や家具・壁への接近時に、AIの要求に関わらず安全な車速（`min_speed`）まで自動減速。スリップや旋回半径膨らみ、衝突衝撃を防止。
- **Phase 3.3 通信ウォッチドッグ (Communication Watchdog)**:
  - Wi-Fi通信断やRust APIサーバーの応答遅延（0.6秒以上）を常時監視。通信途絶時に自動で車両を即時緊急停止し、暴走を防止。通信復帰時は自動再開。
- **Phase 3.4 前方行き止まり自動脱出 (Dead-End Escape)**:
  - 壁や家具の前で2回連続で安全停止した場合、同様に通ってきたルートを逆走して後退・旋回し、自力で脱出。
- **前方障害物ガード (超音波)**:
  - 前方指定距離（デフォルト15cm）以内に家具・壁を検知すると、AIの推論に関わらずモーターを完全強制停止。
- **Disengagement ロギング (直前3秒間の自動記録)**:
  - 急ブレーキ・段差回避・通信断・手動介入 (`Ctrl+C`) の直前3秒間を `data/disengagements/` へ自動保存。
- **手動緊急停止**:
  - ターミナルで **`Ctrl + C`** を押すと、即座にステアリングが0度（中立）に戻りモーターが完全停止。
