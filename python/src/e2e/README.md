# PiCar-X End-to-End (E2E) 自動運転 実践マニュアル

本書は、PiCar-Xにおいて **「データ収集 → モデル学習 → オフライン評価 → 実機自動走行」** の一連のE2E自律走行ループを実行するための手順書です。

> [!NOTE]
> 詳細な内部仕様、モデル構造、データスキーマ、MLOps設計については [E2E_SPECIFICATION.md](../../../E2E_SPECIFICATION.md) を参照してください。

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
1. 車体をコースのスタート位置に置きます。
2. **`R` キーを押して録画を開始**します（画面に「● Pause Recording」と表示され、フレーム数カウントが進みます）。
3. コースの中央をキープするようにスムーズに1〜2周（1,000〜2,000フレーム程度）走らせます。
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
3. **コースへの配置**:
   - 車両をコースのスタート地点に置き、進行方向に向けておきます。

### 2. 実行コマンド (PCのPowerShell)
```powershell
uv run python -m src.e2e.autonomous_driver --model models/v001/model.onnx --speed 35 --safe_dist 15.0
```

#### オプション説明:
- `--model` : 推論に使用するONNXモデルのパス（デフォルト: `models/v001/model.onnx`）
- `--speed` : 走行スピード（`0` 〜 `100`、デフォルト: `35`）
- `--safe_dist` : 強制停止する前方障害物までの距離（cm、デフォルト: `15.0`）

### 3. 走行中の動作と挙動
- **推論ループ (約20Hz)**:
  - PCがRustサーバーからカメラ映像（JPEG）をリアルタイム取得します。
  - ONNXモデルが現在のステアリング角度を推論し、Rustサーバー経由で車輪を自動操舵しながら前進します。
- **Safety Guard (安全機構)**:
  - 超音波センサーが指定距離（デフォルト15cm以内）の障害物を検知した場合、推論値に関わらず**モーターを強制停止**します。
- **緊急停止**:
  - 走行を止めたいときは、ターミナルで **`Ctrl + C`** を押すと、ステアリングが中央（0度）に戻りモーターが完全停止します。
