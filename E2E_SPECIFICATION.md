# PiCar-X End-to-End (E2E) 自律走行 & MLOps システム仕様書

本書は、SunFounder PiCar-X において実装された、**「軽量CNNによる画像入力からの直接舵角予測 (End-to-End)」** および **「データ収集・品質管理・モデル管理 (MLOps)」** の全体アーキテクチャ・各サブシステム詳細仕様・開発ロードマップを一元的に定義する。

> [!NOTE]
> - 全体の開発ロードマップ・進捗状況については [ROADMAP.md](ROADMAP.md) を参照してください。
> - ユーザー向けの操作手順については [python/src/e2e/README.md](python/src/e2e/README.md) を参照してください。

---

## 1. 基本方針 & スコープ

### 1.1 基本思想
- **家庭内フリースペース自律走行**: コース・白線・標識のない家庭内（部屋・廊下・家具の間）を縦横無尽に走行し、障害物や行き止まりを自律回避して開けた空間へナビゲートする。
- **GPU完全不要 (CPU前提)**: PC・Raspberry PiともにCPUのみで高速動作する軽量設計。
- **実機車載アーキテクチャ (Plan A Remote Control)**:
  - PC側: データ収集GUI、データクリーニング、PyTorch学習、ONNX Runtime推論、MLOpsレジストリを担当。
  - Raspberry Pi側: Rust/Axum HTTPサーバー経由でカメラ撮影、超音波センサー計測、モーター・サーボPWM制御を担当。
  - 両者間は標準 HTTP/JSON 通信（Base64 JPEG画像伝送）で疎結合に連携。

### 1.2 対象外 (スコープ外)
- ❌ **巨大モデル（Vision Transformer / LLM制御）**: GPUを要し推論遅延が大きいため採用しない。
- ❌ **重厚な3Dシミュレーター（Unreal Engine / CARLA）**: 独立GPUリソースを消費するため採用しない。
- ❌ **大規模時系列ディープラーニング（LSTM/動画モデル）**: 計算コストとメモリ消費を抑えるため除外。

### 1.3 注力分野
- ⭕ **MLOps & 実験管理**: データセット・モデルのバージョン管理と定量評価の自動化。
- ⭕ **継続学習（Human-in-the-loop）**: 失敗ケースの切り出しとデータリバランスによる実走性能向上。
- ⭕ **車載安全系（Safety Layer）**: AI推論に依存しない多重センサーFail-Safeの構築。
- ⭕ **エッジ完全自律化（Rust推論）**: Raspberry Pi単体で完結する超低遅延ONNX推論。
- ⭕ **ツール前提**: Python環境操作は常に `uv`（`uv run` / `uv pip`）を使用する。

---

## 2. 全体構成図

```mermaid
flowchart TD
    subgraph PC["ローカルPC (Python 3.13 / uv)"]
        GC["データ収集 GUI<br>(gui_collector.py)"]
        RawData[("生データ<br>data/raw/")]
        DC["データクリーナー<br>(cleaner.py)"]
        CleanData[("クリーンデータ<br>data/processed/")]
        TR["モデル学習 & ONNXエクスポート<br>(train.py)"]
        Models[("モデルレジストリ<br>models/vXXX/")]
        EV["モデル評価<br>(evaluate.py)"]
        CP["MLOps 比較 CLI<br>(compare.py)"]
        AD["自律走行エンジン<br>(autonomous_driver.py)"]
    end

    subgraph Pi["Raspberry Pi (Rust / Axum Server :8080)"]
        API["HTTP API サーバー<br>(/api/*)"]
        Calib[("校正ファイル<br>/opt/picar-x/picar-x.conf")]
        Cam["カメラ (rpicam-still)"]
        US["超音波センサー"]
        HW["サーボ & モーター PWM<br>(自動オフセット補正適用)"]
    end

    GC -- "キーボード操縦" --> API
    API -- "Snapshot (Base64 JPEG)" --> GC
    GC --> RawData
    RawData --> DC
    Models -. "残差推論" .-> DC
    DC --> CleanData
    CleanData --> TR
    RawData -. "全結合学習" .-> TR
    TR --> Models
    Models --> EV
    Models --> CP
    RawData & CleanData --> CP
    Models -- "ONNXモデルロード" --> AD
    API -- "リアルタイム画像受信" --> AD
    AD -- "推論舵角 (steer)" --> API
    AD -- "超音波安全監視" --> API
    Calib -. "自動読込" .-> HW
```

---

## 3. 開発ロードマップについて

プロジェクト全体のフェーズ計画（Step 1〜4、Phase 1〜5）および最新の進捗状況は、[ROADMAP.md](ROADMAP.md) に集約・一元管理されています。

---

## 4. 各サブシステム詳細仕様

### 4.1 データ収集サブシステム (`gui_collector.py`)
PC上でカメラ映像をリアルタイム確認しながら手動操縦し、高品質な教師データを記録する。

- **モジュールパス**: `python/src/e2e/gui_collector.py`
- **主要機能**:
  - PySide6 による GUI 操作画面
  - `PicarxClient` (HTTPX 同期通信) を用いたリアルタイム遠隔操作
  - 録画トグル（`R` キー / UIボタン）による必要な走行シーンのみの抽出保存
  - 画像と操舵ラベルの完全同期書き込み
- **操作キーアサイン**:
  - `W`: 前進 (設定速度)
  - `S`: 停止 (車速0)
  - `A`: 左ステアリング (10° 刻み)
  - `D`: 右ステアリング (10° 刻み)
  - `X`: ステアリング中立 (0°)
  - `R`: 録画開始 / 一時停止トグル
- **保存データ形式**: `python/data/raw/<session_timestamp>/`
  - `images/frame_xxxxxx.jpg`: 解像度 160×120 の JPEG 画像
  - `labels.jsonl`: 各フレームの JSONL レコード
    ```json
    {
      "timestamp": 1729000000.123,
      "frame": "frame_000001.jpg",
      "steering": -0.2857,
      "throttle": 0.5,
      "speed": 35,
      "ultrasonic": 42.5
    }
    ```
    ※ `steering` は最大舵角 ±35° を基準に `-1.0` (最大左折) 〜 `+1.0` (最大右折) に正規化。

---

### 4.2 データセット管理 & 自動クリーニング (`dataset.py`, `cleaner.py`)

#### 4.2.1 複数セッション統合データセット (`training/dataset.py`)
- **モジュールパス**: `python/src/e2e/training/dataset.py`
- **クラス**: `DrivingDataset(Dataset)`
- **仕様**:
  - 引数 `data_dir` に「単一セッション」「複数セッションのリスト」「親ディレクトリ (`data/raw`, `data/processed`)」のいずれを指定しても、配下の `labels.jsonl` を自動探索して1つのデータセットに結合。
  - 各画像のパスをセッションごとに正しく解決。
  - 前処理パイプライン:
    1. リサイズ: 120×160 ピクセル (H×W)
    2. Tensor 変換: `[0.0, 1.0]`
    3. 正規化: ImageNet 基準 (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

#### 4.2.2 残差外れ値フィルタ（データクリーナー） (`dataset/cleaner.py`)
- **モジュールパス**: `python/src/e2e/dataset/cleaner.py`
- **クラス / エントリポイント**: `DatasetCleaner`, `clean_dataset()`
- **検出アルゴリズム (Data-Centric AI)**:
  1. 指定された学習済みモデル（`.onnx` または `.pth`）で全フレームのステアリング角 $\hat{y}$ を推論。
  2. 手動入力ラベル $y$ との絶対残差誤差 $e = |\hat{y} - y|$ を算出。
  3. 閾値判定:
     - 絶対誤差閾値 (`--threshold`, デフォルト `0.4` ≒ 約14°以上の乖離) を超えるフレーム
     - またはワーストパーセンタイル (`--top_k_percent`) に該当するフレーム
     これらを「操作遅れ」「衝突後の混乱」「手ブレノイズ」として除外。
- **出力アーティファクト**: `python/data/processed/<output_dir>/`
  - `images/`: セッションID接頭辞を付与したユニークな画像コピー
  - `labels.jsonl`: クリーニング済みの正常ラベル
  - `bad_cases.jsonl`: 除外されたフレーム一覧と除外理由（デバッグ・分析用）
  - `cleaning_report.json`: 除外率、前後MAEの定量的改善レポート
  - `metadata.json`: レジストリ互換の統計メタデータ

---

### 4.3 モデルアーキテクチャ & 学習 (`training/model.py`, `train.py`)

#### 4.3.1 モデル仕様 (`SmallDrivingCNN`)
CPU推論時の超低遅延（数ミリ秒）と高い表現力を両立した4層CNN。

- **入力**: `[Batch, 3, 120, 160]` (RGB画像)
- **出力**: `[Batch, 1]` (予測ステアリング角, `-1.0` 〜 `+1.0`)
- **ネットワーク構造**:
  ```text
  Input (3, 120, 160)
    │
    ├── Conv2d(3 -> 16, 3x3, stride=2, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2) -> (16, 30, 40)
    ├── Conv2d(16 -> 32, 3x3, stride=1, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2) -> (32, 15, 20)
    ├── Conv2d(32 -> 64, 3x3, stride=1, padding=1) + BatchNorm2d + ReLU + MaxPool2d(2) -> (64, 7, 10)
    ├── Conv2d(64 -> 64, 3x3, stride=1, padding=1) + BatchNorm2d + ReLU                 -> (64, 7, 10)
    │
    ├── AdaptiveAvgPool2d((2, 2)) -> (64, 2, 2) = 256次元
    ├── Flatten
    │
    ├── Linear(256 -> 64) + ReLU + Dropout(0.2)
    ├── Linear(64 -> 32) + ReLU
    └── Linear(32 -> 1) + Tanh
  ```

#### 4.3.2 学習仕様 (`training/train.py`)
- **損失関数**: Mean Squared Error (MSE Loss)
- **最適化**: Adam (`lr=1e-3`)
- **データ分割**: Train 80% / Validation 20%
- **エクスポート**:
  - 最良検証ロスの重み: `best_model.pth`
  - ONNX 形式: `model.onnx` (Opset 17, 入出力動的バッチ対応)
  - メタデータ: `metadata.json` (エポック数、バッチサイズ、使用セッション一覧、最良Val Loss)

---

### 4.4 オフライン評価 & MLOps レジストリ (`evaluation/`, `registry/`)

- **オフライン評価 (`evaluate.py`)**:
  - 平均絶対誤差 (MAE) を正規化スケールおよび実角度（±◯度）で算出。
  - 評価結果を `eval_metrics.json` に自動出力。
- **データセットレジストリ (`registry/dataset.py`)**:
  - セッションごとのフレーム数、走行時間、FPS、直進/左折/右折比率、平均操舵角を自動分析。
- **モデルレジストリ (`registry/models.py`)**:
  - 保存済みモデルのバージョン、エポック数、検証損失、MAE、ONNX/PTH成果物の有無を走査。
- **MLOps 比較 CLI (`registry/compare.py`)**:
  - `uv run python -m src.e2e.registry.compare` でターミナル上に Rawデータセット、Processedデータセット、モデル性能を表形式で一覧表示。

---

### 4.5 自律操縦 & セーフティレイヤー (`autonomous_driver.py`)

- **推論ループ (約20Hz)**:
  1. `/api/camera/snapshot` からカメラ画像（Base64 JPEG）を受信
  2. OpenCV / PIL で 120×160 にリサイズ・正規化
  3. ONNX Runtime でステアリング角を推論
  4. 推論値（`-1.0` 〜 `+1.0`）を実機角度（`-35°` 〜 `+35°`）にスケーリング
  5. `/api/drive/steer` へ送信し、一定スロットルで前進
- **多重セーフティガード (Fail-Safe)**:
  - **超音波安全停止**: `/api/sensor/ultrasonic` が指定距離（デフォルト 15.0cm）以下を検知した場合、推論舵角に関わらず**モーターを即時停止**。
  - **緊急停止 (Ctrl+C)**: シグナルハンドラにより、プログラム中断時にステアリング角度0（中立）へ復帰しモーターを停止。

---

### 4.6 ハードウェア校正自動補正レイヤー (`rust/src/picarx/core.rs`)

PiCar-X実機のサーボモーターギア個体差（数度の傾き）を吸収する機構。

- **校正設定ファイルの自動読み込み**:
  - 起動時に `/opt/picar-x/picar-x.conf`、`/etc/picar-x/picar-x.conf`、`~/.config/picar-x/picar-x.conf` を自動探索。
  - `picarx_cam_pan_servo`, `picarx_cam_tilt_servo`, `picarx_dir_servo` をパース。
- **環境変数オーバーライド**:
  - `CAM_PAN_OFFSET`, `CAM_TILT_OFFSET`, `DIR_SERVO_OFFSET` での上書きに対応。
- **駆動時の自動オフセット加算**:
  ```rust
  set_cam_pan_angle(angle + pan_offset);
  set_cam_tilt_angle(angle + tilt_offset);
  set_dir_servo_angle(angle + dir_offset);
  ```
  これにより、Rustサーバー起動直後からカメラが正確に真正面を向き、直進操舵のズレが排除される。

---

## 5. 各フェーズ詳細設計（未実装含む）

### Phase 1: 軽量 MLOps & モデルレジストリ ✅ 完了
- **目的**: データとモデルのライフサイクルを可視化・定量比較できるようにする。
- **実装内容**:
  1. **Dataset Registry (`registry/dataset.py`)**: 各走行ログの統計情報（総フレーム数、走行時間、直進/左折/右折比率、平均舵角）を算出してメタデータ化。
  2. **Model Registry (`registry/models.py`)**: `models/v001`, `models/v002` ... の学習設定、エポック数、検証損失、MAE、入力解像度を一元管理。
  3. **Model Comparator CLI (`registry/compare.py`)**: `uv run python -m src.e2e.registry.compare` で登録済みモデルの性能をターミナル上に表形式で比較表示。
- **計算負荷**: ゼロ（メタデータ管理・集計のみ）

---

### Phase 2: データフライホイール & 継続学習 🔧 一部完了
- **目的**: 失敗した走行データから効率的にリカバリーを学習させ、実走性能を向上させる。
- **実装内容**:
  1. **Disengagement Logger (危険・介入検知)** ✅ 完了:
     - 自動走行中の「手動緊急停止（Ctrl+C）」や「超音波センサー急接近（急ブレーキ作動）」の直前3秒間を自動抽出し、`bad_case` としてタグ付け。
  2. **Dataset Merging & Balancing** ✅ 完了:
     - 複数セッションの統合学習対応済み。残差データクリーナーによる品質向上も実装済み。
  3. **Retraining Pipeline** ✅ 完了:
     - 苦手箇所のデータを補強したデータセットで再学習し、`models/v002`〜`v004` を生成。`clean_v1`/`clean_v2` でクリーンデータセットも生成済み。
- **計算負荷**: 非常に軽量（CPUで数分〜5分程度）

---

### Phase 3: ハイブリッド安全系 & 空間マッピング（Safety & Spatial Mapping）🔧 一部完了
- **目的**: AI推論の誤判定や通信途絶が起きても家具・壁への衝突や段差・階段からの落下を防ぐ多重 Fail-Safe アーキテクチャの構築に加え、後方センサーを持たない PiCar-X のための軌跡記憶と動的マッピングを実現する。
- **実装内容**:
  1. **Phase 3.1: 多重センサー防護 (Cliff Detection)** ✅ 完了:
     - 底面赤外線グレースケールセンサーによる段差・階段落下防止 (src/e2e/safety/cliff_detector.py)。床面途切れ検知時の即時停止と左右方向性を持った自動後退退避。
  2. **Phase 3.2: ダイナミック・スピードリミッター (Dynamic Speed Limiter)** ✅ 完了:
     - 舵角および前方障害物接近距離に応じた動的減速制御 (src/e2e/safety/speed_limiter.py)。直進時は巡航速度、急旋回時や障害物接近時は最低速度までスムーズに自動減速。
  3. **Phase 3.3: 通信監視・ウォッチドッグ (Communication Watchdog)** ✅ 完了:
     - PC-Raspberry Pi 間のハートビート監視 (src/e2e/safety/watchdog.py)。Wi-Fi途絶やAPI連続エラー時に即座に強制停止し暴走を防止。
  4. **Phase 3.4: 直前軌跡メモリ & 安全後退 (Breadcrumb Safety)** ✅ 完了:
     - 直前3〜5秒間の「車速・操舵角・推定相対座標」をリングバッファに常時保持。
     - 後方にセンサーがない制約をカバーするため、段差回避や前方行き止まり時に「直前に走行して安全が確認されているルート」を逆再生して自動後退 (`src/e2e/safety/trajectory_tracker.py`)。
  5. **Phase 3.5: 動的オキュパンシーグリッド (Dynamic Mapping & Visualization)** ✅ 完了:
     - 走行軌跡と前方超音波距離を積算し、2D確率格子地図（Occupancy Grid）をリアルタイム構築。
     - 家具の移動やドアの開閉に対応するため、時間経過に伴う「忘却係数（確率の減衰）」を導入。
     - PC 側画面にリアルタイムで自車位置・通った軌跡・検知した壁や家具のグリッドを描画・可視化。
  6. **Phase 3.6: スタック検知 & 自動脱出 (Stuck & Wiggle Recovery)** ✅ 完了:
     - モーター前進指令中にもかかわらずカメラ画像の変位（フレーム間差分）が動かない状態を検出し、ラグ段差や家具脚への引っかかり・車輪空転を自動検知 (`src/e2e/safety/stuck_detector.py`)。
     - 直前軌跡バック ＋ 左右の車体振り（Wiggle） ＋ 迂回前進の複合シーケンスで自力脱出。
  7. **Disengagement 自動記録連携**:
     - 崖検知、超音波急接近、ウォッチドッグタイムアウト、手動停止 (Ctrl+C) の各事象発生時に、直前3秒間 (30フレーム) を ad_case として自動タグ付け保存。
- **計算負荷**: 非常に軽量（Python側でのシンプルな算術更新・OpenCV描画）
### Phase 4: Raspberry Pi 実機スタンドアロン化（Rust推論移植）⬜ 未着手
- **目的**: Wi-FiやPCを介さず、Raspberry Pi単体・完全自律で動作する「本物のロボット」へ昇華。
- **実装内容**:
  1. **Rust ONNX Runtime (`ort` crate) 組み込み**:
     - PCで学習した `model.onnx` をRaspberry Piへデプロイ。
     - Raspberry Pi上のRustプログラム内でカメラキャプチャからONNX推論、PWM出力までを直接実行。
  2. **超低遅延化**:
     - Wi-Fi通信オーバーヘッド（30〜50ms）がなくなり、20〜30FPSで遅延のない滑らかな自律走行を実現。
  3. **PC側役割の分離**:
     - PCは「走行ログの受信」「学習」「モデル配信」のみを担当し、リアルタイム制御ループから完全に外す。
- **計算負荷**: Raspberry PiのCPUのみで余裕で動作（軽量CNNのため1推論あたり数ms〜十数ms）

---

## 6. ディレクトリ構成

```text
PiCarX/
├── E2E_SPECIFICATION.md           # 本仕様書 (ロードマップ統合)
├── python/
│   ├── pyproject.toml             # uv / Python 依存関係定義 (e2e, dev)
│   ├── data/
│   │   ├── raw/                   # 生データセッション (タイムスタンプ別)
│   │   │   └── <session_id>/
│   │   │       ├── images/
│   │   │       └── labels.jsonl
│   │   └── processed/             # クリーン化済みデータセット
│   │       └── clean_vX/
│   │           ├── images/
│   │           ├── labels.jsonl
│   │           ├── bad_cases.jsonl
│   │           └── cleaning_report.json
│   ├── models/                    # 学習済みモデルレジストリ
│   │   └── vXXX/
│   │       ├── best_model.pth
│   │       ├── model.onnx
│   │       ├── metadata.json
│   │       └── eval_metrics.json
│   ├── src/
│   │   └── e2e/
│   │       ├── README.md          # ユーザー向け操作マニュアル
│   │       ├── autonomous_driver.py # 自律操縦実行スクリプト
│   │       ├── gui_collector.py   # データ収集GUI
│   │       ├── dataset/           # データクリーナー & ユーティリティ
│   │       │   ├── __init__.py
│   │       │   └── cleaner.py
│   │       ├── evaluation/        # 評価スクリプト
│   │       │   └── evaluate.py
│   │       ├── registry/          # MLOps レジストリ & CLI
│   │       │   ├── __init__.py
│   │       │   ├── compare.py
│   │       │   ├── dataset.py
│   │       │   └── models.py
│   │       └── training/          # モデル定義 & 学習
│   │           ├── dataset.py
│   │           ├── model.py
│   │           └── train.py
│   └── tests/                     # ユニットテスト (全30件)
│       ├── test_e2e_cleaner.py
│       └── test_e2e_dataset.py
└── rust/
    └── src/
        └── picarx/
            └── core.rs            # 校正値自動読込付きハードウェア実装
```

---

## 7. 実行コマンド集 (uv前提)

| 目的 | コマンド (PowerShell) |
|---|---|
| **データ収集GUI起動** | `uv run python -m src.e2e.gui_collector` |
| **生データ全結合学習** | `uv run python -m src.e2e.training.train --data_dir data/raw --output_dir models/v004 --epochs 25` |
| **データクリーニング (外れ値除外)** | `uv run python -m src.e2e.dataset.cleaner --data_dir data/raw --model models/v004/model.onnx --threshold 0.4 --output_dir data/processed/clean_v2` |
| **クリーンデータでの再学習** | `uv run python -m src.e2e.training.train --data_dir data/processed/clean_v2 --output_dir models/v005 --epochs 25` |
| **モデルオフライン評価** | `uv run python -m src.e2e.evaluation.evaluate --data_dir data/raw --model_path models/v004/best_model.pth` |
| **MLOps レジストリ確認** | `uv run python -m src.e2e.registry.compare` |
| **実機自律走行** | `uv run python -m src.e2e.autonomous_driver --model models/v004/model.onnx --speed 35` |
| **介入イベント一覧確認** | `uv run python -m src.e2e.dataset.disengagement_manager --list` |
| **Rustクロスコンパイル** | `cd rust/build; make build MODEL=4b` (ローカルPC Docker上) |
| **Rustラズパイ送信** | `cd rust/build; make deploy RPI_HOST=pi@<IP> RPI_DIR=/home/pi/` |
| **CI 静的解析** | `uv run ruff check --config pyproject.toml src tests` |
| **CI テスト実行** | `uv run pytest tests` |

---

## 8. 開発プロセス & 品質ルール

- **Python実行ルール**: 常に `uv` を前提とする（`uv run python -m ...`、`uv pip install ...`）。
- **Rustビルド & デプロイルール**: 実機でのコンパイルは行わず、常にローカルPC上でDockerクロスコンパイル（`rust/build/Makefile`）を行い、`scp`（`make deploy`）でRaspberry Piへバイナリを送信する。
- **コード品質**:
  - リンター/フォーマッタ: `uv run ruff check --config pyproject.toml src tests`
  - ユニットテスト: `uv run pytest tests`
  - 新機能追加時も常にCIテスト全件合格を維持する。
