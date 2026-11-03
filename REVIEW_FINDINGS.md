# PiCar-X 実装レビュー指摘事項

- **レビュー日**: 2026-11-03
- **対象**: `feature/e2e-safety-layer`（コミット `10cd748` 時点）。行番号はこの時点のもの
- **目的**: 最終目標（人と話しながら家の構造を把握して自由に走る）に対し、実装に残る課題を一覧化する
- **修正状況**: 本書の指摘は **未修正のコード課題**。文書側の矛盾は同日に修正済み（[§7](#7-文書の修正記録)）
- **計画との対応**: 「M」は [ROADMAP.md](ROADMAP.md) のマイルストーンを指す

## 検証の範囲

| 区分 | 内容 |
|---|---|
| 実施 | コードと文書の読解、`python/data` の記録ログ解析、Python テスト実行（47件成功）、ruff（1件検出） |
| 未実施 | 実機での動作確認、Rust テストの実行 |
| 推定を含む | 遅延や誤検知の **原因**、部品価格。本文では「推定」と明記する |

## 重要度

- **S1**: 衝突、落下、停止不能、情報漏えいにつながる
- **S2**: 最終目標の実現を妨げる
- **S3**: 品質、保守性、評価の信頼性に関わる

## 1. 実測で判明した仕様との差

記録ログ（`python/data`）の解析結果。

| 項目 | 仕様書の旧記載 | 実測 | 影響 |
|---|---|---|---|
| 走行ループ | 約20Hz | フレーム間隔の中央値 **1.14秒**、p10 1.04秒、p90 2.59秒（2,930間隔） | 全機能の時間前提が崩れる |
| 介入前の記録 | 直前3秒（30フレーム） | 30フレームで約30秒以上 | 失敗直前だけを切り出せない |
| スタック検知 | 約1秒 | 最短でも約25秒（8 + 15 フレームからの計算値） | 実質的に機能しない |
| 学習データ | 継続学習が完了 | 150フレーム（約2.5分） | 家庭内走行の学習には少なすぎる |
| ラベル分布 | — | 左101 / 直進37 / 右12（最大左舵のみで60） | 左に偏る恐れ |
| 通信タイムアウト | 通信断で作動 | 全111イベント中57件。**57件すべて連続エラー0回** | HTTP 失敗ではない発火が大半 |

1秒周期の原因は、撮影ごとの `rpicam-still` プロセス起動と推定する（内訳は未計測）。

## 2. 安全系

| ID | 重要度 | 内容 | 根拠 | 対応案 | M |
|---|---|---|---|---|---|
| SAF-2 | S1 | Pi 側の通信タイムアウト判定 `is_timed_out` が未使用 | `rust/src/safety/mod.rs:45` | TTL 付き走行コマンドと反射ループで置き換える | M1 |
| SAF-4 | S1 | Rust 側に段差検知がない。距離チェックは発進時の1回のみ | `rust/src/api/handlers.rs:34-45` | Pi 内の反射ループ（20Hz以上）で走行中も常時監視 | M1 |
| SAF-5 | S1 | 400ms 移動（未校正の係数で約11cm）の間は PC 側で監視しない | `python/src/e2e/autonomous_driver.py:609` | 同上 | M1 |
| SAF-6 | S1 | 脱出動作の穴。脱出後に軌跡を消去し、続けて発動すると根拠のない直進後退になる。旋回やウィグル中は段差を確認しない | `autonomous_driver.py:216-220,242,284-296`, `python/src/e2e/safety/trajectory_tracker.py:84-85` | 反射層での常時監視を前提に作り直す。軌跡切れの後退は拒否する | M1 |
| SAF-7 | S1 | 読み取り失敗時に前回値を保持し、時刻がない | `rust/src/sensor_monitor/core.rs:49-52` | 全センサー値に時刻を付け、古い値は安全側に倒す | M1 |
| SAF-8 | S1 | 通信失敗を安全扱い（距離 `999`、床 `[999,999,999]`）。距離 `0` も障害物なし扱い | `autonomous_driver.py:438,478,486` | 「不明」を型で表し、不明なら走らない | M1 |
| SAF-12 | S2 | 段差閾値の校正が未使用で150固定。`[8,13,11]` のような値が記録されており、床色による誤検知の可能性あり（推定） | `python/src/e2e/safety/cliff_detector.py:32` | 起動時に校正し、Pi 側で保持 | M1 |
| SAF-13 | S2 | 古い `stop` スレッドが新しい移動を止める。ループを高速化すると顕在化する | `rust/src/api/hardware.rs:30-51` | スレッド方式をやめ、期限管理に統一 | M1 |
| SAF-14 | S2 | ウォッチドッグがループ先頭でしか評価されず、HTTP タイムアウト5秒の間は動かない。誤発火の原因は未特定 | `autonomous_driver.py:393`, `python/src/config.py:35` | Pi 側へ移管。PC 側は表示用途に縮小 | M1 |
| SAF-15 | S2 | センサーのポーリングが 300ms 間隔で、1回の測定中に共有 Mutex を保持する | `rust/src/sensor_monitor/core.rs:9,42-46` | ポーリングを短縮し、モーターとセンサーのロックを分離 | M1 |

## 3. 地図と自己位置

ロードマップの旧記述はズレの原因をスリップとしていたが、先に直すべきバグがある。

| ID | 重要度 | 内容 | 根拠 | 対応案 | M |
|---|---|---|---|---|---|
| MAP-1 | S2 | **地図が左右反転**。右操舵（正の角度）で θ が増え、画面では左に曲がる。旋回方向のテストがない | `python/src/e2e/mapping/occupancy_grid.py:97`, `visualizer.py:224-225` | 符号を修正し、旋回方向のテストを追加 | M4 |
| MAP-2 | S2 | **移動距離を半分以下に記録**。実移動 400ms に対し記録は 200ms が上限。後退用の軌跡も同様 | `autonomous_driver.py:381,591-604,609` | 実際に指令した移動時間を記録する | M4 |
| MAP-3 | S2 | **走っていない移動も記録**。安全ガードの拒否を確認せず、ハートビートも更新する | `autonomous_driver.py:607-611`, `python/src/picarx_client/client.py:101-107` | 応答を確認し、拒否時は記録しない | M4 |
| MAP-4 | S2 | 通信失敗や距離 `0` を地図へ「空き」として書き込む | `autonomous_driver.py:478-482`, `occupancy_grid.py:126-129` | 不明な観測は地図に反映しない | M4 |
| MAP-5 | S2 | 地図の状態を保存・読込できず、終了で失われる（画像保存のみ） | `occupancy_grid.py`, `visualizer.py:362-363` | 場所グラフと共に永続化 | M4 |
| MAP-6 | S2 | 前方超音波1本と推測航法では家全体の地図は作れない | 構成上の制約 | カメラによる場所の認識と場所グラフへ変更。不足すれば M6 でタグとジャイロを追加 | M4 |
| MAP-7 | S3 | 速度係数 `0.008` とホイールベース `0.14` が未校正 | `occupancy_grid.py:23-24` | 実測して設定値にする | M4 |

## 4. 学習

| ID | 重要度 | 内容 | 根拠 | 対応案 | M |
|---|---|---|---|---|---|
| LRN-1 | S2 | 介入データのラベルがモデル自身の出力。承認すると失敗を正解として学習する | `python/src/e2e/autonomous_driver.py:582-588`, `python/src/e2e/dataset/disengagement_manager.py:88,217` | 人の修正操作を記録する方式に変更。それまで承認機能は使わない | M3 |
| LRN-2 | S2 | E2E モデルは舵角のみで、目的地の概念がない | `python/src/e2e/training/model.py` | E2E は短距離の走行に限定。行き先は上位層が担当 | M4, M5 |
| LRN-3 | S2 | データが少なく偏っている（[§1](#1-実測で判明した仕様との差)）。1fps で収集されている | `python/data/raw` | 10fps 以上で再収集。左右反転で補強 | M3 |
| LRN-4 | S3 | MAE が走行性能を示さない。評価手順が学習データを含み、隣接フレームが学習と検証に分かれる | `python/src/e2e/training/train.py:40`, `python/src/e2e/evaluation/evaluate.py:24` | セッション単位で分割し、シードを固定。主指標は「10分あたりの介入回数」 | M3 |
| LRN-5 | S3 | クリーナーはモデルと合わないフレームを除くため、数値が良く見えやすい | `python/src/e2e/dataset/cleaner.py` | 改善の判断は実走の指標で行う | M3 |
| LRN-6 | S3 | 収集データに距離と速度が入らない。`throttle` は前進指令後に停止しても 0.5 のまま | `python/src/e2e/gui_collector.py:259-265,277` | 距離と実速度を保存する。停止中フレームの扱いを決める | M3 |
| LRN-7 | S3 | 塞がり判定がない | — | 超音波を自動ラベルにした補助出力を追加 | M3 |
| LRN-8 | S3 | 学習の再現性がない（シード未固定、データ拡張なし）。`v003` の Val Loss が 0.0 | `train.py:40`, `python/models/v003/metadata.json` | シード固定。検証件数0の場合は学習を失敗扱いにする | M3 |

## 5. 会話 Agent

目標との差が最も大きい領域。

| ID | 重要度 | 内容 | 根拠 | 対応案 | M |
|---|---|---|---|---|---|
| AGT-1 | S2 | 音声入力がない。`listen` は固定文字列を正常応答で返す | `rust/src/api/hardware.rs:98-101` | 音声入力を実装。未実装の間はエラーを返す | M2 |
| AGT-2 | S2 | 電池値を読んでいるが API がない | `rust/src/sensor_monitor/core.rs:45`, `rust/src/api/mod.rs:19-27` | `/api/status` を追加 | M1 |
| AGT-3 | S2 | LLM は画像を見ていない。ツールは解像度と文字数だけを返す | `python/src/tools/camera_tool.py:15-19` | 画像認識の方式を評価する。Gemini へ画像を送る場合は事前に確認する（D-3） | M2 |
| AGT-4 | S2 | Python 側に永続記憶がない。Rust の記憶は旧 GUI 専用 | `python/src/gui/app.py:40`, `rust/src/sample/picarx_sample.rs:100-103` | 記憶を Brain 側へ一本化 | M2 |
| AGT-5 | S2 | Agent の状態フィールドが常に空 | `python/src/agents/state.py:11-13`, `python/src/gui/app.py:277-279` | 状態ストアから毎ターン注入 | M2 |
| AGT-6 | S2 | 走行プロセスと Agent の間で情報共有がない | `autonomous_driver.py`, `graph.py` | 状態ストアとイベント API で接続 | M2, M5 |
| AGT-7 | S2 | 学習状況を参照する経路がない | `python/src/e2e/registry/` | レジストリと走行記録を状態ストアへ公開 | M5 |
| AGT-8 | S1 | LLM がモーターを直接操作できる（速度100、5秒まで） | `python/src/tools/drive_tool.py`, `python/src/models.py:15-16` | LLM は意図のみを出力。実行可否は Body が判断 | M2, M5 |
| AGT-9 | S3 | サブエージェントのノードにエッジがなく到達不能。テストは存在確認のみ | `python/src/agents/graph.py:89-92`, `python/tests/test_agents.py` | 削除して単一 Agent にする | M2 |
| AGT-10 | S3 | 履歴圧縮が各メッセージの先頭50文字の連結で、要約になっていない。切り方により tool の呼び出しと結果の対応が壊れる恐れあり（未検証） | `python/src/token_optimizer.py:42-57` | ターン単位で切り、LLM で要約 | M2 |
| AGT-11 | S3 | `speak_text` ツールと GUI の自動発話で二重に話す可能性がある | `python/src/tools/voice_tool.py`, `python/src/gui/app.py:262` | 発話経路を1つにする | M2 |
| AGT-12 | S3 | 発話を途中で止める手段がない | `rust/src/picarx/core.rs:496-510` | 停止 API を追加し、割り込みに対応 | M2 |
| AGT-13 | S3 | Rust と Python で LLM 実装と記憶が重複 | `rust/src/llm_agent/`, `rust/src/memory/`, `rust/src/sample/picarx_sample.rs` | Brain 移行後に Rust 側を廃止 | M2 |

## 6. 開発基盤

| ID | 重要度 | 内容 | 根拠 | 対応案 | M |
|---|---|---|---|---|---|
| OPS-1 | S3 | Python の CI がない（`rust-ci.yml` のみ） | `.github/workflows/` | ruff と pytest の CI を追加 | M1 |
| OPS-2 | S3 | ruff が1件検出（import 順） | `python/tests/test_e2e_integrated_safety.py:7` | `ruff check --fix` | M1 |
| OPS-3 | S3 | カメラ API がクライアントの `width` と `quality` を無視する | `python/src/picarx_client/client.py:93`, `rust/src/api/handlers.rs:145` | 実装するか引数を削除 | M1 |
| OPS-4 | S3 | 不要な残骸がある | 空ディレクトリ `rust;C`、ルートの `models/` | 不要なら削除 | M1 |
| OPS-5 | S3 | `main` に `ROADMAP.md` がなく、ブランチ間で文書が異なる | `git ls-tree main` | 本ブランチのマージで解消 | M0 |
| OPS-7 | S3 | CI のビルド対象が Pi 3B+ で、実機の Pi 5 と異なる | `.github/workflows/rust-ci.yml:137-149` | `MODEL=5` へ変更 | M1 |
| OPS-8 | S3 | コミットの作者情報に第三者のものを含むメールアドレスがある | `git log --format=%ae` | 本人に確認して判断する | M0 |

## 7. 文書の修正記録

2026-11-03 に次を修正した。

| 文書 | 主な修正 |
|---|---|
| `ROADMAP.md` | 最終目標と完了条件を追加。順序を変更。番号を M0〜M5 に統一。状態表記を見直し。未決定事項を明記 |
| `E2E_SPECIFICATION.md` | 実測の周期を記載。走行ループと安全機能の実態、保存データ形式、ディレクトリ構成を修正 |
| `APPLICATION_SPECIFICATION.md` | メモリの適用範囲、Piper の引数、CI、Python バージョンを修正。会話 Agent の目標仕様を追加 |
| `agentic_ai_architecture_policy.md` | 実装済みと目標を区別し、Agent の方針と音声処理の前提を見直し |
| `python/src/e2e/README.md` | 周期とウォッチドッグに関する誤った記述を修正 |

### 解決済み（2026-11-03）

- `python/.env` と室内写真を全履歴から削除し、GitHub リポジトリを作り直した。新規ミラークローンでキーと JPEG が0件であることを確認した
- ルートに `.gitignore` を追加した
- `articles/` と `rust/build/error_result_temp.txt` を全履歴から除去し、追跡対象外にした（ファイルは手元に保持）
- ローカルのユーザー名とプライベート IP を全履歴で置き換えた

### 解決済み（コード修正。実機は未確認）

| ID | 内容 | 修正 |
|---|---|---|
| SAF-1 | 緊急停止が保持されない | `/api/drive/emergency_stop` と解除 API を追加。通常の `/api/drive/stop` は保持しない |
| SAF-3 | 走行時間と速度の上限がない | サーバー側で `speed` 0〜100、`duration_ms` 50〜5000 を検証 |
| SAF-9 | GUI の緊急停止が押せない時間がある | 通信を背景スレッドへ移し、緊急停止は専用スレッドから送信 |
| SAF-10 | API が認証なし、CORS 全許可 | Bearer トークンを必須にし、CORS を廃止。未設定ならサーバーは起動しない |
| SAF-11 | 黙ってモックで起動する | 明示フラグ `PICARX_ALLOW_MOCK=1` がある時だけモックで起動 |
| OPS-6 | 既定モデルが古い `v001` | `--model` と `--output_dir` を必須にした |
| OPS-9 | `v001/model.onnx` にローカルのユーザー名 | 全履歴から除去し追跡対象外にした。ONNX 出力時にスタックトレース情報を除去 |
