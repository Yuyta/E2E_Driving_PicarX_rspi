# NeuraCPU の評価と PiCar-X への組み込み

調査日: 2026-11-23 / ブランチ: `feature/neuracpu-agent`
対象: `%USERPROFILE%\.gemini\antigravity\scratch\neuracpu_engine`（このフォルダ自体は変更していない）
実機: Intel Core i7-1255U（Pコア2 + Eコア8、12スレッド）、メモリ 16GB、GPU なし

## 結論

1. **NeuraCPU は「独自の高速化エンジン」ではなく、llama.cpp（llama-cpp-python）+ Qwen2.5-3B Q4_K_M + ハイブリッド検索（BM25 + BGE-M3 + RRF）+ Brave Web 検索の組み合わせ**です。モデル単体の速度・賢さは llama.cpp と量子化モデルそのものの性能で、NeuraCPU が足しているのは検索（RAG）です。
2. **高速化は実現できていません。** 既定の設定（Prompt Lookup 投機的デコード、4スレッド）は、同じモデルを Ollama で動かすより遅く、資料付きの回答は 64秒 対 38秒でした。投機的デコードは速くならず、短い出力では生成が約半分の速さになりました。`engine.py` のドラフトモデルは初期化に失敗して実際には使われていません（`LlamaDraftModel() takes no arguments`）。
3. **高精度化は「検索」の部分では本物です。** 資料にしか答えがない質問で、検索なしは 1/23 しか答えられず、検索ありでは 11〜20/23 になりました。ただし NeuraCPU の CLI のままでは指示文を復唱する失敗が多く（正答 11/23）、PiCar-X に組み込んだ改良版は 20/23 で、最初の一言までが約2倍速くなりました（22秒 → 11秒）。
4. **この CPU で待ち時間を決めるのは「読む量」です。** 文章の読み込み（prefill）は 28〜51 トークン/秒しか出ず、1,500トークンの資料を読むだけで 30〜56秒かかります。生成（10〜16 トークン/秒）より、渡す資料を短くする方が効きます。
5. **PiCar-X の会話 GUI で「NeuraCPU（CPU・資料と Web を自動で検索）」を選べるようにしました。** 利用者は話すだけで、走行・記憶・カメラ・雑談・資料検索・Web 検索を AI が1回の判断で選びます（36発言中36正解。NeuraCPU の CLI の判断方式は 13/36）。

## 1. 速度の実測（`bench_speed.py`）

同じ質問を各3回測った中央値。TTFT は最初の一言までの時間。資料回答は約1,550トークンの資料を読んで答える。

| 構成 | 短い会話 TTFT | Agent 初回 TTFT（約390トークン） | 生成 tok/s | Agent 2回目 TTFT | 資料回答 TTFT | 資料回答 合計 | 読み込み tok/s |
|---|---|---|---|---|---|---|---|
| **NeuraCPU 既定**（4スレッド、Prompt Lookup） | 1.3s | 12.1s | 4.9 | 0.48s | 56.1s | 64.1s | 28 |
| llama-cpp-python 4スレッド（Prompt Lookup なし） | 1.5s | 12.0s | 10.4 | 0.44s | 52.2s | 60.9s | 30 |
| 同 6スレッド | 1.4s | 10.3s | 12.5 | 0.37s | 46.4s | 54.7s | 33 |
| 同 8スレッド | 1.4s | 9.8s | 12.7 | 0.35s | 41.3s | 48.5s | 37 |
| 同 10スレッド | 1.1s | 8.9s | 12.4 | 0.37s | 38.9s | 48.4s | 40 |
| **Ollama 同じ GGUF**（10スレッド） | 1.0s | 7.8s | 15.4 | 0.39s | 30.9s | 38.3s | 51 |
| Ollama qwen3.5:4b-text（PiCar-X 既定） | 1.5s | 7.9s | 11.4 | 7.31s | 30.0s | 38.3s | 45 |

分かったこと:

- **Prompt Lookup 投機的デコードは逆効果。** 4スレッド同士で、Agent の短い出力の生成が 10.4 → 4.9 tok/s。会話や資料回答でも速くならない。llama-cpp-python では `logits_all=True`（全トークンの出力を計算）も必要になる。ドラフトの当たりが少ない日本語の短い応答では効かない。Web の調査でも、CPU だけの投機的デコードは「ほぼ効かないか遅くなる」という報告が多い。
- **スレッドは 4 では少ない。** 10 にすると資料回答の TTFT が 25% 短くなる。読み込みは計算量で決まるので、E コアも使った方が速い。
- **同じモデルなら Ollama の方が速い**（読み込み 1.7倍、生成 1.3〜1.5倍）。PiCar-X はすでに Ollama を使っているので、NeuraCPU のモデルを別に読み込む利点はない。
- **qwen3.5 は 2回目の読み直しを省けない場合がある**（7.31s）。qwen3.5 は途中まで巻き戻せない構造（Gated DeltaNet）のため、固定の指示の後ろが変わると全部読み直す。履歴を追記だけにすれば省ける（下の判断の評価で 1.6秒）。Qwen2.5 は普通の Transformer なので常に省ける（0.39s）。
- 副次的な発見: **`localhost` で Ollama に接続すると、新しい接続のたびに約2秒待たされていた**（IPv6 を先に試すため。`127.0.0.1` なら 0.002秒）。PiCar-X の既定を `127.0.0.1` に直した。会話の間隔が空いて接続が切れるたびに、ローカル LLM と画像認識の応答が約2秒遅れていた。

## 2. 正確さの実測（`bench_qa.py`）

- PiCar-X: 操作マニュアル（`manual.html`）にだけ答えがある 14問 + 資料に答えがない 3問
- NeuraCPU: NeuraCPU 付属の例の資料（`docs/`）の 9問 + 答えがない 2問
- 正解は期待する語が回答にあるか。答えがない質問は「分からない」と答えれば正解。

| 構成 | PiCar-X 答えあり | 答えなし | NeuraCPU 答えあり | 答えなし | TTFT 中央値 | 合計 中央値 | 渡した文字数 |
|---|---|---|---|---|---|---|---|
| 検索なし（qwen3.5） | 1/14 | 0/3 | 0/9 | 0/2 | 1.5s | 3.2s | 約75 |
| **NeuraCPU CLI そのまま**（Qwen2.5-3B、llama-cpp-python） | 6/14 | 1/3※ | 5/9 | 0/2※ | 22.3s | 26.0s | 約1,970 |
| CLI と同じ資料と指示 + qwen3.5（Ollama） | 9/14 | 3/3 | 7/9 | 2/2 | 22.5s | 32.6s | 約1,970 |
| **改良版**（PiCar-X に組み込んだ形）+ qwen3.5 | 11/14 | 2/3 | 9/9 | 2/2 | 10.8s | 12.3s | 約830〜1,030 |
| 改良版 + NeuraCPU の Qwen2.5-3B（Ollama） | 11/14 | 3/3 | 9/9 | 2/2 | 7.3s | 8.8s | 約830〜1,030 |

※ 機械的な判定。回答を読むと、CLI の5問は「根拠が不足しているため明記できません」と答えていて、実質は答えを控えている。

NeuraCPU CLI の失敗の多くは、指示文「根拠が不足している点は不足と明記し、関連する引用IDを回答内に示してください」を **そのまま復唱して答えない**ものだった（例: 「根拠が不足しているため、明記します。関連する引用IDは [local:2] を示します。」）。3B のモデルには指示が長く複雑すぎる。

**改良版**で変えたこと（すべて計測して採用）:

1. 資料の索引を 700文字 → 400文字 に細かくした。NeuraCPU は1つの資料から2件までしか返さないので、1冊のマニュアルでは大きく切ると拾える箇所が少ない。
2. 上位3件から、質問と重なる行とその次の行だけを抜き出し、合計 700文字に収めた。答えを含む根拠の割合は、CLI の 1,556文字と同等（PiCar-X 11/14、NeuraCPU 9/9）で、量は半分以下。
3. 出典の絶対パスを省き、ファイル名と行だけにした。
4. 指示を短くし、「読み上げるので1〜2文」「資料に答えがなければ見つからなかったと言う」だけにした。

残る誤り: 改良版 + qwen3.5 でも「緊急停止を押すと Pi は止まりませんが…」のような**資料と逆の答え**が1件あった。資料の言葉を使っていても、3〜4B のモデルは誤読することがある。緊急停止そのものは検索を通らず、GUI の停止ボタンと停止語の即時処理で必ず実行される（変更なし）。

検索方式の比較（答えを含む根拠を LLM に渡せた割合）:

| 方式 | PiCar-X（14問） | NeuraCPU（9問） | 検索時間 |
|---|---|---|---|
| BM25 のみ、CLI の量（1,556文字） | 12 | 9 | 1.5ms |
| BM25 + BGE-M3、CLI の量 | 11 | 8 | 65ms |
| BM25 + BGE-M3、700文字で単純に切る | 9 | 7 | 65ms |
| 改良版（400文字の索引 + 要点 700文字） | 11 | 9 | 65ms |

この程度の小さな資料では、埋め込み検索（BGE-M3）の上乗せは見られなかった。資料が増えて言い換えが多くなると効く可能性があるので残しているが、評価の質問を増やして確かめる価値がある。

## 3. 自律的な判断の実測（`eval_routing.py`）

36の発言（走行6、記憶3、カメラ2、雑談9、資料8、Web 8）で、選んだ行動が正しいか。

| 方式 | 正解 | 1回の判断の中央値 |
|---|---|---|
| NeuraCPU CLI の判断（時事の語なら Web、あいさつなら会話、他はすべて資料） | 13/36 | 0秒 |
| LLM（qwen3.5:4b-text）、発言ごとに新しい会話 | 35/36 | 11.9s |
| **LLM（qwen3.5:4b-text）、1つの会話として**（GUI と同じ） | **36/36** | **1.6s** |
| LLM（Qwen2.5-3B）、発言ごと / 会話 | 33/36 / 32/36 | 0.9s / 2.5s |

- CLI の判断は車体を扱わず、雑談も資料検索に回してしまう。`/web` や `/local` を利用者が付ける前提の作り。
- LLM が1回の JSON で判断する方式は、走行・記憶・カメラを含めて正確。qwen3.5 は会話として履歴を追記していく限り速い。
- 「前回は何の話をしたっけ？」を資料検索と判断した場合でも、記憶に答えがあれば記憶を根拠に答えるようにした。
- 会話と判断した時だけ、「天気」「ニュース」「調べて」などの明らかな手がかりで検索に直す規則も入れた。この36問では LLM が先に正しく判断したので結果は変わらなかった（安全網）。

車体の操作（既存の `llm_eval`、32問）: Qwen2.5-3B は 31/32、中央値 0.57秒。qwen3.5:4b-text は 31/32（2026-11-06 の記録）。

**既定のモデルは qwen3.5:4b-text にした**（判断が最も正確で、今のローカル LLM と同じなので追加のメモリが要らない）。回答を約3秒速くしたい時は、NeuraCPU の Qwen2.5-3B を Ollama に登録して `NEURACPU_MODEL=neuracpu-qwen2.5-3b` にする（判断の正解は 32〜33/36 に下がる）。

## 4. 実際の動作（GUI と同じ部品、車体は偽物、Ollama と検索は本物）

| 発言 | 判断 | 応答までの時間 | 返事（抜粋） |
|---|---|---|---|
| こんにちは！ | reply | 9.0s（最初の1回、モデルの準備込み） | こんにちは！何かお手伝いできることはありますか？ |
| Control Center の開き方を教えて | search_local | 13.5s | ターミナルで「uv run python -m src.gui.web」と実行するか…（出典: manual.html） |
| 少し前に進んで | forward | 1.9s | 前に進みました。（速度50、0.5秒。コードで検証） |
| 日本で一番高い山の高さをネットで調べて | search_web | 14.3s | 富士山は標高 3775.56 メートルです。（出典: Wikipedia「富士山」） |
| 緊急停止の解除はどうするの？ | search_local | 15.1s | 画面の「緊急停止を解除」ボタンを押して戻ります。（出典: manual.html） |

検索サーバーの初回起動は 18秒（マニュアルの埋め込み 52件で 15.5秒。2回目以降はキャッシュを使う）。資料の検索は 0.1秒、Wikipedia は 1.4秒。GUI でこのモデルを選んだ時点で背景で起動する。

## 5. 組み込みの構成

```text
会話 GUI（PySide6）── モデルの一覧で「NeuraCPU（CPU・資料と Web を自動で検索）」
   │  文字でも音声（Whisper）でも同じ経路
   ▼
NeuraCPUAgent（src/agents/neuracpu_agent.py、LocalAgent を拡張）
   ├─ 見る質問・車体の状態 … 従来どおり LLM を呼ばずに答える
   ├─ 判断: Ollama の LLM が JSON を1つ出す
   │     forward/backward/left/right/center/stop … コードで検証して走行（従来どおり）
   │     remember/forget/look/reply … 従来どおり
   │     search_local / search_web … 検索 → 根拠700文字 → 2回目の LLM で1〜2文の答え
   │     （資料に無ければ Web、Web が使えなければ資料。記憶の質問は記憶を根拠に）
   └─ 返事: 画面には出典付き、読み上げは出典番号を除く
         ▼ HTTP（127.0.0.1 のみ）
NeuraCPU 検索サーバー（src/neuracpu/sidecar.py、NeuraCPU の仮想環境で動く別プロセス）
   ├─ 資料: NeuraCPU の LocalRAGEngine（BM25 + BGE-M3 + RRF）、変更は30秒ごとに検知
   └─ Web: BRAVE_SEARCH_API_KEY があれば Brave、なければ日本語版 Wikipedia
```

- 検索する資料: 既定は操作マニュアルと `python/data/knowledge/`（Git に入らない。PDF・Word・Excel・PowerPoint・Markdown などを置ける）。`NEURACPU_SOURCES` に `;` 区切りで指定すると置き換えられる。
- 外へ送るもの: Web 検索の時の検索語（発言から作る）だけ。手元の資料の中身や会話の履歴は送らない。資料や Web の本文は参考データとして渡し、その中の命令には従わせない。
- 設定（環境変数）: `NEURACPU_ROOT`（NeuraCPU の場所）、`NEURACPU_MODEL`（Ollama のモデル）、`NEURACPU_PORT`（既定 18765）、`NEURACPU_SOURCES`、`BRAVE_SEARCH_API_KEY`（任意）。
- Gemini が使えない時の切り替え先にはしない。NeuraCPU を選んでいる時はローカル LLM と同じく 120秒まで待つ。

## 6. さらに速く・正確に・軽くするための改良案（優先順）

最新の論文・情報と今回の実測から、効果が大きい順に並べる。

| # | 改良 | 期待できる効果 | 根拠 | 状態 |
|---|---|---|---|---|
| 1 | 渡す資料を短くする（要点抽出、700文字） | TTFT 約2倍短縮、正答は同等以上 | 本評価。Provence（ICLR 2025）も、文単位の剪定で質を落とさず 50〜80% 削れると報告 | **実装済み** |
| 2 | Prompt Lookup を切る、スレッドを増やす、Ollama で生成 | 資料回答 64秒 → 38秒 | 本評価 | **実装済み**（生成は Ollama） |
| 3 | `localhost` → `127.0.0.1` | 接続ごとに約2秒 | 本評価 | **実装済み** |
| 4 | LLM 1回の判断で行動と検索を選ぶ | 判断 13/36 → 36/36 | 本評価。Adaptive-RAG、Self-Routing RAG（SR-RAG）、RAGRouter-Bench の「軽い判断器で検索の要否を決める」方向と同じ | **実装済み** |
| 5 | 要点抽出を学習済みの剪定器（XProvence など多言語版）に替える | 今の語の重なりより誤読が減る見込み | XProvence（ECIR 2026、16言語で学習） | 未実装。CPU での速度を測る必要あり |
| 6 | 日本語に強い小さな埋め込み（ruri-v3-30m/70m）に替える | 索引作り 15秒の短縮とメモリ削減。JMTEB 検索は BGE-M3 と同等以上の報告 | ruri-v3 のモデルカード、Zenn/Qiita の比較 | 未実装（ダウンロードが必要） |
| 7 | 日本語特化の小型モデル（LFM2.5-1.2B-JP）を判断と回答に試す | 読み込み・生成とも 3〜4B の数倍速い見込み | Liquid AI の公表値（ノート PC の CPU で約100 tok/s）。Q4 で固有名詞を誤る報告もある | 未実装（ダウンロードが必要） |
| 8 | 固定の指示の KV キャッシュを保存して再利用（llama-server `--cache-reuse` 等） | qwen3.5 の判断で、履歴を削った直後の約10秒の読み直しを避ける | llama.cpp のディスカッション、KV 再利用の研究（CacheClip、ProphetKV など） | 未実装 |
| 9 | BitNet b1.58 2B4T | 論文では x86 で llama.cpp 比 2.4〜6.2倍 | bitnet.cpp 論文 | **非推奨**。日本語の評価がなく、別ビルドが必要 |
| 10 | 評価の質問を増やす（実際の会話から50〜100問） | 改良の採否を確実に決められる | — | 今回の質問は 23 + 5 + 36 問で小さい |

## 7. 再現方法

```powershell
$nc = "$env:USERPROFILE\.gemini\antigravity\scratch\neuracpu_engine"
cd <PiCarX のフォルダ>
# 比較のため、NeuraCPU の GGUF を Ollama に登録する（ファイルはコピーされる）
ollama create neuracpu-qwen2.5-3b -f <FROM に GGUF のパスを書いた Modelfile>
& "$nc\.venv\Scripts\python.exe" python/src/neuracpu/bench_speed.py --neuracpu-root $nc --output python/data/neuracpu_bench/speed.json
& "$nc\.venv\Scripts\python.exe" python/src/neuracpu/bench_qa.py --neuracpu-root $nc --output python/data/neuracpu_bench/qa.json
cd python
uv run python -m src.neuracpu.eval_routing --conversation
uv run python -m src.llm_eval --model neuracpu-qwen2.5-3b
```

結果の JSON は `python/data/neuracpu_bench/`（Git に入らない）に残る。質問はすべて架空で、個人情報は含まない。

## 制約と注意

- 計測は1台・1回の条件（電源・温度は記録できず）。同じ構成でも試走と本番で生成速度が 1〜3 tok/s ぶれた。差が小さい比較（スレッド 8 と 10 など）は誤差の範囲。
- 正解の判定は語の一致で、意味の正しさは人が読んで確かめた範囲だけ。答えの表現が違う正解（「データが学習の後に変わっていないか」など）は誤りに数えている。
- Wikipedia は百科事典的な質問には答えられるが、天気・ニュース・相場は答えられない。それらには Brave Search の API キー（要登録。料金と利用条件は Brave の案内を確認）が必要。
- 実際の GUI の画面と、実機の車体ではまだ試していない（GUI の部品はテストで、会話の経路は上の実動作で確認）。

## 参考にした論文・情報

- [Provence: efficient and robust context pruning for RAG（ICLR 2025）](https://arxiv.org/abs/2501.16214) / [XProvence（arXiv 2601.18886）](https://arxiv.org/abs/2601.18886)
- [Lightweight Query Routing for Adaptive RAG: RAGRouter-Bench（arXiv 2604.03455）](https://arxiv.org/pdf/2604.03455) / [Adaptive RAG の解説](https://blog.reachsumit.com/posts/2025/10/learning-to-retrieve/)
- [llama.cpp: KV cache reuse with llama-server](https://github.com/ggml-org/llama.cpp/discussions/13606) / [CPU speculative decoding の調査 issue](https://github.com/ggml-org/llama.cpp/issues/21453) / [投機的デコードで遅くなった例](https://openclawdc.com/blog/speculative-decoding-slower-llama-cpp/)
- [Qualiteg 日本語 LLM ランキング 2026（10B 未満で Qwen3.5-4B が1位）](https://journal.qualiteg.com/llm-ranking-2026/)
- [LiquidAI LFM2.5-1.2B-JP](https://huggingface.co/LiquidAI/LFM2.5-1.2B-JP-202606) / [LFM2.5 の発表](https://www.liquid.ai/en/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai) / [Q4 で固有名詞を誤る検証](https://lilting.ch/en/articles/lfm25-12b-jp-local-test-plan)
- [cl-nagoya/ruri-v3-30m](https://huggingface.co/cl-nagoya/ruri-v3-30m) / [日本語 RAG の埋め込み比較 2026](https://zenn.dev/fp16/articles/aa48dcae23974e) / [中小企業の RAG 埋め込み選定](https://qiita.com/k_nabe/items/a03213a5137dbf8e66c5)
- [bitnet.cpp（arXiv 2410.16144）](https://arxiv.org/abs/2410.16144) / [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)
