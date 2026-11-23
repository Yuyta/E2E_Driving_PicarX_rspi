"""NeuraCPU（CPU だけで動く検索付きの会話）を PiCar-X の会話 Agent から使うための部品。

PiCar-X の仮想環境で動くもの:
- client.py: 検索サーバーの起動と呼び出し
- eval_routing.py: 発言ごとの行動の選び方の評価

NeuraCPU の仮想環境で動くもの（llama-cpp-python と文書の読み込みを使う）:
- sidecar.py: 検索サーバー本体
- bench_speed.py / bench_qa.py: 速度と、検索付き回答の正確さの評価
- eval_data.py: 評価の質問（どちらの環境からも読む。依存なし）
"""
