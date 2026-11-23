"""NeuraCPU の生成速度を、PiCar-X で今使っている Ollama と同じ質問で比べる。

NeuraCPU の仮想環境で実行する（llama-cpp-python が必要）。PiCar-X の他のモジュールは読まない。

    $nc = "$env:USERPROFILE\\.gemini\\antigravity\\scratch\\neuracpu_engine"
    & "$nc\\.venv\\Scripts\\python.exe" python/src/neuracpu/bench_speed.py --neuracpu-root $nc

測る構成:
- neuracpu_default: NeuraCPU の CLI と同じ設定（Qwen2.5-3B Q4_K_M、4スレッド、Prompt Lookup 投機的デコード）
- llama_cpp_t*: 同じモデルで Prompt Lookup を切り、スレッド数だけを変える
- ollama:*: Ollama の HTTP API。同じ GGUF を登録したモデルと、PiCar-X 既定の qwen3.5:4b-text

結果は数値と固定の質問への応答だけで、個人情報は含まない。
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List

# localhost だと Windows では IPv6 を先に試し、毎回約2秒待たされた
OLLAMA_URL = "http://127.0.0.1:11434"
# PiCar-X のローカル Agent の固定指示とほぼ同じ長さ（約700トークン）の指示
AGENT_SYSTEM = (
    "あなたは小型ロボットカー PiCar-X です。発言を読み、JSON を1つだけ出力します。\n"
    "action の意味:\n- forward: 前進する\n- backward: 後退する\n"
    "- left, right, center: 舵を左、右、中央にする\n"
    "- stop: 止まる。「待って」「やめて」「動かないで」も含む\n"
    "- remember: 事実を覚える。key に短い見出し、value に内容を入れる\n"
    "- forget: 事実を忘れる。key に見出しを入れる\n- look: カメラで前を見て、見える物を答える\n"
    "- search_local: 手元の資料を調べて答える\n- search_web: Web で最新の情報を調べて答える\n"
    "- reply: 上のどれでもない会話。text に短い日本語で答える\n"
    "動かすのは、今動くように頼まれた時だけです。電池やセンサーの値は分かりません。\n"
    '例:\n少し前に進んで → {"action":"forward"}\nバックして → {"action":"backward"}\n'
    '右に曲がって → {"action":"right"}\n危ないからやめて → {"action":"stop"}\n'
    '前に何がある？ → {"action":"look"}\n'
    '私の名前はたろうです。覚えておいて → {"action":"remember","key":"ユーザーの名前","value":"たろう"}\n'
    '前回の旅行は楽しかったね → {"action":"reply","text":"楽しかったですね。"}\n'
    '緊急停止の解除方法は？ → {"action":"search_local"}\n'
    '今日のニュースを教えて → {"action":"search_web"}\n'
)
RAG_CONTEXT_FILES = ("troubleshooting_playbook.md", "ai_system_guidelines.md")

CASES = [
    # (名前, system, user, max_tokens)
    (
        "chat_short",
        "日本語で簡潔に答える会話アシスタントです。",
        "こんにちは！今日は何をして遊ぼうか？",
        96,
    ),
    ("agent_turn1", AGENT_SYSTEM, "少し右に曲がりながら前に進んで", 48),
    # 固定の指示が同じなら、2回目は先頭の読み直しが省ける（プレフィックスキャッシュ）
    ("agent_turn2", AGENT_SYSTEM, "好きな食べ物は何？", 48),
    ("rag_answer", None, "推論が遅いときに最初に確認することを3つ挙げてください。", 160),
]


def rag_system(root: Path) -> str:
    docs = "\n\n".join(
        (root / "docs" / name).read_text(encoding="utf-8") for name in RAG_CONTEXT_FILES
    )
    return f"次の資料だけを根拠に、日本語で簡潔に答えてください。\n<資料>\n{docs}\n</資料>"


def median(values: List[float]) -> float:
    return round(statistics.median(values), 3) if values else 0.0


class LlamaRunner:
    def __init__(self, model_path: str, threads: int, speculative: bool, batch: int, ubatch: int):
        from llama_cpp import Llama

        kwargs = dict(
            model_path=model_path,
            n_threads=threads,
            n_threads_batch=threads,
            n_ctx=4096,
            n_batch=batch,
            n_ubatch=ubatch,
            n_gpu_layers=0,
            verbose=False,
        )
        if speculative:
            from llama_cpp.llama_speculative import LlamaPromptLookupDecoding

            kwargs["draft_model"] = LlamaPromptLookupDecoding(max_ngram_size=2, num_pred_tokens=4)
            kwargs["logits_all"] = True
        started = time.perf_counter()
        self.llm = Llama(**kwargs)
        self.load_seconds = time.perf_counter() - started

    def run(self, system: str, user: str, max_tokens: int) -> Dict:
        started = time.perf_counter()
        first = None
        text = ""
        stream = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {}).get("content", "")
            if delta:
                if first is None:
                    first = time.perf_counter()
                text += delta
        end = time.perf_counter()
        first = first or end
        tokens = len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=False))
        prompt_tokens = len(
            self.llm.tokenize((system + user).encode("utf-8"), add_bos=False, special=False)
        )
        decode = end - first
        return {
            "ttft": first - started,
            "decode_tps": (tokens - 1) / decode if decode > 0 and tokens > 1 else 0.0,
            "total": end - started,
            "tokens": tokens,
            "prompt_tokens": prompt_tokens,
            "text": text,
        }

    def close(self):
        self.llm.close()


class OllamaRunner:
    def __init__(self, model: str, threads: int):
        self.model = model
        self.threads = threads
        started = time.perf_counter()
        self._post(
            {
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1, "num_thread": threads, "num_ctx": 4096},
            }
        )
        self.load_seconds = time.perf_counter() - started

    @staticmethod
    def _post(body: dict):
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(request, timeout=600)

    def run(self, system: str, user: str, max_tokens: int) -> Dict:
        body = {
            "model": self.model,
            "stream": True,
            "think": False,
            "keep_alive": "10m",
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {
                "temperature": 0.0,
                "num_predict": max_tokens,
                "num_thread": self.threads,
                "num_ctx": 4096,
            },
        }
        if self.model.startswith("neuracpu-"):
            body.pop("think")  # qwen2.5 は思考モードを持たない
        started = time.perf_counter()
        first = None
        text = ""
        final = {}
        with self._post(body) as response:
            for line in response:
                data = json.loads(line)
                piece = data.get("message", {}).get("content", "")
                if piece and first is None:
                    first = time.perf_counter()
                text += piece
                if data.get("done"):
                    final = data
        end = time.perf_counter()
        first = first or end
        eval_s = final.get("eval_duration", 0) / 1e9
        return {
            "ttft": first - started,
            "decode_tps": final.get("eval_count", 0) / eval_s if eval_s else 0.0,
            "total": end - started,
            "tokens": final.get("eval_count", 0),
            "prompt_tokens": final.get("prompt_eval_count", 0),
            "cached_tokens": final.get("prompt_eval_cached_count", 0),
            "load_s": final.get("load_duration", 0) / 1e9,
            "prefill_tps": final.get("prompt_eval_count", 0)
            / (final.get("prompt_eval_duration", 0) / 1e9)
            if final.get("prompt_eval_duration")
            else 0.0,
            "text": text,
        }

    def close(self):
        try:
            self._post({"model": self.model, "messages": [], "keep_alive": 0}).read()
        except Exception:
            pass


def bench(name: str, factory: Callable[[], object], cases, repeat: int) -> Dict:
    print(f"== {name}", flush=True)
    runner = factory()
    rows = []
    try:
        for case_name, system, user, max_tokens in cases:
            runner.run(system, user + " ", max_tokens)  # 暖機。結果には含めない
            samples = []
            for i in range(repeat):
                if case_name == "agent_turn2":
                    # 同じ固定指示で1ターン目を答えた直後を再現する。先頭の読み直しを省けるかを測る
                    runner.run(system, f"少し右に曲がりながら前に進んで{i}", 48)
                    samples.append(runner.run(system, user, max_tokens))
                else:
                    # 先頭に毎回違う行を置き、前回の読み込み結果を使えない状態（初回）で測る
                    nonce = f"[{time.time_ns()}]\n"
                    samples.append(runner.run(nonce + system, user, max_tokens))
            row = {
                "case": case_name,
                "ttft_s": median([s["ttft"] for s in samples]),
                "decode_tps": median([s["decode_tps"] for s in samples]),
                "total_s": median([s["total"] for s in samples]),
                "tokens": samples[-1]["tokens"],
                "prompt_tokens": samples[-1]["prompt_tokens"],
                "sample": samples[-1]["text"][:200],
            }
            for key in ("prefill_tps", "cached_tokens", "load_s"):
                if key in samples[-1]:
                    row[key] = median([s[key] for s in samples])
            rows.append(row)
            print(
                f"  {case_name:12} TTFT={row['ttft_s']:6.2f}s decode={row['decode_tps']:5.1f} tok/s "
                f"total={row['total_s']:6.2f}s prompt={row['prompt_tokens']} out={row['tokens']}",
                flush=True,
            )
    finally:
        runner.close()
    return {"config": name, "load_seconds": round(runner.load_seconds, 2), "rows": rows}


def power_state() -> str:
    try:
        out = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
        return {"1": "battery", "2": "ac"}.get(out, out or "unknown")
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--neuracpu-root", required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", default="4,6,8,10")
    parser.add_argument("--ollama-models", default="neuracpu-qwen2.5-3b,qwen3.5:4b-text")
    parser.add_argument("--skip", default="", help="省く構成名のカンマ区切り")
    parser.add_argument("--output", default="bench_speed.json")
    args = parser.parse_args()

    root = Path(args.neuracpu_root)
    model = str(root / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf")
    cases = [(n, s if s is not None else rag_system(root), u, m) for n, s, u, m in CASES]
    skip = set(filter(None, args.skip.split(",")))
    configs = [("neuracpu_default", lambda: LlamaRunner(model, 4, True, 256, 128))]
    for t in [int(x) for x in args.threads.split(",") if x]:
        configs.append((f"llama_cpp_t{t}", lambda t=t: LlamaRunner(model, t, False, 512, 512)))
    for name in filter(None, args.ollama_models.split(",")):
        configs.append((f"ollama:{name}", lambda name=name: OllamaRunner(name, 10)))

    results = []
    for name, factory in configs:
        if name in skip:
            continue
        try:
            results.append(bench(name, factory, cases, args.repeat))
        except Exception as error:
            print(f"  [skip] {name}: {error}", flush=True)
            results.append({"config": name, "error": str(error)})
        Path(args.output).write_text(
            json.dumps(
                {
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "cpu": platform.processor(),
                    "python": sys.version.split()[0],
                    "power": power_state(),
                    "repeat": args.repeat,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
