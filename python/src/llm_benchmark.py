"""ローカル LLM を同じ条件で比較する。

    uv run python -m src.llm_benchmark qwen3.5:4b granite4:3b

実際のツール定義と状態の文章を使い、日本語の応答、ツール呼び出し、実測値の扱いを測る。
単発の質問だけでは実際の会話での失敗を見逃したので、履歴を引き継ぐ会話でも測る。
結果は数値と固定の質問への応答だけで、個人情報は含まない。
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import httpx
from langchain_core.utils.function_calling import convert_to_openai_tool

from src.config import settings
from src.tools import ALL_TOOLS

BASE_PROMPT = (
    "あなたはPiCar-Xのメインコントローラです。ユーザーの指示を達成するために適切なツールを呼び出すか、"
    "簡潔に日本語で回答してください。"
)
KNOWN_STATE = (
    "\n\n【現在の車体の状態】\n- 電池: 7.62V、残量の目安 51%\n- 走行: 停止中\n"
    "車体の状態を聞かれたら、上の値だけを使って答えること。値を推測しないこと。"
    "「不明」の項目は分からないと答えること。"
)
UNKNOWN_STATE = (
    "\n\n【現在の車体の状態】\n- 電池: 不明\n- 走行: 停止中\n"
    "車体の状態を聞かれたら、上の値だけを使って答えること。値を推測しないこと。"
    "「不明」の項目は分からないと答えること。"
)
REQUEST_TIMEOUT_SEC = 300.0
NUM_CTX = 4096
JAPANESE_RANGES = ((0x3040, 0x30FF), (0x4E00, 0x9FFF))


def has_japanese(text: str) -> bool:
    return any(lo <= ord(ch) <= hi for ch in text for lo, hi in JAPANESE_RANGES)


@dataclass
class Case:
    name: str
    prompt: str
    system: str = BASE_PROMPT
    expect_tool: Optional[str] = None
    check: Optional[Callable[[str], bool]] = None


CASES = [
    Case("greeting", "こんにちは！元気？", check=has_japanese),
    Case("stop_tool", "止まって", expect_tool="stop_moving"),
    Case("forward_tool", "少し前に進んで", expect_tool="drive_forward"),
    Case("distance_tool", "前の障害物までの距離を測って", expect_tool="read_distance"),
    Case(
        "remember_tool",
        "私の名前はたろうです。覚えておいて",
        expect_tool="remember_fact",
    ),
    Case(
        "battery_known",
        "電池どう？",
        system=BASE_PROMPT + KNOWN_STATE,
        check=lambda text: "7.62" in text or "51" in text,
    ),
    Case(
        "battery_unknown",
        "電池どう？",
        system=BASE_PROMPT + UNKNOWN_STATE,
        check=lambda text: (
            not any(ch.isdigit() for ch in text)
            and ("不明" in text or "分から" in text or "わから" in text)
        ),
    ),
]


CONVERSATION_CONTEXT = (
    "参考情報。発言への回答に必要な時だけ使う。そのまま読み上げない。\n"
    "【現在の車体の状態】\n- 電池: 7.60V、残量の目安 50%\n- 走行: 停止中\n- 前方の距離: 40.0cm\n"
    "車体の状態を聞かれたら、上の値だけを使って答えること。値を推測しないこと。\n\n"
    "【記憶】\n前回の会話の終わり:\n- ユーザー: 昨日はキッチンを探検したよ\n"
    "- あなた: キッチンの探検は楽しかったですね\n"
    "過去のことを聞かれたら、この記憶だけを使って答えること。"
)
CONVERSATION_SYSTEM = (
    BASE_PROMPT + "\n行動や記憶の指示には、説明せず必ずツールを呼び出してください。"
)
CONVERSATION = [
    ("電池どう？", None, "7.6"),
    ("前回は何の話をしたっけ？", None, "キッチン"),
    ("私の名前はたろうです。覚えておいて", "remember_fact", None),
    ("止まって", "stop_moving", None),
    ("少し前に進んで", "drive_forward", None),
    ("今の電池は何ボルト？", None, "7.6"),
    ("ストップ！", "stop_moving", None),
]


def judge_turn(
    expect_tool: Optional[str], expect_text: Optional[str], reply: str, tools: List[str]
) -> bool:
    if expect_tool is not None:
        return expect_tool in tools
    return not tools and expect_text in reply


def run_conversation(client: httpx.Client, model: str, tools: List[dict]) -> dict:
    """履歴を引き継ぎながら話す。ツールは実行せず、完了したものとして履歴に残す。"""
    history: List[dict] = []
    passed = 0
    turns = []
    for question, expect_tool, expect_text in CONVERSATION:
        messages = (
            [{"role": "system", "content": CONVERSATION_SYSTEM}]
            + history
            + [
                {"role": "system", "content": CONVERSATION_CONTEXT},
                {"role": "user", "content": question},
            ]
        )
        started = time.perf_counter()
        response = client.post(
            "/api/chat",
            json={
                "model": model,
                "stream": False,
                "think": False,
                "tools": tools,
                "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
                "messages": messages,
            },
        )
        response.raise_for_status()
        seconds = round(time.perf_counter() - started, 1)
        message = response.json()["message"]
        reply = message.get("content", "").strip()
        calls = message.get("tool_calls") or []
        names = [call["function"]["name"] for call in calls]
        good = judge_turn(expect_tool, expect_text, reply, names)
        passed += good
        print(
            f"  {'OK ' if good else 'NG '}{seconds:>6}s {question} tools={names} {reply!r}",
            flush=True,
        )
        turns.append(
            {"question": question, "passed": good, "sec": seconds, "tools": names, "reply": reply}
        )

        history.append({"role": "user", "content": question})
        assistant = {"role": "assistant", "content": reply}
        if calls:
            assistant["tool_calls"] = calls
        history.append(assistant)
        history.extend(
            {"role": "tool", "tool_name": name, "content": "完了しました"} for name in names
        )
    return {
        "passed": passed,
        "turns": len(turns),
        "median_sec": statistics.median(turn["sec"] for turn in turns),
        "detail": turns,
    }


@dataclass
class CaseResult:
    name: str
    passed: bool
    first_token_sec: float
    total_sec: float
    tokens_per_sec: float
    tool_calls: List[str] = field(default_factory=list)
    reply: str = ""
    error: str = ""


def judge(case: Case, reply: str, tool_calls: List[str]) -> bool:
    if case.expect_tool is not None:
        return case.expect_tool in tool_calls
    if tool_calls:
        return False
    return case.check(reply) if case.check else bool(reply)


def run_case(client: httpx.Client, model: str, case: Case, tools: List[dict]) -> CaseResult:
    body = {
        "model": model,
        "stream": True,
        "think": False,
        "tools": tools,
        "options": {"num_ctx": NUM_CTX, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": case.system},
            {"role": "user", "content": case.prompt},
        ],
    }
    started = time.perf_counter()
    first_token = None
    reply = ""
    tool_calls: List[str] = []
    final: Dict = {}
    with client.stream("POST", "/api/chat", json=body) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            message = chunk.get("message", {})
            if first_token is None and (message.get("content") or message.get("tool_calls")):
                first_token = time.perf_counter() - started
            reply += message.get("content", "")
            tool_calls += [c["function"]["name"] for c in message.get("tool_calls") or []]
            if chunk.get("done"):
                final = chunk
    total = time.perf_counter() - started
    eval_sec = final.get("eval_duration", 0) / 1e9
    return CaseResult(
        name=case.name,
        passed=judge(case, reply, tool_calls),
        first_token_sec=round(first_token if first_token is not None else total, 2),
        total_sec=round(total, 2),
        tokens_per_sec=round(final.get("eval_count", 0) / eval_sec, 1) if eval_sec else 0.0,
        tool_calls=tool_calls,
        reply=reply.strip()[:160],
    )


def loaded_size_gb(client: httpx.Client, model: str) -> Optional[float]:
    for running in client.get("/api/ps").json().get("models", []):
        if running.get("name") == model:
            return round(running.get("size", 0) / 1024**3, 2)
    return None


def benchmark_model(client: httpx.Client, model: str, tools: List[dict]) -> dict:
    # 最初の読み込み時間は別に記録し、応答時間に混ぜない
    started = time.perf_counter()
    client.post("/api/generate", json={"model": model, "keep_alive": "10m"}).raise_for_status()
    load_sec = round(time.perf_counter() - started, 1)

    results = []
    for case in CASES:
        try:
            result = run_case(client, model, case, tools)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            result = CaseResult(case.name, False, 0.0, 0.0, 0.0, error=str(exc)[:160])
        results.append(result)
        mark = "OK " if result.passed else "NG "
        print(
            f"  {mark}{result.name:<16} first {result.first_token_sec:>6}s "
            f"total {result.total_sec:>6}s {result.tokens_per_sec:>6} t/s "
            f"tools={result.tool_calls} {result.reply[:50]!r} {result.error}",
            flush=True,
        )

    print("  -- 会話 --", flush=True)
    conversation = run_conversation(client, model, tools)

    speeds = [r.tokens_per_sec for r in results if r.tokens_per_sec]
    summary = {
        "model": model,
        "load_sec": load_sec,
        "memory_gb": loaded_size_gb(client, model),
        "passed": sum(r.passed for r in results),
        "cases": len(results),
        "first_token_median_sec": statistics.median(r.first_token_sec for r in results),
        "tokens_per_sec_median": statistics.median(speeds) if speeds else 0.0,
        "results": [asdict(r) for r in results],
        "conversation": conversation,
    }
    # 次のモデルの測定にメモリを空ける
    client.post("/api/generate", json={"model": model, "keep_alive": 0})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="ローカル LLM を同じ条件で比較する")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--base-url", default=settings.ollama_base_url)
    parser.add_argument("--output-dir", type=Path, default=Path("data/llm_benchmarks"))
    args = parser.parse_args()

    tools = [convert_to_openai_tool(tool) for tool in ALL_TOOLS]
    summaries = []
    with httpx.Client(base_url=args.base_url, timeout=REQUEST_TIMEOUT_SEC) as client:
        for model in args.models:
            print(f"\n=== {model} ===", flush=True)
            summaries.append(benchmark_model(client, model, tools))

    print(
        f"\n{'model':<20}{'pass':>7}{'talk':>7}{'talk(s)':>9}"
        f"{'first(s)':>10}{'t/s':>8}{'load(s)':>9}{'mem(GB)':>9}"
    )
    for s in summaries:
        print(
            f"{s['model']:<20}{s['passed']:>4}/{s['cases']:<2}"
            f"{s['conversation']['passed']:>4}/{s['conversation']['turns']:<2}"
            f"{s['conversation']['median_sec']:>9}"
            f"{s['first_token_median_sec']:>10}{s['tokens_per_sec_median']:>8}"
            f"{s['load_sec']:>9}{str(s['memory_gb']):>9}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
