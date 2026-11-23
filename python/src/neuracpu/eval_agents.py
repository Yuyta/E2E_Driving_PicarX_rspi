"""従来のローカル LLM の Agent と NeuraCPU の Agent を、同じ発言で比べる。車は動かない。

PiCar-X の仮想環境で実行する。NeuraCPU の検索サーバーは自動で起動する。

    uv run python -m src.neuracpu.eval_agents
    uv run python -m src.neuracpu.eval_agents --agents neuracpu --sets car

比べる Agent:
- local: GUI の「ollama/qwen3.5:4b-text」と同じ LocalAgent（従来のローカル LLM）
- neuracpu: GUI の「NeuraCPU」と同じ NeuraCPUAgent（検索は本物、走行・カメラは記録するだけ）

発言の組:
- car: llm_eval の発言（調整用32、調整に使っていない32、曲がりながら走る10）。判定も llm_eval と同じ
- knowledge: 操作マニュアルにだけ答えがある17問と、一般的な知識12問。期待する語が返事にあれば正解
- conversation: llm_eval の会話（9ターン）を1つの Agent で続けて話す。GUI と同じ使い方の速さ

発言や記憶はすべて架空で、結果に個人情報は含まない。
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from langchain_core.messages import HumanMessage

from src.agents.local_agent import LocalAgent
from src.agents.neuracpu_agent import NeuraCPUAgent
from src.llm_eval import (
    CASES,
    COMPOUND_CASES,
    CONVERSATION,
    HELDOUT_CASES,
    Case,
    RecordingTool,
    make_memory,
    result_row,
    status_client,
)
from src.neuracpu.client import ensure_ollama_model, shared_client
from src.neuracpu.eval_data import (
    GENERAL_QA_HELDOUT,
    PICARX_QA,
    PICARX_QA_HELDOUT,
    UNKNOWN_MARKERS,
)
from src.neuracpu.eval_routing import FakeVision

RESULT_DIR = Path(__file__).resolve().parents[2] / "data" / "neuracpu_bench"
# 答えが変わりにくい一般的な知識。モデルが覚えていても、Web で調べても正解にする
GENERAL_QA = [
    ("富士山の高さは？", ["3776", "3775", "3,776"]),
    ("日本の首都はどこ？", ["東京"]),
    ("Python を作ったのは誰？", ["グイド", "Guido", "ヴァンロッサム", "ヴァン・ロッサム"]),
    ("日本で一番長い川は？", ["信濃川"]),
    ("ダイナマイトを発明したのは誰？", ["ノーベル"]),
    ("太陽系で一番大きい惑星は？", ["木星"]),
    ("将棋の駒は全部で何枚？", ["40"]),
    ("水の化学式は？", ["H2O", "H₂O", "Ｈ２Ｏ"]),
    ("エベレストの標高は？", ["8848", "8849", "8,848", "8,849"]),
    ("1年は何日？", ["365"]),
    ("東京タワーの高さは？", ["333"]),
    ("光の三原色は？", ["赤", "緑", "青"]),
]


# 走行・記憶・資料・一般知識を混ぜた1つの会話。GUI で実際に話す時の速さと正確さを測る
MIXED_CONVERSATION = [
    Case("こんにちは！元気？"),
    Case("Control Center を開くコマンドは？", reply_has="src.gui.web"),
    Case("少し前に進んで", "drive_forward", args={"duration_ms": 500}),
    Case("走行データ収集で録画を始めるキーは？", reply_has="R"),
    Case("右に曲がって", "steer_wheels", args={"angle": 30}),
    Case("富士山の高さは？", reply_has="77"),  # 3776 m、3,776 m、3775.56 m のどれでもよい
    Case("私の名前はたろうです。覚えておいて", "remember_fact", fact=("たろう", True)),
    Case("モデルの採否はどのファイルに記録される？", reply_has="registry.json"),
    Case("待って", "stop_moving"),
    Case("日本の首都はどこ？", reply_has="東京"),
    Case("手動操作で前進する時の速さはいくつ？", reply_has="30"),
    Case("ありがとう"),
]


def knowledge_ok(reply: str, expect: List[str], question: str) -> bool:
    if not expect:
        return any(marker in reply for marker in UNKNOWN_MARKERS)
    if question == "光の三原色は？":
        return all(color in reply for color in expect)
    folded = reply.casefold()
    return any(value.casefold() in folded for value in expect)


def make_agent(kind: str, memory, tools) -> object:
    common = dict(status_client=status_client(), memory=memory, tools=tools, vision=FakeVision())
    if kind == "local":
        return LocalAgent(**common)
    # neuracpu-ctx: 判断と同じプロンプトの続きで答える方式（比較用）
    return NeuraCPUAgent(
        search_client=shared_client(), answer_in_context=kind == "neuracpu-ctx", **common
    )


def recording(memory, calls: List[Tuple[str, dict]]):
    original = {"remember": memory.remember_fact, "forget": memory.forget_fact}

    def remember(key: str, value: str) -> None:
        calls.append(("remember_fact", {"key": key, "value": value}))
        original["remember"](key, value)

    def forget(key: str) -> bool:
        calls.append(("forget_fact", {"key": key}))
        return original["forget"](key)

    memory.remember_fact = remember
    memory.forget_fact = forget


def new_tools(calls):
    return {
        name: RecordingTool(name, calls)
        for name in ("drive_forward", "drive_backward", "steer_wheels", "stop_moving")
    }


async def run_single(kind: str, text: str, directory: str, name: str):
    calls: List[Tuple[str, dict]] = []
    memory = make_memory(directory, name)
    recording(memory, calls)
    agent = make_agent(kind, memory, new_tools(calls))
    started = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=text)]})
    seconds = time.perf_counter() - started
    reply = result["messages"][-1].content
    facts = memory.facts()
    memory.close()
    return reply, calls, facts, seconds, getattr(agent, "last_action", None)


async def run_car(kind: str, directory: str) -> List[Dict]:
    rows = []
    for group, cases in (
        ("tuned", CASES),
        ("heldout", HELDOUT_CASES),
        ("compound", COMPOUND_CASES),
    ):
        for index, case in enumerate(cases):
            reply, calls, facts, seconds, action = await run_single(
                kind, case.text, directory, f"{kind}_{group}_{index}"
            )
            row = result_row(case, reply, calls, facts, seconds)
            row.update(group=group, action=action)
            rows.append(row)
            log(kind, group, row)
    return rows


async def run_knowledge(kind: str, directory: str, datasets=None) -> List[Dict]:
    rows = []
    datasets = datasets or (("manual", PICARX_QA), ("general", GENERAL_QA))
    for group, dataset in datasets:
        for index, (question, expect) in enumerate(dataset):
            reply, calls, _, seconds, action = await run_single(
                kind, question, directory, f"{kind}_{group}_{index}"
            )
            row = {
                "group": group,
                "text": question,
                "passed": knowledge_ok(reply, expect, question) and not calls,
                "answerable": bool(expect),
                "seconds": round(seconds, 2),
                "tools": [name for name, _ in calls],
                "action": action,
                "reply": reply,
            }
            rows.append(row)
            log(kind, group, row)
    return rows


async def run_conversation(kind: str, directory: str, cases=None) -> List[Dict]:
    cases = cases or CONVERSATION
    calls: List[Tuple[str, dict]] = []
    memory = make_memory(directory, f"{kind}_conversation")
    recording(memory, calls)
    agent = make_agent(kind, memory, new_tools(calls))
    history: List = []
    rows = []
    for case in cases:
        calls.clear()
        history.append(HumanMessage(content=case.text))
        started = time.perf_counter()
        result = await agent.ainvoke({"messages": history})
        seconds = time.perf_counter() - started
        history = result["messages"]
        row = result_row(case, history[-1].content, list(calls), memory.facts(), seconds)
        row.update(
            group="conversation" if cases is CONVERSATION else "mixed",
            action=getattr(agent, "last_action", None),
        )
        rows.append(row)
        log(kind, "conversation", row)
    memory.close()
    return rows


def log(kind, group, row):
    mark = "OK" if row["passed"] else "NG"
    reply = str(row["reply"]).replace("\n", " ")[:70]
    print(
        f"[{kind}/{group}] {mark} {row['seconds']:5.2f}s {row['text']} -> {row['tools']} {reply!r}",
        flush=True,
    )


def summarize(rows: List[Dict]) -> Dict:
    seconds = sorted(r["seconds"] for r in rows)
    return {
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "motion_errors": sum(r.get("motion_error", False) for r in rows),
        "median_s": round(statistics.median(seconds), 2),
        "mean_s": round(statistics.mean(seconds), 2),
        "p90_s": round(seconds[int(len(seconds) * 0.9) - 1 if len(seconds) > 1 else 0], 2),
    }


async def warm_up(kind: str, directory: str) -> None:
    """モデルの読み込みと検索サーバーの起動を、計測の前に済ませる"""
    if kind.startswith("neuracpu"):
        await asyncio.to_thread(ensure_ollama_model)
        await asyncio.to_thread(shared_client().ensure_started)
    await run_single(kind, "こんにちは", directory, f"{kind}_warm")


async def run_heldout(kind: str, directory: str) -> List[Dict]:
    """調整に使っていない質問"""
    return await run_knowledge(
        kind,
        directory,
        (("manual_heldout", PICARX_QA_HELDOUT), ("general_heldout", GENERAL_QA_HELDOUT)),
    )


async def run_mixed(kind: str, directory: str) -> List[Dict]:
    return await run_conversation(kind, directory, MIXED_CONVERSATION)


SETS: Dict[str, Callable] = {
    "car": run_car,
    "knowledge": run_knowledge,
    "conversation": run_conversation,
    "mixed": run_mixed,
    "heldout": run_heldout,
}


async def main_async(agents: List[str], sets: List[str]) -> Dict:
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "summary": {}, "rows": {}}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        for kind in agents:
            await warm_up(kind, directory)
            for name in sets:
                rows = await SETS[name](kind, directory)
                report["rows"][f"{kind}/{name}"] = rows
                groups = sorted({r["group"] for r in rows})
                report["summary"][f"{kind}/{name}"] = {
                    "all": summarize(rows),
                    **{g: summarize([r for r in rows if r["group"] == g]) for g in groups},
                }
    shared_client().close()
    return report


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", default="local,neuracpu")
    parser.add_argument("--sets", default="car,knowledge,conversation,mixed,heldout")
    args = parser.parse_args()
    report = asyncio.run(
        main_async(args.agents.split(","), [s for s in args.sets.split(",") if s in SETS])
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"agents_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, summary in report["summary"].items():
        for group, item in summary.items():
            print(
                f"{key:24} {group:12} {item['passed']:3}/{item['total']:<3} "
                f"誤走行 {item['motion_errors']} 中央値 {item['median_s']}s "
                f"平均 {item['mean_s']}s p90 {item['p90_s']}s"
            )
    print(path)


if __name__ == "__main__":
    main()
