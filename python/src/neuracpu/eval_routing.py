"""NeuraCPU の Agent が、発言ごとに正しい行動（走行・記憶・カメラ・雑談・資料・Web）を選ぶかを測る。

PiCar-X の仮想環境で実行する。判断には実際の Ollama を使い、走行・カメラ・検索は記録するだけの偽物。

    uv run python -m src.neuracpu.eval_routing
    uv run python -m src.neuracpu.eval_routing --models qwen3.5:4b-text,neuracpu-qwen2.5-3b

比べる方式:
- neuracpu_rules: NeuraCPU の CLI の判断（時事の語なら Web、あいさつなら会話、他は資料の検索）。
  CLI は車体を扱わないので、走行・記憶・カメラは常に誤りになる
- llm:<モデル>: Agent の LLM が1回の JSON で選ぶ
- llm+rules:<モデル>: LLM が会話と答えた時だけ、発言の明らかな手がかりで検索に直す（Agent の既定）
"""

import argparse
import asyncio
import json
import re
import statistics
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage

from src.agents.neuracpu_agent import NeuraCPUAgent, correct_route
from src.llm_factory import LLMFactory
from src.memory.store import MemoryStore
from src.models import RobotStatus
from src.neuracpu.client import SearchResult
from src.neuracpu.eval_data import ROUTE_CASES
from src.vision.local_vlm import Look

RESULT_DIR = Path(__file__).resolve().parents[2] / "data" / "neuracpu_bench"
ROUTE_OF = {
    "forward": "robot",
    "backward": "robot",
    "left": "robot",
    "right": "robot",
    "center": "robot",
    "stop": "robot",
    "remember": "memory",
    "forget": "memory",
    "look": "look",
    "reply": "chat",
    "search_local": "local",
    "search_web": "web",
}
# NeuraCPU の cli.py の判断をそのまま写したもの
CLI_WEB = re.compile(
    r"最新|現在|今日|今週|今年|直近|最近|ニュース|価格|相場|スケジュール|最新版|最新情報|"
    r"as of|latest|current|today|news|price",
    re.I,
)
CLI_SOCIAL = re.compile(
    r"(こんにちは|こんばんは|おはよう|ありがとう(?:ございます)?|助かりました|やあ|hello|hi|thanks|"
    r"thank you)[!！。\s]*",
    re.I,
)


def cli_route(text: str) -> str:
    if CLI_WEB.search(text):
        return "web"
    if CLI_SOCIAL.fullmatch(text):
        return "chat"
    return "local"


class FakeTool:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return "成功"


class FakeVision:
    def __init__(self):
        self.calls = 0

    async def describe(self, question):
        self.calls += 1
        return Look("机があります。", True)


class FakeSearch:
    async def search(self, kind, query, max_chars=700):
        return SearchResult(True, kind)  # 何も見つからない。答えの LLM は呼ばれない


def status_client():
    from src.llm_eval import FIXED_STATUS

    client = AsyncMock()
    client.get_status = AsyncMock(return_value=RobotStatus.model_validate(FIXED_STATUS))
    return client


class FakeAnswer:
    async def ainvoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(content="記憶から答えました。")


async def run_model(model: str, directory: str, conversation: bool = False):
    from src.agents.local_agent import MAX_REPLY_TOKENS, action_schema
    from src.agents.neuracpu_agent import NEURACPU_ACTIONS

    llm = LLMFactory.create_local(
        model=model,
        temperature=0.0,
        format=action_schema(NEURACPU_ACTIONS),
        num_predict=MAX_REPLY_TOKENS,
    )
    await llm.ainvoke([HumanMessage(content="こんにちは")])  # 読み込み
    rows = []
    history = []
    shared_memory = MemoryStore(str(Path(directory) / f"{model.replace(':', '_')}_conv.db"))
    vision = FakeVision()
    tools = {
        n: FakeTool() for n in ("drive_forward", "drive_backward", "steer_wheels", "stop_moving")
    }
    agent = None
    for index, (text, expected) in enumerate(ROUTE_CASES):
        vision.calls = 0
        for tool in tools.values():
            tool.calls.clear()
        memory = (
            shared_memory
            if conversation
            else MemoryStore(str(Path(directory) / f"{model.replace(':', '_')}_{index}.db"))
        )
        # GUI と同じく、会話の形では1つの Agent を使い続ける
        if agent is None or not conversation:
            agent = NeuraCPUAgent(
                llm=llm,
                answer_llm=FakeAnswer(),
                search_client=FakeSearch(),
                correct_routes=False,
                status_client=status_client(),
                memory=memory,
                tools=tools,
                vision=vision,
                sleep=asyncio.sleep,
            )
        agent.last_action = None
        started = time.perf_counter()
        # 会話の形では履歴を引き継ぐ。固定の指示と履歴の先頭が同じなので、読み直しを省ける
        messages = (history if conversation else []) + [HumanMessage(content=text)]
        result = await agent.ainvoke({"messages": messages})
        history = result["messages"]
        seconds = time.perf_counter() - started
        raw = agent.last_action or ("look" if vision.calls else "state")
        if not conversation:
            memory.close()
        llm_route = ROUTE_OF.get(raw, raw)
        fixed_route = ROUTE_OF.get(correct_route(raw, text), llm_route)
        rows.append(
            {
                "text": text,
                "expected": sorted(expected),
                "action": raw,
                "llm": llm_route,
                "llm_ok": llm_route in expected,
                "llm_rules": fixed_route,
                "llm_rules_ok": fixed_route in expected,
                "seconds": round(seconds, 2),
            }
        )
        print(
            f"[{'OK' if rows[-1]['llm_rules_ok'] else 'NG'}] {seconds:4.1f}s {text} -> "
            f"{raw} ({fixed_route}) 期待 {sorted(expected)}",
            flush=True,
        )
    shared_memory.close()
    return rows


def summarize(name, oks, groups):
    by_group = {}
    for ok, group in zip(oks, groups):
        hit, total = by_group.get(group, (0, 0))
        by_group[group] = (hit + ok, total + 1)
    return {
        "method": name,
        "correct": sum(oks),
        "total": len(oks),
        "by_expected": {g: f"{h}/{t}" for g, (h, t) in sorted(by_group.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3.5:4b-text,neuracpu-qwen2.5-3b")
    parser.add_argument(
        "--conversation", action="store_true", help="1つの会話として履歴を引き継いで測る"
    )
    args = parser.parse_args()
    groups = ["/".join(sorted(expected)) for _, expected in ROUTE_CASES]
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "conversation": args.conversation,
        "summary": [],
        "rows": {},
    }
    cli = [cli_route(text) in expected for text, expected in ROUTE_CASES]
    report["summary"].append(summarize("neuracpu_rules", cli, groups))
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        for model in filter(None, args.models.split(",")):
            rows = asyncio.run(run_model(model, directory, args.conversation))
            report["rows"][model] = rows
            for key, name in (("llm_ok", f"llm:{model}"), ("llm_rules_ok", f"llm+rules:{model}")):
                summary = summarize(name, [r[key] for r in rows], groups)
                summary["median_seconds"] = statistics.median(r["seconds"] for r in rows)
                report["summary"].append(summary)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"routing_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in report["summary"]:
        print(
            f"{item['method']:34} {item['correct']}/{item['total']} {item['by_expected']} "
            f"{item.get('median_seconds', '')}"
        )
    print(path)


if __name__ == "__main__":
    main()
