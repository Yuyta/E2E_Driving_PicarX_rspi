"""ローカル LLM の Agent を、固定の発言で評価する。車は動かない。

    uv run python -m src.llm_eval --repeat 3
    uv run python -m src.llm_eval --model qwen3.5:2b --repeat 3
    uv run python -m src.llm_eval --baseline
    uv run python -m src.llm_eval --conversation
    uv run python -m src.llm_eval --compound

走行ツールは記録するだけの偽物で、状態は固定値を使う。記憶は一時ファイルに作る。
発言や記憶はすべて架空のもので、結果に個人情報は含まない。
--baseline は、ツール定義を使う従来の方式を同じ発言で測る。
"""

import argparse
import asyncio
import json
import statistics
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agents.body_context import format_body_context
from src.agents.graph import CONTEXT_HEADER, ROUTER_PROMPT
from src.agents.local_agent import ACTION_SCHEMA, MAX_REPLY_TOKENS, LocalAgent
from src.llm_factory import LLMFactory
from src.memory.store import MemoryStore
from src.models import RobotStatus
from src.tools import ALL_TOOLS

RESULT_DIR = Path(__file__).resolve().parents[1] / "data" / "llm_benchmarks"
MOTION_TOOLS = {"drive_forward", "drive_backward", "steer_wheels"}
FIXED_STATUS = {
    "mode": "idle",
    "emergency_stop": False,
    "motion": None,
    "steer_angle": 0,
    "blocked": {"forward": None, "backward": None},
    "sensors": {
        "distance": {"cm": 40.0, "age_ms": 12},
        "floor": {"values": [500, 500, 500], "age_ms": 8, "cliff": [False, False, False]},
        "battery": {"voltage": 7.6, "percent": 50, "age_ms": 300},
    },
    "cliff": {
        "guard_enabled": True,
        "calibration": {"baseline": [500, 500, 500], "thresholds": [50, 50, 50]},
    },
    "reflex": {"hz": 39.5, "max_interval_ms": 31.0, "last_tick_age_ms": 4},
    "camera": {"running": True, "fps": 15.0, "frames": 42, "frame_age_ms": 20, "error": None},
    "latest_event_id": 3,
}


@dataclass
class Case:
    text: str
    tool: Optional[str] = None  # 呼ばれるべきツール。None は「何も呼ばない」
    reply_has: Optional[str] = None
    args: Optional[dict] = None
    fact: Optional[Tuple[str, bool]] = None  # (値, 保存されているべきか)
    tools: Optional[Tuple[str, ...]] = None  # 複数のツールを順に呼ぶ時


TURN = ("steer_wheels", "drive_forward", "steer_wheels")
TURN_BACK = ("steer_wheels", "drive_backward", "steer_wheels")
COMPOUND_CASES = [
    Case("前に進みながら右に曲がって", tools=TURN, args={"angle": 30}),
    Case("右に曲がりながら進んで", tools=TURN, args={"angle": 30}),
    Case("左に曲がりながら前に進んで", tools=TURN, args={"angle": -30}),
    Case("少し前に進みながら左に曲がって", tools=TURN, args={"angle": -30}),
    Case("右にカーブしながら前進して", tools=TURN, args={"angle": 30}),
    Case("前に進みながら左へ", tools=TURN, args={"angle": -30}),
    Case("バックしながら右に曲がって", tools=TURN_BACK, args={"angle": 30}),
    Case("左に曲がりながら後ろに下がって", tools=TURN_BACK, args={"angle": -30}),
    Case("前に進みながら曲がって", reply_has="どちら"),
    Case("進みながら右と左を見て"),
]


CASES = [
    # 走行
    Case("少し前に進んで", "drive_forward", args={"duration_ms": 500}),
    Case("前に進んで", "drive_forward"),
    Case("ちょっと進んで", "drive_forward", args={"duration_ms": 500}),
    Case("まっすぐ行って", "drive_forward"),
    Case("前進して", "drive_forward"),
    Case("後ろに下がって", "drive_backward"),
    Case("バックして", "drive_backward"),
    Case("少し戻って", "drive_backward", args={"duration_ms": 500}),
    # 舵
    Case("右に曲がって", "steer_wheels", args={"angle": 30}),
    Case("左を向いて", "steer_wheels", args={"angle": -30}),
    Case("ハンドルを右に切って", "steer_wheels", args={"angle": 30}),
    Case("タイヤをまっすぐにして", "steer_wheels", args={"angle": 0}),
    # 一覧にない言い方の停止
    Case("危ないからやめて", "stop_moving"),
    Case("もう動かないで", "stop_moving"),
    Case("待って", "stop_moving"),
    # 記憶
    Case("私の名前はたろうです。覚えておいて", "remember_fact", fact=("たろう", True)),
    Case("好きな色は青だよ、覚えといて", "remember_fact", fact=("青", True)),
    Case("飼い猫の名前は忘れて", "forget_fact", fact=("ミケ", False)),
    Case("前回は何の話をしたっけ？", reply_has="キッチン"),
    Case("飼い猫の名前は何？", reply_has="ミケ"),
    # 車体の状態
    Case("電池どう？", reply_has="7.6"),
    Case("今の電池は何ボルト？", reply_has="7.6"),
    Case("バッテリー残ってる？", reply_has="50"),
    Case("前の障害物までの距離を測って", reply_has="40"),
    Case("調子はどう？", reply_has="7.6"),
    # 動いてはいけない発言
    Case("こんにちは！元気？"),
    Case("前回の旅行は楽しかったね"),
    Case("進化って面白いよね"),
    Case("右と左どっちが好き？"),
    Case("「前に進んで」って英語で何て言うの？"),
    Case("昨日バックで駐車するのに苦労したよ"),
    Case("今日の天気はどう？"),
]


# プロンプトの例文や検証の言葉の調整に使っていない発言。ここに合わせてプロンプトを調整しない
HELDOUT_CASES = [
    Case("前へお願い", "drive_forward"),
    Case("もうちょい前に進めて", "drive_forward", args={"duration_ms": 500}),
    Case("ゆっくり前に行って", "drive_forward"),
    Case("進んでください", "drive_forward"),
    Case("直進して", "drive_forward"),
    Case("下がって", "drive_backward"),
    Case("うしろに行って", "drive_backward"),
    Case("ちょっとバックお願い", "drive_backward", args={"duration_ms": 500}),
    Case("後退して", "drive_backward"),
    Case("左に曲がって", "steer_wheels", args={"angle": -30}),
    Case("右へ向いて", "steer_wheels", args={"angle": 30}),
    Case("ハンドルを左に", "steer_wheels", args={"angle": -30}),
    Case("舵を真ん中に戻して", "steer_wheels", args={"angle": 0}),
    Case("止めて", "stop_moving"),
    Case("ダメ、危ない", "stop_moving"),
    Case("そこでじっとしてて", "stop_moving"),
    Case("僕の誕生日は5月3日だよ、覚えておいて", "remember_fact", fact=("5月3日", True)),
    Case("妹はさくらって名前なんだ、覚えてね", "remember_fact", fact=("さくら", True)),
    Case("昨日の話、何だったっけ？", reply_has="キッチン"),
    Case("ミケって誰だっけ？", reply_has="猫"),
    Case("電池あとどれくらい？", reply_has="50"),
    Case("充電残量教えて", reply_has="50"),
    Case("壁までの距離は？", reply_has="40"),
    Case("段差はある？", reply_has="床"),
    Case("車の状態を教えて", reply_has="7.6"),
    Case("今日は疲れたよ"),
    Case("前に食べたカレーがおいしかった"),
    Case("左利きなんだ"),
    Case("バックミュージックって何？"),
    Case("走るのは好き？"),
    Case("進路について悩んでる"),
    Case("右の部屋は寒いね"),
]


class RecordingTool:
    def __init__(self, name: str, calls: List[Tuple[str, dict]]):
        self.name = name
        self.calls = calls

    async def ainvoke(self, args: dict) -> str:
        self.calls.append((self.name, args))
        return "停止完了" if self.name == "stop_moving" else "成功"


def make_memory(directory: str, name: str) -> MemoryStore:
    path = Path(directory) / f"{name}.db"
    earlier = MemoryStore(path)
    earlier.add_message("user", "昨日はキッチンを探検したよ")
    earlier.add_message("assistant", "キッチンの探検は楽しかったですね")
    earlier.remember_fact("飼い猫の名前", "ミケ")
    earlier.close()
    return MemoryStore(path)


def judge(case: Case, reply: str, tools: List[Tuple[str, dict]], facts: Dict[str, str]) -> bool:
    names = [name for name, _ in tools]
    expected = list(case.tools) if case.tools else ([case.tool] if case.tool else [])
    if names != expected:
        return False
    if case.reply_has and case.reply_has not in reply:
        return False
    if case.args and any(tools[0][1].get(key) != value for key, value in case.args.items()):
        return False
    if case.fact:
        value, stored = case.fact
        if (value in facts.values()) != stored:
            return False
    return True


def status_client() -> AsyncMock:
    client = AsyncMock()
    client.get_status = AsyncMock(return_value=RobotStatus.model_validate(FIXED_STATUS))
    return client


async def run_local_case(llm, case: Case, directory: str, name: str) -> dict:
    calls: List[Tuple[str, dict]] = []
    tools = {
        tool: RecordingTool(tool, calls)
        for tool in ("drive_forward", "drive_backward", "steer_wheels", "stop_moving")
    }
    memory = make_memory(directory, name)
    original = {"remember": memory.remember_fact, "forget": memory.forget_fact}

    def remember(key: str, value: str) -> None:
        calls.append(("remember_fact", {"key": key, "value": value}))
        original["remember"](key, value)

    def forget(key: str) -> bool:
        calls.append(("forget_fact", {"key": key}))
        return original["forget"](key)

    memory.remember_fact = remember
    memory.forget_fact = forget
    agent = LocalAgent(llm=llm, status_client=status_client(), memory=memory, tools=tools)
    started = time.perf_counter()
    result = await agent.ainvoke({"messages": [HumanMessage(content=case.text)]})
    elapsed = time.perf_counter() - started
    reply = result["messages"][-1].content
    outcome = result_row(case, reply, calls, memory.facts(), elapsed)
    memory.close()
    return outcome


async def run_baseline_case(llm, case: Case, directory: str, name: str) -> dict:
    """従来の方式の1回目の呼び出しだけを測る。ツールは実行しない。"""
    memory = make_memory(directory, name)
    context = "\n\n".join(
        section
        for section in (
            format_body_context(RobotStatus.model_validate(FIXED_STATUS)),
            memory.build_context(case.text),
        )
        if section
    )
    memory.close()
    prompt = [
        SystemMessage(content=ROUTER_PROMPT),
        SystemMessage(content=f"{CONTEXT_HEADER}\n{context}"),
        HumanMessage(content=case.text),
    ]
    started = time.perf_counter()
    response: AIMessage = await llm.ainvoke(prompt)
    elapsed = time.perf_counter() - started
    calls = [(call["name"], {}) for call in response.tool_calls]
    # 従来の方式は引数の基準が違うので、ツールの選択と返事だけを比べる
    plain = Case(case.text, case.tool, case.reply_has)
    # 距離は read_distance でも正解とする
    if case.text == "前の障害物までの距離を測って" and [c[0] for c in calls] == ["read_distance"]:
        return {
            "text": case.text,
            "passed": True,
            "motion_error": False,
            "seconds": round(elapsed, 2),
            "tools": ["read_distance"],
            "reply": "",
        }
    return result_row(plain, str(response.content), calls, {}, elapsed, check_fact=False)


def result_row(case, reply, calls, facts, elapsed, check_fact=True) -> dict:
    checked = (
        case
        if check_fact
        else Case(case.text, case.tool, case.reply_has, case.args, tools=case.tools)
    )
    names = [name for name, _ in calls]
    return {
        "text": case.text,
        "passed": judge(checked, reply, calls, facts),
        # 頼んでいないのに車を動かしたか
        "motion_error": any(
            name in MOTION_TOOLS and name != case.tool and name not in (case.tools or ())
            for name in names
        ),
        "seconds": round(elapsed, 2),
        "tools": names,
        "reply": reply,
    }


CONVERSATION = [
    Case("電池どう？", reply_has="7.6"),
    Case("前回は何の話をしたっけ？", reply_has="キッチン"),
    Case("私の名前はたろうです。覚えておいて", "remember_fact", fact=("たろう", True)),
    Case("待って", "stop_moving"),
    Case("少し前に進んで", "drive_forward", args={"duration_ms": 500}),
    Case("右に曲がって", "steer_wheels", args={"angle": 30}),
    Case("こんにちは！元気？"),
    Case("今の電池は何ボルト？", reply_has="7.6"),
    Case("バックして", "drive_backward"),
]


async def run_conversation(model: Optional[str]) -> dict:
    """履歴を引き継いで話し、2ターン目以降の応答時間を測る"""
    llm = LLMFactory.create_local(
        model=model, temperature=0.0, format=ACTION_SCHEMA, num_predict=MAX_REPLY_TOKENS
    )
    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        calls: List[Tuple[str, dict]] = []
        tools = {
            tool: RecordingTool(tool, calls)
            for tool in ("drive_forward", "drive_backward", "steer_wheels", "stop_moving")
        }
        memory = make_memory(directory, "conversation")
        original_remember = memory.remember_fact

        def remember(key: str, value: str) -> None:
            calls.append(("remember_fact", {"key": key, "value": value}))
            original_remember(key, value)

        memory.remember_fact = remember
        agent = LocalAgent(llm=llm, status_client=status_client(), memory=memory, tools=tools)
        history: List = []
        for case in CONVERSATION:
            calls.clear()
            history.append(HumanMessage(content=case.text))
            started = time.perf_counter()
            result = await agent.ainvoke({"messages": history})
            elapsed = time.perf_counter() - started
            history = result["messages"]
            row = result_row(case, history[-1].content, list(calls), memory.facts(), elapsed)
            rows.append(row)
            mark = "OK" if row["passed"] else "NG"
            print(
                f"[{mark}] {row['seconds']}s {row['text']} -> {row['tools']} {row['reply']!r}",
                flush=True,
            )
        memory.close()
    timed = [row["seconds"] for row in rows]
    return {
        "model": model or "default",
        "mode": "conversation",
        "repeat": 1,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "motion_errors": sum(row["motion_error"] for row in rows),
        "median_seconds": statistics.median(timed),
        "max_seconds": max(timed),
        "rows": rows,
    }


async def evaluate(
    model: Optional[str], repeat: int, baseline: bool, cases: Optional[List[Case]] = None
) -> dict:
    cases = cases or CASES
    if baseline:
        llm = LLMFactory.create_local(model=model, temperature=0.2).bind_tools(ALL_TOOLS)
        run_case = run_baseline_case
    else:
        llm = LLMFactory.create_local(
            model=model, temperature=0.0, format=ACTION_SCHEMA, num_predict=MAX_REPLY_TOKENS
        )
        run_case = run_local_case

    rows = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        for attempt in range(repeat):
            for index, case in enumerate(cases):
                try:
                    row = await run_case(llm, case, directory, f"{attempt}_{index}")
                except Exception as error:
                    row = {
                        "text": case.text,
                        "passed": False,
                        "motion_error": False,
                        "seconds": None,
                        "tools": [],
                        "reply": f"エラー: {error}",
                    }
                rows.append(row)
                mark = "OK" if row["passed"] else "NG"
                print(
                    f"[{mark}] {row['seconds']}s {row['text']} -> {row['tools']} {row['reply']!r}",
                    flush=True,
                )

    timed = [row["seconds"] for row in rows if row["seconds"] is not None]
    return {
        "model": model or "default",
        "mode": "baseline" if baseline else "local_agent",
        "repeat": repeat,
        "passed": sum(row["passed"] for row in rows),
        "total": len(rows),
        "motion_errors": sum(row["motion_error"] for row in rows),
        "median_seconds": statistics.median(timed) if timed else None,
        "max_seconds": max(timed) if timed else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Ollama のモデル名。省略時は既定のモデル")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--conversation", action="store_true")
    parser.add_argument("--heldout", action="store_true", help="調整に使っていない発言で測る")
    parser.add_argument("--compound", action="store_true", help="曲がりながら走る指示を測る")
    args = parser.parse_args()

    if args.conversation:
        report = asyncio.run(run_conversation(args.model))
    else:
        cases = COMPOUND_CASES if args.compound else HELDOUT_CASES if args.heldout else CASES
        report = asyncio.run(evaluate(args.model, args.repeat, args.baseline, cases))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{report['mode']} {report['model']}: {report['passed']}/{report['total']} 合格、"
        f"誤った走行 {report['motion_errors']} 回、中央値 {report['median_seconds']} 秒"
    )
    print(path)


if __name__ == "__main__":
    main()
