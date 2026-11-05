import json
from unittest.mock import AsyncMock

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.body_context import answer_state_question, state_topics
from src.agents.graph import create_agent_graph
from src.agents.local_agent import (
    HISTORY_MAX_MESSAGES,
    LOCAL_PROMPT,
    NOT_UNDERSTOOD,
    REFUSED_MOTION,
    LocalAction,
    LocalAgent,
    allows_motion,
    parse_action,
)
from src.config import ModelPreset
from src.memory.store import MemoryStore
from src.models import RobotStatus
from tests.test_m1_body import status_payload


class FakeTool:
    def __init__(self, result="成功"):
        self.result = result
        self.calls = []

    async def ainvoke(self, args):
        self.calls.append(args)
        return self.result


def make_tools(result="成功"):
    return {
        name: FakeTool(result)
        for name in ("drive_forward", "drive_backward", "steer_wheels", "stop_moving")
    }


def make_status_client(status=None, error=None):
    client = AsyncMock()
    client.get_status = AsyncMock(
        return_value=status or RobotStatus.model_validate(status_payload()), side_effect=error
    )
    return client


def make_llm(*replies):
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=json.dumps(reply, ensure_ascii=False)) for reply in replies]
    )
    return llm


def make_agent(*replies, tools=None, memory=None, result="成功"):
    return LocalAgent(
        llm=make_llm(*replies),
        status_client=make_status_client(),
        memory=memory or MemoryStore(":memory:"),
        tools=tools or make_tools(result),
    )


async def say(agent, text, history=None):
    messages = (history or []) + [HumanMessage(content=text)]
    result = await agent.ainvoke({"messages": messages})
    return result["messages"]


def sent_prompt(agent, call=-1):
    return agent._llm.ainvoke.await_args_list[call].args[0]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action":"forward"}', LocalAction("forward")),
        ('{"action":"reply","text":" はい "}', LocalAction("reply", text="はい")),
        ('{"action":"fly"}', None),
        ("[1]", None),
        ('{"action":"reply","text":"途中で', None),
    ],
)
def test_parse_action(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "action, text, expected",
    [
        ("forward", "少し前に進んで", True),
        ("forward", "前回の旅行は楽しかった", False),
        ("forward", "前に進まないで", False),
        ("forward", "進化って面白いよね", False),
        ("forward", "「前に進んで」って英語で何て言うの？", False),
        ("backward", "昨日バックで駐車するのに苦労したよ", False),
        ("backward", "バックして", True),
        ("backward", "前に進んで", False),
        ("right", "右に曲がって", True),
        ("right", "左に曲がって", False),
        ("center", "タイヤをまっすぐにして", True),
    ],
)
def test_allows_motion(action, text, expected):
    assert allows_motion(action, text) is expected


@pytest.mark.parametrize(
    "text, topics",
    [
        ("電池どう？", ["battery"]),
        ("今の電池は何ボルト？", ["battery"]),
        ("前の障害物までの距離を測って", ["distance"]),
        ("前に進める？", ["movable"]),
        ("調子はどう？", ["summary"]),
        ("電池を買ってきて", []),
        ("昨日の電池はどうだった？", []),
        ("少し前に進んで", []),
        ("こんにちは！元気？", []),
    ],
)
def test_state_topics(text, topics):
    assert state_topics(text) == topics


@pytest.mark.asyncio
async def test_state_question_is_answered_from_the_measurement_without_llm():
    agent = make_agent()
    messages = await say(agent, "今の電池は何ボルト？")
    assert "7.60V" in messages[-1].content
    agent._llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_state_question_reports_unreachable_body():
    client = make_status_client(error=httpx.ConnectError("down"))
    assert "通信できない" in await answer_state_question("電池どう？", client)


@pytest.mark.asyncio
async def test_unknown_battery_is_not_guessed():
    payload = status_payload()
    payload["sensors"]["battery"] = {"voltage": None, "percent": None, "age_ms": None}
    client = make_status_client(status=RobotStatus.model_validate(payload))
    assert await answer_state_question("電池どう？", client) == "電池の値は分かりません。"


@pytest.mark.asyncio
async def test_drive_uses_code_chosen_speed_and_one_llm_call():
    tools = make_tools("前進成功")
    agent = make_agent({"action": "forward"}, tools=tools)
    messages = await say(agent, "少し前に進んで")
    assert tools["drive_forward"].calls == [{"speed": 50, "duration_ms": 500}]
    assert messages[-1].content == "前に進みました。"
    assert agent._llm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_drive_without_a_motion_word_is_refused():
    tools = make_tools()
    agent = make_agent({"action": "forward"}, tools=tools)
    messages = await say(agent, "前回の旅行は楽しかったね")
    assert tools["drive_forward"].calls == []
    assert messages[-1].content == REFUSED_MOTION


@pytest.mark.asyncio
async def test_stop_is_always_allowed():
    tools = make_tools("停止完了")
    agent = make_agent({"action": "stop"}, tools=tools)
    messages = await say(agent, "危ないからやめて")
    assert tools["stop_moving"].calls == [{}]
    assert messages[-1].content == "停止しました。"


@pytest.mark.asyncio
async def test_failed_drive_is_reported_from_the_tool_result():
    agent = make_agent({"action": "forward"}, result="前進失敗")
    messages = await say(agent, "前に進んで")
    assert messages[-1].content == "前進できませんでした（前進失敗）。"


@pytest.mark.asyncio
async def test_remember_saves_a_fact_found_in_the_utterance():
    memory = MemoryStore(":memory:")
    action = {"action": "remember", "key": "ユーザーの名前", "value": "たろう"}
    agent = make_agent(action, memory=memory)
    await say(agent, "私の名前はたろうです。覚えておいて")
    assert memory.facts() == {"ユーザーの名前": "たろう"}


@pytest.mark.parametrize("value", ["覚えておいて", "じろう", ""])
@pytest.mark.asyncio
async def test_remember_rejects_a_wrong_value(value):
    memory = MemoryStore(":memory:")
    agent = make_agent({"action": "remember", "key": "たろう", "value": value}, memory=memory)
    messages = await say(agent, "私の名前はたろうです。覚えておいて")
    assert memory.facts() == {}
    assert messages[-1].content == NOT_UNDERSTOOD


@pytest.mark.asyncio
async def test_forget_removes_a_fact():
    memory = MemoryStore(":memory:")
    memory.remember_fact("飼い猫の名前", "ミケ")
    agent = make_agent({"action": "forget", "key": "飼い猫の名前"}, memory=memory)
    messages = await say(agent, "飼い猫の名前は忘れて")
    assert memory.facts() == {}
    assert messages[-1].content == "飼い猫の名前を忘れました。"


@pytest.mark.asyncio
async def test_broken_output_does_not_move_the_car():
    tools = make_tools()
    agent = LocalAgent(
        llm=AsyncMock(ainvoke=AsyncMock(return_value=AIMessage(content="前に進みます"))),
        status_client=make_status_client(),
        memory=MemoryStore(":memory:"),
        tools=tools,
    )
    messages = await say(agent, "前に進んで")
    assert tools["drive_forward"].calls == []
    assert messages[-1].content == NOT_UNDERSTOOD


@pytest.mark.asyncio
async def test_prompt_prefix_stays_the_same_between_turns():
    agent = make_agent({"action": "reply", "text": "はい"}, {"action": "reply", "text": "うん"})
    first = await say(agent, "こんにちは")
    await say(agent, "元気？", history=first)
    previous, current = sent_prompt(agent, 0), sent_prompt(agent, 1)
    assert current[0].content == LOCAL_PROMPT
    assert [m.content for m in current[: len(previous)]] == [m.content for m in previous]


@pytest.mark.asyncio
async def test_memory_context_is_sent_only_when_needed_and_stays_in_place():
    memory = MemoryStore(":memory:")
    memory.remember_fact("飼い猫の名前", "ミケ")
    agent = make_agent(
        {"action": "reply", "text": "ミケです"},
        {"action": "reply", "text": "はい"},
        memory=memory,
    )
    first = await say(agent, "飼い猫の名前は何？")
    await say(agent, "ありがとう", history=first)
    previous, current = sent_prompt(agent, 0), sent_prompt(agent, 1)
    assert "ミケ" in previous[1].content
    assert [m.content for m in current[: len(previous)]] == [m.content for m in previous]


@pytest.mark.asyncio
async def test_unrelated_utterance_gets_no_memory():
    memory = MemoryStore(":memory:")
    memory.remember_fact("飼い猫の名前", "ミケ")
    agent = make_agent({"action": "reply", "text": "こんにちは"}, memory=memory)
    await say(agent, "こんにちは")
    assert not any("ミケ" in m.content for m in sent_prompt(agent))


@pytest.mark.asyncio
async def test_long_history_is_trimmed_at_once_and_starts_with_a_user_message():
    agent = make_agent({"action": "reply", "text": "はい"})
    history = []
    for i in range(HISTORY_MAX_MESSAGES):
        history += [HumanMessage(content=f"質問{i}"), AIMessage(content=f"回答{i}")]
    await say(agent, "最新", history=history)
    prompt = sent_prompt(agent)
    assert isinstance(prompt[1], HumanMessage)
    assert len(prompt) < HISTORY_MAX_MESSAGES
    assert prompt[-1].content == "最新"


@pytest.mark.asyncio
async def test_tool_history_from_gemini_is_not_sent():
    agent = make_agent({"action": "reply", "text": "はい"})
    call = AIMessage(content="", tool_calls=[{"name": "stop_moving", "args": {}, "id": "x"}])
    history = [HumanMessage(content="止まって"), call, ToolMessage(content="ok", tool_call_id="x")]
    await say(agent, "こんにちは", history=history)
    prompt = sent_prompt(agent)
    assert isinstance(prompt[0], SystemMessage)
    assert [type(m) for m in prompt[1:]] == [HumanMessage, HumanMessage]


def test_local_preset_builds_the_local_agent():
    agent = create_agent_graph(
        preset=ModelPreset.OLLAMA_LOCAL,
        status_client=make_status_client(),
        memory=MemoryStore(":memory:"),
    )
    assert isinstance(agent, LocalAgent)
