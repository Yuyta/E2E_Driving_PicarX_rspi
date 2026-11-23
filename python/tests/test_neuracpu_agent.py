import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import create_agent_graph
from src.agents.neuracpu_agent import (
    NOT_FOUND,
    SEARCH_FAILED,
    NeuraCPUAgent,
    correct_route,
    spoken,
)
from src.config import ModelPreset
from src.llm_factory import LLMFactory
from src.memory.store import MemoryStore
from src.neuracpu.client import SearchResult, Source
from tests.test_local_agent import make_status_client, make_tools


def make_llm(*replies):
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=json.dumps(r, ensure_ascii=False)) for r in replies]
    )
    return llm


class FakeSearch:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    async def search(self, kind, query, max_chars=700):
        self.calls.append((kind, query))
        return self.results.get(kind, SearchResult(True, kind))


MANUAL = SearchResult(
    True,
    "local",
    "local",
    [Source("manual.html", "▲▼ で 0.4 秒だけ前進・後退", "manual.html", "行 120-130")],
)
WIKI = SearchResult(
    True,
    "web",
    "wikipedia",
    [Source("富士山", "標高3776 m", "Wikipedia", "", "https://ja.wikipedia.org/wiki/x")],
)


def make_agent(*replies, answer="0.4秒です[1]。", search=None, tools=None, memory=None):
    answer_llm = AsyncMock()
    answer_llm.ainvoke = AsyncMock(return_value=AIMessage(content=answer))
    return NeuraCPUAgent(
        llm=make_llm(*replies),
        answer_llm=answer_llm,
        search_client=search or FakeSearch({"local": MANUAL, "web": WIKI}),
        status_client=make_status_client(),
        memory=memory or MemoryStore(":memory:"),
        tools=tools or make_tools(),
    )


async def say(agent, text):
    result = await agent.ainvoke({"messages": [HumanMessage(content=text)]})
    return result["messages"][-1]


async def test_local_search_answers_with_sources_and_speaks_without_markers():
    agent = make_agent({"action": "search_local"})
    reply = await say(agent, "▲を押すと何秒進む？")

    assert agent._search.calls == [("local", "▲を押すと何秒進む？")]
    assert "0.4秒です" in reply.content
    assert "出典: manual.html 行 120-130" in reply.content
    assert reply.additional_kwargs["speech"] == "0.4秒です。"
    evidence = agent._answer_llm.ainvoke.await_args.args[0][-1].content
    assert "<資料>" in evidence and "0.4 秒" in evidence


async def test_web_search_cites_wikipedia_title():
    agent = make_agent({"action": "search_web"}, answer="3776メートルです。")
    reply = await say(agent, "富士山の高さは？")

    assert agent._search.calls == [("web", "富士山の高さは？")]
    assert "Wikipedia「富士山」" in reply.content


async def test_empty_local_search_falls_back_to_web():
    search = FakeSearch({"local": SearchResult(True, "local"), "web": WIKI})
    agent = make_agent({"action": "search_local"}, search=search)
    await say(agent, "富士山について資料を調べて")

    assert [kind for kind, _ in search.calls] == ["local", "web"]


async def test_nothing_found_does_not_call_the_answer_llm():
    search = FakeSearch({"local": SearchResult(True, "local"), "web": SearchResult(True, "web")})
    agent = make_agent({"action": "search_web"}, search=search)
    reply = await say(agent, "存在しない言葉を調べて")

    assert reply.content == NOT_FOUND
    agent._answer_llm.ainvoke.assert_not_awaited()


async def test_search_error_is_shown_but_not_spoken():
    search = FakeSearch({"web": SearchResult(False, "web", error="ConnectError: refused")})
    agent = make_agent({"action": "search_web"}, search=search)
    reply = await say(agent, "今日のニュースは？")

    assert "ConnectError" in reply.content
    assert reply.additional_kwargs["speech"] == SEARCH_FAILED


async def test_search_never_moves_the_car():
    tools = make_tools()
    agent = make_agent({"action": "search_web"}, tools=tools)
    await say(agent, "前に進んでって英語でなんて言うか調べて")

    assert all(not tool.calls for tool in tools.values())


async def test_motion_still_goes_through_code_validation():
    tools = make_tools()
    agent = make_agent({"action": "forward"}, tools=tools)
    reply = await say(agent, "少し前に進んで")

    assert tools["drive_forward"].calls == [{"speed": 50, "duration_ms": 500}]
    assert reply.content == "前に進みました。"


async def test_prompt_offers_search_actions():
    agent = make_agent({"action": "reply", "text": "こんにちは"})
    await say(agent, "こんにちは")

    prompt = agent._llm.ainvoke.await_args.args[0]
    assert "search_local" in prompt[0].content and "search_web" in prompt[0].content


@pytest.mark.parametrize(
    "action, text, expected",
    [
        ("reply", "今日の東京の天気は？", "search_web"),
        ("reply", "ネットで llama.cpp について調べて", "search_web"),
        ("reply", "マニュアルで交差検証の使い方を調べて", "search_local"),
        ("reply", "こんにちは", "reply"),
        ("forward", "最新のニュースを見ながら進んで", "forward"),  # 動きの判断は変えない
        ("search_local", "今日の天気は？", "search_local"),  # LLM が調べると決めたら従う
    ],
)
def test_correct_route_only_fixes_replies(action, text, expected):
    assert correct_route(action, text) == expected


def test_spoken_removes_citation_markers():
    assert spoken("0.4秒です[1]。速さは30です［2］") == "0.4秒です。速さは30です"


def test_graph_returns_neuracpu_agent(monkeypatch):
    created = {}
    monkeypatch.setattr(
        LLMFactory, "create_local", staticmethod(lambda **kw: created.setdefault("kw", kw))
    )
    agent = create_agent_graph(
        preset=ModelPreset.NEURACPU, status_client=make_status_client(), memory=None
    )
    assert isinstance(agent, NeuraCPUAgent)
    assert created["kw"]["model"] == "qwen3.5:4b-text"


async def test_memory_question_is_answered_from_memory_not_documents():
    memory = MemoryStore(":memory:")
    memory.remember_fact("前回の話題", "キッチンの掃除")
    agent = make_agent(
        {"action": "search_local"}, answer="キッチンの掃除の話でした。", memory=memory
    )
    reply = await say(agent, "前回は何の話をしたっけ？")

    assert agent._search.calls == []
    assert "出典: 会話の記憶" in reply.content
    assert "キッチン" in agent._answer_llm.ainvoke.await_args.args[0][-1].content
