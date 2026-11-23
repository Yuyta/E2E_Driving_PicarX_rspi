import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import create_agent_graph
from src.agents.local_agent import Motion
from src.agents.neuracpu_agent import (
    NOT_FOUND,
    SEARCH_FAILED,
    NeuraCPUAgent,
    best_sentence,
    correct_route,
    fast_motion,
    is_direct_question,
    spoken,
    ungrounded_numbers,
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


def make_agent(
    *replies, answer="0.4秒です[1]。", search=None, tools=None, memory=None, direct=False
):
    answer_llm = AsyncMock()
    answer_llm.ainvoke = AsyncMock(return_value=AIMessage(content=answer))
    # answer_in_context の時は、判断と同じ LLM が続けて JSON で答える
    # 判断の後、同じ LLM が同じ会話の続きで答えを JSON で返す
    answer_reply = {"action": "reply", "text": answer}
    return NeuraCPUAgent(
        llm=make_llm(*replies, answer_reply),
        answer_llm=answer_llm,
        knowledge_llm=answer_llm,
        search_client=search or FakeSearch({"local": MANUAL, "web": WIKI}),
        status_client=make_status_client(),
        memory=memory or MemoryStore(":memory:"),
        tools=tools or make_tools(),
        direct_search=direct,
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


async def test_answer_in_context_reuses_the_routing_prompt():
    agent = make_agent({"action": "search_local"})
    agent._answer_in_context = True
    await say(agent, "▲を押すと何秒進む？")

    first, second = [call.args[0] for call in agent._llm.ainvoke.await_args_list]
    # 判断と同じプロンプトの続きで答えを作る（先頭を読み直さない）
    assert second[: len(first)] == first
    assert "<資料>" in second[-1].content and "0.4 秒" in second[-1].content
    agent._answer_llm.ainvoke.assert_not_awaited()


async def test_web_search_cites_wikipedia_title():
    agent = make_agent({"action": "search_web"}, answer="3776メートルです。")
    reply = await say(agent, "富士山の高さをネットで調べて")

    assert agent._search.calls == [("web", "富士山の高さをネットで調べて")]
    assert "Wikipedia「富士山」" in reply.content


async def test_empty_local_search_falls_back_to_web():
    search = FakeSearch({"local": SearchResult(True, "local"), "web": WIKI})
    agent = make_agent({"action": "search_local"}, search=search)
    await say(agent, "富士山について資料を調べて")

    assert [kind for kind, _ in search.calls] == ["local", "web"]


async def test_nothing_found_on_the_web_answers_from_the_model_and_says_so():
    search = FakeSearch({"local": SearchResult(True, "local"), "web": SearchResult(True, "web")})
    agent = make_agent({"action": "search_web"}, search=search, answer="分かりません。")
    reply = await say(agent, "存在しない言葉を調べて")

    assert "Web で見つからなかった" in reply.content
    assert reply.additional_kwargs["speech"] == "分かりません。"


async def test_web_error_is_shown_but_not_spoken():
    search = FakeSearch({"web": SearchResult(False, "web", error="HTTPError 429")})
    agent = make_agent({"action": "search_web"}, search=search, answer="晴れの予報です。")
    reply = await say(agent, "今日のニュースは？")

    assert "429" in reply.content
    assert reply.additional_kwargs["speech"] == "晴れの予報です。"


async def test_local_search_error_is_reported():
    search = FakeSearch({"local": SearchResult(False, "local", error="ConnectError: refused")})
    agent = make_agent({"action": "search_local"}, search=search)
    reply = await say(agent, "▲を押すと何秒進む？")

    assert "ConnectError" in reply.content
    assert reply.additional_kwargs["speech"] == SEARCH_FAILED


async def test_search_never_moves_the_car():
    tools = make_tools()
    agent = make_agent({"action": "search_web"}, tools=tools)
    await say(agent, "前に進んでって英語でなんて言うか調べて")

    assert all(not tool.calls for tool in tools.values())


async def test_clear_motion_skips_the_llm():
    tools = make_tools()
    agent = make_agent(tools=tools)
    reply = await say(agent, "少し前に進んで")

    assert tools["drive_forward"].calls == [{"speed": 50, "duration_ms": 500}]
    assert reply.content == "前に進みました。"
    agent._llm.ainvoke.assert_not_awaited()


async def test_unclear_motion_still_goes_through_the_llm_and_code_validation():
    tools = make_tools()
    agent = make_agent({"action": "forward"}, tools=tools)
    reply = await say(agent, "カレーを食べたよ")

    assert all(not tool.calls for tool in tools.values())
    assert "動かしていません" in reply.content


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
        ("reply", "日本の首都はどこ？", "knowledge"),
        ("search_web", "将棋の駒は全部で何枚？", "knowledge"),
        ("search_web", "水の化学式は？", "knowledge"),
        ("search_web", "太陽系で一番大きい惑星は？", "knowledge"),
        ("reply", "こんにちは！元気？", "reply"),
        ("forward", "最新のニュースを見ながら進んで", "forward"),  # 動きの判断は変えない
        ("search_local", "今日の天気は？", "search_web"),  # 時事は Web
        ("search_local", "マニュアルの最新の手順は？", "search_local"),  # 資料を指せば資料
        ("stop", "緊急停止の解除はどうするの？", "search_local"),  # 質問は止まる指示ではない
        ("stop", "危ないからやめて", "stop"),
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
    assert created["kw"]["model"] == "neuracpu-qwen2.5-3b"


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


@pytest.mark.parametrize(
    "text, expected",
    [
        ("少し前に進んで", Motion(drive="forward")),
        ("まっすぐ行って", Motion(drive="forward")),
        ("少し戻って", Motion(drive="backward")),
        ("後退して", Motion(drive="backward")),
        ("右に曲がって", Motion(steer="right")),
        ("左を向いて", Motion(steer="left")),
        ("ハンドルを左に", Motion(steer="left")),
        ("タイヤをまっすぐにして", Motion(steer="center")),
        ("舵を真ん中に戻して", Motion(steer="center")),
        ("右にカーブしながら前進して", Motion(drive="forward", steer="right")),
        ("左に曲がりながら後ろに下がって", Motion(drive="backward", steer="left")),
        ("前へお願い", Motion(drive="forward")),
        # 迷う発言や、動く指示でない発言は LLM に任せる
        ("前に進みながら曲がって", None),
        ("うしろに行って", None),
        ("右の部屋は寒いね", None),
        ("左利きなんだ", None),
        ("前に食べたカレーがおいしかった", None),
        ("右と左どっちが好き？", None),
        ("「前に進んで」って英語で何て言うの？", None),
        ("昨日バックで駐車するのに苦労したよ", None),
        ("危ないからやめて", None),
        ("もう動かないで", None),
        ("緊急停止の解除はどうするの？", None),
        ("Control Center の開き方を教えて", None),
    ],
)
def test_fast_motion(text, expected):
    assert fast_motion(text) == expected


def test_ungrounded_numbers_are_detected():
    evidence = "▲▼ で 0.4 秒だけ前進・後退（速さ 30）"
    assert ungrounded_numbers("0.4秒で、速さは30です。", evidence) == []
    assert ungrounded_numbers("1.5秒です。", evidence) == ["1.5"]
    # 「3」は「30」の一部では根拠にならない
    assert ungrounded_numbers("3秒前進します。", evidence) == ["3"]


async def test_ungrounded_answer_is_replaced_by_the_source_sentence():
    agent = make_agent({"action": "search_local"}, answer="1.5秒進みます。")
    reply = await say(agent, "▲を押すと何秒進む？")

    assert reply.additional_kwargs["speech"] == "資料には「▲▼ で 0.4 秒だけ前進・後退」とあります。"


def test_best_sentence_prefers_overlap():
    evidence = "[1] manual.html\n安全のきまり。\n▲▼ で 0.4 秒だけ前進・後退。"
    assert best_sentence("▲を押すと何秒前進する？", evidence) == "▲▼ で 0.4 秒だけ前進・後退。"


async def test_turn_without_side_asks_which_way_without_the_llm():
    agent = make_agent()
    reply = await say(agent, "前に進みながら曲がって")
    assert "どちら" in reply.content
    agent._llm.ainvoke.assert_not_awaited()


async def test_empty_reply_falls_back_to_a_plain_answer():
    agent = make_agent({"action": "reply"}, answer="ミケは飼い猫です。")
    reply = await say(agent, "ミケって誰？")
    assert reply.content == "ミケは飼い猫です。"


async def test_how_to_question_is_not_answered_with_the_battery_state():
    agent = make_agent({"action": "search_local"}, answer="資料には見つかりませんでした")
    reply = await say(agent, "バッテリーの交換手順は？")
    assert agent._search.calls[0][0] == "local"
    assert "V" not in reply.additional_kwargs.get("speech", reply.content)


async def test_web_answers_are_not_replaced_by_quotes():
    agent = make_agent({"action": "search_web"}, answer="333メートルです。")
    reply = await say(agent, "東京タワーの高さをネットで調べて")
    assert reply.additional_kwargs["speech"] == "333メートルです。"


async def test_unparseable_answer_uses_the_separate_answer_prompt():
    agent = make_agent({"action": "search_local"}, answer="")
    agent._answer_in_context = True
    agent._answer_llm.ainvoke.return_value = AIMessage(content="0.4秒です。")
    reply = await say(agent, "▲を押すと何秒進む？")
    assert reply.additional_kwargs["speech"] == "0.4秒です。"


async def test_clear_document_question_skips_the_routing_llm():
    agent = make_agent(direct=True)
    reply = await say(agent, "▲を押すと何秒進む？")
    assert agent.last_action == "direct:search_local"
    assert "0.4秒です" in reply.content
    agent._llm.ainvoke.assert_not_awaited()


async def test_news_question_goes_straight_to_the_web():
    agent = make_agent(direct=True, answer="晴れです。")
    await say(agent, "今日の東京の天気は？")
    assert agent._search.calls == [("web", "今日の東京の天気は？")]
    agent._llm.ainvoke.assert_not_awaited()


async def test_question_without_a_document_match_goes_to_the_routing_llm():
    search = FakeSearch({"local": SearchResult(True, "local")})
    agent = make_agent(
        {"action": "reply", "text": "東京です。"}, search=search, direct=True, answer="東京です。"
    )
    reply = await say(agent, "日本の首都はどこ？")
    assert reply.content == "東京です。"
    assert agent._llm.ainvoke.await_count == 1
    assert agent.last_action == "knowledge"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Control Center を開くコマンドは？", True),
        ("今日の天気は？", True),
        ("右に曲がってくれる？", False),
        ("前回は何の話をしたっけ？", False),
        ("私の名前は覚えてる？", False),
        ("「前に進んで」って英語で何て言うの？", False),
        ("右と左どっちが好き？", False),
        ("少し前に進んで", False),
    ],
)
def test_is_direct_question(text, expected):
    assert is_direct_question(text) is expected


async def test_general_fact_question_is_answered_by_the_knowledge_model():
    agent = make_agent({"action": "search_web"}, answer="東京タワーは333メートルです。")
    reply = await say(agent, "東京タワーの高さは？")

    assert agent.last_action == "knowledge"
    assert agent._search.calls == []
    assert reply.content == "東京タワーは333メートルです。"


async def test_document_question_with_no_match_does_not_make_things_up():
    search = FakeSearch({"local": SearchResult(True, "local"), "web": SearchResult(True, "web")})
    agent = make_agent({"action": "search_local"}, search=search, answer="1.5kgです。")
    reply = await say(agent, "PiCar-X の車体の重さは何グラム？")

    assert reply.content == NOT_FOUND
    agent._answer_llm.ainvoke.assert_not_awaited()


async def test_screen_question_is_not_a_camera_question():
    agent = make_agent(direct=True, answer="概要があります。")
    vision = AsyncMock()
    agent._vision = vision
    await say(agent, "Control Center のサイドバーには一番上に何がある？")
    vision.describe.assert_not_awaited()
    assert agent.last_action == "direct:search_local"


async def test_memory_miss_falls_back_to_general_knowledge():
    memory = MemoryStore(":memory:")
    memory.remember_fact("ユーザーの名前", "たろう")
    agent = make_agent({"action": "reply"}, memory=memory)
    agent._answer_llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="資料には見つかりませんでした"),
            AIMessage(content="月です。"),
        ]
    )
    agent._knowledge_llm = agent._answer_llm
    reply = await say(agent, "地球の衛星の名前は？")
    assert reply.content == "月です。"
