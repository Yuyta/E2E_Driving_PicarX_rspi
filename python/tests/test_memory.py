import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.graph import create_agent_graph
from src.config import ModelPreset
from src.llm_benchmark import CASES, judge
from src.memory.store import MEMORY_DB_ENV, MemoryStore, extract_keywords, get_memory_store
from src.models import EventsResponse, RobotStatus
from tests.test_m1_body import status_payload


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "memory.db"


def restart(path) -> MemoryStore:
    """アプリの再起動を模して、新しいセッションで開き直す"""
    return MemoryStore(path)


def test_previous_topic_survives_restart(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "今日はキッチンを探検したよ")
    first.add_message("assistant", "キッチンは楽しそうですね")
    first.close()

    second = restart(db_path)
    context = second.build_context("前回何の話をした？")
    assert "キッチンを探検" in context
    assert "前回の会話の終わり" in context
    assert "覚えていないと答える" in context


def test_current_session_is_not_repeated_as_memory(db_path):
    store = MemoryStore(db_path)
    store.add_message("user", "キッチンの話")
    assert store.build_context("キッチン") == ""
    assert store.search("キッチン") == []


def test_empty_memory_adds_nothing(db_path):
    assert MemoryStore(db_path).build_context("こんにちは") == ""


def test_facts_persist_update_and_can_be_forgotten(db_path):
    store = MemoryStore(db_path)
    store.remember_fact("ユーザーの名前", "たろう")
    store.remember_fact("ユーザーの名前", "はなこ")
    store.close()

    store = restart(db_path)
    assert store.facts() == {"ユーザーの名前": "はなこ"}
    assert "ユーザーの名前: はなこ" in store.build_context("こんにちは")

    assert store.forget_fact("ユーザーの名前") is True
    assert store.forget_fact("ユーザーの名前") is False
    assert store.facts() == {}


def test_keywords_drop_hiragana_and_keep_names():
    assert extract_keywords("キッチンの話を覚えてる？") == ["キッチン"]
    assert extract_keywords("電池とPiCarの話") == ["電池", "PiCar"]
    assert extract_keywords("こんにちは") == []


def test_search_finds_two_and_three_character_japanese_words(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "電池が切れそう")
    first.add_message("user", "キッチンへ行った")
    first.add_message("user", "関係のない話")
    first.close()

    store = restart(db_path)
    assert [m.content for m in store.search("電池は？")] == ["電池が切れそう"]
    assert [m.content for m in store.search("キッチンはどこ")] == ["キッチンへ行った"]
    assert store.search("ベランダ") == []


def test_search_ranks_more_matches_first(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "電池の話")
    first.add_message("user", "電池とキッチンの話")
    first.add_message("user", "キッチンの話")
    first.close()

    results = restart(db_path).search("電池 キッチン")
    assert results[0].content == "電池とキッチンの話"
    assert len(results) == 3


def test_search_is_safe_with_fts_syntax_characters(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", 'quote"test OR NOT')
    first.close()
    store = restart(db_path)
    assert store.search('quote" OR NOT content:') != []


def test_old_related_messages_are_found_beyond_the_previous_session(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "猫の名前はミケだよ")
    first.close()
    middle = restart(db_path)
    for index in range(8):
        middle.add_message("user", f"別の話題 {index}")
    middle.close()

    context = restart(db_path).build_context("ミケって誰だっけ")
    assert "関連する過去の会話" in context
    assert "猫の名前はミケ" in context


def test_long_messages_are_shortened_in_the_context(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "あ" * 1000)
    first.close()
    assert "あ" * 201 not in restart(db_path).build_context("")


def test_store_can_be_used_from_several_threads(db_path):
    store = MemoryStore(db_path)
    threads = [
        threading.Thread(target=store.add_message, args=("user", f"話題{i}")) for i in range(20)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(restart(db_path).previous_session_messages(limit=50)) == 20


def test_database_path_can_be_set_by_environment(db_path, monkeypatch):
    monkeypatch.setenv(MEMORY_DB_ENV, str(db_path))
    monkeypatch.setattr("src.memory.store._default_store", None)
    store = get_memory_store()
    assert get_memory_store() is store
    store.close()
    assert db_path.exists()
    monkeypatch.setattr("src.memory.store._default_store", None)


def test_default_database_is_inside_the_ignored_data_directory():
    from src.memory.store import DEFAULT_DB_PATH

    assert DEFAULT_DB_PATH.parts[-3:] == ("data", "memory", "memory.db")
    assert DEFAULT_DB_PATH.parent.parent.parent.name == "python"


@pytest.mark.asyncio
async def test_llm_receives_memory_and_state_after_restart(db_path):
    first = MemoryStore(db_path)
    first.add_message("user", "キッチンを探検したよ")
    first.remember_fact("ユーザーの名前", "たろう")
    first.close()

    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="キッチンの話でした"))
    client = MagicMock()
    client.get_status = AsyncMock(return_value=RobotStatus.model_validate(status_payload()))
    client.get_events = AsyncMock(return_value=EventsResponse(events=[], latest_event_id=0))

    with patch("src.llm_factory.LLMFactory.create", return_value=llm):
        app = create_agent_graph(
            preset=ModelPreset.GEMINI_FLASH, status_client=client, memory=restart(db_path)
        )
    await app.ainvoke(
        {
            "messages": [HumanMessage(content="前回何の話をした？")],
        }
    )

    context = llm.ainvoke.await_args.args[0][-2].content
    assert "キッチンを探検" in context
    assert "ユーザーの名前: たろう" in context
    assert "7.60V" in context


def test_broken_memory_does_not_stop_the_conversation():
    import sqlite3

    from src.agents.graph import memory_context

    broken = MagicMock()
    broken.build_context.side_effect = sqlite3.OperationalError("disk I/O error")
    assert memory_context(broken, [HumanMessage(content="こんにちは")]) == ""


def case_named(name):
    return next(case for case in CASES if case.name == name)


def test_benchmark_judges_tool_calls_and_measured_values():
    assert judge(case_named("stop_tool"), "", ["stop_moving"]) is True
    assert judge(case_named("stop_tool"), "止まります", []) is False
    assert judge(case_named("battery_known"), "電池は7.62Vです", []) is True
    assert judge(case_named("battery_known"), "電池は十分です", []) is False
    assert judge(case_named("battery_unknown"), "電池の状態は不明です", []) is True
    assert judge(case_named("battery_unknown"), "電池は50%で不明です", []) is False
    assert judge(case_named("greeting"), "こんにちは", []) is True
    assert judge(case_named("greeting"), "Hello", []) is False
    assert judge(case_named("greeting"), "こんにちは", ["stop_moving"]) is False


def test_context_goes_right_before_the_latest_message_during_tool_loops():
    from langchain_core.messages import SystemMessage, ToolMessage

    from src.agents.graph import with_context

    messages = [
        HumanMessage(content="前の発言"),
        AIMessage(content="応答"),
        HumanMessage(content="止まって"),
        AIMessage(content="", tool_calls=[{"name": "stop_moving", "args": {}, "id": "1"}]),
        ToolMessage(content="停止完了", tool_call_id="1"),
    ]
    result = with_context(messages, "状態")
    assert isinstance(result[2], SystemMessage)
    assert "状態" in result[2].content
    assert result[3].content == "止まって"
    assert len(result) == len(messages) + 1
    assert len(messages) == 5


def test_benchmark_conversation_requires_real_tool_calls_not_imitated_text():
    from src.llm_benchmark import judge_turn

    assert judge_turn("stop_moving", None, "", ["stop_moving"]) is True
    assert judge_turn("stop_moving", None, "（stop_moving を実行しました）", []) is False
    assert judge_turn(None, "7.6", "電池は7.60Vです", []) is True
    assert judge_turn(None, "7.6", "電池は7.60Vです", ["read_distance"]) is False
