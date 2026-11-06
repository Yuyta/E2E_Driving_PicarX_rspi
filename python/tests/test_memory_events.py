from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage

from src.memory.events import EventRecorder, summarize_event
from src.memory.store import MemoryStore
from src.models import EventsResponse, SafetyEvent
from tests.test_local_agent import make_agent

AT_MS = 1_762_400_000_000


def event(id, kind="reflex_stop", reason="obstacle", at_ms=AT_MS, **extra):
    return SafetyEvent(id=id, at_unix_ms=at_ms, kind=kind, reason=reason, **extra)


def events_client(*responses):
    client = AsyncMock()
    client.get_events = AsyncMock(side_effect=list(responses))
    return client


def test_stops_are_summarized_with_reason_direction_and_distance():
    text = summarize_event(event(1, direction="forward", distance_cm=14.62))
    assert text == "自動で停止した（障害物が近い、前進中、前方 14.6cm）"


def test_routine_ttl_expiry_is_not_remembered():
    assert summarize_event(event(1, reason="ttl_expired")) is None


def test_emergency_stop_is_remembered():
    assert summarize_event(event(1, kind="emergency_stop", reason=None)) == "緊急停止された"


@pytest.mark.asyncio
async def test_recorder_saves_new_events_once_and_survives_a_pi_restart(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = [event(1, at_ms=AT_MS), event(2, reason="cliff", at_ms=AT_MS + 1000)]
    client = events_client(
        EventsResponse(events=first, latest_event_id=2),
        EventsResponse(events=[], latest_event_id=2),
        # Pi の再起動で番号が 1 に戻った。同じ事象は二度保存しない
        EventsResponse(events=[first[0], event(1, at_ms=AT_MS + 9000)], latest_event_id=1),
    )
    recorder = EventRecorder(client, lambda: store)

    assert await recorder.poll() == 2
    assert await recorder.poll() == 0
    assert client.get_events.await_args.kwargs == {"after": 2}
    assert await recorder.poll() == 1
    assert [e.summary for e in store.recent_events()] == [
        "自動で停止した（障害物が近い）",
        "自動で停止した（段差を検知）",
        "自動で停止した（障害物が近い）",
    ]


@pytest.mark.asyncio
async def test_memory_failure_keeps_the_position_so_events_are_retried(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    broken = AsyncMock()
    broken.add_event = lambda *args: (_ for _ in ()).throw(OSError("disk full"))
    response = EventsResponse(events=[event(1)], latest_event_id=1)
    client = events_client(response, response)
    recorder = EventRecorder(client, lambda: broken)

    with pytest.raises(OSError):
        await recorder.poll()
    recorder._memory_getter = lambda: store
    assert await recorder.poll() == 1
    assert client.get_events.await_args.kwargs == {"after": 0}


def test_recent_events_appear_in_the_context_including_this_session(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    at = datetime.fromtimestamp(AT_MS / 1000, tz=timezone.utc)
    store.add_event("reflex_stop", "自動で停止した（段差を検知）", at, "pi:1")
    context = store.build_context("さっき何で止まったの？")
    assert "最近の車体の出来事" in context
    assert "段差を検知" in context
    assert at.astimezone().strftime("%H:%M") in context


def test_old_events_drop_out_of_the_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    for minute in range(8):
        at = datetime(2026, 11, 6, 10, minute, tzinfo=timezone.utc)
        store.add_event("reflex_stop", f"出来事{minute}", at, f"pi:{minute}")
    assert [e.summary for e in store.recent_events(3)] == ["出来事5", "出来事6", "出来事7"]


@pytest.mark.asyncio
async def test_local_agent_gets_events_only_when_asked_about_them(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    store.add_event("reflex_stop", "自動で停止した（段差を検知）", source_key="pi:1")
    agent = make_agent(
        {"action": "reply", "text": "段差を検知して止まりました。"},
        {"action": "reply", "text": "こんにちは。"},
        memory=store,
    )
    asked = [HumanMessage(content="なんで止まったの？")]
    await agent.ainvoke({"messages": asked})
    prompt = agent._llm.ainvoke.await_args.args[0]
    assert any("段差を検知" in str(m.content) for m in prompt)

    await agent.ainvoke({"messages": [HumanMessage(content="こんにちは")]})
    prompt = agent._llm.ainvoke.await_args.args[0]
    assert not any("段差を検知" in str(m.content) for m in prompt)
