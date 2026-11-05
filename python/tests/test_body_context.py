from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.body_context import UNAVAILABLE_CONTEXT, fetch_body_context, format_body_context
from src.agents.graph import ROUTER_PROMPT, create_agent_graph
from src.config import ModelPreset
from src.memory.store import MemoryStore
from src.models import EventsResponse, RobotStatus, SafetyEvent
from tests.test_m1_body import status_payload


def make_status(**kwargs) -> RobotStatus:
    return RobotStatus.model_validate(status_payload(**kwargs))


def stop_event(event_id: int, reason: str) -> SafetyEvent:
    return SafetyEvent(id=event_id, at_unix_ms=0, kind="reflex_stop", reason=reason)


def test_context_contains_measured_values():
    text = format_body_context(make_status())
    assert "7.60V" in text
    assert "50%" in text
    assert "40.0cm" in text
    assert "停止中" in text


def test_unknown_values_are_stated_instead_of_guessed():
    payload = status_payload(forward="distance_unknown", cm=None)
    payload["sensors"]["battery"] = {"voltage": None, "percent": None, "age_ms": None}
    payload["sensors"]["floor"] = {"values": None, "age_ms": None, "cliff": None}
    text = format_body_context(RobotStatus.model_validate(payload))

    assert "- 電池: 不明" in text
    assert "- 前方の距離: 不明" in text
    assert "- 床: 不明" in text
    assert "不可（距離が不明）" in text


def test_far_reading_warns_that_something_may_still_be_there():
    payload = status_payload(cm=343.0)
    payload["sensors"]["distance"]["far"] = True
    text = format_body_context(RobotStatus.model_validate(payload))
    assert "343" not in text
    assert "反射なし" in text


def test_cliff_is_reported():
    text = format_body_context(make_status(forward="cliff", cliff=[True, False, False]))
    assert "段差を検知している" in text
    assert "不可（段差を検知）" in text


def test_only_recent_hazard_stops_are_reported():
    events = [
        stop_event(1, "obstacle"),
        stop_event(2, "ttl_expired"),
        stop_event(3, "cliff"),
        SafetyEvent(id=4, at_unix_ms=0, kind="cliff_calibrated"),
    ]
    text = format_body_context(make_status(), events)
    assert "最近の安全停止: 障害物が近い、段差を検知" in text
    assert "指令の期限切れ" not in text


@pytest.mark.asyncio
async def test_unreachable_robot_does_not_stop_the_conversation():
    client = MagicMock()
    client.get_status = AsyncMock(side_effect=httpx.ConnectTimeout("timed out"))
    assert await fetch_body_context(client) == UNAVAILABLE_CONTEXT


@pytest.mark.asyncio
async def test_event_failure_still_reports_status():
    client = MagicMock()
    client.get_status = AsyncMock(return_value=make_status())
    client.get_events = AsyncMock(side_effect=httpx.ConnectTimeout("timed out"))
    text = await fetch_body_context(client)
    assert "7.60V" in text
    assert "最近の安全停止: なし" in text


@pytest.mark.asyncio
async def test_every_router_turn_sends_the_current_state_to_the_llm():
    llm = MagicMock()
    llm.bind_tools.return_value = llm
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="電池は7.60Vです"))
    client = MagicMock()
    client.get_status = AsyncMock(return_value=make_status())
    client.get_events = AsyncMock(
        return_value=EventsResponse(events=[stop_event(1, "obstacle")], latest_event_id=1)
    )

    with patch("src.llm_factory.LLMFactory.create", return_value=llm):
        app = create_agent_graph(
            preset=ModelPreset.GEMINI_FLASH, status_client=client, memory=MemoryStore(":memory:")
        )
    await app.ainvoke(
        {
            "messages": [HumanMessage(content="電池どう？")],
            "current_agent": "main",
            "is_finished": False,
        }
    )

    sent = llm.ainvoke.await_args.args[0]
    context = sent[-2].content
    assert "7.60V" in context
    assert "最近の安全停止: 障害物が近い" in context
    assert "推測しないこと" in context
    # 発言と固定の指示には混ぜない
    assert sent[-1].content == "電池どう？"
    assert sent[0].content == ROUTER_PROMPT
