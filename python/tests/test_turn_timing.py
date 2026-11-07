import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.metrics.turn_report import load, summarize
from src.metrics.turn_timing import (
    ACK_AUDIO,
    REPLY_AUDIO,
    REPLY_READY,
    TIMINGS_ENV,
    TurnTiming,
    save,
    timings_path,
)
from src.voice.listener import Transcript
from src.voice.speaker import Speaker
from tests.test_gui_emergency_stop import (  # noqa: F401
    FakeRecorder,
    FakeTranscriber,
    send_chat,
    wait_until,
    window,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def test_marks_are_seconds_from_the_end_of_input_and_only_the_first_counts():
    clock = Clock()
    timing = TurnTiming("text", clock=clock)
    clock.now = 101.2
    timing.mark(REPLY_READY)
    clock.now = 101.9
    timing.mark(REPLY_AUDIO)
    clock.now = 105.0
    timing.mark(REPLY_AUDIO)

    record = timing.record()
    assert record["reply_ready_sec"] == 1.2
    assert record["reply_audio_sec"] == pytest.approx(1.9)
    assert record["first_audio_sec"] == pytest.approx(1.9)
    assert "返事の作成 1.2秒" in timing.summary()


def test_acknowledgement_counts_as_the_first_audio():
    clock = Clock()
    timing = TurnTiming("voice", stt_sec=2.4, clock=clock)
    clock.now = 102.6
    timing.mark(ACK_AUDIO)
    clock.now = 112.0
    timing.mark(REPLY_AUDIO)
    assert timing.first_audio() == pytest.approx(2.6)
    assert timing.summary().startswith("文字起こし 2.4秒、相づち 2.6秒")


def test_records_hold_no_conversation_text(tmp_path):
    path = tmp_path / "t.jsonl"
    timing = TurnTiming("text")
    timing.model = "gemini-3.5-flash-lite"
    save(timing, path)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert set(record) == {"at", "input", "kind", "model", "ok", "stt_sec", "first_audio_sec"}


def test_tests_never_write_to_the_real_data_directory():
    assert timings_path() == Path(os.environ[TIMINGS_ENV])
    assert "turn_timings" not in str(timings_path())


def test_report_groups_by_kind_and_model_with_median_and_p90(tmp_path):
    path = tmp_path / "t.jsonl"
    rows = [
        {"kind": "llm", "model": "lite", "input": "text", "ok": True, "reply_audio_sec": v}
        for v in (1.0, 2.0, 3.0, 4.0, 10.0)
    ]
    rows.append(
        {"kind": "llm", "model": "lite", "input": "text", "ok": False, "reply_audio_sec": 99}
    )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n", encoding="utf-8")

    table = summarize(load(path))
    cells = table[("llm", "lite", "text")]
    assert cells["n"] == "5"
    assert cells["reply_audio_sec"] == "3.0 / 10.0"
    assert cells["ack_audio_sec"] is None


@pytest.mark.asyncio
async def test_speaker_reports_when_the_first_sentence_is_queued():
    order = []
    client = AsyncMock()
    client.speak = AsyncMock(side_effect=lambda text: order.append(text) or True)
    await Speaker(client).say(
        "一文目です。二文目です。", on_first_sentence=lambda: order.append("*")
    )
    assert order == ["一文目です。", "*", "二文目です。"]


def saved_records():
    path = timings_path()
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_gui_records_a_chat_turn(window):  # noqa: F811
    win, client, app = window

    with (
        patch("src.agents.graph.create_agent_graph"),
        patch.object(
            win,
            "_invoke_agent_with_recovery",
            AsyncMock(
                return_value={"messages": [HumanMessage("電池どう？"), AIMessage("76%です。")]}
            ),
        ),
        patch.object(win, "_remember_turn"),
    ):
        send_chat(win, "電池どう？")
        assert wait_until(lambda: len(saved_records()) == 1, app)

    record = saved_records()[0]
    assert (record["input"], record["kind"], record["ok"]) == ("text", "llm", True)
    assert record["reply_ready_sec"] <= record["reply_audio_sec"]
    assert wait_until(lambda: "[Time]" in win.chat_history.toPlainText(), app)


def test_gui_records_voice_turns_with_transcription_time(window):  # noqa: F811
    win, client, app = window
    client.stop_sync.return_value = True
    win._recorder = FakeRecorder()
    win._transcriber = FakeTranscriber(Transcript("止まって"))
    win._on_talk_pressed()
    win._on_talk_released()
    assert wait_until(lambda: len(saved_records()) == 1, app)

    record = saved_records()[0]
    assert (record["input"], record["kind"]) == ("voice", "stop")
    assert record["stt_sec"] is not None
    assert record["stopped_sec"] <= record["reply_audio_sec"]


def test_failed_turns_are_recorded_as_not_ok(window):  # noqa: F811
    win, _, app = window
    with (
        patch("src.agents.graph.create_agent_graph"),
        patch.object(win, "_invoke_agent_with_recovery", AsyncMock(side_effect=TimeoutError)),
    ):
        send_chat(win, "こんにちは")
        assert wait_until(lambda: len(saved_records()) == 1, app)
    assert saved_records()[0]["ok"] is False
