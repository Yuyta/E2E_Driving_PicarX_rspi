import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.config import ModelPreset  # noqa: E402
from src.gui.app import PicarxMainWindow  # noqa: E402
from src.gui.async_runner import AsyncRunner  # noqa: E402
from src.memory.store import MemoryStore  # noqa: E402


def test_async_runner_submit_does_not_block_caller():
    runner = AsyncRunner()
    try:
        started = time.monotonic()
        future = runner.submit(asyncio.sleep(0.3, result="done"))
        assert time.monotonic() - started < 0.1
        assert future.result(timeout=2) == "done"
    finally:
        runner.close()


def test_async_runner_close_cancels_pending_work():
    runner = AsyncRunner()
    future = runner.submit(asyncio.sleep(30))
    time.sleep(0.05)
    runner.close()
    assert future.cancelled()


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    client = MagicMock()
    client.get_ultrasonic = AsyncMock(return_value=100.0)
    client.get_camera_snapshot = AsyncMock(side_effect=RuntimeError("no camera"))
    with patch("src.gui.app.PicarxClient", return_value=client):
        win = PicarxMainWindow()
    win.poll_timer.stop()
    win.camera_timer.stop()
    yield win, client, app
    win.close()
    # Linux では、ウィジェットを Python の終了処理に任せるとクラッシュする
    win.deleteLater()
    app.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def wait_until(predicate, app, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_emergency_stop_reaches_robot_while_agent_is_busy(window):
    win, client, app = window
    release = threading.Event()

    async def slow_agent_call():
        await asyncio.get_running_loop().run_in_executor(None, release.wait, 5)

    called = threading.Event()
    client.emergency_stop_sync.side_effect = lambda: called.set() or True

    started = time.monotonic()
    win._async_call(slow_agent_call())
    win._on_emergency_stop()
    assert time.monotonic() - started < 0.5, "UI thread was blocked"

    assert called.wait(timeout=2)
    assert wait_until(lambda: "EMERGENCY" in win.lbl_safety.text(), app)
    release.set()


def test_emergency_stop_runs_even_if_event_loop_is_blocked(window):
    win, client, _ = window
    release = threading.Event()

    async def blocks_the_loop():
        release.wait(5)

    called = threading.Event()
    client.emergency_stop_sync.side_effect = lambda: called.set() or True

    win._async_call(blocks_the_loop())
    time.sleep(0.05)
    win._on_emergency_stop()
    try:
        assert called.wait(timeout=2)
    finally:
        release.set()


def test_chat_turn_is_saved_to_memory(window, tmp_path):
    win, _, _ = window
    store = MemoryStore(tmp_path / "memory.db")
    with patch("src.gui.app.get_memory_store", return_value=store):
        win._remember_turn("電池どう？", "7.60Vです")
    saved = MemoryStore(tmp_path / "memory.db").previous_session_messages()
    assert [(m.role, m.content) for m in saved] == [
        ("user", "電池どう？"),
        ("assistant", "7.60Vです"),
    ]


def test_memory_failure_does_not_stop_the_chat(window):
    win, _, app = window
    with patch("src.gui.app.get_memory_store", side_effect=OSError("disk full")):
        win._remember_turn("こんにちは", "こんにちは")
    assert wait_until(lambda: "Memory Error" in win.chat_history.toPlainText(), app)


def fake_graph(*results):
    graph = MagicMock()
    graph.ainvoke = AsyncMock(side_effect=list(results))
    return graph


def run_recovery(win, first_graph, created_graphs):
    win._agent_graph = first_graph
    win.current_model = ModelPreset.GEMINI_FLASH
    created = []

    def create(preset):
        created.append(preset)
        return created_graphs[preset]

    with (
        patch.object(win, "_create_agent_graph", side_effect=create),
        patch("src.gui.app.asyncio.sleep", AsyncMock()),
    ):
        return asyncio.run(win._invoke_agent_with_recovery()), created


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("429 RESOURCE_EXHAUSTED"),
        OSError("network unreachable"),
        RuntimeError("ConnectError: getaddrinfo failed"),
        asyncio.TimeoutError(),
        RuntimeError("API key not valid"),
    ],
)
def test_quota_or_network_failure_goes_straight_to_local_llm(window, error):
    win, _, _ = window
    answer = {"messages": []}
    local = fake_graph(answer)

    result, created = run_recovery(
        win, fake_graph(error), {ModelPreset.OLLAMA_LOCAL: local}
    )

    assert result is answer
    assert created == [ModelPreset.OLLAMA_LOCAL]


def test_busy_gemini_retries_then_flash_lite_then_local_llm(window):
    win, _, _ = window
    busy = RuntimeError("503 UNAVAILABLE")
    answer = {"messages": []}
    first = fake_graph(busy, busy)
    lite = fake_graph(busy)
    local = fake_graph(answer)

    result, created = run_recovery(
        win,
        first,
        {ModelPreset.GEMINI_FLASH_LITE: lite, ModelPreset.OLLAMA_LOCAL: local},
    )

    assert result is answer
    assert first.ainvoke.await_count == 2
    assert created == [ModelPreset.GEMINI_FLASH_LITE, ModelPreset.OLLAMA_LOCAL]


def test_busy_gemini_recovering_on_retry_stays_on_gemini(window):
    win, _, _ = window
    answer = {"messages": []}
    result, created = run_recovery(
        win, fake_graph(RuntimeError("503 UNAVAILABLE"), answer), {}
    )
    assert result is answer
    assert created == []


def test_programming_errors_are_not_hidden_by_the_fallback(window):
    win, _, _ = window
    with pytest.raises(ValueError):
        run_recovery(win, fake_graph(ValueError("bad state")), {})


def test_local_llm_failure_is_reported_not_retried(window):
    win, _, _ = window
    win.current_model = ModelPreset.OLLAMA_LOCAL
    win._agent_graph = fake_graph(ConnectionError("ollama is not running"))
    with pytest.raises(ConnectionError):
        asyncio.run(win._invoke_agent_with_recovery())


def send_chat(win, text):
    win.chat_input.setText(text)
    win._on_send_chat()


def test_stop_word_stops_at_once_without_the_llm(window, tmp_path):
    win, client, app = window
    client.stop_sync.return_value = True
    store = MemoryStore(tmp_path / "memory.db")
    with (
        patch.object(win, "_start_agent_turn") as agent,
        patch("src.gui.app.get_memory_store", return_value=store),
    ):
        send_chat(win, "止まって！")
        assert wait_until(lambda: "停止しました" in win.chat_history.toPlainText(), app)
    client.stop_sync.assert_called_once()
    agent.assert_not_called()


def test_failed_fast_stop_falls_back_to_the_llm(window):
    win, client, app = window
    client.stop_sync.side_effect = RuntimeError("unreachable")
    with patch.object(win, "_start_agent_turn") as agent:
        send_chat(win, "ストップ")
        assert wait_until(lambda: agent.called, app)
    agent.assert_called_once_with("ストップ")
    assert wait_until(lambda: "停止に失敗" in win.chat_history.toPlainText(), app)


def test_refused_fast_stop_falls_back_to_the_llm(window):
    win, client, app = window
    client.stop_sync.return_value = False
    with patch.object(win, "_start_agent_turn") as agent:
        send_chat(win, "止まれ")
        assert wait_until(lambda: agent.called, app)


def test_stop_with_more_instructions_stops_and_continues_with_the_llm(window):
    win, client, app = window
    client.stop_sync.return_value = True
    text = "そこで止まって、それから右を見て"
    with patch.object(win, "_start_agent_turn") as agent:
        send_chat(win, text)
        assert wait_until(lambda: agent.called, app)
    client.stop_sync.assert_called_once()
    agent.assert_called_once_with(text)


def test_other_words_go_to_the_llm_without_stopping(window):
    win, client, _ = window
    with patch.object(win, "_start_agent_turn") as agent:
        send_chat(win, "危ないからやめて")
    client.stop_sync.assert_not_called()
    agent.assert_called_once_with("危ないからやめて")


def test_failed_emergency_stop_is_reported(window):
    win, client, app = window
    client.emergency_stop_sync.side_effect = RuntimeError("unreachable")

    win._on_emergency_stop()

    assert wait_until(lambda: "緊急停止に失敗" in win.chat_history.toPlainText(), app)
    assert "EMERGENCY" not in win.lbl_safety.text()


def test_emergency_stop_status_survives_telemetry_poll_until_reset(window):
    win, client, app = window
    client.emergency_stop_sync.return_value = True
    client.reset_emergency_stop = AsyncMock(return_value=True)

    win._on_emergency_stop()
    assert wait_until(lambda: "EMERGENCY" in win.lbl_safety.text(), app)

    client.get_ultrasonic.reset_mock()
    win._poll_telemetry()
    assert wait_until(lambda: client.get_ultrasonic.await_count == 1, app)
    assert wait_until(lambda: win.lbl_distance.text() == "100.0 cm", app)
    assert "EMERGENCY" in win.lbl_safety.text()

    win._on_emergency_reset()
    assert wait_until(lambda: "EMERGENCY" not in win.lbl_safety.text(), app)
    client.reset_emergency_stop.assert_awaited_once()
