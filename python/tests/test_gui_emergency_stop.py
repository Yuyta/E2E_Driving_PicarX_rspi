import asyncio
import os
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui.app import PicarxMainWindow  # noqa: E402
from src.gui.async_runner import AsyncRunner  # noqa: E402


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
