"""収集 GUI（`src.e2e.gui_collector`）: 保存するセンサー値、W やパッドを押している間の走行"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.e2e import gui_collector  # noqa: E402
from src.e2e.gamepad import PadInput  # noqa: E402
from src.e2e.gui_collector import sensor_fields  # noqa: E402
from src.models import BatteryStatus, DistanceStatus, FloorStatus, SensorsStatus  # noqa: E402

# --- 各フレームに保存するセンサー値 ---


def status(distance: DistanceStatus):
    sensors = SensorsStatus(
        distance=distance,
        floor=FloorStatus(values=[900, 910, 920]),
        battery=BatteryStatus(voltage=8.06, percent=78),
    )
    return type("Status", (), {"sensors": sensors})()


def test_sensor_fields_keep_distance_floor_and_battery():
    fields = sensor_fields(status(DistanceStatus(cm=42.34)))
    assert fields == {
        "distance_cm": 42.3,
        "distance_far": False,
        "floor": [900, 910, 920],
        "battery_v": 8.06,
    }


def test_far_distance_is_saved_as_none_with_the_far_mark():
    fields = sensor_fields(status(DistanceStatus(cm=343.0, far=True)))
    assert fields["distance_cm"] is None
    assert fields["distance_far"] is True


def test_unreadable_status_saves_none():
    assert set(sensor_fields(None).values()) == {None}


# --- W を押している間の走行（キーの自動の連打に頼らない） ---


class FakeStream:
    def start(self):
        return self

    def stop(self):
        pass

    def wait_for_frame(self, after_seq=0, timeout=0.5):
        time.sleep(0.01)
        return None


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def open_camera_stream(self):
        return FakeStream()

    def move_sync(self, direction, speed, ttl_ms):
        self.calls.append(("move", speed, direction.value))
        return True

    def stop_sync(self):
        self.calls.append(("stop",))
        return True

    def steer_sync(self, angle):
        self.calls.append(("steer", angle))
        return True

    def emergency_stop_sync(self):
        self.calls.append(("emergency",))
        return True


class FakePad:
    """テストから状態を決めるパッド。state が None ならつながっていない"""

    def __init__(self):
        self.available = True
        self.name = "Fake Pad"
        self.state = None

    def poll(self):
        return self.state

    def close(self):
        pass


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(gui_collector, "PicarxClient", FakeClient)
    monkeypatch.setattr(gui_collector, "Gamepad", FakePad)
    # 食い違いを記憶 DB に書かない
    monkeypatch.setattr(gui_collector, "record_motion_event", lambda flag, update: (None, flag))
    QApplication.instance() or QApplication([])
    win = gui_collector.E2ECollectorWindow()
    yield win
    # カメラのスレッドを止めてから破棄する。残すと後の GUI のテストでプロセスが落ちる
    win.drive_timer.stop()
    win.steer_timer.stop()
    win.pad_timer.stop()
    win.camera_worker.stop()
    win.deleteLater()
    QApplication.processEvents()


def moves(client):
    return sum(1 for call in client.calls if call[0] == "move")


def stopped_after_last_move(client):
    """最後の前進の指令の後に停止を送った。舵は止まった後も中央へ戻るので、最後とは限らない"""
    kinds = [call[0] for call in client.calls]
    last_move = max(i for i, kind in enumerate(kinds) if kind == "move")
    return "stop" in kinds[last_move + 1 :]


def steers(client):
    return [call[1] for call in client.calls if call[0] == "steer"]


def test_holding_w_keeps_driving_while_steering(window):
    QTest.keyPress(window, Qt.Key_W)
    QTest.keyPress(window, Qt.Key_A)  # 舵を切っても前進は続く
    QTest.qWait(450)
    assert moves(window.client) >= 4  # 0.1 秒ごとに送り直す
    assert steers(window.client) and steers(window.client)[-1] < 0
    QTest.keyRelease(window, Qt.Key_A)

    QTest.keyRelease(window, Qt.Key_W)
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
    assert stopped_after_last_move(window.client)


def test_a_motion_mismatch_stops_even_while_w_is_held(window):
    QTest.keyPress(window, Qt.Key_W)
    QTest.qWait(150)
    update = type("Update", (), {"check": type("Check", (), {"details": {}})()})()
    window._on_motion_mismatch("no_motion", update)

    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
    assert stopped_after_last_move(window.client)


def test_steering_ramps_while_held_and_returns_to_center(window):
    QTest.keyPress(window, Qt.Key_D)
    QTest.qWait(300)
    mid = window.current_steering
    assert 5 <= mid < gui_collector.MAX_STEER_DEG  # 一気に最大角にならない
    QTest.qWait(600)
    assert window.current_steering == gui_collector.MAX_STEER_DEG

    QTest.keyRelease(window, Qt.Key_D)
    QTest.qWait(250)
    assert 0 < window.current_steering < gui_collector.MAX_STEER_DEG  # なめらかに戻る
    QTest.qWait(700)
    assert window.current_steering == 0
    # 途中の角度も送っている（学習に中間の舵角が残る）
    assert len(set(steers(window.client))) >= 8


# --- プロコン（ゲームパッド）での前進・後退・舵 ---


def directions(client):
    return [call[2] for call in client.calls if call[0] == "move"]


def test_pad_drives_backward_and_stops_when_released(window):
    window.gamepad.state = PadInput(drive=-1)
    QTest.qWait(250)
    assert directions(window.client) and set(directions(window.client)) == {"backward"}
    assert window.command_state.at(time.time()).speed == -gui_collector.REVERSE_SPEED

    window.gamepad.state = PadInput()
    QTest.qWait(100)
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
    assert stopped_after_last_move(window.client)


def test_pad_forward_and_stick_steer_in_proportion(window):
    window.gamepad.state = PadInput(drive=1, steer=0.5)
    QTest.qWait(800)
    assert directions(window.client)[-1] == "forward"
    # 半分倒すと最大角の半分で止まる
    assert window.current_steering == round(gui_collector.MAX_STEER_DEG / 2)


def test_unplugging_the_pad_stops_the_car(window):
    window.gamepad.state = PadInput(drive=1)
    QTest.qWait(150)
    window.gamepad.state = None
    QTest.qWait(100)
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
    assert stopped_after_last_move(window.client)


def test_after_a_mismatch_the_pad_must_be_released_before_driving_again(window):
    window.gamepad.state = PadInput(drive=1)
    QTest.qWait(150)
    update = type("Update", (), {"check": type("Check", (), {"details": {}})()})()
    window._on_motion_mismatch("no_motion", update)
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count  # 押したままでは走り出さない

    window.gamepad.state = PadInput()
    QTest.qWait(60)
    window.gamepad.state = PadInput(drive=-1)  # 離してから後退で抜け出せる
    QTest.qWait(150)
    assert directions(window.client)[-1] == "backward"


def test_plus_toggles_recording_once_per_press(window):
    window.gamepad.state = PadInput(record=True)
    QTest.qWait(150)
    assert window.is_recording
    window.gamepad.state = PadInput()
    QTest.qWait(60)
    assert window.is_recording


def test_x_stops_while_a_is_held_until_released(window):
    window.gamepad.state = PadInput(drive=1)
    QTest.qWait(150)
    window.gamepad.state = PadInput(drive=1, stop=True)
    QTest.qWait(60)
    window.gamepad.state = PadInput(drive=1)
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
    assert stopped_after_last_move(window.client)


def test_home_is_the_emergency_stop(window):
    window.gamepad.state = PadInput(drive=1)
    QTest.qWait(150)
    window.gamepad.state = PadInput(drive=1, emergency=True)
    QTest.qWait(60)
    assert ("emergency",) in window.client.calls
    count = moves(window.client)
    QTest.qWait(250)
    assert moves(window.client) == count
