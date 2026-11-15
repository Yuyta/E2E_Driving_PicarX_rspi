"""Pi 上の自律走行を PC から始めて、見守り、パッドで引き継ぎ、止める"""

import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.e2e.gamepad import PadInput
from src.e2e.pi_autopilot import Notifier, PiAutopilotSession, format_status, response_message
from src.e2e.training.dataset import DrivingDataset
from src.models import DriveDirection


class FakeClient:
    def __init__(self, guard=True):
        self.calls = []
        self.guard = guard
        self.running = True

    def get_status_sync(self):
        self.calls.append("status")
        return SimpleNamespace(
            cliff=SimpleNamespace(guard_enabled=self.guard),
            sensors=SimpleNamespace(distance=SimpleNamespace(cm=80.0, far=False)),
        )

    def calibrate_floor_sync(self):
        self.calls.append("calibrate")
        return True

    def autopilot_start_sync(self, speed, intent, wait_for_help=False):
        self.calls.append(("start", speed, intent))
        self.wait_for_help = wait_for_help
        return {"run_id": "20261114_220000", "model_version": "v006", "run_dir": "/runs/x"}

    def autopilot_status_sync(self):
        self.calls.append("autopilot_status")
        if getattr(self, "status", None) is not None:
            return self.status
        if self.running:
            return {"running": True, "phase": "driving", "intent": "right"}
        return {"running": False, "phase": "stopped", "stopped_reason": "stuck 3 times"}

    def autopilot_intent_sync(self, intent):
        self.calls.append(("intent", intent))

    def autopilot_stop_sync(self, intervention):
        self.calls.append(("stop", intervention))
        return {"running": False}

    def steer_sync(self, angle):
        self.calls.append(("steer", angle))
        return True

    def move_sync(self, direction, speed, ttl_ms):
        self.calls.append(("move", direction, speed, ttl_ms))
        return True

    def emergency_stop_sync(self):
        self.calls.append("emergency")
        return True


class FakePad:
    def __init__(self, inputs):
        self.inputs = list(inputs)

    def poll(self):
        return self.inputs.pop(0) if self.inputs else PadInput()


class FakeStream:
    def __init__(self):
        ok, jpg = cv2.imencode(".jpg", np.full((240, 320, 3), 90, np.uint8))
        self.jpeg = jpg.tobytes()
        self.seq = 0

    def wait_for_frame(self, last_seq, timeout):
        self.seq += 1
        return SimpleNamespace(seq=self.seq, jpeg=self.jpeg)


def test_start_calibrates_the_floor_and_uses_the_pi_model():
    client = FakeClient()
    session = PiAutopilotSession(client, speed=50)
    info = session.start("left")
    assert info["model_version"] == "v006"
    assert client.calls == ["status", "calibrate", ("start", 50, "left")]
    assert PiAutopilotSession(FakeClient(guard=False)).start()["run_id"]


def test_step_reports_when_the_pi_ended_the_run():
    client = FakeClient()
    session = PiAutopilotSession(client)
    session.start()
    assert session.step(0.0) is None
    client.running = False
    assert session.step(0.1) is None  # 状態は 0.5 秒ごとに読む
    assert session.step(0.6) == "stuck 3 times"
    session.stop()
    assert client.calls[-1] == ("stop", True)


def test_gamepad_takeover_drives_by_hand_and_records_corrections(tmp_path: Path):
    client = FakeClient()
    pad = FakePad([PadInput(drive=1, steer=0.5), PadInput(drive=0, steer=0.2), PadInput(stop=True)])
    session = PiAutopilotSession(client, pad, tmp_path / "interventions", stream=FakeStream())
    session.start()
    for t in (0.0, 0.1, 0.2):
        session.step(t)
    moves = [c for c in client.calls if c[0] == "move"]
    assert moves[0] == ("move", DriveDirection.FORWARD, 50, 400)
    # 止まる時も、停止の指令ではなく速度 0 の指令（停止の指令は Pi の自律走行を終える）
    assert moves[1][2] == 0 and moves[2][2] == 0
    assert ("steer", 18) in client.calls
    assert session.takeover_count == 1

    dataset = DrivingDataset(session.corrections.session_dir)
    assert [s["steering"] for s in dataset.samples] == [0.5]  # 止まっているフレームは使わない
    assert dataset.samples[0]["intent"] == "any"  # 引き継ぐ前は状態を読んでいない


def test_intent_buttons_while_not_driving_and_emergency():
    client = FakeClient()
    pad = FakePad([PadInput(intent="left"), PadInput(intent="left"), PadInput(emergency=True)])
    session = PiAutopilotSession(client, pad)
    session.start()
    assert session.step(0.0) is None
    assert session.step(0.1) is None
    assert [c for c in client.calls if c[0] == "intent"] == [("intent", "left")]
    assert session.step(0.2).startswith("emergency")
    assert client.calls[-1] == "emergency"


class FakeNotifier:
    def __init__(self):
        self.events = []

    def help(self, reason):
        self.events.append(("help", reason))

    def caution(self, event):
        self.events.append(("caution", event))


def test_with_a_gamepad_the_pi_waits_for_help_and_the_pc_asks_for_it():
    client = FakeClient()
    notifier = FakeNotifier()
    session = PiAutopilotSession(client, FakePad([]), notifier=notifier)
    session.start()
    assert client.wait_for_help is True
    assert PiAutopilotSession(FakeClient()).wait_for_help is False  # 見守る人がいなければ終える

    def status(phase, safety, **extra):
        client.status = {
            "running": True,
            "phase": phase,
            "interventions": {"human": 0, "safety": safety},
            **extra,
        }

    status("obstacle", 1, last_event="obstacle_stop: obstacle at 13cm")
    session.step(0.0)
    assert notifier.events == [("caution", "obstacle_stop: obstacle at 13cm")]
    status("needs_help", 4, help_reason="stuck 3 times within 30s; needs help")
    session.step(1.0)
    assert notifier.events[-1] == ("help", "stuck 3 times within 30s; needs help")
    assert len(notifier.events) == 2  # 助けを求めた時は注意を重ねない
    session.step(2.0)
    assert len(notifier.events) == 2
    session.step(6.5)  # 5秒たっても待っていれば知らせ直す
    assert len(notifier.events) == 3
    status("driving", 4)  # 人が抜け出させて自律に戻った
    session.step(7.0)
    status("needs_help", 5, help_reason="cliff 2 times within 60s; needs help")
    session.step(8.0)
    assert notifier.events[-1][0] == "help"
    assert session.help_count == 2


def test_notifier_prints_sounds_and_rumbles():
    lines, sounds = [], []
    pad = SimpleNamespace(rumble=lambda strength, ms: sounds.append(("rumble", ms)))
    notifier = Notifier(pad, out=lines.append, beep=sounds.append)
    notifier.help("stuck 3 times")
    notifier.caution("obstacle_stop")
    time.sleep(0.05)
    assert "介入してください" in lines[0] and "stuck 3 times" in lines[0]
    assert "注意" in lines[1]
    assert ("rumble", 600) in sounds and Notifier.HELP_SOUND in sounds


def test_format_status():
    text = format_status(
        {
            "phase": "driving",
            "intent": "any",
            "steer_deg": -7,
            "speed": 48,
            "distance_cm": 123.4,
            "cycle_ms": 66.0,
            "inference_ms": 12.3,
            "interventions": {"human": 1, "safety": 2},
        }
    )
    assert "steer -7°" in text and "spd 48" in text and "human 1 safety 2" in text
    assert "--" in format_status({})


def test_response_message_shows_why_the_pi_refused():
    import httpx

    refused = httpx.Response(422, json={"status": "error", "message": "床センサー2の値が低い"})
    assert response_message(refused) == "床センサー2の値が低い"
    assert response_message(httpx.Response(500, text="boom")) == "boom"
