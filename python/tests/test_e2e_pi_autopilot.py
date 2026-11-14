"""Pi 上の自律走行を PC から始めて、見守り、パッドで引き継ぎ、止める"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from src.e2e.gamepad import PadInput
from src.e2e.pi_autopilot import PiAutopilotSession, format_status
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

    def autopilot_start_sync(self, speed, intent):
        self.calls.append(("start", speed, intent))
        return {"run_id": "20261114_220000", "model_version": "v006", "run_dir": "/runs/x"}

    def autopilot_status_sync(self):
        self.calls.append("autopilot_status")
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
