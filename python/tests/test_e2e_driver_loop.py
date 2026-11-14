"""走行ループを偽の Pi で動かし、Pi の判定どおりに行動することを確かめる。"""

import json
from types import SimpleNamespace

import httpx
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from src.e2e import autonomous_driver  # noqa: E402
from src.e2e.autonomous_driver import MOVE_TTL_MS, RemoteAutonomousDriver  # noqa: E402
from src.e2e.gamepad import PadInput  # noqa: E402
from src.models import DriveDirection, RobotStatus  # noqa: E402
from tests.test_m1_body import status_payload  # noqa: E402


def encoded_frame(value: int = 90) -> bytes:
    ok, data = cv2.imencode(".jpg", np.full((120, 160, 3), value, dtype=np.uint8))
    assert ok
    return data.tobytes()


class FakeStream:
    """指定した枚数を返した後、Ctrl+C を模して走行ループを抜ける"""

    last_error = None

    def __init__(self, frames: int):
        self.frames = frames
        self.seq = 0
        self.stopped = False

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def wait_for_frame(self, after_seq=0, timeout=0.5):
        if self.seq >= self.frames:
            raise KeyboardInterrupt
        self.seq += 1
        return SimpleNamespace(jpeg=encoded_frame(), seq=self.seq, received_at=0.0)


class FakePi:
    """`statuses` を順に返す。最後の要素は以後も返し続ける。"""

    base_url = "http://fake-pi"

    def __init__(self, statuses, frames, calibration_error=None, accept_moves=True):
        self.statuses = [RobotStatus.model_validate(s) for s in statuses]
        self.stream = FakeStream(frames)
        self.calibration_error = calibration_error
        self.accept_moves = accept_moves
        self.moves = []
        self.stream_opened = False

    def get_status_sync(self):
        status = self.statuses[0]
        if len(self.statuses) > 1:
            self.statuses.pop(0)
        return status

    def calibrate_floor_sync(self):
        if self.calibration_error:
            raise self.calibration_error
        return True

    def open_camera_stream(self):
        self.stream_opened = True
        return self.stream

    def move_sync(self, direction, speed=30, ttl_ms=500):
        self.moves.append((direction, speed, ttl_ms))
        return self.accept_moves

    def steer_sync(self, angle):
        return True

    def stop_sync(self):
        return True


class FakeSession:
    def __init__(self, *_args, **_kwargs):
        pass

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="steering")]

    def run(self, _outputs, _inputs):
        return [np.array([[0.0]], dtype=np.float32)]


@pytest.fixture
def make_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomous_driver.ort, "InferenceSession", FakeSession)
    monkeypatch.setattr(autonomous_driver.time, "sleep", lambda _sec: None)

    def build(pi: FakePi) -> RemoteAutonomousDriver:
        driver = RemoteAutonomousDriver(
            model_path="fake.onnx",
            enable_mapping=False,
            enable_stuck_detection=False,
            log_dir=str(tmp_path / "disengagements"),
        )
        driver.client = pi
        return driver

    return build


def directions(pi: FakePi):
    return [move[0] for move in pi.moves]


def test_clear_path_sends_forward_commands_with_ttl(make_driver):
    pi = FakePi([status_payload()], frames=3)
    make_driver(pi).run()

    assert directions(pi) == [DriveDirection.FORWARD] * 3
    assert {move[2] for move in pi.moves} == {MOVE_TTL_MS}
    assert pi.stream.stopped


def test_accepted_moves_become_the_escape_path(make_driver):
    pi = FakePi([status_payload()], frames=2)
    driver = make_driver(pi)
    driver.run()
    assert len(driver.trajectory_tracker.history) == 2


def test_refused_moves_are_not_recorded_as_travel(make_driver):
    pi = FakePi([status_payload()], frames=2, accept_moves=False)
    driver = make_driver(pi)
    driver.run()
    assert len(driver.trajectory_tracker.history) == 0


@pytest.mark.parametrize(
    "reason", ["distance_unknown", "floor_unknown", "cliff_uncalibrated", "emergency_stop"]
)
def test_unknown_or_latched_state_never_commands_motion(make_driver, reason):
    pi = FakePi([status_payload(forward=reason, cm=None)], frames=3)
    make_driver(pi).run()
    assert pi.moves == []


def test_cliff_without_history_refuses_to_reverse_blindly(make_driver):
    blocked = status_payload(forward="cliff", cliff=[False, True, False])
    pi = FakePi([blocked], frames=3)
    make_driver(pi).run()
    assert pi.moves == []


def test_cliff_with_history_reverses_only_what_was_recorded(make_driver):
    clear = status_payload()
    blocked = status_payload(forward="cliff", cliff=[True, False, False])
    # 準備の2回は安全、その後は段差のまま
    pi = FakePi([clear, clear, blocked], frames=1)
    driver = make_driver(pi)
    driver.trajectory_tracker.record(speed=30, steer_angle=10, duration_ms=200, timestamp=0.0)
    driver.run()

    backward = [m for m in pi.moves if m[0] == DriveDirection.BACKWARD]
    assert sum(m[2] for m in backward) == 200
    # 左が段差なので、後退後は右へ旋回して離れる
    assert DriveDirection.FORWARD in directions(pi)


def test_far_reading_keeps_driving(make_driver):
    far = status_payload(cm=343.0)
    far["sensors"]["distance"]["far"] = True
    pi = FakePi([far], frames=2)
    make_driver(pi).run()
    assert directions(pi) == [DriveDirection.FORWARD] * 2


def test_obstacle_waits_before_escaping(make_driver):
    pi = FakePi([status_payload(forward="obstacle", cm=10.0)], frames=3)
    make_driver(pi).run()
    assert pi.moves == []


def test_start_is_refused_when_calibration_fails(make_driver):
    request = httpx.Request("POST", "http://fake-pi/api/safety/calibrate")
    error = httpx.HTTPStatusError(
        "422", request=request, response=httpx.Response(422, text="too dark", request=request)
    )
    pi = FakePi([status_payload()], frames=3, calibration_error=error)
    make_driver(pi).run()
    assert pi.moves == []
    assert not pi.stream_opened


# --- 障害物を避ける判断（意図と、前方が塞がっている確率） ---


class IntentSession:
    """意図を入力に持ち、舵角と塞がっている確率を出す偽のモデル。渡された意図を記録する"""

    blocked = 0.0

    def __init__(self, *_args, **_kwargs):
        self.intents = []

    def get_inputs(self):
        return [SimpleNamespace(name="input"), SimpleNamespace(name="intent")]

    def get_outputs(self):
        return [SimpleNamespace(name="steering"), SimpleNamespace(name="blocked")]

    def run(self, _outputs, feed):
        intent = ("any", "left", "right")[int(np.argmax(feed["intent"][0]))]
        self.intents.append(intent)
        steering = {"any": 0.0, "left": -0.8, "right": 0.8}[intent]
        return [
            np.array([[steering]], dtype=np.float32),
            np.array([[self.blocked]], dtype=np.float32),
        ]


# 走り出す前の準備（_prepare_body）が読む2回分の状態
PREPARE = [status_payload(cm=150.0), status_payload(cm=150.0)]


def avoid_driver(monkeypatch, tmp_path, pi, blocked=0.0, trained=True, **kwargs):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "metadata.json").write_text(
        json.dumps({"blocked_output": {"trained": trained}}), encoding="utf-8"
    )
    session = IntentSession()
    session.blocked = blocked
    monkeypatch.setattr(autonomous_driver.ort, "InferenceSession", lambda *_a, **_k: session)
    monkeypatch.setattr(autonomous_driver.time, "sleep", lambda _sec: None)
    driver = RemoteAutonomousDriver(
        model_path=str(model_dir / "model.onnx"),
        enable_mapping=False,
        enable_stuck_detection=False,
        enable_motion_check=False,
        log_dir=str(tmp_path / "disengagements"),
        **kwargs,
    )
    driver.client = pi
    return driver, session


def test_near_obstacle_turns_with_the_right_intent(monkeypatch, tmp_path):
    pi = FakePi(PREPARE + [status_payload(cm=150.0), status_payload(cm=45.0)], frames=3)
    driver, session = avoid_driver(monkeypatch, tmp_path, pi)
    assert driver.uses_blocked_output
    driver.run()
    # 1枚目は遠いので any、近づいたら right で推論し直す
    assert session.intents[0] == "any"
    assert session.intents[-1] == "right"
    assert driver.last_decision.state == "avoid"


def test_camera_blocked_slows_down_and_turns(monkeypatch, tmp_path):
    pi = FakePi([status_payload(cm=150.0)], frames=3)
    driver, session = avoid_driver(monkeypatch, tmp_path, pi, blocked=0.95)
    driver.run()
    assert {move[1] for move in pi.moves} == {driver.min_speed}  # 最低の速度まで落とす
    assert driver.last_decision.state == "avoid"
    assert driver.last_decision.intent == "right"


def test_untrained_blocked_output_is_ignored(monkeypatch, tmp_path):
    pi = FakePi([status_payload(cm=150.0)], frames=3)
    driver, session = avoid_driver(monkeypatch, tmp_path, pi, blocked=0.95, trained=False)
    driver.run()
    assert not driver.uses_blocked_output
    assert driver.last_decision.state == "cruise"
    assert set(session.intents) == {"any"}


def test_obstacle_stop_switches_the_turn_side(monkeypatch, tmp_path):
    statuses = PREPARE + [status_payload(cm=45.0)]
    pi = FakePi(statuses + [status_payload(forward="obstacle", cm=14.0)], frames=3)
    driver, _ = avoid_driver(monkeypatch, tmp_path, pi)
    driver.run()
    assert driver.last_decision.intent == "left"
    assert driver.avoidance.failures == 1


def test_avoidance_can_be_turned_off(monkeypatch, tmp_path):
    pi = FakePi([status_payload(cm=45.0)], frames=2)
    driver, session = avoid_driver(monkeypatch, tmp_path, pi, enable_avoidance=False, intent="left")
    driver.run()
    assert driver.last_decision is None
    assert set(session.intents) == {"left"}


# --- 介入の記録とパッドでの引き継ぎ ---


class FakeGamepad:
    """周期ごとに inputs を順に返す。尽きたら何も操作していない状態"""

    def __init__(self, inputs):
        self.inputs = list(inputs)

    def poll(self):
        return self.inputs.pop(0) if self.inputs else PadInput()


def read_run(tmp_path):
    runs = list((tmp_path / "runs").glob("*/run.json"))
    assert len(runs) == 1
    return json.loads(runs[0].read_text(encoding="utf-8"))


def test_run_log_records_the_manual_stop(make_driver, tmp_path):
    pi = FakePi([status_payload(cm=150.0)], frames=2)
    make_driver(pi).run()
    run = read_run(tmp_path)
    assert run["ended"] is True
    assert run["interventions"] == {"human": 1, "safety": 0}  # Ctrl+C
    assert run["autonomous_s"] > 0


def test_gamepad_takeover_drives_by_hand_and_records_corrections(monkeypatch, tmp_path):
    pi = FakePi(PREPARE + [status_payload(cm=150.0)], frames=4)
    pad = FakeGamepad([PadInput(), PadInput(drive=1, steer=0.5), PadInput(drive=-1, steer=-1.0)])
    driver, session = avoid_driver(monkeypatch, tmp_path, pi, gamepad=pad)
    driver.run()

    # 1周目は自律、2周目と3周目は人（前進、後退）、4周目は入力なしでも 1.5 秒たつまで人
    assert [m[0] for m in pi.moves] == [
        DriveDirection.FORWARD,
        DriveDirection.FORWARD,
        DriveDirection.BACKWARD,
    ]
    assert len(session.intents) == 1  # 人が運転している間はモデルを使わない
    run = read_run(tmp_path)
    assert run["interventions"]["human"] == 2  # 引き継ぎと Ctrl+C
    corrections = list((tmp_path / "interventions").glob("*/labels.jsonl"))
    records = [json.loads(line) for line in corrections[0].read_text().splitlines()]
    assert [r["steering"] for r in records] == [0.5, -1.0, 0.0]
    assert [r["command"]["speed"] for r in records] == [50, -50, 0]
    assert {r["takeover_id"] for r in records} == {1}


def test_gamepad_home_stops_the_run(monkeypatch, tmp_path):
    pi = FakePi(PREPARE + [status_payload(cm=150.0)], frames=5)
    pad = FakeGamepad([PadInput(), PadInput(emergency=True)])
    driver, _ = avoid_driver(monkeypatch, tmp_path, pi, gamepad=pad)
    driver.run()
    assert len(pi.moves) == 1
    assert read_run(tmp_path)["interventions"] == {"human": 1, "safety": 0}
