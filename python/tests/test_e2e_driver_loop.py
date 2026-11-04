"""走行ループを偽の Pi で動かし、Pi の判定どおりに行動することを確かめる。"""

from types import SimpleNamespace

import httpx
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")

from src.e2e import autonomous_driver  # noqa: E402
from src.e2e.autonomous_driver import MOVE_TTL_MS, RemoteAutonomousDriver  # noqa: E402
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
