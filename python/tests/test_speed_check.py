"""車輪が回る最低の速度の計測（`src.e2e.speed_check`）"""

from types import SimpleNamespace

import pytest

from src.e2e import speed_check


def test_approach_rate_is_positive_when_closing_in():
    points = [(0.5, 100.0), (0.7, 98.0), (0.9, 96.0), (1.1, 94.0)]
    assert speed_check.approach_rate(points) == pytest.approx(10.0)


def test_approach_rate_needs_enough_points():
    assert speed_check.approach_rate([(0.5, 100.0), (0.7, 98.0)]) is None


def trial(mode, speed, moving):
    return {"mode": mode, "speed": speed, "moving": moving}


def test_lowest_moving_speed():
    trials = [trial("start", 30, False), trial("start", 40, True), trial("start", 45, True)]
    assert speed_check.lowest_moving(trials, "start") == 40
    assert speed_check.lowest_moving(trials, "rolling") is None


def test_lowest_moving_ignores_rolling_trials_that_never_got_going():
    cold = {**trial("rolling", 30, True), "warmed_up": False}
    warm = {**trial("rolling", 35, True), "warmed_up": True}
    assert speed_check.lowest_moving([cold, warm], "rolling") == 35


def test_lowest_moving_is_unknown_when_a_faster_speed_did_not_move():
    trials = [trial("start", 35, True), trial("start", 40, False), trial("start", 45, True)]
    assert speed_check.lowest_moving(trials, "start") is None


class WallClient:
    """前進の指令を受けるたびに、その速度に応じて壁へ近づく"""

    def __init__(self, distance=120.0, moves_from=40):
        self.distance = distance
        self.moves_from = moves_from
        self.stops = 0

    def get_status_sync(self):
        return SimpleNamespace(
            sensors=SimpleNamespace(
                distance=SimpleNamespace(cm=self.distance, far=False),
                battery=SimpleNamespace(voltage=8.0, percent=80),
                floor=SimpleNamespace(values=[900, 900, 900], cliff=[False, False, False]),
            ),
            blocked=SimpleNamespace(forward=None),
        )

    def move_sync(self, _direction, speed, ttl_ms):
        if speed >= self.moves_from:
            self.distance -= 1.0
        return True

    def stop_sync(self):
        self.stops += 1
        return True


@pytest.fixture
def fast_clock(monkeypatch):
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds

    monkeypatch.setattr(speed_check.time, "sleep", sleep)
    monkeypatch.setattr(speed_check.time, "monotonic", lambda: now[0])


def test_run_finds_the_lowest_speed_and_always_stops(fast_clock):
    client = WallClient(distance=500.0, moves_from=40)
    result = speed_check.run(client, [30, 40], ["start", "rolling"])
    assert result["aborted"] is None
    assert result["lowest_start"] == 40
    assert result["lowest_rolling"] == 40
    assert client.stops >= 4


def test_run_stops_when_the_wall_is_too_close(fast_clock):
    client = WallClient(distance=40.0)
    result = speed_check.run(client, [30], ["start"])
    assert result["trials"] == []
    assert "置き直して" in result["aborted"]


def test_noisy_distance_raises_the_threshold(fast_clock):
    client = WallClient(distance=500.0, moves_from=40)
    readings = iter([100.0, 102.0] * 20)
    original = client.get_status_sync

    def noisy():
        status = original()
        status.sensors.distance.cm = next(readings, client.distance)
        return status

    client.get_status_sync = noisy
    result = speed_check.run(client, [40], ["start"])
    assert result["noise"]["spread_cm"] == 2.0
    assert result["moved_cm_threshold"] == 6.0
