"""実機の確認スクリプト（M1 の Wi-Fi 切断、M3 の動きの計測）のうち、実機なしで確かめられる部分"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from src.e2e import m1_check, m3_check
from src.e2e.motion.monitor import DEFAULT_CONFIG, MotionMonitor, load_motion_config
from src.models import EventsResponse, SafetyEvent

# --- 動きの設定ファイル ---


def test_repository_motion_config_loads():
    tracker, config = load_motion_config(DEFAULT_CONFIG)
    assert tracker.interval_s == 0.3
    assert config.sustain_s == 1.0
    assert config.speed_cm_s_per_unit is None


def test_motion_config_overrides_and_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "motion.toml"
    path.write_text(
        "[tracker]\nhfov_deg = 60.0\n[checker]\nwheelbase_cm = 9.5\nspeed_cm_s_per_unit = 0.4\n",
        encoding="utf-8",
    )
    tracker, config = load_motion_config(path)
    assert tracker.deg_per_px == pytest.approx(60.0 / 160)
    assert config.wheelbase_cm == 9.5

    path.write_text("[checker]\nsustain = 1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_motion_config(path)


def test_missing_motion_config_uses_defaults(tmp_path: Path):
    monitor = MotionMonitor.from_config(tmp_path / "none.toml")
    assert monitor.checker.config.sustain_s == 1.0


# --- M3 の計測のまとめと判定 ---


def record(t, flow=20.0, yaw=0.0, flags=(), sustained=None, approach=None, valid=True):
    check = {"flags": list(flags)}
    if sustained:
        check["sustained"] = sustained
    return {
        "t": t,
        "motion": {
            "image_valid": valid,
            "edge_flow_px_s": flow,
            "yaw_rate_dps": yaw,
            "approach_cm_s": approach,
        },
        "motion_check": check,
    }


def test_summary_integrates_the_turned_angle():
    records = [record(i * 0.1, yaw=30.0) for i in range(11)]  # 1 秒間、毎秒 30°
    summary = m3_check.summarize(records, "right", 50, 30)

    assert summary["turned_deg"] == pytest.approx(30.0)
    assert summary["rate_hz"] == pytest.approx(10.0)
    assert m3_check.judge(summary) == ["OK: 印なし"]


def test_judge_flags_a_turn_in_the_wrong_direction():
    summary = m3_check.summarize([record(i * 0.1, yaw=10.0) for i in range(5)], "left", 50, -30)
    assert any("左へ舵を切ったのに" in line for line in m3_check.judge(summary))


def test_judge_lifted_needs_a_timely_no_motion():
    records = [record(0.0, flow=0.5, flags=["no_motion"])] + [
        record(1.6, flow=0.5, flags=["no_motion"], sustained="no_motion")
    ]
    summary = m3_check.summarize(records, "lifted", 50, 0)
    assert summary["first_sustained"] == {"flag": "no_motion", "t": 1.6}
    assert m3_check.judge(summary)[0].startswith("OK")

    late = [record(0.0, flow=0.5), record(3.0, flow=0.5, sustained="no_motion")]
    assert m3_check.judge(m3_check.summarize(late, "lifted", 50, 0))[0].startswith("NG")


def test_judge_still_scene_with_flags_is_ng():
    summary = m3_check.summarize(
        [record(0.0, flags=["uncommanded_motion"]), record(0.1)], "walk", 0, 0
    )
    assert m3_check.judge(summary)[0].startswith("NG")


def save(tmp_path: Path, name: str, records, scenario, speed=50, steer=0, hfov=53.5) -> str:
    summary = m3_check.summarize(records, scenario, speed, steer)
    summary["config"] = {"hfov_deg": hfov}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"summary": summary, "records": records}), encoding="utf-8")
    return str(path)


def test_calibrate_suggests_thresholds_and_hfov(tmp_path: Path, capsys):
    still = save(tmp_path, "still", [record(i * 0.1, flow=1.0) for i in range(10)], "still", 0)
    forward = save(tmp_path, "fwd", [record(i * 0.1, flow=16.0) for i in range(10)], "forward")
    push = save(tmp_path, "push", [record(i * 0.1, flow=25.0) for i in range(10)], "push", 0)
    turn = save(tmp_path, "turn", [record(i * 0.1, yaw=40.0) for i in range(11)], "right")
    wall = save(
        tmp_path, "wall", [record(i * 0.1, approach=20.0) for i in range(10)], "forward", 50
    )

    args = SimpleNamespace(
        still=[still],
        moving=[forward],
        push=[push],
        turn=turn,
        measured_deg=48.0,
        straight=forward,
        wall=wall,
    )
    assert m3_check.calibrate(args) == 0
    out = capsys.readouterr().out
    assert "moving_px_s = 4.0" in out  # sqrt(1 * 16)
    assert "uncommanded_px_s = 5.0" in out  # sqrt(1 * 25)
    assert "hfov_deg = 64.2" in out  # 53.5 * 48 / 40
    assert "speed_cm_s_per_unit = 0.400" in out
    assert "straight_max_rate_dps = 15.0" in out


def test_calibrate_without_records_fails(capsys):
    args = SimpleNamespace(
        still=None, moving=None, push=None, turn=None, measured_deg=None, straight=None, wall=None
    )
    assert m3_check.calibrate(args) == 1


# --- M1 の Wi-Fi 切断 ---


class FakeClient:
    """3 回目の前進指令で通信が切れ、2 回失敗した後に戻る"""

    def __init__(self, events, refuse=False):
        self.moves = 0
        self.reconnect_failures = 2
        self.events = events
        self.refuse = refuse
        self.stops = 0

    def get_status_sync(self):
        return SimpleNamespace(latest_event_id=5, blocked=SimpleNamespace(forward="obstacle"))

    def move_sync(self, *_args, **_kwargs):
        self.moves += 1
        if self.refuse:
            return False
        if self.moves >= 3:
            raise httpx.ConnectError("Wi-Fi off")
        return True

    def stop_sync(self):
        if self.reconnect_failures > 0:
            self.reconnect_failures -= 1
            raise httpx.ConnectError("still off")
        self.stops += 1
        return True

    def get_events_sync(self, after=0):
        assert after == 5
        return EventsResponse(events=self.events, latest_event_id=6)


def ttl_stop(latency_ms):
    return SafetyEvent(
        id=6,
        at_unix_ms=1,
        kind="reflex_stop",
        reason="ttl_expired",
        direction="forward",
        stop_latency_ms=latency_ms,
    )


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(m1_check.time, "sleep", lambda _s: None)


def test_wifi_check_passes_when_the_pi_stops_on_ttl(no_sleep):
    client = FakeClient([ttl_stop(30.0)])
    assert m1_check.check_wifi(client, seconds=5.0)
    assert client.stops == 1


def test_wifi_check_fails_when_the_stop_is_late(no_sleep):
    assert not m1_check.check_wifi(FakeClient([ttl_stop(250.0)]), seconds=5.0)


def test_wifi_check_fails_without_a_ttl_stop(no_sleep):
    assert not m1_check.check_wifi(FakeClient([]), seconds=5.0)


def test_wifi_check_stops_when_the_pi_refuses(no_sleep):
    client = FakeClient([], refuse=True)
    client.reconnect_failures = 0
    assert not m1_check.check_wifi(client, seconds=5.0)
