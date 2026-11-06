"""M1: Pi の状態 API、TTL、配信の受信、周期計測を PC 側から検証する。"""

import json
import time

import httpx
import pytest
import respx
from pydantic import ValidationError

from src.e2e.cycle_profiler import CycleProfiler
from src.e2e.m1_check import check_approach, summarize_stop_latency
from src.e2e.safety.body_status import BodyAction, cliff_sensors, decide_action
from src.models import DriveDirection, MoveCommand, RobotStatus, SafetyEvent, SafetyReason
from src.picarx_client.camera_stream import LatestFrameStream, iter_jpeg_frames
from src.picarx_client.client import PicarxClient

BASE_URL = "http://test-robot:8080"


def status_payload(forward=None, cm=40.0, cliff=None):
    return {
        "mode": "idle",
        "emergency_stop": False,
        "motion": None,
        "steer_angle": 0,
        "blocked": {"forward": forward, "backward": None},
        "sensors": {
            "distance": {"cm": cm, "age_ms": 12},
            "floor": {"values": [500, 500, 500], "age_ms": 8, "cliff": cliff},
            "battery": {"voltage": 7.6, "percent": 50, "age_ms": 300},
        },
        "cliff": {
            "guard_enabled": True,
            "calibration": {"baseline": [500, 500, 500], "thresholds": [50, 50, 50]},
        },
        "reflex": {"hz": 39.5, "max_interval_ms": 31.0, "last_tick_age_ms": 4},
        "camera": {"running": True, "fps": 15.0, "frames": 42, "frame_age_ms": 20, "error": None},
        "latest_event_id": 3,
    }


def make_status(**kwargs) -> RobotStatus:
    return RobotStatus.model_validate(status_payload(**kwargs))


@pytest.mark.parametrize(
    "reason, expected",
    [
        (None, BodyAction.DRIVE),
        ("obstacle", BodyAction.AVOID_OBSTACLE),
        ("cliff", BodyAction.ESCAPE_CLIFF),
        ("distance_unknown", BodyAction.WAIT),
        ("floor_unknown", BodyAction.WAIT),
        ("cliff_uncalibrated", BodyAction.WAIT),
        ("reflex_stalled", BodyAction.WAIT),
        ("emergency_stop", BodyAction.WAIT),
    ],
)
def test_decide_action_follows_the_pi(reason, expected):
    action, returned = decide_action(make_status(forward=reason))
    assert action == expected
    assert returned == (SafetyReason(reason) if reason else None)


def test_cliff_sensors_are_read_from_status():
    status = make_status(forward="cliff", cliff=[True, False, False])
    assert cliff_sensors(status) == (True, False, False)
    assert cliff_sensors(make_status()) is None


def test_unknown_sensor_values_stay_none():
    status = make_status(cm=None)
    assert status.sensors.distance.cm is None


@pytest.mark.parametrize("ttl_ms", [49, 1001])
def test_move_command_rejects_ttl_outside_the_pi_range(ttl_ms):
    with pytest.raises(ValidationError):
        MoveCommand(direction=DriveDirection.FORWARD, ttl_ms=ttl_ms)


@respx.mock
def test_move_sync_sends_ttl_and_reports_refusal():
    route = respx.post(f"{BASE_URL}/api/drive/move").mock(
        return_value=httpx.Response(409, json={"status": "refused", "reason": "cliff"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert client.move_sync(DriveDirection.FORWARD, speed=20, ttl_ms=300) is False
    body = json.loads(route.calls.last.request.read())
    assert body == {"direction": "forward", "speed": 20, "ttl_ms": 300}


@respx.mock
def test_status_and_events_are_parsed():
    respx.get(f"{BASE_URL}/api/status").mock(
        return_value=httpx.Response(200, json=status_payload())
    )
    events_route = respx.get(f"{BASE_URL}/api/events").mock(
        return_value=httpx.Response(
            200,
            json={
                "events": [
                    {
                        "id": 4,
                        "at_unix_ms": 1,
                        "kind": "reflex_stop",
                        "reason": "obstacle",
                        "direction": "forward",
                        "distance_cm": 12.0,
                        "stop_latency_ms": 41.5,
                    }
                ],
                "latest_event_id": 4,
            },
        )
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")

    assert client.get_status_sync().sensors.battery.percent == 50
    events = client.get_events_sync(after=3)
    assert events_route.calls.last.request.url.params["after"] == "3"
    assert events.events[0].reason == "obstacle"


@respx.mock
def test_sync_sensor_reads_raise_when_the_value_is_unknown():
    respx.get(f"{BASE_URL}/api/sensor/ultrasonic").mock(
        return_value=httpx.Response(503, json={"status": "error", "message": "距離が不明です"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    with pytest.raises(httpx.HTTPStatusError):
        client.get_ultrasonic_sync()


@respx.mock
def test_calibration_failure_raises():
    respx.post(f"{BASE_URL}/api/safety/calibrate").mock(
        return_value=httpx.Response(422, json={"status": "error", "message": "床が暗すぎます"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    with pytest.raises(httpx.HTTPStatusError):
        client.calibrate_floor_sync()


@pytest.mark.asyncio
@respx.mock
async def test_drive_for_short_duration_sends_a_single_command():
    route = respx.post(f"{BASE_URL}/api/drive/move").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert await client.drive_for(DriveDirection.FORWARD, 30, 500) is True
    assert route.call_count == 1
    assert json.loads(route.calls.last.request.read())["ttl_ms"] == 500


@pytest.mark.asyncio
@respx.mock
async def test_drive_for_long_duration_renews_the_ttl():
    route = respx.post(f"{BASE_URL}/api/drive/move").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert await client.drive_for(DriveDirection.FORWARD, 30, 1100) is True
    assert route.call_count >= 3
    ttls = [json.loads(call.request.read())["ttl_ms"] for call in route.calls]
    assert all(50 <= ttl <= 500 for ttl in ttls)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration_ms", [0, -1, 5001])
async def test_drive_for_rejects_durations_out_of_range(duration_ms):
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    with pytest.raises(ValueError):
        await client.drive_for(DriveDirection.FORWARD, 30, duration_ms)


@pytest.mark.asyncio
@respx.mock
async def test_drive_for_stops_renewing_when_refused():
    route = respx.post(f"{BASE_URL}/api/drive/move").mock(
        side_effect=[
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(409, json={"status": "refused", "reason": "obstacle"}),
        ]
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    with pytest.raises(httpx.HTTPStatusError):
        await client.drive_for(DriveDirection.FORWARD, 30, 3000)
    assert route.call_count == 2


@pytest.mark.asyncio
@respx.mock
async def test_snapshot_no_longer_sends_ignored_parameters():
    route = respx.get(f"{BASE_URL}/api/camera/snapshot").mock(
        return_value=httpx.Response(
            200, json={"image_base64": "AA==", "width": 320, "height": 240, "seq": 1, "age_ms": 5}
        )
    )
    snapshot = await PicarxClient(base_url=BASE_URL).get_camera_snapshot()
    assert (snapshot.width, snapshot.height) == (320, 240)
    assert not route.calls.last.request.url.params


def jpeg(body: bytes) -> bytes:
    return b"\xff\xd8" + body + b"\xff\xd9"


def test_iter_jpeg_frames_parses_multipart_chunks():
    part = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 5\r\n\r\n"
    stream = part + jpeg(b"a") + b"\r\n" + part + jpeg(b"\xff\x00b") + b"\r\n"
    assert list(iter_jpeg_frames([stream])) == [jpeg(b"a"), jpeg(b"\xff\x00b")]


def test_iter_jpeg_frames_handles_markers_split_across_chunks():
    data = jpeg(b"abc")
    assert list(iter_jpeg_frames(data[i : i + 1] for i in range(len(data)))) == [data]


def test_latest_frame_stream_only_returns_newer_frames():
    frames = [jpeg(b"1"), jpeg(b"2")]

    def chunks():
        yield from frames
        time.sleep(5)

    with LatestFrameStream(chunks) as stream:
        deadline = time.monotonic() + 2.0
        latest = stream.wait_for_frame(timeout=2.0)
        while latest and latest.seq < 2 and time.monotonic() < deadline:
            latest = stream.wait_for_frame(after_seq=latest.seq, timeout=1.0)
        assert latest is not None and latest.jpeg == frames[1]
        # 新しいフレームがなければ待って None を返す
        assert stream.wait_for_frame(after_seq=latest.seq, timeout=0.05) is None


def test_latest_frame_stream_reconnects_after_an_error():
    calls = []

    def chunks():
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("refused")
        yield jpeg(b"ok")

    with LatestFrameStream(chunks) as stream:
        frame = stream.wait_for_frame(timeout=3.0)
    assert frame is not None and frame.jpeg == jpeg(b"ok")


def test_cycle_profiler_reports_stage_breakdown(tmp_path):
    ticks = iter([0.0, 0.010, 0.020, 0.100, 0.200])
    profiler = CycleProfiler(clock=lambda: next(ticks))
    profiler.mark_cycle()  # 0.0
    with profiler.stage("capture"):  # 0.010 -> 0.020
        pass
    profiler.mark_cycle()  # 0.100
    profiler.mark_cycle()  # 0.200

    summary = profiler.summary()
    assert summary["capture"]["p50_ms"] == pytest.approx(10.0)
    assert summary["total"]["count"] == 2
    assert profiler.cycle_hz() == pytest.approx(10.0)
    assert "capture" in profiler.format_table()

    saved = json.loads(profiler.save(tmp_path).read_text(encoding="utf-8"))
    assert saved["cycle_hz_p50"] == pytest.approx(10.0)


def test_summarize_stop_latency_only_counts_hazard_stops():
    def event(event_id, kind, reason, latency):
        return SafetyEvent(
            id=event_id, at_unix_ms=0, kind=kind, reason=reason, stop_latency_ms=latency
        )

    events = [
        event(1, "reflex_stop", "obstacle", 30.0),
        event(2, "reflex_stop", "cliff", 80.0),
        event(3, "reflex_stop", "ttl_expired", 500.0),
        event(4, "emergency_stop", None, None),
    ]
    assert summarize_stop_latency(events) == 80.0
    # TTL の停止だけでは合格にしない
    assert summarize_stop_latency([event(1, "reflex_stop", "ttl_expired", 19.0)]) is None


@respx.mock
def test_approach_keeps_renewing_until_the_pi_refuses(monkeypatch):
    monkeypatch.setattr("src.e2e.m1_check.time.sleep", lambda _sec: None)
    move = respx.post(f"{BASE_URL}/api/drive/move").mock(
        side_effect=[
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(200, json={"status": "ok"}),
            httpx.Response(409, json={"status": "refused", "reason": "obstacle"}),
        ]
    )
    stop = respx.post(f"{BASE_URL}/api/drive/stop").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.get(f"{BASE_URL}/api/status").mock(
        return_value=httpx.Response(200, json=status_payload(forward="obstacle", cm=12.0))
    )
    respx.get(f"{BASE_URL}/api/events").mock(
        return_value=httpx.Response(
            200,
            json={
                "events": [
                    {
                        "id": 4,
                        "at_unix_ms": 1,
                        "kind": "reflex_stop",
                        "reason": "obstacle",
                        "distance_cm": 14.9,
                        "stop_latency_ms": 9.0,
                    }
                ],
                "latest_event_id": 4,
            },
        )
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")

    assert check_approach(client, speed=50, seconds=3.0) is True
    assert move.call_count == 3
    assert stop.called
