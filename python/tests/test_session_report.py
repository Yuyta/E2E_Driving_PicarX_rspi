"""収集したセッションの確認（`src.e2e.dataset.session_report`）"""

from src.e2e.dataset.session_report import report_session, warnings


def frame(t, steering=0.0, speed=50, flags=(), seq=None):
    return {
        "timestamp": t,
        "steering": steering,
        "throttle": 0.5 if speed else 0.0,
        "command": {"speed": speed},
        "motion_check": {"flags": list(flags)},
        "stream_seq": seq,
        "floor": [900, 900, 900],
        "battery_v": 8.0,
    }


def test_a_good_session_has_no_warnings():
    steers = [-0.5, -0.3, 0.0, 0.3, 0.5] * 20
    records = [frame(i / 15, s, seq=i) for i, s in enumerate(steers)]
    report = report_session(records)
    assert report["fps"] == 15.0
    assert report["moving_ratio"] == 1.0
    assert report["turn_balance"] == 1.0
    assert report["skipped_stream_frames"] == 0
    assert warnings(report) == []


def test_pauses_do_not_lower_the_frame_rate_or_count_as_skipped_frames():
    records = [frame(i / 15, 0.3 if i % 2 else -0.3, seq=i) for i in range(30)]
    records += [frame(10 + i / 15, 0.3 if i % 2 else -0.3, seq=200 + i) for i in range(30)]
    report = report_session(records)
    assert report["pauses"] == 1
    assert report["fps"] == 15.0
    assert report["skipped_stream_frames"] == 0


def test_the_first_2026_11_09_session_would_have_been_caught():
    # 走行中が少なく、舵角が最大近くに偏り、右に偏っている
    records = [frame(i / 15, 0.0, speed=0) for i in range(80)]
    records += [frame(10 + i / 15, 1.0) for i in range(15)]
    records += [frame(20 + i / 15, -0.4) for i in range(5)]
    notes = warnings(report_session(records))
    assert any("走っているフレーム" in n for n in notes)
    assert any("左右の偏り" in n for n in notes)
    assert any("最大近く" in n for n in notes)


def test_flagged_frames_are_not_counted_as_usable():
    records = [frame(i / 15, 0.3, flags=["no_motion"] if i < 5 else ()) for i in range(20)]
    report = report_session(records)
    assert report["flagged"] == 5
    assert report["usable"] == 15


def test_legacy_sessions_get_no_warnings():
    records = [{"timestamp": i, "steering": -0.5, "throttle": 0.5} for i in range(10)]
    report = report_session(records)
    assert report["legacy"]
    assert warnings(report) == []
