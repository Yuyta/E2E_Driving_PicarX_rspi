from src.e2e.safety.trajectory_tracker import TrajectoryTracker


def test_trajectory_tracker_record():
    tracker = TrajectoryTracker(max_history_points=5)
    assert len(tracker.history) == 0

    # 正常な前進記録
    tracker.record(speed=35, steer_angle=10, duration_ms=100, timestamp=1.0)
    tracker.record(speed=35, steer_angle=20, duration_ms=100, timestamp=1.1)
    assert len(tracker.history) == 2

    # 停止中 (speed=0) は記録されない
    tracker.record(speed=0, steer_angle=20, duration_ms=100, timestamp=1.2)
    assert len(tracker.history) == 2


def test_reverse_sequence_empty():
    tracker = TrajectoryTracker(default_reverse_duration_ms=1000, reverse_speed=30)
    # 履歴がなければ後方の安全を確認できないので後退しない
    assert tracker.get_reverse_sequence() == []


def test_reverse_sequence_unwind():
    tracker = TrajectoryTracker(default_reverse_duration_ms=600, reverse_speed=35)

    # 3ステップ走行: 0° -> 15° -> 30°
    tracker.record(speed=35, steer_angle=0, duration_ms=200, timestamp=1.0)
    tracker.record(speed=35, steer_angle=15, duration_ms=200, timestamp=1.2)
    tracker.record(speed=35, steer_angle=30, duration_ms=200, timestamp=1.4)

    # 600ms 分の逆再生シーケンスを取得
    seq = tracker.get_reverse_sequence(target_duration_ms=600)

    # 逆順に取り出される: 30° -> 15° -> 0°
    assert len(seq) == 3
    assert seq[0] == (30, 35, 200)
    assert seq[1] == (15, 35, 200)
    assert seq[2] == (0, 35, 200)

    # 合計時間がちょうど600ms
    assert sum(s[2] for s in seq) == 600


def test_reverse_sequence_stops_at_the_end_of_history():
    tracker = TrajectoryTracker(default_reverse_duration_ms=1000, reverse_speed=30)

    # 200ms 分しか履歴がない場合
    tracker.record(speed=30, steer_angle=-25, duration_ms=200, timestamp=1.0)

    seq = tracker.get_reverse_sequence(target_duration_ms=800)
    # 記録にある200ms分だけ戻り、その先へは進まない
    assert seq == [(-25, 30, 200)]


def test_reverse_sequence_skips_steps_the_pi_would_reject():
    tracker = TrajectoryTracker(reverse_speed=30)
    tracker.record(speed=30, steer_angle=5, duration_ms=200, timestamp=1.0)
    tracker.record(speed=30, steer_angle=9, duration_ms=20, timestamp=1.2)
    assert tracker.get_reverse_sequence(target_duration_ms=500) == [(5, 30, 200)]


def test_consume_reverse_keeps_the_rest_of_the_history():
    tracker = TrajectoryTracker(reverse_speed=30)
    tracker.record(speed=30, steer_angle=0, duration_ms=200, timestamp=1.0)
    tracker.record(speed=30, steer_angle=10, duration_ms=200, timestamp=1.2)

    tracker.consume_reverse(300)

    assert [(p.steer_angle, p.duration_ms) for p in tracker.history] == [(0, 100)]
    assert tracker.get_reverse_sequence(target_duration_ms=500) == [(0, 30, 100)]
    tracker.consume_reverse(1000)
    assert tracker.get_reverse_sequence() == []


def test_escape_turn_cliff():
    tracker = TrajectoryTracker(reverse_speed=30)

    # 左側が落ちそうな場合 -> 右へ旋回前進 (35°)
    turn = tracker.get_escape_turn(last_cliff_sensors=(True, False, False))
    assert turn[0] == 35

    # 右側が落ちそうな場合 -> 左へ旋回前進 (-35°)
    turn = tracker.get_escape_turn(last_cliff_sensors=(False, False, True))
    assert turn[0] == -35
