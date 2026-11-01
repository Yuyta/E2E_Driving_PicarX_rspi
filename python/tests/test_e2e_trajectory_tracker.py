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
    seq = tracker.get_reverse_sequence()
    # 履歴が空の場合はまっすぐ1000ms後退
    assert len(seq) == 1
    assert seq[0] == (0, 30, 1000)


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


def test_reverse_sequence_partial_and_fill():
    tracker = TrajectoryTracker(default_reverse_duration_ms=1000, reverse_speed=30)

    # 200ms 分しか履歴がない場合
    tracker.record(speed=30, steer_angle=-25, duration_ms=200, timestamp=1.0)

    seq = tracker.get_reverse_sequence(target_duration_ms=800)
    # 200ms分 (-25°) + 残り600msを最後の角度(-25°)で補完
    assert sum(s[2] for s in seq) == 800
    assert seq[0] == (-25, 30, 200)
    assert seq[1] == (-25, 30, 600)


def test_escape_turn_cliff():
    tracker = TrajectoryTracker(reverse_speed=30)

    # 左側が落ちそうな場合 -> 右へ旋回前進 (35°)
    turn = tracker.get_escape_turn(last_cliff_sensors=(True, False, False))
    assert turn[0] == 35

    # 右側が落ちそうな場合 -> 左へ旋回前進 (-35°)
    turn = tracker.get_escape_turn(last_cliff_sensors=(False, False, True))
    assert turn[0] == -35
