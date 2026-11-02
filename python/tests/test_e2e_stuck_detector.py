import numpy as np

from src.e2e.safety.stuck_detector import StuckDetector


def test_stuck_detector_normal_moving():
    detector = StuckDetector(
        history_frames=3,
        min_motion_threshold=5.0,
        stuck_duration_frames=3,
        min_moving_speed=15,
    )

    # 変化するフレーム（正常走行中）
    for i in range(10):
        # フレームごとに輝度が変化する画像
        frame = np.full((120, 160, 3), (i * 25) % 256, dtype=np.uint8)
        is_stuck = detector.update(frame, motor_speed=30)
        assert not is_stuck
    assert not detector.is_stuck


def test_stuck_detector_stopped_motor():
    detector = StuckDetector(
        history_frames=3,
        min_motion_threshold=5.0,
        stuck_duration_frames=3,
        min_moving_speed=15,
    )

    # 同一画像（静止）だが、モーター停止中 (speed=0)
    static_frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(10):
        is_stuck = detector.update(static_frame, motor_speed=0)
        assert not is_stuck
    assert not detector.is_stuck


def test_stuck_detector_trigger_when_stuck():
    detector = StuckDetector(
        history_frames=3,
        min_motion_threshold=5.0,
        stuck_duration_frames=4,
        min_moving_speed=15,
    )

    static_frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # 最初の数フレームはバッファ充填中
    for _ in range(3):
        assert not detector.update(static_frame, motor_speed=30)

    # バッファ充填後、同一画像（動きゼロ）が4フレーム継続
    assert not detector.update(static_frame, motor_speed=30)  # 1フレーム目
    assert not detector.update(static_frame, motor_speed=30)  # 2フレーム目
    assert not detector.update(static_frame, motor_speed=30)  # 3フレーム目
    # 4フレーム目でスタック確定
    assert detector.update(static_frame, motor_speed=30)
    assert detector.is_stuck

    # リセットの確認
    detector.reset()
    assert not detector.is_stuck
    assert detector.consecutive_stuck_count == 0


def test_unstuck_plan_generation():
    detector = StuckDetector()
    plan = detector.get_unstuck_plan(escape_speed=30)

    assert len(plan) >= 4
    # 最初のステップは後退（reverse）であること
    assert plan[0][0] == "reverse"
    assert plan[0][2] == 30  # speed

    # 途中に左右の操舵（forward with steer != 0）が含まれること
    actions = [p[0] for p in plan]
    steers = [p[1] for p in plan]
    assert "forward" in actions
    assert any(s < 0 for s in steers)  # 左操舵
    assert any(s > 0 for s in steers)  # 右操舵
