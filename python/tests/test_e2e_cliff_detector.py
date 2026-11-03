"""
Unit Tests for PiCar-X Cliff Detector (Phase 3.1 Safety Layer)
"""

from src.e2e.safety.cliff_detector import CliffDetector


def test_normal_floor_is_safe():
    detector = CliffDetector(threshold=150)
    # 通常の床面 (反射光がしっかり戻る)
    normal_floor = [800, 850, 820]
    assert detector.is_cliff(normal_floor) is False
    assert detector.get_cliff_status(normal_floor) == (False, False, False)


def test_left_dropoff_escape():
    detector = CliffDetector(threshold=150)
    # 左側センサーだけ床から落ちそう
    left_cliff = [80, 850, 820]
    assert detector.is_cliff(left_cliff) is True
    assert detector.get_cliff_status(left_cliff) == (True, False, False)

    steer, speed, duration = detector.get_escape_action(left_cliff)
    # 左前が崖なので、右にハンドルを切って後退
    assert steer == 30
    assert speed < 0
    assert duration > 0


def test_right_dropoff_escape():
    detector = CliffDetector(threshold=150)
    # 右側センサーだけ床から落ちそう
    right_cliff = [800, 850, 70]
    assert detector.is_cliff(right_cliff) is True
    assert detector.get_cliff_status(right_cliff) == (False, False, True)

    steer, speed, duration = detector.get_escape_action(right_cliff)
    # 右前が崖なので、左にハンドルを切って後退
    assert steer == -30
    assert speed < 0
    assert duration > 0


def test_front_or_total_dropoff_escape():
    detector = CliffDetector(threshold=150)
    # 中央または全体が段差
    total_cliff = [50, 60, 40]
    assert detector.is_cliff(total_cliff) is True
    assert detector.get_cliff_status(total_cliff) == (True, True, True)

    steer, speed, duration = detector.get_escape_action(total_cliff)
    # まっすぐ後退
    assert steer == 0
    assert speed < 0


def test_floor_calibration():
    detector = CliffDetector(threshold=150)
    # サンプル床面 (平均約1000)
    samples = [
        [1000, 1050, 1020],
        [980, 1020, 1000],
        [1020, 1010, 990],
    ]
    new_threshold = detector.calibrate(samples)
    # 最低値約1000の35% ≒ 350前後が閾値になる
    assert 300 <= new_threshold <= 400
    assert detector.threshold == new_threshold

    # 新閾値での判定
    assert detector.is_cliff([800, 800, 800]) is False
    assert detector.is_cliff([250, 800, 800]) is True
