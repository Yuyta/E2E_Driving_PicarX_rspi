import pytest

from src.e2e.safety.speed_limiter import SpeedLimiter


def test_speed_limiter_init_validation():
    # min_speed > base_speed raises error
    with pytest.raises(ValueError):
        SpeedLimiter(base_speed=20, min_speed=35)

    # safe_distance >= slowdown_distance raises error
    with pytest.raises(ValueError):
        SpeedLimiter(safe_distance=30.0, slowdown_distance=20.0)


def test_curve_speed_reduction():
    limiter = SpeedLimiter(base_speed=40, min_speed=20, curve_factor=1.0)

    # 直進時 (|norm| = 0.0) -> base_speed
    assert limiter.calculate_curve_speed(0.0) == 40

    # 最大舵角 (|norm| = 1.0) -> min_speed
    assert limiter.calculate_curve_speed(1.0) == 20
    assert limiter.calculate_curve_speed(-1.0) == 20

    # 50%舵角 -> 中間速度 (30)
    assert limiter.calculate_curve_speed(0.5) == 30
    assert limiter.calculate_curve_speed(-0.5) == 30


def test_obstacle_speed_reduction():
    limiter = SpeedLimiter(
        base_speed=40,
        min_speed=20,
        safe_distance=15.0,
        slowdown_distance=35.0,
    )

    # 遠い障害物 (>= 35cm) -> 減速なし (40)
    assert limiter.calculate_obstacle_speed(50.0) == 40
    assert limiter.calculate_obstacle_speed(35.0) == 40

    # 近い障害物 (<= 15cm) -> 最低速度 (20)
    assert limiter.calculate_obstacle_speed(15.0) == 20
    assert limiter.calculate_obstacle_speed(10.0) == 20

    # 中間距離 (25cm) -> ちょうど中間 (30)
    assert limiter.calculate_obstacle_speed(25.0) == 30

    # None -> 減速なし (40)
    assert limiter.calculate_obstacle_speed(None) == 40


def test_combined_speed_calculation():
    limiter = SpeedLimiter(
        base_speed=40,
        min_speed=20,
        safe_distance=15.0,
        slowdown_distance=35.0,
    )

    # 直進かつ障害物なし -> base_speed
    assert limiter.calculate(steering_norm=0.0, obstacle_distance_cm=50.0) == 40

    # 直進だが障害物が25cmに接近 -> 30に減速
    assert limiter.calculate(steering_norm=0.0, obstacle_distance_cm=25.0) == 30

    # 急旋回 (norm=1.0) かつ障害物なし -> 20に減速
    assert limiter.calculate(steering_norm=1.0, obstacle_distance_cm=50.0) == 20

    # 緩い旋回 (norm=0.5 -> 30) と 障害物接近 (15cm -> 20) -> より遅い20を採用
    assert limiter.calculate(steering_norm=0.5, obstacle_distance_cm=15.0) == 20
