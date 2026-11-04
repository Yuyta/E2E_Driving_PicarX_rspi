"""
Integrated Test for PiCar-X E2E Safety, Mapping & Autonomy Layer
全安全系（段差検知、動的減速、通信監視、直前軌跡後退、スタック検知）と
動的オキュパンシーグリッドが協調して整合性を保って動作することを検証する。
"""

import numpy as np

from src.e2e.mapping.occupancy_grid import OccupancyGridMap
from src.e2e.safety.speed_limiter import SpeedLimiter
from src.e2e.safety.stuck_detector import StuckDetector
from src.e2e.safety.trajectory_tracker import TrajectoryTracker
from src.e2e.safety.watchdog import CommunicationWatchdog


def test_integrated_cliff_and_breadcrumb_escape():
    """段差検知と直前軌跡逆再生の協調動作テスト"""
    tracker = TrajectoryTracker(default_reverse_duration_ms=1000, reverse_speed=25)
    grid_map = OccupancyGridMap(grid_width=100, grid_height=100, cell_size_m=0.05)

    # 1. ロボットが直進前進 (speed=30, steer=0, 500ms)
    tracker.record(speed=30, steer_angle=0, duration_ms=500, timestamp=1.0)
    grid_map.update_odometry(speed=30, steer_angle_deg=0, duration_ms=500)
    assert grid_map.robot_x > 0.0

    # 2. Pi が左側センサーで段差を検知（/api/status の floor.cliff）
    cliff_status = (True, False, False)

    # 3. 直前軌跡から逆再生シーケンスを取得
    reverse_seq = tracker.get_reverse_sequence(target_duration_ms=500)
    assert len(reverse_seq) >= 1
    # 後退を実行し、オドメトリを更新
    for steer, spd, dur in reverse_seq:
        grid_map.update_odometry(speed=-spd, steer_angle_deg=steer, duration_ms=dur)

    # 4. 後退後のエスケープ旋回
    turn_steer, turn_spd, turn_dur = tracker.get_escape_turn(cliff_status)
    # 左が崖だったため、右方向 (+35°) へ旋回前進するはず
    assert turn_steer > 0
    grid_map.update_odometry(speed=turn_spd, steer_angle_deg=turn_steer, duration_ms=turn_dur)

    # 履歴クリアで次の走行へ
    tracker.clear()
    assert len(tracker.history) == 0


def test_integrated_speed_limiter_and_mapping():
    """動的減速とオキュパンシーグリッドの協調動作テスト"""
    speed_limiter = SpeedLimiter(
        base_speed=35,
        min_speed=20,
        safe_distance=15.0,
        slowdown_distance=40.0,
    )
    grid_map = OccupancyGridMap(grid_width=100, grid_height=100, cell_size_m=0.05)

    # 遠い障害物 (50cm) -> 通常車速 (35)
    spd_far = speed_limiter.calculate(steering_norm=0.0, obstacle_distance_cm=50.0)
    assert spd_far == 35

    # 障害物が30cmに接近 -> 動的減速 (20〜35の中間)
    spd_mid = speed_limiter.calculate(steering_norm=0.0, obstacle_distance_cm=30.0)
    assert 20 < spd_mid < 35

    # グリッドマップに障害物をプロット
    grid_map.update_sensor_ultrasonic(distance_cm=30.0)
    # 障害物位置のセル確率が上昇していること
    obs_pt = grid_map.world_to_grid(0.30, 0.0)
    assert obs_pt is not None
    assert grid_map.grid[obs_pt[1], obs_pt[0]] > 0.5


def test_integrated_stuck_detection_and_watchdog():
    """スタック検知と通信ウォッチドッグの整合性テスト"""
    stuck_detector = StuckDetector(
        history_frames=3,
        min_motion_threshold=4.0,
        stuck_duration_frames=3,
        min_moving_speed=15,
    )
    watchdog = CommunicationWatchdog(timeout_sec=0.5, max_consecutive_errors=3)

    static_frame = np.zeros((120, 160, 3), dtype=np.uint8)

    # 正常通信ハートビート
    watchdog.heartbeat()
    assert not watchdog.should_stop

    # モーター駆動中だが静止状態が継続
    for _ in range(6):
        stuck_detector.update(static_frame, motor_speed=30)
        watchdog.heartbeat()

    assert stuck_detector.is_stuck
    assert not watchdog.should_stop  # 通信自体は正常

    # 脱出プランの取得
    plan = stuck_detector.get_unstuck_plan(escape_speed=25)
    assert len(plan) >= 3

    # 脱出後のリセット
    stuck_detector.reset()
    assert not stuck_detector.is_stuck
