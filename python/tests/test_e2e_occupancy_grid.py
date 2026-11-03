import numpy as np

from src.e2e.mapping.occupancy_grid import OccupancyGridMap
from src.e2e.mapping.visualizer import MapVisualizer


def test_coordinate_transforms():
    grid_map = OccupancyGridMap(grid_width=200, grid_height=200, cell_size_m=0.05)

    # 原点 (0m, 0m) -> グリッド中心 (100, 100)
    pt = grid_map.world_to_grid(0.0, 0.0)
    assert pt == (100, 100)

    # 1.0m 前進 (y = +1.0m -> gy = 100 - 20 = 80)
    pt_fwd = grid_map.world_to_grid(0.0, 1.0)
    assert pt_fwd == (100, 80)

    # 逆変換
    wx, wy = grid_map.grid_to_world(100, 80)
    assert abs(wx - 0.0) < 1e-4
    assert abs(wy - 1.0) < 1e-4


def test_odometry_update():
    grid_map = OccupancyGridMap(
        grid_width=200,
        grid_height=200,
        cell_size_m=0.05,
        speed_scale=0.01,  # 30 * 0.01 = 0.3 m/s
    )

    # 初期位置
    assert grid_map.robot_x == 0.0
    assert grid_map.robot_y == 0.0
    assert grid_map.robot_theta == 0.0

    # 直進: speed=30, steer=0, duration=1000ms -> 0.3m 前進 (x方向)
    grid_map.update_odometry(speed=30, steer_angle_deg=0, duration_ms=1000)
    assert abs(grid_map.robot_x - 0.3) < 1e-2
    assert abs(grid_map.robot_y - 0.0) < 1e-2
    assert len(grid_map.path) >= 2


def test_ultrasonic_update():
    grid_map = OccupancyGridMap(grid_width=200, grid_height=200, cell_size_m=0.05)

    origin = grid_map.world_to_grid(0.0, 0.0)
    assert origin is not None

    # 前方40cmに障害物を検知
    # 初期確率はすべて 0.5
    assert grid_map.grid[100, 100] == 0.5

    grid_map.update_sensor_ultrasonic(distance_cm=40.0)

    # 自車直前 (空きスペース) は確率が下がっている (Free)
    assert grid_map.grid[100, 104] < 0.5

    # 40cm先 (約8セル先: gx=108, gy=100) は確率が上がっている (Occupied)
    assert grid_map.grid[100, 108] > 0.5


def test_decay():
    grid_map = OccupancyGridMap(grid_width=200, grid_height=200, decay_rate=0.1)

    # 障害物セル (0.9)
    grid_map.grid[100, 108] = 0.9
    # 空きスペースセル (0.1)
    grid_map.grid[100, 102] = 0.1

    # 減衰実行
    grid_map.decay()

    # 0.9 は 0.5 に向かって下がる (0.9 - (0.9-0.5)*0.1 = 0.86)
    assert grid_map.grid[100, 108] < 0.9
    assert grid_map.grid[100, 108] > 0.5

    # 0.1 は 0.5 に向かって上がる (0.1 + (0.5-0.1)*0.1 = 0.14)
    assert grid_map.grid[100, 102] > 0.1
    assert grid_map.grid[100, 102] < 0.5


def test_visualizer_render():
    grid_map = OccupancyGridMap(grid_width=100, grid_height=100, cell_size_m=0.05)
    visualizer = MapVisualizer(grid_map, display_scale=2, sidebar_width=200)

    dummy_cam = np.zeros((120, 160, 3), dtype=np.uint8)

    canvas = visualizer.render(
        camera_frame_bgr=dummy_cam,
        speed=30,
        steer_angle=15,
        ultrasonic_dist=35.0,
        motion_value=12.5,
        status_text="TEST RUN",
    )

    # 幅 = 100*2 + 200 = 400, 最低高さ = 500
    assert canvas.shape == (500, 400, 3)
    assert canvas.dtype == np.uint8
