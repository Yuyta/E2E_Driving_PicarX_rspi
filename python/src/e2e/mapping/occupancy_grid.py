"""
Phase 3.5: Dynamic Occupancy Grid Map
操舵角・車速による推測航法（Dead Reckoning）と前方超音波センサーから
2D空間の確率占有格子地図（Occupancy Grid）をリアルタイム構築するモジュール。
家具の移動やドアの開閉に対応する時間忘却機能（Decay）を備える。
"""

import math
from typing import List, Optional, Tuple

import numpy as np


class OccupancyGridMap:
    """動的確率オキュパンシーグリッド地図"""

    def __init__(
        self,
        grid_width: int = 200,
        grid_height: int = 200,
        cell_size_m: float = 0.05,
        decay_rate: float = 0.001,
        wheelbase_m: float = 0.14,
        speed_scale: float = 0.008,
    ):
        """
        Args:
            grid_width: 横方向セル数 (200セル * 0.05m = 10m)
            grid_height: 縦方向セル数 (200セル * 0.05m = 10m)
            cell_size_m: 1セルの実空間解像度 (メートル、0.05m = 5cm)
            decay_rate: 各ステップでの忘却減衰率 (0.5への引き戻し係数)
            wheelbase_m: PiCar-Xのホイールベース (メートル、約14cm)
            speed_scale: 指令車速(0〜100)から実車速(m/s)への変換係数 (35 * 0.008 ≈ 0.28m/s)
        """
        self.width = grid_width
        self.height = grid_height
        self.cell_size = float(cell_size_m)
        self.decay_rate = float(decay_rate)
        self.wheelbase = float(wheelbase_m)
        self.speed_scale = float(speed_scale)

        # マップ原点（自車の初期位置：グリッド中心）
        self.origin_x = grid_width // 2
        self.origin_y = grid_height // 2

        # 占有確率マップ: 0.0(空き/Free) 〜 0.5(未探索/Unknown) 〜 1.0(障害物/Occupied)
        self.grid = np.full((grid_height, grid_width), 0.5, dtype=np.float32)

        # 自車姿勢 (x[m], y[m], theta[rad]) ※ 初期位置原点 (0, 0, 0)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0

        # 走行軌跡履歴 (セル座標のリスト [(cx, cy), ...])
        self.path: List[Tuple[int, int]] = [(self.origin_x, self.origin_y)]

    def world_to_grid(self, x_m: float, y_m: float) -> Optional[Tuple[int, int]]:
        """世界座標(m) -> グリッド座標(col, row)の変換"""
        gx = int(round(self.origin_x + (x_m / self.cell_size)))
        # 画像・グリッドの上下反転（y前進を画面上方向にするため）
        gy = int(round(self.origin_y - (y_m / self.cell_size)))

        if 0 <= gx < self.width and 0 <= gy < self.height:
            return (gx, gy)
        return None

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """グリッド座標(col, row) -> 世界座標(m)の変換"""
        x_m = (gx - self.origin_x) * self.cell_size
        y_m = (self.origin_y - gy) * self.cell_size
        return (x_m, y_m)

    def update_odometry(self, speed: int, steer_angle_deg: int, duration_ms: int) -> None:
        """
        車速と操舵角から推測航法（Dead Reckoning）で自車位置を更新。

        Args:
            speed: 車速 (0〜100、負なら後退)
            steer_angle_deg: 前輪ステアリング角度 (-35°〜+35°)
            duration_ms: 経過時間 (ミリ秒)
        """
        if duration_ms <= 0:
            return

        dt = duration_ms / 1000.0
        v = speed * self.speed_scale  # m/s
        steer_rad = math.radians(steer_angle_deg)

        # アッカーマン移動モデル
        if abs(steer_rad) < 1e-4:
            # 直進
            dx = v * math.cos(self.robot_theta) * dt
            dy = v * math.sin(self.robot_theta) * dt
            dtheta = 0.0
        else:
            # 旋回
            dtheta = (v / self.wheelbase) * math.tan(steer_rad) * dt
            dx = v * math.cos(self.robot_theta + dtheta * 0.5) * dt
            dy = v * math.sin(self.robot_theta + dtheta * 0.5) * dt

        self.robot_x += dx
        self.robot_y += dy
        self.robot_theta = (self.robot_theta + dtheta) % (2 * math.pi)

        # 軌跡の記録
        curr_grid = self.world_to_grid(self.robot_x, self.robot_y)
        if curr_grid and (not self.path or curr_grid != self.path[-1]):
            self.path.append(curr_grid)
            # 自車が通過した直下のセルは確実に「空き（0.1）」として更新
            self.grid[curr_grid[1], curr_grid[0]] = 0.1

    def update_sensor_ultrasonic(
        self, distance_cm: float, max_range_cm: float = 100.0
    ) -> None:
        """
        前方超音波センサーの距離観測によるグリッドマップの確率更新。
        Ray casting で通過経路をFree(0.2)、到達点をOccupied(0.85)に更新。
        """
        origin_pt = self.world_to_grid(self.robot_x, self.robot_y)
        if not origin_pt:
            return

        x0, y0 = origin_pt

        # 障害物の世界座標を算出
        is_obstacle_detected = 0.0 < distance_cm < max_range_cm
        sensing_dist_m = (
            (distance_cm / 100.0) if is_obstacle_detected else (max_range_cm / 100.0)
        )

        obs_x_m = self.robot_x + sensing_dist_m * math.cos(self.robot_theta)
        obs_y_m = self.robot_y + sensing_dist_m * math.sin(self.robot_theta)

        end_pt = self.world_to_grid(obs_x_m, obs_y_m)
        if not end_pt:
            return

        x1, y1 = end_pt

        # ブレゼンハムの直線描画で光線上のセルを取得
        ray_cells = self._bresenham_line(x0, y0, x1, y1)

        # 手前の空間: 障害物なし (Free)
        for cx, cy in ray_cells[:-1]:
            # ベイズ的更新 (空きスペース確率を低下)
            curr = self.grid[cy, cx]
            self.grid[cy, cx] = max(0.05, curr * 0.85)

        # 到達点: 実際に障害物を検知した場合のみ Occupied (0.85) に更新
        if is_obstacle_detected:
            # 障害物セルの周辺も含めて少し太めにプロット
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    nx, ny = x1 + dx, y1 + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        curr = self.grid[ny, nx]
                        self.grid[ny, nx] = min(0.95, curr * 1.3)

    def decay(self) -> None:
        """
        時間経過に伴う忘却機能（Decay）。
        家具の配置変更やドアの開閉に対応するため、未知(0.5)へ向けて緩やかに減衰させる。
        """
        # grid = grid + (0.5 - grid) * decay_rate
        self.grid += (0.5 - self.grid) * self.decay_rate

    def clear(self) -> None:
        """地図と軌跡を初期化"""
        self.grid.fill(0.5)
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_theta = 0.0
        self.path = [(self.origin_x, self.origin_y)]

    @staticmethod
    def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        """ブレゼンハムの直線アルゴリズム"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        cx, cy = x0, y0
        while True:
            points.append((cx, cy))
            if cx == x1 and cy == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy

        return points

    def get_explored_ratio(self) -> float:
        """マップ全体の探索済みセル比率 (0.5以外のセルの割合)"""
        explored = np.count_nonzero(np.abs(self.grid - 0.5) > 0.05)
        return float(explored / self.grid.size)
