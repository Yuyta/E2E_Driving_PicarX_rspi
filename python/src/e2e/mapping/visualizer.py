"""
Phase 3.5: Map Visualizer & Live Dashboard UI
動的オキュパンシーグリッド地図、走行軌跡、自車姿勢、超音波ビーム、
カメラライブ映像、および走行テレメトリを統合表示するリアルタイムGUIダッシュボード。
"""

import math
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .occupancy_grid import OccupancyGridMap


class MapVisualizer:
    """オキュパンシーグリッド地図と走行状態のリアルタイムHUDダッシュボード"""

    def __init__(
        self,
        grid_map: OccupancyGridMap,
        window_name: str = "PiCar-X Indoor Spatial Map & Telemetry",
        display_scale: int = 3,
        sidebar_width: int = 300,
    ):
        """
        Args:
            grid_map: 描画対象のオキュパンシーグリッド地図
            window_name: 表示ウィンドウ名
            display_scale: グリッドセル(200x200)の表示倍率 (3倍で600x600)
            sidebar_width: 右側テレメトリHUDパネルの幅 (px)
        """
        self.map = grid_map
        self.window_name = window_name
        self.scale = max(1, display_scale)
        self.map_view_w = grid_map.width * self.scale
        self.map_view_h = grid_map.height * self.scale
        self.sidebar_w = sidebar_width
        self.canvas_w = self.map_view_w + self.sidebar_w
        # サイドバーのHUD情報をすべて収めるため、最低高さを確保
        self.canvas_h = max(500, self.map_view_h)

        self.last_status: str = "NORMAL"
        self.last_status_color: Tuple[int, int, int] = (0, 200, 0)  # BGR

    def render(
        self,
        camera_frame_bgr: Optional[np.ndarray] = None,
        speed: int = 0,
        steer_angle: int = 0,
        ultrasonic_dist: float = 999.0,
        motion_value: float = 0.0,
        status_text: str = "NORMAL DRIVING",
        status_color: Tuple[int, int, int] = (0, 220, 0),
    ) -> np.ndarray:
        """
        最新の地図と走行情報からダッシュボード画像を生成。

        Returns:
            np.ndarray: 生成された (canvas_h, canvas_w, 3) のBGR画像
        """
        self.last_status = status_text
        self.last_status_color = status_color

        # 全体キャンバス (背景: ダークグレー)
        canvas = np.full((self.canvas_h, self.canvas_w, 3), 30, dtype=np.uint8)

        # ------------------------------------------------------------------
        # 1. 2D グリッドマップの描画
        # ------------------------------------------------------------------
        map_img = self._render_grid_layer()

        # 軌跡（パス）の描画
        self._draw_trajectory(map_img)

        # 超音波センシングコーンの描画
        self._draw_ultrasonic_beam(map_img, ultrasonic_dist)

        # 自車アイコン（位置と向き）の描画
        self._draw_robot(map_img)

        # キャンバスの左側にマップを配置
        canvas[0:self.map_view_h, 0:self.map_view_w] = map_img

        # ------------------------------------------------------------------
        # 2. 右側 HUD / テレメトリパネルの描画
        # ------------------------------------------------------------------
        self._render_sidebar(
            canvas,
            camera_frame_bgr,
            speed,
            steer_angle,
            ultrasonic_dist,
            motion_value,
        )

        return canvas

    def _render_grid_layer(self) -> np.ndarray:
        """確率マップ（0.0〜1.0）をカラー画像化しアップスケール"""
        # 0.5(未探索)=暗いグレー(45), 0.0(Free)=白(240), 1.0(Occupied)=赤/オレンジ(220)
        prob = self.map.grid

        # BGRカラーマップの作成
        grid_bgr = np.zeros((self.map.height, self.map.width, 3), dtype=np.uint8)

        # 未探索マスク (0.45 <= p <= 0.55)
        unknown_mask = np.abs(prob - 0.5) <= 0.05
        # 空きスペース (p < 0.45)
        free_mask = prob < 0.45
        # 障害物 (p > 0.55)
        occupied_mask = prob > 0.55

        # 未探索エリア: ダークスレートグレー (40, 45, 50)
        grid_bgr[unknown_mask] = [50, 45, 40]

        # 空きスペース (確率が低いほど明るい白緑)
        free_intensity = (1.0 - (prob[free_mask] / 0.45)) * 180 + 60
        grid_bgr[free_mask, 0] = (free_intensity * 0.9).astype(np.uint8)
        grid_bgr[free_mask, 1] = free_intensity.astype(np.uint8)
        grid_bgr[free_mask, 2] = (free_intensity * 0.9).astype(np.uint8)

        # 障害物エリア (確率が高いほど鮮やかな赤橙)
        occ_intensity = ((prob[occupied_mask] - 0.55) / 0.45) * 155 + 100
        grid_bgr[occupied_mask, 0] = (occ_intensity * 0.2).astype(np.uint8)
        grid_bgr[occupied_mask, 1] = (occ_intensity * 0.4).astype(np.uint8)
        grid_bgr[occupied_mask, 2] = occ_intensity.astype(np.uint8)

        # ニアレストネイバーで拡大 (ドットグリッド感を保持)
        map_scaled = cv2.resize(
            grid_bgr,
            (self.map_view_w, self.map_view_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # 1メートル（20セル）ごとの薄い補助線グリッド
        grid_step_px = int(round(1.0 / self.map.cell_size * self.scale))
        origin_px_x = self.map.origin_x * self.scale
        origin_px_y = self.map.origin_y * self.scale

        # 縦線
        for x in range(origin_px_x % grid_step_px, self.map_view_w, grid_step_px):
            cv2.line(map_scaled, (x, 0), (x, self.map_view_h), (60, 60, 60), 1)
        # 横線
        for y in range(origin_px_y % grid_step_px, self.map_view_h, grid_step_px):
            cv2.line(map_scaled, (0, y), (self.map_view_w, y), (60, 60, 60), 1)

        # 原点軸（十字線）
        cv2.line(
            map_scaled,
            (origin_px_x, 0),
            (origin_px_x, self.map_view_h),
            (80, 80, 90),
            1,
        )
        cv2.line(
            map_scaled,
            (0, origin_px_y),
            (self.map_view_w, origin_px_y),
            (80, 80, 90),
            1,
        )

        return map_scaled

    def _draw_trajectory(self, img: np.ndarray) -> None:
        """走行軌跡（シアンの破線）を描画"""
        if len(self.map.path) < 2:
            return

        points = [
            (pt[0] * self.scale + self.scale // 2, pt[1] * self.scale + self.scale // 2)
            for pt in self.map.path
        ]
        pts_arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_arr], isClosed=False, color=(240, 200, 50), thickness=2)

    def _draw_ultrasonic_beam(self, img: np.ndarray, distance_cm: float) -> None:
        """超音波センサーの検知範囲コーン（扇形）を描画"""
        curr_grid = self.map.world_to_grid(self.map.robot_x, self.map.robot_y)
        if not curr_grid:
            return

        rx = curr_grid[0] * self.scale + self.scale // 2
        ry = curr_grid[1] * self.scale + self.scale // 2

        beam_dist_m = min(1.0, distance_cm / 100.0 if distance_cm > 0 else 1.0)
        dist_px = int(round((beam_dist_m / self.map.cell_size) * self.scale))

        th = self.map.robot_theta
        beam_half_fov = math.radians(15)  # 超音波指向角 ±15度

        # コーンの先端2点
        x_left = int(round(rx + dist_px * math.cos(th - beam_half_fov)))
        y_left = int(round(ry - dist_px * math.sin(th - beam_half_fov)))
        x_right = int(round(rx + dist_px * math.cos(th + beam_half_fov)))
        y_right = int(round(ry - dist_px * math.sin(th + beam_half_fov)))

        cone_pts = np.array([[rx, ry], [x_left, y_left], [x_right, y_right]], dtype=np.int32)
        # 半透明オーバーレイ
        overlay = img.copy()
        color = (0, 165, 255) if distance_cm < 20.0 else (50, 220, 220)
        cv2.fillPoly(overlay, [cone_pts], color)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    def _draw_robot(self, img: np.ndarray) -> None:
        """自車アイコン（円＋進行方向矢印）を描画"""
        curr_grid = self.map.world_to_grid(self.map.robot_x, self.map.robot_y)
        if not curr_grid:
            return

        rx = curr_grid[0] * self.scale + self.scale // 2
        ry = curr_grid[1] * self.scale + self.scale // 2

        # 自車ベース円
        cv2.circle(img, (rx, ry), 8, (255, 120, 0), -1)  # 青
        cv2.circle(img, (rx, ry), 9, (255, 255, 255), 1)

        # 進行方向アロー
        arrow_len = 18
        th = self.map.robot_theta
        tip_x = int(round(rx + arrow_len * math.cos(th)))
        tip_y = int(round(ry - arrow_len * math.sin(th)))
        cv2.arrowedLine(
            img, (rx, ry), (tip_x, tip_y), (0, 0, 255), 2, tipLength=0.35
        )

    def _render_sidebar(
        self,
        canvas: np.ndarray,
        camera_frame_bgr: Optional[np.ndarray],
        speed: int,
        steer_angle: int,
        ultrasonic_dist: float,
        motion_value: float,
    ) -> None:
        """右側テレメトリHUDサイドバーの描画"""
        x_offset = self.map_view_w + 15
        y = 25

        # 境界縦線
        cv2.line(
            canvas,
            (self.map_view_w, 0),
            (self.map_view_w, self.canvas_h),
            (70, 70, 70),
            2,
        )

        # タイトル
        cv2.putText(
            canvas,
            "PiCar-X Telemetry",
            (x_offset, y),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
        )
        y += 25

        # ステータスバッジ
        cv2.rectangle(
            canvas,
            (x_offset, y),
            (x_offset + self.sidebar_w - 30, y + 28),
            self.last_status_color,
            -1,
        )
        cv2.putText(
            canvas,
            f" {self.last_status}",
            (x_offset + 5, y + 20),
            cv2.FONT_HERSHEY_DUPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
        y += 45

        # カメラ映像 (ワイプ表示)
        if camera_frame_bgr is not None and camera_frame_bgr.size > 0:
            cam_w = max(10, self.sidebar_w - 30)
            cam_h = int(round(cam_w * (3 / 4)))  # 4:3
            # キャンバス下端からはみ出ないよう調整
            max_avail_h = max(10, self.canvas_h - y - 180)
            if cam_h > max_avail_h:
                cam_h = max_avail_h
                cam_w = int(round(cam_h * (4 / 3)))

            resized_cam = cv2.resize(camera_frame_bgr, (cam_w, cam_h))
            canvas[y:y + cam_h, x_offset:x_offset + cam_w] = resized_cam
            cv2.rectangle(
                canvas,
                (x_offset, y),
                (x_offset + cam_w, y + cam_h),
                (120, 120, 120),
                1,
            )
            y += cam_h + 20
        else:
            y += 10

        # 計測データ一覧
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        color_white = (220, 220, 220)
        line_height = 22

        total_area = self.map.width * self.map.height * (self.map.cell_size ** 2)
        explored_sqm = self.map.get_explored_ratio() * total_area

        info_items = [
            ("Speed", f"{speed} (Min: {self.map.wheelbase * 100:.0f}cm WB)"),
            ("Steering", f"{steer_angle:+03d}°"),
            (
                "Obstacle",
                f"{ultrasonic_dist:.1f} cm" if ultrasonic_dist < 900 else "Clear",
            ),
            ("Flow Motion", f"{motion_value:04.1f}"),
            ("Position X", f"{self.map.robot_x:+.2f} m"),
            ("Position Y", f"{self.map.robot_y:+.2f} m"),
            ("Heading", f"{math.degrees(self.map.robot_theta):.1f}°"),
            ("Explored Area", f"{explored_sqm:.1f} m²"),
        ]

        for label, val in info_items:
            cv2.putText(canvas, f"{label}:", (x_offset, y), font, font_scale, (160, 160, 160), 1)
            cv2.putText(
                canvas,
                val,
                (x_offset + 120, y),
                font,
                font_scale,
                color_white,
                1,
            )
            y += line_height

        y += 15
        # 操作ヘルプ
        cv2.line(
            canvas,
            (x_offset, y),
            (x_offset + self.sidebar_w - 30, y),
            (60, 60, 60),
            1,
        )
        y += 20
        cv2.putText(canvas, "[C] Clear Map", (x_offset, y), font, 0.4, (140, 140, 140), 1)
        y += 18
        cv2.putText(canvas, "[S] Save Map Screenshot", (x_offset, y), font, 0.4, (140, 140, 140), 1)

    def show(self, canvas: np.ndarray, wait_ms: int = 1) -> int:
        """ウィンドウに表示し、キー入力を返す"""
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("c") or key == ord("C"):
            self.map.clear()
        elif key == ord("s") or key == ord("S"):
            save_dir = Path("data/maps")
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = save_dir / f"map_{int(time.time())}.png"
            cv2.imwrite(str(filename), canvas)
            print(f"\n[MAP SAVED] Saved screenshot to: {filename}")
        return key

    def close(self) -> None:
        """ウィンドウを破棄"""
        cv2.destroyAllWindows()
