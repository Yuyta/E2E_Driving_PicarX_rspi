"""
Phase 3.6: Stuck Detector & Auto-Unstuck Behavior
カメラ画像のオプティカル/視覚変化（フレーム間差分）およびモーター動作状態から
車輪の空転・ラグや家具の脚への引っかかり（スタック状態）を自動検知し、
バックや左右の車体振り（Wiggle）を組み合わせて自動脱出するモジュール。
"""

from collections import deque
from typing import Deque, List, Tuple

import cv2
import numpy as np


class StuckDetector:
    """視覚的変化量に基づくスタック（引っかかり・空転）検知および脱出制御器"""

    def __init__(
        self,
        history_frames: int = 8,
        min_motion_threshold: float = 4.0,
        stuck_duration_frames: int = 15,
        min_moving_speed: int = 15,
    ):
        """
        Args:
            history_frames: 変位比較を行う過去フレームの間隔 (約20Hzで8フレーム = 約0.4秒前)
            min_motion_threshold: 移動判定最小フレーム差分平均値 (これ未満で静止判定)
            stuck_duration_frames: スタックと確定する連続静止フレーム数 (約15フレーム = 約1秒)
            min_moving_speed: 移動中と判定する最低モーター速度
        """
        self.history_frames = max(1, history_frames)
        self.min_motion_threshold = float(min_motion_threshold)
        self.stuck_duration_frames = max(3, stuck_duration_frames)
        self.min_moving_speed = min_moving_speed

        # 過去フレームのグレースケールサムネイル (計算軽量化のため 80x60 に縮小)
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=self.history_frames + 1)
        self.consecutive_stuck_count: int = 0
        self.is_stuck_flag: bool = False
        self.last_motion_value: float = 0.0

    def calculate_motion(self, current_frame_bgr: np.ndarray) -> float:
        """
        最新フレームと約0.4秒前のフレームとの視覚的変化量（ピクセル差分平均）を算出。

        Returns:
            float: 変化量 (0.0〜255.0)。静止時は 0.0〜2.0、前進時は通常 10.0 以上。
        """
        if current_frame_bgr is None or current_frame_bgr.size == 0:
            return 0.0

        # 高速化のために 80x60 に縮小してグレースケール化
        small_gray = cv2.cvtColor(
            cv2.resize(current_frame_bgr, (80, 60)), cv2.COLOR_BGR2GRAY
        )

        self.frame_buffer.append(small_gray)

        # 比較対象の過去フレームがまだ溜まっていない場合
        if len(self.frame_buffer) <= self.history_frames:
            return self.min_motion_threshold * 2.0  # 初期状態は動いているとみなす

        past_frame = self.frame_buffer[0]
        diff = cv2.absdiff(small_gray, past_frame)
        motion_value = float(np.mean(diff))
        self.last_motion_value = motion_value
        return motion_value

    def update(
        self,
        current_frame_bgr: np.ndarray,
        motor_speed: int,
    ) -> bool:
        """
        現在のフレームとモーター車速からスタック状態を更新判定。

        Args:
            current_frame_bgr: 最新のカメラ画像
            motor_speed: 現在の指令車速 (0〜100)

        Returns:
            bool: スタック状態（引っかかり）が確定した場合 True
        """
        # モーターが停止または超低速ならスタック判定しない
        if motor_speed < self.min_moving_speed:
            self.consecutive_stuck_count = 0
            self.is_stuck_flag = False
            return False

        motion = self.calculate_motion(current_frame_bgr)

        # 車輪を回しているのに、カメラの景色が動いていない場合
        if motion < self.min_motion_threshold:
            self.consecutive_stuck_count += 1
        else:
            # しっかり動いていればカウンタリセット
            self.consecutive_stuck_count = max(0, self.consecutive_stuck_count - 2)

        if self.consecutive_stuck_count >= self.stuck_duration_frames:
            self.is_stuck_flag = True
            return True

        self.is_stuck_flag = False
        return False

    def reset(self) -> None:
        """脱出完了時などにカウンタと履歴を初期化"""
        self.consecutive_stuck_count = 0
        self.is_stuck_flag = False
        self.frame_buffer.clear()

    @property
    def is_stuck(self) -> bool:
        return self.is_stuck_flag

    def get_unstuck_plan(self, escape_speed: int = 35) -> List[Tuple[str, int, int, int]]:
        """
        引っかかり・スタックから脱出するための複合アクションシーケンスを生成。
        1. 直前軌跡の逆再生バック (障害物から離れる)
        2. 左右の車体振り (Wiggle / 乗り上げや車輪の引っ掛かりを外す)
        3. 迂回ターン (引っかかったポイントを回避して前進)

        Returns:
            List of (action_type, steer_angle, speed, duration_ms)
            action_type: "reverse" | "forward"
        """
        spd = max(20, min(100, int(escape_speed)))

        plan: List[Tuple[str, int, int, int]] = [
            # 1. まずしっかり直線〜緩やかにバック (1000ms)
            ("reverse", 0, spd, 1000),
            # 2. 車体を左に振る (左ステアリング前進 300ms)
            ("forward", -35, spd, 350),
            # 3. 車体を右に振る (右ステアリング前進 300ms)
            ("forward", 35, spd, 350),
            # 4. 少しバックして体勢を整える (500ms)
            ("reverse", 0, spd, 500),
            # 5. 障害物を避ける方向に大きくハンドルを切って脱出 (400ms)
            ("forward", 35, spd, 450),
        ]
        return plan
