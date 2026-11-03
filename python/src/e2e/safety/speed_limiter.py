"""
Phase 3.2: Dynamic Speed Limiter
ステアリング舵角や障害物接近度合いに応じて走行車速を動的に制御する安全モジュール。
家具や壁の角を曲がる際のオーバーステアやスリップ、強い衝突を防ぎ、直進時は最高速で効率よく巡航する。
"""

from typing import Optional


class SpeedLimiter:
    """舵角および前方障害物距離に基づくダイナミック速度制御器"""

    def __init__(
        self,
        base_speed: int = 35,
        min_speed: int = 20,
        curve_factor: float = 1.0,
        safe_distance: float = 15.0,
        slowdown_distance: float = 40.0,
    ):
        """
        Args:
            base_speed: 直進時の基本巡航速度 (0〜100)
            min_speed: 最大減速時の最低速度 (0〜100, base_speed以下)
            curve_factor: 旋回時の減速感度 (1.0 = 線形、>1.0 = 急旋回時により急減速)
            safe_distance: 完全停止する限界距離 (cm)
            slowdown_distance: 障害物接近に伴う減速を開始する距離 (cm)
        """
        if min_speed > base_speed:
            raise ValueError(
                f"min_speed ({min_speed}) cannot be greater than base_speed ({base_speed})"
            )
        if safe_distance >= slowdown_distance:
            raise ValueError("safe_distance must be less than slowdown_distance")

        self.base_speed = max(0, min(100, int(base_speed)))
        self.min_speed = max(0, min(100, int(min_speed)))
        self.curve_factor = max(0.1, float(curve_factor))
        self.safe_distance = float(safe_distance)
        self.slowdown_distance = float(slowdown_distance)

    def calculate_curve_speed(self, steering_norm: float) -> int:
        """
        ステアリング正規化値 (-1.0〜+1.0) に基づく旋回速度の算出。
        直進 (|norm| = 0.0) -> base_speed
        最大舵角 (|norm| >= 1.0) -> min_speed
        """
        abs_steer = min(1.0, abs(steering_norm))
        # 舵角寄与度を計算 (curve_factor適用)
        steer_penalty = min(1.0, (abs_steer ** self.curve_factor))
        speed_drop = (self.base_speed - self.min_speed) * steer_penalty
        calc_speed = self.base_speed - speed_drop
        return int(round(calc_speed))

    def calculate_obstacle_speed(self, obstacle_distance_cm: Optional[float]) -> int:
        """
        前方障害物距離 (cm) に基づく接近減速速度の算出。
        distance >= slowdown_distance -> base_speed (減速なし)
        distance <= safe_distance -> min_speed (完全停止直前のクリープ走行)
        中間 -> 線形補間
        """
        if obstacle_distance_cm is None or obstacle_distance_cm >= self.slowdown_distance:
            return self.base_speed

        if obstacle_distance_cm <= self.safe_distance:
            return self.min_speed

        ratio = (obstacle_distance_cm - self.safe_distance) / (
            self.slowdown_distance - self.safe_distance
        )
        calc_speed = self.min_speed + (self.base_speed - self.min_speed) * ratio
        return int(round(calc_speed))

    def calculate(
        self, steering_norm: float, obstacle_distance_cm: Optional[float] = None
    ) -> int:
        """
        カーブ減速と障害物接近減速の複合適用（最も安全側＝遅い速度を採用）。

        Returns:
            int: 最終的な指令車速 (min_speed 〜 base_speed)
        """
        curve_spd = self.calculate_curve_speed(steering_norm)
        obs_spd = self.calculate_obstacle_speed(obstacle_distance_cm)

        # より遅い（安全な）方を採用
        target_speed = min(curve_spd, obs_spd)
        # 範囲クランプ
        clamped = max(self.min_speed, min(self.base_speed, target_speed))
        return int(clamped)
