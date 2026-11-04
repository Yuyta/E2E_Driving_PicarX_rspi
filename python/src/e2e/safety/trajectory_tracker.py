"""
Phase 3.4: Trajectory Tracker & Breadcrumb Safety
直前数秒間の走行軌跡（車速・操舵角・時間）を常時記録し、
後方にセンサーのない PiCar-X が「安全に通ってきたルート」を正確に逆再生して
十分な距離まで安全に後退・退避できるようにするモジュール。
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Pi の `/api/drive/move` が受け付ける最短の TTL
MIN_COMMAND_MS = 50


@dataclass
class TrajectoryPoint:
    """走行軌跡の1ステップ"""
    speed: int
    steer_angle: int
    duration_ms: int
    timestamp: float


class TrajectoryTracker:
    """直前軌跡メモリ（パンくずリスト）および安全後退シーケンサ"""

    def __init__(
        self,
        max_history_points: int = 100,
        default_reverse_duration_ms: int = 1200,
        reverse_speed: int = 35,
    ):
        """
        Args:
            max_history_points: 保持する最大ステップ数 (約20Hzで100点 = 5秒分)
            default_reverse_duration_ms: 回避時の目標後退時間 (ミリ秒、1.2秒推奨)
            reverse_speed: 後退時の車速 (0〜100)
        """
        self.max_history_points = max_history_points
        self.default_reverse_duration_ms = max(500, int(default_reverse_duration_ms))
        self.reverse_speed = max(10, min(100, int(reverse_speed)))
        self.history: deque[TrajectoryPoint] = deque(maxlen=max_history_points)

    def record(
        self,
        speed: int,
        steer_angle: int,
        duration_ms: int,
        timestamp: float,
    ) -> None:
        """前進走行時の1ステップを記録"""
        # 実際に前進していた場合のみ記録
        if speed > 0 and duration_ms > 0:
            self.history.append(
                TrajectoryPoint(
                    speed=speed,
                    steer_angle=steer_angle,
                    duration_ms=duration_ms,
                    timestamp=timestamp,
                )
            )

    def clear(self) -> None:
        """履歴をクリア"""
        self.history.clear()

    def get_reverse_sequence(
        self, target_duration_ms: Optional[int] = None
    ) -> List[Tuple[int, int, int]]:
        """
        直前に走行してきた安全な軌跡を逆再生するための後退シーケンスを生成。
        後方にセンサーがないため、記録にない範囲へは後退しない。
        履歴が足りなければ短いシーケンスを返し、空なら空のリストを返す。

        Args:
            target_duration_ms: 後退したい合計時間 (省略時は default_reverse_duration_ms)

        Returns:
            List of (steer_angle, reverse_speed, step_duration_ms)
        """
        target_ms = (
            target_duration_ms
            if target_duration_ms is not None
            else self.default_reverse_duration_ms
        )

        sequence: List[Tuple[int, int, int]] = []
        accumulated_ms = 0

        # 最新（直前）から過去へ向かって逆順に取り出す
        for pt in reversed(self.history):
            if accumulated_ms >= target_ms:
                break

            remaining_ms = target_ms - accumulated_ms
            step_ms = min(pt.duration_ms, remaining_ms)
            if step_ms < MIN_COMMAND_MS:
                # Pi が受け付けない短さの指令は作らない
                accumulated_ms += step_ms
                continue

            # アッカーマンステアリングでは、前進時と同じ舵角で後退することで同じ軌跡を戻る
            sequence.append((pt.steer_angle, self.reverse_speed, step_ms))
            accumulated_ms += step_ms

        return sequence

    def consume_reverse(self, duration_ms: int) -> None:
        """実際に後退した分を履歴から取り除く。残りは次の後退に使える。"""
        remaining_ms = duration_ms
        while self.history and remaining_ms > 0:
            newest = self.history[-1]
            if newest.duration_ms <= remaining_ms:
                remaining_ms -= newest.duration_ms
                self.history.pop()
            else:
                newest.duration_ms -= remaining_ms
                remaining_ms = 0

    def get_escape_turn(
        self, last_cliff_sensors: Optional[Tuple[bool, bool, bool]] = None
    ) -> Tuple[int, int, int]:
        """
        後退完了後に、障害物・段差から車首を横へそらすための回避旋回アクション。

        Args:
            last_cliff_sensors: (left_cliff, mid_cliff, right_cliff)

        Returns:
            (steer_angle, forward_speed, duration_ms)
        """
        if last_cliff_sensors:
            left_cliff, _, right_cliff = last_cliff_sensors
            if left_cliff and not right_cliff:
                # 左が危険だった → 右へ旋回前進して離れる
                return (35, self.reverse_speed, 400)
            elif right_cliff and not left_cliff:
                # 右が危険だった → 左へ旋回前進して離れる
                return (-35, self.reverse_speed, 400)

        # 履歴から判断: 直前が右曲がりなら左へ、左曲がりなら右へターン
        avg_steer = 0.0
        if self.history:
            recent = list(self.history)[-10:]
            avg_steer = sum(p.steer_angle for p in recent) / len(recent)

        turn_steer = -35 if avg_steer >= 0 else 35
        return (turn_steer, self.reverse_speed, 400)
