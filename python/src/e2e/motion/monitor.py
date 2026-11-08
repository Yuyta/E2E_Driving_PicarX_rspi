"""
走行中の動きの監視: 指令の状態を追い、画像と超音波から食い違いを検出する

収集 GUI と自律走行の両方から使う。食い違いが続いたら一度だけ知らせ、解消するまで
同じ食い違いでは知らせない。
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    from .checker import FLAG_TEXT, CheckerConfig, CheckResult, Command, MotionChecker
    from .estimator import MotionSample, MotionTracker
except ImportError:
    from src.e2e.motion.checker import (
        FLAG_TEXT,
        CheckerConfig,
        CheckResult,
        Command,
        MotionChecker,
    )
    from src.e2e.motion.estimator import MotionSample, MotionTracker


class CommandState:
    """送った走行指令を覚え、ある時刻に有効な指令を返す

    走行指令には期限（TTL）があり、期限が切れると Pi が止める。`throttle` を記録するだけでは
    止まった後も前進のままに見える（LRN-6）ので、期限を含めて追う。
    """

    def __init__(self, now: float = 0.0):
        self._speed = 0
        self._steering = 0.0
        self._expires: Optional[float] = None
        self._changed_at = now
        self._camera_until = 0.0

    def move(self, speed: int, ttl_s: float, now: float) -> None:
        """前進は正、後退は負の速度。同じ速度の繰り返しは指令の変化とみなさない"""
        if self._effective_speed(now) != speed:
            self._changed_at = now
        self._speed = speed
        self._expires = now + ttl_s

    def stop(self, now: float) -> None:
        if self._effective_speed(now) != 0:
            self._changed_at = now
        self._speed = 0
        self._expires = None

    def steer(self, steering_deg: float, now: float) -> None:
        if steering_deg != self._steering:
            self._changed_at = now
        self._steering = float(steering_deg)

    def camera_moved(self, now: float, settle_s: float = 0.5) -> None:
        """カメラのパンやチルトを動かした。止まるまでの間は景色の動きを判定しない"""
        self._camera_until = now + settle_s

    def _effective_speed(self, now: float) -> int:
        if self._speed and self._expires is not None and now >= self._expires:
            return 0
        return self._speed

    def at(self, now: float) -> Command:
        speed = self._effective_speed(now)
        changed_at = self._changed_at
        if self._speed and speed == 0 and self._expires is not None:
            # 期限切れで止まった時刻から数える
            changed_at = max(changed_at, self._expires)
        return Command(
            speed=speed,
            steering_deg=self._steering,
            since_s=max(0.0, now - changed_at),
            camera_moving=now < self._camera_until,
        )


@dataclass
class MonitorUpdate:
    command: Command
    sample: MotionSample
    check: CheckResult
    newly_sustained: Optional[str]  # 今回初めて sustain_s 続いた食い違い

    def to_record(self) -> dict:
        """収集データの1フレーム分に加える項目"""
        return {
            "command": self.command.to_record(),
            "motion": self.sample.to_record(),
            "motion_check": self.check.to_record(),
        }


class MotionMonitor:
    def __init__(
        self,
        tracker: Optional[MotionTracker] = None,
        config: Optional[CheckerConfig] = None,
        on_sustained: Optional[Callable[[str, MonitorUpdate], None]] = None,
    ):
        self.tracker = tracker or MotionTracker()
        self.checker = MotionChecker(config)
        self.on_sustained = on_sustained
        self._reported: Optional[str] = None

    def reset(self) -> None:
        self.tracker.reset()
        self.checker.reset()
        self._reported = None

    def update(
        self,
        frame: np.ndarray,
        now: float,
        distance_cm: Optional[float],
        command: Command,
    ) -> Optional[MonitorUpdate]:
        sample = self.tracker.update(frame, now, distance_cm)
        if sample is None:
            return None
        check = self.checker.check(command, sample)

        newly = None
        if check.sustained and check.sustained != self._reported:
            newly = check.sustained
        self._reported = check.sustained
        update = MonitorUpdate(command, sample, check, newly)
        if newly and self.on_sustained:
            self.on_sustained(newly, update)
        return update


def describe(flag: str) -> str:
    return FLAG_TEXT.get(flag, flag)


def record_motion_event(flag: str, update: MonitorUpdate) -> Tuple[bool, str]:
    """会話の Agent が「さっき何で止まったの？」に答えられるよう、記憶へ残す"""
    summary = f"走行を止めた: {describe(flag)}"
    try:
        from src.memory.store import get_memory_store

        saved = get_memory_store().add_event("motion_mismatch", summary)
    except Exception as exc:  # 記憶の失敗で走行の停止を妨げない
        return False, f"{summary}（記憶への保存に失敗: {exc}）"
    return saved, summary
