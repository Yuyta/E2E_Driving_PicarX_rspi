"""
指令と実際の動きの食い違いを検出する（ROADMAP の M3）

次の時に、そのフレームへ印を付ける。
- no_motion: 進む指令なのに、画像も距離も変わらない（空転、引っかかり）
- turn_mismatch: 曲がった量が舵角から想定した量と大きく違う（滑り、舵のズレ）
- uncommanded_motion: 指令していないのに動いている（押された、傾いた床）

判断できない時（特徴点が少ない、超音波が何も捉えていない）は印を付けず、不明として返す。
同じ食い違いが `sustain_s` 続いたら、走行を止めるよう知らせる。
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.models import MIN_MOVING_SPEED

try:
    from .estimator import MotionSample
except ImportError:
    from src.e2e.motion.estimator import MotionSample

NO_MOTION = "no_motion"
TURN_MISMATCH = "turn_mismatch"
UNCOMMANDED_MOTION = "uncommanded_motion"

FLAG_TEXT = {
    NO_MOTION: "進む指令なのに動いていない（空転か引っかかり）",
    TURN_MISMATCH: "曲がり方が舵角と合わない（滑りか舵のズレ）",
    UNCOMMANDED_MOTION: "指令していないのに動いている（押されたか傾いた床）",
}


@dataclass
class Command:
    """そのフレームの時点の走行指令"""

    speed: int  # 前進は正、後退は負、停止は 0（0〜100 の指令値）
    steering_deg: float  # 正は右
    since_s: float  # 指令（速度、向き、舵角）が今の値になってからの時間
    camera_moving: bool = False  # カメラのパンやチルトを動かしている間は景色が動く

    def to_record(self) -> dict:
        return {
            "speed": self.speed,
            "steering_deg": round(self.steering_deg, 2),
            "since_s": round(self.since_s, 3),
            "camera_moving": self.camera_moving,
        }


@dataclass
class CheckerConfig:
    """しきい値は提案値。床に印を付けたコースとわざと空転させる試験で校正する"""

    # これより遅い指令では、動かなくても食い違いとしない（車輪が回らない速度なので）
    min_check_speed: int = MIN_MOVING_SPEED
    settle_s: float = 0.5  # 指令が変わってから判定するまでの時間（加速と惰性の分）
    min_turn_steer_deg: float = 15.0  # これ以上の舵角で曲がる向きを確かめる
    straight_steer_deg: float = 3.0  # これ以下の舵角は直進とみなす
    min_turn_rate_dps: float = 8.0  # 逆向きに曲がったとみなす向きの変化
    straight_max_rate_dps: float = 15.0  # 直進の指令でこれより速く曲がればズレ
    still_cm_s: float = 2.0  # 超音波の距離の変化がこれ未満なら止まっている
    uncommanded_min_inliers: float = 0.6  # 景色全体が動いたとみなす特徴点の割合
    # 曲がり方は、模様の点がこれ以上あり、合う点の割合がこれ以上の時だけ確かめる。
    # 白い壁や床の多い部屋では、点が少ない時の向きの推定がばらつき、直進中のフレームの
    # 半分に誤って印が付いた（2026-11-12）。既定は 0（常に確かめる）
    turn_min_points: int = 0
    turn_min_inliers: float = 0.0
    uncommanded_px_s: float = 8.0  # 指令なしで動いたとみなす画像の動き（ピクセル/秒）
    sustain_s: float = 1.0  # これだけ続いたら走行を止める
    # 速度と舵角から曲がる量を計算するための値。校正するまでは None（向きの符号だけを確かめる）
    speed_cm_s_per_unit: Optional[float] = None
    wheelbase_cm: Optional[float] = None
    turn_ratio_range: Tuple[float, float] = (0.5, 2.0)


@dataclass
class CheckResult:
    flags: List[str] = field(default_factory=list)
    unknown: Optional[str] = None  # 判断できなかった理由
    details: Dict[str, str] = field(default_factory=dict)
    sustained: Optional[str] = None  # sustain_s 以上続いている食い違い

    @property
    def mismatch(self) -> bool:
        return bool(self.flags)

    def to_record(self) -> dict:
        record: dict = {"flags": list(self.flags)}
        if self.unknown:
            record["unknown"] = self.unknown
        if self.details:
            record["details"] = dict(self.details)
        if self.sustained:
            record["sustained"] = self.sustained
        return record


class MotionChecker:
    def __init__(self, config: Optional[CheckerConfig] = None):
        self.config = config or CheckerConfig()
        self._since: Dict[str, float] = {}

    def reset(self) -> None:
        self._since.clear()

    def expected_yaw_rate(self, command: Command) -> Optional[float]:
        """校正済みなら、自転車モデルで想定される向きの変化（度/秒）。未校正なら None"""
        cfg = self.config
        if cfg.speed_cm_s_per_unit is None or cfg.wheelbase_cm is None:
            return None
        speed_cm_s = command.speed * cfg.speed_cm_s_per_unit
        return math.degrees(
            speed_cm_s / cfg.wheelbase_cm * math.tan(math.radians(command.steering_deg))
        )

    def check(self, command: Command, motion: MotionSample) -> CheckResult:
        result = CheckResult()
        cfg = self.config

        if command.camera_moving:
            result.unknown = "camera_moving"
        elif command.since_s < cfg.settle_s:
            result.unknown = "settling"
        elif abs(command.speed) >= cfg.min_check_speed:
            self._check_driving(command, motion, result)
        elif command.speed == 0:
            self._check_stopped(motion, result)
        else:
            result.unknown = "slow_command"

        self._update_sustained(motion.timestamp, result)
        return result

    def _distance_still(self, motion: MotionSample) -> Optional[bool]:
        if motion.approach_cm_s is None:
            return None
        return abs(motion.approach_cm_s) < self.config.still_cm_s

    def _check_driving(self, command: Command, motion: MotionSample, result: CheckResult):
        cfg = self.config
        distance_still = self._distance_still(motion)

        # 1. 動いていない（空転、引っかかり）
        if motion.image_valid:
            if not motion.image_moving and distance_still is not False:
                result.flags.append(NO_MOTION)
                result.details[NO_MOTION] = (
                    f"image {motion.edge_flow_px_s:.1f}px/s, "
                    f"distance {_fmt(motion.approach_cm_s)}cm/s"
                )
                return
        elif distance_still is True:
            result.flags.append(NO_MOTION)
            result.details[NO_MOTION] = f"distance {motion.approach_cm_s:.1f}cm/s, image unknown"
            return
        else:
            result.unknown = motion.image_reason or "image_unknown"
            return

        if not motion.image_moving:
            return

        # 2. 曲がり方（後退では舵角に対して向きの変化が逆になる）
        if motion.points < cfg.turn_min_points or motion.inlier_ratio < cfg.turn_min_inliers:
            result.unknown = "turn_unsure"
            return
        yaw = motion.yaw_rate_dps
        steer = command.steering_deg
        direction = 1.0 if command.speed > 0 else -1.0
        expected = self.expected_yaw_rate(command)

        if expected is not None and abs(steer) >= cfg.min_turn_steer_deg:
            low, high = cfg.turn_ratio_range
            ratio = yaw / expected if expected else math.inf
            if not low <= ratio <= high:
                result.flags.append(TURN_MISMATCH)
                result.details[TURN_MISMATCH] = (
                    f"yaw {yaw:+.1f}dps, expected {expected:+.1f}dps (ratio {ratio:.2f})"
                )
        elif abs(steer) >= cfg.min_turn_steer_deg:
            expected_sign = math.copysign(1.0, steer) * direction
            if yaw * expected_sign < -cfg.min_turn_rate_dps:
                result.flags.append(TURN_MISMATCH)
                result.details[TURN_MISMATCH] = (
                    f"yaw {yaw:+.1f}dps against steering {steer:+.0f}deg"
                )
        elif abs(steer) <= cfg.straight_steer_deg and abs(yaw) > cfg.straight_max_rate_dps:
            result.flags.append(TURN_MISMATCH)
            result.details[TURN_MISMATCH] = f"yaw {yaw:+.1f}dps while steering straight"

    def _check_stopped(self, motion: MotionSample, result: CheckResult):
        cfg = self.config
        if not motion.image_valid:
            result.unknown = motion.image_reason or "image_unknown"
            return
        # 人やペットなど一部だけの動きではなく、景色全体が動いた時だけ印を付ける
        whole_scene = motion.inlier_ratio >= cfg.uncommanded_min_inliers
        if whole_scene and motion.edge_flow_px_s >= cfg.uncommanded_px_s:
            if self._distance_still(motion) is True:
                # 超音波が止まっているなら、カメラだけが動いた可能性が高い
                result.unknown = "image_moved_distance_still"
                return
            result.flags.append(UNCOMMANDED_MOTION)
            result.details[UNCOMMANDED_MOTION] = (
                f"image {motion.edge_flow_px_s:.1f}px/s, yaw {motion.yaw_rate_dps:+.1f}dps"
            )

    def _update_sustained(self, now: float, result: CheckResult):
        for flag in list(self._since):
            if flag not in result.flags:
                del self._since[flag]
        for flag in result.flags:
            self._since.setdefault(flag, now)
        lasting = [
            flag for flag, start in self._since.items() if now - start >= self.config.sustain_s
        ]
        if lasting:
            result.sustained = min(lasting, key=lambda f: self._since[f])


def _fmt(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:+.1f}"
