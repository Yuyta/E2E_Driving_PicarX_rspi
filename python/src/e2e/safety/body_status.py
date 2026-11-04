"""Pi の `/api/status` を、走行ループが取るべき行動に変換する。

危険の判定と停止は Pi の反射ループが行う。ここでは判定を繰り返さず、
Pi の結果に応じて脱出するか待機するかだけを決める。
"""

from enum import Enum
from typing import Optional, Tuple

from src.models import RobotStatus, SafetyReason


class BodyAction(str, Enum):
    DRIVE = "drive"
    AVOID_OBSTACLE = "avoid_obstacle"
    ESCAPE_CLIFF = "escape_cliff"
    WAIT = "wait"


def decide_action(status: RobotStatus) -> Tuple[BodyAction, Optional[SafetyReason]]:
    """前進できない理由から行動を決める。不明な状態では走らない。"""
    reason = status.blocked.forward
    if reason is None:
        return BodyAction.DRIVE, None
    if reason == SafetyReason.OBSTACLE:
        return BodyAction.AVOID_OBSTACLE, reason
    if reason == SafetyReason.CLIFF:
        return BodyAction.ESCAPE_CLIFF, reason
    return BodyAction.WAIT, reason


def cliff_sensors(status: RobotStatus) -> Optional[Tuple[bool, bool, bool]]:
    cliff = status.sensors.floor.cliff
    return (cliff[0], cliff[1], cliff[2]) if cliff else None
