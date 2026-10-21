from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DriveDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    STOP = "stop"


class MoveCommand(BaseModel):
    direction: DriveDirection
    speed: int = Field(default=30, ge=0, le=100, description="走行速度 (0-100)")
    duration_ms: int = Field(default=500, ge=50, le=5000, description="走行時間(ミリ秒)")


class SteerCommand(BaseModel):
    angle: int = Field(default=0, ge=-40, le=40, description="ステアリング角度 (-40〜40)")


class UltrasonicResponse(BaseModel):
    distance_cm: float = Field(description="超音波センサーによる前方障害物距離(cm)")


class GrayscaleResponse(BaseModel):
    left: int
    middle: int
    right: int


class SpeakCommand(BaseModel):
    text: str = Field(description="発話するテキスト（音声データではなく文字列）")


class ListenResponse(BaseModel):
    text: str = Field(description="音声認識されたテキスト")


class CameraSnapshotResponse(BaseModel):
    image_base64: str = Field(description="JPEGエンコードされたBase64画像文字列")
    width: int
    height: int


class RobotStatus(BaseModel):
    distance_cm: Optional[float] = None
    grayscale: Optional[List[int]] = None
    is_moving: bool = False
    current_steering_angle: int = 0
