from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

# Pi 側の `api/handlers.rs` の範囲と揃える
MIN_TTL_MS = 50
MAX_TTL_MS = 1000


class DriveDirection(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    STOP = "stop"


class MoveCommand(BaseModel):
    direction: DriveDirection
    speed: int = Field(default=30, ge=0, le=100, description="走行速度 (0-100)")
    ttl_ms: int = Field(
        default=500,
        ge=MIN_TTL_MS,
        le=MAX_TTL_MS,
        description="この時間内に次の指示が届かなければ Pi が停止する(ミリ秒)",
    )


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
    seq: int = 0
    age_ms: int = 0


class SafetyReason(str, Enum):
    """Pi が停止または走行を拒否した理由"""

    EMERGENCY_STOP = "emergency_stop"
    OBSTACLE = "obstacle"
    CLIFF = "cliff"
    DISTANCE_UNKNOWN = "distance_unknown"
    FLOOR_UNKNOWN = "floor_unknown"
    CLIFF_UNCALIBRATED = "cliff_uncalibrated"
    REFLEX_STALLED = "reflex_stalled"
    TTL_EXPIRED = "ttl_expired"


class DistanceStatus(BaseModel):
    cm: Optional[float] = Field(default=None, description="古い、または読めない時は None")
    age_ms: Optional[int] = None
    far: bool = Field(default=False, description="反射が戻らず、測定範囲より遠いと判断している")


class FloorStatus(BaseModel):
    values: Optional[List[int]] = Field(default=None, description="左、中央、右")
    age_ms: Optional[int] = None
    cliff: Optional[List[bool]] = None


class BatteryStatus(BaseModel):
    voltage: Optional[float] = None
    percent: Optional[int] = None
    age_ms: Optional[int] = None


class SensorsStatus(BaseModel):
    distance: DistanceStatus
    floor: FloorStatus
    battery: BatteryStatus


class MotionStatus(BaseModel):
    direction: DriveDirection
    speed: int
    ttl_remaining_ms: int


class BlockedStatus(BaseModel):
    forward: Optional[SafetyReason] = None
    backward: Optional[SafetyReason] = None


class CliffCalibration(BaseModel):
    baseline: List[int]
    thresholds: List[int]


class CliffStatus(BaseModel):
    guard_enabled: bool
    calibration: Optional[CliffCalibration] = None


class ReflexStatus(BaseModel):
    hz: float
    max_interval_ms: float
    last_tick_age_ms: Optional[int] = None


class CameraStatus(BaseModel):
    running: bool
    fps: float
    frames: int
    frame_age_ms: Optional[int] = None
    error: Optional[str] = None


class AudioStatus(BaseModel):
    speaking: bool = False


class RobotStatus(BaseModel):
    """`/api/status` の応答"""

    mode: str
    emergency_stop: bool
    motion: Optional[MotionStatus] = None
    steer_angle: int
    blocked: BlockedStatus
    sensors: SensorsStatus
    cliff: CliffStatus
    reflex: ReflexStatus
    camera: CameraStatus
    # 古い Pi のバイナリは返さない
    audio: AudioStatus = AudioStatus()
    latest_event_id: int


class SafetyEvent(BaseModel):
    id: int
    at_unix_ms: int
    kind: str
    reason: Optional[str] = None
    direction: Optional[DriveDirection] = None
    distance_cm: Optional[float] = None
    grayscale: Optional[List[int]] = None
    stop_latency_ms: Optional[float] = None
    detail: Optional[str] = None


class EventsResponse(BaseModel):
    events: List[SafetyEvent]
    latest_event_id: int
