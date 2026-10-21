from typing import List, Optional

import httpx

from src.config import settings
from src.models import (
    CameraSnapshotResponse,
    DriveDirection,
    GrayscaleResponse,
    ListenResponse,
    MoveCommand,
    SpeakCommand,
    SteerCommand,
    UltrasonicResponse,
)


class PicarxClient:
    """Raspberry Pi (Rust APIサーバー) と通信する非同期クライアント"""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = base_url or settings.robot_base_url
        self.timeout = timeout or settings.request_timeout_sec

    async def move(
        self, direction: DriveDirection, speed: int = 30, duration_ms: int = 500
    ) -> bool:
        """車両の移動指示を送信"""
        cmd = MoveCommand(direction=direction, speed=speed, duration_ms=duration_ms)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/drive/move", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def steer(self, angle: int) -> bool:
        """ステアリング角度を変更"""
        cmd = SteerCommand(angle=angle)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/drive/steer", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def stop(self) -> bool:
        """緊急停止指示を送信"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/drive/stop")
            resp.raise_for_status()
            return resp.status_code == 200

    async def get_ultrasonic(self) -> float:
        """超音波センサーによる前方障害物距離(cm)を取得"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/sensor/ultrasonic")
            resp.raise_for_status()
            data = UltrasonicResponse.model_validate(resp.json())
            return data.distance_cm

    async def get_grayscale(self) -> List[int]:
        """グレースケールセンサー値を取得"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/sensor/grayscale")
            resp.raise_for_status()
            data = GrayscaleResponse.model_validate(resp.json())
            return [data.left, data.middle, data.right]

    async def speak(self, text: str) -> bool:
        """
        Raspberry Pi 側でテキストを音声合成(TTS)して再生させる。
        ネットワーク上には重い音声データではなく、テキストのみを送信。
        """
        cmd = SpeakCommand(text=text)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/audio/speak", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def listen(self) -> str:
        """
        Raspberry Pi 側でマイク録音・STTされたテキストを取得。
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/audio/listen")
            resp.raise_for_status()
            data = ListenResponse.model_validate(resp.json())
            return data.text

    async def get_camera_snapshot(
        self, width: int = 320, quality: int = 70
    ) -> CameraSnapshotResponse:
        """
        オンデマンドでカメラから軽量圧縮JPEG画像を取得。
        """
        params = {"width": width, "quality": quality}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/api/camera/snapshot", params=params)
            resp.raise_for_status()
            return CameraSnapshotResponse.model_validate(resp.json())

    # --- 同期呼び出し用ラッパー (QThread / CLI / 同期コンテキストで再利用) ---

    def move_sync(
        self, direction: DriveDirection, speed: int = 30, duration_ms: int = 500
    ) -> bool:
        cmd = MoveCommand(direction=direction, speed=speed, duration_ms=duration_ms)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/drive/move", json=cmd.model_dump())
            return resp.status_code == 200

    def steer_sync(self, angle: int) -> bool:
        cmd = SteerCommand(angle=angle)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/drive/steer", json=cmd.model_dump())
            return resp.status_code == 200

    def stop_sync(self) -> bool:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/drive/stop")
            return resp.status_code == 200

    def get_ultrasonic_sync(self) -> float:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/api/sensor/ultrasonic")
            data = UltrasonicResponse.model_validate(resp.json())
            return data.distance_cm

    def get_camera_snapshot_sync(
        self, width: int = 320, quality: int = 70
    ) -> CameraSnapshotResponse:
        params = {"width": width, "quality": quality}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(f"{self.base_url}/api/camera/snapshot", params=params)
            return CameraSnapshotResponse.model_validate(resp.json())

