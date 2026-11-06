import asyncio
import threading
from typing import Iterator, List, Optional

import httpx

from src.config import settings
from src.models import (
    MAX_TTL_MS,
    MIN_TTL_MS,
    CameraSnapshotResponse,
    DriveDirection,
    EventsResponse,
    GrayscaleResponse,
    ListenResponse,
    MoveCommand,
    RobotStatus,
    SpeakCommand,
    SteerCommand,
    UltrasonicResponse,
)
from src.picarx_client.camera_stream import LatestFrameStream

# 長い移動を分割して送る時の更新間隔。1回の通信の遅れで止まらない程度に短くする。
DRIVE_RENEW_MS = 200
DRIVE_RENEW_TTL_MS = 500
# `drive_for` で1回に走れる時間の上限。LLM などの誤った値で走り続けないようにする
MAX_DRIVE_MS = 5000
# カメラの起動を待てるよう、配信の読み取りは制御の通信より長く待つ
STREAM_READ_TIMEOUT_SEC = 5.0


class PicarxClient:
    """Raspberry Pi (Rust APIサーバー) と通信する非同期クライアント"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        api_token: Optional[str] = None,
    ):
        self.base_url = base_url or settings.robot_base_url
        self.timeout = timeout or settings.request_timeout_sec
        token = settings.robot_api_token if api_token is None else api_token
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        # 走行ループで毎回接続し直すと遅くなるので、同期呼び出しでは接続を再利用する
        self._sync_client: Optional[httpx.Client] = None
        self._sync_client_lock = threading.Lock()

    def _sync(self) -> httpx.Client:
        with self._sync_client_lock:
            if self._sync_client is None:
                self._sync_client = httpx.Client(timeout=self.timeout, headers=self._headers)
            return self._sync_client

    async def move(self, direction: DriveDirection, speed: int = 30, ttl_ms: int = 500) -> bool:
        """走行指示を送信する。`ttl_ms` 以内に次の指示がなければ Pi が停止する。"""
        cmd = MoveCommand(direction=direction, speed=speed, ttl_ms=ttl_ms)
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/drive/move", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def drive_for(self, direction: DriveDirection, speed: int, duration_ms: int) -> bool:
        """TTL を更新しながら `duration_ms` だけ走る。

        途中で Pi に拒否された場合や通信が切れた場合は、更新が止まるので車両も止まる。
        """
        if not 0 < duration_ms <= MAX_DRIVE_MS:
            raise ValueError(f"duration_msは1から{MAX_DRIVE_MS}の間である必要があります")
        if duration_ms <= MAX_TTL_MS:
            return await self.move(direction, speed, ttl_ms=max(duration_ms, MIN_TTL_MS))

        loop = asyncio.get_running_loop()
        end = loop.time() + duration_ms / 1000.0
        while True:
            remaining_ms = int((end - loop.time()) * 1000)
            if remaining_ms < MIN_TTL_MS:
                return True
            ttl_ms = min(DRIVE_RENEW_TTL_MS, remaining_ms)
            if not await self.move(direction, speed, ttl_ms=ttl_ms):
                return False
            await asyncio.sleep(min(DRIVE_RENEW_MS, remaining_ms) / 1000.0)

    async def steer(self, angle: int) -> bool:
        """ステアリング角度を変更"""
        cmd = SteerCommand(angle=angle)
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/drive/steer", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def stop(self) -> bool:
        """通常停止を指示する。以後の走行は拒否されない。"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/drive/stop")
            resp.raise_for_status()
            return resp.status_code == 200

    async def reset_emergency_stop(self) -> bool:
        """緊急停止を解除する"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/drive/emergency_stop/reset")
            resp.raise_for_status()
            return resp.status_code == 200

    async def calibrate_floor(self) -> bool:
        """今いる床を基準に段差の閾値を校正する。失敗時は例外を送出する。"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/safety/calibrate")
            resp.raise_for_status()
            return True

    async def get_status(self) -> RobotStatus:
        """電池、センサー、走行モードなど Pi の状態を取得"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(f"{self.base_url}/api/status")
            resp.raise_for_status()
            return RobotStatus.model_validate(resp.json())

    async def get_events(self, after: int = 0) -> EventsResponse:
        """`after` より新しい安全イベントを取得"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(f"{self.base_url}/api/events", params={"after": after})
            resp.raise_for_status()
            return EventsResponse.model_validate(resp.json())

    async def get_ultrasonic(self) -> float:
        """超音波センサーによる前方障害物距離(cm)を取得。不明な時は例外を送出する。"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(f"{self.base_url}/api/sensor/ultrasonic")
            resp.raise_for_status()
            data = UltrasonicResponse.model_validate(resp.json())
            return data.distance_cm

    async def get_grayscale(self) -> List[int]:
        """グレースケールセンサー値を取得。不明な時は例外を送出する。"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
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
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/audio/speak", json=cmd.model_dump())
            resp.raise_for_status()
            return resp.status_code == 200

    async def stop_speech(self) -> bool:
        """Pi で再生中と生成中の発話を止める"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.post(f"{self.base_url}/api/audio/stop")
            resp.raise_for_status()
            return True

    async def listen(self) -> str:
        """
        Raspberry Pi 側でマイク録音・STTされたテキストを取得。
        """
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(f"{self.base_url}/api/audio/listen")
            resp.raise_for_status()
            data = ListenResponse.model_validate(resp.json())
            return data.text

    async def get_camera_snapshot(self) -> CameraSnapshotResponse:
        """カメラの最新フレームを取得。解像度と画質は Pi 側で固定している。"""
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            resp = await client.get(f"{self.base_url}/api/camera/snapshot")
            resp.raise_for_status()
            return CameraSnapshotResponse.model_validate(resp.json())

    # --- 同期呼び出し用ラッパー (QThread / CLI / 同期コンテキストで再利用) ---

    def move_sync(self, direction: DriveDirection, speed: int = 30, ttl_ms: int = 500) -> bool:
        """走行指示を送信する。Pi に拒否された時は `False` を返す。"""
        cmd = MoveCommand(direction=direction, speed=speed, ttl_ms=ttl_ms)
        resp = self._sync().post(f"{self.base_url}/api/drive/move", json=cmd.model_dump())
        return resp.status_code == 200

    def steer_sync(self, angle: int) -> bool:
        cmd = SteerCommand(angle=angle)
        resp = self._sync().post(f"{self.base_url}/api/drive/steer", json=cmd.model_dump())
        return resp.status_code == 200

    def stop_sync(self) -> bool:
        resp = self._sync().post(f"{self.base_url}/api/drive/stop")
        return resp.status_code == 200

    def stop_speech_sync(self) -> bool:
        """発話を止める。共有の接続に左右されないよう、毎回新しい接続で送る。"""
        with httpx.Client(timeout=self.timeout, headers=self._headers) as client:
            resp = client.post(f"{self.base_url}/api/audio/stop")
            resp.raise_for_status()
            return True

    def emergency_stop_sync(self) -> bool:
        """緊急停止を指示する。解除されるまで走行は拒否される。失敗時は例外を送出する。

        共有の接続に左右されないよう、毎回新しい接続で送る。
        """
        with httpx.Client(timeout=self.timeout, headers=self._headers) as client:
            resp = client.post(f"{self.base_url}/api/drive/emergency_stop")
            resp.raise_for_status()
            return True

    def reset_emergency_stop_sync(self) -> bool:
        """緊急停止を解除する。失敗時は例外を送出する。"""
        resp = self._sync().post(f"{self.base_url}/api/drive/emergency_stop/reset")
        resp.raise_for_status()
        return True

    def calibrate_floor_sync(self) -> bool:
        """今いる床を基準に段差の閾値を校正する。失敗時は例外を送出する。"""
        resp = self._sync().post(f"{self.base_url}/api/safety/calibrate")
        resp.raise_for_status()
        return True

    def get_status_sync(self) -> RobotStatus:
        resp = self._sync().get(f"{self.base_url}/api/status")
        resp.raise_for_status()
        return RobotStatus.model_validate(resp.json())

    def get_events_sync(self, after: int = 0) -> EventsResponse:
        resp = self._sync().get(f"{self.base_url}/api/events", params={"after": after})
        resp.raise_for_status()
        return EventsResponse.model_validate(resp.json())

    def get_ultrasonic_sync(self) -> float:
        resp = self._sync().get(f"{self.base_url}/api/sensor/ultrasonic")
        resp.raise_for_status()
        return UltrasonicResponse.model_validate(resp.json()).distance_cm

    def get_grayscale_sync(self) -> List[int]:
        resp = self._sync().get(f"{self.base_url}/api/sensor/grayscale")
        resp.raise_for_status()
        data = GrayscaleResponse.model_validate(resp.json())
        return [data.left, data.middle, data.right]

    def get_camera_snapshot_sync(self) -> CameraSnapshotResponse:
        resp = self._sync().get(f"{self.base_url}/api/camera/snapshot")
        resp.raise_for_status()
        return CameraSnapshotResponse.model_validate(resp.json())

    def _iter_camera_chunks(self) -> Iterator[bytes]:
        timeout = httpx.Timeout(self.timeout, read=STREAM_READ_TIMEOUT_SEC)
        with httpx.Client(timeout=timeout, headers=self._headers) as client:
            with client.stream("GET", f"{self.base_url}/api/camera/stream") as resp:
                resp.raise_for_status()
                yield from resp.iter_bytes()

    def open_camera_stream(self) -> LatestFrameStream:
        """カメラの常時配信を受信する。`with` で使うか `start()` を呼ぶ。"""
        return LatestFrameStream(self._iter_camera_chunks)
