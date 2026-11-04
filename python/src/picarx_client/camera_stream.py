"""Pi の MJPEG 配信を受信し、常に最新のフレームだけを渡す。"""

import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Optional

import httpx

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
MAX_BUFFER_BYTES = 4 * 1024 * 1024
RECONNECT_DELAY_SEC = 0.5


def iter_jpeg_frames(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """バイト列から JPEG を切り出す。JPEG の境界はマーカーで判定する。"""
    buffer = bytearray()
    for chunk in chunks:
        buffer.extend(chunk)
        while True:
            start = buffer.find(SOI)
            if start < 0:
                # 末尾の 0xFF は次の読み込みで SOI になる可能性がある
                del buffer[: -1 if buffer.endswith(b"\xff") else len(buffer)]
                break
            end = buffer.find(EOI, start + len(SOI))
            if end < 0:
                del buffer[:start]
                if len(buffer) > MAX_BUFFER_BYTES:
                    buffer.clear()
                break
            frame_end = end + len(EOI)
            yield bytes(buffer[start:frame_end])
            del buffer[:frame_end]


@dataclass(frozen=True)
class StreamFrame:
    jpeg: bytes
    seq: int
    received_at: float


class LatestFrameStream:
    """別スレッドで受信を続ける。処理が遅れても古いフレームは溜まらない。"""

    def __init__(
        self,
        open_chunks: Callable[[], Iterable[bytes]],
        clock: Callable[[], float] = time.monotonic,
    ):
        self._open_chunks = open_chunks
        self._clock = clock
        self._condition = threading.Condition()
        self._latest: Optional[StreamFrame] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_error: Optional[str] = None

    def start(self) -> "LatestFrameStream":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="camera-stream")
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def __enter__(self) -> "LatestFrameStream":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.stop()

    def _publish(self, jpeg: bytes) -> None:
        with self._condition:
            seq = self._latest.seq + 1 if self._latest else 1
            self._latest = StreamFrame(jpeg=jpeg, seq=seq, received_at=self._clock())
            self._condition.notify_all()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for jpeg in iter_jpeg_frames(self._open_chunks()):
                    if self._stop.is_set():
                        return
                    self._publish(jpeg)
                self.last_error = "配信が終了しました"
            except httpx.HTTPError as exc:
                self.last_error = str(exc) or type(exc).__name__
            self._stop.wait(RECONNECT_DELAY_SEC)

    def wait_for_frame(self, after_seq: int = 0, timeout: float = 0.5) -> Optional[StreamFrame]:
        """`after_seq` より新しいフレームを待つ。届かなければ `None` を返す。"""
        deadline = self._clock() + timeout
        with self._condition:
            while self._latest is None or self._latest.seq <= after_seq:
                remaining = deadline - self._clock()
                if remaining <= 0 or self._stop.is_set():
                    return None
                self._condition.wait(remaining)
            return self._latest
