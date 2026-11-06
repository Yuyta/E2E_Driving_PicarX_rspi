"""PC のマイクで「押して話す」音声入力を行う。

- ボタンを押している間だけ録音する。車体にはマイクがないので、PC のそばで話す
- 音声はメモリ上だけで扱い、ファイルに保存しない（個人情報を残さない）
- 文字起こしは PC のローカルの Whisper で行い、音声をクラウドへ送らない

`sounddevice`、`faster-whisper`、`numpy` は `uv sync --extra voice` で入れる。
入っていなくても GUI は起動できるよう、使う時に読み込む。
"""

import os
import threading
from typing import Callable, List, NamedTuple, Optional

SAMPLE_RATE = 16000
WHISPER_MODEL_ENV = "PICARX_WHISPER_MODEL"
DEFAULT_WHISPER_MODEL = "small"
# 短すぎる録音は押し間違いとみなす
MIN_SECONDS = 0.4
MAX_SECONDS = 30.0
# これより小さい音は声として扱わない。無音から文を作る誤認識を防ぐ
MIN_PEAK = 0.005
# 内蔵マイクの音が小さいので、ピークがこの値になるまで大きくする
TARGET_PEAK = 0.5
MAX_GAIN = 30.0

VOICE_EXTRA_HINT = "音声入力には `uv sync --extra voice` が必要です"
TOO_SHORT = "短すぎて聞き取れませんでした。ボタンを押したまま話してください。"
TOO_QUIET = "声が小さくて聞き取れませんでした。マイクに近づいて話してください。"
NOT_HEARD = "聞き取れませんでした。もう一度お願いします。"


class Transcript(NamedTuple):
    """文字起こしの結果。`text` が空の時は `problem` に利用者へ伝える文がある"""

    text: str
    problem: Optional[str] = None
    seconds: float = 0.0


def normalize(audio):
    """ピークを揃える。増幅は上限までにして、雑音だけを大きくしすぎない"""
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return audio
    gain = min(TARGET_PEAK / peak, MAX_GAIN)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def check_audio(audio) -> Optional[str]:
    """文字起こしに回す前の確認。問題があれば利用者へ伝える文を返す"""
    import numpy as np

    if len(audio) < MIN_SECONDS * SAMPLE_RATE:
        return TOO_SHORT
    if float(np.max(np.abs(audio))) < MIN_PEAK:
        return TOO_QUIET
    return None


class Recorder:
    """押している間だけ録音する"""

    def __init__(self, stream_factory: Optional[Callable] = None):
        self._stream_factory = stream_factory
        self._chunks: List = []
        self._lock = threading.Lock()
        self._stream = None

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self) -> None:
        if self._stream is not None:
            return
        with self._lock:
            self._chunks = []
        factory = self._stream_factory or _default_stream
        self._stream = factory(self._on_audio)
        self._stream.start()

    def stop(self):
        """録音を止めて音声を返す。録音していなければ空の配列を返す"""
        import numpy as np

        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks).astype(np.float32)
        return audio[: int(MAX_SECONDS * SAMPLE_RATE)]

    def _on_audio(self, data, frames, time, status) -> None:
        with self._lock:
            self._chunks.append(data[:, 0].copy())


def _default_stream(callback):
    import sounddevice as sd

    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)


class Transcriber:
    """ローカルの Whisper で日本語を文字にする。モデルは最初に使う時に読み込む"""

    def __init__(self, model_name: Optional[str] = None, model_factory: Optional[Callable] = None):
        self.model_name = model_name or os.getenv(WHISPER_MODEL_ENV) or DEFAULT_WHISPER_MODEL
        self._model_factory = model_factory or _load_whisper
        self._model = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """モデルを読み込む。数秒かかるので、GUI の起動時に別スレッドで呼ぶ"""
        with self._lock:
            if self._model is None:
                self._model = self._model_factory(self.model_name)

    def transcribe(self, audio) -> Transcript:
        problem = check_audio(audio)
        if problem:
            return Transcript("", problem)
        self.load()
        segments, _ = self._model.transcribe(
            normalize(audio), language="ja", beam_size=1, vad_filter=True
        )
        text = "".join(segment.text for segment in segments).strip()
        seconds = len(audio) / SAMPLE_RATE
        return Transcript(text, None if text else NOT_HEARD, seconds)


def _load_whisper(name: str):
    from faster_whisper import WhisperModel

    try:
        # ダウンロード済みなら、起動のたびに更新を確かめに行かない
        return WhisperModel(name, device="cpu", compute_type="int8", local_files_only=True)
    except Exception:
        return WhisperModel(name, device="cpu", compute_type="int8")
