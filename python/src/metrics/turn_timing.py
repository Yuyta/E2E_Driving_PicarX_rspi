"""会話1回ごとの応答時間を測る。M2 の完了条件「相づちと本回答の開始までの目標」を決めるため。

時刻は、入力が終わった時（送信、または押して話すボタンを離した時）からの秒数で記録する。
「流れ始め」は、Pi が最初の文を音声にして再生キューへ入れた時刻で、実際の再生開始に近い。

会話の内容は保存しない。保存先は git の追跡外の `python/data/turn_timings/`。
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

TIMINGS_ENV = "PICARX_TIMINGS_FILE"
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "turn_timings" / "timings.jsonl"

# 印の名前。表示と集計で使う
ACK_AUDIO = "ack_audio"  # 相づちが流れ始めた
STOPPED = "stopped"  # 停止指令が Pi に届いた
REPLY_READY = "reply_ready"  # 返事の文章ができた
REPLY_AUDIO = "reply_audio"  # 返事が流れ始めた

_write_lock = threading.Lock()


class TurnTiming:
    def __init__(
        self,
        input_kind: str,
        started: Optional[float] = None,
        stt_sec: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._clock = clock
        self.started = started if started is not None else clock()
        self.input_kind = input_kind  # text または voice
        self.stt_sec = stt_sec
        self.kind = "llm"  # llm、look、stop、speech_command
        self.model: Optional[str] = None
        self.ok = True
        self.marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        """最初の1回だけ記録する"""
        self.marks.setdefault(name, round(self._clock() - self.started, 3))

    def first_audio(self) -> Optional[float]:
        audio = [self.marks[name] for name in (ACK_AUDIO, REPLY_AUDIO) if name in self.marks]
        return min(audio) if audio else None

    def record(self) -> dict:
        return {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input": self.input_kind,
            "kind": self.kind,
            "model": self.model,
            "ok": self.ok,
            "stt_sec": self.stt_sec,
            "first_audio_sec": self.first_audio(),
            **{f"{name}_sec": value for name, value in self.marks.items()},
        }

    def summary(self) -> str:
        parts = []
        if self.stt_sec is not None:
            parts.append(f"文字起こし {self.stt_sec:.1f}秒")
        labels = (
            (ACK_AUDIO, "相づち"),
            (STOPPED, "停止"),
            (REPLY_READY, "返事の作成"),
            (REPLY_AUDIO, "返事の流れ始め"),
        )
        parts.extend(
            f"{label} {self.marks[name]:.1f}秒" for name, label in labels if name in self.marks
        )
        return "、".join(parts)


def timings_path() -> Path:
    return Path(os.getenv(TIMINGS_ENV) or DEFAULT_PATH)


def save(timing: TurnTiming, path: Optional[Path] = None) -> None:
    """1行の JSON として追記する。失敗は呼び出し元へ返す"""
    target = Path(path) if path is not None else timings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(timing.record(), ensure_ascii=False)
    with _write_lock, target.open("a", encoding="utf-8") as file:
        file.write(line + "\n")
