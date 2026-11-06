"""PC から Pi へ送る発話を、すべてここに通す。

- 応答を文に分けて送る。Pi は1文目を再生しながら2文目を生成するので、話し始めが早くなる
- 新しい発言や緊急停止の時は `interrupt()` で、送っていない文を捨て、Pi の再生も止める
"""

import asyncio
import re
import threading
from typing import Callable, List, Optional

SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")
# 短すぎる文は前の文につなげる（「はい。」だけを1回の合成にしない）
MIN_SENTENCE_CHARS = 4
# 1回の合成が長くならないよう、句点のない長文は読点で切る
MAX_SENTENCE_CHARS = 60
COMMA = re.compile(r"(?<=[、,])")
# 停止の要求が届く前に次の文を送ると、次の文まで止まってしまう
STOP_WAIT_SEC = 2.0


def _split_long(sentence: str) -> List[str]:
    if len(sentence) <= MAX_SENTENCE_CHARS:
        return [sentence]
    parts: List[str] = []
    current = ""
    for piece in COMMA.split(sentence):
        if current and len(current) + len(piece) > MAX_SENTENCE_CHARS:
            parts.append(current)
            current = ""
        current += piece
    if current:
        parts.append(current)
    return parts


def split_sentences(text: str) -> List[str]:
    """読み上げる単位に分ける。空の文は返さない。"""
    sentences: List[str] = []
    for raw in SENTENCE_END.split(text):
        sentence = raw.strip()
        if not sentence:
            continue
        if sentences and len(sentence) < MIN_SENTENCE_CHARS:
            sentences[-1] += sentence
            continue
        sentences.extend(_split_long(sentence))
    return sentences


class Speaker:
    """Pi のスピーカーへの発話を1か所で管理する"""

    def __init__(self, client, on_error: Optional[Callable[[Exception], None]] = None):
        self._client = client
        self._on_error = on_error
        self._generation = 0
        self._generation_lock = threading.Lock()
        self._order: Optional[asyncio.Lock] = None
        self._pending_stops = 0
        self._stop_done = threading.Event()
        self._stop_done.set()

    @property
    def generation(self) -> int:
        with self._generation_lock:
            return self._generation

    async def say(self, text: str) -> bool:
        """文ごとに Pi へ送る。途中で止められたら False を返す。

        送信の失敗は例外のまま返す。呼び出し元が画面に出す。
        """
        generation = self.generation
        # 前の応答の文と混ざらないよう、送信は1つずつ順に行う
        if self._order is None:
            self._order = asyncio.Lock()
        async with self._order:
            if not self._stop_done.is_set():
                await asyncio.to_thread(self._stop_done.wait, STOP_WAIT_SEC)
            for sentence in split_sentences(text):
                if self.generation != generation:
                    return False
                await self._client.speak(sentence)
        return self.generation == generation

    def interrupt(self) -> None:
        """送っていない文を捨て、Pi の再生を止める。どのスレッドからでも呼べ、待たずに戻る。"""
        with self._generation_lock:
            self._generation += 1
            self._pending_stops += 1
            self._stop_done.clear()
        threading.Thread(target=self._stop_on_pi, daemon=True).start()

    def _stop_on_pi(self) -> None:
        try:
            self._client.stop_speech_sync()
        except Exception as error:  # 発話を止められなくても、会話や走行は止めない
            if self._on_error:
                self._on_error(error)
        finally:
            with self._generation_lock:
                self._pending_stops -= 1
                if self._pending_stops == 0:
                    self._stop_done.set()
