"""「止まって」を LLM を通さずに検知する。

停止は LLM の応答時間と判断に任せない。言葉が一覧にない時は LLM の stop_moving が補う。
誤検知しても車が止まるだけなので、広めに検知する。
"""

import re
from enum import Enum

STOP_WORDS = ("止まって", "止まれ", "止まる", "とまって", "とまれ", "ストップ", "停止", "stop")
# 「止まってください！」も停止だけの指示として扱う
POLITE_SUFFIX = re.compile(r"(ください|下さい|くれ|よ|ね)$")
PUNCTUATION = re.compile(r"[\s、。,.!！?？~〜ー…]+")


class StopIntent(str, Enum):
    NONE = "none"
    ONLY_STOP = "only_stop"
    STOP_AND_MORE = "stop_and_more"


def parse_stop_intent(text: str) -> StopIntent:
    normalized = PUNCTUATION.sub("", text).lower()
    if not any(word in normalized for word in STOP_WORDS):
        return StopIntent.NONE
    if PUNCTUATION.sub("", POLITE_SUFFIX.sub("", normalized)) in STOP_WORDS:
        return StopIntent.ONLY_STOP
    return StopIntent.STOP_AND_MORE
