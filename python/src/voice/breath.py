"""Open JTalk がひと息で読む長さを抑えるため、読点を足す。

Pi の Open JTalk は、読点や句点の間をひと息として読む。ひと息が30モーラ前後を超えると、
その先の音が6〜7倍に伸びて、ゆっくり聞こえる（2026-11-07、Docker 上の Open JTalk で確認）。
例: 「11月6日と11月7日に床を校正したこと、」は「こと」が1音あたり約0.5秒になった。

読みとモーラ数は pyopenjtalk で求める。数字や英字は全角に直されるので、文は作り直さず、
元の文の助詞の後ろに「、」を差し込むだけにする。
"""

import threading
from typing import List, Optional

# ひと息の上限。伸びは30モーラ前後から起きたので、余裕を取る
MAX_BREATH_MORAS = 26
# 助詞の後ろで区切るのは、ここまで読んでから。短すぎるひと息を作らない
MIN_BREATH_MORAS = 10
# 助詞が見つからない時は、ここで言葉の切れ目に区切る
HARD_BREATH_MORAS = 22
# Open JTalk が間を取る記号。ここでひと息を数え直す。括弧は間にならなかった
PAUSE_MARKS = set("、。，,．.！？!?\n")

_frontend = None
_frontend_lock = threading.Lock()
_frontend_failed = False


def _run_frontend(text: str) -> Optional[List[dict]]:
    """pyopenjtalk の解析結果を返す。使えない環境では None を返し、文はそのまま読む。"""
    global _frontend, _frontend_failed
    with _frontend_lock:
        if _frontend is None and not _frontend_failed:
            try:
                import pyopenjtalk

                _frontend = pyopenjtalk.run_frontend
            except ImportError:
                _frontend_failed = True
        frontend = _frontend
    if frontend is None:
        return None
    try:
        return frontend(text)
    except Exception:  # 解析に失敗しても、読み上げは止めない
        return None


def preload() -> None:
    """読み込みに約0.5秒かかるので、最初の返事の前に済ませておく。"""
    _run_frontend("準備")


def _is_pause(word: dict) -> bool:
    return word["pos"] == "記号" and any(mark in word["string"] for mark in PAUSE_MARKS)


def _locate(text: str, words: List[dict]) -> List[int]:
    """各言葉の、元の文での開始位置。数字や英字は全角に直されて見つからないので -1 にする。"""
    starts: List[int] = []
    cursor = 0
    for word in words:
        found = text.find(word["string"], cursor)
        if found >= 0 and found - cursor <= 8:
            starts.append(found)
            cursor = found + len(word["string"])
        else:
            starts.append(-1)
    return starts


def _find_cuts(text: str, words: List[dict]) -> List[int]:
    """読点を差し込む、元の文での位置を返す。"""
    starts = _locate(text, words)
    cuts: List[int] = []
    moras = 0
    for index, word in enumerate(words):
        if _is_pause(word):
            moras = 0
            continue
        moras += word["mora_size"]
        if index + 1 >= len(words):
            break
        following = words[index + 1]
        # 言葉がつながる所（「覚えて」と「いる」など）や記号の前では区切らない
        if (
            word["pos"] == "記号"
            or following["pos"] == "記号"
            or following.get("chain_flag") == 1
            or starts[index + 1] <= 0
        ):
            continue
        remaining = 0
        for later in words[index + 1 :]:
            if _is_pause(later):
                break
            remaining += later["mora_size"]
        if moras + remaining <= MAX_BREATH_MORAS:
            continue
        after_particle = word["pos"] == "助詞" and moras >= MIN_BREATH_MORAS
        if after_particle or moras >= HARD_BREATH_MORAS:
            cuts.append(starts[index + 1])
            moras = 0
    return cuts


def add_breath_pauses(text: str) -> str:
    """ひと息が長すぎる所に読点を足した文を返す。足す必要がなければ元の文を返す。"""
    words = _run_frontend(text)
    if not words:
        return text
    cuts = _find_cuts(text, words)
    if not cuts:
        return text
    pieces = []
    last = 0
    for cut in sorted(set(cuts)):
        pieces.append(text[last:cut])
        pieces.append("、")
        last = cut
    pieces.append(text[last:])
    return "".join(pieces)
