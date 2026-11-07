"""Pi の実測値を、毎ターン Agent に渡す文章へ変換する。

Agent が自分の状況を推測で答えないよう、取得できない値は「不明」と明記する。
"""

import re
from typing import Iterable, List, Optional

import httpx

from src.models import RobotStatus, SafetyEvent, SafetyReason
from src.picarx_client.client import PicarxClient

UNKNOWN = "不明"
STATUS_TIMEOUT_SEC = 1.0
RECENT_EVENT_LIMIT = 3
# 期限切れは通常の走行でも起きるので、Agent へは危険による停止だけを伝える
REPORTED_STOP_REASONS = {"obstacle", "cliff", "distance_unknown", "floor_unknown"}

MODE_TEXT = {
    "idle": "停止中",
    "forward": "前進中",
    "backward": "後退中",
    "emergency_stop": "緊急停止中（解除するまで走行できない）",
}
REASON_TEXT = {
    SafetyReason.EMERGENCY_STOP.value: "緊急停止中",
    SafetyReason.OBSTACLE.value: "障害物が近い",
    SafetyReason.CLIFF.value: "段差を検知",
    SafetyReason.DISTANCE_UNKNOWN.value: "距離が不明",
    SafetyReason.FLOOR_UNKNOWN.value: "床センサーが不明",
    SafetyReason.CLIFF_UNCALIBRATED.value: "段差の校正が未完了",
    SafetyReason.REFLEX_STALLED.value: "安全監視が停止している",
    SafetyReason.TTL_EXPIRED.value: "指令の期限切れ",
}

UNAVAILABLE_CONTEXT = (
    "【現在の車体の状態】\n"
    "車体と通信できないため、電池、センサー、走行状態はすべて不明。"
    "聞かれたら、通信できず分からないと答えること。"
)
ANSWER_RULE = (
    "車体の状態を聞かれたら、上の値だけを使って答えること。"
    "値を推測しないこと。「不明」の項目は分からないと答えること。"
)


def _battery_text(status: RobotStatus) -> str:
    battery = status.sensors.battery
    if battery.voltage is None or battery.percent is None:
        return UNKNOWN
    return f"{battery.voltage:.2f}V、残量の目安 {battery.percent}%"


def _distance_text(status: RobotStatus) -> str:
    distance = status.sensors.distance
    if distance.cm is None:
        return UNKNOWN
    if distance.far:
        return "反射なし（遠い、または細い物や音を吸う物があるかもしれない）"
    return f"{distance.cm:.1f}cm"


def _floor_text(status: RobotStatus) -> str:
    floor = status.sensors.floor
    if floor.values is None:
        return UNKNOWN
    if floor.cliff and any(floor.cliff):
        return "段差を検知している"
    return "異常なし" if floor.cliff is not None else "段差の判定は未校正"


def _block_text(reason: Optional[SafetyReason]) -> str:
    if reason is None:
        return "可能"
    return f"不可（{REASON_TEXT.get(reason.value, reason.value)}）"


def _event_text(events: Iterable[SafetyEvent]) -> str:
    stops = [
        REASON_TEXT.get(event.reason, event.reason)
        for event in events
        if event.kind == "reflex_stop" and event.reason in REPORTED_STOP_REASONS
    ]
    return "、".join(stops[-RECENT_EVENT_LIMIT:]) if stops else "なし"


def format_body_context(status: RobotStatus, events: Iterable[SafetyEvent] = ()) -> str:
    """状態を Agent のシステムプロンプトへ入れる文章にする"""
    return "\n".join(
        [
            "【現在の車体の状態】",
            f"- 電池: {_battery_text(status)}",
            f"- 走行: {MODE_TEXT.get(status.mode, status.mode)}",
            f"- 前方の距離: {_distance_text(status)}",
            f"- 床: {_floor_text(status)}",
            f"- 前進: {_block_text(status.blocked.forward)}",
            f"- 後退: {_block_text(status.blocked.backward)}",
            f"- 最近の安全停止: {_event_text(events)}",
            ANSWER_RULE,
        ]
    )


async def fetch_body_context(client) -> str:
    """Pi から状態を取得する。取得できなくても会話は止めない。"""
    try:
        status = await client.get_status()
    except (httpx.HTTPError, ValueError):
        return UNAVAILABLE_CONTEXT
    try:
        events = (await client.get_events()).events
    except (httpx.HTTPError, ValueError):
        events = []
    return format_body_context(status, events)


STATE_TOPICS = {
    "battery": re.compile(r"電池|バッテリー|充電|残量|ボルト"),
    "distance": re.compile(r"距離|障害物|壁まで|何cm|何センチ"),
    "floor": re.compile(r"段差|床"),
    "movable": re.compile(r"進める|動ける|走れる|下がれる|前進できる|後退できる"),
    "summary": re.compile(r"状態|調子|様子"),
}
STATE_QUESTION = re.compile(
    r"[?？]|どう|何|なに|いくつ|いくら|どれ|教えて|測って|はかって|調べて|確認|残って|大丈夫|"
    r"知りたい|か[。！!]*$"
)
# 過去の話や記憶の指示は、今の実測値では答えられない
NOT_CURRENT = re.compile(r"覚え|忘れ|前回|この前|さっき|昨日|きのう|先週|買")
UNREACHABLE_ANSWER = "車体と通信できないため、分かりません。"


def state_topics(text: str) -> List[str]:
    """今の車体の状態を聞く発言なら、その話題を返す。そうでなければ空を返す。"""
    if NOT_CURRENT.search(text) or not STATE_QUESTION.search(text):
        return []
    return [topic for topic, pattern in STATE_TOPICS.items() if pattern.search(text)]


def _distance_sentence(status: RobotStatus) -> str:
    distance = status.sensors.distance
    if distance.cm is None:
        return "前方の距離は分かりません。"
    if distance.far:
        return "前方は超音波の反射がありません。遠いか、細い物や音を吸う物があるかもしれません。"
    return f"前方の距離は{distance.cm:.1f}cmです。"


def _battery_sentence(status: RobotStatus) -> str:
    text = _battery_text(status)
    return "電池の値は分かりません。" if text == UNKNOWN else f"電池は{text}です。"


def _movable_sentence(status: RobotStatus) -> str:
    return (
        f"前進は「{_block_text(status.blocked.forward)}」、"
        f"後退は「{_block_text(status.blocked.backward)}」です。"
    )


def format_state_answer(topics: Iterable[str], status: RobotStatus) -> str:
    """実測値から返事を作る。LLM を通さないので、値を間違えない。"""
    topics = list(topics)
    if "summary" in topics:
        topics = ["battery", "mode", "distance", "movable"]
    sentences = {
        "battery": lambda: _battery_sentence(status),
        "mode": lambda: f"今は{MODE_TEXT.get(status.mode, status.mode)}です。",
        "distance": lambda: _distance_sentence(status),
        "floor": lambda: f"床の状態は「{_floor_text(status)}」です。",
        "movable": lambda: _movable_sentence(status),
    }
    return "".join(sentences[topic]() for topic in topics)


async def answer_state_question(text: str, client) -> Optional[str]:
    """車体の状態の質問なら実測値で答える。別の話題なら None を返す。"""
    topics = state_topics(text)
    if not topics:
        return None
    try:
        status = await (client or PicarxClient(timeout=STATUS_TIMEOUT_SEC)).get_status()
    except (httpx.HTTPError, ValueError):
        return UNREACHABLE_ANSWER
    return format_state_answer(topics, status)
