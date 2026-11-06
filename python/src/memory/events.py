"""Pi の安全イベントを記憶へ保存する。「さっき何で止まったの？」に実際の記録で答えるため。

Pi はイベントを再起動まで一定数だけ保持する。PC が起動した時点で残っている分も取り込む。
"""

from datetime import datetime, timezone
from typing import Optional

from src.agents.body_context import REASON_TEXT, REPORTED_STOP_REASONS
from src.models import SafetyEvent

DIRECTION_TEXT = {"forward": "前進中", "backward": "後退中"}
KIND_TEXT = {
    "emergency_stop": "緊急停止された",
    "emergency_stop_reset": "緊急停止が解除された",
    "stop_failed": "モーターの停止指令に失敗した（Pi が再送する）",
    "cliff_calibrated": "床を校正した",
    "cliff_calibration_failed": "床の校正に失敗した",
}


def summarize_event(event: SafetyEvent) -> Optional[str]:
    """記憶に残す文章にする。通常の走行でも起きる期限切れなどは None を返す。"""
    if event.kind != "reflex_stop":
        return KIND_TEXT.get(event.kind)
    if event.reason not in REPORTED_STOP_REASONS:
        return None
    details = [REASON_TEXT.get(event.reason, event.reason)]
    if event.direction:
        details.append(DIRECTION_TEXT.get(event.direction.value, event.direction.value))
    if event.distance_cm is not None:
        details.append(f"前方 {event.distance_cm:.1f}cm")
    return f"自動で停止した（{'、'.join(details)}）"


def source_key(event: SafetyEvent) -> str:
    return f"pi:{event.at_unix_ms}:{event.kind}:{event.reason or ''}"


class EventRecorder:
    """Pi のイベントを取得して記憶へ保存する。GUI の定期取得から呼ぶ。"""

    def __init__(self, client, memory_getter):
        self._client = client
        self._memory_getter = memory_getter
        self._after = 0

    async def poll(self) -> int:
        """新しいイベントを保存し、保存した件数を返す。

        通信の失敗は呼び出し元へ返す。記憶の失敗では取得位置を進めず、次回に再試行する。
        """
        response = await self._client.get_events(after=self._after)
        memory = self._memory_getter()
        saved = 0
        for event in response.events:
            summary = summarize_event(event)
            if summary is None:
                continue
            occurred_at = datetime.fromtimestamp(event.at_unix_ms / 1000, tz=timezone.utc)
            if memory.add_event(event.kind, summary, occurred_at, source_key(event)):
                saved += 1
        # Pi が再起動して番号が戻っても、Pi が全件を返し、保存済みは識別子で除く
        self._after = response.latest_event_id
        return saved
