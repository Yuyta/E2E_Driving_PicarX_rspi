from typing import List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.config import settings


class TokenOptimizer:
    """API呼び出し時のトークン消費を最小化するための最適化モジュール"""

    def __init__(self):
        self.last_distance_cm: Optional[float] = None

    def should_report_distance(self, current_distance: float) -> bool:
        """
        距離センサーの値が前回から閾値以上変化したかチェック。
        変化が小さい場合はトークン節約のためレポートをスキップ可能。
        """
        if not settings.enable_sensor_diff_filter:
            return True

        if self.last_distance_cm is None:
            self.last_distance_cm = current_distance
            return True

        diff = abs(current_distance - self.last_distance_cm)
        if diff >= settings.sensor_diff_threshold_cm:
            self.last_distance_cm = current_distance
            return True

        return False

    def compress_history(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        メッセージ履歴が長くなった場合に過去の履歴を圧縮し、トークン消費を抑える。
        直近のメッセージを残し、古いメッセージを要約スニペットにまとめる。
        """
        if len(messages) <= settings.max_history_turns:
            return messages

        # 直近 N ターンを残し、それ以前を要約に置換
        keep_count = settings.max_history_turns // 2
        older_messages = messages[:-keep_count]
        recent_messages = messages[-keep_count:]

        summary_points = []
        for msg in older_messages:
            prefix = (
                "User"
                if isinstance(msg, HumanMessage)
                else "Bot"
                if isinstance(msg, AIMessage)
                else "System"
            )
            # 50文字に短縮して要約ポイント化
            text = str(msg.content)[:50]
            summary_points.append(f"- {prefix}: {text}")

        summary_content = "【過去の対話要約】\n" + "\n".join(summary_points)
        compressed = [SystemMessage(content=summary_content)] + recent_messages
        return compressed

    def format_status_for_prompt(
        self, distance_cm: Optional[float], grayscale: Optional[List[int]]
    ) -> str:
        """
        センサー情報を最小限のトークンで表現するコンパクト文字列にフォーマット。
        """
        parts = []
        if distance_cm is not None:
            if self.should_report_distance(distance_cm):
                parts.append(f"前方:{distance_cm:.1f}cm")
        if grayscale is not None:
            parts.append(f"床面:{grayscale}")

        return " | ".join(parts) if parts else ""
