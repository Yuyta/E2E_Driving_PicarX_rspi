"""
Phase 3.3: Communication Watchdog
PCとRaspberry Pi間のWi-Fi通信断、API応答タイムアウト、推論フリーズを監視する安全モジュール。
通信途絶や連続エラー発生時に車体の暴走を防ぐため、即時停止と介入イベント記録をトリガーする。
"""

import time
from typing import Optional


class CommunicationWatchdog:
    """通信監視・ハートビート・タイムアウト検出器"""

    def __init__(
        self,
        timeout_sec: float = 0.6,
        max_consecutive_errors: int = 3,
    ):
        """
        Args:
            timeout_sec: 前回の正常通信/推論からのタイムアウト猶予時間 (秒)
            max_consecutive_errors: 連続通信エラーの許容回数
        """
        self.timeout_sec = float(timeout_sec)
        self.max_consecutive_errors = int(max_consecutive_errors)

        self._last_heartbeat: float = time.time()
        self._consecutive_errors: int = 0
        self._is_triggered: bool = False

    def heartbeat(self, timestamp: Optional[float] = None) -> None:
        """正常通信または推論サイクル完了時に呼び出してタイマーをリセット"""
        self._last_heartbeat = timestamp if timestamp is not None else time.time()
        self._consecutive_errors = 0
        self._is_triggered = False

    def record_error(self) -> bool:
        """
        通信エラー（HTTPリクエスト失敗など）を記録。
        許容回数を超えた場合は True を返す。
        """
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_consecutive_errors:
            self._is_triggered = True
            return True
        return False

    def is_timeout(self, current_time: Optional[float] = None) -> bool:
        """指定秒数以上ハートビートが途切れているか確認"""
        now = current_time if current_time is not None else time.time()
        elapsed = now - self._last_heartbeat
        if elapsed > self.timeout_sec:
            self._is_triggered = True
            return True
        return False

    @property
    def should_stop(self) -> bool:
        """即時停止すべき状態かどうか"""
        return self._is_triggered or self.is_timeout()

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat

    def reset(self) -> None:
        """状態を初期化"""
        self._last_heartbeat = time.time()
        self._consecutive_errors = 0
        self._is_triggered = False
