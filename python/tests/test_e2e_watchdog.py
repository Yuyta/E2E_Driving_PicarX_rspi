import time

from src.e2e.safety.watchdog import CommunicationWatchdog


def test_watchdog_normal_heartbeat():
    dog = CommunicationWatchdog(timeout_sec=0.5, max_consecutive_errors=3)
    assert not dog.is_timeout()
    assert not dog.should_stop
    assert dog.consecutive_errors == 0

    dog.heartbeat()
    assert not dog.should_stop


def test_watchdog_timeout():
    dog = CommunicationWatchdog(timeout_sec=0.1, max_consecutive_errors=3)
    start = time.time()
    dog.heartbeat(timestamp=start)

    # 50ms経過 (まだタイムアウトしない)
    assert not dog.is_timeout(current_time=start + 0.05)
    assert not dog.should_stop

    # 150ms経過 (タイムアウト)
    assert dog.is_timeout(current_time=start + 0.15)
    assert dog.should_stop


def test_watchdog_consecutive_errors():
    dog = CommunicationWatchdog(timeout_sec=1.0, max_consecutive_errors=3)

    assert not dog.record_error()  # 1回目
    assert dog.consecutive_errors == 1
    assert not dog.should_stop

    assert not dog.record_error()  # 2回目
    assert dog.consecutive_errors == 2
    assert not dog.should_stop

    assert dog.record_error()  # 3回目 -> エラー閾値到達
    assert dog.consecutive_errors == 3
    assert dog.should_stop

    # 正常復帰でリセットされるか
    dog.heartbeat()
    assert dog.consecutive_errors == 0
    assert not dog.should_stop
