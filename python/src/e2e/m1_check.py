"""M1 の完了条件を実機で計測する。

    uv run python -m src.e2e.m1_check status
    uv run python -m src.e2e.m1_check camera --seconds 20
    uv run python -m src.e2e.m1_check events
    uv run python -m src.e2e.m1_check ttl --i-understand-the-car-will-move
    uv run python -m src.e2e.m1_check approach --speed 50 --i-understand-the-car-will-move

`ttl` と `approach` 以外は車両を動かさない。
"""

import argparse
import json
import statistics
import time
from typing import List, Optional

from src.models import DriveDirection, SafetyEvent
from src.picarx_client.client import PicarxClient

CAMERA_TARGET_FPS = 10.0
STOP_LATENCY_TARGET_MS = 100.0
HAZARD_REASONS = ("obstacle", "cliff")
# 走行プログラムの TTL と周期に合わせる
APPROACH_TTL_MS = 400
APPROACH_PERIOD_SEC = 0.07
APPROACH_SETTLE_SEC = 0.5
TTL_TEST_SPEED = 20
TTL_TEST_MS = 300
# TTL に加え、反射ループ1周と状態取得の遅れを許容する
TTL_POLL_TIMEOUT_SEC = 2.0
TTL_MARGIN_MS = 100.0


def check_status(client: PicarxClient) -> bool:
    status = client.get_status_sync()
    print(json.dumps(status.model_dump(mode="json"), indent=2, ensure_ascii=False))
    print(f"\n反射ループ: {status.reflex.hz:.1f} Hz（目標 20Hz 以上）")
    return status.reflex.hz >= 20.0


def check_camera(client: PicarxClient, seconds: float) -> bool:
    arrivals: List[float] = []
    with client.open_camera_stream() as stream:
        seq = 0
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = stream.wait_for_frame(after_seq=seq, timeout=2.0)
            if frame is None:
                print(f"フレームが届きません: {stream.last_error}")
                break
            seq = frame.seq
            arrivals.append(frame.received_at)

    if len(arrivals) < 2:
        print("測定に必要なフレーム数がありません")
        return False

    intervals = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    # 配信の開始直後は安定しないので、全体の平均で判定する
    fps = (len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
    print(
        f"フレーム {len(arrivals)} 枚: {fps:.1f} fps, "
        f"間隔 中央値 {statistics.median(intervals):.1f} ms, 最大 {max(intervals):.1f} ms"
    )
    passed = fps >= CAMERA_TARGET_FPS
    print(f"判定: {'合格' if passed else '不合格'}（目標 {CAMERA_TARGET_FPS:.0f}fps 以上）")
    return passed


def summarize_stop_latency(events: List[SafetyEvent]) -> Optional[float]:
    """障害物と段差の停止について、停止までの最大時間を返す。事象がなければ `None`。

    `ttl_expired` は危険の検知ではないので、完了条件の判定に含めない。
    """
    latencies = [
        event.stop_latency_ms
        for event in events
        if event.kind == "reflex_stop"
        and event.reason in HAZARD_REASONS
        and event.stop_latency_ms is not None
    ]
    return max(latencies) if latencies else None


def check_events(client: PicarxClient) -> bool:
    response = client.get_events_sync()
    for event in response.events:
        latency = f"{event.stop_latency_ms:.1f} ms" if event.stop_latency_ms is not None else "-"
        print(
            f"#{event.id} {event.kind} reason={event.reason} "
            f"distance={event.distance_cm} floor={event.grayscale} stop={latency}"
        )
    worst = summarize_stop_latency(response.events)
    if worst is None:
        print("障害物や段差で止まった記録がありません。走行中に止めてから再実行してください")
        return False
    passed = worst <= STOP_LATENCY_TARGET_MS
    print(f"検知から停止まで 最大 {worst:.1f} ms（目標 {STOP_LATENCY_TARGET_MS:.0f}ms 以下）")
    print(f"判定: {'合格' if passed else '不合格'}")
    return passed


def check_ttl(client: PicarxClient) -> bool:
    """短い TTL で一度だけ前進指令を送り、以後何も送らずに停止を確認する。"""
    before = client.get_status_sync().latest_event_id
    print(f"速度 {TTL_TEST_SPEED}、TTL {TTL_TEST_MS}ms で前進します。以後は指令を送りません")
    if not client.move_sync(DriveDirection.FORWARD, speed=TTL_TEST_SPEED, ttl_ms=TTL_TEST_MS):
        print(f"走行を拒否されました: {client.get_status_sync().blocked.forward}")
        return False

    started = time.monotonic()
    try:
        elapsed_ms = None
        while time.monotonic() - started < TTL_POLL_TIMEOUT_SEC:
            if client.get_status_sync().motion is None:
                elapsed_ms = (time.monotonic() - started) * 1000.0
                break
            time.sleep(0.02)
    finally:
        client.stop_sync()

    if elapsed_ms is None:
        print(f"判定: 不合格（{TTL_POLL_TIMEOUT_SEC}秒以内に停止を確認できません）")
        return False

    events = client.get_events_sync(after=before).events
    stopped = [event for event in events if event.kind == "reflex_stop"]
    reason = stopped[-1].reason if stopped else None
    passed = reason == "ttl_expired" and elapsed_ms <= TTL_TEST_MS + TTL_MARGIN_MS
    print(f"停止を確認: {elapsed_ms:.0f} ms 後、理由 {reason}")
    print(f"判定: {'合格' if passed else '不合格'}")
    return passed


def check_approach(client: PicarxClient, speed: int, seconds: float) -> bool:
    """走行ループと同じ間隔で前進指令を更新し続け、Pi が止めるまで連続で走る。"""
    before = client.get_status_sync().latest_event_id
    print(f"速度 {speed} で最大 {seconds:.0f} 秒、連続で前進します")
    refused = False
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not client.move_sync(DriveDirection.FORWARD, speed=speed, ttl_ms=APPROACH_TTL_MS):
                refused = True
                break
            time.sleep(APPROACH_PERIOD_SEC)
    finally:
        client.stop_sync()

    # 惰性で進み終わってから距離を読む
    time.sleep(APPROACH_SETTLE_SEC)
    status = client.get_status_sync()
    print("Pi が前進を拒否しました" if refused else "時間切れで止めました。Pi は止めていません")
    print(f"拒否の理由: {status.blocked.forward}")
    print(f"停止後の距離: {status.sensors.distance.cm} cm、床: {status.sensors.floor.values}")
    for event in client.get_events_sync(after=before).events:
        if event.kind == "reflex_stop" and event.reason in HAZARD_REASONS:
            print(
                f"#{event.id} {event.reason}: 検知時の距離 {event.distance_cm} cm、"
                f"床 {event.grayscale}、停止まで {event.stop_latency_ms:.1f} ms"
            )
    return refused


def main() -> int:
    parser = argparse.ArgumentParser(description="M1 の完了条件を実機で計測する")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status", help="状態を表示する")
    camera = sub.add_parser("camera", help="配信の fps を測る")
    camera.add_argument("--seconds", type=float, default=20.0)
    sub.add_parser("events", help="安全イベントと停止までの時間を表示する")
    ttl = sub.add_parser("ttl", help="切断時に TTL で止まることを確認する（車両が動く）")
    ttl.add_argument(
        "--i-understand-the-car-will-move",
        action="store_true",
        required=True,
        help="車両が少し前進することを了承する",
    )
    approach = sub.add_parser("approach", help="連続走行で障害物や段差に近づける（車両が動く）")
    approach.add_argument("--speed", type=int, default=50)
    approach.add_argument("--seconds", type=float, default=3.0)
    approach.add_argument(
        "--i-understand-the-car-will-move",
        action="store_true",
        required=True,
        help="車両が連続で前進することを了承する",
    )
    args = parser.parse_args()

    client = PicarxClient(timeout=3.0)
    if args.command == "status":
        passed = check_status(client)
    elif args.command == "camera":
        passed = check_camera(client, args.seconds)
    elif args.command == "events":
        passed = check_events(client)
    elif args.command == "approach":
        passed = check_approach(client, args.speed, args.seconds)
    else:
        passed = check_ttl(client)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
