"""M1 の完了条件を実機で計測する。

    uv run python -m src.e2e.m1_check status
    uv run python -m src.e2e.m1_check camera --seconds 20
    uv run python -m src.e2e.m1_check events
    uv run python -m src.e2e.m1_check ttl --i-understand-the-car-will-move
    uv run python -m src.e2e.m1_check approach --speed 50 --i-understand-the-car-will-move
    uv run python -m src.e2e.m1_check wifi --i-understand-the-car-will-move
    uv run python -m src.e2e.m1_check wifi --simulate --ssh_host <Pi> \
        --i-understand-the-car-will-move

`ttl`、`approach`、`wifi` 以外は車両を動かさない。
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Sequence

import httpx

from src.models import DRIVE_SPEED, MIN_MOVING_SPEED, DriveDirection, SafetyEvent
from src.picarx_client.client import PicarxClient

CAMERA_TARGET_FPS = 10.0
STOP_LATENCY_TARGET_MS = 100.0
HAZARD_REASONS = ("obstacle", "cliff")
# 走行プログラムの TTL と周期に合わせる
APPROACH_TTL_MS = 400
APPROACH_PERIOD_SEC = 0.07
APPROACH_SETTLE_SEC = 0.5
TTL_TEST_SPEED = DRIVE_SPEED
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


WIFI_TEST_SPEED = DRIVE_SPEED
WIFI_RECONNECT_TIMEOUT_SEC = 180.0


# 疑似切断: 走り出してから遮断するまでの秒数と、遮断を続ける秒数。
# 遮断中に PC の要求がタイムアウト（3秒）して切断と判断し、解除後に Pi の記録を読む
SIMULATE_CUT_AFTER_SEC = 2.0
SIMULATE_CUT_SECONDS = 8
NETCUT_COMMAND = "sudo -n /usr/local/sbin/picarx-netcut"


def netcut_ssh_command(host: str, key: Optional[str], seconds: int) -> List[str]:
    """Pi の `picarx-netcut` を SSH で呼ぶコマンド。鍵の指定がなければ SSH の設定に任せる。"""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-x"]
    if key:
        command += ["-i", key, "-o", "IdentitiesOnly=yes"]
    return command + [host, f"{NETCUT_COMMAND} {seconds}"]


def check_wifi(
    client: PicarxClient,
    seconds: float,
    netcut: Optional[Sequence[str]] = None,
    speed: int = WIFI_TEST_SPEED,
) -> bool:
    """走行中に PC と Pi の通信を切り、Pi が TTL で止めたことを再接続後に確かめる。

    `netcut` がなければ、利用者が PC の Wi-Fi を切る。あれば走り出して
    `SIMULATE_CUT_AFTER_SEC` 秒後にそのコマンドを実行し、Pi 側で走行サーバーへの
    受信を一定時間破棄する（疑似切断）。PC の他の通信には影響しない。

    Pi は TTL の期限から実際に止めるまでの時間を `stop_latency_ms` に記録する。
    最後に届いた指令から止まるまでの時間は TTL + `stop_latency_ms`。
    """
    before = client.get_status_sync().latest_event_id
    print(f"速度 {speed}、TTL {APPROACH_TTL_MS}ms で最大 {seconds:.0f} 秒前進を続けます")
    if netcut is None:
        print("走り出したら PC の Wi-Fi を切ってください。車が止まったら Wi-Fi を戻してください")
    else:
        print(f"約 {SIMULATE_CUT_AFTER_SEC:.0f} 秒後に Pi 側で通信を遮断します（疑似切断）")
    cutter: Optional[subprocess.Popen] = None
    disconnected = False
    started = time.monotonic()
    deadline = started + seconds
    while time.monotonic() < deadline:
        if netcut is not None and cutter is None:
            if time.monotonic() - started >= SIMULATE_CUT_AFTER_SEC:
                cutter = subprocess.Popen(
                    list(netcut),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
        if cutter is not None and cutter.poll() not in (None, 0):
            # 遮断できなかった。走り続けないよう、すぐ止める
            client.stop_sync()
            report_netcut(cutter)
            print("判定: 未実施（疑似切断に失敗しました）")
            return False
        try:
            if not client.move_sync(DriveDirection.FORWARD, speed=speed, ttl_ms=APPROACH_TTL_MS):
                print(f"Pi が前進を拒否しました: {client.get_status_sync().blocked.forward}")
                client.stop_sync()
                return False
        except httpx.HTTPError:
            disconnected = True
            break
        time.sleep(APPROACH_PERIOD_SEC)

    if not disconnected:
        client.stop_sync()
        report_netcut(cutter)
        print("判定: 未実施（時間内に通信が切れませんでした）")
        return False

    if netcut is None:
        print("通信が切れました。Wi-Fi を戻すのを待っています...")
    else:
        print("通信が切れました。遮断の解除を待っています...")
    reconnect_deadline = time.monotonic() + WIFI_RECONNECT_TIMEOUT_SEC
    events = None
    while time.monotonic() < reconnect_deadline:
        try:
            client.stop_sync()
            events = client.get_events_sync(after=before).events
            break
        except httpx.HTTPError:
            time.sleep(1.0)
    report_netcut(cutter)
    if events is None:
        print(f"判定: 不明（{WIFI_RECONNECT_TIMEOUT_SEC:.0f}秒以内に再接続できません）")
        return False

    stops = [
        e
        for e in events
        if e.kind == "reflex_stop" and e.reason == "ttl_expired" and e.direction == "forward"
    ]
    if not stops:
        print("判定: 不合格（Pi の ttl_expired の停止記録がありません）")
        return False
    latency = stops[0].stop_latency_ms or 0.0
    passed = latency <= TTL_MARGIN_MS
    print(
        f"Pi が ttl_expired で停止: 期限から {latency:.1f} ms、"
        f"最後の指令から約 {APPROACH_TTL_MS + latency:.0f} ms"
    )
    print(f"判定: {'合格' if passed else '不合格'}（期限から {TTL_MARGIN_MS:.0f} ms 以内）")
    return passed


def report_netcut(cutter: Optional[subprocess.Popen]) -> None:
    """疑似切断のコマンドの終了を待ち、失敗していれば理由を表示する。"""
    if cutter is None:
        return
    try:
        out, err = cutter.communicate(timeout=SIMULATE_CUT_SECONDS + 10)
    except subprocess.TimeoutExpired:
        cutter.kill()
        out, err = cutter.communicate()
    if cutter.returncode != 0:
        print(f"疑似切断のコマンドが失敗しました（終了コード {cutter.returncode}）: {err.strip()}")
    for line in out.splitlines():
        print(f"Pi: {line}")


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
    wifi = sub.add_parser("wifi", help="走行中の Wi-Fi 切断で止まることを確認する（車両が動く）")
    wifi.add_argument("--seconds", type=float, default=30.0)
    wifi.add_argument(
        "--speed",
        type=int,
        default=WIFI_TEST_SPEED,
        help=f"前進の速度。{MIN_MOVING_SPEED} 未満では車輪が回らず、止まる様子を確かめられない",
    )
    wifi.add_argument(
        "--simulate",
        action="store_true",
        help="PC の Wi-Fi を切らず、Pi 側で走行サーバーへの受信を一定時間破棄する",
    )
    wifi.add_argument(
        "--ssh_host",
        default=os.environ.get("PICARX_SSH_HOST"),
        help="疑似切断で接続する Pi（SSH の設定名でもよい）。既定は PICARX_SSH_HOST",
    )
    wifi.add_argument(
        "--ssh_key",
        default=os.environ.get("PICARX_SSH_KEY"),
        help="パスフレーズなしの鍵。既定は PICARX_SSH_KEY、なければ SSH の設定に任せる",
    )
    wifi.add_argument(
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
    elif args.command == "wifi":
        netcut = None
        if args.simulate:
            if not args.ssh_host:
                parser.error("--simulate には --ssh_host か PICARX_SSH_HOST が必要です")
            key = str(Path(args.ssh_key).expanduser()) if args.ssh_key else None
            netcut = netcut_ssh_command(args.ssh_host, key, SIMULATE_CUT_SECONDS)
        passed = check_wifi(client, args.seconds, netcut, args.speed)
    else:
        passed = check_ttl(client)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
