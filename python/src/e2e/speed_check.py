"""車輪が回る最低の速度を実機で測る（ROADMAP の M3「車輪が回る最低の速度を測る」）。

壁に向かって走らせ、超音波の距離の変化で、車が実際に進んだかを確かめる。速度ごとに2回測る。

- 発進: 止まった状態から、その速度で走らせる。発進できる最低の速度が分かる
- 走行中: `ROLL_FROM_SPEED` で走らせてから、その速度へ落とす。減速の下限に使える最低の速度が分かる

    uv run python -m src.e2e.speed_check --speeds 30 35 40 45 --i-understand-the-car-will-move

壁の 1〜1.5m 手前に、壁へまっすぐ向けて置く。壁に近づきすぎたら止めて、置き直しを頼む。
記録は `data/m3_checks/` に保存する（画像は含まない）。
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from src.models import DRIVE_SPEED, DriveDirection
from src.picarx_client.client import PicarxClient

OUTPUT_DIR = Path("data/m3_checks")
MOVE_TTL_MS = 400
MOVE_PERIOD_SEC = 0.07
CONTROL_TIMEOUT_SEC = 1.0

TRIAL_SEC = 1.5  # その速度で走らせる時間
SETTLE_SEC = 1.0  # 次の試行の前に、完全に止まるまで待つ時間
ROLL_FROM_SPEED = DRIVE_SPEED  # 走行中の試験で、先に走らせる速度（動くと分かっている値）
ROLL_FROM_SEC = 1.5  # 止まった状態から速度50でも、動き出しに約1秒かかる（2026-11-09 実測）
ROLL_SKIP_SEC = 0.5  # 速度を落とした直後は惰性で進むので、この時間は速さの計算に使わない
# 壁に近づくと Pi が 15cm で止める。1回の試行で進む分の余裕を残し、これより近ければ置き直す
MIN_START_DISTANCE_CM = 50.0
# 「進んだ」とみなす最小の値。実際は、最初に測る静止中の揺れの3倍と比べて大きいほうを使う
MOVED_CM = 3.0
MOVING_CM_S = 1.5  # 速度35で約3cm/s の這うような走りを「動く」と判定できる値
NOISE_SAMPLES = 20
NOISE_FACTOR = 3.0


class TooClose(Exception):
    """壁に近すぎるか、距離が読めない。置き直してから続ける"""


def read_distance(client: PicarxClient, samples: int = 5) -> Optional[float]:
    """静止中の距離。揺れを抑えるため中央値を使う。遠すぎるか読めなければ None"""
    values = []
    for _ in range(samples):
        distance = client.get_status_sync().sensors.distance
        if distance.cm is not None and not distance.far:
            values.append(distance.cm)
        time.sleep(0.05)
    return statistics.median(values) if values else None


def approach_rate(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """(時刻, 距離) から、壁に近づく速さ（cm/s、正が接近）を出す。

    超音波は反射する物が変わると距離が跳ぶので、最小二乗ではなく、
    2点ずつの傾きの中央値（Theil–Sen）を使い、跳びの影響を抑える。
    """
    if len(points) < 3:
        return None
    slopes = [
        (d2 - d1) / (t2 - t1)
        for i, (t1, d1) in enumerate(points)
        for t2, d2 in points[i + 1 :]
        if t2 != t1
    ]
    if not slopes:
        return None
    return -statistics.median(slopes)


def drive(
    client: PicarxClient,
    speed: int,
    seconds: float,
    on_tick: Optional[Callable[[float], None]] = None,
) -> None:
    """前進の指令を TTL 付きで送り続ける。Pi が拒否したら止まる"""
    start = time.monotonic()
    while (elapsed := time.monotonic() - start) < seconds:
        if not client.move_sync(DriveDirection.FORWARD, speed=speed, ttl_ms=MOVE_TTL_MS):
            reason = client.get_status_sync().blocked.forward
            raise TooClose(f"Pi が前進を拒否しました: {reason}（{floor_detail(client)}）")
        if on_tick is not None:
            on_tick(elapsed)
        time.sleep(MOVE_PERIOD_SEC)


def measure_noise(client: PicarxClient) -> Dict:
    """止まったまま距離を読み、揺れの大きさ（最大と最小の差）を出す"""
    values = []
    for _ in range(NOISE_SAMPLES):
        distance = client.get_status_sync().sensors.distance
        if distance.cm is not None and not distance.far:
            values.append(distance.cm)
        time.sleep(0.05)
    if len(values) < NOISE_SAMPLES // 2:
        raise TooClose("静止中に距離が読めません。壁へまっすぐ向けて置き直してください")
    return {
        "median_cm": round(statistics.median(values), 2),
        "spread_cm": round(max(values) - min(values), 2),
    }


def floor_detail(client: PicarxClient) -> str:
    floor = client.get_status_sync().sensors.floor
    return f"床センサー {floor.values}、段差の判定 {floor.cliff}"


def ensure_room(client: PicarxClient) -> float:
    distance = read_distance(client)
    if distance is None or distance < MIN_START_DISTANCE_CM:
        shown = "読めません" if distance is None else f"{distance:.0f}cm"
        raise TooClose(f"壁までの距離が {shown}。壁の 1〜1.5m 手前に置き直してください")
    return distance


def start_trial(client: PicarxClient, speed: int, moved_cm: float = MOVED_CM) -> Dict:
    """止まった状態から走らせ、進んだ距離を測る"""
    time.sleep(SETTLE_SEC)
    before = ensure_room(client)
    try:
        drive(client, speed, TRIAL_SEC)
    finally:
        client.stop_sync()
    time.sleep(SETTLE_SEC / 2)
    after = read_distance(client)
    moved = None if after is None else round(before - after, 1)
    return {
        "mode": "start",
        "speed": speed,
        "before_cm": before,
        "after_cm": after,
        "moved_cm": moved,
        "moving": moved is not None and moved >= moved_cm,
    }


def rolling_trial(client: PicarxClient, speed: int, moving_cm_s: float = MOVING_CM_S) -> Dict:
    """動くと分かっている速度で走らせてから落とし、落とした後の速さを測る"""
    time.sleep(SETTLE_SEC)
    before = ensure_room(client)
    # (段階, 経過秒, 距離)。段階は "from"（先に走らせる速度）と "to"（落とした後）
    series: List[Tuple[str, float, float]] = []

    def sampler(phase: str) -> Callable[[float], None]:
        def sample(elapsed: float) -> None:
            distance = client.get_status_sync().sensors.distance
            if distance.cm is not None and not distance.far:
                series.append((phase, round(elapsed, 3), distance.cm))

        return sample

    try:
        drive(client, ROLL_FROM_SPEED, ROLL_FROM_SEC, on_tick=sampler("from"))
        drive(client, speed, TRIAL_SEC, on_tick=sampler("to"))
    finally:
        client.stop_sync()
    time.sleep(SETTLE_SEC / 2)
    after = read_distance(client)
    to_points = [(t, d) for phase, t, d in series if phase == "to" and t >= ROLL_SKIP_SEC]
    from_points = [(t, d) for phase, t, d in series if phase == "from"]
    rate = approach_rate(to_points)
    from_rate = approach_rate(from_points)
    return {
        "mode": "rolling",
        "speed": speed,
        "from_speed": ROLL_FROM_SPEED,
        "before_cm": before,
        "after_cm": after,
        "moved_cm": None if after is None else round(before - after, 1),
        "from_approach_cm_s": None if from_rate is None else round(from_rate, 1),
        "samples": len(to_points),
        "approach_cm_s": None if rate is None else round(rate, 1),
        # 先に走らせた速度で進んでいなければ、落とした後の判定は意味がない
        "warmed_up": from_rate is not None and from_rate >= moving_cm_s,
        "moving": rate is not None and rate >= moving_cm_s,
        "series": series,
    }


def trial_line(trial: Dict) -> str:
    """1回の試行の要約。距離の前後も出し、おかしな値に気づけるようにする"""
    verdict = "動く" if trial["moving"] else "動かない"
    span = f"{trial['before_cm']} → {trial['after_cm']}cm（{trial['moved_cm']}cm）"
    if trial["mode"] == "start":
        return f"{verdict}: {span}"
    warm = "" if trial["warmed_up"] else "。先の速度で進んでいないので判定は無効"
    return (
        f"{verdict}: 落とした後 {trial['approach_cm_s']}cm/s、"
        f"{trial['from_speed']} の間 {trial['from_approach_cm_s']}cm/s、{span}{warm}"
    )


def lowest_moving(trials: List[Dict], mode: str) -> Optional[int]:
    """その方法で動いた最低の速度。

    それより速い速度で動かなかった試行があれば、境目が揺れているので None。
    走行中の試行のうち、先の速度で進んでいなかったものは使わない。
    """
    rows = sorted(
        (t for t in trials if t["mode"] == mode and t.get("warmed_up", True)),
        key=lambda t: t["speed"],
    )
    moving = [t["speed"] for t in rows if t["moving"]]
    if not moving:
        return None
    lowest = moving[0]
    if any(not t["moving"] for t in rows if t["speed"] > lowest):
        return None
    return lowest


def print_table(trials: List[Dict]) -> None:
    print("\n速度 | 発進（進んだ距離） | 走行中（接近の速さ）")
    for speed in sorted({t["speed"] for t in trials}):
        cells = []
        for mode, key, unit in (("start", "moved_cm", "cm"), ("rolling", "approach_cm_s", "cm/s")):
            row = next((t for t in trials if t["mode"] == mode and t["speed"] == speed), None)
            if row is None:
                cells.append("-")
            elif row[key] is None:
                cells.append("不明")
            else:
                cells.append(f"{'動く' if row['moving'] else '動かない'} {row[key]:.1f}{unit}")
        print(f"{speed:4d} | {cells[0]} | {cells[1]}")


def run(client: PicarxClient, speeds: Sequence[int], modes: Sequence[str]) -> Dict:
    battery = client.get_status_sync().sensors.battery
    print(f"電池 {battery.voltage}V（{battery.percent}%）")
    trials: List[Dict] = []
    aborted = None
    noise = None
    moved_cm, moving_cm_s = MOVED_CM, MOVING_CM_S
    try:
        noise = measure_noise(client)
        moved_cm = max(MOVED_CM, NOISE_FACTOR * noise["spread_cm"])
        # 速さは約1秒分の傾きなので、揺れの幅をその時間で割った値と比べる
        moving_cm_s = max(MOVING_CM_S, noise["spread_cm"] / (TRIAL_SEC - ROLL_SKIP_SEC))
        print(
            f"静止中の距離 {noise['median_cm']}cm、揺れ {noise['spread_cm']}cm。"
            f"進んだとみなす値: 発進 {moved_cm:.1f}cm、走行中 {moving_cm_s:.1f}cm/s"
        )
        for speed in sorted(speeds):
            for mode in modes:
                label = "発進" if mode == "start" else f"走行中（{ROLL_FROM_SPEED} から）"
                print(f"速度 {speed}、{label}...", flush=True)
                if mode == "start":
                    trial = start_trial(client, speed, moved_cm)
                else:
                    trial = rolling_trial(client, speed, moving_cm_s)
                print(f"  {trial_line(trial)}", flush=True)
                trials.append(trial)
    except TooClose as exc:
        aborted = str(exc)
        print(f"中断: {aborted}")
    except httpx.HTTPError as exc:
        aborted = f"通信エラー: {exc}"
        print(f"中断: {aborted}")
    finally:
        try:
            client.stop_sync()
        except httpx.HTTPError:
            pass

    print_table(trials)
    result = {
        "battery_v": battery.voltage,
        "noise": noise,
        "moved_cm_threshold": moved_cm,
        "moving_cm_s_threshold": moving_cm_s,
        "trials": trials,
        "lowest_start": lowest_moving(trials, "start"),
        "lowest_rolling": lowest_moving(trials, "rolling"),
        "aborted": aborted,
    }
    for mode, name in (("start", "発進"), ("rolling", "走行中")):
        lowest = result[f"lowest_{mode}"]
        if any(t["mode"] == mode for t in trials):
            print(f"{name}で動いた最低の速度: {lowest if lowest is not None else '判定できず'}")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="車輪が回る最低の速度を測る（車両が動く）")
    parser.add_argument("--speeds", type=int, nargs="+", default=[30, 35, 40, 45])
    parser.add_argument(
        "--mode", choices=["both", "start", "rolling"], default="both", help="測る方法"
    )
    parser.add_argument("--note", type=str, default=None, help="床の種類などのメモ")
    parser.add_argument(
        "--i-understand-the-car-will-move",
        action="store_true",
        required=True,
        help="車両が壁へ向かって前進することを了承する",
    )
    args = parser.parse_args(argv)
    if any(not 0 < s <= 100 for s in args.speeds):
        parser.error("--speeds は 1〜100 で指定してください")
    modes = ["start", "rolling"] if args.mode == "both" else [args.mode]

    result = run(PicarxClient(timeout=CONTROL_TIMEOUT_SEC), args.speeds, modes)
    if args.note:
        result["note"] = args.note
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_speed.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"保存: {path}")
    return 0 if result["aborted"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
