"""
M3「指令と実際の動きの食い違い」を実機で計測し、しきい値を校正する

場面ごとに数秒走らせ（または止めたまま）、推定した動きと判定を記録する。
記録は `data/m3_checks/` に保存する（画像は含まない）。

    uv run python -m src.e2e.m3_check run --scenario still
    uv run python -m src.e2e.m3_check run --scenario forward --i-understand-the-car-will-move
    uv run python -m src.e2e.m3_check calibrate --still <記録> --moving <記録> ...

場面:
    still    止めたまま（雑音の大きさ）
    walk     止めたまま、前を人が横切る（印が付かないこと）
    push     止めたまま、車体を手で押す（uncommanded_motion が付くこと）
    forward  直進（印が付かないこと。舵0での曲がり癖）
    left     左へ舵を切って前進（向きの変化が負）
    right    右へ舵を切って前進（向きの変化が正）
    lifted   車体を持ち上げて車輪を浮かせ、前進の指令（no_motion で止まること）
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import httpx
import numpy as np

from src.e2e.motion.monitor import CommandState, MotionMonitor, load_motion_config
from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

OUTPUT_DIR = Path("data/m3_checks")
MOVE_TTL_MS = 400
CONTROL_TIMEOUT_SEC = 1.0
FRAME_TIMEOUT_SEC = 1.0
# 車輪を浮かせた時、指令から止まるまでの目標（判定の待ち 0.5 秒 + 継続 1 秒 + 余裕）
LIFTED_STOP_TARGET_SEC = 2.5

MOVING = {"forward": 0, "left": -30, "right": 30, "lifted": 0}
STILL = {"still", "walk", "push"}
SCENARIOS = sorted(set(MOVING) | STILL)

# 場面ごとに期待する印。None は「印が付かないこと」
EXPECTED_FLAG = {"push": "uncommanded_motion", "lifted": "no_motion"}


def percentile(values: List[float], q: float) -> Optional[float]:
    return round(float(np.percentile(values, q)), 3) if values else None


def summarize(records: List[dict], scenario: str, speed: int, steer: int) -> Dict:
    """記録から、しきい値の校正と合否の判断に使う値をまとめる"""
    valid = [r for r in records if r["motion"]["image_valid"]]
    flows = [r["motion"]["edge_flow_px_s"] for r in valid]
    yaws = [r["motion"]["yaw_rate_dps"] for r in valid]
    approaches = [
        r["motion"]["approach_cm_s"] for r in records if r["motion"]["approach_cm_s"] is not None
    ]
    flags = Counter(f for r in records for f in r["motion_check"]["flags"])
    unknown = Counter(r["motion_check"].get("unknown") for r in records)
    unknown.pop(None, None)

    # 向きの変化を時間で積み上げ、場面全体で曲がった角度にする
    turned = 0.0
    for prev, cur in zip(valid, valid[1:]):
        turned += cur["motion"]["yaw_rate_dps"] * (cur["t"] - prev["t"])

    duration = records[-1]["t"] - records[0]["t"] if len(records) > 1 else 0.0
    return {
        "scenario": scenario,
        "speed": speed,
        "steering_deg": steer,
        "samples": len(records),
        "rate_hz": round((len(records) - 1) / duration, 1) if duration > 0 else None,
        "image_valid_ratio": round(len(valid) / len(records), 3) if records else 0.0,
        "edge_flow_px_s": {f"p{q}": percentile(flows, q) for q in (10, 50, 90, 99)},
        "yaw_rate_dps": {f"p{q}": percentile(yaws, q) for q in (5, 50, 95)},
        "turned_deg": round(turned, 1),
        "approach_cm_s_median": round(statistics.median(approaches), 2) if approaches else None,
        "flags": dict(flags),
        "unknown": dict(unknown),
        "first_sustained": next(
            (
                {"flag": r["motion_check"]["sustained"], "t": round(r["t"], 2)}
                for r in records
                if r["motion_check"].get("sustained")
            ),
            None,
        ),
    }


def judge(summary: Dict) -> List[str]:
    """場面ごとの合否の目安。校正前は不合格もありうるので、結果は記録して判断に使う"""
    scenario = summary["scenario"]
    notes = []
    flags = summary["flags"]
    expected = EXPECTED_FLAG.get(scenario)
    if expected:
        first = summary["first_sustained"]
        if first and first["flag"] == expected:
            if scenario == "lifted" and first["t"] > LIFTED_STOP_TARGET_SEC:
                notes.append(
                    f"NG: {expected} まで {first['t']} 秒（目標 {LIFTED_STOP_TARGET_SEC}）"
                )
            else:
                notes.append(f"OK: {expected} が {first['t']} 秒で続いた")
        else:
            notes.append(f"NG: {expected} が続きませんでした（印: {flags}）")
    elif flags:
        notes.append(f"NG: 印が付きました {flags}")
    else:
        notes.append("OK: 印なし")

    yaw = summary["yaw_rate_dps"]["p50"]
    if scenario == "left" and yaw is not None and yaw >= 0:
        notes.append(f"NG: 左へ舵を切ったのに向きの変化が {yaw:+.1f}°/s")
    if scenario == "right" and yaw is not None and yaw <= 0:
        notes.append(f"NG: 右へ舵を切ったのに向きの変化が {yaw:+.1f}°/s")
    if summary["image_valid_ratio"] < 0.8:
        notes.append(
            f"注意: 画像で判断できた割合が {summary['image_valid_ratio']:.0%}。"
            "模様の少ない場所を避けてください"
        )
    return notes


def run_scenario(
    client: PicarxClient, scenario: str, speed: int, steer: Optional[int], seconds: float
) -> Dict:
    tracker, config = load_motion_config()
    monitor = MotionMonitor(tracker=tracker, config=config)
    moving = scenario in MOVING
    steer = MOVING[scenario] if steer is None and moving else (steer or 0)
    state = CommandState(time.time())
    records: List[dict] = []
    stopped_by_check = False
    refused = None

    detail = f"、速度 {speed}、舵 {steer:+d}°" if moving else ""
    print(f"場面: {scenario}、{seconds:.0f} 秒{detail}")
    stream = client.open_camera_stream().start()
    last_seq = 0
    try:
        if moving:
            client.steer_sync(steer)
            state.steer(steer, time.time())
        start = time.time()
        while time.time() - start < seconds:
            frame_msg = stream.wait_for_frame(last_seq, timeout=FRAME_TIMEOUT_SEC)
            if frame_msg is None:
                print(f"\r[NO FRAME] {stream.last_error}", end="", flush=True)
                continue
            last_seq = frame_msg.seq
            frame = cv2.imdecode(np.frombuffer(frame_msg.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if moving:
                if not client.move_sync(DriveDirection.FORWARD, speed=speed, ttl_ms=MOVE_TTL_MS):
                    refused = client.get_status_sync().blocked.forward
                    print(f"\nPi が前進を拒否しました: {refused}")
                    break
                state.move(speed, MOVE_TTL_MS / 1000.0, time.time())

            status = client.get_status_sync()
            distance = None if status.sensors.distance.far else status.sensors.distance.cm
            now = time.time()
            update = monitor.update(frame, now, distance, state.at(now))
            if update is None:
                continue
            records.append({"t": round(now - start, 3), **update.to_record()})
            check = update.check
            print(
                f"\r t={now - start:4.1f}s flow {update.sample.edge_flow_px_s:5.1f}px/s "
                f"yaw {update.sample.yaw_rate_dps:+6.1f}°/s "
                f"flags {check.flags or '-'} {check.unknown or ''}        ",
                end="",
                flush=True,
            )
            if update.newly_sustained and moving:
                # 実際の走行と同じく、食い違いが続いたら止める
                client.stop_sync()
                state.stop(time.time())
                stopped_by_check = True
                print(f"\n{update.newly_sustained} が続いたので止めました")
                break
    finally:
        try:
            client.stop_sync()
            if moving:
                client.steer_sync(0)
        except httpx.HTTPError as exc:
            print(f"\n停止の送信に失敗: {exc}")
        stream.stop()
    print()

    if not records:
        raise RuntimeError("推定値が得られませんでした（カメラの配信を確認してください）")
    summary = summarize(records, scenario, speed if moving else 0, steer if moving else 0)
    summary["stopped_by_check"] = stopped_by_check
    summary["refused"] = refused
    summary["config"] = {
        "hfov_deg": tracker.deg_per_px * 160,
        "moving_px_s": tracker.moving_px_s,
        "interval_s": tracker.interval_s,
    }
    return {"summary": summary, "records": records}


def save_result(result: Dict, scenario: str, note: Optional[str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{scenario}.json"
    if note:
        result["summary"]["note"] = note
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
        f.write("\n")
    return path


def print_summary(summary: Dict) -> None:
    flow = summary["edge_flow_px_s"]
    yaw = summary["yaw_rate_dps"]
    print(f"推定の回数: {summary['samples']}（{summary['rate_hz']} Hz）")
    print(f"画像で判断できた割合: {summary['image_valid_ratio']:.0%}")
    print("画像の動き（px/s）: " + ", ".join(f"{k} {v}" for k, v in flow.items()))
    print("向きの変化（°/s）: " + ", ".join(f"{k} {v}" for k, v in yaw.items()))
    print(f"曲がった角度の推定: {summary['turned_deg']}°")
    print(f"超音波の接近速度（中央値）: {summary['approach_cm_s_median']} cm/s")
    print(f"印: {summary['flags'] or 'なし'}、不明: {summary['unknown'] or 'なし'}")
    for line in judge(summary):
        print(line)


def load_summary(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def calibrate(args) -> int:
    """記録から、configs/motion.toml に書く値を提案する"""
    suggestions = []

    if args.still and args.moving:
        quiet = max(load_summary(p)["edge_flow_px_s"]["p99"] or 0 for p in args.still)
        moving = min(load_summary(p)["edge_flow_px_s"]["p10"] or 0 for p in args.moving)
        print(f"止まっている時の p99 の最大: {quiet} px/s")
        print(f"走っている時の p10 の最小: {moving} px/s")
        if quiet < moving:
            value = round(float(np.sqrt(max(quiet, 0.1) * moving)), 1)
            suggestions.append(f"[tracker] moving_px_s = {value}")
        else:
            print("  止まっている時と走っている時が重なっています。比べる時間差を長くしてください")

    if args.still and args.push:
        quiet = max(load_summary(p)["edge_flow_px_s"]["p99"] or 0 for p in args.still)
        pushed = min(load_summary(p)["edge_flow_px_s"]["p50"] or 0 for p in args.push)
        print(f"止まっている時の p99: {quiet} px/s、押した時の p50: {pushed} px/s")
        if quiet < pushed:
            value = round(float(np.sqrt(max(quiet, 0.1) * pushed)), 1)
            suggestions.append(f"[checker] uncommanded_px_s = {value}")

    if args.turn and args.measured_deg:
        summary = load_summary(args.turn)
        estimated = abs(summary["turned_deg"])
        if estimated > 0:
            hfov = summary["config"]["hfov_deg"] * abs(args.measured_deg) / estimated
            print(f"推定 {estimated}°、実測 {args.measured_deg}°")
            suggestions.append(f"[tracker] hfov_deg = {hfov:.1f}")

    if args.straight:
        summary = load_summary(args.straight)
        drift = summary["yaw_rate_dps"]
        print("舵0での向きの変化（°/s）: " + ", ".join(f"{k} {v}" for k, v in drift.items()))
        edge = max(abs(drift["p5"] or 0), abs(drift["p95"] or 0))
        suggestions.append(f"[checker] straight_max_rate_dps = {max(15.0, round(edge * 1.5, 1))}")

    if args.wall:
        summary = load_summary(args.wall)
        approach = summary["approach_cm_s_median"]
        if approach and summary["speed"]:
            per_unit = approach / summary["speed"]
            print(f"速度 {summary['speed']} で {approach} cm/s")
            suggestions.append(f"[checker] speed_cm_s_per_unit = {per_unit:.3f}")

    if not suggestions:
        print("提案できる値がありません。記録の組み合わせを確認してください（--help）")
        return 1
    print("\nconfigs/motion.toml に書く値の提案:")
    for line in suggestions:
        print(f"  {line}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="M3 の動きの推定を実機で計測する")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="場面ごとに計測する")
    run.add_argument("--scenario", choices=SCENARIOS, required=True)
    run.add_argument("--speed", type=int, default=50)
    run.add_argument("--steer", type=int, default=None, help="舵角（既定は場面ごとの値）")
    run.add_argument("--seconds", type=float, default=5.0)
    run.add_argument("--note", type=str, default=None, help="床の種類や明るさなどのメモ")
    run.add_argument(
        "--i-understand-the-car-will-move",
        action="store_true",
        help="forward、left、right、lifted で車両が前進することを了承する",
    )

    cal = sub.add_parser("calibrate", help="記録から設定の値を提案する")
    cal.add_argument("--still", nargs="+", help="still や walk の記録")
    cal.add_argument("--moving", nargs="+", help="forward、left、right の記録")
    cal.add_argument("--push", nargs="+", help="push の記録")
    cal.add_argument("--turn", help="left か right の記録（--measured_deg と一緒に）")
    cal.add_argument("--measured_deg", type=float, help="その記録で実際に曲がった角度")
    cal.add_argument("--straight", help="forward の記録（舵0での曲がり癖）")
    cal.add_argument("--wall", help="壁へ向かって forward した記録（速度の校正）")

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        return calibrate(args)

    if args.scenario in MOVING and not args.i_understand_the_car_will_move:
        parser.error(
            f"{args.scenario} は車両が前進します。--i-understand-the-car-will-move が必要です"
        )
    client = PicarxClient(timeout=CONTROL_TIMEOUT_SEC)
    result = run_scenario(client, args.scenario, args.speed, args.steer, args.seconds)
    path = save_result(result, args.scenario, args.note)
    print_summary(result["summary"])
    print(f"保存: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
