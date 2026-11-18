"""
記録したセッションの動きを後から推定し、指令との食い違いを数える

走行データは追記のみなので、結果はセッションの中ではなく `data/derived/motion/` へ書く。
しきい値を変えて試す時や、試験走行の記録を確かめる時に使う。

使い方:
    uv run python -m src.e2e.motion.analyze --data_dir data/raw/<session_id>
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2

try:
    from ..dataset.manifest import find_session_dirs
    from .checker import Command, MotionChecker
    from .estimator import MotionTracker
except ImportError:
    from src.e2e.dataset.manifest import find_session_dirs
    from src.e2e.motion.checker import Command, MotionChecker
    from src.e2e.motion.estimator import MotionTracker

MAX_STEERING_DEG = 35.0
# 旧形式の記録には速度の指令値がないので、収集 GUI の既定値とみなす
LEGACY_SPEED = 40


def command_from_record(record: dict) -> Tuple[int, float, bool]:
    """(速度, 舵角, 旧形式か) を返す。新しい記録は `command` を持つ"""
    command = record.get("command")
    if command:
        return int(command.get("speed", 0)), float(command.get("steering_deg", 0.0)), False
    # 旧形式: throttle は前進の指令後に止まっても 0.5 のまま（LRN-6）なので信用できない
    speed = LEGACY_SPEED if float(record.get("throttle", 0.0)) > 0 else 0
    return speed, float(record.get("steering", 0.0)) * MAX_STEERING_DEG, True


def iter_records(session_dir: Path) -> Iterator[dict]:
    with open(session_dir / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def analyze_session(
    session_dir: Path, tracker: Optional[MotionTracker] = None, checker=None
) -> Tuple[List[dict], Dict[str, int]]:
    tracker = tracker or MotionTracker()
    checker = checker or MotionChecker()
    results: List[dict] = []
    counts: Counter = Counter()
    legacy_seen = False
    last_command: Optional[Tuple[int, float]] = None
    command_start = 0.0

    for record in iter_records(session_dir):
        frame_path = session_dir / "images" / record["frame"]
        frame = cv2.imread(str(frame_path))
        if frame is None:
            counts["unreadable"] += 1
            continue
        timestamp = float(record["timestamp"])
        speed, steering, legacy = command_from_record(record)
        legacy_seen |= legacy

        # 記録に since_s がなければ、指令が変わった時刻から数える
        if (speed, steering) != last_command:
            last_command = (speed, steering)
            command_start = timestamp
        since = record.get("command", {}).get("since_s", timestamp - command_start)

        sample = tracker.update(frame, timestamp, record.get("distance_cm"))
        if sample is None:
            counts["warming_up"] += 1
            continue
        check = checker.check(Command(speed, steering, since), sample)
        counts["frames"] += 1
        for flag in check.flags:
            counts[flag] += 1
        if check.unknown:
            counts[f"unknown:{check.unknown}"] += 1
        results.append(
            {"frame": record["frame"], "motion": sample.to_record(), "check": check.to_record()}
        )

    counts["legacy_commands"] = int(legacy_seen)
    return results, dict(counts)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Estimate motion and find command mismatches")
    parser.add_argument("--data_dir", required=True, help="Session or parent folder")
    parser.add_argument("--output_dir", default="data/derived/motion")
    parser.add_argument("--interval", type=float, default=0.3, help="Seconds between frames")
    args = parser.parse_args(argv)

    sessions = find_session_dirs(args.data_dir)
    if not sessions:
        print(f"No sessions with labels.jsonl found in: {args.data_dir}")
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        results, counts = analyze_session(session, MotionTracker(interval_s=args.interval))
        out_file = out_dir / f"{session.name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{session.name}: {counts}")
        if counts.get("legacy_commands"):
            print("  note: old records without commands; no_motion is overcounted (LRN-6)")
        print(f"  -> {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
