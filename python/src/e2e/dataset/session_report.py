"""収集したセッションの中身を確かめる（ROADMAP の M3「10fps 以上で再収集」）。

学習に使える走りになっているかを、セッションごとに数値と注意で出す。

    uv run python -m src.e2e.dataset.session_report --data_dir data/raw
    uv run python -m src.e2e.dataset.session_report --data_dir data/raw/20261110_210000

旧形式（`command` を持たない記録）のセッションは数えるだけで、注意は出さない。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

try:
    from ..training.dataset import is_stopped, resolve_session_dirs
except ImportError:
    from src.e2e.training.dataset import is_stopped, resolve_session_dirs

MAX_STEERING_DEG = 35.0
# 録画を一時停止したとみなす間隔。これより長い間は fps の計算に入れない
PAUSE_SEC = 1.0
# これより大きい舵角（正規化）を「最大近く」とみなす
SATURATED = 0.85
STRAIGHT = 0.05

# 注意を出す目安
MIN_FPS = 10.0
MIN_MOVING_RATIO = 0.5
MIN_TURN_BALANCE = 0.5  # 少ない方の旋回 / 多い方の旋回
MAX_SATURATED_RATIO = 0.5


def load_records(session_dir: Path) -> List[dict]:
    with open(session_dir / "labels.jsonl", "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def direction(steering: float) -> str:
    if steering < -STRAIGHT:
        return "left"
    if steering > STRAIGHT:
        return "right"
    return "straight"


def report_session(records: List[dict]) -> Dict:
    """1セッション分の数値をまとめる"""
    legacy = not any("command" in r for r in records)
    times = [r["timestamp"] for r in records if "timestamp" in r]
    gaps = [b - a for a, b in zip(times, times[1:])]
    recorded = sum(g for g in gaps if g <= PAUSE_SEC)
    intervals = sum(1 for g in gaps if g <= PAUSE_SEC)
    fps = intervals / recorded if recorded > 0 else None

    moving = [r for r in records if not is_stopped(r)]
    flagged = [r for r in records if (r.get("motion_check") or {}).get("flags")]
    usable = [r for r in moving if r not in flagged]
    directions = Counter(direction(float(r.get("steering", 0.0))) for r in usable)
    saturated = sum(1 for r in usable if abs(float(r.get("steering", 0.0))) >= SATURATED)
    turns = (directions["left"], directions["right"])

    # 一時停止の間も配信は進むので、一時停止をまたぐ番号の飛びは取りこぼしに数えない
    skipped = sum(
        b["stream_seq"] - a["stream_seq"] - 1
        for a, b in zip(records, records[1:])
        if a.get("stream_seq") is not None
        and b.get("stream_seq") is not None
        and b.get("timestamp", 0) - a.get("timestamp", 0) <= PAUSE_SEC
        and b["stream_seq"] - a["stream_seq"] > 1
    )
    flags = Counter(f for r in flagged for f in r["motion_check"]["flags"])

    return {
        "legacy": legacy,
        "frames": len(records),
        "recorded_sec": round(recorded, 1),
        "pauses": sum(1 for g in gaps if g > PAUSE_SEC),
        "fps": None if fps is None else round(fps, 1),
        "skipped_stream_frames": skipped,
        "moving": len(moving),
        "moving_ratio": round(len(moving) / len(records), 3) if records else 0.0,
        "flagged": len(flagged),
        "flags": dict(flags),
        "usable": len(usable),
        "directions": {k: directions[k] for k in ("left", "straight", "right")},
        "turn_balance": round(min(turns) / max(turns), 2) if max(turns) else None,
        # 意図ごとのフレーム数（学習に使えるフレーム）。意図を記録していない旧形式は any に数える
        "intents": dict(sorted(Counter(r.get("intent", "any") for r in usable).items())),
        "saturated_ratio": round(saturated / len(usable), 3) if usable else None,
        "steering_deg_histogram": dict(
            sorted(
                Counter(
                    int(round(float(r.get("steering", 0.0)) * MAX_STEERING_DEG / 5.0) * 5)
                    for r in usable
                ).items()
            )
        ),
        "missing": {
            key: sum(1 for r in records if r.get(key) is None)
            for key in ("floor", "battery_v")
            if not legacy
        },
    }


def warnings(report: Dict) -> List[str]:
    """学習に使う前に直したほうがよい点。旧形式のセッションには出さない"""
    if report["legacy"]:
        return []
    notes = []
    if report["fps"] is not None and report["fps"] < MIN_FPS:
        notes.append(f"fps が {report['fps']}（{MIN_FPS:.0f} 以上で集める）")
    if report["moving_ratio"] < MIN_MOVING_RATIO:
        notes.append(
            f"走っているフレームが {report['moving_ratio']:.0%}。止まっている間は録画を止める"
            "（止まっているフレームは学習に使わない）"
        )
    balance = report["turn_balance"]
    if balance is not None and balance < MIN_TURN_BALANCE:
        d = report["directions"]
        notes.append(f"左右の偏り: 左 {d['left']}、右 {d['right']}。少ない方の向きを足す")
    saturated = report["saturated_ratio"]
    if saturated is not None and saturated > MAX_SATURATED_RATIO:
        notes.append(f"舵角が最大近く（±30°以上）のフレームが {saturated:.0%}。なめらかに切る")
    missing = {k: v for k, v in report["missing"].items() if v}
    if missing:
        notes.append(f"センサー値の欠け: {missing}")
    return notes


def print_report(name: str, report: Dict) -> None:
    kind = "（旧形式）" if report["legacy"] else ""
    print(f"\n== {name}{kind}")
    fps = report["fps"]
    print(
        f"フレーム {report['frames']}、録画 {report['recorded_sec']}秒、"
        f"{fps if fps is not None else '-'} fps、一時停止 {report['pauses']}回、"
        f"配信の取りこぼし {report['skipped_stream_frames']}"
    )
    print(
        f"走行中 {report['moving']}（{report['moving_ratio']:.0%}）、"
        f"食い違いの印 {report['flagged']} {report['flags'] or ''}、学習に使える {report['usable']}"
    )
    d = report["directions"]
    print(
        f"向き: 左 {d['left']}、直進 {d['straight']}、右 {d['right']}"
        f"（左右の比 {report['turn_balance']}）、最大近く {report['saturated_ratio']}"
    )
    print(f"舵角（5°刻み）: {report['steering_deg_histogram']}")
    print(f"意図: {report['intents']}")
    notes = warnings(report)
    for note in notes:
        print(f"  注意: {note}")
    if not notes and not report["legacy"]:
        print("  問題なし")


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="収集したセッションの中身を確かめる")
    parser.add_argument("--data_dir", default="data/raw", help="セッションか、その親フォルダ")
    args = parser.parse_args(argv)

    total = Counter()
    for session_dir in resolve_session_dirs(args.data_dir):
        records = load_records(session_dir)
        if not records:
            print(f"\n== {session_dir.name}: 記録なし")
            continue
        report = report_session(records)
        print_report(session_dir.name, report)
        total["frames"] += report["frames"]
        total["usable"] += report["usable"]
        for key, value in report["directions"].items():
            total[key] += value
    print(
        f"\n合計: フレーム {total['frames']}、学習に使える {total['usable']}"
        f"（左 {total['left']}、直進 {total['straight']}、右 {total['right']}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
