"""記録した応答時間を、会話の種類とモデルごとに集計する。

uv run python -m src.metrics.turn_report
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.metrics.turn_timing import timings_path

COLUMNS = (
    ("stt_sec", "文字起こし"),
    ("ack_audio_sec", "相づち"),
    ("stopped_sec", "停止"),
    ("reply_ready_sec", "返事の作成"),
    ("reply_audio_sec", "返事の流れ始め"),
)


def load(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def percentile(values: List[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return ordered[index]


def summarize(records: Iterable[dict]) -> Dict[Tuple[str, str, str], Dict[str, Optional[str]]]:
    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for record in records:
        if record.get("ok"):
            key = (
                record.get("kind") or "-",
                record.get("model") or "-",
                record.get("input") or "-",
            )
            groups[key].append(record)

    table = {}
    for key, rows in sorted(groups.items()):
        cells: Dict[str, Optional[str]] = {"n": str(len(rows))}
        for column, _ in COLUMNS:
            values = [row[column] for row in rows if row.get(column) is not None]
            cells[column] = (
                f"{statistics.median(values):.1f} / {percentile(values, 0.9):.1f}"
                if values
                else None
            )
        table[key] = cells
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--file", type=Path, default=None)
    args = parser.parse_args()
    path = args.file or timings_path()
    table = summarize(load(path))
    if not table:
        print(f"記録がありません: {path}")
        return

    header = ["種類", "モデル", "入力", "回数"] + [label for _, label in COLUMNS]
    print("値は秒。中央値 / 90パーセンタイル。入力が終わった時からの時間")
    print(" | ".join(header))
    for (kind, model, input_kind), cells in table.items():
        row = [kind, model, input_kind, cells["n"]]
        row += [cells[column] or "-" for column, _ in COLUMNS]
        print(" | ".join(row))


if __name__ == "__main__":
    main()
