"""モデル別の介入率（ROADMAP の M3「主指標を 10分あたりの介入回数に変更し、モデル間で比較」）

`data/runs/<run_id>/run.json` を読み、モデルの版ごとに、自律で走った時間、人の介入の回数
（パッドでの引き継ぎ、Ctrl+C）、安全のための自動停止の回数と、それぞれの10分あたりの回数を出す。
自律で走った時間が短いと率がぶれるので、合計の時間も一緒に見ること。

使い方:
    uv run python -m src.e2e.evaluation.interventions --runs_dir data/runs
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PER_SECONDS = 600.0  # 10分


def load_runs(runs_dir) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(Path(runs_dir).glob("*/run.json")):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, ValueError) as exc:
            print(f"Skipping {path}: {exc}", file=sys.stderr)
    return runs


def per_ten_minutes(count: int, seconds: float) -> Optional[float]:
    return round(count * PER_SECONDS / seconds, 2) if seconds > 0 else None


def summarize(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """モデルの版ごとの集計。学習に使ったデータの版が違えば別の行にする"""
    groups: Dict[tuple, Dict[str, Any]] = {}
    for run in runs:
        key = (run.get("model_version"), run.get("data_fingerprint"))
        group = groups.setdefault(
            key,
            {
                "model_version": key[0],
                "data_fingerprint": key[1],
                "runs": 0,
                "autonomous_s": 0.0,
                "human_s": 0.0,
                "human": 0,
                "safety": 0,
            },
        )
        counts = run.get("interventions") or {}
        group["runs"] += 1
        group["autonomous_s"] += float(run.get("autonomous_s") or 0.0)
        group["human_s"] += float(run.get("human_s") or 0.0)
        group["human"] += int(counts.get("human") or 0)
        group["safety"] += int(counts.get("safety") or 0)

    rows = []
    for group in groups.values():
        rows.append(
            {
                **group,
                "autonomous_min": round(group["autonomous_s"] / 60.0, 2),
                "human_per_10min": per_ten_minutes(group["human"], group["autonomous_s"]),
                "safety_per_10min": per_ten_minutes(group["safety"], group["autonomous_s"]),
            }
        )
    # 人の介入が少ない順（時間がないものは最後）
    rows.sort(key=lambda r: (r["human_per_10min"] is None, r["human_per_10min"] or 0.0))
    return rows


def format_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        f"{'model':<12} {'data':<10} {'runs':>4} {'auto min':>9} "
        f"{'human':>6} {'/10min':>7} {'safety':>7} {'/10min':>7}"
    )
    lines = [header, "-" * len(header)]
    for r in rows:
        data = (r["data_fingerprint"] or "-")[:10]
        human_rate = "-" if r["human_per_10min"] is None else f"{r['human_per_10min']:.2f}"
        safety_rate = "-" if r["safety_per_10min"] is None else f"{r['safety_per_10min']:.2f}"
        lines.append(
            f"{str(r['model_version']):<12} {data:<10} {r['runs']:>4} {r['autonomous_min']:>9.2f} "
            f"{r['human']:>6} {human_rate:>7} {r['safety']:>7} {safety_rate:>7}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Interventions per 10 minutes by model")
    parser.add_argument("--runs_dir", default="data/runs")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = parser.parse_args(argv)
    runs = load_runs(args.runs_dir)
    if not runs:
        print(f"No runs in {args.runs_dir}")
        return 1
    rows = summarize(runs)
    print(json.dumps(rows, indent=2, ensure_ascii=False) if args.json else format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
