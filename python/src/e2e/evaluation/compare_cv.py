"""2つの交差検証の結果を比べ、採用してよいかを判定する（ROADMAP の M3「採用の判定（オフライン）」）

`src.e2e.training.cross_validate` の出力（`data/cv/*.json`）を2つ読み、使用中の設定（current）と
候補（candidate）を同じ形式で比べる。同じデータで同じセッションを抜いた結果どうしでなければ
比べない（同じセッションを抜き、同じ数のフレームで評価した結果どうし）。

判定:
- 候補の全体の舵角の MAE が、使用中より `tolerance` を超えて悪ければ採用しない
- 候補が「画像を見ず、意図ごとの平均を答える」基準より悪ければ採用しない
- それ以外は採用してよい。舵角の変化（ふらつき）、行きたい向き別、抜いたセッション別の差は
  判断の材料として出す（最後に採用するのは人）

使い方:
    uv run python -m src.e2e.evaluation.compare_cv \
        --current data/cv/auto_8sessions_turn.json \
        --candidate data/cv/auto_8sessions_turn_flip.json \
        --output data/cv/compare_turn_vs_flip.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ADOPT = "adopt"
KEEP = "keep_current"
NOT_COMPARABLE = "not_comparable"


def load_result(path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _held_out(result: Dict[str, Any]) -> List[str]:
    return [fold["held_out"] for fold in result.get("folds", [])]


def _fold_samples(result: Dict[str, Any]) -> List[Any]:
    return [fold.get("samples") for fold in result.get("folds", [])]


def comparability_problems(current: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    """同じ条件で比べられない理由。空なら比べられる

    同じセッションを抜き、抜いたセッションごとに同じ数のフレームで舵角を評価していれば比べる。
    学習データの指紋（data_fingerprint）は設定を当てた後の学習用のフレームから作るので、曲がりの
    意図の付け直しや、塞がっているかだけに使うフレームでも変わる。違っても止めず、注記にする。
    """
    problems = []
    if not current.get("folds") or not candidate.get("folds"):
        problems.append("no folds")
    if _held_out(current) != _held_out(candidate):
        problems.append("different held-out sessions")
    elif _fold_samples(current) != _fold_samples(candidate):
        problems.append("different number of evaluated samples")
    return problems


def comparability_notes(current: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    a, b = current.get("data_fingerprint"), candidate.get("data_fingerprint")
    if a and b and a != b:
        return [f"training data fingerprints differ ({a[:8]} vs {b[:8]}): check changed params"]
    return []


def changed_params(current: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, List[Any]]:
    a, b = current.get("params", {}), candidate.get("params", {})
    return {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def _by_key(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Dict[str, Optional[float]]]:
    return {
        k: {"current": a.get(k), "candidate": b.get(k), "diff": _diff(a.get(k), b.get(k))}
        for k in list(dict.fromkeys([*a, *b]))
    }


def compare(
    current: Dict[str, Any], candidate: Dict[str, Any], tolerance: float = 0.0
) -> Dict[str, Any]:
    """tolerance: 全体の MAE（正規化した値）で、どこまでの悪化を同じとみなすか"""
    problems = comparability_problems(current, candidate)
    a, b = current.get("overall", {}), candidate.get("overall", {})
    report: Dict[str, Any] = {
        "comparable": not problems,
        "problems": problems,
        "notes": comparability_notes(current, candidate),
        "changed_params": changed_params(current, candidate),
        "code": {"current": current.get("code"), "candidate": candidate.get("code")},
        "overall": {
            "steering_mae": {
                "current": a.get("steering_mae"),
                "candidate": b.get("steering_mae"),
                "diff": _diff(a.get("steering_mae"), b.get("steering_mae")),
            },
            "steering_mae_deg": {
                "current": a.get("steering_mae_deg"),
                "candidate": b.get("steering_mae_deg"),
                "diff": _diff(a.get("steering_mae_deg"), b.get("steering_mae_deg")),
            },
            "steering_change": {
                "current": a.get("steering_change"),
                "candidate": b.get("steering_change"),
                "diff": _diff(a.get("steering_change"), b.get("steering_change")),
            },
        },
        "by_intent": _by_key(
            a.get("steering_mae_by_intent", {}), b.get("steering_mae_by_intent", {})
        ),
        "by_direction": _by_key(
            a.get("steering_mae_by_direction", {}), b.get("steering_mae_by_direction", {})
        ),
        "tolerance": tolerance,
    }

    folds = []
    if not problems:
        for fa, fb in zip(current["folds"], candidate["folds"]):
            folds.append(
                {
                    "held_out": fa["held_out"],
                    "current_deg": fa.get("steering_mae_deg"),
                    "candidate_deg": fb.get("steering_mae_deg"),
                    "diff": _diff(fa.get("steering_mae"), fb.get("steering_mae")),
                }
            )
    report["folds"] = folds
    report["candidate_better_folds"] = sum(
        1 for f in folds if f["diff"] is not None and f["diff"] < 0
    )

    baseline = candidate.get("baselines", {}).get("intent_mean", {}).get("steering_mae")
    report["intent_mean_baseline"] = baseline

    reasons = []
    if problems:
        verdict = NOT_COMPARABLE
        reasons = problems
    else:
        diff = report["overall"]["steering_mae"]["diff"]
        if diff is None:
            reasons.append("overall steering MAE is missing")
        elif diff > tolerance:
            reasons.append(f"overall MAE is worse by {diff:.4f} (tolerance {tolerance:.4f})")
        mae = b.get("steering_mae")
        if baseline is not None and mae is not None and mae >= baseline:
            reasons.append(
                f"not better than the intent_mean baseline ({mae:.4f} >= {baseline:.4f})"
            )
        verdict = KEEP if reasons else ADOPT
    report["verdict"] = verdict
    report["reasons"] = reasons
    return report


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _signed(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:+.4f}"


def format_report(report: Dict[str, Any]) -> str:
    lines = ["=" * 64, "Cross-validation comparison (current -> candidate)", "=" * 64]
    if report["changed_params"]:
        lines.append("Changed params:")
        for key, (a, b) in report["changed_params"].items():
            lines.append(f"  {key}: {a} -> {b}")
    else:
        lines.append("Changed params: none")
    for note in report.get("notes", []):
        lines.append(f"Note: {note}")
    if not report["comparable"]:
        lines.append("NOT COMPARABLE: " + "; ".join(report["problems"]))
        return "\n".join(lines)

    overall = report["overall"]
    lines.append("")
    lines.append(f"{'':<18} {'current':>10} {'candidate':>10} {'diff':>10}")
    for name, key in (("MAE", "steering_mae"), ("MAE (deg)", "steering_mae_deg")):
        row = overall[key]
        digits = 2 if key.endswith("deg") else 4
        lines.append(
            f"{name:<18} {_fmt(row['current'], digits):>10} {_fmt(row['candidate'], digits):>10} "
            f"{_signed(row['diff']):>10}"
        )
    row = overall["steering_change"]
    lines.append(
        f"{'Steering change':<18} {_fmt(row['current']):>10} {_fmt(row['candidate']):>10} "
        f"{_signed(row['diff']):>10}"
    )
    for title, key in (("By intent", "by_intent"), ("By direction", "by_direction")):
        lines.append(f"{title}:")
        for name, row in report[key].items():
            lines.append(
                f"  {name:<16} {_fmt(row['current']):>10} {_fmt(row['candidate']):>10} "
                f"{_signed(row['diff']):>10}"
            )
    lines.append("Folds (deg):")
    for fold in report["folds"]:
        lines.append(
            f"  {fold['held_out']:<16} {_fmt(fold['current_deg'], 2):>10} "
            f"{_fmt(fold['candidate_deg'], 2):>10} {_signed(fold['diff']):>10}"
        )
    lines.append(
        f"Candidate better in {report['candidate_better_folds']} of {len(report['folds'])} folds"
    )
    lines.append("")
    lines.append(f"Verdict: {report['verdict']}")
    for reason in report["reasons"]:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compare two cross-validation results")
    parser.add_argument("--current", required=True, help="CV result of the settings in use")
    parser.add_argument("--candidate", required=True, help="CV result of the candidate")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Overall MAE increase (normalized) still treated as not worse",
    )
    parser.add_argument("--output", help="Save the comparison as JSON")
    args = parser.parse_args(argv)

    report = compare(load_result(args.current), load_result(args.candidate), args.tolerance)
    report["files"] = {"current": str(args.current), "candidate": str(args.candidate)}
    print(format_report(report))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {out}")
    return 2 if report["verdict"] == NOT_COMPARABLE else 0


if __name__ == "__main__":
    sys.exit(main())
