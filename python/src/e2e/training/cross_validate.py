"""
1セッションを抜いて評価する交差検証

セッションが少ない間は固定のテストセットを作らず、各セッションを順に抜いて学習し、
抜いたセッションで評価する（ROADMAP の M3）。モデルの選択に評価データを使わないので、
エポック数は固定し、最後の重みで評価する。

使い方:
    uv run python -m src.e2e.training.cross_validate --data_dir data/raw --output cv.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..evaluation.metrics import predict_all, steering_metrics
    from . import provenance
    from .dataset import DrivingDataset
    from .train import TrainingError, fit, load_params, tolerate_console_encoding
except ImportError:
    from src.e2e.evaluation.metrics import predict_all, steering_metrics
    from src.e2e.training import provenance
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.train import (
        TrainingError,
        fit,
        load_params,
        tolerate_console_encoding,
    )


def cross_validate(data_dir, params: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    params = load_params(overrides=params)
    dataset = DrivingDataset(data_dir)
    sessions = sorted({s["session_id"] for s in dataset.samples})
    if len(sessions) < 2:
        raise TrainingError(f"Cross-validation needs at least 2 sessions, found {len(sessions)}")

    folds: List[Dict[str, Any]] = []
    all_preds: List[float] = []
    all_targets: List[float] = []
    all_sessions: List[str] = []

    for held_out in sessions:
        train_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] != held_out]
        test_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] == held_out]
        log(f"Fold: hold out {held_out} ({len(test_idx)} samples, train {len(train_idx)})")
        model, _, _ = fit(dataset, train_idx, None, params, log=lambda _msg: None)

        preds = predict_all(model, dataset, test_idx)
        targets = [dataset.samples[i]["steering"] for i in test_idx]
        session_ids = [held_out] * len(test_idx)
        metrics = steering_metrics(preds, targets, session_ids)
        folds.append({"held_out": held_out, **metrics})

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_sessions.extend(session_ids)

    return {
        "method": "leave_one_session_out",
        "params": params,
        # すべての抜いたセッションの予測をまとめた指標。採用の判定はこれで比べる
        "overall": steering_metrics(all_preds, all_targets, all_sessions),
        "folds": folds,
        "data_fingerprint": provenance.data_fingerprint(dataset.samples),
        "code": provenance.git_state(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    tolerate_console_encoding()
    parser = argparse.ArgumentParser(description="Leave-one-session-out cross-validation")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--config", type=str, default=None, help="TOML file with parameters")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None, help="Write the result as JSON")
    args = parser.parse_args(argv)

    params = load_params(args.config, {"epochs": args.epochs, "seed": args.seed})
    data_input = args.data_dir[0] if len(args.data_dir) == 1 else args.data_dir
    try:
        result = cross_validate(data_input, params)
    except TrainingError as e:
        print(f"Error: {e}")
        return 1

    overall = result["overall"]
    print(f"\nOverall MAE: {overall['steering_mae']} (±{overall['steering_mae_deg']}°)")
    print(f"By direction: {overall['steering_mae_by_direction']}")
    print(f"Samples by direction: {overall['samples_by_direction']}")
    print(f"Steering change: {overall['steering_change']}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
