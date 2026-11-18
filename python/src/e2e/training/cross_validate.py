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
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from ..evaluation.metrics import (
        blocked_metrics,
        predict_all,
        predict_blocked,
        steering_metrics,
    )
    from . import provenance
    from .train import (
        TrainingError,
        annotation_record,
        fit,
        learns_blocked,
        load_dataset,
        load_params,
        tolerate_console_encoding,
    )
except ImportError:
    from src.e2e.evaluation.metrics import (
        blocked_metrics,
        predict_all,
        predict_blocked,
        steering_metrics,
    )
    from src.e2e.training import provenance
    from src.e2e.training.train import (
        TrainingError,
        annotation_record,
        fit,
        learns_blocked,
        load_dataset,
        load_params,
        tolerate_console_encoding,
    )


def intent_means(dataset, train_idx, intents: List[str]) -> List[float]:
    """学習に使ったフレームの、意図ごとの舵角の平均（画像を見ない答え）。意図がなければ 0"""
    by_intent: Dict[str, List[float]] = {}
    for i in dataset.steer_indices(train_idx):
        sample = dataset.samples[i]
        by_intent.setdefault(sample["intent"], []).append(sample["steering"])
    means = {name: float(np.mean(values)) for name, values in by_intent.items()}
    return [means.get(name, 0.0) for name in intents]


BLOCKED_THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def blocked_detail(
    probs: Sequence[float], labels: Sequence[bool], by_stuck: Sequence[bool]
) -> Dict[str, Any]:
    """塞がっているかの内訳: しきい値ごとの再現率と適合率、ラベルの出どころ別

    `ultrasonic` は引っかかったことから付けたラベルを除いた指標で、この規則を入れる前の交差検証と
    比べやすい。`stuck_recall` は、超音波で見えずに引っかかった場面を画像から当てられた割合。
    """
    sweep = []
    for threshold in BLOCKED_THRESHOLDS:
        m = blocked_metrics(probs, labels, threshold=threshold)
        sweep.append(
            {
                "threshold": threshold,
                "recall": m["recall"],
                "precision": m["precision"],
                "accuracy": m["accuracy"],
            }
        )
    keep = [i for i, stuck in enumerate(by_stuck) if not stuck]
    stuck_idx = [i for i, stuck in enumerate(by_stuck) if stuck]
    ultrasonic = blocked_metrics([probs[i] for i in keep], [labels[i] for i in keep])
    stuck_probs = [probs[i] for i in stuck_idx]
    return {
        "thresholds": sweep,
        "ultrasonic": ultrasonic,
        "stuck_samples": len(stuck_idx),
        "stuck_recall": (
            round(sum(1 for p in stuck_probs if p >= 0.5) / len(stuck_probs), 4)
            if stuck_probs
            else None
        ),
    }


def cross_validate(data_dir, params: Optional[Dict[str, Any]] = None, log=print) -> Dict[str, Any]:
    params = load_params(overrides=params)
    dataset = load_dataset(data_dir, params)
    with_blocked = learns_blocked(params)
    sessions = sorted({s["session_id"] for s in dataset.samples})
    if len(sessions) < 2:
        raise TrainingError(f"Cross-validation needs at least 2 sessions, found {len(sessions)}")

    folds: List[Dict[str, Any]] = []
    all_preds: List[float] = []
    all_targets: List[float] = []
    all_sessions: List[str] = []
    all_intents: List[str] = []
    all_blocked_probs: List[float] = []
    all_blocked_labels: List[bool] = []
    all_blocked_by_stuck: List[bool] = []
    # 画像を見ない答え方。モデルがこれより良ければ、画像を使えている
    all_intent_mean: List[float] = []

    for held_out in sessions:
        train_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] != held_out]
        fold_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] == held_out]
        # 舵角の評価は、舵角の学習に使えるフレームだけで行う
        test_idx = dataset.steer_indices(fold_idx)
        log(f"Fold: hold out {held_out} ({len(test_idx)} samples, train {len(train_idx)})")
        model, _, _ = fit(dataset, train_idx, None, params, log=lambda _msg: None)
        if not test_idx:
            raise TrainingError(f"Held-out session {held_out} has no frames usable for steering")

        preds = predict_all(model, dataset, test_idx)
        targets = [dataset.samples[i]["steering"] for i in test_idx]
        session_ids = [held_out] * len(test_idx)
        intents = [dataset.samples[i]["intent"] for i in test_idx]
        metrics = steering_metrics(preds, targets, session_ids, intents)
        if with_blocked:
            blocked_idx = dataset.blocked_indices(fold_idx)
            probs = predict_blocked(model, dataset, blocked_idx)
            labels = [dataset.samples[i]["blocked"] for i in blocked_idx]
            metrics["blocked"] = blocked_metrics(probs, labels)
            all_blocked_probs.extend(probs)
            all_blocked_labels.extend(labels)
            all_blocked_by_stuck.extend(dataset.samples[i]["blocked_by_stuck"] for i in blocked_idx)
        folds.append({"held_out": held_out, **metrics})

        all_preds.extend(preds)
        all_targets.extend(targets)
        all_sessions.extend(session_ids)
        all_intents.extend(intents)
        all_intent_mean.extend(intent_means(dataset, train_idx, intents))

    overall = steering_metrics(all_preds, all_targets, all_sessions, all_intents)
    baselines = {
        # 常に0°（まっすぐ）
        "zero": steering_metrics([0.0] * len(all_targets), all_targets, all_sessions, all_intents),
        # 学習に使ったセッションの、意図ごとの舵角の平均
        "intent_mean": steering_metrics(all_intent_mean, all_targets, all_sessions, all_intents),
    }
    if with_blocked:
        overall["blocked"] = blocked_metrics(all_blocked_probs, all_blocked_labels)
        overall["blocked_detail"] = blocked_detail(
            all_blocked_probs, all_blocked_labels, all_blocked_by_stuck
        )
    return {
        "method": "leave_one_session_out",
        "params": params,
        # すべての抜いたセッションの予測をまとめた指標。採用の判定はこれで比べる
        "overall": overall,
        "baselines": baselines,
        "folds": folds,
        "data_fingerprint": provenance.data_fingerprint(dataset.samples),
        "annotations": annotation_record(dataset),
        "code": provenance.git_state(),
    }


def main(argv: Optional[List[str]] = None) -> int:
    tolerate_console_encoding()
    parser = argparse.ArgumentParser(description="Leave-one-session-out cross-validation")
    parser.add_argument("--data_dir", type=str, nargs="+", required=True)
    parser.add_argument("--config", type=str, default=None, help="TOML file with parameters")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--blocked_loss_weight",
        type=float,
        default=None,
        help="Weight of the 'way ahead is blocked' loss (0 = do not learn it)",
    )
    parser.add_argument(
        "--turn_intent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Relabel turns with the direction they turned as the intent",
    )
    parser.add_argument(
        "--blocked_stuck",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Label frames where the car was stuck as blocked",
    )
    parser.add_argument(
        "--flip_augment",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Add left-right mirrored copies of the training frames",
    )
    parser.add_argument("--output", type=str, default=None, help="Write the result as JSON")
    args = parser.parse_args(argv)

    params = load_params(
        args.config,
        {
            "epochs": args.epochs,
            "seed": args.seed,
            "blocked_loss_weight": args.blocked_loss_weight,
            "turn_intent": args.turn_intent,
            "flip_augment": args.flip_augment,
            "blocked_stuck": args.blocked_stuck,
        },
    )
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
    print(f"By intent: {overall['steering_mae_by_intent']}")
    print(f"Samples by intent: {overall['samples_by_intent']}")
    for name, baseline in result["baselines"].items():
        print(
            f"Baseline {name}: {baseline['steering_mae']} (±{baseline['steering_mae_deg']}°), "
            f"by intent {baseline['steering_mae_by_intent']}"
        )
    if "blocked" in overall:
        print(f"Blocked: {overall['blocked']}")
        detail = overall["blocked_detail"]
        print(f"Blocked (ultrasonic labels only): {detail['ultrasonic']}")
        print(f"Stuck frames detected: {detail['stuck_recall']} of {detail['stuck_samples']}")
        for row in detail["thresholds"]:
            print(
                f"  threshold {row['threshold']}: "
                f"recall {row['recall']}, precision {row['precision']}"
            )
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
