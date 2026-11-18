"""
走行モデルのオフライン指標

採用の判定（ROADMAP の M3）で使う。学習に使っていないセッションで計算すること。
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    from ..intent import INTENTS
except ImportError:
    from src.e2e.intent import INTENTS

# 舵角の区分。registry/dataset.py と cleaner.py の分布と同じ境界
STRAIGHT_LIMIT = 0.15
MAX_STEERING_DEG = 35.0


def direction_of(steering: float) -> str:
    if steering < -STRAIGHT_LIMIT:
        return "left"
    if steering > STRAIGHT_LIMIT:
        return "right"
    return "straight"


def steering_metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    session_ids: Sequence[str],
    intents: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """舵角の MAE（全体と左、直進、右の別。意図を渡せば意図の別も）と、予測の舵角の変化量

    変化量はセッション内で隣り合うフレームの予測の差の絶対値の平均。大きいほどふらつく。
    サンプルはセッション内で時刻順に並んでいる前提。
    """
    preds = np.asarray(predictions, dtype=np.float64)
    tgts = np.asarray(targets, dtype=np.float64)
    if len(preds) != len(tgts) or len(preds) != len(session_ids):
        raise ValueError("predictions, targets and session_ids must have the same length")
    if len(preds) == 0:
        raise ValueError("No samples to evaluate")

    errors = np.abs(preds - tgts)
    by_direction: Dict[str, Optional[float]] = {}
    counts: Dict[str, int] = {}
    directions = np.array([direction_of(t) for t in tgts])
    for name in ("left", "straight", "right"):
        mask = directions == name
        counts[name] = int(mask.sum())
        by_direction[name] = round(float(errors[mask].mean()), 4) if mask.any() else None

    changes: List[float] = []
    for i in range(1, len(preds)):
        if session_ids[i] == session_ids[i - 1]:
            changes.append(abs(preds[i] - preds[i - 1]))

    mae = float(errors.mean())
    extra: Dict[str, object] = {}
    if intents is not None:
        if len(intents) != len(preds):
            raise ValueError("intents must have the same length as predictions")
        labels = np.array(list(intents))
        extra["steering_mae_by_intent"] = {
            name: round(float(errors[labels == name].mean()), 4) if (labels == name).any() else None
            for name in INTENTS
        }
        extra["samples_by_intent"] = {name: int((labels == name).sum()) for name in INTENTS}
    return {
        "samples": int(len(preds)),
        "steering_mae": round(mae, 4),
        "steering_mae_deg": round(mae * MAX_STEERING_DEG, 2),
        "steering_mae_by_direction": by_direction,
        "samples_by_direction": counts,
        "steering_change": round(float(np.mean(changes)), 4) if changes else None,
        **extra,
    }


def predict_all(model: torch.nn.Module, dataset, indices: Sequence[int]) -> List[float]:
    """指定したサンプルの舵角を順に推論する"""
    model.eval()
    preds: List[float] = []
    with torch.no_grad():
        for idx in indices:
            image, _, _, intent = dataset[idx]
            preds.append(float(model(image.unsqueeze(0), intent.unsqueeze(0)).item()))
    return preds


def blocked_metrics(
    probabilities: Sequence[float], labels: Sequence[bool], threshold: float = 0.5
) -> Dict[str, object]:
    """前方が塞がっているかの判定の指標

    見落とし（塞がっているのに空いていると答える）は衝突につながるので、再現率（recall）を
    主に見る。`always_clear_accuracy` は常に「空いている」と答えた場合の正解率で、比べる基準。
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=bool)
    if len(probs) != len(truth):
        raise ValueError("probabilities and labels must have the same length")
    if len(probs) == 0:
        return {"samples": 0}
    predicted = probs >= threshold
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    eps = 1e-7
    clipped = np.clip(probs, eps, 1 - eps)
    log_loss = -np.mean(truth * np.log(clipped) + (~truth) * np.log(1 - clipped))

    def rounded(value):
        return None if value is None else round(float(value), 4)

    return {
        "samples": int(len(probs)),
        "blocked_samples": int(truth.sum()),
        "threshold": threshold,
        "accuracy": rounded((tp + tn) / len(probs)),
        "always_clear_accuracy": rounded((~truth).mean()),
        "precision": rounded(precision),
        "recall": rounded(recall),
        "f1": rounded(f1),
        "log_loss": rounded(log_loss),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def predict_blocked(model: torch.nn.Module, dataset, indices: Sequence[int]) -> List[float]:
    """指定したサンプルの、前方が塞がっている確率を順に推論する"""
    model.eval()
    probs: List[float] = []
    with torch.no_grad():
        for idx in indices:
            image, _, _, intent = dataset[idx]
            _, logit = model.forward_with_blocked(image.unsqueeze(0), intent.unsqueeze(0))
            probs.append(float(torch.sigmoid(logit).item()))
    return probs


def inference_time_ms(model: torch.nn.Module, runs: int = 50, warmup: int = 5) -> float:
    """1枚あたりの推論時間（PyTorch、CPU、中央値）。PC 上の比較用で、Pi の値ではない"""
    model.eval()
    dummy = torch.zeros(1, 3, 120, 160)
    dummy_intent = torch.zeros(1, len(INTENTS))
    dummy_intent[0, 0] = 1.0
    timings = []
    with torch.no_grad():
        for i in range(warmup + runs):
            start = time.perf_counter()
            model(dummy, dummy_intent)
            if i >= warmup:
                timings.append((time.perf_counter() - start) * 1000.0)
    return round(float(np.median(timings)), 3)
