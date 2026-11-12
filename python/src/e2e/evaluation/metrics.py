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
