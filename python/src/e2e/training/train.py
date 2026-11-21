"""
Training and ONNX Export Script for PiCar-X

学習は CPU で決定的に行い、同じデータと設定なら同じ指標になるようにする（ROADMAP の M3）。
分割はセッション単位で、検証用のセッションを学習に混ぜない。

使い方:
    uv run python -m src.e2e.training.train --data_dir data/raw --output_dir models/v006
    uv run python -m src.e2e.training.train --config configs/train.toml --data_dir data/raw ...
"""

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 以前
    tomllib = None

try:
    from ..evaluation.metrics import (
        blocked_metrics,
        inference_time_ms,
        predict_all,
        predict_blocked,
        steering_metrics,
    )
    from ..intent import INTENT_LEFT, INTENT_RIGHT, INTENTS
    from . import provenance
    from .dataset import BLOCKED_CLEAR_CM, BLOCKED_NEAR_CM, DrivingDataset
    from .model import ExportModel, SmallDrivingCNN
    from .onnx_utils import strip_export_debug_info
except ImportError:
    from src.e2e.evaluation.metrics import (
        blocked_metrics,
        inference_time_ms,
        predict_all,
        predict_blocked,
        steering_metrics,
    )
    from src.e2e.intent import INTENT_LEFT, INTENT_RIGHT, INTENTS
    from src.e2e.training import provenance
    from src.e2e.training.dataset import BLOCKED_CLEAR_CM, BLOCKED_NEAR_CM, DrivingDataset
    from src.e2e.training.model import ExportModel, SmallDrivingCNN
    from src.e2e.training.onnx_utils import strip_export_debug_info

INPUT_HEIGHT = 120
INPUT_WIDTH = 160
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

DEFAULT_PARAMS: Dict[str, Any] = {
    "epochs": 15,
    "batch_size": 32,
    "lr": 1e-3,
    "seed": 42,
    # 検証に使うセッション。空なら名前順で最後のセッション
    "val_sessions": [],
    # 前方が塞がっているかの損失の重み。0 なら学習せず、舵角の学習はこの頭がない時と同じになる
    "blocked_loss_weight": 0.0,
    # 塞がっているかのラベルの境界（cm）。間の距離のフレームはこの学習に使わない
    "blocked_near_cm": BLOCKED_NEAR_CM,
    "blocked_clear_cm": BLOCKED_CLEAR_CM,
    # 前へ進めずに引っかかったフレームを塞がっているとし、前後の「空いている」を不明にする
    "blocked_stuck": True,
    # 曲がりの区間に、曲がった向きを意図として付け直す（dataset.turn_intents）
    "turn_intent": False,
    # 学習用のフレームに、左右反転した複製を加える（舵角の符号と意図の左右を入れ替える）。
    # 評価は元のフレームだけで行う
    "flip_augment": False,
    # 手作業のラベル（data/annotations）を使う。除外の印、意図と塞がっているかの上書き
    "annotations": True,
}


class TrainingError(RuntimeError):
    """学習を始められない、または結果が信用できない時"""


def load_params(
    config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """既定値、設定ファイル、コマンドラインの順に上書きする"""
    params = dict(DEFAULT_PARAMS)
    if config_path:
        if tomllib is None:
            raise TrainingError("Reading a TOML config needs Python 3.11 or later")
        with open(config_path, "rb") as f:
            loaded = tomllib.load(f)
        unknown = set(loaded) - set(DEFAULT_PARAMS)
        if unknown:
            raise TrainingError(f"Unknown keys in {config_path}: {sorted(unknown)}")
        params.update(loaded)
    for key, value in (overrides or {}).items():
        if value is not None:
            params[key] = value
    params["val_sessions"] = list(params["val_sessions"])
    return params


def annotation_record(dataset: DrivingDataset) -> Dict[str, Any]:
    """学習の記録に残す、手作業のラベルの版と使った数"""
    return {
        "hashes": dict(dataset.annotation_hashes),
        "excluded": dataset.annotated_excluded,
        "intent_overridden": dataset.annotated_intent,
        "blocked_overridden": dataset.annotated_blocked,
    }


def learns_blocked(params: Dict[str, Any]) -> bool:
    return float(params.get("blocked_loss_weight", 0.0)) > 0.0


def load_dataset(data_dir, params: Dict[str, Any]) -> DrivingDataset:
    """設定に合わせてデータを読む。塞がっているかを学習する時は、舵角に使わないフレームも残す"""
    return DrivingDataset(
        data_dir,
        keep_for_blocked=learns_blocked(params),
        blocked_near_cm=float(params["blocked_near_cm"]),
        blocked_clear_cm=float(params["blocked_clear_cm"]),
        turn_intent=bool(params["turn_intent"]),
        blocked_stuck=bool(params["blocked_stuck"]),
        annotations=bool(params.get("annotations", True)),
    )


class _Targets(Dataset):
    """学習用に、舵角と塞がっているかのラベルと、それぞれを使うかの印を付ける"""

    def __init__(self, dataset: DrivingDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, steering, _, intent = self.dataset[idx]
        sample = self.dataset.samples[idx]
        blocked = sample["blocked"]
        return (
            image,
            steering,
            intent,
            torch.tensor(1.0 if sample["steer_valid"] else 0.0),
            torch.tensor(1.0 if blocked else 0.0),
            torch.tensor(0.0 if blocked is None else 1.0),
        )


_MIRRORED_NAME = {INTENT_LEFT: INTENT_RIGHT, INTENT_RIGHT: INTENT_LEFT}
# 反転後の意図の各位置へ、元のどの位置の値を持ってくるか（any はそのまま、left と right を交換）
_MIRROR_INTENT_ORDER = [INTENTS.index(_MIRRORED_NAME.get(name, name)) for name in INTENTS]


class _Mirrored(Dataset):
    """番号が n 未満なら元のまま、n 以上なら n を引いたフレームを左右反転して返す（n は元の数）

    画像を左右反転し、舵角の符号を反転し、意図の左右を入れ替える。前方が塞がっているかと、
    使うかの印は左右に関係しないのでそのまま。右へ曲がるフレームが左の半分ほどしかない偏りを補う。
    """

    def __init__(self, base: Dataset, intent_pos: int):
        self.base = base
        self.intent_pos = intent_pos

    def __len__(self) -> int:
        return 2 * len(self.base)

    def __getitem__(self, idx: int):
        n = len(self.base)
        item = self.base[idx % n]
        if idx < n:
            return item
        item = list(item)
        item[0] = torch.flip(item[0], dims=[-1])  # 画像の横方向（C, H, W の W）
        item[1] = -item[1]
        item[self.intent_pos] = item[self.intent_pos][_MIRROR_INTENT_ORDER]
        return tuple(item)


def _training_subset(base: Dataset, train_idx: Sequence[int], params: Dict[str, Any], intent_pos):
    """学習に使う集合。flip_augment なら、各フレームに左右反転した複製を加える"""
    if not params.get("flip_augment"):
        return Subset(base, list(train_idx))
    n = len(base)
    return Subset(_Mirrored(base, intent_pos), list(train_idx) + [n + i for i in train_idx])


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1.0)


def set_seed(seed: int) -> torch.Generator:
    """乱数を固定する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


@contextmanager
def deterministic():
    """学習の間だけ決定的なアルゴリズムに限る。終わったら元の設定に戻す"""
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def split_by_session(
    dataset: DrivingDataset, val_sessions: Sequence[str]
) -> Tuple[List[int], List[int], List[str]]:
    """セッション単位で学習と検証に分ける。どちらかが空なら学習しない"""
    all_sessions = sorted({s["session_id"] for s in dataset.samples})
    if not val_sessions:
        if len(all_sessions) < 2:
            raise TrainingError(
                "Session-based validation needs at least 2 sessions, "
                f"found {len(all_sessions)}: {all_sessions}"
            )
        val_sessions = [all_sessions[-1]]
    unknown = sorted(set(val_sessions) - set(all_sessions))
    if unknown:
        raise TrainingError(f"Validation sessions not in the dataset: {unknown}")

    val_set = set(val_sessions)
    train_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] not in val_set]
    val_idx = [i for i, s in enumerate(dataset.samples) if s["session_id"] in val_set]
    if not train_idx or not val_idx:
        raise TrainingError("Training and validation sets must both be non-empty")
    return train_idx, val_idx, sorted(val_set)


def fit(
    dataset: DrivingDataset,
    train_idx: Sequence[int],
    val_idx: Optional[Sequence[int]],
    params: Dict[str, Any],
    log=print,
) -> Tuple[SmallDrivingCNN, List[Dict[str, float]], Optional[float]]:
    """学習する。検証データがあれば検証損失が最良の重みを、なければ最後の重みを返す"""
    with deterministic():
        return _fit(dataset, train_idx, val_idx, params, log)


def _fit(dataset, train_idx, val_idx, params, log):
    generator = set_seed(int(params["seed"]))
    model = SmallDrivingCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(params["lr"]))
    if learns_blocked(params):
        return _fit_with_blocked(
            dataset, train_idx, val_idx, params, log, model, optimizer, generator
        )

    train_set = _training_subset(dataset, train_idx, params, intent_pos=3)
    train_loader = DataLoader(
        train_set,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = (
        DataLoader(Subset(dataset, list(val_idx)), batch_size=int(params["batch_size"]))
        if val_idx
        else None
    )

    epochs = int(params["epochs"])
    best_val_loss: Optional[float] = None
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, steerings, _, intents in train_loader:
            steerings = steerings.unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images, intents), steerings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_set)

        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, steerings, _, intents in val_loader:
                    loss = criterion(model(images, intents), steerings.unsqueeze(1))
                    val_loss += loss.item() * images.size(0)
            val_loss /= len(val_idx)
            entry["val_loss"] = val_loss
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            log(f"Epoch [{epoch:02d}/{epochs:02d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        else:
            log(f"Epoch [{epoch:02d}/{epochs:02d}] Train: {train_loss:.4f}")
        history.append(entry)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_val_loss


def _fit_with_blocked(dataset, train_idx, val_idx, params, log, model, optimizer, generator):
    """舵角（使えるフレームだけ）と、前方が塞がっているか（ラベルのあるフレームだけ）を一緒に学習する"""
    weight = float(params["blocked_loss_weight"])
    bce = nn.BCEWithLogitsLoss(reduction="none")
    targets = _Targets(dataset)
    batch_size = int(params["batch_size"])
    train_loader = DataLoader(
        _training_subset(targets, train_idx, params, intent_pos=2),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = (
        DataLoader(Subset(targets, list(val_idx)), batch_size=batch_size) if val_idx else None
    )

    def losses(batch):
        images, steerings, intents, steer_mask, blocked, blocked_mask = batch
        steering_pred, blocked_logit = model.forward_with_blocked(images, intents)
        steer_loss = _masked_mean((steering_pred.squeeze(1) - steerings) ** 2, steer_mask)
        blocked_loss = _masked_mean(bce(blocked_logit.squeeze(1), blocked), blocked_mask)
        return steer_loss, blocked_loss

    def run(loader, train: bool) -> Tuple[float, float]:
        steer_sum, blocked_sum, batches = 0.0, 0.0, 0
        for batch in loader:
            if train:
                optimizer.zero_grad()
                steer_loss, blocked_loss = losses(batch)
                (steer_loss + weight * blocked_loss).backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    steer_loss, blocked_loss = losses(batch)
            steer_sum += steer_loss.item()
            blocked_sum += blocked_loss.item()
            batches += 1
        batches = max(batches, 1)
        return steer_sum / batches, blocked_sum / batches

    epochs = int(params["epochs"])
    best_val_loss: Optional[float] = None
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        steer_loss, blocked_loss = run(train_loader, train=True)
        entry = {
            "epoch": epoch,
            "train_loss": steer_loss + weight * blocked_loss,
            "train_steering_loss": steer_loss,
            "train_blocked_loss": blocked_loss,
        }
        message = f"Epoch [{epoch:02d}/{epochs:02d}] Train: {entry['train_loss']:.4f}"
        if val_loader is not None:
            model.eval()
            steer_loss, blocked_loss = run(val_loader, train=False)
            val_loss = steer_loss + weight * blocked_loss
            entry.update(
                {
                    "val_loss": val_loss,
                    "val_steering_loss": steer_loss,
                    "val_blocked_loss": blocked_loss,
                }
            )
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            message += f" | Val: {val_loss:.4f}"
        log(message)
        history.append(entry)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history, best_val_loss


def export_onnx(model: SmallDrivingCNN, onnx_path: Path) -> None:
    dummy_input = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    dummy_intent = torch.zeros(1, len(INTENTS))
    dummy_intent[0, 0] = 1.0
    # 1つ目の出力は舵角のまま（自律走行は outputs[0] を読む）。2つ目は塞がっている確率
    torch.onnx.export(
        ExportModel(model).eval(),
        (dummy_input, dummy_intent),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input", "intent"],
        output_names=["steering", "blocked"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "intent": {0: "batch_size"},
            "steering": {0: "batch_size"},
            "blocked": {0: "batch_size"},
        },
    )
    # 新しい exporter（既定）は進捗に絵文字を出す。cp932 の端末では main() で置き換えて表示する。
    # 旧 exporter（dynamo=False）は AdaptiveAvgPool2d の 7x10 -> 2x2 を出力できないので使わない
    strip_export_debug_info(onnx_path)


def train(data_dir, output_model_dir, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = load_params(overrides=params) if params is not None else load_params()
    output_dir = Path(output_model_dir)

    print(f"Loading dataset from: {data_dir}")
    dataset = load_dataset(data_dir, params)
    if not dataset.steer_indices():
        raise TrainingError("Dataset is empty")

    train_idx, val_idx, val_sessions = split_by_session(dataset, params["val_sessions"])
    train_sessions = sorted({dataset.samples[i]["session_id"] for i in train_idx})
    print(f"Train sessions: {', '.join(train_sessions)} ({len(train_idx)} samples)")
    print(f"Validation sessions: {', '.join(val_sessions)} ({len(val_idx)} samples)")

    model, history, best_val_loss = fit(dataset, train_idx, val_idx, params)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pth")
    print("\nExporting model to ONNX...")
    export_onnx(model, output_dir / "model.onnx")

    val_steer = dataset.steer_indices(val_idx)
    if not val_steer:
        raise TrainingError("Validation sessions have no frames usable for steering")
    preds = predict_all(model, dataset, val_steer)
    validation = steering_metrics(
        preds,
        [dataset.samples[i]["steering"] for i in val_steer],
        [dataset.samples[i]["session_id"] for i in val_steer],
        [dataset.samples[i]["intent"] for i in val_steer],
    )
    if learns_blocked(params):
        val_blocked = dataset.blocked_indices(val_idx)
        validation["blocked"] = blocked_metrics(
            predict_blocked(model, dataset, val_blocked),
            [dataset.samples[i]["blocked"] for i in val_blocked],
        )

    metadata = {
        "model_version": output_dir.name,
        # 学習を終えた時刻（モデルの一覧で最新版を決めるのに使う）
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "best_val_loss": best_val_loss,
        "input_shape": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
        # 2つ目の入力。any, left, right の順の 0 / 1（ONNX の入力名は intent）
        "intent_input": {"shape": [1, len(INTENTS)], "order": list(INTENTS)},
        "output": "steering (-1.0 to 1.0)",
        # ONNX の2つ目の出力。trained が False なら学習していないので使わない
        "blocked_output": {
            "name": "blocked",
            "meaning": "probability that the way ahead is blocked (0.0 to 1.0)",
            "trained": learns_blocked(params),
            "label_near_cm": params["blocked_near_cm"],
            "label_clear_cm": params["blocked_clear_cm"],
            "label_stuck": params["blocked_stuck"],
        },
        "sessions": train_sessions + val_sessions,
        "total_samples": len(dataset),
        "params": params,
        "split": {
            "method": "session",
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
        },
        "preprocessing": {
            "resize_hw": [INPUT_HEIGHT, INPUT_WIDTH],
            "normalize_mean": NORMALIZE_MEAN,
            "normalize_std": NORMALIZE_STD,
            "color": "RGB",
        },
        # 検証セッションはモデルの選択にも使うので、偏りのない比較には交差検証を使う
        "validation_metrics": validation,
        "inference_ms_pc": inference_time_ms(model),
        "data": {
            "fingerprint": provenance.data_fingerprint(dataset.samples),
            "sessions": provenance.session_records(dataset.source_session_dirs),
            # 指令と実際の動きが食い違ったので学習から外したフレーム
            "skipped_motion_mismatch": dataset.skipped_motion_mismatch,
            # 前進の指令がなく止まっていたので学習から外したフレーム
            "skipped_stopped": dataset.skipped_stopped,
            # 舵角には使わず、塞がっているかの学習だけに使ったフレーム
            "kept_for_blocked_only": dataset.kept_for_blocked_only,
            # 曲がりの区間なので、意図を左右へ付け直したフレーム（turn_intent）
            "relabeled_turn_intent": dataset.relabeled_turn_intent,
            # 引っかかったので塞がっているとしたフレームと、その近くで不明にしたフレーム
            "stuck_blocked": dataset.stuck_blocked,
            "stuck_unknown": dataset.stuck_unknown,
            # 手作業のラベル（data/annotations）。セッションごとの中身のハッシュと、使った数
            "annotations": annotation_record(dataset),
            "samples_by_intent": {
                name: sum(1 for s in dataset.samples if s["intent"] == name and s["steer_valid"])
                for name in INTENTS
            },
            "blocked_labels": {
                "blocked": sum(1 for s in dataset.samples if s["blocked"] is True),
                "clear": sum(1 for s in dataset.samples if s["blocked"] is False),
                "unknown": sum(1 for s in dataset.samples if s["blocked"] is None),
            },
        },
        # 学習データを集めた車体の版（configs/body.toml）
        "body": provenance.body_record(dataset.source_session_dirs),
        "code": provenance.git_state(),
        "environment": provenance.environment(),
        "history": history,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Validation MAE: {validation['steering_mae']} (±{validation['steering_mae_deg']}°)")
    if "blocked" in validation:
        print(f"Validation blocked: {validation['blocked']}")
    print(f"✓ Training finished. Metadata saved at: {output_dir / 'metadata.json'}")
    return metadata


def tolerate_console_encoding() -> None:
    """Windows の cp932 の端末で表示できない文字があっても、学習を止めない"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: Optional[List[str]] = None) -> int:
    tolerate_console_encoding()
    parser = argparse.ArgumentParser(description="Train PiCar-X E2E Steering Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset (single session, parent folder like data/raw, or multiple paths)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for trained model"
    )
    parser.add_argument("--config", type=str, default=None, help="TOML file with parameters")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--val_sessions", type=str, nargs="+", default=None, help="Session IDs for validation"
    )
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

    args = parser.parse_args(argv)
    params = load_params(
        args.config,
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "val_sessions": args.val_sessions,
            "blocked_loss_weight": args.blocked_loss_weight,
            "turn_intent": args.turn_intent,
            "flip_augment": args.flip_augment,
            "blocked_stuck": args.blocked_stuck,
        },
    )
    data_input = args.data_dir[0] if len(args.data_dir) == 1 else args.data_dir
    try:
        train(data_input, args.output_dir, params)
    except TrainingError as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
