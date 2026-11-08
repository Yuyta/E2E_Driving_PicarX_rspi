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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 以前
    tomllib = None

try:
    from ..evaluation.metrics import inference_time_ms, predict_all, steering_metrics
    from . import provenance
    from .dataset import DrivingDataset
    from .model import SmallDrivingCNN
    from .onnx_utils import strip_export_debug_info
except ImportError:
    from src.e2e.evaluation.metrics import inference_time_ms, predict_all, steering_metrics
    from src.e2e.training import provenance
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.model import SmallDrivingCNN
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

    train_loader = DataLoader(
        Subset(dataset, list(train_idx)),
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
        for images, steerings, _ in train_loader:
            steerings = steerings.unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), steerings)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_idx)

        entry = {"epoch": epoch, "train_loss": train_loss}
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, steerings, _ in val_loader:
                    loss = criterion(model(images), steerings.unsqueeze(1))
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


def export_onnx(model: SmallDrivingCNN, onnx_path: Path) -> None:
    dummy_input = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["steering"],
        dynamic_axes={"input": {0: "batch_size"}, "steering": {0: "batch_size"}},
    )
    # 新しい exporter（既定）は進捗に絵文字を出す。cp932 の端末では main() で置き換えて表示する。
    # 旧 exporter（dynamo=False）は AdaptiveAvgPool2d の 7x10 -> 2x2 を出力できないので使わない
    strip_export_debug_info(onnx_path)


def train(data_dir, output_model_dir, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = load_params(overrides=params) if params is not None else load_params()
    output_dir = Path(output_model_dir)

    print(f"Loading dataset from: {data_dir}")
    dataset = DrivingDataset(data_dir)
    if len(dataset) == 0:
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

    preds = predict_all(model, dataset, val_idx)
    validation = steering_metrics(
        preds,
        [dataset.samples[i]["steering"] for i in val_idx],
        [dataset.samples[i]["session_id"] for i in val_idx],
    )

    metadata = {
        "model_version": output_dir.name,
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "best_val_loss": best_val_loss,
        "input_shape": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
        "output": "steering (-1.0 to 1.0)",
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
        },
        "code": provenance.git_state(),
        "environment": provenance.environment(),
        "history": history,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Validation MAE: {validation['steering_mae']} (±{validation['steering_mae_deg']}°)")
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

    args = parser.parse_args(argv)
    params = load_params(
        args.config,
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "val_sessions": args.val_sessions,
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
