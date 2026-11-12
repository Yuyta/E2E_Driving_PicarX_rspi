"""
Evaluation Script for PiCar-X E2E Steering Model
舵角の MAE（全体と左、直進、右の別）、予測の舵角の変化量、推論時間を算出する。

指定したデータ全体で評価するので、学習に使ったデータを含めると良く見える。
モデルの比較には交差検証（`src.e2e.training.cross_validate`）を使う。
"""

import argparse
import json
from pathlib import Path

import torch

try:
    from ..training.dataset import DrivingDataset
    from ..training.model import SmallDrivingCNN
    from .metrics import inference_time_ms, predict_all, steering_metrics
except ImportError:
    from src.e2e.evaluation.metrics import inference_time_ms, predict_all, steering_metrics
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.model import SmallDrivingCNN


def evaluate(data_dir, model_path):
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {data_dir}")

    dataset = DrivingDataset(data_dir)
    model = SmallDrivingCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    indices = list(range(len(dataset)))
    preds = predict_all(model, dataset, indices)
    detail = steering_metrics(
        preds,
        [dataset.samples[i]["steering"] for i in indices],
        [dataset.samples[i]["session_id"] for i in indices],
        [dataset.samples[i]["intent"] for i in indices],
    )

    metrics = {
        "total_samples": detail["samples"],
        "steering_mae_normalized": detail["steering_mae"],
        "steering_mae_degrees": detail["steering_mae_deg"],
        "steering_mae_by_direction": detail["steering_mae_by_direction"],
        "samples_by_direction": detail["samples_by_direction"],
        "steering_change": detail["steering_change"],
        "steering_mae_by_intent": detail["steering_mae_by_intent"],
        "samples_by_intent": detail["samples_by_intent"],
        "inference_ms_pc": inference_time_ms(model),
        "data_dir": str(data_dir),
    }

    print("\n--- Evaluation Results ---")
    print(f"Total Samples  : {metrics['total_samples']}")
    print(f"Steering MAE   : {metrics['steering_mae_normalized']} (Normalized -1.0 to 1.0)")
    print(f"Steering MAE   : ±{metrics['steering_mae_degrees']}° (Degrees)")
    print(f"By direction   : {metrics['steering_mae_by_direction']}")
    print(f"Steering change: {metrics['steering_change']}")
    print(f"By intent      : {metrics['steering_mae_by_intent']}")
    print(f"Inference (PC) : {metrics['inference_ms_pc']} ms")

    eval_file = Path(model_path).parent / "eval_metrics.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PiCar-X Steering Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset (single session, parent folder like data/raw, or multiple paths)",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to .pth model file")

    args = parser.parse_args()
    data_input = args.data_dir[0] if len(args.data_dir) == 1 else args.data_dir
    evaluate(data_input, args.model_path)
