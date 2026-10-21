"""
Evaluation Script for PiCar-X E2E Steering Model
MAE (Mean Absolute Error) および 予測結果の統計を算出する
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from ..training.dataset import DrivingDataset
    from ..training.model import SmallDrivingCNN
except ImportError:
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.model import SmallDrivingCNN


def evaluate(data_dir, model_path):
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {data_dir}")

    dataset = DrivingDataset(data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallDrivingCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_mae = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, steerings, _ in loader:
            images = images.to(device)
            steerings = steerings.unsqueeze(1).to(device)
            predictions = model(images)

            mae = torch.abs(predictions - steerings).sum().item()
            total_mae += mae
            total_samples += images.size(0)

    avg_mae = total_mae / max(total_samples, 1)
    # 角度（度数法）への換算（最大舵角35度の場合）
    avg_mae_deg = avg_mae * 35.0

    metrics = {
        "total_samples": total_samples,
        "steering_mae_normalized": round(avg_mae, 4),
        "steering_mae_degrees": round(avg_mae_deg, 2),
    }

    print("\n--- Evaluation Results ---")
    print(f"Total Samples : {metrics['total_samples']}")
    print(f"Steering MAE  : {metrics['steering_mae_normalized']} (Normalized -1.0 to 1.0)")
    print(f"Steering MAE  : ±{metrics['steering_mae_degrees']}° (Degrees)")
    model_dir = Path(model_path).parent
    eval_file = model_dir / "eval_metrics.json"
    try:
        import json
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass

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
