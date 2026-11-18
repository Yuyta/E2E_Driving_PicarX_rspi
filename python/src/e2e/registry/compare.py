"""
Model and Dataset Comparator CLI for PiCar-X MLOps
登録されているデータセット一覧およびモデルバージョンの性能を表形式で比較表示するCLIツール。
"""

import argparse
from pathlib import Path

from .dataset import DatasetRegistry
from .models import ModelRegistry


def show_datasets(data_dir: str = "data/raw", title: str = "Dataset Registry Summary"):
    reg = DatasetRegistry(data_dir)
    sessions = reg.list_sessions()

    print("\n" + "=" * 80)
    print(f"             PiCar-X {title} ({data_dir})")
    print("=" * 80)

    if not sessions:
        print(f"No dataset sessions found in: {data_dir}")
        print("=" * 80 + "\n")
        return

    header = (
        f"{'Session ID':<22} | {'Frames':<7} | {'Duration':<9} | {'FPS':<5} | "
        f"{'Straight':<9} | {'Left':<6} | {'Right':<6}"
    )
    print(header)
    print("-" * 80)

    for s in sessions:
        dist = s.get("distribution", {})
        row = (
            f"{s['session_id']:<22} | "
            f"{s['total_frames']:<7} | "
            f"{s['duration_sec']:<7.1f}s | "
            f"{s['fps']:<5.1f} | "
            f"{dist.get('straight_ratio', 0.0):<9.1%} | "
            f"{dist.get('left_ratio', 0.0):<6.1%} | "
            f"{dist.get('right_ratio', 0.0):<6.1%}"
        )
        print(row)
    print("=" * 80 + "\n")


def show_models(models_dir: str = "models"):
    reg = ModelRegistry(models_dir)
    models = reg.list_models()

    print("\n" + "=" * 108)
    print("                      PiCar-X Model Registry & Benchmark")
    print("=" * 108)

    if not models:
        print(f"No trained models found in: {models_dir}")
        print("=" * 108 + "\n")
        return

    header = (
        f"{'Model Version':<14} | {'Epochs':<6} | {'Batch':<5} | {'Val Loss':<9} | "
        f"{'MAE (Norm)':<10} | {'MAE (Deg)':<10} | {'Artifacts':<9} | {'Data':<8} | "
        f"{'Commit':<8}"
    )
    print(header)
    print("-" * 108)

    for m in models:
        ev = m.get("evaluation", {})
        # オフライン評価がなければ、学習の時の検証の値（metadata.json）を出す
        val = m.get("validation_metrics", {})
        norm = ev.get("steering_mae_normalized", val.get("steering_mae"))
        deg = ev.get("steering_mae_degrees", val.get("steering_mae_deg"))
        mae_norm = f"{norm:.4f}" if norm is not None else "--"
        mae_deg = f"±{deg:.1f}°" if deg is not None else "--"

        artifacts = []
        if m.get("has_weights"):
            artifacts.append("PTH")
        if m.get("has_onnx"):
            artifacts.append("ONNX")
        art_str = "+".join(artifacts) if artifacts else "None"

        val_loss = f"{m['best_val_loss']:.4f}" if m.get("best_val_loss") is not None else "--"
        # 来歴は M3 以降に学習したモデルだけが持つ
        data_hash = (m.get("data", {}).get("fingerprint") or "--")[:8]
        code = m.get("code", {})
        commit = (code.get("commit") or "--")[:7] + ("*" if code.get("dirty") else "")

        row = (
            f"{m.get('model_version', 'unknown'):<14} | "
            f"{m.get('epochs', '--'):<6} | "
            f"{m.get('batch_size', '--'):<5} | "
            f"{val_loss:<9} | "
            f"{mae_norm:<10} | "
            f"{mae_deg:<10} | "
            f"{art_str:<9} | "
            f"{data_hash:<8} | "
            f"{commit:<8}"
        )
        print(row)
    print("Data: hash of the training samples. Commit: * means uncommitted changes.")
    print("=" * 108 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PiCar-X MLOps Registry & Comparator")
    parser.add_argument(
        "--data_dir", type=str, default="data/raw", help="Path to raw data directory"
    )
    parser.add_argument(
        "--models_dir", type=str, default="models", help="Path to models directory"
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["all", "datasets", "models"],
        default="all",
        help="Target to inspect",
    )

    args = parser.parse_args()

    if args.target in ("all", "datasets"):
        show_datasets(args.data_dir, title="Raw Datasets")
        proc_path = Path("data/processed")
        if proc_path.exists() and any(proc_path.iterdir()):
            show_datasets("data/processed", title="Processed / Cleaned Datasets")
    if args.target in ("all", "models"):
        show_models(args.models_dir)


if __name__ == "__main__":
    main()
