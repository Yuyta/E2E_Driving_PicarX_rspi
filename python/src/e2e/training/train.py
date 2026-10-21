"""
Training and ONNX Export Script for PiCar-X
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import DrivingDataset
    from .model import SmallDrivingCNN
except ImportError:
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.model import SmallDrivingCNN


def train(data_dir, output_model_dir="models/v001", epochs=15, batch_size=32, lr=1e-3):
    output_dir = Path(output_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_dir}")
    full_dataset = DrivingDataset(data_dir)
    total_len = len(full_dataset)
    session_names = [s.name for s in full_dataset.session_dirs]
    print(f"Loaded {len(session_names)} session(s): {', '.join(session_names)}")
    print(f"Total samples: {total_len}")

    if total_len == 0:
        print("Error: Dataset is empty.")
        return

    # Train / Val 分割 (80% / 20%)
    val_len = int(total_len * 0.2)
    train_len = total_len - val_len
    train_set, val_set = random_split(full_dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SmallDrivingCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, steerings, _ in train_loader:
            images = images.to(device)
            steerings = steerings.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, steerings)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= train_len

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, steerings, _ in val_loader:
                images = images.to(device)
                steerings = steerings.unsqueeze(1).to(device)
                outputs = model(images)
                loss = criterion(outputs, steerings)
                val_loss += loss.item() * images.size(0)

        val_loss /= max(val_len, 1)

        log_msg = f"Epoch [{epoch:02d}/{epochs:02d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}"
        print(log_msg)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pth")

    # ONNXエクスポート
    print("\nExporting model to ONNX...")
    model.load_state_dict(torch.load(output_dir / "best_model.pth"))
    model.eval()

    dummy_input = torch.randn(1, 3, 120, 160, device=device)
    onnx_path = output_dir / "model.onnx"
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
    print(f"✓ ONNX model saved at: {onnx_path}")

    # メタデータ保存
    metadata = {
        "model_version": output_dir.name,
        "epochs": epochs,
        "batch_size": batch_size,
        "best_val_loss": best_val_loss,
        "input_shape": [1, 3, 120, 160],
        "output": "steering (-1.0 to 1.0)",
        "sessions": session_names,
        "total_samples": total_len,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Training finished. Metadata saved at: {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PiCar-X E2E Steering Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to dataset (single session, parent folder like data/raw, or multiple paths)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/v001", help="Output directory for trained model"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    data_input = args.data_dir[0] if len(args.data_dir) == 1 else args.data_dir
    train(data_input, args.output_dir, args.epochs, args.batch_size, args.lr)
