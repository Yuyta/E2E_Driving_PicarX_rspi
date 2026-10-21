"""
PyTorch Dataset and DataLoader for PiCar-X Driving Data
単一セッション、親フォルダ (例: data/raw/) 配下の全セッション、または複数セッションの統合に対応。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_session_dirs(
    data_dir: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Path]:
    """
    指定されたパスから labels.jsonl を含む有効なセッションディレクトリ一覧を解決する。

    Args:
        data_dir: 単一セッションフォルダ、親フォルダ (例: data/raw)、またはパスのリスト/タプル

    Returns:
        labels.jsonl を含む Path のリスト
    """
    targets: List[Path] = []

    if isinstance(data_dir, (list, tuple)):
        for item in data_dir:
            targets.extend(resolve_session_dirs(item))
        return sorted(list(dict.fromkeys(targets)))

    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # 直下に labels.jsonl がある場合は単一セッション
    if (p / "labels.jsonl").exists():
        return [p]

    # 直下にない場合、サブディレクトリから labels.jsonl を探索
    if p.is_dir():
        found = [d for d in sorted(p.iterdir()) if d.is_dir() and (d / "labels.jsonl").exists()]
        if found:
            return found
        # 再帰的に探索 (data/processed/clean_v1 など深い階層にも対応)
        recursive_found = sorted(
            list(dict.fromkeys(lp.parent for lp in p.rglob("labels.jsonl")))
        )
        if recursive_found:
            return recursive_found

    raise FileNotFoundError(f"No valid session directory with labels.jsonl found in: {p}")


class DrivingDataset(Dataset):
    """
    収集された画像とステアリング角のデータセット (複数セッション統合対応)
    """

    def __init__(self, data_dir: Union[str, Path, Sequence[Union[str, Path]]], transform=None):
        """
        Args:
            data_dir: 単一セッションフォルダ、親フォルダ (data/raw)、またはそのリスト
            transform: torchvision 変換パイプライン
        """
        self.session_dirs = resolve_session_dirs(data_dir)
        self.samples: List[Dict[str, Any]] = []

        for s_dir in self.session_dirs:
            labels_file = s_dir / "labels.jsonl"
            images_dir = s_dir / "images"

            with open(labels_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    record = json.loads(line_str)
                    frame_name = record.get("frame", "")

                    # 画像ファイルパスの解決
                    img_path = images_dir / frame_name
                    if not img_path.exists() and (s_dir / frame_name).exists():
                        img_path = s_dir / frame_name

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "steering": float(record.get("steering", 0.0)),
                            "throttle": float(record.get("throttle", 0.0)),
                            "session_id": s_dir.name,
                            "session_dir": s_dir,
                            "raw_record": record,
                        }
                    )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((120, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        steering = torch.tensor(sample["steering"], dtype=torch.float32)
        throttle = torch.tensor(sample["throttle"], dtype=torch.float32)

        return image, steering, throttle

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """サンプルの画像パスやメタデータを取得"""
        return self.samples[idx]
