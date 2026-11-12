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

try:
    from ..intent import one_hot
except ImportError:
    from src.e2e.intent import one_hot


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
        recursive_found = sorted(list(dict.fromkeys(lp.parent for lp in p.rglob("labels.jsonl"))))
        if recursive_found:
            return recursive_found

    raise FileNotFoundError(f"No valid session directory with labels.jsonl found in: {p}")


def is_stopped(record: Dict[str, Any]) -> bool:
    """記録から、前進の指令がなく止まっていたと分かるフレームか。分からなければ False

    新しい記録はそのフレームで有効な指令（`command.speed`）、収集の記録は `throttle`、
    介入の記録は `speed` を持つ。どれもない古い記録は、止まっていたとは判断しない。
    """
    command = record.get("command")
    if isinstance(command, dict) and "speed" in command:
        return float(command["speed"]) <= 0.0
    if "throttle" in record:
        return float(record["throttle"]) <= 0.0
    if "speed" in record:
        return float(record["speed"]) <= 0.0
    return False


class DrivingDataset(Dataset):
    """
    収集された画像とステアリング角のデータセット (複数セッション統合対応)
    """

    def __init__(
        self,
        data_dir: Union[str, Path, Sequence[Union[str, Path]]],
        transform=None,
        include_excluded: bool = False,
        include_motion_mismatch: bool = False,
        include_stopped: bool = False,
    ):
        """
        Args:
            data_dir: 単一セッションフォルダ、親フォルダ (data/raw)、またはそのリスト
            transform: torchvision 変換パイプライン
            include_excluded: False の場合、exclude: true のデータ（未承認の介入データ等）を除外
            include_motion_mismatch: False の場合、指令と実際の動きが食い違ったフレーム
                （`motion_check.flags` が空でないもの。空転、滑りなど）を除外
            include_stopped: False の場合、前進の指令がないフレーム（`throttle` が 0）を除外。
                止まっている間の舵角は、その景色でどう曲がるかの手本にならない
        """
        self.session_dirs = resolve_session_dirs(data_dir)
        self.samples: List[Dict[str, Any]] = []
        self.skipped_motion_mismatch = 0
        self.skipped_stopped = 0

        for s_dir in self.session_dirs:
            labels_file = s_dir / "labels.jsonl"
            images_dir = s_dir / "images"

            with open(labels_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    record = json.loads(line_str)

                    # 学習除外フラグの確認 (未承認の介入データや外れ値データをスキップ)
                    if not include_excluded and (
                        record.get("exclude", False) or record.get("exclude_from_training", False)
                    ):
                        continue

                    # 指令と実際の動きが食い違ったフレームは、誤った操作の例になるので学習に使わない
                    if not include_motion_mismatch and (record.get("motion_check") or {}).get(
                        "flags"
                    ):
                        self.skipped_motion_mismatch += 1
                        continue

                    if not include_stopped and is_stopped(record):
                        self.skipped_stopped += 1
                        continue

                    frame_name = record.get("frame", "")

                    # クリーナーの出力は画像を複製せず、元のセッションを `source_dir` で参照する
                    source_dir = record.get("source_dir")
                    if source_dir:
                        origin_dir = (s_dir / source_dir).resolve()
                        img_path = origin_dir / "images" / frame_name
                        if not img_path.exists() and (origin_dir / frame_name).exists():
                            img_path = origin_dir / frame_name
                    else:
                        origin_dir = s_dir
                        img_path = images_dir / frame_name
                        if not img_path.exists() and (s_dir / frame_name).exists():
                            img_path = s_dir / frame_name

                    self.samples.append(
                        {
                            "image_path": img_path,
                            "steering": float(record.get("steering", 0.0)),
                            "throttle": float(record.get("throttle", 0.0)),
                            # 意図のない古い記録は any
                            "intent": record.get("intent") or "any",
                            # 分割と集計は元のセッション単位で行う。
                            # 画像を複製していた旧形式の出力は cleaned_meta から元を読む
                            "session_id": (
                                origin_dir.name
                                if source_dir
                                else record.get("cleaned_meta", {}).get("original_session")
                                or s_dir.name
                            ),
                            "session_dir": origin_dir,
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
        intent = torch.tensor(one_hot(sample["intent"]), dtype=torch.float32)

        return image, steering, throttle, intent

    @property
    def source_session_dirs(self) -> List[Path]:
        """サンプルの元になったセッション（クリーナーの出力なら参照先の元データ）"""
        return sorted({sample["session_dir"] for sample in self.samples})

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """サンプルの画像パスやメタデータを取得"""
        return self.samples[idx]
