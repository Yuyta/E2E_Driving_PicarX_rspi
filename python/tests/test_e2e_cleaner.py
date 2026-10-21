import json
from pathlib import Path

import torch
from PIL import Image

from src.e2e.dataset.cleaner import DatasetCleaner
from src.e2e.training.model import SmallDrivingCNN


def test_cleaner_filter_samples():
    cleaner = DatasetCleaner.__new__(DatasetCleaner)
    cleaner.threshold = 0.4
    cleaner.top_k_percent = None

    analyzed = [
        {"abs_error": 0.1, "raw_record": {"frame": "f1.jpg"}},
        {"abs_error": 0.5, "raw_record": {"frame": "f2.jpg"}},
        {"abs_error": 0.2, "raw_record": {"frame": "f3.jpg"}},
        {"abs_error": 0.8, "raw_record": {"frame": "f4.jpg"}},
    ]

    clean, bad = cleaner.filter_samples(analyzed)
    assert len(clean) == 2
    assert len(bad) == 2
    assert [item["raw_record"]["frame"] for item in clean] == ["f1.jpg", "f3.jpg"]
    assert [item["raw_record"]["frame"] for item in bad] == ["f2.jpg", "f4.jpg"]


def test_cleaner_top_k_filter():
    cleaner = DatasetCleaner.__new__(DatasetCleaner)
    cleaner.threshold = None
    cleaner.top_k_percent = 25.0  # 4サンプルのうち上位25% (最も悪い1件)

    analyzed = [
        {"abs_error": 0.1, "raw_record": {"frame": "f1.jpg"}},
        {"abs_error": 0.2, "raw_record": {"frame": "f2.jpg"}},
        {"abs_error": 0.3, "raw_record": {"frame": "f3.jpg"}},
        {"abs_error": 0.9, "raw_record": {"frame": "f4.jpg"}},
    ]

    clean, bad = cleaner.filter_samples(analyzed)
    assert len(clean) == 3
    assert len(bad) == 1
    assert bad[0]["raw_record"]["frame"] == "f4.jpg"


def test_cleaner_pipeline(tmp_path: Path):
    # ダミーモデル (.pth) の保存
    model_path = tmp_path / "dummy.pth"
    model = SmallDrivingCNN()
    torch.save(model.state_dict(), model_path)

    # ダミーデータの作成
    data_dir = tmp_path / "session_1"
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True)

    records = []
    for i in range(4):
        name = f"frame_{i:04d}.jpg"
        img = Image.new("RGB", (160, 120), color=(100, 100, 100))
        img.save(img_dir / name)
        # ステアリング角を0.0と1.0で極端に分ける
        records.append({"frame": name, "steering": 0.0 if i < 3 else 1.0, "throttle": 0.5})

    with open(data_dir / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    output_dir = tmp_path / "cleaned"
    cleaner = DatasetCleaner(model_path=model_path, threshold=0.5)
    report = cleaner.clean(data_dir=data_dir, output_dir=output_dir)

    assert report["total_samples"] == 4
    assert (output_dir / "labels.jsonl").exists()
    assert (output_dir / "cleaning_report.json").exists()
    assert (output_dir / "metadata.json").exists()
