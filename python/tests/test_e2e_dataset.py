import json
from pathlib import Path

import pytest
from PIL import Image

from src.e2e.training.dataset import DrivingDataset, resolve_session_dirs


@pytest.fixture
def mock_dataset_tree(tmp_path: Path):
    """2つのセッションを持つ模擬データセット階層を作成"""
    raw_dir = tmp_path / "raw"
    session1 = raw_dir / "session_1"
    session2 = raw_dir / "session_2"

    for s_dir, count, steer_val in [(session1, 3, 0.2), (session2, 2, -0.4)]:
        img_dir = s_dir / "images"
        img_dir.mkdir(parents=True)
        labels = []
        for i in range(count):
            img_name = f"frame_{i:04d}.jpg"
            img = Image.new("RGB", (160, 120), color=(100, 150, 200))
            img.save(img_dir / img_name)
            labels.append(
                {
                    "frame": img_name,
                    "steering": steer_val,
                    "throttle": 0.5,
                    "timestamp": 1000.0 + i,
                }
            )

        with open(s_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for rec in labels:
                f.write(json.dumps(rec) + "\n")

    return raw_dir, session1, session2


def test_resolve_session_dirs(mock_dataset_tree):
    raw_dir, session1, session2 = mock_dataset_tree

    # 単一セッション
    assert resolve_session_dirs(session1) == [session1]

    # 親ディレクトリ
    resolved = resolve_session_dirs(raw_dir)
    assert len(resolved) == 2
    assert session1 in resolved
    assert session2 in resolved

    # リスト指定
    resolved_list = resolve_session_dirs([str(session1), str(session2)])
    assert len(resolved_list) == 2


def test_driving_dataset_single_session(mock_dataset_tree):
    _, session1, _ = mock_dataset_tree
    dataset = DrivingDataset(session1)
    assert len(dataset) == 3
    assert len(dataset.session_dirs) == 1

    img, steer, throttle = dataset[0]
    assert img.shape == (3, 120, 160)
    assert abs(steer.item() - 0.2) < 1e-4
    assert abs(throttle.item() - 0.5) < 1e-4


def test_driving_dataset_merged_sessions(mock_dataset_tree):
    raw_dir, _, _ = mock_dataset_tree
    # 親ディレクトリを指定して2つのセッション (3 + 2 = 5サンプル) を統合
    dataset = DrivingDataset(raw_dir)
    assert len(dataset) == 5
    assert len(dataset.session_dirs) == 2

    # 各サンプルが取得できるか
    for i in range(5):
        img, steer, _ = dataset[i]
        assert img.shape == (3, 120, 160)
