"""
Unit Tests for PiCar-X Disengagement Logger & Manager (Phase 2.1)
"""

import json

import numpy as np
import pytest

from src.e2e.dataset.disengagement_manager import (
    DisengagementBuffer,
    DisengagementManager,
)
from src.e2e.training.dataset import DrivingDataset


@pytest.fixture
def dummy_frame():
    """160x120のダミーBGR画像"""
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_disengagement_buffer_and_save(tmp_path, dummy_frame):
    buffer = DisengagementBuffer(max_frames=5)

    # 7フレーム追加（バッファサイズ5を超えるので直近5個のみ残る）
    for i in range(7):
        buffer.append(
            frame_bgr=dummy_frame,
            steering_norm=0.1 * i,
            steer_angle=int(3.5 * i),
            ultrasonic_dist=50.0 - i * 5.0,
            speed=35,
        )

    assert len(buffer.buffer) == 5

    # イベント保存
    saved_dir = buffer.save_event(
        trigger="obstacle_stop",
        output_base_dir=tmp_path,
        model_name="test_model.onnx",
        note="test note",
    )

    assert saved_dir is not None
    assert saved_dir.exists()
    assert (saved_dir / "event.json").exists()
    assert (saved_dir / "labels.jsonl").exists()
    assert (saved_dir / "images").exists()

    # event.json の検証
    with open(saved_dir / "event.json", "r", encoding="utf-8") as f:
        event_meta = json.load(f)
    assert event_meta["trigger"] == "obstacle_stop"
    assert event_meta["frames_count"] == 5
    assert event_meta["status"] == "pending_review"
    assert event_meta["exclude_from_training"] is True

    # labels.jsonl の検証
    with open(saved_dir / "labels.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 5
    first_record = json.loads(lines[0])
    assert first_record["exclude"] is True
    assert first_record["tag"] == "disengagement"


def test_disengagement_manager_review_flow(tmp_path, dummy_frame):
    buffer = DisengagementBuffer(max_frames=3)
    for _ in range(3):
        buffer.append(
            frame_bgr=dummy_frame,
            steering_norm=0.0,
            steer_angle=0,
            ultrasonic_dist=10.0,
            speed=35,
        )

    event_dir = buffer.save_event(
        trigger="manual_intervention",
        output_base_dir=tmp_path,
        model_name="v001",
    )
    event_id = event_dir.name

    manager = DisengagementManager(base_dir=tmp_path)
    events = manager.list_events()
    assert len(events) == 1
    assert events[0]["event_id"] == event_id
    assert events[0]["status"] == "pending_review"

    # 1. 誤検知として reject
    ok = manager.reject_event(event_id, delete=False)
    assert ok is True
    with open(event_dir / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["status"] == "rejected"
    assert meta["exclude_from_training"] is True

    # 2. 真の失敗ケースとして approve
    ok = manager.approve_event(event_id)
    assert ok is True
    with open(event_dir / "event.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["status"] == "approved"
    assert meta["exclude_from_training"] is False

    # 3. 物理削除 delete
    ok = manager.reject_event(event_id, delete=True)
    assert ok is True
    assert not event_dir.exists()


def test_dataset_excludes_disengagement_by_default(tmp_path, dummy_frame):
    buffer = DisengagementBuffer(max_frames=3)
    for _ in range(3):
        buffer.append(
            frame_bgr=dummy_frame,
            steering_norm=0.0,
            steer_angle=0,
            ultrasonic_dist=12.0,
            speed=35,
        )

    # 未承認のイベント保存 (exclude: True)
    event_dir = buffer.save_event(
        trigger="obstacle_stop",
        output_base_dir=tmp_path,
        model_name="v001",
    )

    # デフォルトでは exclude: true は読み込まれず0件になる
    dataset = DrivingDataset(data_dir=tmp_path, include_excluded=False)
    assert len(dataset) == 0

    # include_excluded=True の場合のみ読み込まれる
    dataset_with_excluded = DrivingDataset(data_dir=tmp_path, include_excluded=True)
    assert len(dataset_with_excluded) == 3

    # approve 後は exclude: false になるため、デフォルトでも読み込まれる
    manager = DisengagementManager(base_dir=tmp_path)
    manager.approve_event(event_dir.name)

    dataset_approved = DrivingDataset(data_dir=tmp_path, include_excluded=False)
    assert len(dataset_approved) == 3
