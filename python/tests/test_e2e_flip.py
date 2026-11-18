"""左右反転によるデータの補強: 画像、舵角、意図の反転と、学習での使い方"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.e2e.intent import one_hot
from src.e2e.training import train as train_module
from src.e2e.training.cross_validate import cross_validate
from src.e2e.training.dataset import DrivingDataset
from src.e2e.training.train import _Mirrored, _Targets, _training_subset, load_params, train


def make_session(root: Path, name: str, frames) -> Path:
    """frames: (steering, intent) の並び。画像の左半分を赤、右半分を青にする"""
    session = root / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, (steering, intent) in enumerate(frames):
            frame = f"frame_{i:04d}.png"
            image = Image.new("RGB", (160, 120), color=(0, 0, 255))
            image.paste((255, 0, 0), (0, 0, 80, 120))
            image.save(images / frame)
            record = {
                "frame": frame,
                "steering": steering,
                "throttle": 0.5,
                "intent": intent,
                "distance_cm": 20.0 if i % 2 else 150.0,
            }
            f.write(json.dumps(record) + "\n")
    return session


def test_mirrored_frame_flips_the_image_steering_and_intent(tmp_path: Path):
    dataset = DrivingDataset(make_session(tmp_path, "s1", [(0.6, "right"), (-0.3, "any")]))
    mirrored = _Mirrored(dataset, intent_pos=3)
    assert len(mirrored) == 4

    image, steering, throttle, intent = mirrored[0]  # 元のまま
    assert steering.item() == pytest.approx(0.6) and intent.tolist() == one_hot("right")

    m_image, m_steering, m_throttle, m_intent = mirrored[2]  # 0番の反転
    assert torch.equal(m_image, torch.flip(image, dims=[-1]))
    # 元は左が赤（赤のチャンネルが大きい）、反転すると右が赤
    assert image[0, 60, 10] > image[0, 60, 150]
    assert m_image[0, 60, 150] > m_image[0, 60, 10]
    assert m_steering.item() == pytest.approx(-0.6)
    assert m_intent.tolist() == one_hot("left")
    assert m_throttle.item() == throttle.item()

    assert mirrored[3][3].tolist() == one_hot("any")  # おまかせはそのまま


def test_mirrored_targets_keep_the_masks_and_blocked_label(tmp_path: Path):
    dataset = DrivingDataset(make_session(tmp_path, "s1", [(0.6, "left"), (0.2, "right")]))
    targets = _Targets(dataset)
    mirrored = _Mirrored(targets, intent_pos=2)
    original, flipped = targets[1], mirrored[3]
    assert flipped[1].item() == -original[1].item()
    assert flipped[2].tolist() == one_hot("left")
    for k in (3, 4, 5):  # 舵角を使うか、塞がっているか、そのラベルを使うか
        assert flipped[k].item() == original[k].item()


def test_training_subset_adds_mirrored_copies_only_when_asked(tmp_path: Path):
    dataset = DrivingDataset(make_session(tmp_path, "s1", [(0.1, "any")] * 3))
    assert len(_training_subset(dataset, [0, 2], {"flip_augment": False}, intent_pos=3)) == 2
    subset = _training_subset(dataset, [0, 2], {"flip_augment": True}, intent_pos=3)
    assert subset.indices == [0, 2, 3, 5]


def sessions(root: Path) -> Path:
    raw = root / "raw"
    make_session(raw, "s1", [(0.5, "right"), (0.0, "any"), (-0.5, "left"), (0.0, "any")])
    make_session(raw, "s2", [(0.4, "right"), (-0.4, "left"), (0.0, "any")])
    return raw


def test_flip_is_off_by_default_and_doubles_training_only(tmp_path: Path, monkeypatch):
    assert load_params()["flip_augment"] is False
    seen = []
    original = train_module._Mirrored.__getitem__

    def spy(self, idx):
        seen.append(idx)
        return original(self, idx)

    monkeypatch.setattr(train_module._Mirrored, "__getitem__", spy)
    raw = sessions(tmp_path)
    metadata = train(raw, tmp_path / "model", {"epochs": 1, "batch_size": 2, "flip_augment": True})
    assert metadata["params"]["flip_augment"] is True
    # 学習は s1 の4フレーム（0〜3）と、その反転（全体の7フレームの後ろの 7〜10）。
    # 検証（s2）は元のフレームだけ
    assert sorted(seen) == [0, 1, 2, 3, 7, 8, 9, 10]
    assert metadata["validation_metrics"]["samples"] == 3


def test_cross_validation_with_flip_and_blocked(tmp_path: Path):
    raw = sessions(tmp_path)
    params = {"epochs": 1, "batch_size": 2, "flip_augment": True, "blocked_loss_weight": 0.5}
    result = cross_validate(raw, params, log=lambda _m: None)
    assert result["overall"]["samples"] == 7
    assert result["params"]["flip_augment"] is True
