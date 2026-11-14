"""行きたい向き（意図）を入力にする走行モデル: 記録の読み込み、分岐型のモデル、ONNX、指標、推論"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from src.e2e import autonomous_driver
from src.e2e.evaluation.metrics import steering_metrics
from src.e2e.intent import INTENTS, intent_index, intent_of, one_hot
from src.e2e.training.dataset import DrivingDataset
from src.e2e.training.model import SmallDrivingCNN
from src.e2e.training.train import export_onnx, train

# --- 意図の値 ---


def test_intent_values_and_one_hot():
    assert INTENTS == ("any", "left", "right")  # モデルの頭の並び。変えると学習済みと合わない
    assert intent_index(None) == 0  # 記録のない古いフレームは おまかせ
    assert one_hot("left") == [0.0, 1.0, 0.0]
    assert one_hot("right") == [0.0, 0.0, 1.0]
    assert intent_of(True, True) == "any"
    with pytest.raises(ValueError):
        intent_index("straight")


# --- 記録の読み込み ---


def make_session(root: Path, name: str, frames) -> Path:
    """frames: (steering, intent か None) の並び"""
    session = root / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, (steering, intent) in enumerate(frames):
            frame = f"frame_{i:04d}.jpg"
            Image.new("RGB", (160, 120), color=(60 + i * 7, 80, 120)).save(images / frame)
            record = {"frame": frame, "steering": steering, "throttle": 0.5}
            if intent is not None:
                record["intent"] = intent
            f.write(json.dumps(record) + "\n")
    return session


def test_dataset_reads_the_intent_and_treats_missing_as_any(tmp_path: Path):
    session = make_session(tmp_path, "s1", [(0.0, None), (-0.6, "left"), (0.6, "right")])
    dataset = DrivingDataset(session)
    assert [s["intent"] for s in dataset.samples] == ["any", "left", "right"]
    _, _, _, intent = dataset[1]
    assert intent.tolist() == [0.0, 1.0, 0.0]


# --- 分岐型のモデル ---


def test_each_intent_uses_its_own_head():
    torch.manual_seed(0)
    model = SmallDrivingCNN().eval()
    image = torch.rand(1, 3, 120, 160)
    with torch.no_grad():
        outputs = {name: model(image, torch.tensor([one_hot(name)])).item() for name in INTENTS}
        default = model(image).item()
    assert default == outputs["any"]  # 意図を省くと おまかせ
    assert len(set(outputs.values())) == 3  # 頭ごとに違う値

    # 1つの頭の出力だけを変えても、ほかの意図の出力は変わらない
    with torch.no_grad():
        model.heads[1][2].bias.add_(1.0)
        after_any = model(image, torch.tensor([one_hot("any")])).item()
    assert after_any == outputs["any"]


def test_onnx_takes_the_intent_and_matches_pytorch(tmp_path: Path):
    torch.manual_seed(0)
    model = SmallDrivingCNN().eval()
    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    session = ort.InferenceSession(str(path))
    assert [i.name for i in session.get_inputs()] == ["input", "intent"]
    images = torch.rand(2, 3, 120, 160)
    intents = torch.tensor([one_hot("left"), one_hot("right")])
    onnx_out = session.run(None, {"input": images.numpy(), "intent": intents.numpy()})[0]
    with torch.no_grad():
        torch_out = model(images, intents).numpy()
    assert np.allclose(onnx_out, torch_out, atol=1e-5)


# --- 学習と指標 ---


def test_steering_metrics_by_intent():
    metrics = steering_metrics(
        [0.0, -0.5, 0.5], [0.1, -0.7, 0.5], ["s", "s", "s"], ["any", "left", "right"]
    )
    assert metrics["steering_mae_by_intent"] == {"any": 0.1, "left": 0.2, "right": 0.0}
    assert metrics["samples_by_intent"] == {"any": 1, "left": 1, "right": 1}
    # 意図を渡さなければ、これまでと同じ項目だけ
    assert "steering_mae_by_intent" not in steering_metrics([0.0], [0.0], ["s"])


def test_training_records_the_intent_input(tmp_path: Path):
    raw = tmp_path / "raw"
    make_session(raw, "s1", [(0.0, None), (-0.6, "left"), (0.5, "right"), (0.1, None)])
    make_session(raw, "s2", [(0.0, "any"), (-0.4, "left"), (0.4, "right")])
    metadata = train(raw, tmp_path / "model", {"epochs": 1})

    assert metadata["intent_input"] == {"shape": [1, 3], "order": ["any", "left", "right"]}
    assert metadata["data"]["samples_by_intent"] == {"any": 3, "left": 2, "right": 2}
    assert metadata["validation_metrics"]["samples_by_intent"] == {"any": 1, "left": 1, "right": 1}


# --- 自律走行での推論 ---


class FakeSession:
    def __init__(self, names):
        self.names = names
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.names]

    def get_outputs(self):
        return [SimpleNamespace(name="steering")]

    def run(self, _outputs, feed):
        self.feeds.append(feed)
        return [np.array([[0.0]], dtype=np.float32)]


def make_driver(monkeypatch, tmp_path, names, intent="any"):
    """避ける判断は切り、渡した意図がそのままモデルへ届くかを見る"""
    session = FakeSession(names)
    monkeypatch.setattr(autonomous_driver.ort, "InferenceSession", lambda *_a, **_k: session)
    driver = autonomous_driver.RemoteAutonomousDriver(
        model_path="fake.onnx",
        enable_mapping=False,
        enable_stuck_detection=False,
        log_dir=str(tmp_path / "disengagements"),
        intent=intent,
        enable_avoidance=False,
    )
    return driver, session


def fed_intent(driver, session):
    driver.infer(np.zeros((1, 3, 120, 160), dtype=np.float32))
    feed = session.feeds[-1]
    return feed["intent"].tolist() if "intent" in feed else None


def test_driver_feeds_the_intent_to_a_model_that_takes_it(monkeypatch, tmp_path):
    driver, session = make_driver(monkeypatch, tmp_path, ["input", "intent"], intent="right")
    assert fed_intent(driver, session) == [[0.0, 0.0, 1.0]]
    driver.set_intent("left")
    assert fed_intent(driver, session) == [[0.0, 1.0, 0.0]]
    with pytest.raises(ValueError):
        driver.set_intent("straight")


def test_driver_refuses_an_intent_the_model_cannot_take(monkeypatch, tmp_path):
    driver, session = make_driver(monkeypatch, tmp_path, ["input"])
    assert fed_intent(driver, session) is None
    with pytest.raises(ValueError):
        driver.set_intent("left")
