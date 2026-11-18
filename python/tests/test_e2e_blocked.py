"""前方が塞がっているかの補助出力: 超音波からのラベル、読み込み、モデル、ONNX、学習、交差検証"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
from PIL import Image

from src.e2e.evaluation.metrics import blocked_metrics
from src.e2e.intent import one_hot
from src.e2e.training.cross_validate import cross_validate
from src.e2e.training.dataset import DrivingDataset, blocked_label, blocked_labels, is_stuck
from src.e2e.training.model import SmallDrivingCNN
from src.e2e.training.train import export_onnx, train

# --- 超音波からのラベル ---


@pytest.mark.parametrize(
    "record, expected",
    [
        ({"distance_cm": 20.0}, True),
        ({"distance_cm": 39.9}, True),
        ({"distance_cm": 40.0}, None),  # 境界の間は決めない
        ({"distance_cm": 59.9}, None),
        ({"distance_cm": 60.0}, False),
        ({"distance_cm": None, "distance_far": True}, False),  # 反射が戻らないのは遠い時
        ({"distance_cm": None, "distance_far": False}, None),
        ({}, None),  # 距離のない古い記録
    ],
)
def test_blocked_label_from_the_ultrasonic_distance(record, expected):
    assert blocked_label(record) is expected


def test_blocked_label_rejects_reversed_bounds():
    with pytest.raises(ValueError):
        blocked_label({"distance_cm": 50.0}, near_cm=70.0, clear_cm=60.0)


# --- 読み込み ---


def make_session(root: Path, name: str, frames) -> Path:
    """frames: 記録の dict の並び。frame と画像はここで作る。画像の色は塞がっているかで変える"""
    session = root / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, extra in enumerate(frames):
            frame = f"frame_{i:04d}.jpg"
            near = (extra.get("distance_cm") or 999) < 40
            color = (220, 30, 30) if near else (30, 30, 220)
            Image.new("RGB", (160, 120), color=color).save(images / frame)
            record = {
                "frame": frame,
                "steering": 0.0,
                "throttle": 0.5,
                "intent": "any",
                "command": {"speed": 50, "steering_deg": 0.0},
                **extra,
            }
            f.write(json.dumps(record) + "\n")
    return session


def stopped(distance):
    return {"distance_cm": distance, "throttle": 0.0, "command": {"speed": 0}}


def test_dataset_keeps_frames_left_out_of_steering_only_when_asked(tmp_path: Path):
    session = make_session(
        tmp_path,
        "s1",
        [
            {"distance_cm": 100.0, "steering": 0.1},
            stopped(20.0),  # 止まっている
            {"distance_cm": 25.0, "command": {"speed": -50}},  # 下がっている
            {"distance_cm": 30.0, "exclude": True},  # 曲がれなかった
            {"distance_cm": 30.0, "motion_check": {"flags": ["no_motion"]}},  # 空転
            stopped(50.0),  # 境界の間なので、塞がっているかにも使えない
        ],
    )

    # 引っかかったフレームの扱い（blocked_stuck）は下の別のテストで確かめる
    plain = DrivingDataset(session, blocked_stuck=False)
    assert len(plain) == 1  # これまでと同じ
    assert plain.kept_for_blocked_only == 0
    assert plain.samples[0]["steer_valid"] is True
    assert plain.samples[0]["blocked"] is False

    both = DrivingDataset(session, keep_for_blocked=True, blocked_stuck=False)
    assert [s["steer_valid"] for s in both.samples] == [True, False, False, False, False]
    assert [s["blocked"] for s in both.samples] == [False, True, True, True, True]
    assert both.kept_for_blocked_only == 4
    assert both.steer_indices() == [0]
    assert both.blocked_indices() == [0, 1, 2, 3, 4]
    # 舵角から外した理由の数は、残したかどうかに関わらず同じ
    assert (both.skipped_stopped, both.skipped_motion_mismatch) == (
        plain.skipped_stopped,
        plain.skipped_motion_mismatch,
    )


def stuck(distance=None, far=True):
    """前へ進む指令なのに動けない。反射が戻らず「遠い」と読めることが多い"""
    return {
        "distance_cm": distance,
        "distance_far": far,
        "command": {"speed": 50},
        "motion_check": {"flags": ["no_motion"]},
    }


def test_is_stuck_needs_a_forward_command():
    assert is_stuck(stuck())
    assert not is_stuck({**stuck(), "command": {"speed": -50}})  # 下がっている
    assert not is_stuck({**stuck(), "motion_check": {"flags": ["turn_mismatch"]}})
    assert not is_stuck({"distance_far": True, "command": {"speed": 50}})
    # 指令のない古い記録は throttle で決める
    assert is_stuck({"throttle": 0.5, "motion_check": {"flags": ["no_motion"]}})


def test_stuck_frames_are_blocked_and_nearby_clear_labels_unknown():
    far = {"distance_cm": None, "distance_far": True}
    records = [far] * 5 + [stuck(), stuck(80.0, False)] + [far] * 2 + [{"distance_cm": 20.0}]
    assert blocked_labels(records, stuck=False) == [False] * 5 + [False, False] + [False] * 2 + [
        True
    ]
    labels = blocked_labels(records, stuck_context=2)
    assert labels == [False] * 3 + [None] * 2 + [True, True] + [None] * 2 + [True]


def test_dataset_counts_the_stuck_labels(tmp_path: Path):
    frames = [{"distance_cm": None, "distance_far": True}] * 20 + [stuck()] * 3
    session = make_session(tmp_path, "s1", frames)
    dataset = DrivingDataset(session, keep_for_blocked=True)
    assert dataset.stuck_blocked == 3
    assert dataset.stuck_unknown == 15  # 前の 15 フレーム（約1秒）
    # 舵角に使えるフレームは、塞がっているかが不明になっても残る
    assert [s["blocked"] for s in dataset.samples] == [False] * 5 + [None] * 15 + [True] * 3
    assert dataset.blocked_indices() == [0, 1, 2, 3, 4, 20, 21, 22]
    assert DrivingDataset(session, keep_for_blocked=True, blocked_stuck=False).stuck_blocked == 0


# --- モデルと ONNX ---


def test_forward_returns_the_same_steering_as_forward_with_blocked():
    torch.manual_seed(0)
    model = SmallDrivingCNN().eval()
    image = torch.rand(2, 3, 120, 160)
    intent = torch.tensor([one_hot("left"), one_hot("any")])
    with torch.no_grad():
        steering, logit = model.forward_with_blocked(image, intent)
        assert torch.equal(model(image, intent), steering)
    assert logit.shape == (2, 1)


def test_onnx_outputs_steering_first_and_a_blocked_probability(tmp_path: Path):
    torch.manual_seed(0)
    model = SmallDrivingCNN().eval()
    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    session = ort.InferenceSession(str(path))
    assert [o.name for o in session.get_outputs()] == ["steering", "blocked"]
    images = torch.rand(2, 3, 120, 160)
    intents = torch.tensor([one_hot("any"), one_hot("right")])
    steering, blocked = session.run(None, {"input": images.numpy(), "intent": intents.numpy()})
    with torch.no_grad():
        torch_steering, logit = model.forward_with_blocked(images, intents)
    assert np.allclose(steering, torch_steering.numpy(), atol=1e-5)
    assert np.allclose(blocked, torch.sigmoid(logit).numpy(), atol=1e-5)
    assert ((blocked >= 0) & (blocked <= 1)).all()


# --- 指標 ---


def test_blocked_metrics():
    metrics = blocked_metrics([0.9, 0.2, 0.7, 0.1], [True, True, False, False])
    assert metrics["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["always_clear_accuracy"] == 0.5
    assert blocked_metrics([], [])["samples"] == 0
    # 塞がっている例がなければ、再現率は決まらない
    assert blocked_metrics([0.1], [False])["recall"] is None


# --- 学習と交差検証 ---


def colored_sessions(root: Path) -> Path:
    """赤い画像は前が近い、青い画像は遠い。近いフレームの半分は止まっている"""
    raw = root / "raw"
    for name in ("s1", "s2"):
        frames = []
        for i in range(8):
            frames.append({"distance_cm": 150.0 + i, "steering": 0.0})
            frames.append({"distance_cm": 20.0 + i, "steering": 0.8})
            frames.append(stopped(15.0 + i))
        make_session(raw, name, frames)
    return raw


def test_training_without_the_weight_ignores_the_blocked_head(tmp_path: Path):
    raw = colored_sessions(tmp_path)
    metadata = train(raw, tmp_path / "model", {"epochs": 1, "batch_size": 8})
    assert metadata["blocked_output"]["trained"] is False
    assert metadata["data"]["kept_for_blocked_only"] == 0
    assert "blocked" not in metadata["validation_metrics"]
    assert metadata["validation_metrics"]["samples"] == 16  # 止まっているフレームは使わない


def test_training_learns_whether_the_way_ahead_is_blocked(tmp_path: Path):
    raw = colored_sessions(tmp_path)
    params = {"epochs": 8, "batch_size": 8, "blocked_loss_weight": 1.0}
    metadata = train(raw, tmp_path / "model", params)

    assert metadata["blocked_output"]["trained"] is True
    assert metadata["data"]["kept_for_blocked_only"] == 16
    assert metadata["data"]["blocked_labels"] == {"blocked": 32, "clear": 16, "unknown": 0}
    validation = metadata["validation_metrics"]
    assert validation["samples"] == 16  # 舵角は止まっているフレームを除いて評価する
    assert validation["blocked"]["samples"] == 24
    assert validation["blocked"]["recall"] == 1.0
    assert validation["blocked"]["accuracy"] == 1.0
    assert "train_blocked_loss" in metadata["history"][0]

    session = ort.InferenceSession(str(tmp_path / "model" / "model.onnx"))
    assert [o.name for o in session.get_outputs()] == ["steering", "blocked"]


def test_cross_validation_reports_blocked_metrics(tmp_path: Path):
    raw = colored_sessions(tmp_path)
    result = cross_validate(
        raw, {"epochs": 1, "batch_size": 8, "blocked_loss_weight": 1.0}, log=lambda _m: None
    )
    assert result["overall"]["samples"] == 32
    assert result["overall"]["blocked"]["samples"] == 48
    assert all(fold["blocked"]["samples"] == 24 for fold in result["folds"])

    plain = cross_validate(raw, {"epochs": 1, "batch_size": 8}, log=lambda _m: None)
    assert "blocked" not in plain["overall"]


# --- Pi の tract で読める形での書き出し ---


@pytest.mark.parametrize("in_size, out_size", [(10, 4), (15, 4), (7, 2), (8, 4)])
def test_matrix_pool_matches_adaptive_avg_pool(in_size, out_size):
    from src.e2e.training.model import MatrixAdaptivePool

    torch.manual_seed(0)
    x = torch.rand(2, 5, in_size, in_size + 3)
    pool = MatrixAdaptivePool((in_size, in_size + 3), (out_size, out_size))
    expected = torch.nn.AdaptiveAvgPool2d(out_size)(x)
    assert torch.allclose(pool(x), expected, atol=1e-6)


def test_onnx_uses_only_ops_that_tract_reads(tmp_path: Path):
    import onnx

    torch.manual_seed(0)
    path = tmp_path / "model.onnx"
    export_onnx(SmallDrivingCNN().eval(), path)
    ops = {node.op_type for node in onnx.load(str(path)).graph.node}
    # AdaptiveAvgPool2d の 10 x 15 -> 4 x 4 は GatherND になり、tract が読めなかった（2026-11-14）
    assert not ops & {"GatherND", "Gather", "Where"}
    assert "MatMul" in ops


def test_cross_validation_reports_stuck_recall_and_thresholds(tmp_path: Path):
    raw = tmp_path / "raw"
    for name in ("s1", "s2"):
        frames = []
        for i in range(8):
            frames.append({"distance_cm": 150.0 + i, "steering": 0.0})
            frames.append({"distance_cm": 20.0 + i, "steering": 0.8})
        # 赤い画像（前が近い時と同じ色）で、超音波には映らずに引っかかった
        frames += [{**stuck(15.0, False), "distance_cm": 15.0}] * 2
        frames += [{**stuck(), "distance_cm": 39.0}]
        make_session(raw, name, frames)

    result = cross_validate(
        raw,
        {"epochs": 1, "batch_size": 8, "blocked_loss_weight": 1.0},
        log=lambda _m: None,
    )
    detail = result["overall"]["blocked_detail"]
    assert detail["stuck_samples"] == 0  # 超音波でも近いので、引っかかったことで変わらない
    assert [row["threshold"] for row in detail["thresholds"]] == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert detail["ultrasonic"]["samples"] == result["overall"]["blocked"]["samples"]


def test_blocked_detail_separates_the_stuck_labels():
    from src.e2e.training.cross_validate import blocked_detail

    detail = blocked_detail(
        [0.9, 0.2, 0.7, 0.1, 0.3],
        [True, True, True, False, False],
        [False, False, True, False, False],
    )
    assert detail["stuck_samples"] == 1
    assert detail["stuck_recall"] == 1.0
    assert detail["ultrasonic"]["samples"] == 4
    assert detail["thresholds"][0]["recall"] == 1.0  # 0.2 以上を塞がっているとすると全部当たる
