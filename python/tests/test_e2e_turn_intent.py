"""曲がりの区間に、曲がった向きを意図として付け直す（自動収集のデータ）と、交差検証の基準"""

import json
from pathlib import Path

from PIL import Image

from src.e2e.training.cross_validate import cross_validate
from src.e2e.training.dataset import DrivingDataset, turn_intents
from src.e2e.training.train import load_params, train


def rec(state, steering, intent="any"):
    return {"policy_state": state, "steering": steering, "intent": intent}


def test_avoid_turn_and_its_tail_get_the_direction_they_turned():
    records = [
        rec("cruise", 0.0),
        rec("avoid", 0.0),  # 曲がり始め。舵はまだ中央
        rec("avoid", 0.5),
        rec("avoid", 1.0),
        rec("cruise", 0.6),  # 舵が中央へ戻るまでの続き
        rec("cruise", 0.1),
        rec("cruise", 0.0),
    ]
    assert turn_intents(records) == ["any", "right", "right", "right", "right", "any", "any"]


def test_left_after_backing_out_is_left():
    records = [
        rec("avoid", 1.0),
        rec("backup", 1.0),
        rec("pivot", -1.0),
        rec("avoid", -1.0),
        rec("cruise", -0.5),
        rec("cruise", 0.0),
    ]
    assert turn_intents(records) == ["right", "any", "any", "left", "left", "any"]


def test_recorded_side_intents_are_kept_and_other_states_untouched():
    records = [
        rec("lead", 0.0, "left"),
        rec("open_turn", -1.0, "left"),
        rec("cruise", -0.4),
        rec("approach", 0.0, "right"),
        rec("cruise", 0.8),  # 曲がりの区間の後ではない曲がりは変えない
        {"steering": -1.0},  # 状態のない記録（人の収集）
    ]
    assert turn_intents(records) == ["left", "left", "left", "right", "any", "any"]


def test_a_turn_that_never_steers_keeps_any():
    assert turn_intents([rec("avoid", 0.0), rec("avoid", 0.1)]) == ["any", "any"]


def make_session(root: Path, name: str, records) -> Path:
    session = root / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            frame = f"frame_{i:04d}.jpg"
            Image.new("RGB", (160, 120), color=(40 + i * 9, 90, 140)).save(images / frame)
            full = {"frame": frame, "throttle": 0.5, "command": {"speed": 50}, **record}
            f.write(json.dumps(full) + "\n")
    return session


def turning_sessions(root: Path) -> Path:
    raw = root / "raw"
    for name in ("s1", "s2"):
        make_session(
            raw,
            name,
            [rec("cruise", 0.0), rec("avoid", 1.0), rec("cruise", 0.5)]
            + [rec("cruise", 0.0), rec("avoid", -1.0), rec("cruise", -0.5)],
        )
    return raw


def test_dataset_relabels_only_when_asked(tmp_path: Path):
    raw = turning_sessions(tmp_path)
    plain = DrivingDataset(raw)
    assert {s["intent"] for s in plain.samples} == {"any"}
    assert plain.relabeled_turn_intent == 0

    relabeled = DrivingDataset(raw, turn_intent=True)
    assert [s["intent"] for s in relabeled.samples[:6]] == [
        "any",
        "right",
        "right",
        "any",
        "left",
        "left",
    ]
    assert [s["intent_recorded"] for s in relabeled.samples[:6]] == ["any"] * 6
    assert relabeled.relabeled_turn_intent == 8


def test_turn_intent_is_off_by_default_and_recorded_in_metadata(tmp_path: Path):
    assert load_params()["turn_intent"] is False
    raw = turning_sessions(tmp_path)
    metadata = train(raw, tmp_path / "model", {"epochs": 1, "batch_size": 4, "turn_intent": True})
    assert metadata["params"]["turn_intent"] is True
    assert metadata["data"]["relabeled_turn_intent"] == 8
    assert metadata["data"]["samples_by_intent"] == {"any": 4, "left": 4, "right": 4}


def test_cross_validation_reports_image_free_baselines(tmp_path: Path):
    raw = turning_sessions(tmp_path)
    result = cross_validate(raw, {"epochs": 1, "batch_size": 4, "turn_intent": True}, log=print)
    zero = result["baselines"]["zero"]
    intent_mean = result["baselines"]["intent_mean"]
    # 曲がりの舵角は ±1.0 と ±0.5、直進は 0.0
    assert zero["steering_mae"] == round((1.0 + 0.5) * 2 / 6, 4)
    # 意図ごとの平均は右 0.75、左 -0.75、おまかせ 0 なので、誤差は曲がりで 0.25 ずつ
    assert intent_mean["steering_mae"] == round(0.25 * 4 / 6, 4)
    assert intent_mean["samples"] == result["overall"]["samples"]
