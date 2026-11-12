"""M3 のデータとモデルの管理: マニフェスト、画像を複製しないクリーナー、再現できる学習"""

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.e2e.dataset.cleaner import DatasetCleaner
from src.e2e.dataset.manifest import (
    MANIFEST_NAME,
    build_manifest,
    verify_session,
    write_manifest,
)
from src.e2e.evaluation.metrics import steering_metrics
from src.e2e.training.cross_validate import cross_validate
from src.e2e.training.dataset import DrivingDataset
from src.e2e.training.model import SmallDrivingCNN
from src.e2e.training.train import TrainingError, load_params, split_by_session, train


def make_session(root: Path, name: str, steerings, shade: int = 100) -> Path:
    session = root / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, steering in enumerate(steerings):
            frame = f"frame_{i:04d}.jpg"
            # 画像ごとに少し変えて、学習が意味を持つようにする
            Image.new("RGB", (160, 120), color=(shade + i * 5, 80, 120)).save(images / frame)
            f.write(json.dumps({"frame": frame, "steering": steering, "throttle": 0.5}) + "\n")
    return session


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    make_session(raw, "s1", [-0.5, -0.4, 0.0, 0.1], shade=60)
    make_session(raw, "s2", [0.0, 0.3, 0.5], shade=110)
    make_session(raw, "s3", [-0.2, 0.0, 0.4], shade=160)
    return raw


# --- マニフェスト ---


def test_manifest_verifies_an_unchanged_session(raw_dir: Path):
    session = raw_dir / "s1"
    manifest = write_manifest(session)

    assert manifest["file_count"] == 5  # labels.jsonl と画像4枚
    assert "labels.jsonl" in manifest["files"]
    assert verify_session(session).ok


def test_manifest_detects_changed_missing_and_added_files(raw_dir: Path):
    session = raw_dir / "s1"
    write_manifest(session)

    (session / "labels.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "images" / "frame_0000.jpg").unlink()
    (session / "images" / "extra.jpg").write_bytes(b"x")

    result = verify_session(session)
    assert not result.ok
    assert result.changed == ["labels.jsonl"]
    assert result.missing == ["images/frame_0000.jpg"]
    assert result.added == ["images/extra.jpg"]


def test_manifest_is_never_overwritten(raw_dir: Path):
    write_manifest(raw_dir / "s1")
    with pytest.raises(FileExistsError):
        write_manifest(raw_dir / "s1")


def test_manifest_ignores_derived_metadata(raw_dir: Path):
    session = raw_dir / "s1"
    write_manifest(session)
    (session / "metadata.json").write_text("{}", encoding="utf-8")
    assert verify_session(session).ok


def test_manifest_hash_depends_on_content(raw_dir: Path):
    before = build_manifest(raw_dir / "s1")["hash"]
    (raw_dir / "s1" / "labels.jsonl").write_text("{}\n", encoding="utf-8")
    assert build_manifest(raw_dir / "s1")["hash"] != before


def test_session_without_manifest_is_not_ok(raw_dir: Path):
    result = verify_session(raw_dir / "s2")
    assert not result.has_manifest
    assert not result.ok
    assert not (raw_dir / "s2" / MANIFEST_NAME).exists()


# --- クリーナー: 画像を複製しない ---


def test_cleaner_references_raw_images_instead_of_copying(tmp_path: Path, raw_dir: Path):
    model_path = tmp_path / "model.pth"
    torch.save(SmallDrivingCNN().state_dict(), model_path)
    out = tmp_path / "processed" / "clean"

    DatasetCleaner(model_path=model_path, threshold=None).clean(raw_dir, out)

    assert not (out / "images").exists()
    records = [json.loads(line) for line in (out / "labels.jsonl").read_text().splitlines()]
    assert len(records) == 10
    assert all(not Path(r["source_dir"]).is_absolute() for r in records)

    dataset = DrivingDataset(out)
    assert len(dataset) == 10
    # 分割と集計は元のセッション単位
    assert {s["session_id"] for s in dataset.samples} == {"s1", "s2", "s3"}
    assert {p.name for p in dataset.source_session_dirs} == {"s1", "s2", "s3"}
    image, steering, _, _ = dataset[0]
    assert image.shape == (3, 120, 160)


def test_legacy_cleaned_output_keeps_the_original_session(tmp_path: Path):
    legacy = tmp_path / "clean_v1"
    (legacy / "images").mkdir(parents=True)
    Image.new("RGB", (160, 120)).save(legacy / "images" / "s9_frame_0000.jpg")
    record = {
        "frame": "s9_frame_0000.jpg",
        "steering": 0.1,
        "cleaned_meta": {"original_session": "s9"},
    }
    (legacy / "labels.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = DrivingDataset(legacy)
    assert dataset.samples[0]["session_id"] == "s9"
    assert dataset.samples[0]["image_path"].exists()


# --- 設定と分割 ---


def test_load_params_merges_file_and_overrides(tmp_path: Path):
    config = tmp_path / "train.toml"
    config.write_text('epochs = 3\nseed = 7\nval_sessions = ["s2"]\n', encoding="utf-8")

    params = load_params(str(config), {"seed": 9, "lr": None})

    assert params["epochs"] == 3
    assert params["seed"] == 9  # コマンドラインが優先
    assert params["lr"] == 1e-3  # None は上書きしない
    assert params["val_sessions"] == ["s2"]


def test_load_params_rejects_unknown_keys(tmp_path: Path):
    config = tmp_path / "train.toml"
    config.write_text("epoch = 3\n", encoding="utf-8")
    with pytest.raises(TrainingError):
        load_params(str(config))


def test_split_holds_out_whole_sessions(raw_dir: Path):
    dataset = DrivingDataset(raw_dir)

    train_idx, val_idx, val_sessions = split_by_session(dataset, [])

    assert val_sessions == ["s3"]
    assert {dataset.samples[i]["session_id"] for i in val_idx} == {"s3"}
    assert "s3" not in {dataset.samples[i]["session_id"] for i in train_idx}


def test_split_refuses_a_single_session(raw_dir: Path):
    with pytest.raises(TrainingError):
        split_by_session(DrivingDataset(raw_dir / "s1"), [])


def test_split_refuses_unknown_validation_session(raw_dir: Path):
    with pytest.raises(TrainingError):
        split_by_session(DrivingDataset(raw_dir), ["nope"])


# --- 指標 ---


def test_steering_metrics_by_direction_and_change():
    metrics = steering_metrics(
        predictions=[-0.4, 0.0, 0.5, 0.1],
        targets=[-0.5, 0.1, 0.3, 0.0],
        session_ids=["a", "a", "a", "b"],
    )

    assert metrics["steering_mae"] == pytest.approx(0.125)
    assert metrics["steering_mae_by_direction"] == {"left": 0.1, "straight": 0.1, "right": 0.2}
    assert metrics["samples_by_direction"] == {"left": 1, "straight": 2, "right": 1}
    # セッションをまたぐ差は数えない: |0.0+0.4| と |0.5-0.0| の平均
    assert metrics["steering_change"] == pytest.approx(0.45)


def test_steering_metrics_marks_missing_directions():
    metrics = steering_metrics([0.0], [0.0], ["a"])
    assert metrics["steering_mae_by_direction"]["left"] is None
    assert metrics["steering_change"] is None


# --- 再現できる学習 ---


def test_training_is_reproducible_and_records_provenance(tmp_path: Path, raw_dir: Path):
    write_manifest(raw_dir / "s1")
    params = {"epochs": 2, "batch_size": 4, "seed": 3}

    first = train(raw_dir, tmp_path / "m1", params)
    second = train(raw_dir, tmp_path / "m2", params)

    assert first["validation_metrics"] == second["validation_metrics"]
    assert first["best_val_loss"] == second["best_val_loss"]
    assert first["data"]["fingerprint"] == second["data"]["fingerprint"]

    assert first["split"]["val_sessions"] == ["s3"]
    assert first["preprocessing"]["resize_hw"] == [120, 160]
    sessions = {s["session_id"]: s for s in first["data"]["sessions"]}
    assert sessions["s1"]["manifest_ok"] is True
    assert sessions["s2"]["manifest_hash"] is None
    assert first["environment"]["torch"] == torch.__version__
    assert "commit" in first["code"]
    assert (tmp_path / "m1" / "model.onnx").exists()

    saved = json.loads((tmp_path / "m1" / "metadata.json").read_text(encoding="utf-8"))
    assert saved["data"]["fingerprint"] == first["data"]["fingerprint"]


def test_different_seed_changes_the_result(tmp_path: Path, raw_dir: Path):
    a = train(raw_dir, tmp_path / "a", {"epochs": 2, "batch_size": 4, "seed": 1})
    b = train(raw_dir, tmp_path / "b", {"epochs": 2, "batch_size": 4, "seed": 2})
    assert a["best_val_loss"] != b["best_val_loss"]


def test_cross_validation_holds_out_each_session(raw_dir: Path):
    result = cross_validate(raw_dir, {"epochs": 1, "batch_size": 4}, log=lambda _m: None)

    assert [f["held_out"] for f in result["folds"]] == ["s1", "s2", "s3"]
    assert result["overall"]["samples"] == 10
    assert result["method"] == "leave_one_session_out"


def test_cross_validation_needs_two_sessions(raw_dir: Path):
    with pytest.raises(TrainingError):
        cross_validate(raw_dir / "s1", {"epochs": 1}, log=lambda _m: None)
