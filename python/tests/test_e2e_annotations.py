"""手作業のラベル（data/annotations）と、データビューアの表と統計、学習での使われ方"""

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from src.e2e.dataset.annotations import (
    AnnotationError,
    AnnotationStore,
    SessionAnnotations,
    default_annotations_dir,
)
from src.e2e.dataset.frame_table import (
    EXCLUDED_BY_HAND,
    EXCLUDED_STOPPED,
    USED,
    FrameTable,
    Settings,
    count_by,
    crosstab,
    histogram,
    summary,
)
from src.e2e.training.dataset import DrivingDataset
from src.e2e.training.train import annotation_record


def make_session(data_root: Path, name: str, records) -> Path:
    session = data_root / "raw" / name
    images = session / "images"
    images.mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            frame = f"frame_{i:06d}.jpg"
            Image.new("RGB", (16, 12), (i * 10 % 255, 0, 0)).save(images / frame)
            f.write(json.dumps({"index": i, "frame": frame, "timestamp": 100.0 + i, **record}))
            f.write("\n")
    return session


def moving(steering=0.0, distance=None, intent="any", **extra):
    return {
        "steering": steering,
        "throttle": 0.5,
        "intent": intent,
        "distance_cm": distance,
        "distance_far": distance is None,
        "command": {"speed": 50, "steering_deg": steering * 35},
        **extra,
    }


def stopped():
    return {"steering": 0.0, "throttle": 0.0, "command": {"speed": 0}}


@pytest.fixture
def session(tmp_path):
    return make_session(
        tmp_path / "data",
        "20261116_120000",
        [moving(0.0, 100), moving(0.5, 30), moving(-1.0, 80), stopped(), moving(0.0)],
    )


# ---- 注釈の保存 ----


def test_annotations_live_outside_the_session(session, tmp_path):
    assert default_annotations_dir(session) == (tmp_path / "data" / "annotations").resolve()
    store = AnnotationStore()
    notes = store.session(session)
    notes.add_tag(["frame_000001.jpg"], "脚")
    store.save()
    assert notes.path == (tmp_path / "data" / "annotations" / "20261116_120000.json").resolve()
    # 元のセッションには何も増えない（マニフェストが変わらない）
    assert sorted(p.name for p in session.iterdir()) == ["images", "labels.jsonl"]


def test_round_trip_and_clearing(tmp_path):
    path = tmp_path / "a.json"
    notes = SessionAnnotations("s", path)
    assert notes.add_tag(["f1", "f2"], "脚") == 2
    assert notes.add_tag(["f1"], "脚") == 0  # 同じタグは重ねない
    notes.set_field(["f1"], "exclude", True)
    notes.set_field(["f2"], "blocked", None)  # 「不明にする」という上書き
    notes.set_field(["f2"], "intent", "left")
    notes.save()

    loaded = SessionAnnotations("s", path)
    assert loaded.get("f1") == {"tags": ["脚"], "exclude": True}
    assert loaded.get("f2") == {"tags": ["脚"], "blocked": None, "intent": "left"}
    assert loaded.content_hash() == notes.content_hash()

    loaded.remove_tag(["f1", "f2"], "脚")
    loaded.set_field(["f1"], "exclude", False)
    loaded.clear_field(["f2"], "blocked")
    loaded.clear_field(["f2"], "intent")
    assert loaded.frames == {}
    assert loaded.content_hash() is None
    loaded.save()
    assert json.loads(path.read_text(encoding="utf-8"))["frames"] == {}


def test_bad_values_are_refused(tmp_path):
    notes = SessionAnnotations("s", tmp_path / "a.json")
    with pytest.raises(AnnotationError):
        notes.set_field(["f"], "intent", "up")
    with pytest.raises(AnnotationError):
        notes.set_field(["f"], "blocked", "yes")
    with pytest.raises(AnnotationError):
        notes.set_field(["f"], "steering", 0.3)
    with pytest.raises(AnnotationError):
        notes.add_tag(["f"], "  ")


def test_nothing_is_written_without_annotations(session, tmp_path):
    store = AnnotationStore()
    store.session(session)
    store.save()
    assert not (tmp_path / "data" / "annotations").exists()


def test_failed_save_keeps_the_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "a.json"
    notes = SessionAnnotations("s", path)
    notes.add_tag(["f1"], "脚")
    notes.save()
    before = path.read_text(encoding="utf-8")

    notes.add_tag(["f2"], "布団")

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken_replace)
    with pytest.raises(OSError):
        notes.save()
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.json"]  # 一時ファイルは残らない


# ---- 学習での使われ方 ----


def test_training_uses_the_annotations(session):
    plain = DrivingDataset(session, annotations=False)
    assert len(plain.samples) == 4  # 止まっている1枚を除く

    store = AnnotationStore()
    notes = store.session(session)
    notes.set_field(["frame_000000.jpg"], "exclude", True)
    notes.set_field(["frame_000002.jpg"], "intent", "left")
    notes.set_field(["frame_000004.jpg"], "blocked", True)
    notes.add_tag(["frame_000002.jpg"], "脚")
    store.save()

    dataset = DrivingDataset(session, keep_for_blocked=True)
    by_frame = {s["image_path"].name: s for s in dataset.samples}
    assert "frame_000000.jpg" not in by_frame  # 除外は塞がっているかの学習にも使わない
    assert by_frame["frame_000002.jpg"]["intent"] == "left"
    assert by_frame["frame_000002.jpg"]["intent_recorded"] == "any"
    assert by_frame["frame_000002.jpg"]["tags"] == ["脚"]
    assert by_frame["frame_000004.jpg"]["blocked"] is True
    assert by_frame["frame_000004.jpg"]["blocked_by_stuck"] is False

    record = annotation_record(dataset)
    assert record["excluded"] == 1
    assert record["intent_overridden"] == 1
    assert record["blocked_overridden"] == 1
    assert record["hashes"] == {session.name: notes.content_hash()}

    # 使わない設定なら元のまま
    off = DrivingDataset(session, annotations=False)
    assert [s["image_path"].name for s in off.samples] == [
        s["image_path"].name for s in plain.samples
    ]


# ---- 表と統計 ----


def test_frame_table_rows(session):
    table = FrameTable([session.parent], Settings(turn_intent=False))
    assert len(table) == 5
    first = table.rows[0]
    assert first["session"] == "20261116_120000"
    assert first["source"] == "raw"
    assert first["time_s"] == 0.0
    assert first["direction"] == "まっすぐ"
    assert first["blocked"] == "空き"
    assert table.rows[1]["blocked"] == "塞がり"
    assert table.rows[2]["steering_deg"] == -35.0
    assert table.rows[3]["training"] == EXCLUDED_STOPPED
    assert table.image_path(2).name == "frame_000002.jpg"

    notes, frames = table.frames([0, 2])[0]
    notes.set_field(frames, "exclude", True)
    notes.add_tag(frames, "ぶれ")
    table.refresh([0, 2])
    assert table.rows[0]["training"] == EXCLUDED_BY_HAND
    assert table.rows[2]["tags"] == "ぶれ"
    assert table.rows[2]["overrides"] == "exclude"


def test_statistics():
    rows = [
        {"training": USED, "tags": "脚,布団", "steering_deg": -35.0, "source": "raw"},
        {"training": USED, "tags": "", "steering_deg": 0.0, "source": "raw"},
        {"training": EXCLUDED_STOPPED, "tags": "脚", "steering_deg": 35.0, "source": "x"},
        {"training": USED, "tags": "", "steering_deg": None, "source": "x"},
    ]
    assert count_by(rows, "training") == [(USED, 3), (EXCLUDED_STOPPED, 1)]
    assert dict(count_by(rows, "tags")) == {"脚": 2, "布団": 1, "(なし)": 2}

    bins, missing = histogram(rows, "steering_deg", bins=2)
    assert missing == 1
    assert [n for _, _, n in bins] == [1, 2]

    assert summary(rows, "steering_deg") == {
        "count": 3,
        "min": -35.0,
        "median": 0.0,
        "mean": 0.0,
        "max": 35.0,
    }
    row_labels, col_labels, counts = crosstab(rows, "source", "training")
    assert counts[("raw", USED)] == 2
    assert counts[("x", EXCLUDED_STOPPED)] == 1
    assert set(row_labels) == {"raw", "x"}
