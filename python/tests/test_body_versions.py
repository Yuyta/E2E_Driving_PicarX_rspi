"""車体の版（configs/body.toml）と、来歴へのつながり"""

import json
from datetime import datetime
from pathlib import Path

import pytest

from src.e2e.registry import body, lineage
from src.gui.workflows import PHASES, PHASES_BY_ID, WORKFLOWS

BODY = """
[[versions]]
id = "b1"
since = "2026-11-09T00:00"
title = "最初"
equipment = ["カメラ"]
[versions.calibration]
motor_balance = 0

[[versions]]
id = "b2"
since = "2026-11-11T00:00"
title = "補正"
equipment = ["カメラ"]
[versions.calibration]
motor_balance = 10
"""


def write_body(path: Path, text: str = BODY) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------- body.toml


def test_versions_are_sorted_and_found_by_time(tmp_path):
    versions = body.load_versions(write_body(tmp_path / "body.toml"))
    assert [v["id"] for v in versions] == ["b1", "b2"]
    assert body.version_at(versions, "20261110_120000")["id"] == "b1"
    assert body.version_at(versions, "20261112_235654")["id"] == "b2"
    assert body.version_at(versions, datetime(2026, 11, 1)) is None
    assert body.version_at(versions, "2026-11-12T10:00:00+0900")["id"] == "b2"
    assert body.version_at(versions, None) is None


def test_missing_file_is_empty_and_broken_file_is_an_error(tmp_path):
    assert body.load_versions(tmp_path / "none.toml") == []
    dup = BODY + '\n[[versions]]\nid = "b1"\nsince = "2026-11-20"\n'
    with pytest.raises(body.BodyError, match="重複"):
        body.load_versions(write_body(tmp_path / "dup.toml", dup))
    with pytest.raises(body.BodyError, match="since"):
        body.load_versions(write_body(tmp_path / "nosince.toml", '[[versions]]\nid = "x"\n'))


def test_fingerprint_changes_only_with_content(tmp_path):
    a = body.load_versions(write_body(tmp_path / "a.toml"))
    b = body.load_versions(
        write_body(tmp_path / "b.toml", BODY.replace("motor_balance = 10", "motor_balance = 12"))
    )
    assert a[0]["fingerprint"] == b[0]["fingerprint"]
    assert a[1]["fingerprint"] != b[1]["fingerprint"]


def test_add_appends_a_new_version_without_touching_old_ones(tmp_path):
    path = write_body(tmp_path / "body.toml")
    before = body.load_versions(path)
    added = body.add_version(
        path,
        "b3",
        'M6 の後 "ジャイロ"',
        "2026-11-25T14:00",
        ["カメラ", "ジャイロ"],
        {"motor_balance": 0},
    )
    after = body.load_versions(path)
    assert [v["id"] for v in after] == ["b1", "b2", "b3"]
    assert after[0]["fingerprint"] == before[0]["fingerprint"]
    assert added["title"] == 'M6 の後 "ジャイロ"' and added["calibration"] == {"motor_balance": 0}
    with pytest.raises(body.BodyError, match="すでに"):
        body.add_version(path, "b3", "again", "2026-11-26")
    with pytest.raises(body.BodyError, match="後に"):
        body.add_version(path, "b4", "past", "2026-11-10")


def test_cli_add_and_show(tmp_path, capsys):
    path = write_body(tmp_path / "body.toml")
    code = body.main(
        [
            "--file",
            str(path),
            "add",
            "--id",
            "b3",
            "--title",
            "t",
            "--since",
            "2026-12-01",
            "--equipment",
            "カメラ",
            "--set",
            "motor_balance=5",
            "--set",
            "note=abc",
        ]
    )
    assert code == 0
    assert body.load_versions(path)[-1]["calibration"] == {"motor_balance": 5, "note": "abc"}
    assert body.main(["--file", str(path), "show"]) == 0
    assert "b3" in capsys.readouterr().out
    assert body.main(["--file", str(path), "add", "--id", "b3", "--title", "t"]) == 1


def test_record_session_writes_the_current_version(tmp_path):
    path = write_body(tmp_path / "body.toml")
    session = tmp_path / "s"
    session.mkdir()
    record = body.record_session(session, path)
    assert record["id"] == "b2"
    assert body.read_session_record(session)["fingerprint"] == record["fingerprint"]
    # 版のファイルが壊れていても収集は止めない
    assert (
        body.record_session(session, write_body(tmp_path / "bad.toml", "[[versions]]\nid=")) is None
    )


def test_resolve_prefers_the_record_and_flags_rewritten_versions(tmp_path):
    versions = body.load_versions(write_body(tmp_path / "body.toml"))
    by_date = body.resolve(versions, None, "20261110_000000")
    assert by_date == {"id": "b1", "source": "date", "known": True, "changed": False}
    recorded = body.resolve(versions, {"id": "b2", "fingerprint": "old"}, "20261110_000000")
    assert recorded["id"] == "b2" and recorded["source"] == "recorded" and recorded["changed"]


def test_repository_body_file_is_valid():
    versions = body.load_versions()
    assert versions, "configs/body.toml に車体の版が1つ以上ある"
    assert body.current(versions) is not None


# ---------------------------------------------------------------- lineage


def make_session(root: Path, session_id: str, recorded=None) -> None:
    session = root / "data" / "raw" / session_id
    (session / "images").mkdir(parents=True)
    (session / "labels.jsonl").write_text(
        '{"timestamp": 0.0, "steering": 0.0}\n{"timestamp": 1.0, "steering": 0.2}\n',
        encoding="utf-8",
    )
    write_json(session / "manifest.json", {"hash": "m1", "files": {}})
    if recorded:
        write_json(session / "body.json", recorded)


def make_model(root: Path, version: str, sessions, body_record=None) -> None:
    meta = {
        "model_version": version,
        "created_at": "2026-11-15T10:00:00+0900",
        "split": {"train_sessions": sessions[:-1], "val_sessions": sessions[-1:]},
        "preprocessing": {"resize_hw": [120, 160]},
        "data": {
            "fingerprint": f"fp-{version}",
            "sessions": [
                {"session_id": s, "manifest_hash": "m1", "manifest_ok": True} for s in sessions
            ],
        },
        "code": {"commit": "a" * 40, "dirty": False},
    }
    if body_record:
        meta["body"] = body_record
    write_json(root / "models" / version / "metadata.json", meta)
    (root / "models" / version / "model.onnx").write_bytes(b"onnx")


def build(root: Path, body_file: Path):
    return lineage.build_catalog(
        models_dir=root / "models",
        data_dirs=[root / "data" / "raw"],
        annotations_dir=root / "data" / "annotations",
        cv_dir=root / "data" / "cv",
        runs_dir=root / "data" / "runs",
        deployments_file=root / "data" / "deployments.jsonl",
        jobs_dir=root / "data" / "jobs",
        body_file=body_file,
    )


def body_messages(model):
    return [(c["level"], c["message"]) for c in model["checks"] if c["key"] == "body"]


def test_catalog_links_bodies_to_sessions_models_and_runs(tmp_path):
    body_file = write_body(tmp_path / "configs" / "body.toml")
    make_session(tmp_path, "20261110_100000")  # b1（日時から）
    make_session(tmp_path, "20261112_100000")  # b2（日時から）
    make_session(tmp_path, "20261112_110000")
    make_model(tmp_path, "v001", ["20261110_100000", "20261112_100000"])
    make_model(tmp_path, "v002", ["20261112_100000", "20261112_110000"])
    write_json(
        tmp_path / "data" / "runs" / "r1" / "run.json",
        {
            "run_id": "r1",
            "model_version": "v001",
            "where": "pi",
            "started_at": datetime(2026, 11, 13, 9).timestamp(),
            "autonomous_s": 60.0,
        },
    )
    c = build(tmp_path, body_file)
    sessions = {s["session_id"]: s for s in c["sessions"]}
    assert sessions["20261110_100000"]["body_version"] == "b1"
    assert sessions["20261112_100000"]["body_source"] == "date"
    models = {m["version"]: m for m in c["models"]}
    assert models["v001"]["body_versions"] == ["b1", "b2"]
    assert any("混ぜて" in msg for _, msg in body_messages(models["v001"]))
    assert ("ok", "車体の版 b2（2 セッション） のデータで学習") in body_messages(models["v002"])
    # v001 は b1 と b2 で学習し、b2 の時期に走った（学習データに含まれる版なので食い違いではない）
    assert models["v001"]["field"]["body_mismatch"] == 0
    assert c["runs"][0]["body_version"] == "b2"
    assert "started_ts" not in c["runs"][0]
    bodies = {b["id"]: b for b in c["bodies"]}
    assert bodies["b2"]["current"] and bodies["b2"]["models"] == ["v001", "v002"]
    assert bodies["b1"]["sessions"] == ["20261110_100000"]
    assert c["summary"]["body_current"] == "b2"
    assert "since_dt" not in bodies["b1"]
    json.dumps(c)  # 画面へそのまま送れる


def test_models_trained_on_an_older_body_are_flagged(tmp_path):
    body_file = write_body(tmp_path / "configs" / "body.toml")
    make_session(tmp_path, "20261110_100000")
    make_session(tmp_path, "20261110_110000")
    make_model(tmp_path, "v001", ["20261110_100000", "20261110_110000"])
    model = build(tmp_path, body_file)["models"][0]
    assert any(level == "warn" and "今の車体（b2）" in msg for level, msg in body_messages(model))
    assert model["health"] == "warn"


def test_recorded_body_wins_and_rewrites_are_flagged(tmp_path):
    body_file = write_body(tmp_path / "configs" / "body.toml")
    make_session(tmp_path, "20261112_100000", {"id": "b2", "fingerprint": "stale"})
    make_session(tmp_path, "20261112_110000")
    make_model(
        tmp_path,
        "v001",
        ["20261112_100000", "20261112_110000"],
        {
            "sessions": {"20261112_100000": "b2", "20261112_110000": "b2"},
            "versions": {"b2": "stale"},
        },
    )
    c = build(tmp_path, body_file)
    session = next(s for s in c["sessions"] if s["session_id"] == "20261112_100000")
    assert session["body_source"] == "recorded" and session["body_changed"]
    model = c["models"][0]
    assert model["body_source"] == "metadata"
    assert any("書き換えられた" in msg for _, msg in body_messages(model))


def test_broken_body_file_does_not_break_the_catalog(tmp_path):
    make_session(tmp_path, "20261112_100000")
    c = build(tmp_path, write_body(tmp_path / "configs" / "body.toml", "[[versions]]\nid = 1\n"))
    assert c["summary"]["body_error"] and c["bodies"] == []
    assert c["sessions"][0]["body_version"] is None


def test_without_body_versions_nothing_is_flagged(tmp_path):
    make_session(tmp_path, "20261112_100000")
    make_session(tmp_path, "20261112_110000")
    make_model(tmp_path, "v001", ["20261112_100000", "20261112_110000"])
    model = build(tmp_path, tmp_path / "configs" / "none.toml")["models"][0]
    assert body_messages(model) == []


# ---------------------------------------------------------------- phases


def test_every_workflow_belongs_to_a_phase_and_phases_form_a_loop():
    assert all(w.phase in PHASES_BY_ID for w in WORKFLOWS)
    numbers = [p.number for p in PHASES if p.number is not None]
    assert numbers == list(range(len(numbers)))
    used = {w.phase for w in WORKFLOWS}
    assert all(p.id in used for p in PHASES if p.number is not None)


def test_manual_uses_the_same_phase_colors():
    manual = (Path(__file__).resolve().parents[1] / "src/gui/web/static/manual.html").read_text(
        encoding="utf-8"
    )
    for phase in PHASES:
        assert f"--ph: {phase.color}" in manual, phase.id


def test_every_phase_has_an_icon_for_both_guis():
    assert all(p.icon.startswith("<") for p in PHASES)
