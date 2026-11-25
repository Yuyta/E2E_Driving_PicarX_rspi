import json
from pathlib import Path

import pytest

from src.e2e.registry import lineage, stages


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def make_session(root: Path, session_id: str, manifest_hash: str = "m-" + "0" * 8) -> Path:
    session = root / "data" / "raw" / session_id
    (session / "images").mkdir(parents=True)
    lines = [
        {"timestamp": 0.0, "steering": 0.0},
        {"timestamp": 1.0, "steering": -0.5},
        {"timestamp": 2.0, "steering": 0.5},
        {"timestamp": 3.0, "steering": 0.0},
    ]
    (session / "labels.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8"
    )
    write_json(session / "manifest.json", {"hash": manifest_hash, "files": {}})
    write_json(session / "collector.json", {"collector": "auto_v3"})
    return session


def make_model(root: Path, version: str, sessions, val, fingerprint="fp1", manifest="m-00000000"):
    model = root / "models" / version
    write_json(
        model / "metadata.json",
        {
            "model_version": version,
            "created_at": f"2026-11-1{version[-1]}T10:00:00+0900",
            "split": {"train_sessions": sessions, "val_sessions": val},
            "params": {"epochs": 3},
            "history": [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.3}],
            "validation_metrics": {"steering_mae": 0.1, "steering_mae_deg": 3.5},
            "preprocessing": {"resize_hw": [120, 160]},
            "data": {
                "fingerprint": fingerprint,
                "sessions": [
                    {"session_id": s, "manifest_hash": manifest, "manifest_ok": True}
                    for s in sessions + val
                ],
            },
            "code": {"commit": "a" * 40, "dirty": False},
        },
    )
    (model / "model.onnx").write_bytes(b"onnx")
    return model


@pytest.fixture
def repo(tmp_path):
    make_session(tmp_path, "s1")
    make_session(tmp_path, "s2")
    make_session(tmp_path, "s3")
    (tmp_path / "data" / "raw" / "broken" / "images").mkdir(parents=True)
    make_model(tmp_path, "v001", ["s1"], ["s2"])
    make_model(tmp_path, "v002", ["s1", "s2"], ["s3"], fingerprint="fp2")
    write_json(
        tmp_path / "data" / "cv" / "cv_fp2.json",
        {"method": "leave_one_session_out", "data_fingerprint": "fp2", "overall": {}, "folds": []},
    )
    write_json(
        tmp_path / "data" / "runs" / "r1" / "run.json",
        {
            "run_id": "r1",
            "model_version": "v002",
            "data_fingerprint": "fp2",
            "where": "pi",
            "started_at": 1794833938.0,
            "autonomous_s": 300.0,
            "interventions": {"human": 2, "safety": 1},
        },
    )
    return tmp_path


def catalog(root: Path):
    return lineage.build_catalog(
        models_dir=root / "models",
        data_dirs=[root / "data" / "raw"],
        annotations_dir=root / "data" / "annotations",
        cv_dir=root / "data" / "cv",
        runs_dir=root / "data" / "runs",
        deployments_file=root / "deployments.jsonl",
        jobs_dir=root / "data" / "jobs",
    )


def test_links_sessions_models_cv_and_runs(repo):
    c = catalog(repo)
    models = {m["version"]: m for m in c["models"]}
    assert models["v002"]["cv"] == ["cv_fp2"]
    assert models["v001"]["cv"] == []
    assert models["v002"]["field"]["runs"] == 1
    assert models["v002"]["field"]["human_per_10min"] == 4.0
    sessions = {s["session_id"]: s for s in c["sessions"]}
    assert sessions["s3"]["used_by"] == ["v002"]
    assert sessions["s1"]["used_by"] == ["v001", "v002"]
    edges = {(e["from"], e["to"], e["kind"]) for e in c["graph"]["edges"]}
    assert ("session:s3", "model:v002", "val") in edges
    assert ("model:v002", "cv:cv_fp2", "same_data") in edges
    assert ("model:v002", "field:v002", "drove") in edges


def test_training_history_is_not_replaced_by_stage_history(repo):
    stages.set_stage(repo / "models", "v001", "approved", "good")
    m = next(m for m in catalog(repo)["models"] if m["version"] == "v001")
    assert m["history"] == [{"epoch": 1, "train_loss": 0.2, "val_loss": 0.3}]
    assert m["stage_history"][-1]["to"] == "approved"


def test_changed_data_after_training_is_an_error(repo):
    write_json(repo / "data" / "raw" / "s3" / "manifest.json", {"hash": "different", "files": {}})
    models = {m["version"]: m for m in catalog(repo)["models"]}
    assert models["v002"]["health"] == "error"
    assert any(c["key"] == "manifest" and "s3" in c["message"] for c in models["v002"]["checks"])
    assert models["v001"]["health"] == "ok"


def test_usable_needs_approval_and_no_errors(repo):
    c = catalog(repo)
    assert c["summary"]["latest"] == "v002"
    assert c["summary"]["latest_approved"] is None
    stages.set_stage(repo / "models", "v001", "approved", "ok")
    stages.set_stage(repo / "models", "v002", "approved", "ok")
    assert catalog(repo)["summary"]["latest_approved"] == "v002"
    (repo / "models" / "v002" / "model.onnx").unlink()
    c = catalog(repo)
    assert {m["version"]: m["usable"] for m in c["models"]} == {"v001": True, "v002": False}
    assert c["summary"]["latest_approved"] == "v001"


def test_incomplete_sessions_are_reported(repo):
    c = catalog(repo)
    assert [s["session_id"] for s in c["summary"]["incomplete_sessions"]] == ["broken"]
    assert "broken" not in {s["session_id"] for s in c["sessions"]}


def test_catalog_does_not_write_into_data(repo):
    before = sorted(p.relative_to(repo) for p in repo.rglob("*"))
    catalog(repo)
    assert sorted(p.relative_to(repo) for p in repo.rglob("*")) == before


def test_training_job_is_linked_to_model(repo):
    write_json(
        repo / "data" / "jobs" / "j1.json",
        {"id": "j1", "command": "python -m src.e2e.training.train", "produces": "models/v002"},
    )
    models = {m["version"]: m for m in catalog(repo)["models"]}
    assert models["v002"]["created_by"]["id"] == "j1"
    assert models["v001"]["created_by"] is None


def test_set_stage_validates_and_keeps_history(repo):
    with pytest.raises(stages.StageError):
        stages.set_stage(repo / "models", "v001", "production")
    with pytest.raises(stages.StageError):
        stages.set_stage(repo / "models", "v999", "approved")
    stages.set_stage(repo / "models", "v001", "approved", "first")
    result = stages.set_stage(repo / "models", "v001", "archived", "replaced by v002")
    assert [(h["from"], h["to"]) for h in result["history"]] == [
        ("candidate", "approved"),
        ("approved", "archived"),
    ]


def test_deployed_version_follows_push_and_rollback(tmp_path):
    log = tmp_path / "deployments.jsonl"
    stages.record_deployment("push", "v001", True, path=log)
    stages.record_deployment("push", "v002", True, path=log)
    stages.record_deployment("push", "v003", False, path=log)
    stages.record_deployment("install", "v004", True, path=log)
    assert stages.deployed_version(stages.load_deployments(log))["version"] == "v002"
    stages.record_deployment("rollback", None, True, path=log)
    assert stages.deployed_version(stages.load_deployments(log))["version"] == "v001"


def test_deploy_cli_records_without_host(tmp_path, monkeypatch):
    from src.e2e.deploy import pi

    class Done:
        returncode = 0

    monkeypatch.setattr(pi.subprocess, "run", lambda *a, **k: Done())
    assert pi.main(["--host", "pi@example", "rollback"]) == 0
    events = stages.load_deployments()
    assert events[-1]["action"] == "rollback" and events[-1]["ok"] is True
    assert "example" not in json.dumps(events)


def test_runs_keep_the_pi_server_build(tmp_path):
    """走行の記録の code.server（Pi の走行サーバーのビルド）を構成の画面で見せる"""
    import json as _json

    from src.e2e.registry import lineage as _lineage

    run = tmp_path / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "run.json").write_text(
        _json.dumps(
            {"run_id": "r1", "where": "pi", "code": {"server": "picar-x-rust 0.1.0 (abc)"}}
        ),
        encoding="utf-8",
    )
    runs = _lineage.collect_runs(tmp_path / "runs")
    assert runs[0]["server_build"] == "picar-x-rust 0.1.0 (abc)"
