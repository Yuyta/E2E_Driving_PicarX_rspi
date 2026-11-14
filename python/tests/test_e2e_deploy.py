"""Pi へ送る走行モデルのバンドルと、送信、戻し、走行の記録の取り込み"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.e2e.deploy import pi
from src.e2e.deploy.bundle import BundleError, build_bundle, sha256_file

V006 = Path(__file__).resolve().parents[1] / "models" / "v006"


def test_bundle_has_the_manifest_the_pi_checks(tmp_path: Path):
    bundle = build_bundle(V006, tmp_path)
    assert bundle.name == "v006"
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == 1
    assert set(manifest["files"]) == {"model.onnx", "model.onnx.data"}
    for name, digest in manifest["files"].items():
        assert sha256_file(bundle / name) == digest
    assert manifest["input"]["width"] == 160 and manifest["input"]["height"] == 120
    assert manifest["input"]["color"] == "RGB" and manifest["input"]["layout"] == "NCHW"
    assert manifest["intents"] == ["any", "left", "right"]
    assert manifest["outputs"] == ["steering", "blocked"]
    assert manifest["blocked_trained"] is False
    assert manifest["steering_scale_deg"] == 35.0
    assert manifest["data_fingerprint"].startswith("f1eaaabf")


def test_a_model_without_metadata_is_not_bundled(tmp_path: Path):
    model = tmp_path / "v999"
    model.mkdir()
    for name in ("model.onnx", "model.onnx.data"):
        shutil.copy(V006 / name, model / name)
    with pytest.raises(BundleError, match="metadata"):
        build_bundle(model, tmp_path / "out")
    with pytest.raises(BundleError):
        build_bundle(tmp_path / "missing", tmp_path / "out")


class Recorder:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def __call__(self, cmd, **_kwargs):
        self.calls.append(cmd)
        code = 1 if self.fail_at is not None and len(self.calls) == self.fail_at else 0
        return SimpleNamespace(returncode=code)


def test_push_uploads_then_installs_and_activates_on_the_pi():
    run = Recorder()
    assert pi.push("pi@rspi", V006, "~/picar-x-rust", run=run) == 0
    assert run.calls[0][:2] == ["ssh", "pi@rspi"]
    assert run.calls[1][0] == "scp" and run.calls[1][-1] == "pi@rspi:/tmp/picarx-model-upload/v006"
    install = run.calls[2][2]
    assert "~/picar-x-rust model install /tmp/picarx-model-upload/v006 --activate" in install
    assert "rm -rf" in install  # 送ったフォルダは片付ける


def test_push_stops_when_the_upload_fails():
    run = Recorder(fail_at=2)
    assert pi.push("pi@rspi", V006, "~/picar-x-rust", run=run) == 1
    assert len(run.calls) == 2  # インストールしない


def test_rollback_list_and_pull_runs(tmp_path: Path, monkeypatch):
    run = Recorder()
    monkeypatch.setattr(pi.subprocess, "run", run)
    monkeypatch.setenv(pi.HOST_ENV, "pi@rspi")
    assert pi.main(["rollback"]) == 0
    assert run.calls[-1] == ["ssh", "pi@rspi", "~/picar-x-rust model rollback"]
    assert pi.main(["list"]) == 0
    assert run.calls[-1][-1].endswith("model list")
    assert pi.main(["pull-runs", "--runs_dir", str(tmp_path / "runs")]) == 0
    assert run.calls[-1] == ["scp", "-r", "pi@rspi:~/picarx/runs/*", str(tmp_path / "runs")]


def test_host_is_required(monkeypatch):
    monkeypatch.delenv(pi.HOST_ENV, raising=False)
    with pytest.raises(SystemExit):
        pi.main(["list"])
