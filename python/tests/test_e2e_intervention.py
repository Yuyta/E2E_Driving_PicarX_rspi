"""介入の記録、パッドでの引き継ぎ、人の修正操作の保存、モデル別の介入率"""

import json
from pathlib import Path

import numpy as np

from src.e2e.evaluation.interventions import format_table, load_runs, summarize
from src.e2e.gamepad import PadInput
from src.e2e.intervention import (
    CorrectionRecorder,
    RunLog,
    Takeover,
    model_identity,
    pad_active,
)
from src.e2e.training.dataset import DrivingDataset

# --- モデルの版 ---


def write_metadata(model_dir: Path, version="v006", fingerprint="abc123") -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model_version": version,
                "data": {"fingerprint": fingerprint},
                "code": {"commit": "deadbeef"},
            }
        ),
        encoding="utf-8",
    )
    return model_dir / "model.onnx"


def test_model_identity_reads_the_metadata(tmp_path: Path):
    model = write_metadata(tmp_path / "v006")
    identity = model_identity(model)
    assert identity["model_version"] == "v006"
    assert identity["data_fingerprint"] == "abc123"
    assert identity["model_commit"] == "deadbeef"
    # metadata がなければフォルダ名だけ
    assert model_identity(tmp_path / "other" / "model.onnx")["model_version"] == "other"


# --- 走行の記録 ---


def test_run_log_counts_human_and_safety_interventions(tmp_path: Path):
    model = write_metadata(tmp_path / "models" / "v006")
    log = RunLog(tmp_path / "runs", model, {"speed": 50}, now=1000.0)
    log.add_time(30.0, human=False)
    log.add_time(5.0, human=True)
    log.intervention("obstacle_stop", 1031.0, "Pi reported an obstacle")
    log.intervention("human_takeover", 1032.0, "Gamepad input", takeover_id=1)
    log.intervention("manual_intervention", 1040.0, "Ctrl+C")
    log.close(now=1041.0)

    run = json.loads((log.dir / "run.json").read_text(encoding="utf-8"))
    assert run["model_version"] == "v006"
    assert run["data_fingerprint"] == "abc123"
    assert run["interventions"] == {"human": 2, "safety": 1}
    assert run["autonomous_s"] == 30.0
    assert run["human_s"] == 5.0
    assert run["ended"] is True
    assert run["settings"] == {"speed": 50}
    lines = (log.dir / "interventions.jsonl").read_text(encoding="utf-8").splitlines()
    kinds = [json.loads(line)["kind"] for line in lines]
    assert kinds == ["safety", "human", "human"]
    assert json.loads(lines[1])["takeover_id"] == 1


# --- 引き継ぎ ---


def test_pad_active():
    assert not pad_active(None)
    assert not pad_active(PadInput())
    assert not pad_active(PadInput(intent="left"))  # 意図だけでは運転しない
    assert pad_active(PadInput(drive=1))
    assert pad_active(PadInput(steer=-0.3))
    assert pad_active(PadInput(stop=True))


def test_takeover_starts_on_input_and_releases_after_quiet_time():
    takeover = Takeover(release_s=1.5)
    assert not takeover.update(PadInput(), 0.0).human
    started = takeover.update(PadInput(steer=0.5), 1.0)
    assert started.human and started.started
    assert takeover.update(None, 2.0).human  # パッドが一瞬読めなくても続ける
    assert takeover.update(PadInput(), 2.4).human
    ended = takeover.update(PadInput(), 2.5)
    assert not ended.human and ended.ended


def test_takeover_hands_back_on_y_and_waits_through_a_pause():
    takeover = Takeover(release_s=8.0)
    takeover.update(PadInput(drive=1), 0.0)
    assert takeover.update(PadInput(), 5.0).human  # 手を止めて考えていても続ける
    ended = takeover.update(PadInput(resume=True), 5.5)
    assert ended.ended and not ended.human
    assert not takeover.update(PadInput(resume=True), 5.6).human  # Y だけでは引き継がない


def test_takeover_emergency():
    takeover = Takeover()
    takeover.update(PadInput(drive=1), 0.0)
    decision = takeover.update(PadInput(emergency=True), 0.1)
    assert decision.emergency and decision.ended and not decision.human


# --- 修正操作の保存 ---


def test_corrections_are_readable_as_training_data(tmp_path: Path):
    recorder = CorrectionRecorder(
        tmp_path / "interventions", "20261114_120000", {"model_version": "v006"}
    )
    frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    recorder.record(frame, 1.0, PadInput(drive=1, steer=0.6), 50, 21, "right", 45.0, False, 1)
    recorder.record(
        frame, 1.1, PadInput(drive=1, steer=-0.4, intent="left"), 50, -14, "right", None, True, 1
    )
    recorder.record(frame, 1.2, PadInput(drive=0, steer=0.0), 0, 0, "right", 30.0, False, 1)

    dataset = DrivingDataset(recorder.session_dir)
    # 止まっているフレームは舵角の学習に使わない
    assert [s["steering"] for s in dataset.samples] == [0.6, -0.4]
    # L / R を押していなければ、引き継いだ時の自律の意図
    assert [s["intent"] for s in dataset.samples] == ["right", "left"]
    assert dataset.samples[0]["raw_record"]["model_version"] == "v006"
    assert dataset.samples[1]["blocked"] is False  # 反射なしは遠い


# --- 介入率の集計 ---


def write_run(runs: Path, run_id, version, autonomous_s, human, safety, fingerprint="f1"):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "model_version": version,
                "data_fingerprint": fingerprint,
                "autonomous_s": autonomous_s,
                "human_s": 0.0,
                "interventions": {"human": human, "safety": safety},
            }
        ),
        encoding="utf-8",
    )


def test_summary_per_ten_minutes_by_model(tmp_path: Path):
    runs = tmp_path / "runs"
    write_run(runs, "a", "v006", 300.0, 1, 2)
    write_run(runs, "b", "v006", 300.0, 2, 0)
    write_run(runs, "c", "v007", 600.0, 1, 1)
    write_run(runs, "d", "v008", 0.0, 0, 1)

    rows = summarize(load_runs(runs))
    assert [r["model_version"] for r in rows] == ["v007", "v006", "v008"]
    v006 = rows[1]
    assert (v006["runs"], v006["autonomous_min"], v006["human"]) == (2, 10.0, 3)
    assert v006["human_per_10min"] == 3.0
    assert v006["safety_per_10min"] == 2.0
    assert rows[2]["human_per_10min"] is None  # 自律の時間がない
    assert "v006" in format_table(rows)


def test_summary_separates_models_trained_on_different_data(tmp_path: Path):
    runs = tmp_path / "runs"
    write_run(runs, "a", "v006", 60.0, 0, 0, fingerprint="f1")
    write_run(runs, "b", "v006", 60.0, 0, 0, fingerprint="f2")
    assert len(summarize(load_runs(runs))) == 2
