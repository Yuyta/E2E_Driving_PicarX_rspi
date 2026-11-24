"""門番の試験（回帰テスト）: タグからの場面集め、採点、合否、比べる元、Pi へ送る前の確認"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.e2e.dataset.annotations import AnnotationStore
from src.e2e.deploy import pi
from src.e2e.evaluation import gate
from src.e2e.evaluation.gate import (
    LEFT,
    RIGHT,
    STRAIGHT,
    GateCase,
    GateCriteria,
    GateError,
    collect_cases,
    judge,
    passes,
    score,
)
from src.e2e.registry.stages import load_deployments, record_deployment, set_stage

V006 = Path(__file__).resolve().parents[1] / "models" / "v006"


def make_session(root: Path, name: str, frames: int, intents=None) -> Path:
    session = root / "raw" / name
    (session / "images").mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(frames):
            frame = f"frame_{i:06d}.jpg"
            shade = (i * 37) % 256
            Image.new("RGB", (160, 120), (shade, 80, 255 - shade)).save(session / "images" / frame)
            record = {"index": i, "frame": frame, "steering": 0.0}
            if intents:
                record["intent"] = intents[i]
            f.write(json.dumps(record) + "\n")
    return session


def tag(store: AnnotationStore, session: Path, frames, name: str) -> None:
    notes = store.session(session)
    notes.add_tag([f"frame_{i:06d}.jpg" for i in frames], name)
    notes.save()


def case(expect: str, i: int = 0) -> GateCase:
    return GateCase("s", f"frame_{i:06d}.jpg", expect)


# ----------------------------------------------------------------------------- 採点の決まり


def test_each_kind_of_scene_has_its_own_pass_rule():
    c = GateCriteria(turn_min_deg=5.0, straight_max_deg=4.0)
    assert passes(LEFT, -6.0, c) and not passes(LEFT, -4.0, c) and not passes(LEFT, 10.0, c)
    assert passes(RIGHT, 5.0, c) and not passes(RIGHT, 2.0, c)
    assert passes(STRAIGHT, -3.9, c) and not passes(STRAIGHT, 4.5, c)
    with pytest.raises(ValueError):
        passes("up", 0.0, c)


def test_criteria_reject_nonsense_values():
    with pytest.raises(ValueError):
        GateCriteria(turn_min_deg=0)
    with pytest.raises(ValueError):
        GateCriteria(max_drop=2.0)
    with pytest.raises(ValueError):
        GateCriteria(min_rate={"up": 0.5})


def test_score_counts_by_kind_and_keeps_every_frame():
    cases = [case(LEFT, 0), case(LEFT, 1), case(STRAIGHT, 2)]
    result = score(cases, [-20.0, 3.0, 1.0], GateCriteria())
    assert result["by_kind"][LEFT] == {"cases": 2, "passed": 1, "rate": 0.5}
    assert result["by_kind"][STRAIGHT] == {"cases": 1, "passed": 1, "rate": 1.0}
    assert result["by_kind"][RIGHT]["rate"] is None
    assert [f["passed"] for f in result["frames"]] == [True, False, True]


def test_too_few_scenes_do_not_pass():
    criteria = GateCriteria(min_cases=3)
    result = score([case(RIGHT), case(STRAIGHT, 1)], [20.0, 0.0], criteria)
    reasons = judge(result, criteria)
    assert len(reasons) == 2
    assert "避ける場面" in reasons[0] and "まっすぐの場面" in reasons[1]


def test_low_pass_rate_fails_and_a_good_model_passes():
    criteria = GateCriteria(min_cases=2)
    cases = [case(LEFT, 0), case(RIGHT, 1), case(STRAIGHT, 2), case(STRAIGHT, 3)]
    good = score(cases, [-20.0, 20.0, 0.0, 1.0], criteria)
    assert judge(good, criteria) == []
    bad = score(cases, [-20.0, 20.0, 0.0, 15.0], criteria)  # まっすぐの所で曲がった
    assert any("まっすぐ: 合格率 50%" in r for r in judge(bad, criteria))


def test_dropping_below_the_baseline_fails_even_above_the_minimum():
    criteria = GateCriteria(min_cases=1, min_rate={LEFT: 0.0, RIGHT: 0.0, STRAIGHT: 0.0})
    cases = [case(LEFT, i) for i in range(10)] + [case(STRAIGHT, 10)]
    baseline = score(cases, [-20.0] * 10 + [0.0], criteria)
    candidate = score(cases, [-20.0] * 8 + [0.0, 0.0] + [0.0], criteria)
    reasons = judge(candidate, criteria, baseline)
    assert reasons and "比べる元の 100%" in reasons[0]
    assert gate.regressions(candidate, baseline) == ["s/frame_000008.jpg", "s/frame_000009.jpg"]
    # 下がり方が許容の範囲なら合格
    small = score(cases, [-20.0] * 10 + [0.0], criteria)
    assert judge(small, criteria, baseline) == []


# ----------------------------------------------------------------------------- 場面を集める


def test_cases_come_from_the_data_viewer_tags(tmp_path: Path):
    session = make_session(tmp_path, "20261201_000000", 6, intents=["any"] * 5 + ["left"])
    store = AnnotationStore(tmp_path / "annotations")
    tag(store, session, [0, 1], "門番:右へ避ける")
    tag(store, session, [2], "gate:straight")
    tag(store, session, [3], "脚")  # 門番ではないタグは使わない
    tag(store, session, [5], "門番:まっすぐ")
    store.session(session).set_field(["frame_000002.jpg"], "intent", "right")
    store.save()

    cases, warnings = collect_cases([tmp_path / "raw"], AnnotationStore(tmp_path / "annotations"))
    assert warnings == []
    # 意図はタグで決める。記録の意図（left）と注釈の上書き（right）は使わない
    assert [(c.frame, c.expect, c.intent) for c in cases] == [
        ("frame_000000.jpg", RIGHT, "right"),
        ("frame_000001.jpg", RIGHT, "right"),
        ("frame_000002.jpg", STRAIGHT, "any"),
        ("frame_000005.jpg", STRAIGHT, "any"),
    ]
    assert gate.suite_fingerprint(cases) == gate.suite_fingerprint(list(reversed(cases)))
    assert gate.suite_fingerprint([]) is None


def test_conflicting_gate_tags_stop_the_test(tmp_path: Path):
    session = make_session(tmp_path, "s1", 2)
    store = AnnotationStore(tmp_path / "annotations")
    tag(store, session, [0], "門番:左へ避ける")
    tag(store, session, [0], "門番:右へ避ける")
    with pytest.raises(GateError, match="矛盾"):
        collect_cases([tmp_path / "raw"], AnnotationStore(tmp_path / "annotations"))


def test_missing_images_are_skipped_with_a_warning(tmp_path: Path):
    session = make_session(tmp_path, "s1", 2)
    store = AnnotationStore(tmp_path / "annotations")
    tag(store, session, [0, 1], "門番:まっすぐ")
    (session / "images" / "frame_000001.jpg").unlink()
    cases, warnings = collect_cases(
        [tmp_path / "raw", tmp_path / "missing"], AnnotationStore(tmp_path / "annotations")
    )
    assert [c.frame for c in cases] == ["frame_000000.jpg"]
    assert len(warnings) == 1 and "frame_000001.jpg" in warnings[0]


# ----------------------------------------------------------------------------- モデルを動かす


def fake_model(tmp_path: Path, name: str) -> Path:
    model = tmp_path / "models" / name
    model.mkdir(parents=True)
    shutil.copy(V006 / "metadata.json", model / "metadata.json")
    (model / "model.onnx").write_bytes(name.encode())
    return model


def constant_predictor(values):
    """版ごとに、全フレームへ同じ舵角（度）を返す"""

    def factory(model_dir: Path):
        return lambda images, intents: np.full(len(images), values[model_dir.name])

    return factory


def test_run_gate_compares_with_the_baseline(tmp_path: Path):
    session = make_session(tmp_path, "s1", 4)
    cases = [GateCase(str(session), f"frame_{i:06d}.jpg", RIGHT) for i in range(4)]
    criteria = GateCriteria(min_cases=0, min_rate={RIGHT: 0.5})
    new, old = fake_model(tmp_path, "v008"), fake_model(tmp_path, "v007")
    report = gate.run_gate(
        new, cases, old, criteria, predictor_factory=constant_predictor({"v008": 1.0, "v007": 25})
    )
    assert not report["passed"]
    assert report["baseline"]["version"] == "v007"
    assert len(report["baseline"]["regressed_frames"]) == 4
    assert report["model_sha256"] and report["suite"]["cases"] == 4
    lines = gate.summary_lines(report)
    assert lines[-1].startswith("  - ") and any("不合格" in line for line in lines)

    path = gate.write_report(new, report)
    assert path.name == "gate.json" and gate.report_is_current(gate.load_report(new), new)
    (new / "model.onnx").write_bytes(b"retrained")
    assert not gate.report_is_current(gate.load_report(new), new)


def test_v006_runs_on_recorded_images(tmp_path: Path):
    """Pi へ送る ONNX をそのまま動かし、学習と同じ前処理で舵角（度）を出す"""
    session = make_session(tmp_path, "s1", 3)
    cases = [GateCase(str(session), f"frame_{i:06d}.jpg", STRAIGHT) for i in range(3)]
    pre = gate.load_preprocessing(V006)
    degrees = gate.predict_cases(cases, gate.onnx_predictor(V006), pre)
    assert len(degrees) == 3
    assert all(-35.0 <= d <= 35.0 for d in degrees)
    image = gate.load_image(session / "images" / "frame_000000.jpg", pre)
    assert image.shape == (3, 120, 160) and image.dtype == np.float32


def test_a_model_without_metadata_cannot_be_tested(tmp_path: Path):
    with pytest.raises(GateError):
        gate.load_preprocessing(tmp_path)
    with pytest.raises(GateError):
        gate.onnx_predictor(tmp_path)


# ----------------------------------------------------------------------------- 比べる元


def test_baseline_is_the_model_on_the_pi_then_the_approved_one(tmp_path: Path):
    models = tmp_path / "models"
    for name in ("v006", "v007", "v008"):
        fake_model(tmp_path, name)
    assert gate.default_baseline(models, "v008") is None
    set_stage(models, "v006", "approved")
    assert gate.default_baseline(models, "v008") == "v006"
    assert gate.default_baseline(models, "v006") is None  # 自分とは比べない
    record_deployment("push", "v007", True)
    assert gate.default_baseline(models, "v008") == "v007"


# ----------------------------------------------------------------------------- Pi へ送る前


def failing_report(model_dir, *_args, **_kwargs):
    return {
        "model_version": Path(model_dir).name,
        "passed": False,
        "reasons": ["まっすぐ: 合格率 50%（1/2）が基準 90% より低い"],
        "suite": {"cases": 2, "warnings": []},
        "by_kind": {k: {"cases": 0, "passed": 0, "rate": None} for k in gate.EXPECTATIONS},
        "baseline": None,
    }


def test_push_refuses_a_model_that_fails_the_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(pi.gate, "gate_model", failing_report)
    monkeypatch.setattr(pi, "push", lambda *a, **k: calls.append(a) or 0)
    assert pi.main(["--host", "pi@rspi", "push", "--model_dir", str(V006)]) == 1
    assert calls == []
    event = load_deployments()[-1]
    assert (event["action"], event["version"], event["ok"]) == ("push", "v006", False)
    assert event["detail"].startswith("gate failed: まっすぐ")


def test_push_refuses_when_the_gate_cannot_run(monkeypatch):
    def broken(*_a, **_k):
        raise GateError("no scenes")

    monkeypatch.setattr(pi.gate, "gate_model", broken)
    monkeypatch.setattr(pi, "push", lambda *a, **k: pytest.fail("must not push"))
    assert pi.main(["--host", "pi@rspi", "push", "--model_dir", str(V006)]) == 1
    assert load_deployments()[-1]["detail"] == "gate error: no scenes"


def test_a_person_can_send_without_the_gate_by_giving_a_reason(monkeypatch):
    monkeypatch.setattr(pi.gate, "gate_model", lambda *a, **k: pytest.fail("must not test"))
    monkeypatch.setattr(pi, "push", lambda *a, **k: 0)
    argv = ["--host", "pi@rspi", "push", "--model_dir", str(V006)]
    assert pi.main(argv + ["--skip_gate", "  "]) == 1  # 理由のない省略は認めない
    assert pi.main(argv + ["--skip_gate", "門番の場面をまだ選んでいない"]) == 0
    event = load_deployments()[-1]
    assert event["ok"] and event["detail"] == "gate skipped: 門番の場面をまだ選んでいない"


def test_push_goes_ahead_when_the_gate_passes(monkeypatch):
    def passing(model_dir, *_a, **_k):
        report = failing_report(model_dir)
        report.update(passed=True, reasons=[])
        return report

    sent = []
    monkeypatch.setattr(pi.gate, "gate_model", passing)
    monkeypatch.setattr(pi, "push", lambda host, model_dir, *a, **k: sent.append(model_dir) or 0)
    assert pi.main(["--host", "pi@rspi", "push", "--model_dir", str(V006)]) == 0
    assert sent == [str(V006)]
    assert load_deployments()[-1]["detail"] == "gate passed (2 frames)"


def test_lineage_shows_the_gate_result_and_whether_it_is_current(tmp_path: Path):
    from src.e2e.registry import lineage

    model = fake_model(tmp_path, "v007")
    assert lineage.collect_models(tmp_path / "models")[0]["gate"] is None  # まだ試していない
    session = make_session(tmp_path, "s1", 2)
    cases = [GateCase(str(session), f"frame_{i:06d}.jpg", STRAIGHT) for i in range(2)]
    report = gate.run_gate(
        model,
        cases,
        criteria=GateCriteria(min_cases=0),
        predictor_factory=constant_predictor({"v007": 0.0}),
    )
    gate.write_report(model, report)
    found = lineage.collect_models(tmp_path / "models")[0]
    assert found["gate"]["passed"] and found["gate"]["current"] and found["gate"]["cases"] == 2
    checks = {c["key"]: c for c in lineage.model_checks(found, {})}
    assert checks["gate"]["level"] == "ok"

    (model / "model.onnx").write_bytes(b"retrained")
    found = lineage.collect_models(tmp_path / "models")[0]
    assert not found["gate"]["current"]
    checks = {c["key"]: c for c in lineage.model_checks(found, {})}
    assert checks["gate"]["level"] == "warn"
