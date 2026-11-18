"""2つの交差検証の結果の比較と、採用の判定"""

import copy
import json
from pathlib import Path

from src.e2e.evaluation.compare_cv import ADOPT, KEEP, NOT_COMPARABLE, compare, main


def cv_result(mae=0.16, folds=(0.15, 0.17), flip=False, fingerprint="f1"):
    return {
        "params": {"epochs": 15, "flip_augment": flip},
        "data_fingerprint": fingerprint,
        "code": {"commit": "abc", "dirty": False},
        "overall": {
            "samples": 200,
            "steering_mae": mae,
            "steering_mae_deg": round(mae * 35, 2),
            "steering_change": 0.03,
            "steering_mae_by_intent": {"any": 0.003, "left": 0.28, "right": 0.32},
            "steering_mae_by_direction": {"left": 0.26, "straight": 0.07, "right": 0.43},
        },
        "baselines": {"intent_mean": {"steering_mae": 0.19}},
        "folds": [
            {
                "held_out": f"s{i}",
                "samples": 100,
                "steering_mae": m,
                "steering_mae_deg": round(m * 35, 2),
            }
            for i, m in enumerate(folds)
        ],
    }


def test_a_better_candidate_is_adopted():
    report = compare(cv_result(0.16), cv_result(0.15, folds=(0.14, 0.18), flip=True))
    assert report["verdict"] == ADOPT
    assert report["changed_params"] == {"flip_augment": [False, True]}
    assert report["overall"]["steering_mae"]["diff"] == -0.01
    assert report["candidate_better_folds"] == 1
    assert report["by_intent"]["right"]["diff"] == 0.0


def test_a_worse_candidate_keeps_the_current_settings_unless_within_tolerance():
    current, candidate = cv_result(0.159), cv_result(0.1598)
    assert compare(current, candidate)["verdict"] == KEEP
    assert compare(current, candidate, tolerance=0.001)["verdict"] == ADOPT


def test_a_candidate_must_beat_the_intent_mean_baseline():
    report = compare(cv_result(0.25), cv_result(0.2))
    assert report["verdict"] == KEEP
    assert "intent_mean" in report["reasons"][0]


def test_results_with_different_folds_are_not_compared():
    current = cv_result()
    other_folds = copy.deepcopy(current)
    other_folds["folds"][0]["held_out"] = "s9"
    report = compare(current, other_folds)
    assert report["verdict"] == NOT_COMPARABLE
    assert report["folds"] == []
    other_samples = copy.deepcopy(current)
    other_samples["folds"][1]["samples"] = 90
    assert compare(current, other_samples)["verdict"] == NOT_COMPARABLE


def test_a_different_fingerprint_is_only_a_note():
    """曲がりの意図の付け直しなど、設定でも学習用のフレームの指紋は変わる"""
    report = compare(cv_result(0.16), cv_result(0.15, fingerprint="f2"))
    assert report["verdict"] == ADOPT
    assert "fingerprints differ" in report["notes"][0]
    assert compare(cv_result(), cv_result())["notes"] == []


def test_cli_saves_the_comparison(tmp_path: Path, capsys):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(cv_result(0.16)), encoding="utf-8")
    b.write_text(json.dumps(cv_result(0.15, flip=True)), encoding="utf-8")
    out = tmp_path / "cmp" / "a_vs_b.json"
    assert main(["--current", str(a), "--candidate", str(b), "--output", str(out)]) == 0
    assert "Verdict: adopt" in capsys.readouterr().out
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["verdict"] == ADOPT
    assert saved["files"]["current"] == str(a)

    c = tmp_path / "c.json"
    c.write_text(json.dumps(cv_result(folds=(0.15,))), encoding="utf-8")
    assert main(["--current", str(a), "--candidate", str(c)]) == 2
