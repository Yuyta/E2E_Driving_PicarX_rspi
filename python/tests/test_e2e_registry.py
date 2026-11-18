"""モデルの一覧: オフライン評価がなければ、学習の時の検証の値を出す"""

import json
from pathlib import Path

from src.e2e.registry.compare import show_models


def write_model(models: Path, version: str, metadata: dict, evaluation=None) -> None:
    model_dir = models / version
    model_dir.mkdir(parents=True)
    metadata = {"model_version": version, **metadata}
    (model_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    if evaluation is not None:
        (model_dir / "eval_metrics.json").write_text(json.dumps(evaluation), encoding="utf-8")


def row(output: str, version: str) -> str:
    return next(line for line in output.splitlines() if line.startswith(version))


def test_models_show_the_validation_mae_without_an_offline_evaluation(tmp_path: Path, capsys):
    models = tmp_path / "models"
    validation = {"validation_metrics": {"steering_mae": 0.1328, "steering_mae_deg": 4.65}}
    write_model(models, "v006", validation)
    write_model(
        models,
        "v007",
        validation,
        {"steering_mae_normalized": 0.2, "steering_mae_degrees": 7.0},
    )
    write_model(models, "v008", {})

    show_models(str(models))
    output = capsys.readouterr().out
    assert "0.1328" in row(output, "v006") and "4.7" in row(output, "v006")
    assert "0.2000" in row(output, "v007")  # オフライン評価を優先する
    assert row(output, "v008").count("--") >= 2
