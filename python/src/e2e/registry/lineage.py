"""データとモデルの来歴（トレーサビリティ）

走行データのセッション → 手作業のラベル → 学習したモデル → 交差検証 → Pi への配置 → 走行の記録
を、ファイルに残っている来歴だけからつなぐ。読むだけで、データのフォルダには何も書かない。

つなぎ方:
- セッション → モデル: `metadata.json` の `split.train_sessions` / `val_sessions`
- ラベル → モデル: `data.annotations.hashes`（セッションごとのラベルの中身のハッシュ）
- 交差検証 ↔ モデル: 学習データの中身のハッシュ（`data_fingerprint`）が同じもの
- モデル → 走行: `run.json` の `model_version`（データのハッシュが違えば印を付ける）
- 車体の版 → セッション・走行: セッションの `body.json`、なければ日時と `configs/body.toml`

使い方（確認用の CLI）:
    uv run python -m src.e2e.registry.lineage --summary
"""

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from ..dataset.manifest import load_manifest, verify_session
    from . import body, stages
except ImportError:
    from src.e2e.dataset.manifest import load_manifest, verify_session
    from src.e2e.registry import body, stages


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# マニフェストは全ファイルの一覧で大きい（1セッション 2,000 件以上）。画面を開くたびに読み直さない
# よう、更新の時刻と大きさが同じなら前の結果を使う
_MANIFEST_CACHE: Dict[str, Any] = {}


def _manifest_summary(session_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(session_dir) / "manifest.json"
    try:
        stat = path.stat()
    except OSError:
        return None
    key = str(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _MANIFEST_CACHE.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    manifest = load_manifest(session_dir)
    summary = (
        {"hash": manifest.get("hash"), "created_at": manifest.get("created_at")}
        if manifest
        else None
    )
    _MANIFEST_CACHE[key] = (stamp, summary)
    return summary


def _iso(timestamp: Optional[float]) -> Optional[str]:
    if not timestamp:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(timestamp))


def _version_key(name: str):
    match = re.search(r"(\d+)$", name)
    return (0, int(match.group(1)), name) if match else (1, 0, name)


def _short_hash(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- sessions


def session_stats(session_dir: Path) -> Optional[Dict[str, Any]]:
    """セッションの集計。保存済みの metadata.json があれば使い、なければ labels.jsonl から数える"""
    labels = session_dir / "labels.jsonl"
    meta = _read_json(session_dir / "metadata.json") or {}
    if "total_frames" not in meta:
        if not labels.is_file():
            return None
        steerings, timestamps = [], []
        with open(labels, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                steerings.append(float(record.get("steering", 0.0)))
                if "timestamp" in record:
                    timestamps.append(float(record["timestamp"]))
        if not steerings:
            return None
        total = len(steerings)
        duration = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0
        meta = {
            "total_frames": total,
            "duration_sec": round(duration, 2),
            "fps": round(total / duration, 1) if duration > 0 else 0.0,
            "distribution": {
                "straight_ratio": round(sum(abs(s) <= 0.15 for s in steerings) / total, 3),
                "left_ratio": round(sum(s < -0.15 for s in steerings) / total, 3),
                "right_ratio": round(sum(s > 0.15 for s in steerings) / total, 3),
            },
        }
    return meta


def collect_sessions(data_dirs: Iterable[Path], annotations_dir: Optional[Path] = None):
    sessions = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue
        for path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            stats = session_stats(path)
            if stats is None:
                continue
            manifest = _manifest_summary(path)
            collector = _read_json(path / "collector.json") or {}
            annotation = (annotations_dir / f"{path.name}.json") if annotations_dir else None
            sessions.append(
                {
                    "session_id": path.name,
                    "source": data_dir.name,
                    "path": str(path),
                    "frames": stats.get("total_frames", 0),
                    "duration_sec": stats.get("duration_sec", 0.0),
                    "fps": stats.get("fps", 0.0),
                    "distribution": stats.get("distribution", {}),
                    "collector": collector.get("collector", "manual" if collector == {} else None),
                    "is_cleaned": bool(stats.get("is_cleaned")),
                    "manifest_hash": manifest.get("hash") if manifest else None,
                    "manifest_created_at": manifest.get("created_at") if manifest else None,
                    "annotation_hash": _short_hash(annotation) if annotation else None,
                    "body_recorded": body.read_session_record(path),
                }
            )
    return sessions


def incomplete_sessions(data_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """labels.jsonl のないセッション（記録が途中で止まったもの）。学習には使われない"""
    found = []
    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        if not data_dir.is_dir():
            continue
        for path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            labels = path / "labels.jsonl"
            if labels.is_file() and labels.stat().st_size > 0:
                continue
            images = path / "images"
            count = sum(1 for _ in images.iterdir()) if images.is_dir() else 0
            found.append({"session_id": path.name, "source": data_dir.name, "images": count})
    return found


# --------------------------------------------------------------------------- models


def _stage_fields(stage: Dict[str, Any]) -> Dict[str, Any]:
    # 学習の履歴（history）と名前がぶつからないようにする
    return {
        "stage": stage["stage"],
        "note": stage["note"],
        "stage_updated_at": stage["updated_at"],
        "stage_history": stage["history"],
    }


def _gate_summary(report: Optional[Dict[str, Any]], onnx_sha256: Optional[str]):
    """門番の試験の結果（`gate.json`、`evaluation/gate.py`）の要約。試していなければ None"""
    if not report:
        return None
    baseline = report.get("baseline") or {}
    return {
        "passed": bool(report.get("passed")),
        # 試した後に model.onnx が変わっていれば、その結果は今のモデルのものではない
        "current": bool(onnx_sha256) and report.get("model_sha256") == onnx_sha256,
        "tested_at": report.get("tested_at"),
        "reasons": list(report.get("reasons") or []),
        "cases": (report.get("suite") or {}).get("cases"),
        "by_kind": report.get("by_kind") or {},
        "baseline_version": baseline.get("version"),
        "baseline_by_kind": baseline.get("by_kind") or {},
        "regressed_frames": len(baseline.get("regressed_frames") or []),
    }


def collect_models(models_dir: Path) -> List[Dict[str, Any]]:
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        return []
    registry = stages.load_registry(models_dir)
    models = []
    for path in sorted(
        (p for p in models_dir.iterdir() if p.is_dir()), key=lambda p: _version_key(p.name)
    ):
        meta_path = path / "metadata.json"
        meta = _read_json(meta_path)
        if meta is None:
            continue
        version = meta.get("model_version") or path.name
        split = meta.get("split") or {}
        data = meta.get("data") or {}
        evaluation = _read_json(path / "eval_metrics.json")
        onnx_sha256 = _short_hash(path / "model.onnx")
        models.append(
            {
                "version": path.name,
                "declared_version": version,
                "path": str(path),
                "created_at": meta.get("created_at") or _iso(meta_path.stat().st_mtime),
                "created_at_source": "metadata" if meta.get("created_at") else "file_time",
                "artifacts": {
                    "metadata": True,
                    "onnx": (path / "model.onnx").is_file(),
                    "onnx_data": (path / "model.onnx.data").is_file(),
                    "weights": (path / "best_model.pth").is_file(),
                    "eval": evaluation is not None,
                },
                "onnx_sha256": onnx_sha256,
                "gate": _gate_summary(_read_json(path / "gate.json"), onnx_sha256),
                "train_sessions": list(split.get("train_sessions") or meta.get("sessions") or []),
                "val_sessions": list(split.get("val_sessions") or []),
                "train_samples": split.get("train_samples"),
                "val_samples": split.get("val_samples"),
                "total_samples": meta.get("total_samples"),
                "data_fingerprint": data.get("fingerprint"),
                "data_sessions": list(data.get("sessions") or []),
                "annotations": (data.get("annotations") or {}).get("hashes") or {},
                "data_filters": {
                    k: v
                    for k, v in data.items()
                    if k not in ("fingerprint", "sessions", "annotations")
                },
                "params": meta.get("params") or {},
                "best_val_loss": meta.get("best_val_loss"),
                "validation": meta.get("validation_metrics") or {},
                "evaluation": evaluation,
                "inference_ms_pc": meta.get("inference_ms_pc"),
                "blocked_trained": bool((meta.get("blocked_output") or {}).get("trained")),
                "code": meta.get("code") or {},
                "body_record": meta.get("body") or {},
                "environment": meta.get("environment") or {},
                "history": meta.get("history") or [],
                "preprocessing": meta.get("preprocessing") or {},
                **_stage_fields(stages.stage_of(registry, path.name)),
            }
        )
    return models


# --------------------------------------------------------------------------- checks


def model_checks(model: Dict[str, Any], sessions_by_id: Dict[str, Dict[str, Any]]):
    """来歴と成果物の確認。level は ok / warn / error"""
    checks = []

    def add(key, level, message):
        checks.append({"key": key, "level": level, "message": message})

    art = model["artifacts"]
    add("onnx", "ok" if art["onnx"] else "error", "ONNX あり" if art["onnx"] else "ONNX がない")
    if model["declared_version"] != model["version"]:
        add(
            "version",
            "warn",
            f"フォルダ名 {model['version']} と記録の版 {model['declared_version']} が違う",
        )
    if not model["preprocessing"].get("resize_hw"):
        add("preprocessing", "error", "前処理の値がない（Pi へ送れない）")
    if not model["data_fingerprint"]:
        add("fingerprint", "warn", "学習データのハッシュがない（M3 より前のモデル）")

    code = model["code"]
    if not code.get("commit"):
        add("commit", "warn", "学習したコードのコミットが記録されていない")
    elif code.get("dirty"):
        add("commit", "warn", f"未コミットの変更がある状態で学習した ({code['commit'][:7]}*)")
    else:
        add("commit", "ok", f"コミット {code['commit'][:7]} で学習")

    recorded = {s.get("session_id"): s for s in model["data_sessions"]}
    used = model["train_sessions"] + model["val_sessions"]
    missing = [s for s in used if s not in sessions_by_id]
    changed, unverified = [], []
    for session_id in used:
        record = recorded.get(session_id) or {}
        current = sessions_by_id.get(session_id)
        if current is None:
            continue
        if not record.get("manifest_hash"):
            unverified.append(session_id)
        elif current.get("manifest_hash") != record.get("manifest_hash"):
            changed.append(session_id)
        if record.get("manifest_ok") is False:
            changed.append(session_id)
    if missing:
        add("sessions", "warn", f"学習データがこの PC にない: {', '.join(missing)}")
    if changed:
        add("manifest", "error", f"学習後にデータが変わった: {', '.join(sorted(set(changed)))}")
    elif unverified:
        add("manifest", "warn", f"マニフェストのないセッション: {', '.join(unverified)}")
    elif used and not missing:
        add("manifest", "ok", "学習データは学習時のまま（マニフェストのハッシュが一致）")

    gate = model.get("gate")
    if gate is not None:
        # 試していないモデルには印を付けない（Pi へ送る時に deploy.pi が試す）
        if not gate["current"]:
            add("gate", "warn", "門番の試験の後に model.onnx が変わった（試し直す）")
        elif gate["passed"]:
            add("gate", "ok", f"門番の試験に合格（{gate['cases']} フレーム）")
        else:
            add("gate", "warn", "門番の試験に不合格: " + "; ".join(gate["reasons"][:2]))

    for session_id, recorded_hash in model["annotations"].items():
        current = sessions_by_id.get(session_id, {}).get("annotation_hash")
        if current and recorded_hash and current != recorded_hash:
            add("annotations", "warn", f"学習後にラベルが変わった: {session_id}")
    return checks


def health(checks: List[Dict[str, Any]]) -> str:
    levels = {c["level"] for c in checks}
    return "error" if "error" in levels else "warn" if "warn" in levels else "ok"


def verify_model_data(model: Dict[str, Any], data_dirs: Iterable[Path]) -> List[Dict[str, Any]]:
    """学習データのセッションを、マニフェストの全ファイルのハッシュで確かめ直す（時間がかかる）"""
    results = []
    roots = [Path(d) for d in data_dirs]
    for session_id in model["train_sessions"] + model["val_sessions"]:
        path = next((r / session_id for r in roots if (r / session_id).is_dir()), None)
        if path is None:
            results.append({"session_id": session_id, "ok": None, "problems": ["見つからない"]})
            continue
        result = verify_session(path)
        problems = [
            *(f"なし: {p}" for p in result.missing),
            *(f"変更: {p}" for p in result.changed),
            *(f"追加: {p}" for p in result.added),
        ]
        if not result.has_manifest:
            problems.insert(0, "マニフェストがない")
        results.append({"session_id": session_id, "ok": result.ok, "problems": problems[:20]})
    return results


# --------------------------------------------------------------------------- cv / runs


def collect_cv(cv_dir: Path) -> List[Dict[str, Any]]:
    results = []
    cv_dir = Path(cv_dir)
    if not cv_dir.is_dir():
        return results
    for path in sorted(cv_dir.glob("*.json")):
        data = _read_json(path)
        if not data:
            continue
        entry = {
            "name": path.stem,
            "path": str(path),
            "modified_at": _iso(path.stat().st_mtime),
            "kind": "comparison" if "verdict" in data else "cross_validation",
        }
        if entry["kind"] == "cross_validation":
            entry.update(
                {
                    "method": data.get("method"),
                    "data_fingerprint": data.get("data_fingerprint"),
                    "params": data.get("params") or {},
                    "overall": data.get("overall") or {},
                    "baselines": data.get("baselines") or {},
                    "folds": [
                        {
                            "held_out": f.get("held_out"),
                            "samples": f.get("samples"),
                            "steering_mae": f.get("steering_mae"),
                            "steering_mae_deg": f.get("steering_mae_deg"),
                        }
                        for f in data.get("folds") or []
                    ],
                    "code": data.get("code") or {},
                }
            )
        else:
            entry.update(
                {
                    "verdict": data.get("verdict"),
                    "reasons": data.get("reasons") or [],
                    "comparable": data.get("comparable"),
                    "changed_params": data.get("changed_params") or {},
                    "overall": data.get("overall") or {},
                    "files": data.get("files") or {},
                }
            )
        results.append(entry)
    return results


def collect_runs(runs_dir: Path) -> List[Dict[str, Any]]:
    runs = []
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return runs
    for path in sorted(runs_dir.glob("*/run.json")):
        data = _read_json(path)
        if not data:
            continue
        counts = data.get("interventions") or {}
        runs.append(
            {
                "run_id": data.get("run_id") or path.parent.name,
                "model_version": data.get("model_version"),
                "data_fingerprint": data.get("data_fingerprint"),
                "model_commit": data.get("model_commit"),
                # Pi の走行サーバーのビルド（例: picar-x-rust 0.1.0 (6c139ea)）
                "server_build": (data.get("code") or {}).get("server"),
                "where": data.get("where"),
                "started_at": _iso(data.get("started_at")),
                "started_ts": data.get("started_at"),
                "autonomous_s": float(data.get("autonomous_s") or 0.0),
                "human_s": float(data.get("human_s") or 0.0),
                "human": int(counts.get("human") or 0),
                "safety": int(counts.get("safety") or 0),
                "end_reason": data.get("end_reason"),
                "ended": data.get("ended"),
                "settings": data.get("settings") or {},
            }
        )
    return runs


def collect_jobs(jobs_dir: Path) -> List[Dict[str, Any]]:
    """Control Center から動かしたジョブの記録（コマンド、時刻、結果）"""
    jobs = []
    jobs_dir = Path(jobs_dir)
    if not jobs_dir.is_dir():
        return jobs
    for path in sorted(jobs_dir.glob("*.json")):
        data = _read_json(path)
        if data:
            jobs.append(
                {
                    k: data.get(k)
                    for k in (
                        "id",
                        "workflow",
                        "title",
                        "command",
                        "started_at",
                        "ended_at",
                        "status",
                        "exit_code",
                        "produces",
                    )
                }
            )
    return jobs


# --------------------------------------------------------------------------- body


def _load_bodies(body_file: Optional[Path]):
    """車体の版。ファイルに誤りがあれば空にして、理由を画面に出す（来歴の画面は止めない）"""
    if body_file is None:
        return [], None
    try:
        return body.load_versions(body_file), None
    except body.BodyError as exc:
        return [], str(exc)


def _attach_bodies(bodies, sessions, runs) -> None:
    for session in sessions:
        resolved = body.resolve(bodies, session.pop("body_recorded", None), session["session_id"])
        session["body_version"] = resolved["id"]
        session["body_source"] = resolved["source"]
        session["body_changed"] = resolved["changed"]
    for run in runs:
        version = body.version_at(bodies, run.pop("started_ts", None))
        run["body_version"] = version["id"] if version else None


def body_checks(model, sessions_by_id, bodies, current_body) -> List[Dict[str, Any]]:
    """学習データの車体の版の確認。版が1つも登録されていなければ何も言わない"""
    record = model.get("body_record") or {}
    recorded = record.get("sessions") or {}
    per_session: Dict[str, Optional[str]] = {}
    for session_id in model["train_sessions"] + model["val_sessions"]:
        if recorded.get(session_id):
            per_session[session_id] = recorded[session_id]
        elif session_id in sessions_by_id:
            per_session[session_id] = sessions_by_id[session_id].get("body_version")
        else:
            per_session[session_id] = None
    counts: Dict[str, int] = {}
    for version in per_session.values():
        if version:
            counts[version] = counts.get(version, 0) + 1
    model["body_versions"] = sorted(counts)
    model["body_source"] = "metadata" if recorded else "date"
    if not bodies:
        return []
    checks = []

    def add(level, message):
        checks.append({"key": "body", "level": level, "message": message})

    names = ", ".join(f"{v}（{n} セッション）" for v, n in sorted(counts.items()))
    unknown = [s for s, v in per_session.items() if not v]
    if len(counts) > 1:
        add("warn", f"違う車体の版のデータを混ぜて学習した: {names}")
    if unknown:
        add("warn", f"車体の版が分からないセッション: {', '.join(unknown)}")
    by_id = {v["id"]: v for v in bodies}
    changed = [
        v
        for v, fp in (record.get("versions") or {}).items()
        if v in by_id and fp and by_id[v]["fingerprint"] != fp
    ]
    if changed:
        add("warn", f"学習の後に車体の版の中身が書き換えられた: {', '.join(changed)}")
    if current_body and counts and current_body["id"] not in counts:
        add(
            "warn",
            f"学習データの車体（{', '.join(sorted(counts))}）と今の車体（{current_body['id']}）"
            "が違う。走らせる前に動きを確かめる",
        )
    elif len(counts) == 1 and not changed:
        add("ok", f"車体の版 {names} のデータで学習")
    return checks


# --------------------------------------------------------------------------- catalog


def build_catalog(
    models_dir: Path,
    data_dirs: Iterable[Path],
    annotations_dir: Path,
    cv_dir: Path,
    runs_dir: Path,
    deployments_file: Optional[Path] = None,
    jobs_dir: Optional[Path] = None,
    body_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """画面に出す全体像。モデル、データ、評価、配置、走行と、その間のつながり"""
    data_dirs = list(data_dirs)
    sessions = collect_sessions(data_dirs, annotations_dir)
    sessions_by_id = {s["session_id"]: s for s in sessions}
    models = collect_models(models_dir)
    cv = collect_cv(cv_dir)
    runs = collect_runs(runs_dir)
    bodies, body_error = _load_bodies(body_file)
    _attach_bodies(bodies, sessions, runs)
    deployments = stages.load_deployments(deployments_file)
    deployed = stages.deployed_version(deployments)
    jobs = collect_jobs(jobs_dir) if jobs_dir else []

    by_fp: Dict[str, List[str]] = {}
    for model in models:
        if model["data_fingerprint"]:
            by_fp.setdefault(model["data_fingerprint"], []).append(model["version"])

    current_body = body.current(bodies) if bodies else None
    for model in models:
        checks = model_checks(model, sessions_by_id)
        checks += body_checks(model, sessions_by_id, bodies, current_body)
        model["checks"] = checks
        model["health"] = health(checks)
        model_runs = [r for r in runs if r["model_version"] == model["version"]]
        autonomous = sum(r["autonomous_s"] for r in model_runs)
        human = sum(r["human"] for r in model_runs)
        model["field"] = {
            "runs": len(model_runs),
            "autonomous_min": round(autonomous / 60.0, 2),
            "human": human,
            "safety": sum(r["safety"] for r in model_runs),
            "human_per_10min": round(human * 600.0 / autonomous, 2) if autonomous > 0 else None,
            "fingerprint_mismatch": sum(
                1
                for r in model_runs
                if r["data_fingerprint"]
                and model["data_fingerprint"]
                and r["data_fingerprint"] != model["data_fingerprint"]
            ),
            # 学習データと違う車体の版で走った回数
            "body_mismatch": sum(
                1
                for r in model_runs
                if r["body_version"]
                and model["body_versions"]
                and r["body_version"] not in model["body_versions"]
            ),
        }
        model["cv"] = [
            c["name"]
            for c in cv
            if c.get("data_fingerprint") and c["data_fingerprint"] == model["data_fingerprint"]
        ]
        model["deployed"] = model["version"] == deployed["version"]
        made_by = [
            j for j in jobs if j.get("produces") and Path(j["produces"]).name == model["version"]
        ]
        model["created_by"] = made_by[-1] if made_by else None
        model["deployments"] = [d for d in deployments if d.get("version") == model["version"]]
        # 有効 = 採用の判断をしていて、来歴と成果物に誤りがない
        model["usable"] = model["stage"] == "approved" and model["health"] != "error"

    for session in sessions:
        session["used_by"] = [
            m["version"]
            for m in models
            if session["session_id"] in m["train_sessions"] + m["val_sessions"]
        ]

    latest = max(
        models, key=lambda m: (m["created_at"] or "", _version_key(m["version"])), default=None
    )
    approved = [m for m in models if m["usable"]]
    latest_approved = max(
        approved, key=lambda m: (m["created_at"] or "", _version_key(m["version"])), default=None
    )
    pi_runs = sorted((r for r in runs if r["where"] == "pi"), key=lambda r: r["started_at"] or "")
    last_pi_run = pi_runs[-1] if pi_runs else None

    body_list = body.public(bodies)
    for entry in body_list:
        mine = [s for s in sessions if s["body_version"] == entry["id"]]
        entry["sessions"] = [s["session_id"] for s in mine]
        entry["frames"] = sum(s["frames"] for s in mine)
        entry["models"] = [m["version"] for m in models if entry["id"] in m["body_versions"]]
        entry["runs"] = sum(1 for r in runs if r["body_version"] == entry["id"])
        entry["current"] = bool(current_body and current_body["id"] == entry["id"])

    return {
        "generated_at": stages.now_iso(),
        "models": models,
        "sessions": sessions,
        "cv": cv,
        "runs": runs,
        "deployments": deployments,
        "bodies": body_list,
        "summary": {
            "body_current": current_body["id"] if current_body else None,
            "body_error": body_error,
            "models": len(models),
            "usable_models": len(approved),
            "sessions": len(sessions),
            "frames": sum(s["frames"] for s in sessions),
            "hours": round(sum(s["duration_sec"] or 0 for s in sessions) / 3600.0, 2),
            "runs": len(runs),
            "latest": latest["version"] if latest else None,
            "latest_approved": latest_approved["version"] if latest_approved else None,
            "deployed": deployed,
            "last_pi_run": {
                "model_version": last_pi_run["model_version"],
                "server_build": last_pi_run["server_build"],
                "run_id": last_pi_run["run_id"],
                "started_at": last_pi_run["started_at"],
            }
            if last_pi_run
            else None,
            "unused_sessions": [s["session_id"] for s in sessions if not s["used_by"]],
            "incomplete_sessions": incomplete_sessions(data_dirs),
        },
        "graph": lineage_graph(models, sessions, cv, runs),
    }


def lineage_graph(models, sessions, cv, runs) -> Dict[str, Any]:
    """来歴の図の節と辺。列（column）は データ → ラベル → モデル → 評価・走行"""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen = set()

    def node(node_id, kind, label, column, **extra):
        if node_id in seen:
            return
        seen.add(node_id)
        nodes.append({"id": node_id, "kind": kind, "label": label, "column": column, **extra})

    for s in sessions:
        node(
            f"session:{s['session_id']}",
            "session",
            s["session_id"],
            0,
            frames=s["frames"],
            body=s.get("body_version"),
        )
        if s.get("annotation_hash"):
            node(f"annotation:{s['session_id']}", "annotation", f"ラベル {s['session_id']}", 1)
            edges.append(
                {
                    "from": f"session:{s['session_id']}",
                    "to": f"annotation:{s['session_id']}",
                    "kind": "labels",
                }
            )
    for m in models:
        node(
            f"model:{m['version']}", "model", m["version"], 2, stage=m["stage"], health=m["health"]
        )
        for role, ids in (("train", m["train_sessions"]), ("val", m["val_sessions"])):
            for session_id in ids:
                if f"session:{session_id}" not in seen:
                    node(f"session:{session_id}", "session", session_id, 0, missing=True)
                edges.append(
                    {"from": f"session:{session_id}", "to": f"model:{m['version']}", "kind": role}
                )
        for session_id in m["annotations"]:
            if f"annotation:{session_id}" in seen:
                edges.append(
                    {
                        "from": f"annotation:{session_id}",
                        "to": f"model:{m['version']}",
                        "kind": "labels",
                    }
                )
    for c in cv:
        if c["kind"] != "cross_validation":
            continue
        targets = [
            m
            for m in models
            if c.get("data_fingerprint") and m["data_fingerprint"] == c["data_fingerprint"]
        ]
        if not targets:
            continue
        node(f"cv:{c['name']}", "cv", c["name"], 3)
        for m in targets:
            edges.append(
                {"from": f"model:{m['version']}", "to": f"cv:{c['name']}", "kind": "same_data"}
            )
    # 走行は1回ずつではなく、モデルごとにまとめる（評価と同じ列に並べる）
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        if r["model_version"] and f"model:{r['model_version']}" in seen:
            by_model.setdefault(r["model_version"], []).append(r)
    for version, group in by_model.items():
        node(
            f"field:{version}",
            "field",
            f"{version} の走行",
            3,
            runs=len(group),
            human=sum(r["human"] for r in group),
            safety=sum(r["safety"] for r in group),
            autonomous_min=round(sum(r["autonomous_s"] for r in group) / 60.0, 1),
        )
        edges.append({"from": f"model:{version}", "to": f"field:{version}", "kind": "drove"})
    return {"nodes": nodes, "edges": edges}


def default_catalog(base: Path = Path(".")) -> Dict[str, Any]:
    base = Path(base)
    return build_catalog(
        models_dir=base / "models",
        data_dirs=[base / "data" / "raw", base / "data" / "processed"],
        annotations_dir=base / "data" / "annotations",
        cv_dir=base / "data" / "cv",
        runs_dir=base / "data" / "runs",
        deployments_file=None,
        jobs_dir=base / "data" / "jobs",
        body_file=base / "configs" / "body.toml",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Show model and data lineage")
    parser.add_argument("--summary", action="store_true", help="Print the summary only")
    args = parser.parse_args(argv)
    catalog = default_catalog()
    output = catalog["summary"] if args.summary else catalog
    print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
