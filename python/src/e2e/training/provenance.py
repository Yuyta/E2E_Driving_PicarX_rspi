"""
学習の記録（来歴）

モデルから、使ったデータ、コード、環境、設定を一意にたどれるようにする（ROADMAP の M3）。
"""

import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from ..dataset.manifest import load_manifest, sha256_file, verify_session
except ImportError:
    from src.e2e.dataset.manifest import load_manifest, sha256_file, verify_session

PYTHON_DIR = Path(__file__).resolve().parents[3]
UV_LOCK = PYTHON_DIR / "uv.lock"


def git_state(cwd: Path = PYTHON_DIR) -> Dict[str, Any]:
    """コミットと、未コミットの変更の有無。git がなければ None を入れる"""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def environment() -> Dict[str, Any]:
    import torch

    versions: Dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(terse=True),
    }
    for name in ("torchvision", "onnx", "onnxruntime", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", None)
        except ImportError:
            versions[name] = None
    versions["uv_lock_sha256"] = sha256_file(UV_LOCK) if UV_LOCK.exists() else None
    return versions


def data_fingerprint(samples: Iterable[Dict[str, Any]]) -> str:
    """学習に使ったサンプルの中身そのもののハッシュ

    画像の内容、舵角、元のセッションを順に連ねる。データの置き場所が変わっても、
    中身が同じなら同じ値になる。
    """
    digest = hashlib.sha256()
    for sample in samples:
        image_hash = sha256_file(Path(sample["image_path"]))
        digest.update(
            f"{sample['session_id']}\t{image_hash}\t{sample['steering']:.6f}\n".encode("utf-8")
        )
    return digest.hexdigest()


def session_records(session_dirs: Iterable[Path]) -> List[Dict[str, Optional[Any]]]:
    """使ったセッションと、そのマニフェストのハッシュ。マニフェストと食い違えば印を付ける"""
    records = []
    for session_dir in session_dirs:
        manifest = load_manifest(session_dir)
        if manifest is None:
            records.append({"session_id": session_dir.name, "manifest_hash": None})
            continue
        records.append(
            {
                "session_id": session_dir.name,
                "manifest_hash": manifest.get("hash"),
                "manifest_ok": verify_session(session_dir).ok,
            }
        )
    return records


def body_record(session_dirs: Iterable[Path]) -> Dict[str, Any]:
    """学習データを集めた車体の版（configs/body.toml）。セッションごとの版と、各版の中身のハッシュ

    車体の版のファイルが読めない時も学習は止めず、理由を残す。
    """
    try:
        from ..registry import body
    except ImportError:
        from src.e2e.registry import body
    try:
        versions = body.load_versions()
    except body.BodyError as exc:
        return {"error": str(exc), "sessions": {}, "versions": {}}
    by_id = {v["id"]: v for v in versions}
    sessions: Dict[str, Optional[str]] = {}
    for session_dir in session_dirs:
        session_dir = Path(session_dir)
        resolved = body.resolve(versions, body.read_session_record(session_dir), session_dir.name)
        sessions[session_dir.name] = resolved["id"]
    used = sorted({v for v in sessions.values() if v})
    return {
        "sessions": sessions,
        "versions": {v: by_id[v]["fingerprint"] for v in used if v in by_id},
    }
