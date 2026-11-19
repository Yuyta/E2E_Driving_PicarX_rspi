"""モデルの採否（ステージ）と、Pi への配置の記録

- ステージは `models/registry.json` に置く。モデルと一緒に git で残し、いつ誰の判断で
  採用・却下したかを変更の履歴としてたどれるようにする
- 配置の記録は `data/deployments.jsonl`（1行1件、追記のみ）。接続先のホスト名は残さない

ステージ:
    candidate  学習しただけで、まだ判断していない（既定）
    approved   有効。走行に使ってよい
    rejected   使わない（性能や来歴に問題がある）
    archived   以前は使っていたが、今は使わない
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REGISTRY_FILE = "registry.json"
REGISTRY_FORMAT = 1
DEPLOYMENTS_FILE = Path("data") / "deployments.jsonl"
# テストで本物の記録に書かないように、置き場所を変えられる
DEPLOYMENTS_ENV = "PICARX_DEPLOYMENTS_FILE"

STAGES = ("candidate", "approved", "rejected", "archived")
DEFAULT_STAGE = "candidate"
STAGE_LABELS = {
    "candidate": "候補",
    "approved": "有効",
    "rejected": "却下",
    "archived": "廃止",
}


class StageError(ValueError):
    pass


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_registry(models_dir) -> Dict[str, Any]:
    path = Path(models_dir) / REGISTRY_FILE
    if not path.is_file():
        return {"format": REGISTRY_FORMAT, "models": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StageError(f"{path} を読めません: {exc}") from exc
    data.setdefault("format", REGISTRY_FORMAT)
    data.setdefault("models", {})
    return data


def save_registry(models_dir, registry: Dict[str, Any]) -> None:
    path = Path(models_dir) / REGISTRY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def stage_of(registry: Dict[str, Any], version: str) -> Dict[str, Any]:
    entry = registry.get("models", {}).get(version) or {}
    return {
        "stage": entry.get("stage", DEFAULT_STAGE),
        "note": entry.get("note", ""),
        "updated_at": entry.get("updated_at"),
        "history": list(entry.get("history", [])),
    }


def set_stage(models_dir, version: str, stage: str, note: str = "") -> Dict[str, Any]:
    """ステージを変え、履歴に残す。モデルのフォルダがなければ変えない"""
    if stage not in STAGES:
        raise StageError(f"ステージは {', '.join(STAGES)} のどれかです: {stage}")
    if not (Path(models_dir) / version / "metadata.json").is_file():
        raise StageError(f"モデル {version} がありません")
    registry = load_registry(models_dir)
    entry = registry["models"].setdefault(version, {"history": []})
    previous = entry.get("stage", DEFAULT_STAGE)
    at = now_iso()
    entry["history"] = list(entry.get("history", [])) + [
        {"from": previous, "to": stage, "at": at, "note": note}
    ]
    entry["stage"] = stage
    entry["note"] = note
    entry["updated_at"] = at
    save_registry(models_dir, registry)
    return stage_of(registry, version)


def deployments_path() -> Path:
    return Path(os.environ.get(DEPLOYMENTS_ENV) or DEPLOYMENTS_FILE)


def record_deployment(
    action: str,
    version: Optional[str],
    ok: bool,
    path: Optional[Path] = None,
    detail: str = "",
) -> Dict[str, Any]:
    """Pi への配置の操作（push / install / rollback）を1行追記する"""
    event = {"at": now_iso(), "action": action, "version": version, "ok": ok, "detail": detail}
    path = Path(path) if path else deployments_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event


def load_deployments(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = Path(path) if path else deployments_path()
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def deployed_version(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """記録からわかる Pi の使用中の版

    push で使用中にした版を積み、rollback で1つ戻す（Pi の `model rollback` と同じ）。
    記録の外で Pi を操作した場合は合わないので、走行の記録（run.json）とも並べて見せる。
    """
    stack: List[str] = []
    last = None
    for event in events:
        if not event.get("ok"):
            continue
        if event.get("action") == "push" and event.get("version"):
            stack.append(event["version"])
            last = event
        elif event.get("action") == "rollback" and stack:
            stack.pop()
            last = event
    return {"version": stack[-1] if stack else None, "at": last.get("at") if last else None}
