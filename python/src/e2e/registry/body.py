"""車体の版（ハードウェアの構成と校正）

走行データは、どの車体で集めたかで意味が変わる（M6 で前輪の修理やジャイロの追加をすると、
同じ操作でも動きが変わる）。そこで車体の版を `configs/body.toml` に人が書いて残し、
セッション・モデル・走行の来歴に車体の版をつなぐ。

- 版は `since`（使い始めた日時）の順に並べる。ある時刻の版は、その時刻より前で最後の版
- 収集の時に、その時の版をセッションの `body.json` に書く（`record_session`）。
  書いていない古いセッションは、セッション名の日時から版を決める
- 各版の中身のハッシュ（fingerprint）も残し、後から版の中身を書き換えたら分かるようにする

使い方:
    uv run python -m src.e2e.registry.body show
    uv run python -m src.e2e.registry.body add --id b2 --title "M6 の後" \\
        --equipment "ジャイロ（I2C）" --set motor_balance=0
"""

import argparse
import hashlib
import json
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

PYTHON_DIR = Path(__file__).resolve().parents[3]
BODY_FILE = PYTHON_DIR / "configs" / "body.toml"
SESSION_FILE = "body.json"
VERSION_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,32}$")
SESSION_TIME = re.compile(r"^(\d{8}_\d{6})")

When = Union[datetime, float, int, str, None]


class BodyError(ValueError):
    """車体の版のファイルが正しくない"""


def _parse_since(value: Any, version_id: str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "year") and not isinstance(value, datetime):  # TOML の日付
        return datetime(value.year, value.month, value.day)
    try:
        return datetime.fromisoformat(str(value)).replace(tzinfo=None)
    except ValueError as exc:
        raise BodyError(f"{version_id}: since の日時を読めません ({value})") from exc


def fingerprint(entry: Dict[str, Any]) -> str:
    """版の中身（構成・校正・メモ）のハッシュ。並び順や書き方の違いでは変わらない"""
    content = {k: v for k, v in entry.items() if k not in ("fingerprint", "since_dt")}
    text = json.dumps(content, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_versions(path: Path = BODY_FILE) -> List[Dict[str, Any]]:
    """版の一覧（since の古い順）。ファイルがなければ空。誤りがあれば BodyError"""
    path = Path(path)
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BodyError(f"{path.name} を読めません: {exc}") from exc
    versions = []
    seen = set()
    for raw in data.get("versions") or []:
        version_id = str(raw.get("id") or "")
        if not VERSION_ID.match(version_id):
            raise BodyError(f"版の id が正しくありません: {version_id!r}")
        if version_id in seen:
            raise BodyError(f"版の id が重複しています: {version_id}")
        seen.add(version_id)
        if "since" not in raw:
            raise BodyError(f"{version_id}: since（使い始めた日時）がありません")
        since = _parse_since(raw["since"], version_id)
        entry = {
            "id": version_id,
            "since": since.isoformat(timespec="minutes"),
            "title": str(raw.get("title") or ""),
            "equipment": [str(x) for x in raw.get("equipment") or []],
            "calibration": dict(raw.get("calibration") or {}),
            "notes": str(raw.get("notes") or ""),
        }
        entry["fingerprint"] = fingerprint(entry)
        entry["since_dt"] = since
        versions.append(entry)
    versions.sort(key=lambda v: v["since_dt"])
    return versions


def _to_datetime(when: When) -> Optional[datetime]:
    if when is None or when == "":
        return None
    if isinstance(when, datetime):
        return (
            when.replace(tzinfo=None)
            if when.tzinfo is None
            else when.astimezone().replace(tzinfo=None)
        )
    if isinstance(when, (int, float)):
        return datetime.fromtimestamp(when)
    text = str(when)
    match = SESSION_TIME.match(text)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    try:
        parsed = datetime.fromisoformat(re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", text))
    except ValueError:
        return None
    return parsed.astimezone().replace(tzinfo=None) if parsed.tzinfo else parsed


def version_at(versions: List[Dict[str, Any]], when: When) -> Optional[Dict[str, Any]]:
    """その時刻に使っていた版。時刻が分からない、または最初の版より前なら None"""
    moment = _to_datetime(when)
    if moment is None:
        return None
    found = None
    for version in versions:
        if version["since_dt"] <= moment:
            found = version
    return found


def current(versions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return version_at(versions, datetime.now())


def snapshot(version: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """記録に残す形（版の id と中身のハッシュ、構成と校正の写し）"""
    if version is None:
        return None
    return {
        "id": version["id"],
        "fingerprint": version["fingerprint"],
        "title": version["title"],
        "equipment": version["equipment"],
        "calibration": version["calibration"],
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }


def record_session(session_dir: Path, path: Path = BODY_FILE) -> Optional[Dict[str, Any]]:
    """収集を始める時に、今の車体の版をセッションに書く。失敗しても収集は止めない"""
    try:
        record = snapshot(current(load_versions(path)))
    except BodyError as exc:
        print(f"⚠ 車体の版を記録できません: {exc}", file=sys.stderr)
        return None
    if record is None:
        return None
    try:
        with open(Path(session_dir) / SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        print(f"⚠ 車体の版を記録できません: {exc}", file=sys.stderr)
        return None
    return record


def read_session_record(session_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((Path(session_dir) / SESSION_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("id") else None


def resolve(
    versions: List[Dict[str, Any]], recorded: Optional[Dict[str, Any]], when: When
) -> Dict[str, Any]:
    """記録があればそれを、なければ時刻から版を決める

    source: recorded（収集の時に記録）/ date（日時から推定）/ None（分からない）
    changed: 記録した後に版の中身（configs/body.toml）が書き換えられた
    """
    by_id = {v["id"]: v for v in versions}
    if recorded and recorded.get("id"):
        version = by_id.get(recorded["id"])
        return {
            "id": recorded["id"],
            "source": "recorded",
            "known": version is not None,
            "changed": bool(
                version
                and recorded.get("fingerprint")
                and recorded["fingerprint"] != version["fingerprint"]
            ),
        }
    version = version_at(versions, when)
    return {
        "id": version["id"] if version else None,
        "source": "date" if version else None,
        "known": version is not None,
        "changed": False,
    }


def public(versions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """画面と JSON に出す形（datetime を除く）"""
    return [{k: v for k, v in version.items() if k != "since_dt"} for version in versions]


# --------------------------------------------------------------------------- CLI


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    # JSON の文字列は TOML の基本文字列としてそのまま読める
    return json.dumps(str(value), ensure_ascii=False)


def _parse_setting(text: str):
    if "=" not in text:
        raise BodyError(f"--set は 名前=値 で指定してください: {text}")
    key, raw = (part.strip() for part in text.split("=", 1))
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
        raise BodyError(f"校正の名前が正しくありません: {key}")
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    return key, raw


def add_version(
    path: Path,
    version_id: str,
    title: str,
    since: Optional[str] = None,
    equipment: Iterable[str] = (),
    calibration: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """新しい版をファイルの末尾に足す（既存の版は書き換えない）"""
    path = Path(path)
    versions = load_versions(path)
    if not VERSION_ID.match(version_id):
        raise BodyError(f"版の id が正しくありません: {version_id!r}")
    if any(v["id"] == version_id for v in versions):
        raise BodyError(f"版の id がすでにあります: {version_id}")
    moment = _parse_since(since, version_id) if since else datetime.now().replace(second=0)
    moment = moment.replace(microsecond=0)
    if versions and moment <= versions[-1]["since_dt"]:
        raise BodyError(
            f"since は最後の版（{versions[-1]['id']}: {versions[-1]['since']}）より後にしてください"
        )
    lines = [
        "",
        "[[versions]]",
        f"id = {_toml_value(version_id)}",
        f"since = {_toml_value(moment.isoformat(timespec='minutes'))}",
        f"title = {_toml_value(title)}",
        f"equipment = {_toml_value(list(equipment))}",
    ]
    if notes:
        lines.append(f"notes = {_toml_value(notes)}")
    lines.append("[versions.calibration]")
    for key, value in (calibration or {}).items():
        lines.append(f"{key} = {_toml_value(value)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")
    added = next(v for v in load_versions(path) if v["id"] == version_id)
    return added


def _print_versions(versions: List[Dict[str, Any]]) -> None:
    if not versions:
        print(f"車体の版がまだありません（{BODY_FILE.name}）。add で登録してください")
        return
    now = current(versions)
    for version in versions:
        mark = "● 使用中" if now and version["id"] == now["id"] else "  "
        print(f"{mark} {version['id']}  {version['since']} から  {version['title']}")
        if version["equipment"]:
            print(f"     装備: {', '.join(version['equipment'])}")
        if version["calibration"]:
            values = ", ".join(f"{k}={v}" for k, v in version["calibration"].items())
            print(f"     校正: {values}")
        if version["notes"]:
            print(f"     メモ: {version['notes']}")
        print(f"     中身のハッシュ: {version['fingerprint']}")


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Body (hardware) versions for data lineage")
    parser.add_argument("--file", type=Path, default=BODY_FILE, help="body.toml path")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("show", help="List body versions")
    add = sub.add_parser("add", help="Register a new body version")
    add.add_argument("--id", required=True, help="e.g. b2")
    add.add_argument("--title", required=True)
    add.add_argument("--since", help="ISO date/time (default: now)")
    add.add_argument("--equipment", action="append", default=[], help="Repeatable")
    add.add_argument("--set", action="append", default=[], help="Calibration name=value")
    add.add_argument("--notes", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            calibration = dict(_parse_setting(s) for s in args.set)
            version = add_version(
                args.file,
                args.id,
                args.title,
                args.since,
                args.equipment,
                calibration,
                args.notes,
            )
            print(f"✓ 車体の版 {version['id']} を登録しました（{version['since']} から）")
            print("  これ以降に集めたセッションは、この版の車体のデータとして扱われます")
            return 0
        _print_versions(load_versions(args.file))
        return 0
    except BodyError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
