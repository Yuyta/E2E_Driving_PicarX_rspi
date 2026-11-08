"""
走行データのマニフェスト（ファイルのハッシュの一覧）

走行データはセッション単位で追記のみとし、書き換えない（ROADMAP の M3）。
各セッションに `manifest.json` を置き、後から変わっていないかを確かめられるようにする。

使い方（1行目は未作成のセッションに作る。2行目は変わっていないか確かめる）:
    uv run python -m src.e2e.dataset.manifest --data_dir data/raw
    uv run python -m src.e2e.dataset.manifest --data_dir data/raw --verify
"""

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Union

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

# 集計から作り直せるファイルは対象外にする
DERIVED_FILES = {MANIFEST_NAME, "metadata.json"}


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files(session_dir: Path) -> Dict[str, Dict[str, Union[str, int]]]:
    """セッション内のデータファイルのハッシュと大きさを、相対パス（`/` 区切り）をキーにして返す"""
    files: Dict[str, Dict[str, Union[str, int]]] = {}
    for path in sorted(session_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(session_dir).as_posix()
        if rel in DERIVED_FILES:
            continue
        files[rel] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return files


def combined_hash(files: Dict[str, Dict[str, Union[str, int]]]) -> str:
    """ファイルの一覧全体を表すハッシュ。パスと内容のどちらが変わっても値が変わる"""
    lines = "".join(f"{rel}\t{info['sha256']}\n" for rel, info in sorted(files.items()))
    return sha256_bytes(lines.encode("utf-8"))


def build_manifest(session_dir: Union[str, Path]) -> Dict:
    session_dir = Path(session_dir)
    files = collect_files(session_dir)
    return {
        "version": MANIFEST_VERSION,
        "session_id": session_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(files),
        "hash": combined_hash(files),
        "files": files,
    }


def load_manifest(session_dir: Union[str, Path]) -> Dict | None:
    path = Path(session_dir) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(session_dir: Union[str, Path]) -> Dict:
    """マニフェストを作る。既にある場合は書き換えない（追記のみの運用のため）"""
    session_dir = Path(session_dir)
    if (session_dir / MANIFEST_NAME).exists():
        raise FileExistsError(f"Manifest already exists: {session_dir / MANIFEST_NAME}")
    manifest = build_manifest(session_dir)
    with open(session_dir / MANIFEST_NAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return manifest


@dataclass
class VerifyResult:
    session_id: str
    has_manifest: bool
    missing: List[str] = field(default_factory=list)
    changed: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.has_manifest and not (self.missing or self.changed or self.added)


def verify_session(session_dir: Union[str, Path]) -> VerifyResult:
    """マニフェストと今のファイルを比べる。追加も変更として扱う（閉じたセッションには足さない）"""
    session_dir = Path(session_dir)
    manifest = load_manifest(session_dir)
    result = VerifyResult(session_id=session_dir.name, has_manifest=manifest is not None)
    if manifest is None:
        return result

    expected = manifest.get("files", {})
    actual = collect_files(session_dir)
    for rel, info in expected.items():
        if rel not in actual:
            result.missing.append(rel)
        elif actual[rel]["sha256"] != info["sha256"]:
            result.changed.append(rel)
    result.added = sorted(rel for rel in actual if rel not in expected)
    return result


def find_session_dirs(data_dir: Union[str, Path]) -> List[Path]:
    """`labels.jsonl` を持つセッションを探す。空のセッションは対象外"""
    root = Path(data_dir)
    if (root / "labels.jsonl").exists():
        return [root]
    return sorted(p.parent for p in root.rglob("labels.jsonl"))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create or verify per-session data manifests")
    parser.add_argument("--data_dir", default="data/raw", help="Session or parent folder")
    parser.add_argument("--verify", action="store_true", help="Check sessions against manifests")
    args = parser.parse_args(argv)

    sessions = find_session_dirs(args.data_dir)
    if not sessions:
        print(f"No sessions with labels.jsonl found in: {args.data_dir}")
        return 1

    failed = 0
    for session in sessions:
        if args.verify:
            result = verify_session(session)
            if not result.has_manifest:
                print(f"NO MANIFEST  {session}")
                failed += 1
            elif result.ok:
                print(f"OK           {session}")
            else:
                failed += 1
                print(f"CHANGED      {session}")
                for label, items in (
                    ("missing", result.missing),
                    ("changed", result.changed),
                    ("added", result.added),
                ):
                    for rel in items:
                        print(f"    {label}: {rel}")
        elif load_manifest(session) is None:
            manifest = write_manifest(session)
            print(f"CREATED      {session} ({manifest['file_count']} files)")
        else:
            print(f"EXISTS       {session}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
