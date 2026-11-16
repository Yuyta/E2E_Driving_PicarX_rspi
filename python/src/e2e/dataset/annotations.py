"""
走行データへの手作業のラベル（注釈）

走行データはセッション単位で追記のみとし、書き換えない（ROADMAP の M3、`manifest.py`）。
そこで、人が付けるラベルはセッションの外の `data/annotations/<セッションID>.json` に置く。
元の `labels.jsonl` とマニフェストは変わらず、注釈を消せばいつでも元のデータに戻る。

フレームごとに持てる値:
    tags: 自由なタグ（「脚」「布団」「ぶれ」など）。学習には使わず、絞り込みと集計に使う
    exclude: True なら学習と評価に使わない
    intent: 学習に使う意図を上書きする（any / left / right）
    blocked: 前方が塞がっているかのラベルを上書きする（True / False / None = 不明）
    note: 自由なメモ

`intent` と `blocked` は、キーがあれば上書きする（`blocked: None` は「不明にする」）。
キーがなければ自動で付けたラベル（意図の付け直し、超音波、引っかかり）を使う。
"""

import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from ..intent import INTENTS
except ImportError:
    from src.e2e.intent import INTENTS

ANNOTATIONS_DIRNAME = "annotations"
SCHEMA_VERSION = 1

# 上書きできる学習のラベル
OVERRIDE_FIELDS = ("intent", "blocked")
FIELDS = ("tags", "exclude", "note") + OVERRIDE_FIELDS


class AnnotationError(ValueError):
    """注釈の値が正しくない"""


def default_annotations_dir(session_dir: Union[str, Path]) -> Path:
    """セッション（`data/raw/<ID>` や `data/interventions/<ID>`）に対応する注釈の置き場所"""
    return Path(session_dir).resolve().parent.parent / ANNOTATIONS_DIRNAME


def _check(field: str, value: Any) -> Any:
    if field == "tags":
        if not isinstance(value, (list, tuple)):
            raise AnnotationError("tags must be a list")
        tags = []
        for tag in value:
            tag = str(tag).strip()
            if tag and tag not in tags:
                tags.append(tag)
        return tags
    if field == "exclude":
        return bool(value)
    if field == "note":
        return str(value)
    if field == "intent":
        if value not in INTENTS:
            raise AnnotationError(f"intent must be one of {INTENTS}, got {value!r}")
        return value
    if field == "blocked":
        if value not in (True, False, None):
            raise AnnotationError(f"blocked must be true, false or null, got {value!r}")
        return value
    raise AnnotationError(f"unknown field {field!r}")


def _is_empty(field: str, value: Any) -> bool:
    """値がないのと同じか。上書きのラベルは、キーがあること自体に意味がある"""
    if field == "tags":
        return not value
    if field == "exclude":
        return value is False
    if field == "note":
        return value == ""
    return False


class SessionAnnotations:
    """1セッションの注釈。フレームはファイル名（`frame_000123.jpg`）で指す"""

    def __init__(self, session_id: str, path: Path):
        self.session_id = session_id
        self.path = Path(path)
        self.frames: Dict[str, Dict[str, Any]] = {}
        self.dirty = False
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise AnnotationError(f"{self.path}: unsupported version {version!r}")
        frames = data.get("frames") or {}
        for frame, entry in frames.items():
            self.frames[frame] = {field: _check(field, entry[field]) for field in entry}

    def get(self, frame: str) -> Dict[str, Any]:
        """フレームの注釈（ないなら空）。返した辞書を書き換えても保存されない"""
        entry = self.frames.get(frame, {})
        return {k: (list(v) if isinstance(v, list) else v) for k, v in entry.items()}

    def set_field(self, frames: Iterable[str], field: str, value: Any) -> int:
        """値を設定する。変わったフレームの数を返す"""
        value = _check(field, value)
        changed = 0
        for frame in frames:
            entry = self.frames.setdefault(frame, {})
            if _is_empty(field, value):
                if field in entry:
                    del entry[field]
                    changed += 1
            elif entry.get(field, object()) != value or field not in entry:
                entry[field] = list(value) if isinstance(value, list) else value
                changed += 1
            if not entry:
                del self.frames[frame]
        self.dirty = self.dirty or changed > 0
        return changed

    def clear_field(self, frames: Iterable[str], field: str) -> int:
        """値を消して、自動のラベル（上書きなし）に戻す"""
        if field not in FIELDS:
            raise AnnotationError(f"unknown field {field!r}")
        changed = 0
        for frame in frames:
            entry = self.frames.get(frame)
            if entry and field in entry:
                del entry[field]
                changed += 1
                if not entry:
                    del self.frames[frame]
        self.dirty = self.dirty or changed > 0
        return changed

    def add_tag(self, frames: Iterable[str], tag: str) -> int:
        tag = tag.strip()
        if not tag:
            raise AnnotationError("empty tag")
        changed = 0
        for frame in frames:
            tags = self.frames.get(frame, {}).get("tags", [])
            if tag not in tags:
                changed += self.set_field([frame], "tags", tags + [tag])
        return changed

    def remove_tag(self, frames: Iterable[str], tag: str) -> int:
        changed = 0
        for frame in frames:
            tags = self.frames.get(frame, {}).get("tags", [])
            if tag in tags:
                changed += self.set_field([frame], "tags", [t for t in tags if t != tag])
        return changed

    def to_json(self) -> Dict[str, Any]:
        return {
            "version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "frames": {frame: self.frames[frame] for frame in sorted(self.frames)},
        }

    def save(self) -> None:
        """書き途中で止まっても前の内容が残るよう、一時ファイルに書いてから置き換える"""
        if not self.dirty:
            return
        if not self.frames and not self.path.exists():
            self.dirty = False
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.to_json(), f, ensure_ascii=False, indent=1)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        self.dirty = False

    def content_hash(self) -> Optional[str]:
        """学習の記録に残すための、注釈の中身のハッシュ（更新時刻は含めない）。なければ None"""
        if not self.frames:
            return None
        body = json.dumps(
            {frame: self.frames[frame] for frame in sorted(self.frames)},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()


class AnnotationStore:
    """複数セッションの注釈。セッションごとのファイルを必要になった時に読む"""

    def __init__(self, annotations_dir: Optional[Union[str, Path]] = None):
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None
        self.sessions: Dict[str, SessionAnnotations] = {}

    def path_for(self, session_dir: Union[str, Path]) -> Path:
        session_dir = Path(session_dir)
        base = self.annotations_dir or default_annotations_dir(session_dir)
        return base / f"{session_dir.name}.json"

    def session(self, session_dir: Union[str, Path]) -> SessionAnnotations:
        session_dir = Path(session_dir)
        key = str(session_dir.resolve())
        if key not in self.sessions:
            self.sessions[key] = SessionAnnotations(session_dir.name, self.path_for(session_dir))
        return self.sessions[key]

    def save(self) -> None:
        for annotations in self.sessions.values():
            annotations.save()

    @property
    def dirty(self) -> bool:
        return any(a.dirty for a in self.sessions.values())

    def all_tags(self) -> List[str]:
        tags = set()
        for annotations in self.sessions.values():
            for entry in annotations.frames.values():
                tags.update(entry.get("tags", []))
        return sorted(tags)
