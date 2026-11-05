"""会話と人物の事実を保存する。再起動後も前回の話題を答えられるようにする。

DB には個人情報が入るので、git の追跡対象外の場所に置く。
"""

import os
import re
import sqlite3
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

MEMORY_DB_ENV = "PICARX_MEMORY_DB"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "memory" / "memory.db"

# trigram は3文字以上しか検索できないので、2文字の語は LIKE で探す
FTS_MIN_CHARS = 3
# 文からひらがなを除いた語を検索語にする（「キッチンの話」→「キッチン」）
KEYWORD_PATTERN = re.compile(r"[一-龥々ァ-ヶーA-Za-z0-9]{2,}")
MAX_KEYWORDS = 8

PREVIOUS_SESSION_MESSAGES = 6
RELATED_MESSAGES = 5
MAX_FACTS = 30
MAX_SNIPPET_CHARS = 200

ROLE_TEXT = {"user": "ユーザー", "assistant": "あなた"}
MEMORY_RULE = (
    "過去のことを聞かれたら、この記憶だけを使って答えること。"
    "記憶にないことは、覚えていないと答えること。"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS messages_after_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_after_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class StoredMessage(NamedTuple):
    id: int
    session_id: str
    role: str
    content: str
    created_at: str


def extract_keywords(text: str) -> List[str]:
    seen = dict.fromkeys(KEYWORD_PATTERN.findall(text))
    return list(seen)[:MAX_KEYWORDS]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _snippet(text: str) -> str:
    return text if len(text) <= MAX_SNIPPET_CHARS else text[:MAX_SNIPPET_CHARS] + "…"


def _format_line(message: StoredMessage) -> str:
    role = ROLE_TEXT.get(message.role, message.role)
    return f"- {message.created_at[:10]} {role}: {_snippet(message.content)}"


class MemoryStore:
    def __init__(self, path: Optional[Path | str] = None):
        target = str(path) if path is not None else os.getenv(MEMORY_DB_ENV) or str(DEFAULT_DB_PATH)
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self._lock = threading.Lock()
        # GUI の背景スレッドからも使うので、ロックで直列化する
        self._db = sqlite3.connect(target, check_same_thread=False)
        with self._lock, self._db:
            self._db.executescript(SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add_message(self, role: str, content: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (self.session_id, role, content, _now()),
            )

    def remember_fact(self, key: str, value: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=excluded.updated_at",
                (key, value, _now()),
            )

    def forget_fact(self, key: str) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute("DELETE FROM facts WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def facts(self) -> Dict[str, str]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM facts ORDER BY updated_at DESC, key LIMIT ?", (MAX_FACTS,)
            ).fetchall()
        return dict(rows)

    def previous_session_messages(self, limit: int = PREVIOUS_SESSION_MESSAGES):
        """今回より前の会話のうち、最新のものを古い順に返す"""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, session_id, role, content, created_at FROM messages "
                "WHERE session_id != ? ORDER BY id DESC LIMIT ?",
                (self.session_id, limit),
            ).fetchall()
        return [StoredMessage(*row) for row in reversed(rows)]

    def search(self, query: str, limit: int = RELATED_MESSAGES) -> List[StoredMessage]:
        """今回より前の会話から、一致する語が多く新しい順に探す"""
        hits: Counter = Counter()
        with self._lock:
            for keyword in extract_keywords(query):
                if len(keyword) >= FTS_MIN_CHARS:
                    quoted = '"' + keyword.replace('"', '""') + '"'
                    rows = self._db.execute(
                        "SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?", (quoted,)
                    )
                else:
                    rows = self._db.execute(
                        "SELECT id FROM messages WHERE content LIKE ?", (f"%{keyword}%",)
                    )
                hits.update(row[0] for row in rows.fetchall())

            if not hits:
                return []
            marks = ",".join("?" * len(hits))
            rows = self._db.execute(
                "SELECT id, session_id, role, content, created_at FROM messages "
                f"WHERE id IN ({marks}) AND session_id != ?",
                (*hits.keys(), self.session_id),
            ).fetchall()
        messages = [StoredMessage(*row) for row in rows]
        messages.sort(key=lambda m: (hits[m.id], m.id), reverse=True)
        return messages[:limit]

    def build_context(
        self,
        query: str,
        previous_limit: int = PREVIOUS_SESSION_MESSAGES,
        related_limit: int = RELATED_MESSAGES,
    ) -> str:
        """Agent のシステムプロンプトへ入れる文章を作る。記憶がなければ空文字を返す。"""
        facts = self.facts()
        previous = self.previous_session_messages(previous_limit)
        shown = {message.id for message in previous}
        related = [m for m in self.search(query, related_limit) if m.id not in shown]
        if not (facts or previous or related):
            return ""

        lines = ["【記憶】"]
        if facts:
            lines.append("覚えている事実:")
            lines.extend(f"- {key}: {value}" for key, value in facts.items())
        if previous:
            lines.append("前回の会話の終わり:")
            lines.extend(_format_line(message) for message in previous)
        if related:
            lines.append("関連する過去の会話:")
            lines.extend(_format_line(message) for message in related)
        lines.append(MEMORY_RULE)
        return "\n".join(lines)


_default_store: Optional[MemoryStore] = None
_default_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    """アプリ全体で共有する記憶を返す。最初に使う時に DB を開く。"""
    global _default_store
    with _default_store_lock:
        if _default_store is None:
            _default_store = MemoryStore()
        return _default_store
