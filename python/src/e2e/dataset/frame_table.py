"""
走行データのフレームを、表と統計で見るための1行ずつの値（データビューアが使う）

学習と同じ規則（止まっているフレーム、食い違い、意図の付け直し、塞がっているかのラベル）で
「学習に使うか」と「学習で使う意図とラベル」を計算し、手作業のラベル（`annotations`）を重ねる。
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..evaluation.metrics import MAX_STEERING_DEG, STRAIGHT_LIMIT
    from ..intent import INTENT_ANY
    from ..training.dataset import (
        BLOCKED_CLEAR_CM,
        BLOCKED_NEAR_CM,
        blocked_labels,
        is_stopped,
        resolve_session_dirs,
        turn_intents,
    )
    from .annotations import AnnotationStore
except ImportError:
    from src.e2e.dataset.annotations import AnnotationStore
    from src.e2e.evaluation.metrics import MAX_STEERING_DEG, STRAIGHT_LIMIT
    from src.e2e.intent import INTENT_ANY
    from src.e2e.training.dataset import (
        BLOCKED_CLEAR_CM,
        BLOCKED_NEAR_CM,
        blocked_labels,
        is_stopped,
        resolve_session_dirs,
        turn_intents,
    )


@dataclass(frozen=True)
class Column:
    key: str
    title: str
    # "cat"（種類）か "num"（数値）。統計で棒グラフと度数分布のどちらにするか
    kind: str
    # 最初から表に出すか
    visible: bool = True


COLUMNS: Tuple[Column, ...] = (
    Column("session", "セッション", "cat"),
    Column("index", "番号", "num"),
    Column("time_s", "経過秒", "num"),
    Column("warnings", "注意", "cat"),
    Column("tags", "タグ", "cat"),
    Column("training", "学習", "cat"),
    Column("motion", "動き", "cat"),
    Column("steering_deg", "舵角°", "num"),
    Column("direction", "向き", "cat"),
    Column("intent", "意図(学習)", "cat"),
    Column("distance_cm", "距離cm", "num"),
    Column("blocked", "塞がり", "cat"),
    Column("overrides", "手作業", "cat"),
    Column("note", "メモ", "cat"),
    Column("source", "種類", "cat", False),
    Column("collector", "収集", "cat", False),
    Column("speed", "速度", "num", False),
    Column("intent_recorded", "意図(記録)", "cat", False),
    Column("policy_state", "走行状態", "cat", False),
    Column("motion_flags", "食い違い", "cat", False),
    Column("battery_v", "電池V", "num", False),
    Column("floor_min", "床の最小", "num", False),
    Column("frame", "画像", "cat", False),
)
COLUMN_BY_KEY = {c.key: c for c in COLUMNS}
# 1つのフレームに複数の値がある列（カンマ区切り）
MULTI_VALUE_KEYS = ("tags", "overrides", "motion_flags", "warnings")

USED = "使う"
EXCLUDED_BY_HAND = "除外(手作業)"
EXCLUDED_STOPPED = "止まっている"
EXCLUDED_MISMATCH = "動きの食い違い"
EXCLUDED_FLAG = "除外(記録)"

# 怪しいフレームの印（`warnings_of`）。学習から自動で外すものではなく、人が確かめる手がかり
WARN_MISMATCH = "動きの食い違い"
WARN_STUCK = "引っかかり"
WARN_AGAINST_INTENT = "意図と逆の舵"
WARN_TURN_IN_CLEAR = "空いているのに曲がる"
WARN_STRAIGHT_NEAR = "近いのに直進"
WARN_JUMP = "舵の急変"
WARNINGS = (
    WARN_MISMATCH,
    WARN_STUCK,
    WARN_AGAINST_INTENT,
    WARN_TURN_IN_CLEAR,
    WARN_STRAIGHT_NEAR,
    WARN_JUMP,
)
# 意図が any で前が空いているのに、これより大きく切っている（約17°）。2026-11-16 の介入の修正は、
# 左へ寄る車を戻すために、空いていても右へ 20〜30° 切っていた
TURN_IN_CLEAR = 0.5
# これより近いのにまっすぐ進んでいる（cm）
STRAIGHT_NEAR_CM = 20.0
# 1フレームでこれより大きく舵が変わった（左いっぱいから右へ、など。約42°）
STEERING_JUMP = 1.2


@dataclass
class Settings:
    """学習の設定のうち、ラベルの計算に使う値（`configs/train.toml`）"""

    turn_intent: bool = True
    blocked_stuck: bool = True
    blocked_near_cm: float = BLOCKED_NEAR_CM
    blocked_clear_cm: float = BLOCKED_CLEAR_CM

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "Settings":
        return cls(
            turn_intent=bool(params.get("turn_intent", True)),
            blocked_stuck=bool(params.get("blocked_stuck", True)),
            blocked_near_cm=float(params.get("blocked_near_cm", BLOCKED_NEAR_CM)),
            blocked_clear_cm=float(params.get("blocked_clear_cm", BLOCKED_CLEAR_CM)),
        )


def direction_of(steering: float) -> str:
    if steering < -STRAIGHT_LIMIT:
        return "左"
    if steering > STRAIGHT_LIMIT:
        return "右"
    return "まっすぐ"


def motion_of(record: Dict[str, Any]) -> str:
    command = record.get("command") or {}
    speed = command.get("speed")
    if speed is None:
        speed = record.get("throttle", record.get("speed"))
    if speed is None:
        return "不明"
    speed = float(speed)
    if speed > 0:
        return "前進"
    if speed < 0:
        return "後退"
    return "停止"


def warnings_of(
    record: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    intent: str,
    blocked: Optional[bool],
) -> List[str]:
    """人が確かめたほうがよいフレームの理由。前進中のフレームの舵の規則と、動きの食い違い"""
    found = []
    flags = (record.get("motion_check") or {}).get("flags") or []
    if "no_motion" in flags:
        found.append(WARN_STUCK)
    if any(f != "no_motion" for f in flags):
        found.append(WARN_MISMATCH)
    if motion_of(record) != "前進":
        return found
    steering = float(record.get("steering", 0.0))
    if (intent == "left" and steering > STRAIGHT_LIMIT) or (
        intent == "right" and steering < -STRAIGHT_LIMIT
    ):
        found.append(WARN_AGAINST_INTENT)
    if intent == INTENT_ANY and blocked is False and abs(steering) >= TURN_IN_CLEAR:
        found.append(WARN_TURN_IN_CLEAR)
    distance = record.get("distance_cm")
    if (
        distance is not None
        and float(distance) < STRAIGHT_NEAR_CM
        and abs(steering) <= STRAIGHT_LIMIT
    ):
        found.append(WARN_STRAIGHT_NEAR)
    if previous is not None and motion_of(previous) == "前進":
        if abs(steering - float(previous.get("steering", 0.0))) >= STEERING_JUMP:
            found.append(WARN_JUMP)
    return found


def blocked_text(value: Optional[bool]) -> str:
    if value is True:
        return "塞がり"
    if value is False:
        return "空き"
    return "不明"


@dataclass
class SessionData:
    session_dir: Path
    records: List[Dict[str, Any]]
    # 自動で付けた学習のラベル（手作業の上書きの前）
    auto_intents: List[str]
    auto_blocked: List[Optional[bool]]


def load_session(session_dir: Path, settings: Settings) -> SessionData:
    with open(session_dir / "labels.jsonl", "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if settings.turn_intent:
        intents = turn_intents(records)
    else:
        intents = [r.get("intent") or INTENT_ANY for r in records]
    blocked = blocked_labels(
        records, settings.blocked_near_cm, settings.blocked_clear_cm, stuck=settings.blocked_stuck
    )
    return SessionData(session_dir, records, intents, blocked)


def build_row(
    session: SessionData, i: int, note: Dict[str, Any], start: Optional[float]
) -> Dict[str, Any]:
    """1フレームの表の値。`note` はそのフレームの手作業のラベル"""
    record = session.records[i]
    steering = float(record.get("steering", 0.0))
    command = record.get("command") or {}
    flags = (record.get("motion_check") or {}).get("flags") or []
    floor = record.get("floor")
    timestamp = record.get("timestamp")

    intent = note.get("intent", session.auto_intents[i])
    blocked = note["blocked"] if "blocked" in note else session.auto_blocked[i]

    if note.get("exclude"):
        training = EXCLUDED_BY_HAND
    elif record.get("exclude") or record.get("exclude_from_training"):
        training = EXCLUDED_FLAG
    elif flags:
        training = EXCLUDED_MISMATCH
    elif is_stopped(record):
        training = EXCLUDED_STOPPED
    else:
        training = USED

    overrides = [k for k in ("exclude", "intent", "blocked") if k in note]
    previous = session.records[i - 1] if i > 0 else None
    warnings = warnings_of(record, previous, intent, blocked)
    return {
        "session": session.session_dir.name,
        "source": session.session_dir.parent.name,
        "index": record.get("index", i),
        "time_s": (
            round(float(timestamp) - start, 2)
            if timestamp is not None and start is not None
            else None
        ),
        "collector": record.get("collector") or "",
        "motion": motion_of(record),
        "speed": command.get("speed"),
        "steering_deg": round(steering * MAX_STEERING_DEG, 1),
        "direction": direction_of(steering),
        "intent_recorded": record.get("intent") or INTENT_ANY,
        "intent": intent,
        "policy_state": record.get("policy_state") or "",
        "distance_cm": record.get("distance_cm"),
        "blocked": blocked_text(blocked),
        "motion_flags": ",".join(flags),
        "battery_v": record.get("battery_v"),
        "floor_min": min(floor) if isinstance(floor, list) and floor else None,
        "training": training,
        "warnings": ",".join(warnings),
        "tags": ",".join(note.get("tags", [])),
        "overrides": ",".join(overrides),
        "note": note.get("note", ""),
        "frame": record.get("frame", ""),
    }


class FrameTable:
    """読み込んだセッションのフレームの表。手作業のラベルを変えたら `refresh` で行を作り直す"""

    def __init__(
        self,
        data_dirs: Sequence[Path],
        settings: Optional[Settings] = None,
        store: Optional[AnnotationStore] = None,
    ):
        self.settings = settings or Settings()
        self.store = store or AnnotationStore()
        self.sessions: List[SessionData] = []
        for session_dir in resolve_session_dirs([Path(d) for d in data_dirs]):
            self.sessions.append(load_session(session_dir, self.settings))
            self.store.session(session_dir)
        # 行 → (セッションの番号, フレームの番号)
        self.keys: List[Tuple[int, int]] = []
        self.rows: List[Dict[str, Any]] = []
        # セッションの最初の時刻（経過秒の起点）
        self._starts = [
            next(
                (float(r["timestamp"]) for r in s.records if r.get("timestamp") is not None),
                None,
            )
            for s in self.sessions
        ]
        for s, session in enumerate(self.sessions):
            notes = self.store.session(session.session_dir)
            for i in range(len(session.records)):
                frame = session.records[i].get("frame", "")
                self.keys.append((s, i))
                self.rows.append(build_row(session, i, notes.get(frame), self._starts[s]))

    def __len__(self) -> int:
        return len(self.rows)

    def session_of(self, row: int) -> SessionData:
        return self.sessions[self.keys[row][0]]

    def record(self, row: int) -> Dict[str, Any]:
        s, i = self.keys[row]
        return self.sessions[s].records[i]

    def image_path(self, row: int) -> Path:
        session = self.session_of(row)
        record = self.record(row)
        frame = record.get("frame", "")
        source_dir = record.get("source_dir")
        base = (session.session_dir / source_dir).resolve() if source_dir else session.session_dir
        path = base / "images" / frame
        return path if path.exists() else base / frame

    def note(self, row: int) -> Dict[str, Any]:
        session = self.session_of(row)
        return self.store.session(session.session_dir).get(self.record(row).get("frame", ""))

    def refresh(self, rows: Sequence[int]) -> None:
        """手作業のラベルを変えた行の値を作り直す"""
        for row in rows:
            s, i = self.keys[row]
            self.rows[row] = build_row(self.sessions[s], i, self.note(row), self._starts[s])

    def by_session(self, rows: Sequence[int]) -> Dict[int, List[int]]:
        """行をセッションごとに分ける（注釈はセッション単位で保存するため）"""
        groups: Dict[int, List[int]] = {}
        for row in rows:
            groups.setdefault(self.keys[row][0], []).append(row)
        return groups

    def frames(self, rows: Sequence[int]) -> List[Tuple[Any, List[str]]]:
        """行を、(そのセッションの注釈, フレーム名の一覧) にまとめる"""
        result = []
        for s, group in self.by_session(rows).items():
            notes = self.store.session(self.sessions[s].session_dir)
            result.append((notes, [self.record(r).get("frame", "") for r in group]))
        return result


# ---- 統計 ----


def count_by(rows: Sequence[Dict[str, Any]], key: str) -> List[Tuple[str, int]]:
    """種類ごとの数（多い順）。タグは1つのフレームに複数あるので、タグごとに数える"""
    counter: Counter = Counter()
    for row in rows:
        value = row.get(key)
        if key in MULTI_VALUE_KEYS:
            parts = [p for p in str(value or "").split(",") if p]
            counter.update(parts or ["(なし)"])
        else:
            counter[_category(value)] += 1
    return counter.most_common()


def _category(value: Any) -> str:
    if value is None or value == "":
        return "(なし)"
    return str(value)


def histogram(
    rows: Sequence[Dict[str, Any]], key: str, bins: int = 20
) -> Tuple[List[Tuple[float, float, int]], int]:
    """数値の度数分布。[(下限, 上限, 数)] と、値のないフレームの数"""
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    missing = len(rows) - len(values)
    if not values:
        return [], missing
    low, high = min(values), max(values)
    if low == high:
        return [(low, high, len(values))], missing
    width = (high - low) / bins
    counts = [0] * bins
    for v in values:
        counts[min(int((v - low) / width), bins - 1)] += 1
    return [(low + k * width, low + (k + 1) * width, counts[k]) for k in range(bins)], missing


def summary(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
    values = sorted(float(row[key]) for row in rows if isinstance(row.get(key), (int, float)))
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    n = len(values)
    median = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
    return {
        "count": n,
        "min": values[0],
        "median": round(median, 3),
        "mean": round(sum(values) / n, 3),
        "max": values[-1],
    }


def crosstab(
    rows: Sequence[Dict[str, Any]], row_key: str, col_key: str
) -> Tuple[List[str], List[str], Dict[Tuple[str, str], int]]:
    """2つの種類の組み合わせの数。行と列の見出しは数の多い順"""
    counts: Counter = Counter()
    for row in rows:
        counts[(_category(row.get(row_key)), _category(row.get(col_key)))] += 1
    row_totals: Counter = Counter()
    col_totals: Counter = Counter()
    for (a, b), n in counts.items():
        row_totals[a] += n
        col_totals[b] += n
    return (
        [a for a, _ in row_totals.most_common()],
        [b for b, _ in col_totals.most_common()],
        dict(counts),
    )
