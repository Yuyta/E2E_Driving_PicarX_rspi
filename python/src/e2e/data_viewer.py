"""
走行データのビューア（学習データの確認、手作業のラベル付け、分布と統計）

使い方:
    uv run python -m src.e2e.data_viewer
    uv run python -m src.e2e.data_viewer --data_dir data/raw data/interventions

フレーム
- 表で一覧し、列の見出しで並べ替え、見出しの右クリックで表示する列を選ぶ
- 怪しいフレーム（「注意」の列、`frame_table.warnings_of`）は黄色、手作業で除外したフレームは
  赤い取り消し線、学習に使わないフレームは灰色で表示する。「表示」で注意のあるものだけに絞れる
- 表の下の帯は、今のセッションの全フレームの流れ（状態の色と舵角の線）。ドラッグした範囲の
  フレームを選ぶ。表でも Shift+クリックで範囲を選べる。I で範囲の始まり、O で終わりを決める
- 選んだフレームにまとめて、タグ、学習からの除外、意図と前方の塞がりの上書き、メモを付ける
- Space で再生と停止、X で除外の切り替え、Ctrl+Z で元に戻す

統計
- 絞り込んだフレームの、列ごとの数と割合（数値は度数分布）、2つの列の組み合わせの数
- 表の行をダブルクリックすると、その値で絞り込んでフレームへ戻る

手作業のラベルは `data/annotations/<セッションID>.json` に、変えるたびに保存する。元の走行データ
（labels.jsonl、マニフェスト）は書き換えない。学習は `configs/train.toml` の `annotations = true`
でこのラベルを使う。
"""

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PySide6.QtCharts import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QHorizontalBarSeries,
    QHorizontalStackedBarSeries,
    QValueAxis,
)
from PySide6.QtCore import (
    QAbstractTableModel,
    QItemSelection,
    QItemSelectionModel,
    QMargins,
    QModelIndex,
    QPointF,
    QRectF,
    QSettings,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStyledItemDelegate,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from .dataset.annotations import AnnotationStore
    from .dataset.frame_table import (
        COLUMN_BY_KEY,
        COLUMNS,
        EXCLUDED_BY_HAND,
        EXCLUDED_FLAG,
        EXCLUDED_MISMATCH,
        EXCLUDED_STOPPED,
        MULTI_VALUE_KEYS,
        USED,
        FrameTable,
        Settings,
        count_by,
        crosstab,
        histogram,
        summary,
    )
    from .intent import INTENTS
except ImportError:
    from src.e2e.dataset.annotations import AnnotationStore
    from src.e2e.dataset.frame_table import (
        COLUMN_BY_KEY,
        COLUMNS,
        EXCLUDED_BY_HAND,
        EXCLUDED_FLAG,
        EXCLUDED_MISMATCH,
        EXCLUDED_STOPPED,
        MULTI_VALUE_KEYS,
        USED,
        FrameTable,
        Settings,
        count_by,
        crosstab,
        histogram,
        summary,
    )
    from src.e2e.intent import INTENTS

PYTHON_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIRS = ("data/raw", "data/interventions")
TRAIN_CONFIG = PYTHON_DIR / "configs" / "train.toml"
PLAY_INTERVAL_MS = 66  # 収集と同じ 15fps
NONE_TEXT = "(なし)"
AUTO_TEXT = "自動"
OTHERS_TEXT = "その他"
# 棒グラフに出す種類の数。多い時は残りを「その他」にまとめる
MAX_BARS = 25
MAX_STACKS = 8
HISTOGRAM_BINS = 14

# ---- 色 ----

FONT_FAMILY = "Yu Gothic UI"
C_BG = "#f4f6f9"
C_SURFACE = "#ffffff"
C_BORDER = "#dde2e8"
C_TEXT = "#1f2933"
C_MUTED = "#8a94a3"
C_ACCENT = "#2f6fde"
C_ACCENT_SOFT = "#e6efff"
C_DANGER = "#d64545"
C_DANGER_SOFT = "#fde9e9"
C_WARN = "#b7791f"
C_WARN_SOFT = "#fff3d1"
C_USED = "#3aa776"
C_UNUSED = "#cfd5dd"
# 棒グラフの系列の色（積み上げ）
SERIES_COLORS = (
    "#2f6fde",
    "#3aa776",
    "#e0a020",
    "#d64545",
    "#8a5cd6",
    "#2ba3b5",
    "#d6669e",
    "#7b8794",
    "#a0aec0",
)

# 意味のある値は、表と帯と同じ色で描く
VALUE_COLORS = {
    USED: C_USED,
    EXCLUDED_STOPPED: C_UNUSED,
    EXCLUDED_FLAG: "#a0aec0",
    EXCLUDED_MISMATCH: "#e0a020",
    EXCLUDED_BY_HAND: C_DANGER,
    NONE_TEXT: C_UNUSED,
    OTHERS_TEXT: "#a0aec0",
}

STYLE = f"""
QMainWindow, QWidget#page {{ background: {C_BG}; }}
QWidget {{ color: {C_TEXT}; font-family: "{FONT_FAMILY}"; font-size: 10pt; }}
QFrame#card {{
    background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 8px;
}}
QLabel#cardTitle {{ font-weight: 600; font-size: 10.5pt; }}
QLabel#muted {{ color: {C_MUTED}; }}
QLabel#count {{ color: {C_MUTED}; padding: 2px 4px; }}
QToolBar {{ background: {C_SURFACE}; border: none; border-bottom: 1px solid {C_BORDER};
    padding: 4px 8px; spacing: 6px; }}
QToolButton {{ padding: 4px 10px; border-radius: 6px; }}
QToolButton:hover {{ background: {C_ACCENT_SOFT}; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 18px; margin-right: 2px;
    border-bottom: 2px solid transparent; color: {C_MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ color: {C_ACCENT}; border-bottom: 2px solid {C_ACCENT}; }}
QPushButton {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 6px;
    padding: 5px 12px; }}
QPushButton:hover {{ border-color: {C_ACCENT}; color: {C_ACCENT}; }}
QPushButton:pressed {{ background: {C_ACCENT_SOFT}; }}
QPushButton#primary {{ background: {C_ACCENT}; border-color: {C_ACCENT}; color: white; }}
QPushButton#primary:hover {{ background: #285fc0; color: white; }}
QPushButton#danger {{ background: {C_DANGER}; border-color: {C_DANGER}; color: white; }}
QPushButton#danger:hover {{ background: #b93a3a; color: white; }}
QComboBox, QLineEdit {{ background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 6px; padding: 4px 8px; min-height: 20px; }}
QComboBox:focus, QLineEdit:focus {{ border-color: {C_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QTableView, QTableWidget {{ background: {C_SURFACE}; border: 1px solid {C_BORDER};
    border-radius: 6px; gridline-color: transparent;
    selection-background-color: {C_ACCENT_SOFT}; selection-color: {C_TEXT}; }}
QHeaderView::section {{ background: {C_SURFACE}; color: {C_MUTED}; font-weight: 600;
    border: none; border-bottom: 1px solid {C_BORDER}; padding: 6px 8px; }}
QListWidget#chips {{ background: transparent; border: none; }}
QListWidget#chips::item {{ background: {C_ACCENT_SOFT}; color: {C_ACCENT}; border-radius: 10px;
    padding: 2px 10px; margin: 2px; }}
QListWidget#chips::item:selected {{ background: {C_ACCENT}; color: white; }}
QPlainTextEdit {{ background: {C_SURFACE}; border: 1px solid {C_BORDER}; border-radius: 6px;
    font-family: "Consolas", "{FONT_FAMILY}"; font-size: 9pt; color: #3e4c59; }}
QSplitter::handle {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: {C_UNUSED}; border-radius: 5px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {C_UNUSED}; border-radius: 5px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def card(title: Optional[str] = None) -> Tuple[QFrame, QVBoxLayout]:
    """白い角丸の枠。見出しがあれば上に置く"""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(12, 10, 12, 12)
    layout.setSpacing(8)
    if title:
        label = QLabel(title)
        label.setObjectName("cardTitle")
        layout.addWidget(label)
    return frame, layout


def button(text: str, kind: str = "", slot: Optional[Callable] = None) -> QPushButton:
    b = QPushButton(text)
    if kind:
        b.setObjectName(kind)
    if slot is not None:
        b.clicked.connect(slot)
    b.setCursor(Qt.PointingHandCursor)
    return b


# ---- 絞り込みの条件 ----

_RANGE = re.compile(r"^\s*(-?[\d.]+)\s*\.\.\s*(-?[\d.]+)\s*$")
_COMPARE = re.compile(r"^\s*(<=|>=|<|>|=)?\s*(-?[\d.]+)\s*$")


class Condition:
    """表の列の値の条件。種類の列は値が同じか（タグなどは含むか）、数値の列は比較と範囲"""

    def __init__(self, key: str, negate: bool, text: str):
        self.key = key
        self.negate = negate
        self.text = text.strip()
        self.kind = COLUMN_BY_KEY[key].kind
        self._test = self._compile()

    def _compile(self) -> Callable[[Any], bool]:
        if self.text == NONE_TEXT:
            return lambda v: v is None or v == ""
        if self.kind == "num":
            m = _RANGE.match(self.text)
            if m:
                low, high = float(m.group(1)), float(m.group(2))
                return lambda v: isinstance(v, (int, float)) and low <= v <= high
            m = _COMPARE.match(self.text)
            if not m:
                raise ValueError(f"数値の条件が読めません: {self.text}（例: >10, <=5, 10..20）")
            op, num = m.group(1) or "=", float(m.group(2))
            ops = {
                "<": lambda v: v < num,
                "<=": lambda v: v <= num,
                ">": lambda v: v > num,
                ">=": lambda v: v >= num,
                "=": lambda v: math.isclose(v, num, abs_tol=1e-9),
            }
            check = ops[op]
            return lambda v: isinstance(v, (int, float)) and check(float(v))
        if self.key in MULTI_VALUE_KEYS:
            return lambda v: self.text in str(v or "").split(",")
        return lambda v: str(v if v is not None else "") == self.text

    def __call__(self, row: Dict[str, Any]) -> bool:
        return self._test(row.get(self.key)) != self.negate

    def label(self) -> str:
        return f"{COLUMN_BY_KEY[self.key].title} {'≠' if self.negate else '='} {self.text}  ✕"


# 「表示」の選択肢: 名前と、行が当てはまるか
QUICK_FILTERS: Tuple[Tuple[str, Optional[Callable[[Dict[str, Any]], bool]]], ...] = (
    ("すべて", None),
    ("注意のあるフレーム", lambda r: bool(r["warnings"])),
    ("手作業のラベルあり", lambda r: bool(r["tags"] or r["overrides"] or r["note"])),
    ("学習に使う", lambda r: r["training"] == USED),
    ("学習に使わない", lambda r: r["training"] != USED),
    ("手作業で除外", lambda r: r["training"] == EXCLUDED_BY_HAND),
)


def sort_key(value: Any) -> Tuple[int, Any]:
    """値のないものを最後に、数値は数値として並べる"""
    if value is None or value == "":
        return (2, "")
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


# ---- 表のモデル ----


class FrameModel(QAbstractTableModel):
    """絞り込みと並べ替えをモデルの中で行う（2万行でも速いように、Python の sorted を使う）"""

    def __init__(self, table: FrameTable):
        super().__init__()
        self.table = table
        self.columns = list(COLUMNS)
        self.conditions: List[Condition] = []
        self.quick: Optional[Callable[[Dict[str, Any]], bool]] = None
        self.sort_column: Optional[str] = None
        self.sort_order = Qt.AscendingOrder
        self.visible: List[int] = list(range(len(table)))
        self._where: Dict[int, int] = {r: i for i, r in enumerate(self.visible)}
        self._strike = QFont()
        self._strike.setStrikeOut(True)
        self._bold = QFont()
        self._bold.setBold(True)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.visible)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self.table.rows[self.visible[index.row()]]
        key = self.columns[index.column()].key
        value = row.get(key)
        if role == Qt.DisplayRole:
            if value is None:
                return ""
            if key == "warnings" and value:
                return "⚠ " + str(value).replace(",", "、")
            if key in ("tags", "overrides") and value:
                return str(value).replace(",", "、")
            return str(value)
        excluded = row["training"] == EXCLUDED_BY_HAND
        if role == Qt.BackgroundRole:
            if excluded:
                return QBrush(QColor(C_DANGER_SOFT))
            if row["warnings"]:
                return QBrush(QColor(C_WARN_SOFT))
            return None
        if role == Qt.ForegroundRole:
            if key == "warnings" and value:
                return QBrush(QColor(C_WARN))
            if key == "tags" and value:
                return QBrush(QColor(C_ACCENT))
            if excluded:
                return QBrush(QColor(C_DANGER))
            if row["training"] != USED:
                return QBrush(QColor(C_MUTED))
            return None
        if role == Qt.FontRole:
            if excluded:
                return self._strike
            if key in ("warnings", "tags") and value:
                return self._bold
            return None
        if role == Qt.TextAlignmentRole and self.columns[index.column()].kind == "num":
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ToolTipRole and key in ("warnings", "tags", "note") and value:
            return str(value).replace(",", "\n")
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.columns[section].title
        return None

    def sort(self, column: int, order=Qt.AscendingOrder) -> None:
        self.sort_column = self.columns[column].key if 0 <= column < len(self.columns) else None
        self.sort_order = order
        self.apply()

    def apply(self) -> None:
        """条件と並べ替えを掛け直す"""
        self.beginResetModel()
        rows = self.table.rows
        quick = self.quick
        visible = [
            i
            for i in range(len(rows))
            if (quick is None or quick(rows[i])) and all(c(rows[i]) for c in self.conditions)
        ]
        if self.sort_column:
            key = self.sort_column
            visible.sort(
                key=lambda i: sort_key(rows[i].get(key)),
                reverse=self.sort_order == Qt.DescendingOrder,
            )
        self.visible = visible
        self._where = {r: i for i, r in enumerate(visible)}
        self.endResetModel()

    def refresh_rows(self, table_rows: Sequence[int]) -> None:
        """手作業のラベルを変えた行の表示を更新する（絞り込みは掛け直さない）"""
        for r in table_rows:
            i = self._where.get(r)
            if i is not None:
                self.dataChanged.emit(self.index(i, 0), self.index(i, len(self.columns) - 1))

    def table_row(self, view_row: int) -> int:
        return self.visible[view_row]

    def view_row(self, table_row: int) -> Optional[int]:
        return self._where.get(table_row)


# ---- 画像 ----


class FrameImage(QLabel):
    """フレームの画像と、舵角の線（緑）"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 220)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background: #11161d; border-radius: 6px; color: {C_MUTED};")
        self._pixmap: Optional[QPixmap] = None
        self._steering_deg = 0.0

    def show_frame(self, path: Optional[Path], steering_deg: float) -> None:
        self._steering_deg = steering_deg
        self._pixmap = QPixmap(str(path)) if path and path.exists() else None
        if self._pixmap is None or self._pixmap.isNull():
            self._pixmap = None
            self.setText("画像がありません")
        self._render()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(scaled)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(40, 220, 90), 4, Qt.SolidLine, Qt.RoundCap))
        w, h = scaled.width(), scaled.height()
        base = QPointF(w / 2, h - 4)
        angle = math.radians(self._steering_deg)
        length = h * 0.35
        tip = QPointF(base.x() + length * math.sin(angle), base.y() - length * math.cos(angle))
        painter.drawLine(base, tip)
        painter.end()
        self.setPixmap(scaled)


# ---- セッションの流れ ----

# 帯の状態の色。重なる時は強いほう（後ろ）を描く
_STATUS_ORDER = ("unused", "used", "warning", "excluded")
_STATUS_COLOR = {
    "unused": C_UNUSED,
    "used": C_USED,
    "warning": "#e0a020",
    "excluded": C_DANGER,
}


def frame_status(row: Dict[str, Any]) -> str:
    if row["training"] == EXCLUDED_BY_HAND:
        return "excluded"
    if row["warnings"]:
        return "warning"
    if row["training"] == USED:
        return "used"
    return "unused"


class SessionTimeline(QWidget):
    """1セッションの全フレームの流れ。状態の帯、タグの印、舵角の線。ドラッグで範囲を選ぶ"""

    range_selected = Signal(int, int, int)  # セッションの番号, 始まり, 終わり（フレームの番号）

    BAND_TOP = 22
    BAND_H = 12
    TAG_H = 4

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(110)
        self.setMouseTracking(True)
        self.setCursor(Qt.IBeamCursor)
        self.table: Optional[FrameTable] = None
        self.session_index: Optional[int] = None
        self.rows: List[int] = []  # 表の行（フレームの順）
        self.current: Optional[int] = None  # フレームの番号
        self.selection: Optional[Tuple[int, int]] = None
        self._drag_start: Optional[int] = None
        self._hover: Optional[int] = None

    def set_session(self, table: FrameTable, session_index: int) -> None:
        self.table = table
        self.session_index = session_index
        self.rows = [r for r, (s, _) in enumerate(table.keys) if s == session_index]
        self.selection = None
        self.update()

    def set_current(self, position: Optional[int]) -> None:
        self.current = position
        self.update()

    def set_selection(self, selection: Optional[Tuple[int, int]]) -> None:
        self.selection = selection
        self.update()

    def _plot_rect(self) -> QRectF:
        # 右に、舵角の線の向きの文字を置く余白を残す
        return QRectF(8, 4, max(1, self.width() - 56), self.height() - 8)

    def _x(self, position: float) -> float:
        rect = self._plot_rect()
        n = max(1, len(self.rows))
        return rect.left() + rect.width() * position / n

    def _position_at(self, x: float) -> int:
        rect = self._plot_rect()
        n = len(self.rows)
        p = int((x - rect.left()) / rect.width() * n)
        return min(max(p, 0), max(0, n - 1))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self._plot_rect()
        painter.fillRect(self.rect(), QColor(C_SURFACE))
        if not self.rows or self.table is None:
            painter.setPen(QColor(C_MUTED))
            painter.drawText(
                rect, Qt.AlignCenter, "フレームを選ぶと、そのセッションの流れを表示します"
            )
            return
        rows = [self.table.rows[r] for r in self.rows]
        n = len(rows)
        width = int(rect.width())

        # 見出し
        session = self.table.sessions[self.session_index].session_dir.name
        painter.setPen(QColor(C_MUTED))
        text = f"{session}  ・  {n:,} フレーム  ・  ドラッグで範囲を選択"
        if self._hover is not None:
            hover = rows[self._hover]
            text += f"  ・  #{hover['index']}  {hover['time_s']} 秒  {hover['steering_deg']}°"
        painter.drawText(QRectF(rect.left(), 0, rect.width(), 18), Qt.AlignLeft, text)

        # 状態の帯とタグの印（1ピクセルに複数のフレームが入る時は強い状態を描く）
        band_top = self.BAND_TOP
        for x in range(width):
            lo = int(x * n / width)
            hi = max(lo + 1, int((x + 1) * n / width))
            chunk = rows[lo:hi]
            if not chunk:
                continue
            status = max((frame_status(r) for r in chunk), key=_STATUS_ORDER.index)
            painter.fillRect(
                QRectF(rect.left() + x, band_top, 1, self.BAND_H), QColor(_STATUS_COLOR[status])
            )
            if any(r["tags"] for r in chunk):
                painter.fillRect(
                    QRectF(rect.left() + x, band_top - self.TAG_H - 1, 1, self.TAG_H),
                    QColor(C_ACCENT),
                )

        # 舵角の線（上が右、下が左）
        trace_top = band_top + self.BAND_H + 8
        trace_h = rect.bottom() - trace_top
        mid = trace_top + trace_h / 2
        painter.setPen(QPen(QColor(C_BORDER), 1, Qt.DashLine))
        painter.drawLine(QPointF(rect.left(), mid), QPointF(rect.right(), mid))
        path = QPainterPath()
        step = max(1, n // (width * 2))
        for k in range(0, n, step):
            y = mid - (rows[k]["steering_deg"] or 0.0) / 35.0 * (trace_h / 2 - 2)
            x = self._x(k + 0.5)
            if k == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.setPen(QPen(QColor("#5b6b7f"), 1.2))
        painter.drawPath(path)
        painter.setPen(QColor(C_MUTED))
        # 線と重ならないよう、右端の外側に置く
        painter.drawText(QRectF(rect.right() + 6, trace_top - 2, 40, 14), Qt.AlignLeft, "右 ↑")
        painter.drawText(QRectF(rect.right() + 6, rect.bottom() - 13, 40, 14), Qt.AlignLeft, "左 ↓")

        # 選んだ範囲と今のフレーム
        if self.selection is not None:
            a, b = self.selection
            x0, x1 = self._x(a), self._x(b + 1)
            painter.fillRect(
                QRectF(x0, band_top - 6, max(2.0, x1 - x0), rect.bottom() - band_top + 6),
                QColor(47, 111, 222, 45),
            )
            painter.setPen(QPen(QColor(C_ACCENT), 1))
            painter.drawRect(
                QRectF(x0, band_top - 6, max(2.0, x1 - x0), rect.bottom() - band_top + 6)
            )
        if self.current is not None:
            x = self._x(self.current + 0.5)
            painter.setPen(QPen(QColor(C_TEXT), 2))
            painter.drawLine(QPointF(x, band_top - 6), QPointF(x, rect.bottom()))

    def mousePressEvent(self, event) -> None:
        if self.rows and event.button() == Qt.LeftButton:
            self._drag_start = self._position_at(event.position().x())
            self.set_selection((self._drag_start, self._drag_start))

    def mouseMoveEvent(self, event) -> None:
        if not self.rows:
            return
        p = self._position_at(event.position().x())
        self._hover = p
        if self._drag_start is not None:
            self.set_selection((min(self._drag_start, p), max(self._drag_start, p)))
        else:
            self.update()

    def leaveEvent(self, event) -> None:
        self._hover = None
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None or self.session_index is None:
            return
        p = self._position_at(event.position().x())
        a, b = min(self._drag_start, p), max(self._drag_start, p)
        self._drag_start = None
        self.range_selected.emit(self.session_index, a, b)


# ---- 統計の表の割合の棒 ----


class ShareDelegate(QStyledItemDelegate):
    """割合の列に、値の長さの棒を敷く"""

    def paint(self, painter, option, index) -> None:
        share = index.data(Qt.UserRole)
        if isinstance(share, float):
            r = option.rect.adjusted(6, 6, -6, -6)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(C_ACCENT_SOFT))
            painter.drawRoundedRect(QRectF(r), 3, 3)
            painter.setBrush(QColor("#9dbcf2"))
            painter.drawRoundedRect(QRectF(r.left(), r.top(), r.width() * share, r.height()), 3, 3)
            painter.restore()
        super().paint(painter, option, index)


# ---- 画面 ----


class DataViewer(QMainWindow):
    def __init__(self, data_dirs: Sequence[Path], settings: Optional[Settings] = None):
        super().__init__()
        self.setWindowTitle("PiCar-X 走行データ")
        self.resize(1600, 960)
        # INI 形式にする（テストで置き場所を変えられ、利用者の設定に触れない）
        self.qsettings = QSettings(
            QSettings.IniFormat, QSettings.UserScope, "PiCarX", "data_viewer"
        )
        self.data_dirs = [Path(d) for d in data_dirs]
        self.label_settings = settings or load_label_settings()
        self.store = AnnotationStore()
        self.undo_stack: List[List[Tuple[Any, str, Optional[Dict[str, Any]]]]] = []
        self.table = FrameTable(self.data_dirs, self.label_settings, self.store)
        self.model = FrameModel(self.table)
        self.range_start: Optional[int] = None  # I で決めた範囲の始まり（表の行）
        self._stats_drill: Optional[Tuple[str, Optional[str]]] = None

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_frames_tab(), "フレーム")
        self.tabs.addTab(self._build_stats_tab(), "統計")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)
        self._build_toolbar()

        self.play_timer = QTimer(self)
        self.play_timer.setInterval(PLAY_INTERVAL_MS)
        self.play_timer.timeout.connect(self._step)
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_play)
        QShortcut(QKeySequence("X"), self, activated=self.toggle_exclude)
        QShortcut(QKeySequence("I"), self, activated=self.mark_range_start)
        QShortcut(QKeySequence("O"), self, activated=self.mark_range_end)
        QShortcut(QKeySequence.Undo, self, activated=self.undo)

        self._restore_columns()
        self._after_filter()

    # ---- 構成 ----

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("データ")
        bar.setMovable(False)
        for text, slot in (
            ("フォルダを追加…", self.add_folder),
            ("読み込み直す", self.reload),
            ("元に戻す (Ctrl+Z)", self.undo),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            bar.addAction(action)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        bar.addWidget(spacer)
        self.source_label = QLabel()
        self.source_label.setObjectName("muted")
        bar.addWidget(self.source_label)
        self._update_source_label()

    def _build_frames_tab(self) -> QWidget:
        # 絞り込み
        self.quick_filter = QComboBox()
        for name, _ in QUICK_FILTERS:
            self.quick_filter.addItem(name)
        self.quick_filter.currentIndexChanged.connect(self._on_quick_filter)
        self.filter_column = QComboBox()
        for column in COLUMNS:
            self.filter_column.addItem(column.title, column.key)
        self.filter_column.currentIndexChanged.connect(self._fill_filter_values)
        self.filter_op = QComboBox()
        self.filter_op.addItems(["=", "≠"])
        self.filter_value = QComboBox()
        self.filter_value.setEditable(True)
        self.filter_value.setMinimumWidth(200)
        self.filter_value.lineEdit().returnPressed.connect(self.add_condition)
        self.condition_list = QListWidget()
        self.condition_list.setObjectName("chips")
        self.condition_list.setFlow(QListWidget.LeftToRight)
        self.condition_list.setWrapping(True)
        self.condition_list.setMaximumHeight(34)
        self.condition_list.setToolTip("クリックした条件を外します")
        self.condition_list.itemClicked.connect(self._remove_condition_item)
        self.count_label = QLabel()
        self.count_label.setObjectName("count")

        filter_card, filter_layout = card()
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("表示"))
        row1.addWidget(self.quick_filter)
        row1.addSpacing(16)
        row1.addWidget(QLabel("条件"))
        row1.addWidget(self.filter_column)
        row1.addWidget(self.filter_op)
        row1.addWidget(self.filter_value, 1)
        row1.addWidget(button("追加", "primary", self.add_condition))
        row1.addWidget(button("すべて外す", "", self.clear_conditions))
        filter_layout.addLayout(row1)
        row2 = QHBoxLayout()
        row2.addWidget(self.condition_list, 1)
        row2.addWidget(self.count_label)
        filter_layout.addLayout(row2)
        filter_layout.addWidget(self._legend())

        # 表
        self.view = QTableView()
        self.view.setModel(self.model)
        # 並べ替えを有効にすると今の見出しで並べ替わるので、最初は記録の順のままにする
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.view.setSortingEnabled(True)
        self.view.setShowGrid(False)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(24)
        header = self.view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setDefaultSectionSize(90)
        header.setHighlightSections(False)
        header.setContextMenuPolicy(Qt.CustomContextMenu)
        header.customContextMenuRequested.connect(self._column_menu)
        for key, width in (("session", 130), ("warnings", 190), ("tags", 120), ("training", 110)):
            header.resizeSection([c.key for c in self.model.columns].index(key), width)
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self.view.selectionModel().currentRowChanged.connect(self._on_selection)

        self.timeline = SessionTimeline()
        self.timeline.range_selected.connect(self.select_frame_range)
        timeline_card, timeline_layout = card()
        timeline_layout.setContentsMargins(6, 6, 6, 6)
        timeline_layout.addWidget(self.timeline)

        table_card, table_layout = card()
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self.view)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 6, 12)
        left_layout.setSpacing(10)
        left_layout.addWidget(filter_card)
        left_layout.addWidget(table_card, 1)
        left_layout.addWidget(timeline_card)

        # 画像、ラベル、記録
        self.image = FrameImage()
        self.frame_title = QLabel("フレームを選んでください")
        self.frame_title.setObjectName("cardTitle")
        self.frame_info = QLabel("")
        self.frame_info.setObjectName("muted")
        self.frame_info.setWordWrap(True)
        image_card, image_layout = card()
        image_layout.addWidget(self.image, 1)
        image_layout.addWidget(self.frame_title)
        image_layout.addWidget(self.frame_info)

        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        details_card, details_layout = card("記録の中身")
        details_layout.addWidget(self.details)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 12, 12, 12)
        right_layout.setSpacing(10)
        right_layout.addWidget(image_card, 5)
        right_layout.addWidget(self._build_editor())
        right_layout.addWidget(details_card, 3)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([1000, 600])
        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self._fill_filter_values()
        return page

    def _legend(self) -> QWidget:
        """表と帯の色の見かた"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)
        for color, text in (
            (C_USED, "学習に使う"),
            (C_UNUSED, "使わない（止まっている等）"),
            ("#e0a020", "注意あり（確かめたい）"),
            (C_DANGER, "手作業で除外"),
            (C_ACCENT, "タグあり"),
        ):
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background: {color}; border-radius: 3px;")
            label = QLabel(text)
            label.setObjectName("muted")
            layout.addWidget(swatch)
            layout.addWidget(label)
        layout.addStretch(1)
        hint = QLabel("Shift+クリック / 下の帯をドラッグ / I と O で範囲を選択")
        hint.setObjectName("muted")
        layout.addWidget(hint)
        return widget

    def _build_editor(self) -> QWidget:
        frame, layout = card("ラベル（選んだすべてのフレームに付きます）")
        self.selection_label = QLabel("フレームを選んでください")
        self.selection_label.setObjectName("muted")
        layout.addWidget(self.selection_label)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        self.tag_input = QComboBox()
        self.tag_input.setEditable(True)
        self.tag_input.lineEdit().setPlaceholderText("例: 脚, 布団, ぶれ, 要確認")
        self.tag_input.lineEdit().returnPressed.connect(self.add_tag)
        grid.addWidget(QLabel("タグ"), 0, 0)
        grid.addWidget(self.tag_input, 0, 1)
        grid.addWidget(button("付ける", "primary", self.add_tag), 0, 2)
        grid.addWidget(button("外す", "", self.remove_tag), 0, 3)

        grid.addWidget(QLabel("学習"), 1, 0)
        grid.addWidget(button("除外する (X)", "danger", lambda: self.set_exclude(True)), 1, 1, 1, 2)
        grid.addWidget(button("除外を戻す", "", lambda: self.set_exclude(False)), 1, 3)

        self.intent_box = QComboBox()
        self.intent_box.addItem(AUTO_TEXT, None)
        for intent in INTENTS:
            self.intent_box.addItem(intent, intent)
        grid.addWidget(QLabel("意図"), 2, 0)
        grid.addWidget(self.intent_box, 2, 1, 1, 2)
        grid.addWidget(button("設定", "", self.apply_intent), 2, 3)

        self.blocked_box = QComboBox()
        self.blocked_box.addItem(AUTO_TEXT, "auto")
        self.blocked_box.addItem("塞がり", True)
        self.blocked_box.addItem("空き", False)
        self.blocked_box.addItem("不明", None)
        grid.addWidget(QLabel("前方"), 3, 0)
        grid.addWidget(self.blocked_box, 3, 1, 1, 2)
        grid.addWidget(button("設定", "", self.apply_blocked), 3, 3)

        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("自由なメモ")
        grid.addWidget(QLabel("メモ"), 4, 0)
        grid.addWidget(self.note_input, 4, 1, 1, 2)
        grid.addWidget(button("保存", "", self.apply_note), 4, 3)
        layout.addLayout(grid)

        self._refresh_tag_choices()
        return frame

    def _build_stats_tab(self) -> QWidget:
        self.stats_field = QComboBox()
        self.stats_cross = QComboBox()
        self.stats_cross.addItem("(なし)", None)
        for column in COLUMNS:
            self.stats_field.addItem(column.title, column.key)
            if column.kind == "cat":
                self.stats_cross.addItem(column.title, column.key)
        self.stats_field.setCurrentIndex(self.stats_field.findData("training"))
        self.stats_field.currentIndexChanged.connect(self.update_stats)
        self.stats_cross.currentIndexChanged.connect(self.update_stats)

        controls_card, controls_layout = card()
        controls = QHBoxLayout()
        controls.addWidget(QLabel("集計する列"))
        controls.addWidget(self.stats_field)
        controls.addSpacing(16)
        controls.addWidget(QLabel("組み合わせる列"))
        controls.addWidget(self.stats_cross)
        controls.addStretch(1)
        controls_layout.addLayout(controls)
        self.stats_summary = QLabel()
        self.stats_summary.setObjectName("muted")
        self.stats_summary.setWordWrap(True)
        controls_layout.addWidget(self.stats_summary)

        self.chart = QChart()
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.chart_view.setStyleSheet("background: transparent;")
        chart_card, chart_layout = card()
        chart_layout.addWidget(self.chart_view)

        self.stats_table = QTableWidget()
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stats_table.setShowGrid(False)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.verticalHeader().setDefaultSectionSize(26)
        self.stats_table.cellDoubleClicked.connect(self._stats_drill_down)
        self.share_delegate = ShareDelegate(self.stats_table)
        table_card, table_layout = card()
        table_layout.addWidget(self.stats_table, 1)
        hint = QLabel("行をダブルクリックすると、その値で絞り込んでフレームの一覧へ戻ります")
        hint.setObjectName("muted")
        table_layout.addWidget(hint)

        lower = QSplitter()
        lower.addWidget(chart_card)
        lower.addWidget(table_card)
        lower.setSizes([950, 600])

        page = QWidget()
        page.setObjectName("page")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(controls_card)
        layout.addWidget(lower, 1)
        return page

    # ---- 列の表示 ----

    def _column_menu(self, pos) -> None:
        menu = QMenu(self)
        header = self.view.horizontalHeader()
        for i, column in enumerate(self.model.columns):
            action = menu.addAction(column.title)
            action.setCheckable(True)
            action.setChecked(not header.isSectionHidden(i))
            action.toggled.connect(lambda checked, i=i: self._set_column_visible(i, checked))
        menu.addSeparator()
        menu.addAction("最初の表示に戻す", self._default_columns)
        menu.exec(header.mapToGlobal(pos))

    def _set_column_visible(self, i: int, visible: bool) -> None:
        self.view.horizontalHeader().setSectionHidden(i, not visible)
        self._save_columns()

    def _default_columns(self) -> None:
        for i, column in enumerate(self.model.columns):
            self.view.horizontalHeader().setSectionHidden(i, not column.visible)
        self._save_columns()

    def _save_columns(self) -> None:
        # 列ごとに表示するかを残す。後から増えた列は、その列の最初の表示に従う
        header = self.view.horizontalHeader()
        visible = {c.key: not header.isSectionHidden(i) for i, c in enumerate(self.model.columns)}
        self.qsettings.setValue("column_visible", json.dumps(visible))

    def _restore_columns(self) -> None:
        try:
            saved = json.loads(self.qsettings.value("column_visible", "{}") or "{}")
        except (TypeError, ValueError):
            saved = {}
        if not isinstance(saved, dict):
            saved = {}
        for i, column in enumerate(self.model.columns):
            visible = bool(saved.get(column.key, column.visible))
            self.view.horizontalHeader().setSectionHidden(i, not visible)

    # ---- 絞り込み ----

    def _fill_filter_values(self) -> None:
        key = self.filter_column.currentData()
        self.filter_value.clear()
        if COLUMN_BY_KEY[key].kind == "num":
            self.filter_value.lineEdit().setPlaceholderText("例: >10, <=5, 10..20, (なし)")
            self.filter_value.addItem(NONE_TEXT)
        else:
            self.filter_value.lineEdit().setPlaceholderText("値を選ぶか入力")
            values = [v for v, _ in count_by(self.table.rows, key)][:300]
            self.filter_value.addItems(values)
        self.filter_value.setCurrentText("")

    def _on_quick_filter(self, index: int) -> None:
        self.model.quick = QUICK_FILTERS[index][1]
        self.model.apply()
        self._after_filter()

    def add_condition(self) -> None:
        text = self.filter_value.currentText().strip()
        if not text:
            return
        try:
            condition = Condition(
                self.filter_column.currentData(), self.filter_op.currentText() == "≠", text
            )
        except ValueError as e:
            QMessageBox.warning(self, "絞り込み", str(e))
            return
        self._push_condition(condition)
        self.model.apply()
        self._after_filter()

    def _push_condition(self, condition: Condition) -> None:
        self.model.conditions.append(condition)
        self.condition_list.addItem(QListWidgetItem(condition.label()))

    def _remove_condition_item(self, item: QListWidgetItem) -> None:
        row = self.condition_list.row(item)
        self.condition_list.takeItem(row)
        del self.model.conditions[row]
        self.model.apply()
        self._after_filter()

    def remove_condition(self) -> None:
        item = self.condition_list.currentItem()
        if item is not None:
            self._remove_condition_item(item)

    def clear_conditions(self) -> None:
        self.condition_list.clear()
        self.model.conditions.clear()
        self.model.apply()
        self._after_filter()

    def _after_filter(self) -> None:
        rows = self.visible_rows()
        used = sum(1 for r in rows if r["training"] == USED)
        warned = sum(1 for r in rows if r["warnings"])
        self.count_label.setText(
            f"{len(rows):,} / {len(self.table):,} フレーム ・ 学習に使う {used:,}"
            f" ・ 注意 {warned:,}"
        )
        if self.tabs.currentIndex() == 1:
            self.update_stats()

    def visible_rows(self) -> List[Dict[str, Any]]:
        return [self.table.rows[i] for i in self.model.visible]

    # ---- 選択と表示 ----

    def selected_table_rows(self) -> List[int]:
        rows = sorted({i.row() for i in self.view.selectionModel().selectedRows()})
        return [self.model.table_row(r) for r in rows]

    def current_table_row(self) -> Optional[int]:
        index = self.view.currentIndex()
        return self.model.table_row(index.row()) if index.isValid() else None

    def _on_selection(self, *_) -> None:
        selected = self.selected_table_rows()
        current = self.current_table_row()
        if current is None and selected:
            current = selected[0]
        self._describe_selection(selected)
        if current is None:
            return
        row = self.table.rows[current]
        self.image.show_frame(self.table.image_path(current), float(row["steering_deg"] or 0.0))
        self.frame_title.setText(f"{row['session']}  #{row['index']}  ・  {row['time_s']} 秒")
        info = (
            f"{row['motion']} ・ 舵 {row['steering_deg']}° ・ 意図 {row['intent']}"
            f" ・ 距離 {row['distance_cm'] if row['distance_cm'] is not None else '-'} cm"
            f" ・ 前方 {row['blocked']} ・ 学習 {row['training']}"
        )
        if row["warnings"]:
            info += f"\n⚠ {row['warnings'].replace(',', '、')}"
        if row["tags"]:
            info += f"\nタグ: {row['tags'].replace(',', '、')}"
        self.frame_info.setText(info)
        note = self.table.note(current)
        self.note_input.setText(note.get("note", ""))
        record = self.table.record(current)
        self.details.setPlainText(
            "手作業のラベル: "
            + json.dumps(note, ensure_ascii=False)
            + "\n\n"
            + json.dumps(record, ensure_ascii=False, indent=1)
        )
        session_index, position = self.table.keys[current]
        if self.timeline.session_index != session_index or self.timeline.table is not self.table:
            self.timeline.set_session(self.table, session_index)
        self.timeline.set_current(position)
        self._show_selection_on_timeline(selected, session_index)

    def _describe_selection(self, selected: List[int]) -> None:
        if not selected:
            self.selection_label.setText("フレームを選んでください")
            return
        groups = self.table.by_session(selected)
        if len(groups) == 1:
            ((s, rows),) = groups.items()
            positions = [self.table.keys[r][1] for r in rows]
            first = self.table.rows[rows[0]]
            text = (
                f"{len(selected):,} フレームを選択中 ・ {first['session']}"
                f" の #{self.table.record(self._row_at(s, min(positions)))['index']}"
                f"〜#{self.table.record(self._row_at(s, max(positions)))['index']}"
            )
        else:
            text = f"{len(selected):,} フレームを選択中 ・ {len(groups)} セッション"
        if self.range_start is not None:
            text += " ・ 範囲の始まりを決めています（O で終わり）"
        self.selection_label.setText(text)

    def _row_at(self, session_index: int, position: int) -> int:
        """セッションの番号とフレームの番号から、表の行"""
        return self.timeline_rows(session_index)[position]

    def timeline_rows(self, session_index: int) -> List[int]:
        if self.timeline.session_index == session_index and self.timeline.table is self.table:
            return self.timeline.rows
        return [r for r, (s, _) in enumerate(self.table.keys) if s == session_index]

    def _show_selection_on_timeline(self, selected: List[int], session_index: int) -> None:
        positions = [
            self.table.keys[r][1] for r in selected if self.table.keys[r][0] == session_index
        ]
        if len(positions) > 1:
            self.timeline.set_selection((min(positions), max(positions)))
        elif self.timeline._drag_start is None:
            self.timeline.set_selection(None)

    def select_frame_range(self, session_index: int, start: int, end: int) -> None:
        """セッションのフレーム（番号 start〜end）を表で選ぶ。絞り込みで隠れた行は選べない"""
        rows = self.timeline_rows(session_index)[start : end + 1]
        view_rows = sorted(v for v in (self.model.view_row(r) for r in rows) if v is not None)
        hidden = len(rows) - len(view_rows)
        selection = QItemSelection()
        last = len(self.model.columns) - 1
        run_start = prev = None
        for v in view_rows + [None]:
            if v is not None and prev is not None and v == prev + 1:
                prev = v
                continue
            if run_start is not None:
                selection.select(self.model.index(run_start, 0), self.model.index(prev, last))
            run_start = prev = v
        model = self.view.selectionModel()
        model.select(selection, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        if view_rows:
            current = self.model.view_row(rows[0])
            if current is not None:
                model.setCurrentIndex(self.model.index(current, 0), QItemSelectionModel.NoUpdate)
                self.view.scrollTo(self.model.index(current, 0))
        self.timeline.set_selection((start, end))
        if hidden:
            self.statusBar().showMessage(
                f"絞り込みで隠れている {hidden:,} フレームは選ばれていません", 6000
            )
        self._on_selection()

    def mark_range_start(self) -> None:
        self.range_start = self.current_table_row()
        self._on_selection()
        if self.range_start is not None:
            self.statusBar().showMessage(
                "範囲の始まりを決めました。終わりのフレームで O を押します", 6000
            )

    def mark_range_end(self) -> None:
        end = self.current_table_row()
        start, self.range_start = self.range_start, None
        if start is None or end is None:
            return
        (s0, p0), (s1, p1) = self.table.keys[start], self.table.keys[end]
        if s0 != s1:
            QMessageBox.information(
                self, "範囲", "範囲の始まりと終わりは同じセッションで選んでください"
            )
            return
        self.select_frame_range(s0, min(p0, p1), max(p0, p1))

    def _step(self) -> None:
        index = self.view.currentIndex()
        nxt = (index.row() + 1) if index.isValid() else 0
        if nxt >= self.model.rowCount():
            self.play_timer.stop()
            return
        self.view.selectRow(nxt)

    def toggle_play(self) -> None:
        typing = self.focusWidget() in (self.tag_input.lineEdit(), self.note_input)
        if self.tabs.currentIndex() != 0 or typing:
            return
        if self.play_timer.isActive():
            self.play_timer.stop()
        else:
            self.play_timer.start()

    # ---- ラベルの変更 ----

    def _change(self, rows: Sequence[int], apply: Callable[[Any, List[str]], int]) -> int:
        """選んだ行の注釈を変えて保存し、元に戻せるよう前の値を残す"""
        if not rows:
            QMessageBox.information(self, "ラベル", "フレームを選んでください")
            return 0
        before = []
        for notes, frames in self.table.frames(rows):
            for frame in frames:
                entry = notes.frames.get(frame)
                before.append((notes, frame, copy.deepcopy(entry)))
        changed = 0
        for notes, frames in self.table.frames(rows):
            changed += apply(notes, frames)
        if changed:
            self.undo_stack.append(before)
            self._commit(rows)
            self.statusBar().showMessage(f"{changed:,} フレームを変更して保存しました", 4000)
        return changed

    def _commit(self, rows: Sequence[int]) -> None:
        try:
            self.store.save()
        except OSError as e:
            QMessageBox.critical(self, "保存できません", str(e))
        self.table.refresh(rows)
        self.model.refresh_rows(rows)
        self._refresh_tag_choices()
        self.timeline.update()
        self._on_selection()
        self._after_filter()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        before = self.undo_stack.pop()
        touched = set()
        for notes, frame, entry in before:
            if entry is None:
                notes.frames.pop(frame, None)
            else:
                notes.frames[frame] = entry
            notes.dirty = True
            touched.add((notes.session_id, frame))
        rows = [
            r
            for r in range(len(self.table))
            if (self.table.session_of(r).session_dir.name, self.table.record(r).get("frame"))
            in touched
        ]
        self._commit(rows)
        self.statusBar().showMessage(f"{len(rows):,} フレームを元に戻しました", 4000)

    def add_tag(self) -> None:
        tag = self.tag_input.currentText().strip()
        if tag:
            self._change(self.selected_table_rows(), lambda n, f: n.add_tag(f, tag))

    def remove_tag(self) -> None:
        tag = self.tag_input.currentText().strip()
        if tag:
            self._change(self.selected_table_rows(), lambda n, f: n.remove_tag(f, tag))

    def set_exclude(self, exclude: bool) -> None:
        self._change(self.selected_table_rows(), lambda n, f: n.set_field(f, "exclude", exclude))

    def toggle_exclude(self) -> None:
        rows = self.selected_table_rows()
        if not rows:
            return
        exclude = not self.table.note(rows[0]).get("exclude", False)
        self.set_exclude(exclude)

    def apply_intent(self) -> None:
        value = self.intent_box.currentData()
        if value is None:
            self._change(self.selected_table_rows(), lambda n, f: n.clear_field(f, "intent"))
        else:
            self._change(self.selected_table_rows(), lambda n, f: n.set_field(f, "intent", value))

    def apply_blocked(self) -> None:
        value = self.blocked_box.currentData()
        if value == "auto":
            self._change(self.selected_table_rows(), lambda n, f: n.clear_field(f, "blocked"))
        else:
            self._change(self.selected_table_rows(), lambda n, f: n.set_field(f, "blocked", value))

    def apply_note(self) -> None:
        text = self.note_input.text()
        self._change(self.selected_table_rows(), lambda n, f: n.set_field(f, "note", text))

    def _refresh_tag_choices(self) -> None:
        current = self.tag_input.currentText()
        self.tag_input.clear()
        self.tag_input.addItems(self.store.all_tags())
        self.tag_input.setCurrentText(current)

    # ---- 統計 ----

    def _on_tab_changed(self, index: int) -> None:
        self.play_timer.stop()
        if index == 1:
            self.update_stats()

    def update_stats(self) -> None:
        rows = self.visible_rows()
        key = self.stats_field.currentData()
        cross = self.stats_cross.currentData()
        total = len(rows)
        used = sum(1 for r in rows if r["training"] == USED)
        warned = sum(1 for r in rows if r["warnings"])
        by_hand = sum(1 for r in rows if r["overrides"] or r["tags"])
        filters = [c.label().rstrip(" ✕") for c in self.model.conditions]
        if self.model.quick is not None:
            filters.insert(0, self.quick_filter.currentText())
        text = (
            f"対象 {total:,} フレーム ・ 学習に使う {used:,} ・ 注意 {warned:,}"
            f" ・ 手作業のラベルあり {by_hand:,} ・ 絞り込み: {'、'.join(filters) or 'なし'}"
        )
        column = COLUMN_BY_KEY[key]
        if column.kind == "num" and not cross:
            s = summary(rows, key)
            text += (
                f"\n{column.title}: 値あり {s['count']:,} ・ 最小 {s['min']}"
                f" ・ 中央値 {s['median']} ・ 平均 {s['mean']} ・ 最大 {s['max']}"
            )
        self.stats_summary.setText(text)

        if cross:
            self._show_crosstab(rows, key, cross)
        elif column.kind == "num":
            bins, missing = histogram(rows, key, bins=HISTOGRAM_BINS)
            labels = [f"{low:.0f}〜{high:.0f}" for low, high, _ in bins]
            counts = [n for _, _, n in bins]
            if missing:
                labels.append(NONE_TEXT)
                counts.append(missing)
            self._fill_count_table(column.title, labels, counts, total, drill=False)
            self._draw_histogram(column.title, labels, counts)
        else:
            pairs = count_by(rows, key)
            labels = [p[0] for p in pairs]
            counts = [p[1] for p in pairs]
            self._fill_count_table(column.title, labels, counts, total)
            self._draw_horizontal(column.title, *self._top(labels, counts))

    @staticmethod
    def _top(labels: List[str], counts: List[int]) -> Tuple[List[str], List[int]]:
        """多すぎる種類は、多い順に MAX_BARS 個まで出し、残りを「その他」にまとめる"""
        if len(labels) <= MAX_BARS:
            return labels, counts
        return labels[: MAX_BARS - 1] + [OTHERS_TEXT], counts[: MAX_BARS - 1] + [
            sum(counts[MAX_BARS - 1 :])
        ]

    def _fill_count_table(
        self, title: str, labels: List[str], counts: List[int], total: int, drill: bool = True
    ) -> None:
        self._stats_drill = (self.stats_field.currentData(), None) if drill else None
        table = self.stats_table
        table.clear()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels([title, "フレーム数", "割合"])
        table.setRowCount(len(labels))
        for i, (label, n) in enumerate(zip(labels, counts)):
            table.setItem(i, 0, QTableWidgetItem(label))
            count_item = QTableWidgetItem(f"{n:,}")
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(i, 1, count_item)
            share = n / total if total else 0.0
            share_item = QTableWidgetItem(f"{100 * share:.1f}%")
            share_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            share_item.setData(Qt.UserRole, float(share))
            table.setItem(i, 2, share_item)
        table.setItemDelegateForColumn(2, self.share_delegate)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.resizeSection(2, 160)

    def _show_crosstab(self, rows: List[Dict[str, Any]], key: str, cross: str) -> None:
        self._stats_drill = (key, cross)
        row_labels, col_labels, counts = crosstab(rows, key, cross)
        table = self.stats_table
        table.clear()
        table.setItemDelegateForColumn(2, None)
        shown_cols = col_labels[:MAX_STACKS]
        table.setColumnCount(len(shown_cols) + 2)
        table.setHorizontalHeaderLabels([COLUMN_BY_KEY[key].title] + shown_cols + ["計"])
        table.setRowCount(len(row_labels))
        for i, a in enumerate(row_labels):
            table.setItem(i, 0, QTableWidgetItem(a))
            row_total = sum(counts.get((a, b), 0) for b in col_labels)
            for j, b in enumerate(shown_cols):
                n = counts.get((a, b), 0)
                item = QTableWidgetItem(f"{n:,}" if n else "")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, j + 1, item)
            total_item = QTableWidgetItem(f"{row_total:,}")
            total_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(i, len(shown_cols) + 1, total_item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for j in range(1, len(shown_cols) + 2):
            header.setSectionResizeMode(j, QHeaderView.ResizeToContents)

        # グラフは行の種類ごとの積み上げ横棒。多すぎる種類は「その他」にまとめる
        bar_rows = row_labels[:MAX_BARS]
        series: Dict[str, List[int]] = {}
        for b in shown_cols:
            series[b] = [counts.get((a, b), 0) for a in bar_rows]
        rest = col_labels[MAX_STACKS:]
        if rest:
            series[OTHERS_TEXT] = [sum(counts.get((a, b), 0) for b in rest) for a in bar_rows]
        self._draw_horizontal(
            f"{COLUMN_BY_KEY[key].title} × {COLUMN_BY_KEY[cross].title}", bar_rows, series
        )

    # ---- グラフ ----

    def _new_chart(self, title: str) -> QChart:
        # 軸を外して付け直すと古い軸が残ることがあるので、グラフごと作り直す
        old = self.chart
        chart = QChart()
        chart.setTitle(title)
        title_font = QFont(FONT_FAMILY, 11)
        title_font.setBold(True)
        chart.setTitleFont(title_font)
        chart.setTitleBrush(QColor(C_TEXT))
        chart.setBackgroundVisible(False)
        chart.setMargins(QMargins(4, 4, 4, 4))
        chart.setAnimationOptions(QChart.NoAnimation)
        chart.legend().setAlignment(Qt.AlignBottom)
        chart.legend().setFont(QFont(FONT_FAMILY, 9))
        self.chart = chart
        self.chart_view.setChart(chart)
        old.deleteLater()
        return chart

    @staticmethod
    def _style_axis(axis, grid: bool) -> None:
        axis.setLabelsFont(QFont(FONT_FAMILY, 9))
        axis.setLabelsColor(QColor("#52606d"))
        axis.setLinePenColor(QColor(C_BORDER))
        axis.setGridLineVisible(grid)
        axis.setGridLineColor(QColor("#edf0f4"))

    def _draw_horizontal(self, title: str, labels: List[str], series_counts: Any) -> None:
        """種類の数の横棒（多い順に上から）。series_counts は数の列か、系列名→数の列"""
        chart = self._new_chart(title)
        if isinstance(series_counts, dict):
            series = QHorizontalStackedBarSeries()
            groups = series_counts
        else:
            series = QHorizontalBarSeries()
            groups = {"": series_counts}
        # 横棒は下から描くので、多い順に上から並ぶよう逆にする
        labels = list(reversed(labels))
        top = 0.0
        for k, (name, counts) in enumerate(groups.items()):
            bar_set = QBarSet(name or title)
            bar_set.append([float(n) for n in reversed(counts)])
            color = QColor(VALUE_COLORS.get(name, SERIES_COLORS[k % len(SERIES_COLORS)]))
            bar_set.setColor(color)
            bar_set.setBorderColor(color)
            bar_set.setLabelColor(QColor(C_TEXT))
            series.append(bar_set)
        if isinstance(series_counts, dict):
            totals = [sum(c[i] for c in groups.values()) for i in range(len(labels))]
            top = max(totals, default=0)
        else:
            top = max(series_counts, default=0)
            series.setLabelsVisible(True)
            series.setLabelsPosition(QHorizontalBarSeries.LabelsOutsideEnd)
            series.setLabelsFormat("@value")
        series.setBarWidth(0.7)
        chart.addSeries(series)
        y_axis = QBarCategoryAxis()
        y_axis.append([self._short(label) for label in labels])
        x_axis = QValueAxis()
        x_axis.setRange(0, max(1.0, top) * 1.12)
        x_axis.setLabelFormat("%d")
        x_axis.applyNiceNumbers()
        self._style_axis(y_axis, grid=False)
        self._style_axis(x_axis, grid=True)
        chart.addAxis(y_axis, Qt.AlignLeft)
        chart.addAxis(x_axis, Qt.AlignBottom)
        series.attachAxis(y_axis)
        series.attachAxis(x_axis)
        chart.legend().setVisible(isinstance(series_counts, dict))

    def _draw_histogram(self, title: str, labels: List[str], counts: List[int]) -> None:
        chart = self._new_chart(title)
        series = QBarSeries()
        bar_set = QBarSet(title)
        bar_set.append([float(n) for n in counts])
        bar_set.setColor(QColor(C_ACCENT))
        bar_set.setBorderColor(QColor(C_ACCENT))
        series.append(bar_set)
        series.setBarWidth(0.9)
        chart.addSeries(series)
        x_axis = QBarCategoryAxis()
        # 区間の始まりだけを出して、ラベルが重ならないようにする
        x_axis.append([label.split("〜")[0] if "〜" in label else label for label in labels])
        y_axis = QValueAxis()
        y_axis.setRange(0, max(1, max(counts, default=0)) * 1.08)
        y_axis.setLabelFormat("%d")
        y_axis.applyNiceNumbers()
        self._style_axis(x_axis, grid=False)
        self._style_axis(y_axis, grid=True)
        x_axis.setTitleText(f"{title}（区間の始まり）")
        x_axis.setTitleFont(QFont(FONT_FAMILY, 9))
        chart.addAxis(x_axis, Qt.AlignBottom)
        chart.addAxis(y_axis, Qt.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
        chart.legend().setVisible(False)

    @staticmethod
    def _short(label: str, limit: int = 24) -> str:
        return label if len(label) <= limit else label[: limit - 1] + "…"

    def _stats_drill_down(self, row: int, column: int) -> None:
        """統計の表の値で絞り込んで、フレームの一覧へ戻る"""
        if not self._stats_drill:
            return
        key, cross = self._stats_drill
        conditions = [Condition(key, False, self.stats_table.item(row, 0).text())]
        if cross and 0 < column < self.stats_table.columnCount() - 1:
            header = self.stats_table.horizontalHeaderItem(column).text()
            conditions.append(Condition(cross, False, header))
        for condition in conditions:
            self._push_condition(condition)
        self.model.apply()
        self._after_filter()
        self.tabs.setCurrentIndex(0)

    # ---- データ ----

    def _update_source_label(self) -> None:
        self.source_label.setText(
            ", ".join(str(d) for d in self.data_dirs) + f"（{len(self.table.sessions)} セッション）"
        )

    def add_folder(self) -> None:
        start = str(PYTHON_DIR / "data")
        folder = QFileDialog.getExistingDirectory(self, "走行データのフォルダ", start)
        if folder:
            self.data_dirs.append(Path(folder))
            self.reload()

    def reload(self) -> None:
        try:
            self.store.save()
            store = AnnotationStore()
            table = FrameTable(self.data_dirs, self.label_settings, store)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "読み込めません", str(e))
            return
        self.store = store
        self.undo_stack.clear()
        self.range_start = None
        self.table = table
        self.model.table = table
        self.model.apply()
        self.timeline.session_index = None
        self.timeline.rows = []
        self.timeline.update()
        self._update_source_label()
        self._fill_filter_values()
        self._refresh_tag_choices()
        self._after_filter()

    def closeEvent(self, event) -> None:
        self.play_timer.stop()
        self.store.save()
        super().closeEvent(event)


def load_label_settings(config: Path = TRAIN_CONFIG) -> Settings:
    """学習の設定（意図の付け直し、塞がっているかの境界）を、学習と同じに読む"""
    if not config.exists():
        return Settings()
    try:
        import tomllib
    except ImportError:  # Python 3.10 以前
        return Settings()
    with open(config, "rb") as f:
        return Settings.from_params(tomllib.load(f))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="View, label and summarize driving data")
    parser.add_argument(
        "--data_dir",
        nargs="+",
        default=None,
        help="Session folders or their parents (default: data/raw data/interventions)",
    )
    args = parser.parse_args(argv)
    if args.data_dir:
        dirs = [Path(d) for d in args.data_dir]
    else:
        dirs = [PYTHON_DIR / d for d in DEFAULT_DATA_DIRS if (PYTHON_DIR / d).exists()]
    if not dirs:
        print("No data folder found. Pass --data_dir.")
        return 1
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    viewer = DataViewer(dirs)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
