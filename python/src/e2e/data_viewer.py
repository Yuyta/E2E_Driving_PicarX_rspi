"""
走行データのビューア（学習データの確認、手作業のラベル付け、分布と統計）

使い方:
    uv run python -m src.e2e.data_viewer
    uv run python -m src.e2e.data_viewer --data_dir data/raw data/interventions

- フレーム: 表で一覧し、列の見出しで並べ替え、見出しの右クリックで表示する列を選ぶ。
  条件で絞り込み、選んだフレーム（複数可）にタグ、除外、意図と塞がっているかの上書き、メモを付ける。
  画像の上の線は舵角。Space で再生と停止、X で除外の切り替え、Ctrl+Z で元に戻す
- 統計: 絞り込んだフレームの、列ごとの数と割合（数値は度数分布）、2つの列の組み合わせの数。
  表の行をダブルクリックすると、その値で絞り込んでフレームへ戻る

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

from PySide6.QtCharts import QBarCategoryAxis, QBarSeries, QBarSet, QChart, QChartView, QValueAxis
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QPointF, QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QColor, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
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
MULTI_VALUE_KEYS = ("tags", "overrides", "motion_flags")
NONE_TEXT = "(なし)"
AUTO_TEXT = "自動"


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
        return f"{COLUMN_BY_KEY[self.key].title} {'≠' if self.negate else '='} {self.text}"


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
        self.sort_column: Optional[str] = None
        self.sort_order = Qt.AscendingOrder
        self.visible: List[int] = list(range(len(table)))

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
            return "" if value is None else str(value)
        if role == Qt.TextAlignmentRole and self.columns[index.column()].kind == "num":
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole and row.get("training") != "使う":
            return QColor(Qt.gray)
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
        visible = [i for i in range(len(rows)) if all(c(rows[i]) for c in self.conditions)]
        if self.sort_column:
            key = self.sort_column
            visible.sort(
                key=lambda i: sort_key(rows[i].get(key)),
                reverse=self.sort_order == Qt.DescendingOrder,
            )
        self.visible = visible
        self.endResetModel()

    def refresh_rows(self, table_rows: Sequence[int]) -> None:
        """手作業のラベルを変えた行の表示を更新する（絞り込みは掛け直さない）"""
        where = {r: i for i, r in enumerate(self.visible)}
        for r in table_rows:
            if r in where:
                i = where[r]
                self.dataChanged.emit(self.index(i, 0), self.index(i, len(self.columns) - 1))

    def table_row(self, view_row: int) -> int:
        return self.visible[view_row]

    def view_row(self, table_row: int) -> Optional[int]:
        try:
            return self.visible.index(table_row)
        except ValueError:
            return None


# ---- 画像 ----


class FrameImage(QLabel):
    """フレームの画像と、舵角の線（緑）"""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
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
        painter.setPen(QPen(QColor(40, 220, 90), 4))
        w, h = scaled.width(), scaled.height()
        base = QPointF(w / 2, h - 4)
        angle = math.radians(self._steering_deg)
        length = h * 0.35
        painter.drawLine(
            base, QPointF(base.x() + length * math.sin(angle), base.y() - length * math.cos(angle))
        )
        painter.end()
        self.setPixmap(scaled)


# ---- 画面 ----


class DataViewer(QMainWindow):
    def __init__(self, data_dirs: Sequence[Path], settings: Optional[Settings] = None):
        super().__init__()
        self.setWindowTitle("PiCar-X 走行データ")
        self.resize(1500, 900)
        self.qsettings = QSettings("PiCarX", "data_viewer")
        self.data_dirs = [Path(d) for d in data_dirs]
        self.label_settings = settings or load_label_settings()
        self.store = AnnotationStore()
        self.undo_stack: List[List[Tuple[Any, str, Optional[Dict[str, Any]]]]] = []
        self.table = FrameTable(self.data_dirs, self.label_settings, self.store)
        self.model = FrameModel(self.table)

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
        QShortcut(QKeySequence.Undo, self, activated=self.undo)

        self._restore_columns()
        self._after_filter()

    # ---- 構成 ----

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("データ")
        bar.setMovable(False)
        open_action = QAction("フォルダを追加…", self)
        open_action.triggered.connect(self.add_folder)
        bar.addAction(open_action)
        reload_action = QAction("読み込み直す", self)
        reload_action.triggered.connect(self.reload)
        bar.addAction(reload_action)
        undo_action = QAction("元に戻す (Ctrl+Z)", self)
        undo_action.triggered.connect(self.undo)
        bar.addAction(undo_action)
        self.source_label = QLabel()
        bar.addWidget(self.source_label)
        self._update_source_label()

    def _build_frames_tab(self) -> QWidget:
        # 絞り込み
        self.filter_column = QComboBox()
        for column in COLUMNS:
            self.filter_column.addItem(column.title, column.key)
        self.filter_column.currentIndexChanged.connect(self._fill_filter_values)
        self.filter_op = QComboBox()
        self.filter_op.addItems(["=", "≠"])
        self.filter_value = QComboBox()
        self.filter_value.setEditable(True)
        self.filter_value.setMinimumWidth(180)
        self.filter_value.lineEdit().returnPressed.connect(self.add_condition)
        add_button = QPushButton("条件を追加")
        add_button.clicked.connect(self.add_condition)
        self.condition_list = QListWidget()
        self.condition_list.setMaximumHeight(70)
        remove_button = QPushButton("選んだ条件を外す")
        remove_button.clicked.connect(self.remove_condition)
        clear_button = QPushButton("すべて外す")
        clear_button.clicked.connect(self.clear_conditions)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("絞り込み:"))
        filter_row.addWidget(self.filter_column)
        filter_row.addWidget(self.filter_op)
        filter_row.addWidget(self.filter_value, 1)
        filter_row.addWidget(add_button)
        condition_row = QHBoxLayout()
        condition_row.addWidget(self.condition_list, 1)
        buttons = QVBoxLayout()
        buttons.addWidget(remove_button)
        buttons.addWidget(clear_button)
        condition_row.addLayout(buttons)
        self.count_label = QLabel()

        # 表
        self.view = QTableView()
        self.view.setModel(self.model)
        # 並べ替えを有効にすると今の見出しで並べ替わるので、最初は記録の順のままにする
        self.view.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.view.setSortingEnabled(True)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.verticalHeader().setVisible(False)
        self.view.verticalHeader().setDefaultSectionSize(20)
        self.view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.horizontalHeader().customContextMenuRequested.connect(self._column_menu)
        self.view.selectionModel().selectionChanged.connect(self._on_selection)
        self.view.horizontalHeader().sortIndicatorChanged.connect(lambda *_: self._after_filter())

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(filter_row)
        left_layout.addLayout(condition_row)
        left_layout.addWidget(self.count_label)
        left_layout.addWidget(self.view, 1)

        # 画像、ラベル、記録
        self.image = FrameImage()
        self.selection_label = QLabel("フレームを選んでください")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.image, 3)
        right_layout.addWidget(self.selection_label)
        right_layout.addWidget(self._build_editor())
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        right_layout.addWidget(self.details, 2)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self._fill_filter_values()
        return splitter

    def _build_editor(self) -> QWidget:
        box = QGroupBox("手作業のラベル（選んだすべてのフレームに付く）")
        form = QFormLayout(box)

        self.tag_input = QComboBox()
        self.tag_input.setEditable(True)
        self.tag_input.lineEdit().setPlaceholderText("例: 脚, 布団, ぶれ, 要確認")
        self.tag_input.lineEdit().returnPressed.connect(self.add_tag)
        add_tag = QPushButton("タグを付ける")
        add_tag.clicked.connect(self.add_tag)
        remove_tag = QPushButton("タグを外す")
        remove_tag.clicked.connect(self.remove_tag)
        tag_row = QHBoxLayout()
        tag_row.addWidget(self.tag_input, 1)
        tag_row.addWidget(add_tag)
        tag_row.addWidget(remove_tag)
        form.addRow("タグ", tag_row)

        exclude = QPushButton("学習から除外 (X)")
        exclude.clicked.connect(lambda: self.set_exclude(True))
        include = QPushButton("除外を取り消す")
        include.clicked.connect(lambda: self.set_exclude(False))
        exclude_row = QHBoxLayout()
        exclude_row.addWidget(exclude)
        exclude_row.addWidget(include)
        form.addRow("除外", exclude_row)

        self.intent_box = QComboBox()
        self.intent_box.addItem(AUTO_TEXT, None)
        for intent in INTENTS:
            self.intent_box.addItem(intent, intent)
        intent_apply = QPushButton("意図を設定")
        intent_apply.clicked.connect(self.apply_intent)
        intent_row = QHBoxLayout()
        intent_row.addWidget(self.intent_box, 1)
        intent_row.addWidget(intent_apply)
        form.addRow("意図", intent_row)

        self.blocked_box = QComboBox()
        self.blocked_box.addItem(AUTO_TEXT, "auto")
        self.blocked_box.addItem("塞がり", True)
        self.blocked_box.addItem("空き", False)
        self.blocked_box.addItem("不明", None)
        blocked_apply = QPushButton("塞がりを設定")
        blocked_apply.clicked.connect(self.apply_blocked)
        blocked_row = QHBoxLayout()
        blocked_row.addWidget(self.blocked_box, 1)
        blocked_row.addWidget(blocked_apply)
        form.addRow("前方", blocked_row)

        self.note_input = QLineEdit()
        note_apply = QPushButton("メモを保存")
        note_apply.clicked.connect(self.apply_note)
        note_row = QHBoxLayout()
        note_row.addWidget(self.note_input, 1)
        note_row.addWidget(note_apply)
        form.addRow("メモ", note_row)

        self._refresh_tag_choices()
        return box

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

        controls = QHBoxLayout()
        controls.addWidget(QLabel("集計する列:"))
        controls.addWidget(self.stats_field)
        controls.addWidget(QLabel("組み合わせる列:"))
        controls.addWidget(self.stats_cross)
        controls.addStretch(1)

        self.stats_summary = QLabel()
        self.stats_summary.setWordWrap(True)
        self.chart = QChart()
        self.chart.legend().setVisible(False)
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.Antialiasing)
        self.stats_table = QTableWidget()
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.cellDoubleClicked.connect(self._stats_drill_down)
        self.stats_table.verticalHeader().setVisible(False)

        lower = QSplitter()
        lower.addWidget(self.chart_view)
        lower.addWidget(self.stats_table)
        lower.setStretchFactor(0, 3)
        lower.setStretchFactor(1, 2)
        lower.setSizes([900, 500])

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addLayout(controls)
        layout.addWidget(self.stats_summary)
        layout.addWidget(lower, 1)
        layout.addWidget(
            QLabel("表の行をダブルクリックすると、その値で絞り込んでフレームの一覧へ戻ります")
        )
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
        header = self.view.horizontalHeader()
        shown = [c.key for i, c in enumerate(self.model.columns) if not header.isSectionHidden(i)]
        self.qsettings.setValue("columns", json.dumps(shown))

    def _restore_columns(self) -> None:
        try:
            shown = json.loads(self.qsettings.value("columns", "null") or "null")
        except (TypeError, ValueError):
            shown = None
        for i, column in enumerate(self.model.columns):
            visible = column.key in shown if isinstance(shown, list) else column.visible
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
        self.model.conditions.append(condition)
        self.condition_list.addItem(condition.label())
        self.model.apply()
        self._after_filter()

    def remove_condition(self) -> None:
        row = self.condition_list.currentRow()
        if row < 0:
            return
        self.condition_list.takeItem(row)
        del self.model.conditions[row]
        self.model.apply()
        self._after_filter()

    def clear_conditions(self) -> None:
        self.condition_list.clear()
        self.model.conditions.clear()
        self.model.apply()
        self._after_filter()

    def _after_filter(self) -> None:
        shown = len(self.model.visible)
        rows = self.visible_rows()
        used = sum(1 for r in rows if r["training"] == "使う")
        self.count_label.setText(f"{shown:,} / {len(self.table):,} フレーム（学習に使う {used:,}）")
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
        current = self.current_table_row()
        selected = self.selected_table_rows()
        if current is None and selected:
            current = selected[0]
        if current is None:
            return
        row = self.table.rows[current]
        self.image.show_frame(self.table.image_path(current), float(row["steering_deg"] or 0.0))
        self.selection_label.setText(
            f"{len(selected)} フレームを選択中 ・ 表示: {row['session']} #{row['index']}"
        )
        note = self.table.note(current)
        self.note_input.setText(note.get("note", ""))
        record = self.table.record(current)
        self.details.setPlainText(
            "手作業のラベル: "
            + json.dumps(note, ensure_ascii=False)
            + "\n\n"
            + json.dumps(record, ensure_ascii=False, indent=1)
        )

    def _step(self) -> None:
        index = self.view.currentIndex()
        nxt = (index.row() + 1) if index.isValid() else 0
        if nxt >= self.model.rowCount():
            self.play_timer.stop()
            return
        self.view.selectRow(nxt)

    def toggle_play(self) -> None:
        if self.tabs.currentIndex() != 0 or self.focusWidget() in (
            self.tag_input.lineEdit(),
            self.note_input,
        ):
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
        return changed

    def _commit(self, rows: Sequence[int]) -> None:
        try:
            self.store.save()
        except OSError as e:
            QMessageBox.critical(self, "保存できません", str(e))
        self.table.refresh(rows)
        self.model.refresh_rows(rows)
        self._refresh_tag_choices()
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
        used = sum(1 for r in rows if r["training"] == "使う")
        by_hand = sum(1 for r in rows if r["overrides"] or r["tags"])
        conditions = "、".join(c.label() for c in self.model.conditions) or "なし"
        text = (
            f"対象 {total:,} フレーム（絞り込み: {conditions}） ・ 学習に使う {used:,}"
            f" ・ 手作業のラベルあり {by_hand:,}"
        )
        column = COLUMN_BY_KEY[key]
        if column.kind == "num":
            s = summary(rows, key)
            text += (
                f"\n{column.title}: 値あり {s['count']:,} ・ 最小 {s['min']}"
                f" ・ 中央値 {s['median']} ・ 平均 {s['mean']} ・ 最大 {s['max']}"
            )
        self.stats_summary.setText(text)

        if cross:
            self._show_crosstab(rows, key, cross)
        elif column.kind == "num":
            bins, missing = histogram(rows, key)
            labels = [f"{low:.1f}〜{high:.1f}" for low, high, _ in bins]
            counts = [n for _, _, n in bins]
            if missing:
                labels.append(NONE_TEXT)
                counts.append(missing)
            self._show_counts(column.title, labels, counts, total, drill=False)
        else:
            pairs = count_by(rows, key)
            self._show_counts(column.title, [p[0] for p in pairs], [p[1] for p in pairs], total)

    def _show_counts(
        self, title: str, labels: List[str], counts: List[int], total: int, drill: bool = True
    ) -> None:
        self._stats_drill = (self.stats_field.currentData(), None) if drill else None
        self.stats_table.clear()
        self.stats_table.setColumnCount(3)
        self.stats_table.setHorizontalHeaderLabels([title, "フレーム数", "割合"])
        self.stats_table.setRowCount(len(labels))
        for i, (label, n) in enumerate(zip(labels, counts)):
            self.stats_table.setItem(i, 0, QTableWidgetItem(label))
            count_item = QTableWidgetItem(f"{n:,}")
            count_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(i, 1, count_item)
            share = QTableWidgetItem(f"{100 * n / total:.1f}%" if total else "-")
            share.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(i, 2, share)
        self.stats_table.resizeColumnsToContents()
        self._draw_bars(title, labels[:40], {"": counts[:40]})

    def _show_crosstab(self, rows: List[Dict[str, Any]], key: str, cross: str) -> None:
        self._stats_drill = (key, cross)
        row_labels, col_labels, counts = crosstab(rows, key, cross)
        row_labels = row_labels[:60]
        col_labels = col_labels[:12]
        self.stats_table.clear()
        self.stats_table.setColumnCount(len(col_labels) + 2)
        self.stats_table.setHorizontalHeaderLabels([COLUMN_BY_KEY[key].title] + col_labels + ["計"])
        self.stats_table.setRowCount(len(row_labels))
        for i, a in enumerate(row_labels):
            self.stats_table.setItem(i, 0, QTableWidgetItem(a))
            row_total = 0
            for j, b in enumerate(col_labels):
                n = counts.get((a, b), 0)
                row_total += n
                item = QTableWidgetItem(f"{n:,}" if n else "")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.stats_table.setItem(i, j + 1, item)
            total_item = QTableWidgetItem(f"{row_total:,}")
            total_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.stats_table.setItem(i, len(col_labels) + 1, total_item)
        self.stats_table.resizeColumnsToContents()
        series = {b: [counts.get((a, b), 0) for a in row_labels[:30]] for b in col_labels}
        self._draw_bars(COLUMN_BY_KEY[key].title, row_labels[:30], series)

    def _draw_bars(
        self, title: str, labels: List[str], series_counts: Dict[str, List[int]]
    ) -> None:
        # 軸を外して付け直すと古い軸が残ることがあるので、グラフごと作り直す
        old = self.chart
        self.chart = QChart()
        self.chart_view.setChart(self.chart)
        old.deleteLater()
        series = QBarSeries()
        top = 0
        for name, counts in series_counts.items():
            bar_set = QBarSet(name or title)
            bar_set.append([float(n) for n in counts])
            series.append(bar_set)
            top = max([top] + counts)
        self.chart.addSeries(series)
        x_axis = QBarCategoryAxis()
        x_axis.append(labels)
        if len(labels) > 8:
            x_axis.setLabelsAngle(-60)
        y_axis = QValueAxis()
        y_axis.setRange(0, max(1, top) * 1.05)
        y_axis.setLabelFormat("%d")
        self.chart.addAxis(x_axis, Qt.AlignBottom)
        self.chart.addAxis(y_axis, Qt.AlignLeft)
        series.attachAxis(x_axis)
        series.attachAxis(y_axis)
        self.chart.legend().setVisible(len(series_counts) > 1)
        self.chart.setTitle(title)

    def _stats_drill_down(self, row: int, column: int) -> None:
        """統計の表の値で絞り込んで、フレームの一覧へ戻る"""
        if not getattr(self, "_stats_drill", None):
            return
        key, cross = self._stats_drill
        conditions = [Condition(key, False, self.stats_table.item(row, 0).text())]
        if cross and 0 < column < self.stats_table.columnCount() - 1:
            header = self.stats_table.horizontalHeaderItem(column).text()
            conditions.append(Condition(cross, False, header))
        for condition in conditions:
            self.model.conditions.append(condition)
            self.condition_list.addItem(condition.label())
        self.model.apply()
        self._after_filter()
        self.tabs.setCurrentIndex(0)

    # ---- データ ----

    def _update_source_label(self) -> None:
        self.source_label.setText(
            "  読み込み中: "
            + ", ".join(str(d) for d in self.data_dirs)
            + f"（{len(self.table.sessions)} セッション）"
        )

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "走行データのフォルダ", str(PYTHON_DIR / "data")
        )
        if folder:
            self.data_dirs.append(Path(folder))
            self.reload()

    def reload(self) -> None:
        try:
            self.store.save()
            self.store = AnnotationStore()
            table = FrameTable(self.data_dirs, self.label_settings, self.store)
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "読み込めません", str(e))
            return
        self.undo_stack.clear()
        self.table = table
        self.model.table = table
        self.model.apply()
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
    viewer = DataViewer(dirs)
    viewer.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
