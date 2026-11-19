"""PiCar-X のデスクトップ画面（Qt）の共通の見た目

色はブラウザの Control Center（src/gui/web/static/app.css）と同じ値を使い、
どの画面を開いても同じ見た目になるようにする。
"""

FONT_FAMILY = "Yu Gothic UI"

BG = "#f6f6f3"
SURFACE = "#fcfcfb"
SURFACE_2 = "#f1f0ec"
BORDER = "#e1e0d9"
BORDER_STRONG = "#c3c2b7"
TEXT = "#1b1b1a"
TEXT_2 = "#52514e"
MUTED = "#76756f"
ACCENT = "#2a78d6"
ACCENT_HOVER = "#256abf"
ACCENT_SOFT = "#e6f0fc"
DANGER = "#d03b3b"
DANGER_HOVER = "#b53030"
DANGER_SOFT = "#fbe7e7"
WARN = "#8a5a00"
WARN_SOFT = "#fdf3dc"
GOOD = "#0ca30c"
GOOD_INK = "#006300"
GOOD_SOFT = "#e5f4e5"
WARN_STRONG = "#fab219"
CAMERA_BG = "#101214"
SIDEBAR = "#15171c"
USED = "#1baf7a"
UNUSED = "#d6d5ce"

APP_STYLE = f"""
QMainWindow, QMainWindow > QWidget, QDialog {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-family: "{FONT_FAMILY}", "Segoe UI", sans-serif; font-size: 10pt; }}
QLabel {{ background: transparent; }}
QGroupBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 12px;
    margin-top: 0; padding: 40px 14px 14px 14px; font-weight: 700; font-size: 10.5pt;
}}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left;
    left: 16px; top: 13px; padding: 0; color: {TEXT}; background: transparent; }}
QGroupBox > QWidget {{ background: transparent; }}
QGroupBox QLabel {{ font-weight: normal; font-size: 10pt; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 6px 8px; selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus,
QPlainTextEdit:focus {{ border: 1px solid {ACCENT}; }}
QPushButton {{
    background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER};
    border-radius: 7px; padding: 7px 14px; min-height: 20px;
}}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; background: {ACCENT_SOFT}; }}
QPushButton:pressed {{ background: #d6e6fa; }}
QPushButton:disabled {{ color: #a3a29b; background: {SURFACE_2}; border-color: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white;
    font-weight: 600; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; color: white; }}
QPushButton#danger {{ background: {DANGER}; border-color: {DANGER}; color: white;
    font-weight: 700; }}
QPushButton#danger:hover {{ background: {DANGER_HOVER}; color: white; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
/* −・＋ のボタンを横に置く数値の欄。中の矢印は幅 0 で隠す（NoButtons にすると、スタイルを
   当てた時に Qt が数字を描かない） */
QSpinBox#stepper {{ padding: 2px 6px; min-height: 26px; font-size: 11pt; font-weight: 700; }}
QSpinBox#stepper::up-button, QSpinBox#stepper::down-button {{ width: 0px; border: none; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; padding: 8px 18px; margin-right: 2px;
    border-bottom: 2px solid transparent; color: {MUTED}; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTableView, QTableWidget, QListWidget {{ background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 8px; selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {SURFACE}; color: {TEXT_2}; font-weight: 600;
    border: none; border-bottom: 1px solid {BORDER_STRONG}; padding: 6px 8px; }}
QProgressBar {{ background: {SURFACE_2}; border: none; border-radius: 4px; height: 8px;
    text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
/* ヘッダー（各画面の上の帯。Control Center の上の帯と同じ形） */
QFrame#header {{ background: {SURFACE}; border: none; border-bottom: 1px solid {BORDER}; }}
QLabel#headerMark {{ background: {ACCENT}; color: white; border-radius: 9px;
    font-size: 15pt; font-weight: 700; min-width: 36px; max-width: 36px;
    min-height: 36px; max-height: 36px; qproperty-alignment: AlignCenter; }}
QLabel#headerTitle {{ font-size: 14pt; font-weight: 700; }}
QLabel#headerSub {{ color: {MUTED}; font-size: 9pt; }}
/* 数値のタイル */
QFrame#tile {{ background: {SURFACE_2}; border: none; border-radius: 10px; }}
QLabel#tileLabel {{ color: {MUTED}; font-size: 9pt; }}
QLabel#tileValue {{ font-size: 15pt; font-weight: 700; }}
QLabel#tileValueSmall {{ font-size: 10.5pt; font-weight: 600; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#hint {{ color: {MUTED}; font-size: 9pt; }}
/* 状態のバッジ。色だけでなく文字でも状態を書く */
QLabel#pill {{ border-radius: 12px; padding: 3px 12px; font-weight: 700;
    background: {SURFACE_2}; color: {TEXT_2}; border: 1px solid {BORDER}; }}
QLabel#pill[level="good"] {{ background: {GOOD_SOFT}; color: {GOOD_INK};
    border-color: {GOOD_SOFT}; }}
QLabel#pill[level="warn"] {{ background: {WARN_SOFT}; color: {WARN}; border-color: {WARN_SOFT}; }}
QLabel#pill[level="bad"] {{ background: {DANGER}; color: white; border-color: {DANGER}; }}
QLabel#pill[level="accent"] {{ background: {ACCENT_SOFT}; color: {ACCENT_HOVER};
    border-color: {ACCENT_SOFT}; }}
QLabel[level="bad"] {{ color: {DANGER}; }}
QLabel[level="accent"] {{ color: {ACCENT}; }}
/* カメラ */
QLabel#cameraView {{ background: {CAMERA_BG}; color: #8d93a0; border-radius: 10px; }}
/* キーの表示 */
QLabel#kbd {{ background: {SURFACE}; border: 1px solid {BORDER_STRONG}; border-bottom-width: 2px;
    border-radius: 5px; padding: 1px 7px; font-family: "Consolas"; font-weight: 700;
    font-size: 9pt; color: {TEXT}; }}
/* 方向のボタン */
QPushButton#pad {{ min-width: 72px; min-height: 46px; font-size: 13pt; border-radius: 10px; }}
QPushButton#pad:pressed {{ background: {ACCENT}; color: white; }}
QPushButton#estop {{ background: {DANGER}; border: none; color: white; font-weight: 800;
    font-size: 11pt; padding: 10px 18px; border-radius: 9px; }}
QPushButton#estop:hover {{ background: {DANGER_HOVER}; color: white; }}
QPushButton#record {{ background: {DANGER}; border: none; color: white; font-weight: 700;
    font-size: 11pt; padding: 12px 20px; border-radius: 10px; }}
QPushButton#record:hover {{ background: {DANGER_HOVER}; color: white; }}
QPushButton#record[recording="true"] {{ background: {WARN_STRONG}; color: #0b0b0b; }}
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {UNUSED}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_STRONG}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {UNUSED}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""


def set_level(widget, level: str = "") -> None:
    """状態の色を変える（good / warn / bad / accent / 空）。スタイルを当て直す"""
    widget.setProperty("level", level)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_property(widget, name: str, value) -> None:
    widget.setProperty(name, value)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def header(title: str, subtitle: str = "", mark: str = "P"):
    """画面の上の帯。右側に部品を足せるように (帯, 右側のレイアウト) を返す"""
    from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout

    bar = QFrame()
    bar.setObjectName("header")
    row = QHBoxLayout(bar)
    row.setContentsMargins(20, 12, 20, 12)
    row.setSpacing(12)
    icon = QLabel(mark)
    icon.setObjectName("headerMark")
    row.addWidget(icon)
    text = QVBoxLayout()
    text.setSpacing(0)
    name = QLabel(title)
    name.setObjectName("headerTitle")
    text.addWidget(name)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("headerSub")
        text.addWidget(sub)
    row.addLayout(text)
    row.addStretch(1)
    right = QHBoxLayout()
    right.setSpacing(10)
    row.addLayout(right)
    return bar, right


def tile(label: str, value, small: bool = False):
    """ラベルと値のタイル。value は表示に使う QLabel（画面の処理が文字を書き換える）"""
    from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

    box = QFrame()
    box.setObjectName("tile")
    layout = QVBoxLayout(box)
    layout.setContentsMargins(12, 9, 12, 10)
    layout.setSpacing(2)
    caption = QLabel(label)
    caption.setObjectName("tileLabel")
    value.setObjectName("tileValueSmall" if small else "tileValue")
    layout.addWidget(caption)
    layout.addWidget(value)
    return box


def pill(text: str, level: str = ""):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    label.setObjectName("pill")
    label.setAlignment(Qt.AlignCenter)
    label.setFixedHeight(30)
    set_level(label, level)
    return label


def key_row(keys, text: str):
    """キーの表示と説明の1行"""
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    for key in keys:
        cap = QLabel(key)
        cap.setObjectName("kbd")
        layout.addWidget(cap)
    desc = QLabel(text)
    desc.setWordWrap(True)
    layout.addWidget(desc, 1)
    return row
