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
USED = "#1baf7a"
UNUSED = "#d6d5ce"

APP_STYLE = f"""
QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; }}
QWidget {{ font-family: "{FONT_FAMILY}", "Segoe UI", sans-serif; font-size: 10pt; }}
QLabel {{ background: transparent; }}
QGroupBox {{
    background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 14px; padding: 14px 10px 10px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; color: {TEXT_2}; }}
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
QToolTip {{ background: {SURFACE}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px 6px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {UNUSED}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {BORDER_STRONG}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {UNUSED}; border-radius: 4px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
"""
