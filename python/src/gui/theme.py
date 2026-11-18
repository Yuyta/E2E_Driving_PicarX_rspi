"""Shared visual theme for PiCar-X desktop windows."""

APP_STYLE = """
QMainWindow, QWidget { background: #f4f6f9; color: #1f2933; }
QWidget { font-family: "Yu Gothic UI", "Segoe UI", sans-serif; font-size: 10pt; }
QLabel { background: transparent; }
QGroupBox {
    background: #ffffff; border: 1px solid #dde2e8; border-radius: 8px;
    margin-top: 14px; padding: 14px 10px 10px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #344054; }
QLineEdit, QComboBox, QSpinBox, QTextEdit, QPlainTextEdit {
    background: #ffffff; color: #1f2933; border: 1px solid #cfd6df;
    border-radius: 6px; padding: 6px 8px; selection-background-color: #2f6fde;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus {
    border: 1px solid #2f6fde;
}
QPushButton {
    background: #ffffff; color: #263445; border: 1px solid #cfd6df;
    border-radius: 6px; padding: 7px 12px; min-height: 20px;
}
QPushButton:hover { border-color: #2f6fde; color: #2f6fde; background: #f7faff; }
QPushButton:pressed { background: #e6efff; }
QPushButton:disabled { color: #98a2b3; background: #f2f4f7; }
QComboBox::drop-down { border: 0; width: 22px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #c6ced8; border-radius: 4px; min-height: 24px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""
