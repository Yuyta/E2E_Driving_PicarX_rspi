"""Unified desktop launcher for PiCar-X workflows."""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QProcess
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PYTHON_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Workflow:
    title: str
    description: str
    module: str
    group: str
    moves_car: bool = False
    arguments: str = ""
    examples: tuple[tuple[str, str], ...] = ()


WORKFLOWS = (
    Workflow("会話・手動操作", "車体の状態確認、会話、手動操作", "src.gui.app", "操作"),
    Workflow(
        "走行データ収集",
        "カメラを見ながら操作と走行データを記録",
        "src.e2e.gui_collector",
        "データ",
    ),
    Workflow("データ確認・ラベル", "走行データの確認、注釈、統計", "src.e2e.data_viewer", "データ"),
    Workflow(
        "モデル学習",
        "記録データからモデルを学習し ONNX を出力",
        "src.e2e.training.train",
        "学習・評価",
        arguments="--config configs/train.toml --data_dir data/raw --output_dir models/v007",
    ),
    Workflow(
        "交差検証",
        "セッションを分けてモデルを評価",
        "src.e2e.training.cross_validate",
        "学習・評価",
        arguments="--config configs/train.toml --data_dir data/raw",
    ),
    Workflow(
        "モデル比較", "登録済みモデルの評価値を比較", "src.e2e.registry.compare", "学習・評価"
    ),
    Workflow(
        "安全チェック",
        "M1 の車体・安全機能を確認",
        "src.e2e.m1_check",
        "車体チェック",
        arguments="status",
        examples=(
            ("状態", "status"),
            ("カメラ計測", "camera --seconds 20"),
            ("安全イベント", "events"),
        ),
    ),
    Workflow(
        "動きの計測",
        "M3 の動き計測・校正",
        "src.e2e.m3_check",
        "車体チェック",
        arguments="run --scenario still",
        examples=(
            ("静止", "run --scenario still"),
            ("直進（車体が動く）", "run --scenario forward --i-understand-the-car-will-move"),
            ("校正", "calibrate"),
        ),
    ),
    Workflow(
        "速度チェック",
        "車輪が回り始める速度を計測",
        "src.e2e.speed_check",
        "車体チェック",
        True,
        "--i-understand-the-car-will-move",
        (("速度確認", "--speeds 30 35 40 45 --i-understand-the-car-will-move"),),
    ),
    Workflow(
        "Pi 自律走行", "選択したモデルで Pi の車体を自律走行", "src.e2e.pi_autopilot", "走行", True
    ),
    Workflow(
        "自動データ収集",
        "ルールに従って車体を走らせデータを記録",
        "src.e2e.auto_collector",
        "走行",
        True,
        "--i-understand-the-car-will-move",
        (("150秒（車体が動く）", "--seconds 150 --i-understand-the-car-will-move"),),
    ),
)


class WorkflowCard(QFrame):
    def __init__(self, workflow: Workflow, launch):
        super().__init__()
        self.setObjectName("workflowCard")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        self.setMinimumHeight(92)
        title = QLabel(workflow.title)
        title.setObjectName("cardTitle")
        description = QLabel(workflow.description)
        description.setWordWrap(True)
        description.setObjectName("muted")
        button = QPushButton("起動" if not workflow.moves_car else "確認して起動…")
        button.setObjectName("launchButton")
        button.clicked.connect(lambda: launch(workflow))
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)
        text_layout.addWidget(title)
        text_layout.addWidget(description)
        layout.addLayout(text_layout, 1)
        button.setFixedSize(132, 38)
        layout.addWidget(button)


class ArgumentsDialog(QDialog):
    def __init__(self, workflow: Workflow, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{workflow.title} の起動設定")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("操作の例を選ぶか、下の引数を編集してください。"))
        self.preset = QComboBox()
        self.preset.addItem("選択してください", None)
        for label, value in workflow.examples:
            self.preset.addItem(label, value)
        if workflow.examples:
            layout.addWidget(QLabel("操作の例"))
            layout.addWidget(self.preset)
        form = QFormLayout()
        self.arguments = QLineEdit(workflow.arguments)
        self.arguments.setClearButtonEnabled(True)
        form.addRow("実行引数", self.arguments)
        layout.addLayout(form)
        example_text = workflow.arguments or "追加の引数はありません。"
        hint = QLabel(f"既定値・例: {example_text}")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)
        self.preset.currentIndexChanged.connect(self._apply_preset)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("起動")
        buttons.button(QDialogButtonBox.Cancel).setText("キャンセル")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_preset(self, index: int):
        value = self.preset.itemData(index)
        if value is not None:
            self.arguments.setText(value)


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PiCar-X Control Center")
        self.resize(1080, 860)
        self.setMinimumSize(760, 620)
        self.processes: list[QProcess] = []
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        header = QLabel("PiCar-X Control Center")
        header.setObjectName("header")
        layout.addWidget(header)
        subtitle = QLabel("操作、データ、学習、評価をここから始められます")
        subtitle.setObjectName("muted")
        layout.addWidget(subtitle)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        for col in range(3):
            grid.setColumnStretch(col, 1)
        groups = list(dict.fromkeys(workflow.group for workflow in WORKFLOWS))
        row = 0
        for group in groups:
            label = QLabel(group)
            label.setObjectName("sectionTitle")
            grid.addWidget(label, row, 0, 1, 3)
            row += 1
            entries = [workflow for workflow in WORKFLOWS if workflow.group == group]
            for col, workflow in enumerate(entries):
                grid.addWidget(WorkflowCard(workflow, self._launch), row, col)
            row += 1
        content = QWidget()
        content.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        # 長時間起動しても、ログ全量の保持・再描画で画面が重くならないようにする。
        self.log.document().setMaximumBlockCount(2000)
        self.log.setPlaceholderText("起動したツールの出力がここに表示されます")
        self.log.setMaximumHeight(105)
        layout.addWidget(QLabel("実行ログ"))
        layout.addWidget(self.log)
        self.setCentralWidget(root)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #f5f7fa; color: #172033; font-size: 14px; }
            #header { font-size: 28px; font-weight: 700; }
            #muted { color: #64748b; }
            #sectionTitle { font-size: 17px; font-weight: 650; padding-top: 8px; }
            #workflowCard { background: white; border: 1px solid #dbe2ea;
                border-radius: 10px; min-height: 102px; }
            #cardTitle { font-size: 15px; font-weight: 650; }
            #workflowCard QLabel { background: transparent; }
            QScrollArea, QScrollArea QWidget { background: transparent; }
            QScrollBar:vertical { background: transparent; width: 10px; margin: 4px 2px; }
            QScrollBar::handle:vertical { background: #b8c4d4; min-height: 28px;
                border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #8192a8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent; }
            QPushButton { background: #2563eb; color: white; border: 0;
                border-radius: 6px; padding: 9px; }
            QPushButton#launchButton { background-color: #2563eb; color: #ffffff;
                border: 0; border-radius: 7px; font-weight: 600; min-width: 132px;
                max-width: 132px; min-height: 38px; max-height: 38px; }
            QPushButton:hover { background: #1d4ed8; }
            QTextEdit { background: white; border: 1px solid #dbe2ea; border-radius: 6px; }
        """)

    def _launch(self, workflow: Workflow):
        if workflow.moves_car:
            answer = QMessageBox.warning(
                self,
                "車体が動きます",
                f"{workflow.title} を起動すると車体が動く可能性があります。"
                "周囲の安全を確認しましたか？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return

        dialog = ArgumentsDialog(workflow, self)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            import shlex

            parsed_arguments = shlex.split(dialog.arguments.text())
        except ValueError as error:
            QMessageBox.warning(self, "引数を確認してください", str(error))
            return

        process = QProcess(self)
        process.setWorkingDirectory(str(PYTHON_DIR))
        environment = process.processEnvironment()
        environment.insert("PYTHONPATH", str(PYTHON_DIR))
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(["-m", workflow.module, *parsed_arguments])
        process.readyReadStandardOutput.connect(
            lambda p=process, name=workflow.title: self._append_output(
                name, p.readAllStandardOutput()
            )
        )
        process.readyReadStandardError.connect(
            lambda p=process, name=workflow.title: self._append_output(
                name, p.readAllStandardError()
            )
        )
        process.finished.connect(lambda *_args, p=process: self._reap(p))
        process.errorOccurred.connect(
            lambda _error, p=process, name=workflow.title: self.log.append(
                f"起動エラー ({name}): {p.errorString()}"
            )
        )
        self.processes.append(process)
        self.log.append(f"▶ {workflow.title}: {workflow.module}")
        process.start()

    def _append_output(self, name: str, data):
        output = bytes(data).decode("utf-8", errors="replace").strip()
        if output:
            self.log.append(f"[{name}] {output}")

    def _reap(self, process: QProcess):
        if process in self.processes:
            self.processes.remove(process)
        process.deleteLater()

    def closeEvent(self, event):
        active = [process for process in self.processes if process.state() != QProcess.NotRunning]
        if active:
            answer = QMessageBox.question(
                self,
                "実行中のツール",
                "起動中のツールも終了しますか？",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                event.ignore()
                return
            for process in active:
                process.terminate()
        event.accept()


def main():
    os.chdir(PYTHON_DIR)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
