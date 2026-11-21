"""Unified desktop launcher for PiCar-X workflows."""

import os
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, Qt
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

from src.gui import theme
from src.gui.workflows import PHASES, WORKFLOWS, Phase, Workflow, workflows_in

PYTHON_DIR = Path(__file__).resolve().parents[2]
WEB_PORT = 8765

__all__ = ["WORKFLOWS", "Workflow", "LauncherWindow"]


def phase_mark(phase: Phase) -> str:
    return "共" if phase.number is None else str(phase.number)


class PhaseBadge(QLabel):
    """フェーズの番号の丸。色だけに頼らず、番号でも見分けられるようにする"""

    def __init__(self, phase: Phase, size: int = 24):
        super().__init__(phase_mark(phase))
        self.setFixedSize(size, size)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            f"background: {phase.color}; color: white; border-radius: {size // 2}px;"
            f"font-weight: 700; font-size: {9 if size < 24 else 10}pt;"
        )


class PhaseFlow(QFrame):
    """0 準備 → 1 収集 → … → 6 運用 → 1 へ戻る、の流れ"""

    def __init__(self):
        super().__init__()
        self.setObjectName("phaseFlow")
        row = QHBoxLayout(self)
        row.setContentsMargins(12, 8, 12, 8)
        row.setSpacing(6)
        loop = [phase for phase in PHASES if phase.number is not None]
        for i, phase in enumerate(loop):
            if i:
                arrow = QLabel("→")
                arrow.setObjectName("muted")
                row.addWidget(arrow)
            row.addWidget(PhaseBadge(phase, 20))
            name = QLabel(phase.title)
            name.setObjectName("phaseName")
            name.setToolTip(phase.summary)
            row.addWidget(name)
        back = QLabel("→ 1 収集へ戻る")
        back.setObjectName("muted")
        row.addWidget(back)
        row.addStretch(1)


class PhaseHeader(QWidget):
    def __init__(self, phase: Phase):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 12, 0, 0)
        row.setSpacing(8)
        row.addWidget(PhaseBadge(phase))
        title = QLabel(phase.title)
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        summary = QLabel(phase.summary)
        summary.setObjectName("muted")
        row.addWidget(summary, 1)


class WorkflowCard(QFrame):
    def __init__(self, workflow: Workflow, launch, color: str = theme.BORDER):
        super().__init__()
        self.setObjectName("workflowCard")
        # 左の色の帯で、どのフェーズのツールかが分かるようにする
        self.setStyleSheet(f"#workflowCard {{ border-left: 4px solid {color}; }}")
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
        button.setObjectName("carButton" if workflow.moves_car else "launchButton")
        button.clicked.connect(lambda: launch(workflow))
        text_layout = QVBoxLayout()
        text_layout.setSpacing(5)
        text_layout.addWidget(title)
        text_layout.addWidget(description)
        layout.addLayout(text_layout, 1)
        button.setFixedSize(116, 36)
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
        self.web_process: QProcess | None = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        header = QLabel("PiCar-X Control Center")
        header.setObjectName("header")
        web_button = QPushButton("Control Center を開く（ブラウザ）")
        web_button.setObjectName("primary")
        web_button.setToolTip("モデルとデータの来歴、採否、比較、配置、ジョブの記録を管理する画面")
        web_button.clicked.connect(self._open_web)
        header_row = QHBoxLayout()
        header_row.addWidget(header)
        header_row.addStretch(1)
        header_row.addWidget(web_button)
        layout.addLayout(header_row)
        # 開発の流れ（MLOps のフェーズ）を1行で見せる。下の区分も同じ順番と色
        layout.addWidget(PhaseFlow())

        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(10)
        # フェーズの見出しと、そのフェーズのツールのカード。幅に合わせて列の数を変えて並べ直す
        self.sections = [
            (PhaseHeader(phase), [WorkflowCard(w, self._launch, phase.color) for w in entries])
            for phase in PHASES
            if (entries := workflows_in(phase.id))
        ]
        self.columns = 0
        self._arrange(3)
        content = QWidget()
        content.setLayout(self.grid)
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
        self.setStyleSheet(
            theme.APP_STYLE
            + f"""
            #header {{ font-size: 26px; font-weight: 700; }}
            #muted {{ color: {theme.MUTED}; }}
            #sectionTitle {{ font-size: 16px; font-weight: 650; color: {theme.TEXT}; }}
            #phaseFlow {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER};
                border-radius: 10px; }}
            #phaseName {{ font-weight: 600; color: {theme.TEXT}; }}
            #workflowCard {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER};
                border-radius: 10px; }}
            #workflowCard:hover {{ border-color: {theme.BORDER_STRONG}; }}
            #cardTitle {{ font-size: 14px; font-weight: 650; }}
            #workflowCard QLabel {{ background: transparent; }}
            #movesBadge {{ color: {theme.DANGER}; font-size: 12px; font-weight: 600; }}
            QScrollArea, QScrollArea QWidget {{ background: transparent; }}
            QPushButton#launchButton {{ background-color: {theme.ACCENT}; color: #ffffff;
                border: 0; border-radius: 7px; font-weight: 600; padding: 0 8px;
                min-width: 100px; max-width: 100px; min-height: 36px; max-height: 36px; }}
            QPushButton#launchButton:hover {{ background-color: {theme.ACCENT_HOVER}; }}
            QPushButton#carButton {{ background-color: {theme.DANGER}; color: #ffffff;
                border: 0; border-radius: 7px; font-weight: 600; padding: 0 8px;
                min-width: 100px; max-width: 100px; min-height: 36px; max-height: 36px; }}
            QPushButton#carButton:hover {{ background-color: {theme.DANGER_HOVER}; }}
            QTextEdit {{ font-family: "Consolas", "{theme.FONT_FAMILY}"; font-size: 9pt; }}
            """
        )

    def _arrange(self, columns: int):
        if columns == self.columns:
            return
        self.columns = columns
        while self.grid.count():  # 置き場所だけを外す（カードはそのまま使う）
            self.grid.takeAt(0)
        for col in range(3):
            self.grid.setColumnStretch(col, 1 if col < columns else 0)
        row = 0
        for header, cards in self.sections:
            self.grid.addWidget(header, row, 0, 1, columns)
            row += 1
            for i, card in enumerate(cards):
                self.grid.addWidget(card, row + i // columns, i % columns)
            row += (len(cards) + columns - 1) // columns

    def resizeEvent(self, event):
        # 狭い時は2列にして、カードが右へはみ出さないようにする
        self._arrange(3 if event.size().width() >= 1000 else 2)
        super().resizeEvent(event)

    def _launch(self, workflow: Workflow):
        if workflow.confirm and not workflow.moves_car:
            answer = QMessageBox.question(
                self,
                "確認",
                workflow.confirm,
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Yes:
                return
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

        self._start(workflow.title, workflow.module, parsed_arguments)

    def _open_web(self):
        if self.web_process is not None and self.web_process.state() != QProcess.NotRunning:
            # もう動いている。ブラウザだけ開き直す
            import webbrowser

            webbrowser.open(f"http://127.0.0.1:{WEB_PORT}/")
            return
        self.web_process = self._start("Control Center", "src.gui.web", ["--port", str(WEB_PORT)])

    def _start(self, title: str, module: str, arguments: list) -> QProcess:
        process = QProcess(self)
        process.setWorkingDirectory(str(PYTHON_DIR))
        environment = process.processEnvironment()
        environment.insert("PYTHONPATH", str(PYTHON_DIR))
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(["-m", module, *arguments])
        process.readyReadStandardOutput.connect(
            lambda p=process, name=title: self._append_output(name, p.readAllStandardOutput())
        )
        process.readyReadStandardError.connect(
            lambda p=process, name=title: self._append_output(name, p.readAllStandardError())
        )
        process.finished.connect(lambda *_args, p=process: self._reap(p))
        process.errorOccurred.connect(
            lambda _error, p=process, name=title: self.log.append(
                f"起動エラー ({name}): {p.errorString()}"
            )
        )
        self.processes.append(process)
        self.log.append(f"▶ {title}: {module}")
        process.start()
        return process

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
