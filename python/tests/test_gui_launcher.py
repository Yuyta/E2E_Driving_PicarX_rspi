import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from src.gui.launcher import WORKFLOWS, LauncherWindow  # noqa: E402


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_workflows_have_arguments_for_required_cli_options():
    workflows = {workflow.module: workflow for workflow in WORKFLOWS}
    assert "--data_dir" in workflows["src.e2e.training.train"].arguments
    assert "--output_dir" in workflows["src.e2e.training.train"].arguments
    assert "--data_dir" in workflows["src.e2e.training.cross_validate"].arguments
    assert workflows["src.e2e.m1_check"].arguments == "status"
    assert workflows["src.e2e.m3_check"].arguments == "run --scenario still"
    assert "--i-understand-the-car-will-move" in workflows["src.e2e.speed_check"].arguments
    assert "--i-understand-the-car-will-move" in workflows["src.e2e.auto_collector"].arguments


def test_action_buttons_are_inside_visible_cards(app):
    window = LauncherWindow()
    window.show()
    app.processEvents()
    try:
        cards = window.findChildren(type(window.findChild(QPushButton).parent()))
        for card in (child for child in cards if child.objectName() == "workflowCard"):
            button = card.findChild(QPushButton)
            assert button is not None and button.isVisible()
            assert card.rect().contains(button.geometry())
    finally:
        window.close()
