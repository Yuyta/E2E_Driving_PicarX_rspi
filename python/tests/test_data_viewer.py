"""データビューアの画面（絞り込み、並べ替え、ラベルの変更と元に戻す、統計）"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtCore import QItemSelectionModel, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from src.e2e.data_viewer import Condition, DataViewer  # noqa: E402
from src.e2e.dataset.frame_table import EXCLUDED_BY_HAND, Settings  # noqa: E402
from tests.test_e2e_annotations import make_session, moving, stopped  # noqa: E402


@pytest.fixture
def app(tmp_path):
    # 利用者の設定（レジストリ）に触れない
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp_path / "settings"))
    return QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(app, tmp_path):
    data = tmp_path / "data"
    make_session(data, "20261116_120000", [moving(0.0, 100), moving(0.5, 30), stopped()])
    make_session(data, "20261116_130000", [moving(-1.0, 20), moving(0.0)])
    window = DataViewer([data / "raw"], Settings(turn_intent=False))
    yield window
    window.close()


def select(viewer, view_rows):
    viewer.view.clearSelection()
    for r in view_rows:
        viewer.view.selectionModel().select(
            viewer.model.index(r, 0),
            QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )
    viewer.view.selectionModel().setCurrentIndex(
        viewer.model.index(view_rows[0], 0), QItemSelectionModel.NoUpdate
    )


def test_filter_and_sort(viewer):
    assert viewer.model.rowCount() == 5
    viewer.filter_column.setCurrentIndex(viewer.filter_column.findData("motion"))
    viewer.filter_value.setCurrentText("前進")
    viewer.add_condition()
    assert viewer.model.rowCount() == 4

    viewer.filter_column.setCurrentIndex(viewer.filter_column.findData("distance_cm"))
    viewer.filter_value.setCurrentText("<50")
    viewer.add_condition()
    assert viewer.model.rowCount() == 2
    assert "2 / 5" in viewer.count_label.text()

    col = [c.key for c in viewer.model.columns].index("steering_deg")
    viewer.model.sort(col, Qt.AscendingOrder)
    assert [viewer.table.rows[i]["steering_deg"] for i in viewer.model.visible] == [-35.0, 17.5]

    viewer.clear_conditions()
    assert viewer.model.rowCount() == 5


def test_numeric_conditions():
    assert Condition("distance_cm", False, "10..30")({"distance_cm": 20})
    assert not Condition("distance_cm", False, ">=30")({"distance_cm": 20})
    assert Condition("distance_cm", True, "(なし)")({"distance_cm": 5})
    assert Condition("tags", False, "脚")({"tags": "布団,脚"})
    with pytest.raises(ValueError):
        Condition("distance_cm", False, "abc")


def test_label_selected_frames_and_undo(viewer, tmp_path):
    select(viewer, [0, 1])
    viewer.tag_input.setCurrentText("脚")
    viewer.add_tag()
    viewer.set_exclude(True)
    rows = viewer.table.rows
    assert rows[0]["tags"] == "脚" and rows[1]["tags"] == "脚"
    assert rows[0]["training"] == EXCLUDED_BY_HAND
    saved = tmp_path / "data" / "annotations" / "20261116_120000.json"
    assert saved.exists()
    assert "脚" in viewer.store.all_tags()

    viewer.undo()  # 除外を戻す
    assert viewer.table.rows[0]["training"] != EXCLUDED_BY_HAND
    assert viewer.table.rows[0]["tags"] == "脚"
    viewer.undo()  # タグを戻す
    assert viewer.table.rows[0]["tags"] == ""
    assert '"frames": {}' in saved.read_text(encoding="utf-8")


def test_override_intent_and_blocked(viewer):
    select(viewer, [3])  # 2つ目のセッションの最初
    viewer.intent_box.setCurrentIndex(viewer.intent_box.findData("left"))
    viewer.apply_intent()
    viewer.blocked_box.setCurrentIndex(viewer.blocked_box.findText("空き"))
    viewer.apply_blocked()
    row = viewer.table.rows[3]
    assert row["intent"] == "left"
    assert row["blocked"] == "空き"
    assert row["overrides"] == "intent,blocked"

    viewer.intent_box.setCurrentIndex(0)  # 自動に戻す
    viewer.apply_intent()
    assert viewer.table.rows[3]["intent"] == "any"


def test_stats_and_drill_down(viewer):
    viewer.tabs.setCurrentIndex(1)
    viewer.stats_field.setCurrentIndex(viewer.stats_field.findData("motion"))
    texts = {
        viewer.stats_table.item(r, 0).text(): viewer.stats_table.item(r, 1).text()
        for r in range(viewer.stats_table.rowCount())
    }
    assert texts == {"前進": "4", "停止": "1"}

    viewer.stats_field.setCurrentIndex(viewer.stats_field.findData("steering_deg"))
    assert viewer.stats_table.rowCount() > 1  # 度数分布
    assert "中央値" in viewer.stats_summary.text()

    viewer.stats_field.setCurrentIndex(viewer.stats_field.findData("session"))
    viewer.stats_cross.setCurrentIndex(viewer.stats_cross.findData("motion"))
    first = viewer.stats_table.item(0, 0).text()
    viewer._stats_drill_down(0, 0)
    assert viewer.tabs.currentIndex() == 0
    assert {viewer.table.rows[i]["session"] for i in viewer.model.visible} == {first}


def test_new_columns_follow_their_default_after_saved_settings(app, viewer, tmp_path):
    # 列を増やす前の設定（記録にない列）があっても、増えた列は最初の表示に従う
    viewer.qsettings.setValue("column_visible", '{"session": true}')
    again = DataViewer([tmp_path / "data" / "raw"], Settings(turn_intent=False))
    try:
        keys = [c.key for c in again.model.columns]
        assert not again.view.horizontalHeader().isSectionHidden(keys.index("warnings"))
        assert again.view.horizontalHeader().isSectionHidden(keys.index("battery_v"))
    finally:
        again.close()


def test_hidden_columns_are_remembered(app, viewer, tmp_path):
    col = [c.key for c in viewer.model.columns].index("distance_cm")
    viewer._set_column_visible(col, False)
    again = DataViewer([tmp_path / "data" / "raw"], Settings(turn_intent=False))
    try:
        assert again.view.horizontalHeader().isSectionHidden(col)
    finally:
        again.close()


def test_quick_filter_shows_frames_with_warnings(app, tmp_path):
    data = tmp_path / "data"
    make_session(data, "20261116_140000", [moving(0.0, 100), moving(0.9, 150), moving(0.0, 10)])
    window = DataViewer([data / "raw"], Settings(turn_intent=False))
    try:
        window.quick_filter.setCurrentIndex(1)  # 注意のあるフレーム
        assert [window.table.rows[i]["index"] for i in window.model.visible] == [1, 2]
        assert "注意 2" in window.count_label.text()
        window.quick_filter.setCurrentIndex(0)
        assert window.model.rowCount() == 3
    finally:
        window.close()


def test_range_selection_from_the_timeline_and_marks(viewer):
    viewer.select_frame_range(0, 0, 2)  # 1つ目のセッションの3フレーム
    assert viewer.selected_table_rows() == [0, 1, 2]
    assert viewer.timeline.selection == (0, 2)
    viewer.set_exclude(True)
    assert [r["training"] for r in viewer.table.rows[:3]] == [EXCLUDED_BY_HAND] * 3
    assert viewer.table.rows[3]["training"] != EXCLUDED_BY_HAND

    # I と O で、2つ目のセッションの範囲を選ぶ
    select(viewer, [4])
    viewer.mark_range_start()
    select(viewer, [3])
    viewer.mark_range_end()
    assert viewer.selected_table_rows() == [3, 4]

    # 絞り込みで隠れた行は選ばない
    viewer.filter_column.setCurrentIndex(viewer.filter_column.findData("motion"))
    viewer.filter_value.setCurrentText("前進")
    viewer.add_condition()
    viewer.select_frame_range(0, 0, 2)
    assert viewer.selected_table_rows() == [0, 1]
