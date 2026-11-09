"""収集 GUI が各フレームに保存するセンサー値"""

from src.e2e.gui_collector import sensor_fields
from src.models import (
    BatteryStatus,
    DistanceStatus,
    FloorStatus,
    SensorsStatus,
)


def status(distance: DistanceStatus):
    sensors = SensorsStatus(
        distance=distance,
        floor=FloorStatus(values=[900, 910, 920]),
        battery=BatteryStatus(voltage=8.06, percent=78),
    )
    return type("Status", (), {"sensors": sensors})()


def test_sensor_fields_keep_distance_floor_and_battery():
    fields = sensor_fields(status(DistanceStatus(cm=42.34)))
    assert fields == {
        "distance_cm": 42.3,
        "distance_far": False,
        "floor": [900, 910, 920],
        "battery_v": 8.06,
    }


def test_far_distance_is_saved_as_none_with_the_far_mark():
    fields = sensor_fields(status(DistanceStatus(cm=343.0, far=True)))
    assert fields["distance_cm"] is None
    assert fields["distance_far"] is True


def test_unreadable_status_saves_none():
    assert set(sensor_fields(None).values()) == {None}
