"""プロコンの生の値から操作を決める（`src.e2e.gamepad`）"""

import pygame
import pytest

from src.e2e import gamepad
from src.e2e.gamepad import AXIS_MAX, SWITCH_BUTTONS, PadInput, stick_value, to_input


def pad(*pressed, left_x=0, zl=0, zr=0):
    return to_input(left_x=left_x, zl=zl, zr=zr, pressed=pressed)


def test_nothing_pressed_is_idle():
    assert pad() == PadInput()


def test_a_and_zr_drive_forward_b_and_zl_backward():
    assert pad("A").drive == 1
    assert pad(zr=AXIS_MAX).drive == 1
    assert pad("B").drive == -1
    assert pad(zl=AXIS_MAX).drive == -1


def test_forward_and_backward_together_do_not_move():
    assert pad("A", "B").drive == 0
    assert pad(zr=AXIS_MAX, zl=AXIS_MAX).drive == 0


def test_light_trigger_touch_is_ignored():
    assert pad(zr=AXIS_MAX * 0.3).drive == 0


def test_stick_center_wobble_is_ignored_and_full_tilt_is_full_lock():
    assert stick_value(AXIS_MAX * 0.1) == 0.0
    assert stick_value(AXIS_MAX) == 1.0
    assert stick_value(-32768) == -1.0
    assert 0.0 < stick_value(AXIS_MAX * 0.5) < 0.5


def test_dpad_left_right_override_the_stick():
    assert pad("DPAD_LEFT", left_x=AXIS_MAX).steer == -1.0
    assert pad("DPAD_RIGHT").steer == 1.0
    assert pad(left_x=-AXIS_MAX).steer == -1.0


def test_system_buttons():
    state = pad("PLUS", "MINUS", "X", "HOME")
    assert state.record and state.center and state.stop and state.emergency
    assert state.resume  # 自律走行では＋で運転を返す
    assert pad("Y") == PadInput()  # Y は使わない（B の隣で押し間違えやすい）


def test_face_buttons_are_read_by_position():
    # SDL は位置で返す設定（Xbox の配置の名前）。プロコンは右が A、下が B、上が X、左が Y
    assert SWITCH_BUTTONS["A"] == "CONTROLLER_BUTTON_B"
    assert SWITCH_BUTTONS["B"] == "CONTROLLER_BUTTON_A"
    assert SWITCH_BUTTONS["X"] == "CONTROLLER_BUTTON_Y"
    assert SWITCH_BUTTONS["Y"] == "CONTROLLER_BUTTON_X"


# --- 開けなかった時は止まらずに開き直す ---


class FlakyController:
    """最初の数回は SDL が開けない（スティックの校正値を読めない）プロコン"""

    failures = 1

    def __init__(self, index):
        if FlakyController.failures > 0:
            FlakyController.failures -= 1
            raise pygame.error("Couldn't load stick calibration")
        self.name = "Pro Controller"

    def attached(self):
        return True

    def get_button(self, button):
        return False

    def get_axis(self, axis):
        return 0

    def quit(self):
        pass


@pytest.fixture
def flaky_pad(monkeypatch):
    clock = {"now": 100.0}
    monkeypatch.setattr(gamepad.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(gamepad.pygame.joystick, "get_count", lambda: 1)
    monkeypatch.setattr(gamepad.sdl_controller, "is_controller", lambda index: True)
    monkeypatch.setattr(gamepad.sdl_controller, "Controller", FlakyController)
    FlakyController.failures = 1
    pad = gamepad.Gamepad()
    yield pad, clock
    pad._close()


def test_a_failed_open_is_retried_instead_of_raising(flaky_pad):
    pad, clock = flaky_pad
    assert pad.poll() is None
    assert "stick calibration" in pad.error

    clock["now"] += 0.5  # すぐには開き直さない
    assert pad.poll() is None
    assert FlakyController.failures == 0 and pad.name is None

    clock["now"] += gamepad.REOPEN_INTERVAL_S
    assert pad.poll() == PadInput()
    assert pad.name == "Pro Controller" and pad.error is None


def test_l_and_r_set_the_intent_without_moving_the_car():
    assert pad("L").intent == "left"
    assert pad("R").intent == "right"
    assert pad().intent == "any"
    assert pad("L", "R").intent == "any"  # 両方は決めない
    assert pad("L") == PadInput(intent="left")  # 走行と舵は変えない
