"""収集 GUI をゲームパッド（Switch のプロコン）で操作する。

配置は Nintendo のレースゲームと同じ考え方にする。右手の親指で進む・下がる、左手の親指で舵を切る。

    [ZL] 後退                                       前進 [ZR]
    [L]  意図: 左                                   意図: 右 [R]
      (左スティック) 舵                 (X) 停止
      [十字 ←→] 舵いっぱい          (Y) －   (A) 前進
                                        (B) 後退
          [－] 舵を中央へ   [HOME] 緊急停止   [＋] 録画トグル

- A / ZR を押している間は前進、B / ZL を押している間は後退（両方なら止まる）
- X は S キーと同じ停止。押していた走行の操作をすべて離すまで走り出さない
- HOME は画面の EMERGENCY STOP と同じ。戻すには画面の Reset を押す
- L / R を押している間は、行きたい向き（意図）を左・右として記録する。押していなければおまかせ
  （両方なら おまかせ）。車の動きは変えない（ROADMAP の M3「行きたい向きの入力」）

SDL（pygame-ce）の GameController として読む。SDL は既定では A/B/X/Y を刻印の文字で返すが、
古い版や読み方によって入れ替わることがあるので、ボタンの位置で読む設定にして、
位置（下・右・上・左）からプロコンの刻印（B・A・X・Y）を決める。
pygame-ce が入っていない時やパッドがつながっていない時は、何もしない。
"""

import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional

# SDL のウィンドウを開かないので、ウィンドウの外でもパッドの入力を受け取る
os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
# 面のボタンを刻印の文字ではなく位置で返す（下 = SDL の A、右 = B、左 = X、上 = Y）
os.environ["SDL_GAMECONTROLLER_USE_BUTTON_LABELS"] = "0"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
    from pygame._sdl2 import controller as sdl_controller
except ImportError:  # pragma: no cover - e2e の追加の依存を入れていない時
    pygame = None
    sdl_controller = None

AXIS_MAX = 32767.0
# スティックの中央付近の揺れは無視する
STICK_DEADZONE = 0.15
# プロコンの ZL / ZR はオンかオフだけ。アナログのトリガーでも半分を超えたら押したとみなす
TRIGGER_THRESHOLD = 0.5
# 開けなかった時（Bluetooth が不安定でスティックの校正値を読めない時など）に開き直す間隔
REOPEN_INTERVAL_S = 1.0

# プロコンの刻印と、SDL が位置で返すボタン（SDL の名前は Xbox の配置）
SWITCH_BUTTONS = {
    "B": "CONTROLLER_BUTTON_A",  # 下
    "A": "CONTROLLER_BUTTON_B",  # 右
    "Y": "CONTROLLER_BUTTON_X",  # 左
    "X": "CONTROLLER_BUTTON_Y",  # 上
    "MINUS": "CONTROLLER_BUTTON_BACK",
    "PLUS": "CONTROLLER_BUTTON_START",
    "HOME": "CONTROLLER_BUTTON_GUIDE",
    "DPAD_LEFT": "CONTROLLER_BUTTON_DPAD_LEFT",
    "DPAD_RIGHT": "CONTROLLER_BUTTON_DPAD_RIGHT",
    "L": "CONTROLLER_BUTTON_LEFTSHOULDER",
    "R": "CONTROLLER_BUTTON_RIGHTSHOULDER",
}

# 行きたい向き（意図）。上の層（M5 の見回りと行き先）が走行モデルへ渡す値と同じ
INTENT_ANY = "any"
INTENT_LEFT = "left"
INTENT_RIGHT = "right"


def intent_of(left: bool, right: bool) -> str:
    """左右どちらか一方だけを押している時はその向き。どちらも、またはどちらでもなければおまかせ"""
    if left == right:
        return INTENT_ANY
    return INTENT_LEFT if left else INTENT_RIGHT


@dataclass(frozen=True)
class PadInput:
    """ある時点のパッドの状態"""

    drive: int = 0  # 1: 前進、-1: 後退、0: なし
    steer: float = 0.0  # -1.0（左いっぱい）〜 1.0（右いっぱい）
    record: bool = False  # ＋
    center: bool = False  # －
    stop: bool = False  # X
    emergency: bool = False  # HOME
    intent: str = INTENT_ANY  # L / R


def stick_value(raw: float, deadzone: float = STICK_DEADZONE) -> float:
    """スティックの生の値（-32768〜32767）を -1〜1 にする。中央付近は 0、外側は 0 から始める"""
    value = max(-1.0, min(1.0, raw / AXIS_MAX))
    if abs(value) < deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def to_input(left_x: float, zl: float, zr: float, pressed: Iterable[str]) -> PadInput:
    """左スティックと ZL / ZR の生の値（SDL の範囲）と、押しているボタンの刻印から操作を決める"""
    pressed = set(pressed)
    forward = "A" in pressed or zr / AXIS_MAX >= TRIGGER_THRESHOLD
    backward = "B" in pressed or zl / AXIS_MAX >= TRIGGER_THRESHOLD
    drive = 0 if forward == backward else (1 if forward else -1)

    left, right = "DPAD_LEFT" in pressed, "DPAD_RIGHT" in pressed
    if left != right:
        steer = -1.0 if left else 1.0
    else:
        steer = stick_value(left_x)
    return PadInput(
        drive=drive,
        steer=steer,
        record="PLUS" in pressed,
        center="MINUS" in pressed,
        stop="X" in pressed,
        emergency="HOME" in pressed,
        intent=intent_of("L" in pressed, "R" in pressed),
    )


class Gamepad:
    """最初につながった GameController を読む。抜けたら次につながったものを使う"""

    def __init__(self):
        self.available = pygame is not None
        self._pad = None
        self.name: Optional[str] = None
        self.pressed: set = set()  # 最後に読んだ時に押していたボタンの刻印（確認用）
        self.error: Optional[str] = None  # 最後に開けなかった理由。開けたら None
        self._next_open = 0.0
        if self.available:
            pygame.init()
            pygame.joystick.init()
            sdl_controller.init()

    def poll(self) -> Optional[PadInput]:
        """今の状態。パッドがなければ None"""
        if not self.available:
            return None
        for event in pygame.event.get():
            if event.type == pygame.CONTROLLERDEVICEREMOVED:
                self._close()
        if self._pad is None:
            self._open_first()
        if self._pad is None:
            return None
        pad = self._pad
        try:
            if not pad.attached():
                self._close()
                return None
            self.pressed = {
                label
                for label, name in SWITCH_BUTTONS.items()
                if pad.get_button(getattr(pygame, name))
            }
            return to_input(
                left_x=pad.get_axis(pygame.CONTROLLER_AXIS_LEFTX),
                zl=pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT),
                zr=pad.get_axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT),
                pressed=self.pressed,
            )
        except pygame.error as exc:
            # 読んでいる途中で切れた。つながっていないとみなし、少し待って開き直す
            self.error = str(exc)
            self._close()
            self._next_open = time.monotonic() + REOPEN_INTERVAL_S
            return None

    def _open_first(self) -> None:
        now = time.monotonic()
        if now < self._next_open:
            return
        for index in range(pygame.joystick.get_count()):
            if sdl_controller.is_controller(index):
                try:
                    self._pad = sdl_controller.Controller(index)
                except pygame.error as exc:
                    # SDL がプロコンからスティックの校正値を読めなかった時など。少し待って開き直す
                    self.error = str(exc)
                    self._next_open = now + REOPEN_INTERVAL_S
                    return
                self.name = self._pad.name
                self.error = None
                return

    def _close(self) -> None:
        if self._pad is not None:
            try:
                self._pad.quit()
            except Exception:
                pass
        self._pad = None
        self.name = None
        self.pressed = set()

    def close(self) -> None:
        self._close()
        if self.available:
            sdl_controller.quit()
            pygame.joystick.quit()


def main() -> None:
    """パッドが読めるかと、ボタンの刻印が合っているかを確かめる（Ctrl+C で終わる）"""
    pad = Gamepad()
    if not pad.available:
        print("pygame-ce がない: uv sync --extra e2e")
        return
    print("ボタンを1つずつ押し、表示される刻印がプロコンの刻印と同じかを確かめる")
    last = None
    try:
        while True:
            state = pad.poll()
            shown = (state, tuple(sorted(pad.pressed)), pad.error)
            if shown != last:
                buttons = " ".join(sorted(pad.pressed)) or "-"
                name = pad.name or f"(パッドなし: {pad.error or '未接続'})"
                print(f"{name}  押している: {buttons}  {state}")
                last = shown
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        pad.close()


if __name__ == "__main__":
    main()
