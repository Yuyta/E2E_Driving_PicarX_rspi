"""Pi 上の自律走行を PC から始めて、見守り、止める（ROADMAP の M3、走行のループは Pi で動く）

推論、避ける判断、後退、介入の記録は Pi（`rust/src/autopilot`）が行う。PC がなくても走れる。
ここでは次のことだけをする。

- 段差の校正をしてから、Pi の使用中のモデルで走行を始める
- Pi の状態を一定の間隔で表示する。Pi が走行を終えたら（引っかかりが続いた、カメラが止まった
  など）理由を表示して終わる
- Ctrl+C で止める（人の介入として Pi が記録する）
- `--gamepad` なら、パッドを操作すると人が運転を引き継ぐ。＋を押すか、8秒何も触らなければ
  自律に戻す（手を止めて考えている間に自律が割り込まないよう、短くしない）。Pi は人の指令が
  来ると自律の指令を止め、途絶えて 1.5 秒で自律に戻る。人が運転した間の画像と操作は、PC の
  `data/interventions/<run_id>/` に学習データの形で保存する。HOME は緊急停止（走行も
  終わる）。運転していない時に L / R を押すと、行きたい向き（意図）を Pi へ送る
- `--gamepad` の時は、介入してほしい時に知らせる。Pi が自分で抜け出せない時（引っかかりや段差が
  続く、後退を繰り返す）は、走行を終えずに止まって待つので、音とパッドの振動と表示で
  「介入してください」と知らせ、人が抜け出させるまで5秒ごとに繰り返す。抜け出させたら＋で
  自律に戻す。障害物や段差で止まった時、引っかかった時は、短い音と表示で注意を促す

使い方:
    uv run python -m src.e2e.pi_autopilot --speed 50
    uv run python -m src.e2e.pi_autopilot --gamepad --intent left
"""

import argparse
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import httpx
import numpy as np

from src.e2e.intent import INTENT_ANY, INTENTS
from src.models import DRIVE_SPEED, DriveDirection
from src.picarx_client.client import AutopilotError, PicarxClient

try:
    from .gamepad import Gamepad, PadInput
    from .intervention import CorrectionRecorder, Takeover
except ImportError:
    from src.e2e.gamepad import Gamepad, PadInput
    from src.e2e.intervention import CorrectionRecorder, Takeover

MAX_STEER_DEG = 35
# 人が運転している間の指令の期限
HUMAN_TTL_MS = 400
STATUS_PERIOD_S = 0.5
# 人の助けを待っている間、この間隔で知らせ直す
HELP_REMIND_S = 5.0
# 引き継いだ運転は＋で返す。押し忘れた時だけ、この時間何も触らなければ自律に戻す
TAKEOVER_RELEASE_S = 8.0
# 人の指令が止められた時の表示の間隔
REFUSED_REMIND_S = 2.0


def _beep(pattern) -> None:
    """(周波数 Hz, 長さ ms) の並びを鳴らす。Windows 以外は端末のベル"""
    try:
        import winsound

        for freq, ms in pattern:
            winsound.Beep(freq, ms)
    except (ImportError, RuntimeError):
        print("\a", end="", flush=True)


class Notifier:
    """人に介入を頼む知らせ。音は別のスレッドで鳴らし、見守りの周期を止めない"""

    HELP_SOUND = ((880, 200), (660, 200), (880, 200))
    CAUTION_SOUND = ((990, 120),)

    def __init__(self, gamepad=None, out=print, beep=_beep):
        self.gamepad = gamepad
        self.out = out
        self.beep = beep

    def _sound(self, pattern) -> None:
        threading.Thread(target=self.beep, args=(pattern,), daemon=True).start()

    def help(self, reason: str) -> None:
        self.out(
            f"\n[介入してください] 車が自分で抜け出せません: {reason}\n"
            "  パッドで車を動かして抜け出させてください。＋で自律に戻ります。"
            "終えるなら Ctrl+C"
        )
        self._sound(self.HELP_SOUND)
        if self.gamepad is not None and hasattr(self.gamepad, "rumble"):
            self.gamepad.rumble(1.0, 600)

    def caution(self, event: str) -> None:
        self.out(f"\n[注意] {event}（危なければパッドで引き継いでください）")
        self._sound(self.CAUTION_SOUND)


def response_message(response: httpx.Response) -> str:
    """Pi の応答の message（なければ本文）"""
    try:
        message = response.json().get("message")
    except ValueError:
        message = None
    return message or response.text or f"HTTP {response.status_code}"


def format_status(status: dict) -> str:
    def num(key, fmt):
        value = status.get(key)
        return "--" if value is None else format(value, fmt)

    counts = status.get("interventions") or {}
    return (
        f"[{status.get('phase', '?'):>8}] intent {status.get('intent') or '-':<5} "
        f"steer {num('steer_deg', '+d')}° spd {num('speed', 'd')} "
        f"dist {num('distance_cm', '.0f')}cm cycle {num('cycle_ms', '.0f')}ms "
        f"infer {num('inference_ms', '.1f')}ms | human {counts.get('human', 0)} "
        f"safety {counts.get('safety', 0)}"
    )


class PiAutopilotSession:
    """Pi の自律走行の1回分。時刻は引数で受け取り、実機なしで確かめられるようにする"""

    def __init__(
        self,
        client,
        gamepad=None,
        corrections_dir="data/interventions",
        speed: int = DRIVE_SPEED,
        stream=None,
        notifier: Optional[Notifier] = None,
    ):
        self.client = client
        self.gamepad = gamepad
        self.corrections_dir = Path(corrections_dir)
        self.speed = speed
        self.stream = stream
        self.takeover = Takeover(release_s=TAKEOVER_RELEASE_S)
        self.takeover_count = 0
        self.info: dict = {}
        self.corrections: Optional[CorrectionRecorder] = None
        self.last_status: dict = {}
        self._next_status = 0.0
        self._last_seq = 0
        self._sent_intent: Optional[str] = None
        # 見守る人（パッド）がいる時だけ、Pi は止まって助けを待つ
        self.wait_for_help = gamepad is not None
        self.notifier = notifier or Notifier(gamepad)
        self.help_count = 0
        self._safety_seen = 0
        self._next_help_notice: Optional[float] = None
        self._next_refused_notice = 0.0
        self.cliff_guard = True

    def start(self, intent: str = INTENT_ANY) -> dict:
        # 前の PC が途中で落ちると、Pi の走行が残って次を始められない。残っていれば終える
        leftover = self.client.autopilot_status_sync()
        if leftover.get("running"):
            run_id = leftover.get("run_id")
            self.notifier.out(f"[NOTE] A run was still going on the Pi ({run_id}). Stopping it.")
            self.client.autopilot_stop_sync(intervention=False)
        status = self.client.get_status_sync()
        self.cliff_guard = status.cliff.guard_enabled
        if status.cliff.guard_enabled:
            self.client.calibrate_floor_sync()
        self.info = self.client.autopilot_start_sync(
            speed=self.speed, intent=intent, wait_for_help=self.wait_for_help
        )
        self._sent_intent = intent
        identity = {"model_version": self.info.get("model_version")}
        self.corrections = CorrectionRecorder(
            self.corrections_dir, self.info.get("run_id", "pi_run"), identity
        )
        return self.info

    def step(self, now: float) -> Optional[str]:
        """1周分。Pi の走行が終わっていれば、その理由を返す"""
        pad = self.gamepad.poll() if self.gamepad is not None else None
        hand = self.takeover.update(pad, now)
        if hand.emergency:
            self.client.emergency_stop_sync()
            return "emergency stop (HOME on the gamepad)"
        if hand.started:
            self.takeover_count += 1
            self.notifier.out("\n[TAKEN OVER] You are driving. Press + to hand back.")
        if hand.ended and not hand.emergency:
            self.notifier.out("\n[AUTOPILOT] Handed back to the autopilot.")
        if hand.human:
            self._drive_by_hand(pad or PadInput(), now)
        elif pad is not None and pad.intent != self._sent_intent:
            # 運転していない時の L / R は、上の層の意図として送る
            self.client.autopilot_intent_sync(pad.intent)
            self._sent_intent = pad.intent

        if now >= self._next_status:
            self._next_status = now + STATUS_PERIOD_S
            self.last_status = self.client.autopilot_status_sync()
            if not self.last_status.get("running"):
                return self.last_status.get("stopped_reason") or "stopped"
            self._notify(self.last_status, now)
        return None

    def _notify(self, status: dict, now: float) -> None:
        """Pi の状態の変化から、介入を頼むか注意を促す"""
        safety = (status.get("interventions") or {}).get("safety", 0)
        if safety > self._safety_seen:
            self._safety_seen = safety
            if status.get("phase") != "needs_help":
                self.notifier.caution(status.get("last_event") or "safety stop")
        if status.get("phase") == "needs_help":
            if self._next_help_notice is None:
                self.help_count += 1
            if self._next_help_notice is None or now >= self._next_help_notice:
                self.notifier.help(status.get("help_reason") or status.get("detail") or "")
                self._next_help_notice = now + HELP_REMIND_S
        else:
            self._next_help_notice = None

    def _drive_by_hand(self, pad: PadInput, now: float) -> None:
        steer = int(round(pad.steer * MAX_STEER_DEG))
        moving = pad.drive != 0 and not pad.stop
        self.client.steer_sync(steer)
        direction = DriveDirection.BACKWARD if pad.drive < 0 else DriveDirection.FORWARD
        # 止まる時も速度 0 の指令を送る（停止の指令は Pi の自律走行を終えてしまうので使わない）
        accepted = self.client.move_sync(
            direction, speed=self.speed if moving else 0, ttl_ms=HUMAN_TTL_MS
        )
        if moving and not accepted and now >= self._next_refused_notice:
            # Pi の安全装置が人の指令も止めている。黙っていると「介入できない」ように見える
            self._next_refused_notice = now + REFUSED_REMIND_S
            way = "forward" if pad.drive > 0 else "backward"
            self.notifier.out(
                f"\n[REFUSED] The Pi blocked driving {way} (obstacle or floor sensors). "
                "Try the other direction."
            )
        self._record_correction(pad if moving else replace(pad, drive=0), now, steer, moving)

    def _record_correction(self, pad: PadInput, now: float, steer: int, moving: bool) -> None:
        if self.stream is None or self.corrections is None:
            return
        frame = self.stream.wait_for_frame(self._last_seq, timeout=0.05)
        if frame is None:
            return
        self._last_seq = frame.seq
        image = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return
        try:
            distance = self.client.get_status_sync().sensors.distance
            cm, far = distance.cm, bool(distance.far)
        except httpx.HTTPError:
            cm, far = None, False
        self.corrections.record(
            cv2.resize(image, (160, 120)),
            now,
            pad,
            self.speed if moving else 0,
            steer,
            self.last_status.get("intent") or INTENT_ANY,
            cm,
            far,
            self.takeover_count,
        )

    def stop(self) -> dict:
        return self.client.autopilot_stop_sync(intervention=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the autopilot on the Pi and watch it")
    parser.add_argument("--speed", type=int, default=DRIVE_SPEED)
    parser.add_argument("--intent", choices=INTENTS, default=INTENT_ANY)
    parser.add_argument("--gamepad", action="store_true", help="Take over with the gamepad")
    parser.add_argument("--corrections_dir", default="data/interventions")
    args = parser.parse_args(argv)

    client = PicarxClient(timeout=2.0)
    gamepad = Gamepad() if args.gamepad else None
    stream = client.open_camera_stream().start() if args.gamepad else None
    session = PiAutopilotSession(client, gamepad, args.corrections_dir, args.speed, stream)
    try:
        model = client.autopilot_model_sync()
        if not model.get("verified"):
            print(f"[START REFUSED] Model on the Pi: {model.get('error')}")
            return 1
        info = session.start(args.intent)
    except httpx.HTTPStatusError as e:
        # 段差の校正を断られた時など。Pi の理由をそのまま出す
        print(f"[START REFUSED] {response_message(e.response)}")
        return 1
    except (AutopilotError, httpx.HTTPError) as e:
        print(f"[START REFUSED] {e}")
        return 1
    print(
        f"Autopilot on the Pi: run {info['run_id']}, model {info['model_version']}, "
        f"server {info.get('server_build', 'unknown')}"
    )
    if not session.cliff_guard:
        print("Floor sensors on the Pi: OFF (PICARX_DISABLE_CLIFF_GUARD=1). Keep away from steps.")
    print("Press Ctrl+C to stop." + (" Move the gamepad to take over." if gamepad else ""))
    reason = None
    try:
        while reason is None:
            try:
                reason = session.step(time.time())
            except httpx.HTTPError as e:
                # 通信が切れても Pi は走り続けるか、自分で止まる。表示だけして続ける
                print(f"\r[NO CONNECTION] {e}", end="", flush=True)
            if session.last_status:
                print("\r" + format_status(session.last_status), end="", flush=True)
            time.sleep(0.05)
        print(f"\n[ENDED ON THE PI] {reason}")
    except KeyboardInterrupt:
        print("\n[MANUAL INTERVENTION] Stopping the autopilot.")
        try:
            final = session.stop()
            print(format_status(final))
        except (AutopilotError, httpx.HTTPError) as e:
            print(f"Could not reach the Pi ({e}). Use the emergency stop if the car still moves.")
            return 1
    finally:
        if stream is not None:
            stream.stop()
    if session.corrections is not None and session.corrections.count:
        print(
            f"Corrections: {session.corrections.session_dir} ({session.corrections.count} frames)"
        )
    if session.help_count:
        print(f"Help requested {session.help_count} time(s) during the run.")
    print(f"Run log on the Pi: {info.get('run_dir')}")
    print("Copy the Pi run logs with: uv run python -m src.e2e.deploy.pi pull-runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
