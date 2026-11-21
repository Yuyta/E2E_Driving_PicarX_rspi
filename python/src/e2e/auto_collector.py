"""決まったルールで走り、走行データを集める（ROADMAP の M3）

人の操作では「基本はまっすぐ、塞がれたら空いている側へ曲がる、分かれ道では意図の向きへ」を
揃えにくかったので、超音波の距離を見て次のルールで走る。画像から舵角を学ぶモデルの手本にする。

- 前が空いていれば、まっすぐ進む（意図は any）
- 前が近づいたら曲がる。意図がなければ右を先に試し、曲がれなければ下がって左を試す
  （any の頭は「右が空いていれば右、塞がっていれば左」を学ぶ）
- 一部の曲がりでは、曲がり始める前から意図（left / right）を付け、その向きへ曲がる。
  障害物の手前と、開けた場所での曲がりの両方で付ける。左右の数は釣り合うように選ぶ
- 曲がれなかった（下がることになった）曲がりのフレームは、学習から外す印（exclude）を付けて
  保存する。
  そのため、曲がりの間のラベルは結果が分かるまで書かずに持っておく
- 下がっている間のフレームは throttle が負になり、学習では自動で外れる

Pi の反射が、障害物（15cm）、段差、不明なセンサー値、指令の期限切れで止める。
このスクリプトが止まっても、最後の指令の期限（0.6秒）で車は止まる。

使い方（車が動く。周りを片付けてから）:
    uv run python -m src.e2e.auto_collector --seconds 150 --i-understand-the-car-will-move
"""

import argparse
import json
import random
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional

from src.e2e.intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT
from src.e2e.registry import body as body_versions
from src.models import DRIVE_SPEED, DriveDirection

LEFT, RIGHT = -1, 1
SIDE_INTENT = {LEFT: INTENT_LEFT, RIGHT: INTENT_RIGHT}
COLLECTOR_VERSION = "auto_v3"
# 前進の指令を出しているのに、模様の見える画像が止まっている（柔らかい物を押している）か、
# 画像が真っ暗（布に埋もれてカメラがふさがった）状態がこの時間続いたら、塞がれたとみなす。
# 超音波は布の反射を拾わず、食い違いの検出も画像が読めないと判断しない（2026-11-12）。
# 白い壁のように模様が少ないだけの時は、走っていても画像の動きが読めないので数えない
STALLED_S = 1.5
# 画像の明るさ（0〜255 の平均）がこれより暗ければ、カメラがふさがれたとみなす
DARK_MEAN = 35.0
# 舵0°での向きの変化を、信頼できる推定だけで測る時の条件（模様の点の数と、合う点の割合）
CONFIDENT_POINTS = 60
CONFIDENT_INLIERS = 0.6
# 指令を出してからこの時間は、動き出す途中なので数えない
STALL_GRACE_S = 0.6


@dataclass
class PolicyConfig:
    """走り方の値。距離は cm、時間は秒、舵角は度"""

    avoid_cm: float = 50.0  # これより近いと曲がり始める
    clear_cm: float = 90.0  # 曲がっている間、これより遠くなれば曲がり終えたとみなす
    clear_frames: int = 4  # 遠い状態がこのフレーム数続いたら曲がり終える
    # 曲がり始めてこの時間は、曲がり終えたとみなさない。斜めの壁では反射が戻らず、
    # 遠いと誤って読むため（2026-11-12 の試し走りで 0.3 秒で曲がり終えていた）
    avoid_min_s: float = 1.0
    approach_exit_frames: int = 6  # 近づいている途中で、遠い値がこのフレーム数続いたら意図を外す
    approach_cm: float = 100.0  # これより近づいたら、意図を付けるかを1回だけ決める
    open_cm: float = 150.0  # 開けた場所での曲がりは、これより遠い時だけ始める
    turn_deg: float = 30.0  # 曲がる時の舵角
    steer_rate_dps: float = 60.0  # 舵を切る速さ（人の収集と同じ）
    avoid_max_s: float = 4.0  # これだけ曲がっても前が空かなければ、曲がれなかったとみなす
    backup_s: float = 2.0  # 下がる時間（1.2 秒では角から抜けられなかった）
    # 下がった後、止まったまま舵を反対側へ切り替える時間。走りながら切り替えると、
    # 舵が回り切る前にまた壁へ近づいた（2026-11-12）
    pivot_s: float = 1.1
    lead_s: float = 1.2  # 開けた場所での曲がりで、意図を付けてから曲がり始めるまで
    open_turn_s: float = 1.6  # 開けた場所での曲がりの長さ
    cruise_min_s: float = 2.5  # まっすぐ走ってから、開けた場所での曲がりを考えるまで
    open_turn_per_s: float = 0.25  # 開けた場所で、1秒あたり曲がりを始める確率
    intent_at_obstacle: float = 0.7  # 障害物の手前で意図を付ける確率
    any_first_side: int = RIGHT  # 意図がない時に先に試す向き
    max_failures: int = 6  # 続けて曲がれなかったら、閉じ込められたとみなして止める


@dataclass
class Observation:
    now: float
    distance_cm: Optional[float]  # 何も捉えていない時（遠い時を含む）は None
    blocked: bool = False  # 前進を Pi が拒否した、または動けていない（食い違いが続いた）


@dataclass
class Action:
    drive: int  # 1: 前進、-1: 後退、0: 止まる
    target_deg: float
    intent: str
    state: str
    # 曲がりの始まりと結果。"begin"、"success"、"fail"、なし
    attempt: Optional[str] = None
    done: bool = False  # 閉じ込められたので止める


@dataclass
class RulePolicy:
    """超音波の距離から、走る向き、舵角、意図を決める。実機なしで確かめられるよう時刻は引数で受け取る"""

    config: PolicyConfig = field(default_factory=PolicyConfig)
    rng: random.Random = field(default_factory=random.Random)
    state: str = "cruise"
    side: int = RIGHT
    intent: str = INTENT_ANY
    since: float = 0.0
    clear_count: int = 0
    approach_decided: bool = False
    failures: int = 0
    tried_sides: List[int] = field(default_factory=list)
    intent_counts: Dict[str, int] = field(default_factory=lambda: {INTENT_LEFT: 0, INTENT_RIGHT: 0})
    _last_now: Optional[float] = None

    def _enter(self, state: str, now: float) -> None:
        self.state = state
        self.since = now
        self.clear_count = 0

    def _balanced_side(self) -> int:
        left, right = self.intent_counts[INTENT_LEFT], self.intent_counts[INTENT_RIGHT]
        if left == right:
            return self.rng.choice((LEFT, RIGHT))
        return LEFT if left < right else RIGHT

    def _begin_avoid(self, side: int, intent: str, now: float) -> Action:
        self.side = side
        self.intent = intent
        self.tried_sides.append(side)
        self._enter("avoid", now)
        return self._action(1, side * self.config.turn_deg, attempt="begin")

    def _action(self, drive: int, target: float, attempt: Optional[str] = None) -> Action:
        return Action(drive, target, self.intent, self.state, attempt)

    def step(self, obs: Observation) -> Action:
        c = self.config
        now = obs.now
        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        if self.intent != INTENT_ANY and self.state not in ("backup", "pivot"):
            self.intent_counts[self.intent] += 1
        d = obs.distance_cm
        near = d is not None and d < c.avoid_cm
        elapsed = now - self.since

        if self.state == "backup":
            if elapsed < c.backup_s:
                # 下がりながら曲がれなかった側へ舵を切ると、車の向きは反対側へ回る
                return self._action(-1, self.side * c.turn_deg)
            other = -self.side
            # 両側とも試したら、右から試し直す
            if other in self.tried_sides and self.side in self.tried_sides:
                self.tried_sides.clear()
                other = c.any_first_side
            self.side = other
            self._enter("pivot", now)
            return self._action(0, other * c.turn_deg)

        if self.state == "pivot":
            if elapsed < c.pivot_s:
                return self._action(0, self.side * c.turn_deg)
            return self._begin_avoid(self.side, INTENT_ANY, now)

        if self.state == "avoid":
            if d is None or d > c.clear_cm:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if self.clear_count >= c.clear_frames and elapsed >= c.avoid_min_s:
                self.failures = 0
                self.tried_sides.clear()
                self.intent = INTENT_ANY
                self.approach_decided = True  # 曲がり終えた直後に同じ壁で意図を決め直さない
                self._enter("cruise", now)
                return self._action(1, 0.0, attempt="success")
            if obs.blocked or elapsed > c.avoid_max_s:
                self.failures += 1
                self.intent = INTENT_ANY
                self._enter("backup", now)
                action = self._action(-1, self.side * c.turn_deg, attempt="fail")
                action.done = self.failures >= c.max_failures
                return action
            return self._action(1, self.side * c.turn_deg)

        # まっすぐ、意図を付けて近づく、開けた場所での曲がり（の前後）
        if obs.blocked:
            # まっすぐ走っている時に止められた（低い物にぶつかったなど）。曲がりと同じに扱う
            self.failures += 1
            self.intent = INTENT_ANY
            self.side = c.any_first_side
            self._enter("backup", now)
            action = self._action(-1, self.side * c.turn_deg)
            action.done = self.failures >= c.max_failures
            return action

        if near:
            if self.state in ("approach", "lead", "open_turn"):
                return self._begin_avoid(self.side, self.intent, now)
            return self._begin_avoid(c.any_first_side, INTENT_ANY, now)

        if self.state == "approach":
            if d is None or d > c.approach_cm + 30:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if self.clear_count >= c.approach_exit_frames:
                # 近づく前に前が空いた（斜めに抜けた）。意図を外して走り続ける
                self.intent = INTENT_ANY
                self.approach_decided = False
                self._enter("cruise", now)
            return self._action(1, 0.0)

        if self.state == "lead":
            if elapsed >= c.lead_s:
                self._enter("open_turn", now)
                return self._action(1, self.side * c.turn_deg)
            return self._action(1, 0.0)

        if self.state == "open_turn":
            if elapsed >= c.open_turn_s:
                self.intent = INTENT_ANY
                self._enter("cruise", now)
                return self._action(1, 0.0)
            return self._action(1, self.side * c.turn_deg)

        # cruise
        if d is None or d > c.approach_cm + 30:
            self.approach_decided = False
        if d is not None and d < c.approach_cm and not self.approach_decided:
            self.approach_decided = True
            if self.rng.random() < c.intent_at_obstacle:
                self.side = self._balanced_side()
                self.intent = SIDE_INTENT[self.side]
                self._enter("approach", now)
                return self._action(1, 0.0)
        open_space = d is None or d > c.open_cm
        if open_space and elapsed >= c.cruise_min_s and self.rng.random() < c.open_turn_per_s * dt:
            self.side = self._balanced_side()
            self.intent = SIDE_INTENT[self.side]
            self._enter("lead", now)
        return self._action(1, 0.0)


def ramp(current: float, target: float, rate_dps: float, dt: float) -> float:
    """舵を一定の速さで目標へ近づける"""
    step = rate_dps * max(dt, 0.0)
    if current < target:
        return min(target, current + step)
    return max(target, current - step)


class AttemptBuffer:
    """曲がりの間のラベルを持っておき、結果が分かったら書く。曲がれなかったら学習から外す印を付ける"""

    def __init__(self, write):
        self.write = write
        self.pending: List[dict] = []
        self.active = False

    def add(self, record: dict) -> None:
        if self.active:
            self.pending.append(record)
        else:
            self.write(record)

    def begin(self) -> None:
        self.flush()
        self.active = True

    def resolve(self, success: bool) -> None:
        for record in self.pending:
            if not success:
                record["exclude"] = True
                record["exclude_reason"] = "failed_turn"
            self.write(record)
        self.pending = []
        self.active = False

    def flush(self) -> None:
        # 結果が分からないまま終わった曲がりは、学習から外す
        if self.pending:
            self.resolve(success=False)
        self.active = False


def smoothed_distance(history: Deque[Optional[float]]) -> Optional[float]:
    """直近の距離の中央値。何も捉えていない値（None）が半分以上なら None"""
    values = [v for v in history if v is not None]
    if len(values) * 2 <= len(history):
        return None
    values.sort()
    return values[len(values) // 2]


class StallDetector:
    """前進しているはずなのに画像が動かない時間を数える"""

    def __init__(self, stalled_s: float = STALLED_S, grace_s: float = STALL_GRACE_S):
        self.stalled_s = stalled_s
        self.grace_s = grace_s
        self.since: Optional[float] = None

    def update(self, now: float, forward_for_s: Optional[float], view: str) -> bool:
        """forward_for_s: 前進の指令が続いている時間（前進していなければ None）。
        view: moving（画像が動いている）、still（模様が見えて止まっている）、
        dark（真っ暗）、unknown（模様が少なくて読めない）"""
        if forward_for_s is None or forward_for_s < self.grace_s or view not in ("still", "dark"):
            self.since = None
            return False
        if self.since is None:
            self.since = now
        if now - self.since >= self.stalled_s:
            self.since = None
            return True
        return False


def view_of(sample, brightness: float) -> str:
    """画像の状態。sample は動きの推定（MotionSample か None）"""
    if brightness < DARK_MEAN:
        return "dark"
    if sample is None or not sample.image_valid:
        return "unknown"
    return "moving" if sample.image_moving else "still"


def straight_drift(records: List[dict]) -> Dict[str, Optional[float]]:
    """舵0°で前進している間の向きの変化（°/s、負が左）を、信頼できる推定だけで測る"""
    values = []
    for r in records:
        command = r.get("command") or {}
        motion = r.get("motion") or {}
        if (
            abs(command.get("steering_deg", 1.0)) < 0.5
            and command.get("speed", 0) > 0
            and command.get("since_s", 0) > 1.0
            and motion.get("image_valid")
            and motion.get("points", 0) >= CONFIDENT_POINTS
            and motion.get("inlier_ratio", 0) >= CONFIDENT_INLIERS
            and motion.get("yaw_rate_dps") is not None
        ):
            values.append(float(motion["yaw_rate_dps"]))
    values.sort()
    median = values[len(values) // 2] if values else None
    return {"samples": len(values), "yaw_dps": None if median is None else round(median, 2)}


def run(
    seconds: float,
    output_base: Path,
    seed: Optional[int],
    min_battery_v: float,
    steer_trim_deg: float = 0.0,
) -> Path:
    import cv2
    import numpy as np

    from src.e2e.gui_collector import MOVE_TTL_MS, REVERSE_SPEED, sensor_fields
    from src.e2e.motion.monitor import CommandState, MotionMonitor
    from src.picarx_client.client import PicarxClient

    config = PolicyConfig()
    rng = random.Random(seed)
    policy = RulePolicy(config=config, rng=rng)
    client = PicarxClient()
    command_state = CommandState(time.time())
    monitor = MotionMonitor.from_config()

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = output_base / session_id
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_path = session_dir / "labels.jsonl"
    with open(session_dir / "collector.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "collector": COLLECTOR_VERSION,
                "seed": seed,
                "policy": asdict(config),
                # ラベルの舵角は補正前の値。車へ送る舵角はこれを足した値
                "steer_trim_deg": steer_trim_deg,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
        f.write("\n")
    # どの車体（装備と校正）で集めたかを残す（configs/body.toml の今の版）
    body_versions.record_session(session_dir)
    labels = open(labels_path, "a", encoding="utf-8")

    def write(record: dict) -> None:
        labels.write(json.dumps(record) + "\n")

    buffer = AttemptBuffer(write)
    stall = StallDetector()
    distances: Deque[Optional[float]] = deque(maxlen=3)
    steer_value = 0.0
    sent_angle: Optional[int] = None
    frame_count = 0
    stats = {"frames": 0, "fails": 0, "successes": 0, "refused": 0, "stalled": 0}
    stop_reason = "time"

    status = client.get_status_sync()
    if (
        status.sensors.battery.voltage is not None
        and status.sensors.battery.voltage < min_battery_v
    ):
        raise SystemExit(f"Battery {status.sensors.battery.voltage} V is below {min_battery_v} V")
    client.steer_sync(0)
    stream = client.open_camera_stream().start()
    print(f"Session: {session_dir}")
    start = time.time()
    last_seq = 0
    last_now: Optional[float] = None
    blocked_next = False
    try:
        while time.time() - start < seconds:
            frame = stream.wait_for_frame(last_seq, timeout=1.0)
            if frame is None:
                continue
            last_seq = frame.seq
            try:
                status = client.get_status_sync()
            except Exception:
                status = None
            now = time.time()
            sensors = sensor_fields(status)
            voltage = sensors["battery_v"]
            if voltage is not None and voltage < min_battery_v:
                stop_reason = f"battery {voltage} V"
                break
            distances.append(sensors["distance_cm"])
            img = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
            update = None
            if img is not None:
                update = monitor.update(img, now, sensors["distance_cm"], command_state.at(now))
            # 前へ進んでいるのに動けていない（引っかかった）時は、曲がれなかったとみなす
            current = command_state.at(now)
            stuck = (
                update is not None and update.newly_sustained == "no_motion" and current.speed > 0
            )
            brightness = float(img.mean()) if img is not None else 0.0
            stalled = stall.update(
                now,
                current.since_s if current.speed > 0 else None,
                view_of(None if update is None else update.sample, brightness),
            )
            if stalled:
                stats["stalled"] += 1

            # 保存するラベルは、このフレームの時点で送っていた指令（前のフレームで決めたもの）
            if img is not None:
                filename = f"frame_{frame_count:06d}.jpg"
                cv2.imwrite(str(images_dir / filename), cv2.resize(img, (160, 120)))
                command = command_state.at(now)
                record = {
                    "index": frame_count,
                    "timestamp": now,
                    "frame": filename,
                    "steering": round((sent_angle or 0) / 35.0, 4),
                    "throttle": 0.5 if command.speed > 0 else (-0.5 if command.speed < 0 else 0.0),
                    "intent": policy.intent,
                    "stream_seq": frame.seq,
                    **sensors,
                    "collector": COLLECTOR_VERSION,
                    "policy_state": policy.state,
                    "steer_trim_deg": steer_trim_deg,
                }
                if update is not None:
                    record.update(update.to_record())
                else:
                    record["command"] = command.to_record()
                buffer.add(record)
                frame_count += 1
                stats["frames"] = frame_count

            action = policy.step(
                Observation(
                    now, smoothed_distance(distances), blocked=blocked_next or stuck or stalled
                )
            )
            blocked_next = False
            if action.attempt == "begin":
                buffer.begin()
            elif action.attempt in ("success", "fail"):
                buffer.resolve(action.attempt == "success")
                stats["successes" if action.attempt == "success" else "fails"] += 1
                if action.attempt == "fail":
                    monitor.reset()
                    distances.clear()
            if action.done:
                stop_reason = "trapped (too many failed turns)"
                break

            dt = 0.0 if last_now is None else now - last_now
            last_now = now
            steer_value = ramp(steer_value, action.target_deg, config.steer_rate_dps, dt)
            angle = int(round(steer_value))
            if angle != sent_angle:
                client.steer_sync(int(round(angle + steer_trim_deg)))
                sent_angle = angle
                command_state.steer(angle, now)

            if action.drive > 0:
                ok = client.move_sync(DriveDirection.FORWARD, DRIVE_SPEED, MOVE_TTL_MS)
                speed = DRIVE_SPEED
            elif action.drive < 0:
                ok = client.move_sync(DriveDirection.BACKWARD, REVERSE_SPEED, MOVE_TTL_MS)
                speed = -REVERSE_SPEED
            else:
                client.stop_sync()
                ok, speed = True, 0
            if ok and speed:
                command_state.move(speed, MOVE_TTL_MS / 1000.0, now)
            elif not ok:
                command_state.stop(now)
                stats["refused"] += 1
                if action.drive < 0:
                    stop_reason = "backward refused"
                    break
                blocked_next = True
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    finally:
        try:
            client.stop_sync()
            client.steer_sync(0)
        finally:
            stream.stop()
            buffer.flush()
            labels.close()
    stats["intent_frames"] = policy.intent_counts
    with open(labels_path, encoding="utf-8") as f:
        stats["straight_drift"] = straight_drift([json.loads(line) for line in f])
    print(f"Stopped: {stop_reason}. {stats}")
    return session_dir


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Collect driving data with a rule-based driver")
    parser.add_argument("--seconds", type=float, default=150.0)
    parser.add_argument("--output", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min_battery_v", type=float, default=7.0)
    parser.add_argument(
        "--steer_trim", type=float, default=0.0, help="Degrees added to every steering command"
    )
    parser.add_argument("--i-understand-the-car-will-move", action="store_true", dest="confirmed")
    args = parser.parse_args(argv)
    if not args.confirmed:
        parser.error(
            "The car drives by itself. Clear the area and pass --i-understand-the-car-will-move"
        )
    run(args.seconds, Path(args.output), args.seed, args.min_battery_v, args.steer_trim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
