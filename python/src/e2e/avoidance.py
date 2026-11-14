"""障害物を避ける判断（自律走行で、走行モデルへ渡す意図を決める）

走行モデルは「おまかせ」ではまっすぐ進み、曲がる向きは意図で受け取る（ROADMAP の M3、
2026-11-14 に曲がりの区間へ曲がった向きの意図を付け直した）。そこで、いつ、どちらへ曲がるかを
ここで決める。収集のルール（auto_collector）と同じ考え方にする。

- 曲がり始め: 超音波が近い、またはモデルの「前方が塞がっている確率」が高い状態が続いた。
  カメラの判断は、超音波で見えない細い家具の脚などへの備え
- 向き: 上の層（M5 の見回りと行き先）の意図が左右ならその向き。おまかせなら右を先に試し、
  曲がれなかったら次は反対側
- 曲がり終え: 超音波もカメラも空いている状態が続いた
- 減速: カメラが塞がっていそうと判断している間は、確実に動く最低の速度まで落とす

時刻は引数で受け取り、実機なしで確かめられるようにする。
"""

from dataclasses import dataclass, field
from typing import Optional

try:
    from .intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, INTENTS
except ImportError:
    from src.e2e.intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, INTENTS

CRUISE = "cruise"
AVOID = "avoid"

_OTHER_SIDE = {INTENT_LEFT: INTENT_RIGHT, INTENT_RIGHT: INTENT_LEFT}


@dataclass
class AvoidanceConfig:
    """距離は cm、時間は秒、確率は 0〜1。距離と時間は auto_collector の PolicyConfig に合わせる"""

    avoid_cm: float = 50.0  # これより近いと曲がり始める
    clear_cm: float = 90.0  # 曲がっている間、これより遠ければ空いているとみなす
    clear_frames: int = 4  # 空いている状態がこのフレーム数続いたら曲がり終える
    avoid_min_s: float = 1.0  # 曲がり始めてこの時間は、曲がり終えたとみなさない
    avoid_max_s: float = 4.0  # これだけ曲がっても空かなければ、反対側へ切り替える
    first_side: str = INTENT_RIGHT  # おまかせの時に先に試す向き
    # カメラの判断。入りと出で値を変え、境目で行ったり来たりしないようにする
    blocked_enter: float = 0.6  # 塞がっている確率がこれ以上で「塞がっていそう」
    blocked_exit: float = 0.4  # 曲がっている間、これより低ければ空いているとみなす
    blocked_frames: int = 2  # 塞がっていそうな状態がこのフレーム数続いたら曲がり始める
    slow_probability: float = 0.5  # これ以上なら最低の速度まで落とす

    def __post_init__(self) -> None:
        if self.first_side not in _OTHER_SIDE:
            raise ValueError(f"first_side must be left or right, got {self.first_side!r}")
        if not 0.0 <= self.blocked_exit <= self.blocked_enter <= 1.0:
            raise ValueError("Expected 0 <= blocked_exit <= blocked_enter <= 1")


@dataclass
class Decision:
    intent: str  # 走行モデルへ渡す意図
    slow: bool  # 最低の速度まで落とす
    state: str
    # 曲がりの出来事。"begin"、"success"、"switch"（曲がれず反対側へ）、なし
    event: Optional[str] = None
    reason: str = ""


@dataclass
class AvoidancePolicy:
    config: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    state: str = CRUISE
    side: str = INTENT_RIGHT
    next_side: Optional[str] = None  # おまかせで次に試す向き。None なら first_side
    since: float = 0.0
    clear_count: int = 0
    blocked_count: int = 0
    failures: int = 0  # 続けて曲がれなかった回数（成功すると 0）

    def step(
        self,
        now: float,
        distance_cm: Optional[float],
        blocked_probability: Optional[float] = None,
        upper_intent: str = INTENT_ANY,
    ) -> Decision:
        """distance_cm は反射が戻らない（遠い）時と分からない時は None。
        blocked_probability はモデルがその出力を持たない時は None"""
        if upper_intent not in INTENTS:
            raise ValueError(f"Unknown intent: {upper_intent!r}")
        c = self.config
        camera_blocked = blocked_probability is not None and blocked_probability >= c.blocked_enter
        self.blocked_count = self.blocked_count + 1 if camera_blocked else 0
        slow = blocked_probability is not None and blocked_probability >= c.slow_probability
        near = distance_cm is not None and distance_cm < c.avoid_cm

        if self.state == CRUISE:
            seen_by_camera = self.blocked_count >= c.blocked_frames
            if near or seen_by_camera:
                if upper_intent in _OTHER_SIDE:
                    side = upper_intent
                else:
                    side = self.next_side or c.first_side
                self._enter_avoid(side, now)
                reason = "ultrasonic" if near else "camera"
                return Decision(self.side, slow, self.state, "begin", reason)
            return Decision(upper_intent, slow, self.state)

        # 曲がっている
        ultrasonic_clear = distance_cm is None or distance_cm > c.clear_cm
        camera_clear = blocked_probability is None or blocked_probability < c.blocked_exit
        self.clear_count = self.clear_count + 1 if (ultrasonic_clear and camera_clear) else 0
        elapsed = now - self.since
        if self.clear_count >= c.clear_frames and elapsed >= c.avoid_min_s:
            self.state = CRUISE
            self.failures = 0
            self.next_side = None
            self.clear_count = 0
            self.blocked_count = 0
            return Decision(upper_intent, slow, self.state, "success")
        if elapsed > c.avoid_max_s:
            return self._switch_side(now, slow, "timeout")
        return Decision(self.side, slow, self.state)

    def on_stopped_by_obstacle(self, now: float) -> Decision:
        """Pi が障害物で前進を止めた。曲がれなかったので、次は反対側へ曲がる"""
        if self.state == CRUISE:
            # まっすぐ走っていて止められた（超音波の手前で急に現れた物など）
            self._enter_avoid(self.next_side or self.config.first_side, now)
            self.failures += 1
            return Decision(self.side, True, self.state, "begin", "stopped")
        return self._switch_side(now, True, "stopped")

    def _enter_avoid(self, side: str, now: float) -> None:
        self.state = AVOID
        self.side = side
        self.since = now
        self.clear_count = 0

    def _switch_side(self, now: float, slow: bool, reason: str) -> Decision:
        self.failures += 1
        other = _OTHER_SIDE[self.side]
        self.next_side = other
        self._enter_avoid(other, now)
        return Decision(self.side, slow, self.state, "switch", reason)
