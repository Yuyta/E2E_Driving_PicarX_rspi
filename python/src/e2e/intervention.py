"""自律走行の介入の記録と、人の修正操作（ROADMAP の M3「介入時に人の修正操作を記録」）

- 走行ごとの記録（`RunLog`）: `data/runs/<run_id>/run.json` に、モデルの版、学習に使ったデータの
  版、コードの版、自律で走った時間を、`interventions.jsonl` に介入を1行ずつ残す。介入は人の介入
  （パッドで操作を引き継いだ、Ctrl+C）と、安全のための自動停止（障害物、段差、通信、食い違い）に
  分ける。主指標は「10分あたりの人の介入回数」（`src.e2e.evaluation.interventions` で集計）
- 引き継ぎ（`Takeover`）: 走行中にパッドを操作したら人が運転を引き継ぎ、入力がなくなって
  しばらくたったら自律に戻す
- 修正操作の記録（`CorrectionRecorder`）: 引き継いでいる間の画像と人の操作を、`DrivingDataset` で
  読める形で `data/interventions/<run_id>/` に保存する。収集のデータ（data/raw）とは分け、学習に
  使う時は `--data_dir data/raw data/interventions` のように明示する

時刻は引数で受け取り、実機なしで確かめられるようにする。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2

try:
    from .gamepad import PadInput
    from .intent import INTENT_ANY
    from .training import provenance
except ImportError:
    from src.e2e.gamepad import PadInput
    from src.e2e.intent import INTENT_ANY
    from src.e2e.training import provenance

HUMAN = "human"
SAFETY = "safety"
# 人の介入とみなす記録の種類。ほかは安全のための自動停止
HUMAN_TRIGGERS = ("human_takeover", "manual_intervention")


def model_identity(model_path) -> Dict[str, Any]:
    """モデルの隣の metadata.json から、版と学習に使ったデータの版。なければ分かる範囲だけ"""
    path = Path(model_path)
    identity: Dict[str, Any] = {
        "model_path": str(path),
        "model_version": path.parent.name,
        "data_fingerprint": None,
        "model_commit": None,
    }
    try:
        with open(path.parent / "metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError):
        return identity
    identity["model_version"] = metadata.get("model_version") or identity["model_version"]
    identity["data_fingerprint"] = (metadata.get("data") or {}).get("fingerprint")
    identity["model_commit"] = (metadata.get("code") or {}).get("commit")
    return identity


class RunLog:
    """1回の自律走行の記録。介入のたびに追記し、run.json はその都度書き直す（途中で落ちても残る）"""

    def __init__(self, base_dir, model_path, settings: Optional[Dict[str, Any]] = None, now=None):
        now = time.time() if now is None else now
        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        self.dir = Path(base_dir) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.started_at = now
        self.autonomous_s = 0.0
        self.human_s = 0.0
        self.counts = {HUMAN: 0, SAFETY: 0}
        self.info: Dict[str, Any] = {
            "run_id": self.run_id,
            **model_identity(model_path),
            "code": provenance.git_state(),
            "settings": settings or {},
        }
        self._write(now)

    def add_time(self, seconds: float, human: bool) -> None:
        """周期ごとに、自律と人の運転のどちらだったかの時間を足す"""
        if human:
            self.human_s += max(0.0, seconds)
        else:
            self.autonomous_s += max(0.0, seconds)

    def intervention(self, trigger: str, now: float, note: str = "", **extra) -> Dict[str, Any]:
        kind = HUMAN if trigger in HUMAN_TRIGGERS else SAFETY
        self.counts[kind] += 1
        record = {
            "time": now,
            "elapsed_s": round(now - self.started_at, 3),
            "autonomous_s": round(self.autonomous_s, 3),
            "trigger": trigger,
            "kind": kind,
            "note": note,
            **extra,
        }
        with open(self.dir / "interventions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write(now)
        return record

    def close(self, now=None) -> None:
        self._write(time.time() if now is None else now, ended=True)

    def _write(self, now: float, ended: bool = False) -> None:
        data = {
            **self.info,
            "started_at": self.started_at,
            "updated_at": now,
            "ended": ended,
            "autonomous_s": round(self.autonomous_s, 3),
            "human_s": round(self.human_s, 3),
            "interventions": dict(self.counts),
        }
        with open(self.dir / "run.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")


def pad_active(pad: Optional[PadInput]) -> bool:
    """人が運転しようとしているか（進む・下がる、舵、停止のどれか）"""
    if pad is None:
        return False
    return pad.drive != 0 or pad.steer != 0.0 or pad.stop


@dataclass
class TakeoverDecision:
    human: bool  # 人が運転している
    started: bool = False  # この周期で引き継いだ
    ended: bool = False  # この周期で自律に戻した
    emergency: bool = False  # HOME。走行を終える


@dataclass
class Takeover:
    """パッドの操作で人が運転を引き継ぎ、入力がなくなって release_s たったら自律に戻す

    ＋（resume）を押せば、待たずに自律に戻す。release_s を長くすれば、人が手を止めて考えている
    間に自律が割り込まない
    """

    release_s: float = 1.5
    human: bool = False
    started_at: float = 0.0
    last_input: float = 0.0

    def update(self, pad: Optional[PadInput], now: float) -> TakeoverDecision:
        if pad is not None and pad.emergency:
            was_human = self.human
            self.human = False
            return TakeoverDecision(human=False, ended=was_human, emergency=True)
        if self.human and pad is not None and pad.resume:
            self.human = False
            return TakeoverDecision(human=False, ended=True)
        active = pad_active(pad)
        if active:
            self.last_input = now
            if not self.human:
                self.human = True
                self.started_at = now
                return TakeoverDecision(human=True, started=True)
            return TakeoverDecision(human=True)
        if self.human and now - self.last_input >= self.release_s:
            self.human = False
            return TakeoverDecision(human=False, ended=True)
        return TakeoverDecision(human=self.human)


@dataclass
class CorrectionRecorder:
    """人が運転している間の画像と操作を、収集と同じ形（labels.jsonl と images/）で保存する"""

    base_dir: Path
    run_id: str
    identity: Dict[str, Any] = field(default_factory=dict)
    count: int = 0

    @property
    def session_dir(self) -> Path:
        return Path(self.base_dir) / self.run_id

    def record(
        self,
        frame_bgr,
        now: float,
        pad: PadInput,
        speed: int,
        steer_deg: float,
        policy_intent: str,
        distance_cm: Optional[float],
        distance_far: bool,
        takeover_id: int,
    ) -> None:
        images = self.session_dir / "images"
        images.mkdir(parents=True, exist_ok=True)
        frame = f"frame_{self.count:06d}.jpg"
        cv2.imwrite(str(images / frame), frame_bgr)
        signed = speed if pad.drive > 0 else (-speed if pad.drive < 0 else 0)
        # 人が L / R を押していればその向き。押していなければ、引き継いだ時に自律が使っていた意図
        intent = pad.intent if pad.intent != INTENT_ANY else policy_intent
        record = {
            "index": self.count,
            "timestamp": now,
            "frame": frame,
            "steering": float(pad.steer),
            "throttle": signed / 100.0,
            "intent": intent,
            "intent_pressed": pad.intent,
            "policy_intent": policy_intent,
            "distance_cm": distance_cm,
            "distance_far": distance_far,
            "command": {"speed": signed, "steering_deg": float(steer_deg)},
            "collector": "human_correction",
            "run_id": self.run_id,
            "takeover_id": takeover_id,
            "model_version": self.identity.get("model_version"),
            "data_fingerprint": self.identity.get("data_fingerprint"),
        }
        with open(self.session_dir / "labels.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.count += 1
