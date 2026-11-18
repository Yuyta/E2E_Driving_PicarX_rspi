"""行きたい向き（意図）

走行モデルは「障害物を避けて進む」技能で、どちらへも曲がれる場所では上の層（M5 の見回りと行き先）が
意図を与える（ROADMAP の M3「行きたい向きの入力」）。
収集ではプロコンの L / R を押している間を記録する。
"""

from typing import List, Optional

INTENT_ANY = "any"
INTENT_LEFT = "left"
INTENT_RIGHT = "right"
# モデルの入力と出力の頭（ヘッド）の並び。変えると学習済みのモデルと合わなくなる
INTENTS = (INTENT_ANY, INTENT_LEFT, INTENT_RIGHT)


def intent_of(left: bool, right: bool) -> str:
    """左右どちらか一方だけを押している時はその向き。どちらも、またはどちらでもなければおまかせ"""
    if left == right:
        return INTENT_ANY
    return INTENT_LEFT if left else INTENT_RIGHT


def intent_index(value: Optional[str]) -> int:
    """記録の意図をモデルの番号にする。意図のない古い記録は おまかせ"""
    if value is None:
        return 0
    if value not in INTENTS:
        raise ValueError(f"Unknown intent: {value!r} (expected one of {INTENTS})")
    return INTENTS.index(value)


def one_hot(value: Optional[str]) -> List[float]:
    """モデルへ渡す形（any, left, right の順の 0 / 1）"""
    vector = [0.0] * len(INTENTS)
    vector[intent_index(value)] = 1.0
    return vector
