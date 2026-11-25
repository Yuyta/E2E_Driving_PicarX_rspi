"""Control Center の「構成」画面: 役割ごとの部品、安全に関わる部分の隔離、部品ごとの版

部品の一覧はここで決める（ツールの一覧の `workflows.py` と同じ考え方）。中身のコードは変えず、
どの部品が何の役割を持ち、どこで動き、安全にどう関わり、版を何で確かめるかだけを書く。

安全の区分:
    core     安全の要。Pi の中だけで、どの判断や指示よりも先に止める
    related  安全に関わる。指令は安全の要の許可を通る、または読み込む物を確かめる
    isolated 安全から切り離す。モーターを直接動かせず、意図や指令を Pi の API へ渡すだけ

版の出どころ（versions のキー）:
    pi_server  Pi の走行サーバーのビルド（最後の走行の記録 run.json の code.server）
    model      Pi で使用中の走行モデル（デプロイの記録、なければ最後の走行）
    pc_code    PC のコードのコミット（git）
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

SAFETY_LEVELS = ("core", "related", "isolated")
PLACES = ("pi", "pc")
VERSION_SOURCES = ("pi_server", "model", "pc_code")


@dataclass(frozen=True)
class Component:
    id: str
    title: str
    place: str
    role: str
    safety: str
    # 安全との関わり方（隔離のされ方）
    safety_note: str
    interface: str
    source: str
    version: str

    def __post_init__(self) -> None:
        if self.place not in PLACES:
            raise ValueError(f"{self.id}: place must be one of {PLACES}")
        if self.safety not in SAFETY_LEVELS:
            raise ValueError(f"{self.id}: safety must be one of {SAFETY_LEVELS}")
        if self.version not in VERSION_SOURCES:
            raise ValueError(f"{self.id}: version must be one of {VERSION_SOURCES}")


COMPONENTS = (
    # ---------------------------------------------------------------- Pi（体、Rust）
    Component(
        "reflex",
        "反射ループ",
        "pi",
        "障害物、段差、指令の期限切れ、分からないセンサー値、緊急停止で止める（約40Hz）",
        "core",
        "Pi の中だけで判断し、PC や LLM の指示より先に止める。PC が途切れても止まる",
        "走行の指令はすべてここの許可を通る",
        "rust/src/safety/",
        "pi_server",
    ),
    Component(
        "sensors",
        "センサーの監視",
        "pi",
        "超音波と床センサーを時刻付きで読み、古い値や分からない値を区別する",
        "related",
        "古い値や分からない値は、反射ループが安全側（止める）に扱う",
        "/api/status の sensors",
        "rust/src/sensor_monitor/, rust/src/picarx/",
        "pi_server",
    ),
    Component(
        "camera",
        "カメラ配信",
        "pi",
        "カメラの画像を常時配信する（約15fps）",
        "related",
        "画像が止まると自律走行を終える",
        "/api/camera/stream, /api/camera/snapshot",
        "rust/src/rpi_camera/",
        "pi_server",
    ),
    Component(
        "autopilot",
        "走行ループ",
        "pi",
        "走行モデルの推論、避ける判断、引っかかりの検出、後退、走行の記録",
        "related",
        "指令は反射ループの許可を通る。PC がなくても Pi だけで走る",
        "/api/autopilot/start, stop, intent, status",
        "rust/src/autopilot/",
        "pi_server",
    ),
    Component(
        "model_store",
        "モデルの保管",
        "pi",
        "送られたモデルのハッシュと入力の大きさを確かめてから使用中にし、1つ前へ戻せる",
        "related",
        "ハッシュが合わないモデルは読み込まない",
        "picar-x-rust model install / rollback",
        "rust/src/autopilot/store.rs, rust/src/model_cli.rs",
        "model",
    ),
    Component(
        "api",
        "走行 API",
        "pi",
        "PC からの指令と状態の窓口。トークンの確認、期限付きの走行の指令、緊急停止の保持",
        "related",
        "走行の時間と速度を Pi の側で確かめる。緊急停止は明示的に解除するまで保つ",
        "/api/status, /api/drive/*, /api/events",
        "rust/src/api/",
        "pi_server",
    ),
    # ---------------------------------------------------------------- PC（脳、Python）
    Component(
        "agent",
        "会話 Agent",
        "pc",
        "人と話し、自分の状況と記憶を踏まえて答える。動きは意図として出す",
        "isolated",
        "モーターを直接動かせない。意図や指令は Pi の API を通り、Pi が断れる",
        "Pi の API のクライアント",
        "python/src/agents/",
        "pc_code",
    ),
    Component(
        "perception",
        "音声と画像の認識",
        "pc",
        "音声認識（Whisper）と見る質問（VLM）を PC のローカルで行う",
        "isolated",
        "認識の結果は会話にだけ使い、走行の安全の根拠にしない",
        "会話 Agent から呼ぶ",
        "python/src/voice/, python/src/vision/",
        "pc_code",
    ),
    Component(
        "memory",
        "記憶",
        "pc",
        "会話と出来事を SQLite + FTS5 に残して検索する",
        "isolated",
        "走行に関わらない",
        "会話 Agent から呼ぶ",
        "python/src/memory/",
        "pc_code",
    ),
    Component(
        "mlops",
        "学習と評価",
        "pc",
        "走行データの管理、学習、交差検証、門番の試験、Pi へのデプロイ",
        "isolated",
        "門番の試験に合格しないモデルは Pi へ送らない。読み込むかは Pi が決める",
        "deploy.pi push（ssh）",
        "python/src/e2e/",
        "pc_code",
    ),
    Component(
        "gui",
        "Control Center と画面",
        "pc",
        "トレーサビリティ、採否、デプロイ、ジョブの画面と、会話・収集・データ確認の画面",
        "isolated",
        "車が動くツールは安全の確認をしてから起動する。緊急停止を送れる",
        "ブラウザ（127.0.0.1）",
        "python/src/gui/",
        "pc_code",
    ),
)


def _short(commit: Optional[str]) -> Optional[str]:
    return commit[:7] if commit else None


def versions(summary: Dict[str, Any], git: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """版の出どころごとの今の値。分からなければ value は None"""
    last_run = summary.get("last_pi_run") or {}
    deployed = summary.get("deployed") or {}
    model = deployed.get("version") or last_run.get("model_version")
    if deployed.get("version"):
        model_note = "デプロイの記録から"
    elif model:
        model_note = "最後の走行の記録から"
    else:
        model_note = "記録なし"
    return {
        "pi_server": {
            "value": last_run.get("server_build"),
            "at": last_run.get("started_at"),
            "note": "最後の走行の記録（run.json）から" if last_run else "走行の記録なし",
        },
        "model": {"value": model, "at": deployed.get("at"), "note": model_note},
        "pc_code": {
            "value": _short(git.get("commit")),
            "dirty": bool(git.get("dirty")),
            "note": "未コミットの変更あり" if git.get("dirty") else "git のコミット",
        },
        "body": {"value": summary.get("body_current"), "note": "configs/body.toml の今の版"},
    }


def view(summary: Dict[str, Any], git: Dict[str, Any]) -> Dict[str, Any]:
    """「構成」画面の中身（部品の一覧と版）"""
    items: List[Dict[str, Any]] = [asdict(c) for c in COMPONENTS]
    return {"components": items, "versions": versions(summary, git)}
