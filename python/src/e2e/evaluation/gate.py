"""門番の試験（回帰テスト）: 新しいモデルを、選んだ場面で1フレームずつ採点する（ROADMAP の M3）

データビューアで、過去にぶつかった、詰まった、避け損ねた場面と、普通にまっすぐ走れた場面に
門番のタグを付けておく。新しいモデル（Pi へ送る `model.onnx`）をそのフレームで動かし、
避ける側へ切るか、まっすぐの所で曲がらないかを採点する。`deploy.pi push` は合格しないモデルを
送らない（理由を表示し、人が明示すれば送れる）。

門番のタグ（データビューアの「タグ」に入力する。英語の別名も使える）:
    門番:左へ避ける  (gate:left)      左へ切るべき場面。舵角が -turn_min_deg 以下で合格
    門番:右へ避ける  (gate:right)     右へ切るべき場面。舵角が +turn_min_deg 以上で合格
    門番:まっすぐ    (gate:straight)  曲がらない場面。舵角の大きさが straight_max_deg 以下で合格

モデルへ渡す意図はタグで決める（記録の意図と注釈の上書きは使わない）。左右へ避ける場面は
その向きの意図、まっすぐの場面はおまかせ（any）。走行モデルは「障害物を避けて進む」技能で、
どちらへ曲がるかは意図で受け取り、おまかせではまっすぐ進む（ROADMAP の D-6）。いつ、どちらへ
曲がるかは Pi の避ける判断（`avoidance.rs`）が超音波で決める。そこで門番は「避ける向きを
渡されたら、その向きへはっきり切るか」と「何も渡されない時に、勝手に曲がらないか」を確かめる。
記録の意図は人がパッドで押していた向きで、場面の正解とは限らない（2026-11-24、v006 の結果が
画像ではなく記録の意図でほぼ決まっていたので改めた）。

合格の条件（すべて満たす）:
    1. 避ける場面（左右の合計）とまっすぐの場面が、それぞれ min_cases 以上ある
    2. 種類（左、右、まっすぐ）ごとの合格率が min_rate 以上
    3. 比べる元のモデル（Pi で使用中、なければ有効な版）より、種類ごとの合格率が max_drop を
       超えて下がっていない

記録した画像の再生なので、モデルの判断で次の画像が変わる閉じたループは試せない。
数値は提案値で、実機の結果を見て調整する。

使い方:
    uv run python -m src.e2e.evaluation.gate --model_dir models/v007
    uv run python -m src.e2e.evaluation.gate --model_dir models/v007 --baseline models/v006
    uv run python -m src.e2e.evaluation.gate --list
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from ..dataset.annotations import GATE_TAGS, AnnotationStore
    from ..intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, one_hot
    from ..registry.stages import deployed_version, load_deployments, load_registry
    from ..training.dataset import resolve_session_dirs
    from .metrics import MAX_STEERING_DEG, STRAIGHT_LIMIT
except ImportError:
    from src.e2e.dataset.annotations import GATE_TAGS, AnnotationStore
    from src.e2e.evaluation.metrics import MAX_STEERING_DEG, STRAIGHT_LIMIT
    from src.e2e.intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, one_hot
    from src.e2e.registry.stages import deployed_version, load_deployments, load_registry
    from src.e2e.training.dataset import resolve_session_dirs

GATE_FILE = "gate.json"
GATE_FORMAT = 1
MODEL_FILE = "model.onnx"

LEFT = "left"
RIGHT = "right"
STRAIGHT = "straight"
EXPECTATIONS = (LEFT, RIGHT, STRAIGHT)
# 採点の時にモデルへ渡す意図。避ける場面はその向き、まっすぐはおまかせ
INTENT_FOR = {LEFT: INTENT_LEFT, RIGHT: INTENT_RIGHT, STRAIGHT: INTENT_ANY}
EXPECTATION_LABELS = {LEFT: "左へ避ける", RIGHT: "右へ避ける", STRAIGHT: "まっすぐ"}


# 走行データの既定の置き場所（自動収集と、介入の修正操作）
DEFAULT_DATA_DIRS = ("data/raw", "data/interventions")
DEFAULT_MODELS_DIR = "models"

# 舵角（度）を受け取り、(フレーム数, 3, H, W) の画像と (フレーム数, 3) の意図から予測する関数
Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]


class GateError(RuntimeError):
    """試験を始められない（モデルや試験の場面がない、タグが矛盾している）"""


@dataclass(frozen=True)
class GateCase:
    session_dir: str
    frame: str
    expect: str
    intent: str = INTENT_ANY

    @property
    def session_id(self) -> str:
        return Path(self.session_dir).name

    @property
    def key(self) -> str:
        return f"{self.session_id}/{self.frame}"


@dataclass(frozen=True)
class GateCriteria:
    """角度は度、割合は 0〜1"""

    turn_min_deg: float = round(STRAIGHT_LIMIT * MAX_STEERING_DEG, 2)  # 5.25°
    straight_max_deg: float = round(STRAIGHT_LIMIT * MAX_STEERING_DEG, 2)
    min_rate: Dict[str, float] = field(
        default_factory=lambda: {LEFT: 0.7, RIGHT: 0.7, STRAIGHT: 0.9}
    )
    max_drop: float = 0.05
    min_cases: int = 5

    def __post_init__(self) -> None:
        if self.turn_min_deg <= 0 or self.straight_max_deg <= 0:
            raise ValueError("turn_min_deg and straight_max_deg must be positive")
        if not 0.0 <= self.max_drop <= 1.0:
            raise ValueError("max_drop must be between 0 and 1")
        for name, rate in self.min_rate.items():
            if name not in EXPECTATIONS or not 0.0 <= rate <= 1.0:
                raise ValueError(f"Bad min_rate entry {name}={rate}")


def passes(expect: str, steer_deg: float, criteria: GateCriteria) -> bool:
    if expect == LEFT:
        return steer_deg <= -criteria.turn_min_deg
    if expect == RIGHT:
        return steer_deg >= criteria.turn_min_deg
    if expect == STRAIGHT:
        return abs(steer_deg) <= criteria.straight_max_deg
    raise ValueError(f"Unknown expectation {expect!r}")


# ----------------------------------------------------------------------------- 場面を集める


def _read_records(session_dir: Path) -> Dict[str, Dict[str, Any]]:
    records = {}
    with open(session_dir / "labels.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("frame"):
                records[record["frame"]] = record
    return records


def collect_cases(
    data_dirs: Sequence[Union[str, Path]] = DEFAULT_DATA_DIRS,
    store: Optional[AnnotationStore] = None,
) -> Tuple[List[GateCase], List[str]]:
    """門番のタグの付いたフレームを集める。(場面, 注意) を返す

    無いフォルダは飛ばす。1つのフレームに矛盾するタグ（左と右など）があれば GateError。
    画像や記録のないフレームは注意に入れて飛ばす（消したセッションのタグが残っている時など）。
    """
    store = store or AnnotationStore()
    sessions: List[Path] = []
    for data_dir in data_dirs:
        if not Path(data_dir).exists():
            continue
        try:
            sessions.extend(resolve_session_dirs(data_dir))
        except FileNotFoundError:
            continue
    cases: List[GateCase] = []
    warnings: List[str] = []
    for session_dir in dict.fromkeys(sessions):
        notes = store.session(session_dir)
        tagged = {}
        for frame, entry in notes.frames.items():
            expects = {GATE_TAGS[t] for t in entry.get("tags", []) if t in GATE_TAGS}
            if len(expects) > 1:
                raise GateError(
                    f"{session_dir.name}/{frame} に矛盾する門番のタグがあります: {sorted(expects)}"
                )
            if expects:
                tagged[frame] = expects.pop()
        if not tagged:
            continue
        records = _read_records(session_dir)
        for frame in sorted(tagged):
            expect = tagged[frame]
            record = records.get(frame)
            image = session_dir / "images" / frame
            if record is None or not image.is_file():
                warnings.append(f"{session_dir.name}/{frame}: 記録か画像がないので飛ばしました")
                continue
            cases.append(GateCase(str(session_dir), frame, expect, INTENT_FOR[expect]))
    return cases, warnings


def suite_fingerprint(cases: Sequence[GateCase]) -> Optional[str]:
    """試験の場面の組み合わせのハッシュ（モデルどうしを同じ試験で比べたかを確かめる）"""
    if not cases:
        return None
    body = "\n".join(sorted(f"{c.session_id}/{c.frame}\t{c.expect}\t{c.intent}" for c in cases))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- モデルを動かす


def load_preprocessing(model_dir: Path) -> Dict[str, Any]:
    metadata_path = Path(model_dir) / "metadata.json"
    if not metadata_path.is_file():
        raise GateError(f"{metadata_path} がありません")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pre = metadata.get("preprocessing") or {}
    if not (pre.get("resize_hw") and pre.get("normalize_mean") and pre.get("normalize_std")):
        raise GateError(f"{metadata_path} に前処理の値がありません")
    return pre


def load_image(path: Path, preprocessing: Dict[str, Any]) -> np.ndarray:
    """学習（`training/dataset.py`）と Pi（`autopilot/preprocess.rs`）と同じ前処理。(3, H, W)"""
    from PIL import Image

    height, width = preprocessing["resize_hw"]
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(preprocessing["normalize_mean"], dtype=np.float32)
    std = np.asarray(preprocessing["normalize_std"], dtype=np.float32)
    return np.transpose((array - mean) / std, (2, 0, 1))


def onnx_predictor(model_dir: Path) -> Predictor:
    """Pi へ送る `model.onnx` で舵角（度）を出す関数"""
    import onnxruntime as ort

    path = Path(model_dir) / MODEL_FILE
    if not path.is_file():
        raise GateError(f"{path} がありません")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    names = [i.name for i in session.get_inputs()]
    has_intent = "intent" in names
    image_name = names[0]

    def predict(images: np.ndarray, intents: np.ndarray) -> np.ndarray:
        out = []
        # 書き出した ONNX のバッチの大きさは 1 なので1枚ずつ動かす
        for image, intent in zip(images, intents):
            feed = {image_name: image[None].astype(np.float32)}
            if has_intent:
                feed["intent"] = intent[None].astype(np.float32)
            steering = float(session.run(None, feed)[0].reshape(-1)[0])
            out.append(steering * MAX_STEERING_DEG)
        return np.asarray(out, dtype=np.float64)

    return predict


def predict_cases(
    cases: Sequence[GateCase], predictor: Predictor, preprocessing: Dict[str, Any]
) -> List[float]:
    if not cases:
        return []
    images = np.stack(
        [load_image(Path(c.session_dir) / "images" / c.frame, preprocessing) for c in cases]
    )
    intents = np.asarray([one_hot(c.intent) for c in cases], dtype=np.float32)
    return [round(float(v), 2) for v in predictor(images, intents)]


# ----------------------------------------------------------------------------- 採点


def score(
    cases: Sequence[GateCase], steer_deg: Sequence[float], criteria: GateCriteria
) -> Dict[str, Any]:
    """種類ごとの場面の数、合格の数、合格率と、フレームごとの結果"""
    if len(cases) != len(steer_deg):
        raise ValueError("cases and steer_deg must have the same length")
    by_kind = {name: {"cases": 0, "passed": 0, "rate": None} for name in EXPECTATIONS}
    frames = []
    for case, deg in zip(cases, steer_deg):
        ok = passes(case.expect, deg, criteria)
        kind = by_kind[case.expect]
        kind["cases"] += 1
        kind["passed"] += int(ok)
        frames.append(
            {
                "frame": case.key,
                "expect": case.expect,
                "intent": case.intent,
                "steer_deg": deg,
                "passed": ok,
            }
        )
    for kind in by_kind.values():
        if kind["cases"]:
            kind["rate"] = round(kind["passed"] / kind["cases"], 4)
    return {"by_kind": by_kind, "frames": frames}


def judge(
    result: Dict[str, Any],
    criteria: GateCriteria,
    baseline: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """不合格の理由の一覧（空なら合格）"""
    reasons = []
    by_kind = result["by_kind"]
    avoid_cases = by_kind[LEFT]["cases"] + by_kind[RIGHT]["cases"]
    if avoid_cases < criteria.min_cases:
        reasons.append(
            f"避ける場面が {avoid_cases} フレームしかありません（{criteria.min_cases} 以上必要）。"
            "データビューアで「門番:左へ避ける」か「門番:右へ避ける」のタグを付けてください"
        )
    if by_kind[STRAIGHT]["cases"] < criteria.min_cases:
        reasons.append(
            f"まっすぐの場面が {by_kind[STRAIGHT]['cases']} フレームしかありません"
            f"（{criteria.min_cases} 以上必要）。「門番:まっすぐ」のタグを付けてください"
        )
    for name in EXPECTATIONS:
        kind = by_kind[name]
        if kind["rate"] is None:
            continue
        need = criteria.min_rate.get(name, 0.0)
        if kind["rate"] < need:
            reasons.append(
                f"{EXPECTATION_LABELS[name]}: 合格率 {kind['rate']:.0%}"
                f"（{kind['passed']}/{kind['cases']}）が基準 {need:.0%} より低い"
            )
        if baseline is not None:
            base_rate = baseline["by_kind"][name]["rate"]
            if base_rate is not None and kind["rate"] < base_rate - criteria.max_drop:
                reasons.append(
                    f"{EXPECTATION_LABELS[name]}: 合格率 {kind['rate']:.0%} が比べる元の "
                    f"{base_rate:.0%} より {criteria.max_drop:.0%} を超えて下がった"
                )
    return reasons


def regressions(result: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    """比べる元は合格し、新しいモデルは不合格になったフレーム"""
    base = {f["frame"]: f["passed"] for f in baseline["frames"]}
    return [f["frame"] for f in result["frames"] if base.get(f["frame"]) and not f["passed"]]


# ----------------------------------------------------------------------------- 比べる元


def default_baseline(models_dir: Union[str, Path], candidate: str) -> Optional[str]:
    """比べる元の版: 配置の記録で Pi が使用中の版、なければ有効（approved）の最新の版"""
    models_dir = Path(models_dir)

    def usable(version: Optional[str]) -> bool:
        return bool(version) and version != candidate and (models_dir / version).is_dir()

    deployed = deployed_version(load_deployments()).get("version")
    if usable(deployed):
        return deployed
    try:
        registry = load_registry(models_dir)
    except ValueError:
        return None
    approved = [
        (entry.get("updated_at") or "", version)
        for version, entry in registry.get("models", {}).items()
        if entry.get("stage") == "approved" and usable(version)
    ]
    return max(approved)[1] if approved else None


# ----------------------------------------------------------------------------- まとめ


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_gate(
    model_dir: Union[str, Path],
    cases: Sequence[GateCase],
    baseline_dir: Optional[Union[str, Path]] = None,
    criteria: Optional[GateCriteria] = None,
    predictor_factory: Callable[[Path], Predictor] = onnx_predictor,
    warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    """試験をして結果（`gate.json` の中身）を返す。合否は `passed`"""
    criteria = criteria or GateCriteria()
    model_dir = Path(model_dir)
    preprocessing = load_preprocessing(model_dir)
    result = score(
        cases, predict_cases(cases, predictor_factory(model_dir), preprocessing), criteria
    )

    baseline_report = None
    if baseline_dir is not None:
        baseline_dir = Path(baseline_dir)
        base_pre = load_preprocessing(baseline_dir)
        baseline = score(
            cases, predict_cases(cases, predictor_factory(baseline_dir), base_pre), criteria
        )
        baseline_report = {
            "version": baseline_dir.name,
            "by_kind": baseline["by_kind"],
            "regressed_frames": regressions(result, baseline),
            "frames": baseline["frames"],
        }
    reasons = judge(result, criteria, baseline_report)
    return {
        "format": GATE_FORMAT,
        "model_version": model_dir.name,
        "model_sha256": _sha256(model_dir / MODEL_FILE),
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "passed": not reasons,
        "reasons": reasons,
        "suite": {
            "fingerprint": suite_fingerprint(cases),
            "cases": len(cases),
            "sessions": sorted({c.session_id for c in cases}),
            "warnings": list(warnings),
        },
        "criteria": asdict(criteria),
        "by_kind": result["by_kind"],
        "frames": result["frames"],
        "baseline": baseline_report,
    }


def write_report(model_dir: Union[str, Path], report: Dict[str, Any]) -> Path:
    """`models/<版>/gate.json` に書く（フレームの名前と数値だけで、画像は含まない）"""
    path = Path(model_dir) / GATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def load_report(model_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    path = Path(model_dir) / GATE_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def report_is_current(report: Optional[Dict[str, Any]], model_dir: Union[str, Path]) -> bool:
    """記録した結果が、今の model.onnx を試したものか"""
    if not report:
        return False
    return report.get("model_sha256") == _sha256(Path(model_dir) / MODEL_FILE)


def summary_lines(report: Dict[str, Any]) -> List[str]:
    lines = [f"門番の試験: {report['model_version']}（{report['suite']['cases']} フレーム）"]
    base = report.get("baseline")
    for name in EXPECTATIONS:
        kind = report["by_kind"][name]
        if not kind["cases"]:
            continue
        text = (
            f"  {EXPECTATION_LABELS[name]}: {kind['passed']}/{kind['cases']} ({kind['rate']:.0%})"
        )
        if base and base["by_kind"][name]["rate"] is not None:
            text += f"  比べる元 {base['version']}: {base['by_kind'][name]['rate']:.0%}"
        lines.append(text)
    if base and base["regressed_frames"]:
        shown = ", ".join(base["regressed_frames"][:10])
        more = len(base["regressed_frames"]) - 10
        lines.append(f"  前は合格し、今回は不合格: {shown}" + (f" ほか {more}" if more > 0 else ""))
    for warning in report["suite"]["warnings"]:
        lines.append(f"  注意: {warning}")
    if report["passed"]:
        lines.append("合格")
    else:
        lines.append("不合格:")
        lines.extend(f"  - {reason}" for reason in report["reasons"])
    return lines


def gate_model(
    model_dir: Union[str, Path],
    data_dirs: Sequence[Union[str, Path]] = DEFAULT_DATA_DIRS,
    baseline: Optional[str] = "auto",
    models_dir: Optional[Union[str, Path]] = None,
    criteria: Optional[GateCriteria] = None,
    store: Optional[AnnotationStore] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """場面を集め、試験をして、`gate.json` に残す。baseline は版のフォルダ、"auto"、None"""
    model_dir = Path(model_dir)
    models_dir = Path(models_dir) if models_dir else model_dir.parent
    cases, warnings = collect_cases(data_dirs, store)
    baseline_dir = None
    if baseline == "auto":
        version = default_baseline(models_dir, model_dir.name)
        baseline_dir = models_dir / version if version else None
    elif baseline:
        baseline_dir = Path(baseline)
    report = run_gate(model_dir, cases, baseline_dir, criteria, warnings=warnings)
    if write:
        write_report(model_dir, report)
    return report


def list_suite(cases: Sequence[GateCase], warnings: Sequence[str]) -> List[str]:
    lines = [f"門番の場面: {len(cases)} フレーム"]
    counts: Dict[Tuple[str, str], int] = {}
    for case in cases:
        counts[(case.session_id, case.expect)] = counts.get((case.session_id, case.expect), 0) + 1
    for (session, expect), n in sorted(counts.items()):
        lines.append(f"  {session}  {EXPECTATION_LABELS[expect]}: {n}")
    lines.extend(f"  注意: {w}" for w in warnings)
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Regression gate for driving models")
    parser.add_argument("--model_dir", help="models/<version> to test")
    parser.add_argument(
        "--baseline",
        default="auto",
        help="models/<version> to compare with, 'auto' (the one on the Pi or approved) or 'none'",
    )
    parser.add_argument("--data_dir", nargs="+", default=list(DEFAULT_DATA_DIRS))
    parser.add_argument("--min_cases", type=int, default=GateCriteria.min_cases)
    parser.add_argument("--max_drop", type=float, default=GateCriteria.max_drop)
    parser.add_argument("--list", action="store_true", help="Only list the gate frames")
    parser.add_argument("--no_write", action="store_true", help="Do not write gate.json")
    args = parser.parse_args(argv)

    try:
        if args.list:
            cases, warnings = collect_cases(args.data_dir)
            print("\n".join(list_suite(cases, warnings)))
            return 0
        if not args.model_dir:
            parser.error("--model_dir is required unless --list is given")
        baseline = None if args.baseline.lower() == "none" else args.baseline
        criteria = GateCriteria(min_cases=args.min_cases, max_drop=args.max_drop)
        report = gate_model(
            args.model_dir, args.data_dir, baseline, criteria=criteria, write=not args.no_write
        )
    except GateError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print("\n".join(summary_lines(report)))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
