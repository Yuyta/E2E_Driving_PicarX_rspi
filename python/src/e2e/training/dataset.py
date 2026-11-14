"""
PyTorch Dataset and DataLoader for PiCar-X Driving Data
単一セッション、親フォルダ (例: data/raw/) 配下の全セッション、または複数セッションの統合に対応。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from ..evaluation.metrics import STRAIGHT_LIMIT
    from ..intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, one_hot
except ImportError:
    from src.e2e.evaluation.metrics import STRAIGHT_LIMIT
    from src.e2e.intent import INTENT_ANY, INTENT_LEFT, INTENT_RIGHT, one_hot


def resolve_session_dirs(
    data_dir: Union[str, Path, Sequence[Union[str, Path]]],
) -> List[Path]:
    """
    指定されたパスから labels.jsonl を含む有効なセッションディレクトリ一覧を解決する。

    Args:
        data_dir: 単一セッションフォルダ、親フォルダ (例: data/raw)、またはパスのリスト/タプル

    Returns:
        labels.jsonl を含む Path のリスト
    """
    targets: List[Path] = []

    if isinstance(data_dir, (list, tuple)):
        for item in data_dir:
            targets.extend(resolve_session_dirs(item))
        return sorted(list(dict.fromkeys(targets)))

    p = Path(data_dir)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    # 直下に labels.jsonl がある場合は単一セッション
    if (p / "labels.jsonl").exists():
        return [p]

    # 直下にない場合、サブディレクトリから labels.jsonl を探索
    if p.is_dir():
        found = [d for d in sorted(p.iterdir()) if d.is_dir() and (d / "labels.jsonl").exists()]
        if found:
            return found
        # 再帰的に探索 (data/processed/clean_v1 など深い階層にも対応)
        recursive_found = sorted(list(dict.fromkeys(lp.parent for lp in p.rglob("labels.jsonl"))))
        if recursive_found:
            return recursive_found

    raise FileNotFoundError(f"No valid session directory with labels.jsonl found in: {p}")


def is_stopped(record: Dict[str, Any]) -> bool:
    """記録から、前進の指令がなく止まっていたと分かるフレームか。分からなければ False

    新しい記録はそのフレームで有効な指令（`command.speed`）、収集の記録は `throttle`、
    介入の記録は `speed` を持つ。どれもない古い記録は、止まっていたとは判断しない。
    """
    command = record.get("command")
    if isinstance(command, dict) and "speed" in command:
        return float(command["speed"]) <= 0.0
    if "throttle" in record:
        return float(record["throttle"]) <= 0.0
    if "speed" in record:
        return float(record["speed"]) <= 0.0
    return False


# 前方が塞がっているかのラベルの境界（cm）。間の距離は超音波のばらつきで揺れるので、どちらとも
# 決めずに学習から外す。auto_collector は 50cm より近いと曲がり始める（ROADMAP の M3「前方が
# 塞がっているかの補助出力」）
BLOCKED_NEAR_CM = 40.0
BLOCKED_CLEAR_CM = 60.0


def blocked_label(
    record: Dict[str, Any],
    near_cm: float = BLOCKED_NEAR_CM,
    clear_cm: float = BLOCKED_CLEAR_CM,
) -> Optional[bool]:
    """超音波の距離から、前方が塞がっているか。決められなければ None

    近ければ True、遠いか反射が戻らない（`distance_far`）なら False。距離のない古い記録、
    距離が境界の間、反射がなく遠いとも分からない時は None。細い家具の脚や音を吸う物は
    超音波で見えないので、False のラベルにも誤りが混じる。
    """
    if near_cm > clear_cm:
        raise ValueError(f"near_cm ({near_cm}) must not exceed clear_cm ({clear_cm})")
    distance = record.get("distance_cm")
    if distance is None:
        return False if record.get("distance_far") else None
    distance = float(distance)
    if distance < near_cm:
        return True
    if distance >= clear_cm:
        return False
    return None


# 自動収集（auto_collector）で、障害物を避ける曲がりと開けた場所での曲がりの状態
TURN_STATES = ("avoid", "open_turn")


def turn_intents(records: Sequence[Dict[str, Any]]) -> List[str]:
    """曲がりの区間に、実際に曲がった向きを意図として付け直す（1セッションの記録を時刻順に渡す）

    自動収集は前が近いと「右を先に試し、曲がれなければ下がって左へ」曲がり、前が空くまで
    曲がり続ける。どちらへ、いつまで曲がるかは直前の経過で決まり、1枚の画像からは分からない
    （2026-11-14 の交差検証で、常に0°と答えるより悪かった原因）。そこで、どちらへ曲がるかは
    上の層が意図で与え、いつ曲がり始めるかは前方が塞がっているかの出力で決めることにし、
    曲がりの区間（`policy_state` が `TURN_STATES`）と、その後に舵が中央へ戻るまでの `cruise` の
    フレームに、曲がった向きを付ける。記録にもともと左右の意図があるフレームは変えない。
    """
    intents = [record.get("intent") or INTENT_ANY for record in records]
    i = 0
    while i < len(records):
        if records[i].get("policy_state") not in TURN_STATES:
            i += 1
            continue
        end = i
        while end < len(records) and records[end].get("policy_state") in TURN_STATES:
            end += 1
        # 曲がり終えた後も、舵は一定の速さで中央へ戻るので、しばらく曲がったまま走る
        while (
            end < len(records)
            and records[end].get("policy_state") == "cruise"
            and abs(float(records[end].get("steering", 0.0))) > STRAIGHT_LIMIT
        ):
            end += 1
        total = sum(
            float(r.get("steering", 0.0))
            for r in records[i:end]
            if abs(float(r.get("steering", 0.0))) > STRAIGHT_LIMIT
        )
        if total:
            side = INTENT_LEFT if total < 0 else INTENT_RIGHT
            for k in range(i, end):
                if intents[k] == INTENT_ANY:
                    intents[k] = side
        i = end
    return intents


class DrivingDataset(Dataset):
    """
    収集された画像とステアリング角のデータセット (複数セッション統合対応)
    """

    def __init__(
        self,
        data_dir: Union[str, Path, Sequence[Union[str, Path]]],
        transform=None,
        include_excluded: bool = False,
        include_motion_mismatch: bool = False,
        include_stopped: bool = False,
        keep_for_blocked: bool = False,
        blocked_near_cm: float = BLOCKED_NEAR_CM,
        blocked_clear_cm: float = BLOCKED_CLEAR_CM,
        turn_intent: bool = False,
    ):
        """
        Args:
            data_dir: 単一セッションフォルダ、親フォルダ (data/raw)、またはそのリスト
            transform: torchvision 変換パイプライン
            include_excluded: False の場合、exclude: true のデータ（未承認の介入データ等）を除外
            include_motion_mismatch: False の場合、指令と実際の動きが食い違ったフレーム
                （`motion_check.flags` が空でないもの。空転、滑りなど）を除外
            include_stopped: False の場合、前進の指令がないフレーム（`throttle` が 0）を除外。
                止まっている間の舵角は、その景色でどう曲がるかの手本にならない
            keep_for_blocked: True の場合、上の3つで舵角の学習から外すフレームも、前方が
                塞がっているかのラベルが決まれば残す（`steer_valid` が False）。止まった時、
                下がった時、曲がれなかった時のフレームは、前が近い画像の多くを占める
            blocked_near_cm, blocked_clear_cm: 前方が塞がっているかのラベルの境界（`blocked_label`）
            turn_intent: True の場合、曲がりの区間に曲がった向きを意図として付け直す
                （`turn_intents`）。記録のままの意図は `intent_recorded` に残る
        """
        self.session_dirs = resolve_session_dirs(data_dir)
        self.samples: List[Dict[str, Any]] = []
        self.skipped_motion_mismatch = 0
        self.skipped_stopped = 0
        # 舵角の学習からは外し、前方が塞がっているかの学習だけに残したフレーム
        self.kept_for_blocked_only = 0
        # 曲がりの区間なので、意図を おまかせ から左右へ付け直したフレーム
        self.relabeled_turn_intent = 0

        for s_dir in self.session_dirs:
            labels_file = s_dir / "labels.jsonl"
            images_dir = s_dir / "images"

            with open(labels_file, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            recorded = [record.get("intent") or INTENT_ANY for record in records]
            intents = turn_intents(records) if turn_intent else recorded

            for record, intent, intent_recorded in zip(records, intents, recorded):
                blocked = blocked_label(record, blocked_near_cm, blocked_clear_cm)
                steer_valid = True

                # 学習除外フラグの確認 (未承認の介入データや外れ値データをスキップ)
                if not include_excluded and (
                    record.get("exclude", False) or record.get("exclude_from_training", False)
                ):
                    steer_valid = False

                # 指令と実際の動きが食い違ったフレームは、誤った操作の例になるので学習に使わない
                elif not include_motion_mismatch and (record.get("motion_check") or {}).get(
                    "flags"
                ):
                    self.skipped_motion_mismatch += 1
                    steer_valid = False

                elif not include_stopped and is_stopped(record):
                    self.skipped_stopped += 1
                    steer_valid = False

                if not steer_valid:
                    if not (keep_for_blocked and blocked is not None):
                        continue
                    self.kept_for_blocked_only += 1
                elif intent != intent_recorded:
                    self.relabeled_turn_intent += 1

                frame_name = record.get("frame", "")

                # クリーナーの出力は画像を複製せず、元のセッションを `source_dir` で参照する
                source_dir = record.get("source_dir")
                if source_dir:
                    origin_dir = (s_dir / source_dir).resolve()
                    img_path = origin_dir / "images" / frame_name
                    if not img_path.exists() and (origin_dir / frame_name).exists():
                        img_path = origin_dir / frame_name
                else:
                    origin_dir = s_dir
                    img_path = images_dir / frame_name
                    if not img_path.exists() and (s_dir / frame_name).exists():
                        img_path = s_dir / frame_name

                self.samples.append(
                    {
                        "image_path": img_path,
                        "steering": float(record.get("steering", 0.0)),
                        "throttle": float(record.get("throttle", 0.0)),
                        # 意図のない古い記録は any。turn_intent なら曲がりの区間は付け直した向き
                        "intent": intent,
                        "intent_recorded": intent_recorded,
                        # False なら舵角の学習と評価には使わない（keep_for_blocked で残した）
                        "steer_valid": steer_valid,
                        # 前方が塞がっているか。None は不明
                        "blocked": blocked,
                        # 分割と集計は元のセッション単位で行う。
                        # 画像を複製していた旧形式の出力は cleaned_meta から元を読む
                        "session_id": (
                            origin_dir.name
                            if source_dir
                            else record.get("cleaned_meta", {}).get("original_session")
                            or s_dir.name
                        ),
                        "session_dir": origin_dir,
                        "raw_record": record,
                    }
                )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((120, 160)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image_path = sample["image_path"]

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        steering = torch.tensor(sample["steering"], dtype=torch.float32)
        throttle = torch.tensor(sample["throttle"], dtype=torch.float32)
        intent = torch.tensor(one_hot(sample["intent"]), dtype=torch.float32)

        return image, steering, throttle, intent

    @property
    def source_session_dirs(self) -> List[Path]:
        """サンプルの元になったセッション（クリーナーの出力なら参照先の元データ）"""
        return sorted({sample["session_dir"] for sample in self.samples})

    def steer_indices(self, indices: Optional[Sequence[int]] = None) -> List[int]:
        """舵角の学習と評価に使うサンプル"""
        pool = range(len(self.samples)) if indices is None else indices
        return [i for i in pool if self.samples[i]["steer_valid"]]

    def blocked_indices(self, indices: Optional[Sequence[int]] = None) -> List[int]:
        """前方が塞がっているかのラベルが決まっているサンプル"""
        pool = range(len(self.samples)) if indices is None else indices
        return [i for i in pool if self.samples[i]["blocked"] is not None]

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """サンプルの画像パスやメタデータを取得"""
        return self.samples[idx]
