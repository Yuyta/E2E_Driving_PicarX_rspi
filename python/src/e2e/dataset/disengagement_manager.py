"""
PiCar-X Disengagement Manager & Logger Utilities
家庭内自律走行における介入・緊急停止（壁・家具接近、スタック、人間介入）データの管理ツール。
意味のない誤検知データの削除や、学習からの自動除外・承認を制御する。
"""

import argparse
import json
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


class DisengagementBuffer:
    """
    直近N秒間の走行フレーム（画像、推論舵角、超音波距離等）をメモリ上に保持するリングバッファ
    """

    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.buffer: deque = deque(maxlen=max_frames)

    def append(
        self,
        frame_bgr,
        steering_norm: float,
        steer_angle: int,
        ultrasonic_dist: float,
        speed: int,
    ):
        """1フレーム分のデータをバッファに追加"""
        record = {
            "timestamp": time.time(),
            "frame": frame_bgr.copy() if frame_bgr is not None else None,
            "steering": float(steering_norm),
            "steer_angle": int(steer_angle),
            "ultrasonic": float(ultrasonic_dist),
            "speed": int(speed),
        }
        self.buffer.append(record)

    def save_event(
        self,
        trigger: str,
        output_base_dir: Path,
        model_name: str = "unknown",
        note: str = "",
    ) -> Optional[Path]:
        """
        バッファにたまっている直前データをファイルに保存する。
        デフォルトで exclude_from_training: true （学習除外）としてタグ付け。

        Args:
            trigger: 停止理由 ('obstacle_stop', 'manual_intervention' 等)
            output_base_dir: 出力先親ディレクトリ (例: data/disengagements)
            model_name: 使用していたモデル名
            note: メモ

        Returns:
            保存されたイベントディレクトリのパス (データが空の場合は None)
        """
        if not self.buffer:
            return None

        output_base_dir = Path(output_base_dir)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        event_dir = output_base_dir / f"{timestamp_str}_{trigger}"
        images_dir = event_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        labels = []
        frames_to_save = list(self.buffer)

        for i, item in enumerate(frames_to_save):
            frame_name = f"frame_{i:06d}.jpg"
            img_path = images_dir / frame_name

            if item["frame"] is not None:
                cv2.imwrite(str(img_path), item["frame"])

            record = {
                "timestamp": item["timestamp"],
                "frame": frame_name,
                "steering": item["steering"],
                "steer_angle": item["steer_angle"],
                "ultrasonic": item["ultrasonic"],
                "speed": item["speed"],
                "tag": "disengagement",
                "trigger": trigger,
                "exclude": True,  # デフォルトで学習除外
            }
            labels.append(record)

        # labels.jsonl を書き出し
        with open(event_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for rec in labels:
                f.write(json.dumps(rec) + "\n")

        # イベントメタデータ event.json を書き出し
        event_meta = {
            "event_id": event_dir.name,
            "timestamp": timestamp_str,
            "trigger": trigger,
            "frames_count": len(labels),
            "model_used": model_name,
            "status": "pending_review",  # pending_review, approved, rejected
            "exclude_from_training": True,
            "note": note,
        }
        with open(event_dir / "event.json", "w", encoding="utf-8") as f:
            json.dump(event_meta, f, indent=2, ensure_ascii=False)

        return event_dir


class DisengagementManager:
    """
    保存された介入イベントの管理・レビュー・除外・削除を行うマネージャー
    """

    def __init__(self, base_dir: Path = Path("data/disengagements")):
        self.base_dir = Path(base_dir)

    def list_events(self) -> List[Dict[str, Any]]:
        """全イベントの概要を取得"""
        if not self.base_dir.exists():
            return []

        events = []
        for d in sorted(self.base_dir.iterdir()):
            if not d.is_dir():
                continue
            meta_file = d / "event.json"
            if meta_file.exists():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        meta["dir_path"] = str(d)
                        events.append(meta)
                except Exception:
                    pass
        return events

    def reject_event(self, event_id: str, delete: bool = False) -> bool:
        """
        意味のない誤検知イベント（人が前を横切っただけ、単に走行終了した等）を無効化または削除する。

        Args:
            event_id: イベントID（フォルダ名）
            delete: True の場合フォルダごと物理削除、False の場合 rejected 状態に更新

        Returns:
            成功したか否か
        """
        target_dir = self.base_dir / event_id
        if not target_dir.exists():
            return False

        if delete:
            shutil.rmtree(target_dir)
            return True

        meta_file = target_dir / "event.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "rejected"
            meta["exclude_from_training"] = True
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        # labels.jsonl の exclude を確実に True に更新
        labels_file = target_dir / "labels.jsonl"
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(labels_file, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip():
                        r = json.loads(line)
                        r["exclude"] = True
                        r["status"] = "rejected"
                        f.write(json.dumps(r) + "\n")
        return True

    def approve_event(self, event_id: str) -> bool:
        """
        真の失敗ケース（家具の脚を回避できなかった、行き止まりでスタックした等）を承認する。
        これにより、後の苦手克服学習用データセットとして利用可能になる。
        """
        target_dir = self.base_dir / event_id
        if not target_dir.exists():
            return False

        meta_file = target_dir / "event.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["status"] = "approved"
            meta["exclude_from_training"] = False  # 学習対象として承認
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

        # labels.jsonl の exclude を解除
        labels_file = target_dir / "labels.jsonl"
        if labels_file.exists():
            with open(labels_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            with open(labels_file, "w", encoding="utf-8") as f:
                for line in lines:
                    if line.strip():
                        r = json.loads(line)
                        r["exclude"] = False
                        r["status"] = "approved"
                        f.write(json.dumps(r) + "\n")
        return True


def print_events_table(events: List[Dict[str, Any]]):
    """イベント一覧を綺麗に表形式でターミナル表示"""
    print("\n" + "=" * 80)
    print("       PiCar-X Disengagement Events (Indoor Home Navigation)")
    print("=" * 80)
    print(
        f"{'Event ID':<32} | {'Trigger':<20} | {'Frames':<6} | {'Status':<14} | {'Training'}"
    )
    print("-" * 80)
    if not events:
        print("  No disengagement events recorded.")
    else:
        for ev in events:
            ev_id = ev.get("event_id", "")
            trigger = ev.get("trigger", "")
            frames = ev.get("frames_count", 0)
            status = ev.get("status", "unknown")
            training = (
                "EXCLUDED" if ev.get("exclude_from_training", True) else "ELIGIBLE"
            )
            print(
                f"{ev_id:<32} | {trigger:<20} | {frames:<6} | {status:<14} | {training}"
            )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="PiCar-X Disengagement Event Manager"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="data/disengagements",
        help="Disengagements root directory",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all recorded events"
    )
    parser.add_argument(
        "--reject",
        type=str,
        metavar="EVENT_ID",
        help="Reject an event (meaningless or false alarm)",
    )
    parser.add_argument(
        "--delete",
        type=str,
        metavar="EVENT_ID",
        help="Permanently delete an event directory",
    )
    parser.add_argument(
        "--approve",
        type=str,
        metavar="EVENT_ID",
        help="Approve a true failure event for retraining",
    )

    args = parser.parse_args()
    manager = DisengagementManager(base_dir=Path(args.base_dir))

    if args.reject:
        ok = manager.reject_event(args.reject, delete=False)
        if ok:
            print(f"[SUCCESS] Event '{args.reject}' marked as REJECTED (excluded).")
        else:
            print(f"[ERROR] Event '{args.reject}' not found.")
    elif args.delete:
        ok = manager.reject_event(args.delete, delete=True)
        if ok:
            print(f"[SUCCESS] Event '{args.delete}' DELETED permanently.")
        else:
            print(f"[ERROR] Event '{args.delete}' not found.")
    elif args.approve:
        ok = manager.approve_event(args.approve)
        if ok:
            print(
                f"[SUCCESS] Event '{args.approve}' APPROVED for retraining."
            )
        else:
            print(f"[ERROR] Event '{args.approve}' not found.")
    else:
        events = manager.list_events()
        print_events_table(events)


if __name__ == "__main__":
    main()
