"""
Dataset Registry for PiCar-X E2E Driving Data
走行ログ（labels.jsonl）の集計・統計情報の抽出・メタデータ管理を行う。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class DatasetRegistry:
    """データセットのメタデータ抽出と管理"""

    def __init__(self, raw_data_dir: str = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)

    def analyze_session(self, session_path: Path) -> Optional[Dict[str, Any]]:
        """1つの走行セッションフォルダを解析し統計メタデータを生成"""
        labels_file = session_path / "labels.jsonl"
        meta_file = session_path / "metadata.json"

        # クリーニング済みデータセットなどの既存カスタムメタデータを優先
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
                if existing_meta.get("is_cleaned"):
                    return existing_meta
            except Exception:
                pass

        if not labels_file.exists():
            return None

        records = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        if not records:
            return None

        total_frames = len(records)
        timestamps = [r["timestamp"] for r in records if "timestamp" in r]
        duration_sec = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0

        steerings = [r["steering"] for r in records]
        throttles = [r.get("throttle", 0.0) for r in records]

        # 操舵の分布集計 (直進: -0.15〜0.15, 左: <-0.15, 右: >0.15)
        straight_count = sum(1 for s in steerings if abs(s) <= 0.15)
        left_count = sum(1 for s in steerings if s < -0.15)
        right_count = sum(1 for s in steerings if s > 0.15)

        avg_steering = sum(steerings) / total_frames
        avg_throttle = sum(throttles) / total_frames

        stats = {
            "session_id": session_path.name,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 2),
            "fps": round(total_frames / duration_sec, 1) if duration_sec > 0 else 0.0,
            "distribution": {
                "straight_ratio": round(straight_count / total_frames, 3),
                "left_ratio": round(left_count / total_frames, 3),
                "right_ratio": round(right_count / total_frames, 3),
            },
            "avg_steering": round(avg_steering, 4),
            "avg_throttle": round(avg_throttle, 4),
        }

        # 既存メタデータがある場合は拡張フィールドを維持
        meta_file = session_path / "metadata.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    old_meta = json.load(f)
                for k, v in old_meta.items():
                    if k not in stats:
                        stats[k] = v
            except Exception:
                pass

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats

    def list_sessions(self) -> List[Dict[str, Any]]:
        """raw_data_dir 配下の全セッションを解析・一覧化"""
        if not self.raw_data_dir.exists():
            return []

        results = []
        for p in sorted(self.raw_data_dir.iterdir()):
            if p.is_dir():
                stats = self.analyze_session(p)
                if stats:
                    results.append(stats)
        return results
