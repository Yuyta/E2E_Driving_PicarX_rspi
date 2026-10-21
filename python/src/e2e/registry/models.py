"""
Model Registry for PiCar-X E2E Steering Models
各モデルバージョン（v001, v002...）の成果物、学習履歴、評価指標を一元管理する。
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelRegistry:
    """モデルバージョンのメタデータ管理と一覧化"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)

    def get_model_info(self, version_dir: Path) -> Optional[Dict[str, Any]]:
        """1つのモデルバージョンのメタデータを取得"""
        meta_file = version_dir / "metadata.json"
        if not meta_file.exists():
            return None

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None

        # 成果物の存在確認
        meta["has_onnx"] = (version_dir / "model.onnx").exists()
        meta["has_weights"] = (version_dir / "best_model.pth").exists()

        eval_file = version_dir / "eval_metrics.json"
        if eval_file.exists():
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    meta["evaluation"] = json.load(f)
            except Exception:
                pass

        return meta

    def list_models(self) -> List[Dict[str, Any]]:
        """models_dir 配下の全モデルをバージョン順に取得"""
        if not self.models_dir.exists():
            return []

        models = []
        for p in sorted(self.models_dir.iterdir()):
            if p.is_dir():
                info = self.get_model_info(p)
                if info:
                    models.append(info)
        return models
