"""
PiCar-X Dataset Cleaner (Outlier & Bad-Data Filter)
学習済みモデル（ONNX または PyTorch）の推論誤差（残差）を算出し、
運転ミスや外れ値（壁衝突・急ハンドル・遅延操舵など）のノイズデータを自動で除外する。
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

try:
    from ..training.dataset import DrivingDataset
    from ..training.model import SmallDrivingCNN
except ImportError:
    from src.e2e.training.dataset import DrivingDataset
    from src.e2e.training.model import SmallDrivingCNN

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class DatasetCleaner:
    """
    モデルの推論誤差に基づくデータクリーナー
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        threshold: Optional[float] = 0.4,
        top_k_percent: Optional[float] = None,
    ):
        """
        Args:
            model_path: 評価に使用するモデルパス (.onnx または .pth)
            threshold: 許容するステアリング誤差の上限 (正規化 -1.0〜1.0、例: 0.4 ≒ 約14度)
            top_k_percent: 誤差が大きい上位K%を除外する場合に指定 (例: 5.0 で上位5%除外)
        """
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.top_k_percent = top_k_percent

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((120, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # モデルのロード (ONNX優先、なければPyTorch)
        self.is_onnx = self.model_path.suffix.lower() == ".onnx"
        if self.is_onnx:
            if ort is None:
                raise ImportError("onnxruntime is required to run ONNX models.")
            self.ort_session = ort.InferenceSession(str(self.model_path))
            self.input_name = self.ort_session.get_inputs()[0].name
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.torch_model = SmallDrivingCNN().to(self.device)
            self.torch_model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            self.torch_model.eval()

    def predict(self, image: Image.Image) -> float:
        """単一画像のステアリング角を推論 (-1.0 to 1.0)"""
        tensor = self.transform(image).unsqueeze(0)

        if self.is_onnx:
            input_feed = {self.input_name: tensor.numpy()}
            output = self.ort_session.run(None, input_feed)[0]
            return float(output[0][0])
        else:
            with torch.no_grad():
                tensor = tensor.to(self.device)
                output = self.torch_model(tensor)
                return float(output.cpu().item())

    def analyze_dataset(
        self, data_dir: Union[str, Path, List[Union[str, Path]]]
    ) -> List[Dict[str, Any]]:
        """全サンプルに対して推論を実行し、誤差を付与したレコードリストを作成"""
        dataset = DrivingDataset(data_dir)
        results: List[Dict[str, Any]] = []

        print(
            f"Analyzing {len(dataset)} samples across {len(dataset.session_dirs)} session(s)..."
        )

        for i in range(len(dataset)):
            info = dataset.get_sample_info(i)
            img_path = info["image_path"]

            try:
                img = Image.open(img_path).convert("RGB")
                pred_steering = self.predict(img)
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")
                continue

            target_steering = info["steering"]
            abs_error = abs(pred_steering - target_steering)

            item = {
                "index": i,
                "session_id": info["session_id"],
                "image_path": img_path,
                "target_steering": target_steering,
                "pred_steering": pred_steering,
                "abs_error": abs_error,
                "throttle": info["throttle"],
                "raw_record": info["raw_record"],
            }
            results.append(item)

        return results

    def filter_samples(
        self, analyzed: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """誤差に基づいて良質サンプルと除外対象（Bad Cases）に分類"""
        if not analyzed:
            return [], []

        # 判定用フラグ初期化
        for item in analyzed:
            item["is_bad"] = False
            item["bad_reason"] = None

        # 1. 閾値チェック
        if self.threshold is not None:
            for item in analyzed:
                if item["abs_error"] > self.threshold:
                    item["is_bad"] = True
                    item["bad_reason"] = (
                        f"Error {item['abs_error']:.3f} > threshold {self.threshold:.3f}"
                    )

        # 2. ワーストK%の除外
        if self.top_k_percent is not None and self.top_k_percent > 0:
            errors = [item["abs_error"] for item in analyzed]
            cutoff_val = float(np.percentile(errors, 100.0 - self.top_k_percent))
            for item in analyzed:
                if item["abs_error"] >= cutoff_val and not item["is_bad"]:
                    item["is_bad"] = True
                    item["bad_reason"] = (
                        f"In top {self.top_k_percent}% worst errors (>= {cutoff_val:.3f})"
                    )

        clean = [item for item in analyzed if not item["is_bad"]]
        bad = [item for item in analyzed if item["is_bad"]]

        return clean, bad

    def clean(
        self,
        data_dir: Union[str, Path, List[Union[str, Path]]],
        output_dir: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        データを検査・フィルタリングし、クリーンなデータセットとレポートを出力
        """
        out_p = Path(output_dir)
        out_images_dir = out_p / "images"
        out_images_dir.mkdir(parents=True, exist_ok=True)

        analyzed = self.analyze_dataset(data_dir)
        clean_samples, bad_samples = self.filter_samples(analyzed)

        total_count = len(analyzed)
        clean_count = len(clean_samples)
        bad_count = len(bad_samples)

        mae_before = (
            float(np.mean([s["abs_error"] for s in analyzed])) if total_count > 0 else 0.0
        )
        mae_after = (
            float(np.mean([s["abs_error"] for s in clean_samples])) if clean_count > 0 else 0.0
        )

        print("\nWriting cleaned dataset...")

        # クリーンデータの保存 (画像コピー & labels.jsonl 作成)
        labels_file = out_p / "labels.jsonl"
        with open(labels_file, "w", encoding="utf-8") as f_clean:
            for new_idx, item in enumerate(clean_samples):
                src_img = item["image_path"]
                # 衝突防止のため session_id をプレフィックスに付与
                dst_frame_name = f"{item['session_id']}_{src_img.name}"
                dst_img = out_images_dir / dst_frame_name

                # 画像をコピー
                if src_img.resolve() != dst_img.resolve():
                    shutil.copy2(src_img, dst_img)

                # レコード構築
                new_record = dict(item["raw_record"])
                new_record["frame"] = dst_frame_name
                new_record["cleaned_meta"] = {
                    "original_session": item["session_id"],
                    "pred_steering": round(item["pred_steering"], 4),
                    "abs_error": round(item["abs_error"], 4),
                }
                f_clean.write(json.dumps(new_record, ensure_ascii=False) + "\n")

        # 除外されたバッドケースの記録
        bad_cases_file = out_p / "bad_cases.jsonl"
        with open(bad_cases_file, "w", encoding="utf-8") as f_bad:
            for item in bad_samples:
                bad_record = {
                    "session_id": item["session_id"],
                    "frame": item["image_path"].name,
                    "target_steering": round(item["target_steering"], 4),
                    "pred_steering": round(item["pred_steering"], 4),
                    "abs_error": round(item["abs_error"], 4),
                    "reason": item.get("bad_reason"),
                }
                f_bad.write(json.dumps(bad_record, ensure_ascii=False) + "\n")

        # クリーニングレポート保存
        report = {
            "model_used": str(self.model_path),
            "threshold": self.threshold,
            "top_k_percent": self.top_k_percent,
            "total_samples": total_count,
            "clean_samples": clean_count,
            "removed_samples": bad_count,
            "removal_rate": round(bad_count / total_count, 4) if total_count > 0 else 0.0,
            "mae_before": round(mae_before, 4),
            "mae_before_deg": round(mae_before * 35.0, 2),
            "mae_after": round(mae_after, 4),
            "mae_after_deg": round(mae_after * 35.0, 2),
            "labels_file": str(labels_file),
            "bad_cases_file": str(bad_cases_file),
        }

        with open(out_p / "cleaning_report.json", "w", encoding="utf-8") as f_rep:
            json.dump(report, f_rep, indent=2, ensure_ascii=False)

        # DatasetRegistry 互換の metadata.json を出力
        steerings = [s["target_steering"] for s in clean_samples]
        straight_cnt = sum(1 for s in steerings if abs(s) <= 0.15)
        left_cnt = sum(1 for s in steerings if s < -0.15)
        right_cnt = sum(1 for s in steerings if s > 0.15)

        metadata = {
            "session_id": out_p.name,
            "total_frames": clean_count,
            "duration_sec": 0.0,
            "fps": 0.0,
            "distribution": {
                "straight_ratio": round(straight_cnt / clean_count, 3) if clean_count > 0 else 0.0,
                "left_ratio": round(left_cnt / clean_count, 3) if clean_count > 0 else 0.0,
                "right_ratio": round(right_cnt / clean_count, 3) if clean_count > 0 else 0.0,
            },
            "avg_steering": round(float(np.mean(steerings)), 4) if clean_count > 0 else 0.0,
            "avg_throttle": 0.0,
            "is_cleaned": True,
            "cleaning_report": report,
        }
        with open(out_p / "metadata.json", "w", encoding="utf-8") as f_meta:
            json.dump(metadata, f_meta, indent=2, ensure_ascii=False)

        # コンソール表示
        print("\n" + "=" * 60)
        print("          PiCar-X Dataset Cleaning Summary")
        print("=" * 60)
        print(f"Model Used        : {self.model_path}")
        print(f"Total Samples     : {total_count}")
        print(f"Clean Samples     : {clean_count} (Kept)")
        print(f"Removed Samples   : {bad_count} ({report['removal_rate']:.1%})")
        print(f"MAE Before        : {report['mae_before']} (±{report['mae_before_deg']}°)")
        print(f"MAE After (Clean) : {report['mae_after']} (±{report['mae_after_deg']}°)")
        print(f"Clean Dataset     : {out_p}")
        print(f"Bad Cases Log     : {bad_cases_file}")
        print("\nNext: You can now train a new model using this cleaned dataset:")
        print(
            "  uv run python -m src.e2e.training.train "
            f"--data_dir {out_p} --output_dir models/v004"
        )
        print()

        return report


def clean_dataset(
    data_dir: Union[str, Path, List[Union[str, Path]]],
    model_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/processed/clean_v1",
    threshold: Optional[float] = 0.4,
    top_k_percent: Optional[float] = None,
) -> Dict[str, Any]:
    """スクリプト/外部から直接呼び出すヘルパー関数"""
    cleaner = DatasetCleaner(
        model_path=model_path, threshold=threshold, top_k_percent=top_k_percent
    )
    return cleaner.clean(data_dir=data_dir, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean PiCar-X Dataset by filtering out bad/erroneous driving frames"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        nargs="+",
        default=["data/raw"],
        help="Input dataset path(s) or parent folder (e.g. data/raw)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.onnx or .pth) used for evaluating errors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/clean_v1",
        help="Destination directory for cleaned dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Maximum steering error tolerance (-1.0 to 1.0, default: 0.4)",
    )
    parser.add_argument(
        "--top_k_percent",
        type=float,
        default=None,
        help="Optional: Drop top K percent of worst error samples (e.g. 5.0 for top 5%)",
    )

    args = parser.parse_args()
    data_input = args.data_dir[0] if len(args.data_dir) == 1 else args.data_dir
    clean_dataset(
        data_dir=data_input,
        model_path=args.model,
        output_dir=args.output_dir,
        threshold=args.threshold,
        top_k_percent=args.top_k_percent,
    )
