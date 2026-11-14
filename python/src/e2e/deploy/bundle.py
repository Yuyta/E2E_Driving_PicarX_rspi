"""Pi へ送る走行モデルのフォルダ（バンドル）を作る（ROADMAP の M3「Pi へのモデルの配置」）

`models/<版>/` の ONNX（`model.onnx` と重みの `model.onnx.data`）をコピーし、
`manifest.json` を書く。
マニフェストには、各ファイルの SHA256、入力の大きさ、正規化の値、意図の順番、出力の名前、
前方が塞がっているかの出力を学習したか、ソフトウェアの版を入れる。Pi（`rust/src/autopilot/model.rs`）は
ハッシュと入力の大きさを確かめてから読み込み、合わなければ読み込まない。

使い方:
    uv run python -m src.e2e.deploy.bundle --model_dir models/v006 --output dist/models
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    from ..intent import INTENTS
except ImportError:
    from src.e2e.intent import INTENTS

MANIFEST_FORMAT = 1
MODEL_FILE = "model.onnx"
# 学習の舵角は ±35° を -1〜1 にしたもの（収集の steering = 舵角 / 35）
STEERING_SCALE_DEG = 35.0
# Pi の tract（0.21）が読めない ONNX の演算（AdaptiveAvgPool を割り切れない大きさで書き出すと出る）
TRACT_UNSUPPORTED_OPS = {"GatherND"}


class BundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def onnx_files(model_dir: Path) -> list:
    """ONNX と、外部に分けた重みのファイル"""
    files = [model_dir / MODEL_FILE]
    if not files[0].is_file():
        raise BundleError(f"{files[0]} がありません")
    import onnx

    model = onnx.load(str(files[0]), load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                path = model_dir / entry.value
                if path not in files:
                    files.append(path)
    ops = {node.op_type for node in model.graph.node}
    unsupported = ops & TRACT_UNSUPPORTED_OPS
    if unsupported:
        raise BundleError(f"Pi の tract が読めない演算があります: {sorted(unsupported)}")
    return files


def build_manifest(model_dir: Path, files) -> Dict[str, Any]:
    import onnxruntime as ort

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        raise BundleError(f"{metadata_path} がありません（学習の記録のないモデルは送らない）")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pre = metadata.get("preprocessing") or {}
    height, width = pre.get("resize_hw") or (None, None)
    if not (height and width and pre.get("normalize_mean") and pre.get("normalize_std")):
        raise BundleError("metadata.json に前処理の値（resize_hw、normalize_mean/std）がありません")

    session = ort.InferenceSession(str(model_dir / MODEL_FILE))
    inputs = [i.name for i in session.get_inputs()]
    outputs = [o.name for o in session.get_outputs()]
    if inputs != ["input", "intent"]:
        raise BundleError(f"入力が input と intent ではありません: {inputs}")
    shape = session.get_inputs()[0].shape
    if list(shape[1:]) != [3, height, width]:
        raise BundleError(f"ONNX の入力 {shape} が前処理の大きさ {height}x{width} と違います")

    environment = metadata.get("environment") or {}
    return {
        "format": MANIFEST_FORMAT,
        "model_version": metadata.get("model_version") or model_dir.name,
        "data_fingerprint": (metadata.get("data") or {}).get("fingerprint"),
        "model_commit": (metadata.get("code") or {}).get("commit"),
        "files": {path.name: sha256_file(path) for path in files},
        "input": {
            "width": int(width),
            "height": int(height),
            "scale": 255.0,
            "mean": [float(v) for v in pre["normalize_mean"]],
            "std": [float(v) for v in pre["normalize_std"]],
            "color": pre.get("color", "RGB"),
            "layout": "NCHW",
        },
        "intents": list(INTENTS),
        "outputs": outputs,
        "blocked_trained": bool((metadata.get("blocked_output") or {}).get("trained")),
        "steering_scale_deg": STEERING_SCALE_DEG,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "software": {
            k: str(v)
            for k, v in environment.items()
            if k in ("torch", "onnx", "onnxruntime", "python")
        },
    }


def build_bundle(model_dir, output_root) -> Path:
    """`output_root/<版>/` にバンドルを作り、そのフォルダを返す"""
    model_dir = Path(model_dir)
    files = onnx_files(model_dir)
    manifest = build_manifest(model_dir, files)
    bundle = Path(output_root) / manifest["model_version"]
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    for path in files:
        shutil.copy2(path, bundle / path.name)
    # コピーした後のハッシュで確かめる
    for name, digest in manifest["files"].items():
        if sha256_file(bundle / name) != digest:
            raise BundleError(f"{name} のコピーが元と違います")
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return bundle


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the model bundle to send to the Pi")
    parser.add_argument("--model_dir", required=True, help="models/<version>")
    parser.add_argument("--output", default="dist/models", help="Folder for the bundles")
    args = parser.parse_args(argv)
    try:
        bundle = build_bundle(args.model_dir, args.output)
    except BundleError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    print(
        f"Bundle: {bundle} (version {manifest['model_version']}, files {list(manifest['files'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
