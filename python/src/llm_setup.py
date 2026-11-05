"""ローカル LLM のテキスト専用モデルを作る。

    uv run python -m src.llm_setup

元のモデルから画像処理部を外すと、実測でメモリのピークが 4.28GB から 3.08GB に減った。
ローカル LLM には画像を渡さないので、応答は変わらない。重みのファイルは共有される。
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

BASE_MODEL = "qwen3.5:4b"
TEXT_MODEL = "qwen3.5:4b-text"
# 語の繰り返しを抑える値。元の 1.5 より 0 のほうが評価で良かった（31/32 と 30/32）
PRESENCE_PENALTY = 0.0


def weights_path(modelfile: str) -> str:
    """Modelfile の最初の FROM が重みのファイル。2番目の FROM は画像処理部"""
    for line in modelfile.splitlines():
        if line.startswith("FROM "):
            return line[len("FROM ") :].strip()
    raise ValueError("Modelfile に FROM がありません")


def build_modelfile(weights: str, presence_penalty: float) -> str:
    return (
        f"FROM {weights}\n"
        "RENDERER qwen3.5\n"
        "PARSER qwen3.5\n"
        "PARAMETER top_k 20\n"
        "PARAMETER top_p 0.95\n"
        f"PARAMETER presence_penalty {presence_penalty}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default=TEXT_MODEL)
    parser.add_argument("--presence-penalty", type=float, default=PRESENCE_PENALTY)
    args = parser.parse_args()

    shown = subprocess.run(
        ["ollama", "show", "--modelfile", BASE_MODEL],
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout
    content = build_modelfile(weights_path(shown), args.presence_penalty)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Modelfile"
        path.write_text(content, encoding="utf-8")
        subprocess.run(["ollama", "create", args.name, "-f", str(path)], check=True)
    print(f"作成しました: {args.name}")


if __name__ == "__main__":
    main()
