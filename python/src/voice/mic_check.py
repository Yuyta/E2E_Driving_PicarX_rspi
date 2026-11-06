"""PC のマイクで1回録音し、Whisper のモデルごとに文字起こしの結果と時間を比べる。

    uv run python -m src.voice.mic_check --models small large-v3-turbo

録音はメモリ上だけで扱い、ファイルに保存しない。
"""

import argparse
import time

from src.voice.listener import SAMPLE_RATE, Recorder, Transcriber


def level_text(audio) -> str:
    import numpy as np

    frame = SAMPLE_RATE // 10
    usable = audio[: len(audio) // frame * frame].reshape(-1, frame)
    db = 20 * np.log10(np.maximum(np.sqrt(np.mean(usable**2, axis=1)), 1e-9))
    return (
        f"音量 dBFS: 中央 {np.median(db):.1f} / 最大 {db.max():.1f}、"
        f"ピーク {np.abs(audio).max():.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--models", nargs="+", default=["small"])
    args = parser.parse_args()

    transcribers = []
    for name in args.models:
        print(f"{name} を読み込んでいます...", flush=True)
        started = time.monotonic()
        transcriber = Transcriber(name)
        transcriber.load()
        print(f"  {time.monotonic() - started:.1f}秒")
        transcribers.append(transcriber)

    recorder = Recorder()
    print("3秒後に録音を始めます。普段の距離から話しかけてください。")
    for count in (3, 2, 1):
        print(count, flush=True)
        time.sleep(1)
    print(f"録音中（{args.seconds:.0f}秒）...", flush=True)
    recorder.start()
    time.sleep(args.seconds)
    audio = recorder.stop()
    print(level_text(audio))

    for transcriber in transcribers:
        started = time.monotonic()
        result = transcriber.transcribe(audio)
        elapsed = time.monotonic() - started
        print(f"[{transcriber.model_name}] {elapsed:.1f}秒: {result.text or result.problem}")


if __name__ == "__main__":
    main()
