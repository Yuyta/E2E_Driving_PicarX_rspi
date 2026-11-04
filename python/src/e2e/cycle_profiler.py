"""走行ループの1周を工程ごとに計測する（撮影、通信、推論、指令など）。"""

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

TOTAL = "total"


def _percentile(sorted_values: List[float], ratio: float) -> float:
    index = min(len(sorted_values) - 1, int(len(sorted_values) * ratio))
    return sorted_values[index]


class CycleProfiler:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._cycle_start: Optional[float] = None

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = self._clock()
        try:
            yield
        finally:
            self._samples[name].append((self._clock() - start) * 1000.0)

    def mark_cycle(self) -> None:
        """周の先頭で呼ぶ。前回の呼び出しからの時間を1周の時間として記録する。"""
        now = self._clock()
        if self._cycle_start is not None:
            self._samples[TOTAL].append((now - self._cycle_start) * 1000.0)
        self._cycle_start = now

    def summary(self) -> Dict[str, Dict[str, float]]:
        result = {}
        for name, values in self._samples.items():
            if not values:
                continue
            ordered = sorted(values)
            result[name] = {
                "count": len(ordered),
                "p50_ms": round(_percentile(ordered, 0.50), 1),
                "p95_ms": round(_percentile(ordered, 0.95), 1),
                "max_ms": round(ordered[-1], 1),
            }
        return result

    def cycle_hz(self) -> Optional[float]:
        total = self.summary().get(TOTAL)
        if not total or total["p50_ms"] <= 0:
            return None
        return round(1000.0 / total["p50_ms"], 1)

    def format_table(self) -> str:
        lines = [f"{'stage':<12}{'count':>8}{'p50 ms':>10}{'p95 ms':>10}{'max ms':>10}"]
        for name, row in self.summary().items():
            lines.append(
                f"{name:<12}{row['count']:>8}{row['p50_ms']:>10}{row['p95_ms']:>10}{row['max_ms']:>10}"
            )
        hz = self.cycle_hz()
        if hz is not None:
            lines.append(f"cycle: {hz} Hz (p50)")
        return "\n".join(lines)

    def save(self, output_dir: Path) -> Path:
        """数値のみを保存する。画像や場所に関する情報は含めない。"""
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"cycle_{time.strftime('%Y%m%d_%H%M%S')}.json"
        payload = {"cycle_hz_p50": self.cycle_hz(), "stages": self.summary()}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path
