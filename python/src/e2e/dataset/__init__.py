"""
Dataset utilities for PiCar-X E2E Self Driving
"""

from typing import Any


def __getattr__(name: str) -> Any:
    if name in ("DatasetCleaner", "clean_dataset"):
        from .cleaner import DatasetCleaner, clean_dataset

        return DatasetCleaner if name == "DatasetCleaner" else clean_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DatasetCleaner", "clean_dataset"]
