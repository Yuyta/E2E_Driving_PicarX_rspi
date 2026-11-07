import pytest

import src.memory.store as memory_store
from src.memory.store import MEMORY_DB_ENV
from src.metrics.turn_timing import TIMINGS_ENV


@pytest.fixture(autouse=True)
def keep_test_data_out_of_the_data_directory(tmp_path, monkeypatch):
    """テストの会話と応答時間を、利用者の記憶と記録に混ぜない"""
    monkeypatch.setenv(TIMINGS_ENV, str(tmp_path / "timings.jsonl"))
    monkeypatch.setenv(MEMORY_DB_ENV, str(tmp_path / "memory.db"))
    # アプリ全体で共有する記憶は最初に開いた DB を使い続けるので、テストごとに開き直す
    monkeypatch.setattr(memory_store, "_default_store", None)
