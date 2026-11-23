import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class ModelPreset(str, Enum):
    """利用可能なLLMモデルプリセット"""

    GEMINI_FLASH = "gemini-3.8-flash"  # 無料枠は1日20回しかない
    GEMINI_FLASH_LITE = "gemini-3.5-flash-lite"  # デフォルト: 低遅延・低コスト
    GEMINI_PRO = "gemini-3.1-pro-preview"  # 高度な推論・複雑な計画
    OLLAMA_LOCAL = (
        "ollama/qwen3.5:4b-text"  # ローカルLLM (完全オフライン)。PC の CPU で実測して選定
    )
    # CPU だけで動く NeuraCPU。手元の資料と Web の検索を、発言ごとに AI が自分で選ぶ
    NEURACPU = "neuracpu/qwen3.5:4b-text"


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NEURACPU_DEFAULT_ROOT = Path.home() / ".gemini" / "antigravity" / "scratch" / "neuracpu_engine"
# 会話で調べる手元の資料。操作マニュアルと、利用者が資料を置くフォルダ（Git には入らない）
NEURACPU_DEFAULT_SOURCES = (
    PROJECT_ROOT / "python" / "src" / "gui" / "web" / "static" / "manual.html",
    PROJECT_ROOT / "python" / "data" / "knowledge",
)


def _paths_from_env(name: str, default) -> tuple:
    value = os.getenv(name, "")
    return tuple(Path(p) for p in value.split(";") if p.strip()) if value else tuple(default)


class Settings(BaseModel):
    """アプリケーション設定"""

    # LLM設定
    default_model: ModelPreset = Field(
        # Flash は無料枠が1日20回で、数ターンで尽きる（2026-11-06、利用者の判断）
        default=ModelPreset.GEMINI_FLASH_LITE,
        description="デフォルトのLLMモデル",
    )
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # localhost だと Windows では IPv6 を先に試し、接続のたびに約2秒待たされた（2026-11-23 実測）
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    )

    # NeuraCPU（資料と Web の検索）。検索の処理は NeuraCPU の仮想環境で別のプロセスとして動かす
    neuracpu_root: Path = Field(
        default_factory=lambda: Path(os.getenv("NEURACPU_ROOT", str(NEURACPU_DEFAULT_ROOT)))
    )
    # 判断と回答に使う Ollama のモデル。qwen3.5:4b-text は判断が最も正確だった。
    # NeuraCPU の Qwen2.5-3B（ollama create neuracpu-qwen2.5-3b）は回答が約3秒速い
    neuracpu_model: str = Field(
        default_factory=lambda: os.getenv(
            "NEURACPU_MODEL", ModelPreset.NEURACPU.value.split("/", 1)[1]
        )
    )
    neuracpu_port: int = Field(default_factory=lambda: int(os.getenv("NEURACPU_PORT", "18765")))
    neuracpu_sources: tuple = Field(
        default_factory=lambda: _paths_from_env("NEURACPU_SOURCES", NEURACPU_DEFAULT_SOURCES)
    )

    # Raspberry Pi (Rust APIサーバー) 接続設定
    robot_base_url: str = Field(
        default_factory=lambda: os.getenv("ROBOT_BASE_URL", "http://raspberrypi.local:8080")
    )
    robot_api_token: str = Field(default_factory=lambda: os.getenv("ROBOT_API_TOKEN", ""))
    request_timeout_sec: float = Field(default=5.0)

    # トークン節約設定
    enable_sensor_diff_filter: bool = Field(
        default=True, description="センサー値の変化が小さい場合にプロンプトから除外してトークン節約"
    )
    sensor_diff_threshold_cm: float = Field(default=2.0, description="距離センサーの変化閾値(cm)")
    max_history_turns: int = Field(default=10, description="要約を実行する最大会話ターン数")


settings = Settings()
