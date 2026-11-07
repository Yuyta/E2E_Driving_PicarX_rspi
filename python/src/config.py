import os
from enum import Enum

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


class Settings(BaseModel):
    """アプリケーション設定"""

    # LLM設定
    default_model: ModelPreset = Field(
        # Flash は無料枠が1日20回で、数ターンで尽きる（2026-11-06、利用者の判断）
        default=ModelPreset.GEMINI_FLASH_LITE,
        description="デフォルトのLLMモデル",
    )
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
