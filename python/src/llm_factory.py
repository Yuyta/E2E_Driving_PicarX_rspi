from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import ModelPreset, settings


class LLMFactory:
    """4種のモデル（Gemini 3種 + Ollama 1種）を動的に切り替えるファクトリ"""

    @classmethod
    def create(
        cls,
        preset: Optional[ModelPreset] = None,
        temperature: float = 0.2,
        api_key: Optional[str] = None,
    ) -> BaseChatModel:
        """指定されたモデルプリセットに基づいてChatModelインスタンスを生成"""
        selected = preset or settings.default_model
        if isinstance(selected, str):
            selected = ModelPreset(selected)
        key = api_key or settings.gemini_api_key

        if selected in (
            ModelPreset.GEMINI_FLASH,
            ModelPreset.GEMINI_FLASH_LITE,
            ModelPreset.GEMINI_PRO,
        ):
            from langchain_google_genai import ChatGoogleGenerativeAI

            # モデル名のマッピング
            model_map = {
                model: model.value for model in ModelPreset if model != ModelPreset.OLLAMA_LOCAL
            }
            model_name = model_map[selected]

            # APIキーが空の場合は環境変数 fallback
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=key if key else None,
                # トークン節約のための設定
                max_retries=5,
            )

        elif selected == ModelPreset.OLLAMA_LOCAL:
            from langchain_ollama import ChatOllama

            # プリセット名 "ollama/qwen2.5:7b" からモデル名抽出
            raw_name = selected.value.replace("ollama/", "")
            return ChatOllama(
                model=raw_name,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )

        else:
            raise ValueError(f"サポートされていないモデルプリセットです: {selected}")
