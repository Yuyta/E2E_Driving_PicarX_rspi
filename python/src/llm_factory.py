from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from src.config import ModelPreset, settings

OLLAMA_NUM_CTX = 4096
# 読み込みに時間がかかるので、会話の間はメモリに残す
OLLAMA_KEEP_ALIVE = "10m"
# 既定の6より、最初の応答が速かった。すべて使うと他の処理が重くなるので2つ残す
OLLAMA_NUM_THREAD = 10


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
            return LLMFactory.create_local(temperature=temperature)

        else:
            raise ValueError(f"サポートされていないモデルプリセットです: {selected}")

    @staticmethod
    def create_local(
        model: Optional[str] = None, temperature: float = 0.2, **options
    ) -> BaseChatModel:
        """Ollama のモデルを生成する。比較のため、プリセット以外のモデルも指定できる。"""
        from langchain_ollama import ChatOllama

        # プリセット名 "ollama/qwen3.5:4b-text" からモデル名抽出
        raw_name = model or ModelPreset.OLLAMA_LOCAL.value.replace("ollama/", "")
        return ChatOllama(
            model=raw_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=OLLAMA_NUM_CTX,
            keep_alive=OLLAMA_KEEP_ALIVE,
            num_thread=options.pop("num_thread", OLLAMA_NUM_THREAD),
            # 思考モードは応答を遅くするので、会話では使わない
            reasoning=False,
            **options,
        )
