from src.config import ModelPreset
from src.llm_factory import LLMFactory


def test_llm_factory_presets():
    # 1. Gemini Flash (デフォルト)
    flash = LLMFactory.create(preset=ModelPreset.GEMINI_FLASH, api_key="dummy_key")
    assert flash.model == "gemini-3.8-flash"

    # 2. Gemini Flash-Lite
    lite = LLMFactory.create(preset=ModelPreset.GEMINI_FLASH_LITE, api_key="dummy_key")
    assert lite.model == "gemini-3.5-flash-lite"

    # 3. Gemini Pro
    pro = LLMFactory.create(preset=ModelPreset.GEMINI_PRO, api_key="dummy_key")
    assert pro.model == "gemini-3.1-pro-preview"

    # 4. Ollama Local
    ollama = LLMFactory.create(preset=ModelPreset.OLLAMA_LOCAL)
    assert ollama.model == "qwen2.5:7b"
