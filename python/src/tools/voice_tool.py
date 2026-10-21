from langchain_core.tools import tool

from src.picarx_client.client import PicarxClient

_client = PicarxClient()


@tool
async def speak_text(text: str) -> str:
    """
    Raspberry Pi 側でテキストを音声合成(TTS)してスピーカーから再生します。
    重い音声ファイルは送信せず、テキストのみを実機に伝達します。
    """
    try:
        ok = await _client.speak(text=text)
        return f"発話完了: 「{text}」" if ok else "発話失敗"
    except Exception as e:
        return f"発話エラー: {str(e)}"
