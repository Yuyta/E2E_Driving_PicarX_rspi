from langchain_core.tools import tool

from src.vision.local_vlm import LocalVision

_vision = LocalVision()


@tool
async def look(question: str) -> str:
    """前方のカメラで見て、見える物を答えます。今何が見えるか聞かれた時に使います。
    画像は PC の中で認識し、説明の文章だけを返します。約10秒かかります。
    細い家具の脚などを見落とすことがあるので、走行してよいかの判断には使いません。
    questionには、利用者が知りたいことを短く書きます。"""
    result = await _vision.describe(question)
    if result.ok:
        return f"カメラで見た結果: {result.text}"
    return f"{result.text}（詳細: {result.detail}）"
