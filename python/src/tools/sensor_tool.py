from langchain_core.tools import tool

from src.picarx_client.client import PicarxClient

_client = PicarxClient()


@tool
async def read_distance() -> str:
    """前方の超音波センサーで障害物までの距離(cm)を測定して返します。"""
    try:
        distance = await _client.get_ultrasonic()
        return f"前方障害物までの距離: {distance:.1f}cm"
    except Exception as e:
        return f"距離測定エラー: {str(e)}"


@tool
async def read_grayscale() -> str:
    """床面のグレースケールセンサー値(左, 中央, 右)を取得します。ライン追従や崖検知に使用します。"""
    try:
        values = await _client.get_grayscale()
        return f"床面センサー値: [左:{values[0]}, 中央:{values[1]}, 右:{values[2]}]"
    except Exception as e:
        return f"センサー測定エラー: {str(e)}"
