from langchain_core.tools import tool

from src.picarx_client.client import PicarxClient

_client = PicarxClient()


@tool
async def take_snapshot(width: int = 320, quality: int = 70) -> str:
    """
    カメラからオンデマンドで軽量撮影を行います。
    トークンを節約するため、常時送信ではなく必要なときのみ呼び出します。
    """
    try:
        data = await _client.get_camera_snapshot(width=width, quality=quality)
        # 画像サイズ等のメタデータを返し、実画像データはエージェントステートまたはキャッシュに格納
        return (
            f"撮影成功: 解像度 {data.width}x{data.height}, Base64長: {len(data.image_base64)}文字"
        )
    except Exception as e:
        return f"撮影エラー: {str(e)}"
