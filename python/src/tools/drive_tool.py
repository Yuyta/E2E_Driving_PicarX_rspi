from langchain_core.tools import tool

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

_client = PicarxClient()


@tool
async def drive_forward(speed: int = 30, duration_ms: int = 500) -> str:
    """PiCar-Xを前進させます。speedは0-100、duration_msは走行時間(ミリ秒)です。"""
    try:
        ok = await _client.move(DriveDirection.FORWARD, speed=speed, duration_ms=duration_ms)
        return f"前進成功 (速度:{speed}, 時間:{duration_ms}ms)" if ok else "前進失敗"
    except Exception as e:
        return f"前進エラー: {str(e)}"


@tool
async def drive_backward(speed: int = 30, duration_ms: int = 500) -> str:
    """PiCar-Xを後退させます。speedは0-100、duration_msは走行時間(ミリ秒)です。"""
    try:
        ok = await _client.move(DriveDirection.BACKWARD, speed=speed, duration_ms=duration_ms)
        return f"後退成功 (速度:{speed}, 時間:{duration_ms}ms)" if ok else "後退失敗"
    except Exception as e:
        return f"後退エラー: {str(e)}"


@tool
async def steer_wheels(angle: int) -> str:
    """ステアリング角度を変更します。angleは-40(左折)〜+40(右折)です。0で直進方向に戻ります。"""
    try:
        ok = await _client.steer(angle=angle)
        return f"ステアリング設定完了: {angle}度" if ok else "ステアリング設定失敗"
    except Exception as e:
        return f"ステアリングエラー: {str(e)}"


@tool
async def stop_moving() -> str:
    """PiCar-Xの走行を停止します。"""
    try:
        ok = await _client.stop()
        return "停止完了" if ok else "停止失敗"
    except Exception as e:
        return f"停止エラー: {str(e)}"
