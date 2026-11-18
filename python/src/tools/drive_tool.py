from langchain_core.tools import tool

from src.models import DRIVE_SPEED, MIN_MOVING_SPEED, DriveDirection
from src.picarx_client.client import PicarxClient

_client = PicarxClient()


def _moving_speed(speed: int) -> int:
    """0 以外で車輪が回らない遅さの速度は、動く最低の速度に上げる。"""
    if 0 < speed < MIN_MOVING_SPEED:
        return MIN_MOVING_SPEED
    return speed


@tool
async def drive_forward(speed: int = DRIVE_SPEED, duration_ms: int = 500) -> str:
    """PiCar-Xを前進させます。speedは40-100(40未満では車輪が回らないので40にします。普段は50)、duration_msは走行時間(ミリ秒)です。"""
    speed = _moving_speed(speed)
    try:
        ok = await _client.drive_for(DriveDirection.FORWARD, speed, duration_ms)
        return f"前進成功 (速度:{speed}, 時間:{duration_ms}ms)" if ok else "前進失敗"
    except Exception as e:
        return f"前進エラー: {str(e)}"


@tool
async def drive_backward(speed: int = DRIVE_SPEED, duration_ms: int = 500) -> str:
    """PiCar-Xを後退させます。speedは40-100(40未満では車輪が回らないので40にします。普段は50)、duration_msは走行時間(ミリ秒)です。"""
    speed = _moving_speed(speed)
    try:
        ok = await _client.drive_for(DriveDirection.BACKWARD, speed, duration_ms)
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
