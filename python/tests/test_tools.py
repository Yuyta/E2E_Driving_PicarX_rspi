import httpx
import pytest
import respx

from src.config import settings
from src.tools.drive_tool import drive_forward, stop_moving
from src.tools.sensor_tool import read_distance
from src.tools.voice_tool import speak_text


@pytest.mark.asyncio
@respx.mock
async def test_tools_execution():
    respx.post(f"{settings.robot_base_url}/api/drive/move").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.post(f"{settings.robot_base_url}/api/drive/stop").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    respx.get(f"{settings.robot_base_url}/api/sensor/ultrasonic").mock(
        return_value=httpx.Response(200, json={"distance_cm": 25.0})
    )
    respx.post(f"{settings.robot_base_url}/api/audio/speak").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )

    # 1. Drive tool
    res_drive = await drive_forward.ainvoke({"speed": 30, "duration_ms": 500})
    assert "前進成功" in res_drive

    # 2. Stop tool
    res_stop = await stop_moving.ainvoke({})
    assert "停止完了" in res_stop

    # 3. Sensor tool
    res_sensor = await read_distance.ainvoke({})
    assert "25.0cm" in res_sensor

    # 4. Voice tool
    res_voice = await speak_text.ainvoke({"text": "テスト発話"})
    assert "テスト発話" in res_voice
