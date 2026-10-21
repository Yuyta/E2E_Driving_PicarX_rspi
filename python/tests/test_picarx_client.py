import httpx
import pytest
import respx

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

BASE_URL = "http://test-robot:8080"


@pytest.mark.asyncio
@respx.mock
async def test_client_move():
    route = respx.post(f"{BASE_URL}/api/drive/move").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL)
    result = await client.move(DriveDirection.FORWARD, speed=40, duration_ms=1000)

    assert result is True
    assert route.called
    payload = route.calls.last.request.read().decode()
    assert '"direction":"forward"' in payload


@pytest.mark.asyncio
@respx.mock
async def test_client_get_ultrasonic():
    respx.get(f"{BASE_URL}/api/sensor/ultrasonic").mock(
        return_value=httpx.Response(200, json={"distance_cm": 42.5})
    )
    client = PicarxClient(base_url=BASE_URL)
    dist = await client.get_ultrasonic()
    assert dist == 42.5


@pytest.mark.asyncio
@respx.mock
async def test_client_speak():
    route = respx.post(f"{BASE_URL}/api/audio/speak").mock(
        return_value=httpx.Response(200, json={"status": "spoken"})
    )
    client = PicarxClient(base_url=BASE_URL)
    ok = await client.speak("こんにちは")
    assert ok is True
    payload = route.calls.last.request.read().decode()
    assert "こんにちは" in payload
