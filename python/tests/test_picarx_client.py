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
    result = await client.move(DriveDirection.FORWARD, speed=40, ttl_ms=1000)

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


@respx.mock
def test_client_sends_bearer_token_on_sync_and_async_paths():
    route = respx.post(f"{BASE_URL}/api/drive/stop").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert client.stop_sync() is True
    assert route.calls.last.request.headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
@respx.mock
async def test_client_async_request_sends_bearer_token():
    route = respx.get(f"{BASE_URL}/api/sensor/ultrasonic").mock(
        return_value=httpx.Response(200, json={"distance_cm": 1.0})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    await client.get_ultrasonic()
    assert route.calls.last.request.headers["authorization"] == "Bearer secret"


@respx.mock
def test_client_omits_header_when_token_is_empty():
    route = respx.post(f"{BASE_URL}/api/drive/stop").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="")
    client.stop_sync()
    assert "authorization" not in route.calls.last.request.headers


@respx.mock
def test_emergency_stop_sync_uses_latching_endpoint():
    route = respx.post(f"{BASE_URL}/api/drive/emergency_stop").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert client.emergency_stop_sync() is True
    assert route.called


@respx.mock
def test_emergency_stop_sync_raises_on_failure():
    respx.post(f"{BASE_URL}/api/drive/emergency_stop").mock(
        return_value=httpx.Response(401, json={"status": "error"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="wrong")
    with pytest.raises(httpx.HTTPStatusError):
        client.emergency_stop_sync()


@pytest.mark.asyncio
@respx.mock
async def test_reset_emergency_stop_uses_reset_endpoint():
    route = respx.post(f"{BASE_URL}/api/drive/emergency_stop/reset").mock(
        return_value=httpx.Response(200, json={"status": "ok"})
    )
    client = PicarxClient(base_url=BASE_URL, api_token="secret")
    assert await client.reset_emergency_stop() is True
    assert route.called
