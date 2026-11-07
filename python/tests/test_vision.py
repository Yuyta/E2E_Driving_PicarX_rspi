import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.local_agent import LocalAgent
from src.llm_factory import OLLAMA_NUM_CTX
from src.memory.store import MemoryStore
from src.tools import ALL_TOOLS
from src.vision.local_vlm import (
    CAMERA_FAILED,
    VLM_FAILED,
    LocalVision,
    Look,
    is_look_question,
)
from tests.test_local_agent import make_llm, make_status_client, make_tools


class Camera:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def get_camera_snapshot(self):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(image_base64="QUJD")


def vlm(reply="青い箱が見えます。", error=None):
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=reply), side_effect=error)
    return llm


@pytest.mark.parametrize(
    "text",
    [
        "何が見える？",
        "前に何がある？",
        "周りの様子どう？",
        "カメラに何か映ってる？",
        "ドアは見えますか",
    ],
)
def test_questions_about_what_is_visible_are_detected(text):
    assert is_look_question(text)


@pytest.mark.parametrize(
    "text",
    ["さっき何が見えた？", "電池どう？", "前に進んで", "昨日キッチンに何があった？"],
)
def test_other_questions_are_not_look_questions(text):
    assert not is_look_question(text)


@pytest.mark.asyncio
async def test_one_snapshot_is_sent_to_the_local_model_with_the_question():
    llm = vlm()
    result = await LocalVision(client=Camera(), llm=llm).describe("何が見える？")

    assert result.ok and result.text == "青い箱が見えます。"
    content = llm.ainvoke.await_args.args[0][0].content
    assert "質問: 何が見える？" in content[0]["text"]
    assert content[1]["image_url"] == "data:image/jpeg;base64,QUJD"


@pytest.mark.asyncio
async def test_camera_failure_is_reported_without_calling_the_model():
    llm = vlm()
    camera = Camera(error=httpx.ConnectError("unreachable"))
    result = await LocalVision(client=camera, llm=llm).describe("何が見える？")
    assert (result.ok, result.text) == (False, CAMERA_FAILED)
    llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("llm", [vlm(error=httpx.ConnectError("ollama down")), vlm(reply="  ")])
async def test_model_failure_or_empty_answer_is_reported(llm):
    result = await LocalVision(client=Camera(), llm=llm).describe("何が見える？")
    assert (result.ok, result.text) == (False, VLM_FAILED)


def test_gemini_agent_can_look():
    assert "look" in {tool.name for tool in ALL_TOOLS}


@pytest.mark.asyncio
async def test_look_tool_returns_only_text_to_gemini():
    from src.tools import vision_tool

    with patch.object(vision_tool, "_vision") as vision:
        vision.describe = AsyncMock(return_value=Look("ドアが見えます。", True, 9.0))
        result = await vision_tool.look.ainvoke({"question": "何が見える？"})
    assert result == "カメラで見た結果: ドアが見えます。"


def local_agent(vision, *replies):
    return LocalAgent(
        llm=make_llm(*replies),
        status_client=make_status_client(),
        memory=MemoryStore(":memory:"),
        tools=make_tools(),
        vision=vision,
    )


@pytest.mark.asyncio
async def test_local_agent_answers_look_questions_without_the_text_llm():
    vision = AsyncMock()
    vision.describe = AsyncMock(return_value=Look("青い箱が見えます。", True))
    agent = local_agent(vision)
    result = await agent.ainvoke({"messages": [HumanMessage(content="周りの様子どう？")]})

    assert result["messages"][-1].content == "青い箱が見えます。"
    agent._llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_local_agent_can_choose_look_itself():
    vision = AsyncMock()
    vision.describe = AsyncMock(return_value=Look("椅子があります。", True))
    agent = local_agent(vision, {"action": "look"})
    result = await agent.ainvoke({"messages": [HumanMessage(content="ちょっと前を確認して")]})
    assert result["messages"][-1].content == "椅子があります。"


@pytest.mark.asyncio
async def test_local_agent_keeps_error_details_out_of_speech():
    vision = AsyncMock()
    vision.describe = AsyncMock(return_value=Look(CAMERA_FAILED, False, detail="timeout"))
    agent = local_agent(vision)
    message = (await agent.ainvoke({"messages": [HumanMessage(content="何が見える？")]}))[
        "messages"
    ][-1]
    assert "timeout" in message.content
    assert message.additional_kwargs["speech"] == CAMERA_FAILED


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.mark.asyncio
async def test_prefetched_look_is_used_by_the_next_describe():
    camera, llm = Camera(), vlm()
    vision = LocalVision(client=camera, llm=llm)
    assert vision.prefetch("何が見える？")
    result = await vision.describe("前方に見える物")
    assert (result.ok, result.text) == (True, "青い箱が見えます。")
    # 先に見た1回だけで、もう一度は見ない
    assert (camera.calls, llm.ainvoke.await_count) == (1, 1)
    # 使った結果は捨てる。次は見直す
    await vision.describe("何が見える？")
    assert camera.calls == 2


@pytest.mark.asyncio
async def test_old_prefetched_look_is_not_used():
    clock = Clock()
    camera = Camera()
    vision = LocalVision(client=camera, llm=vlm(), clock=clock)
    vision.prefetch("何が見える？")
    await asyncio.sleep(0.01)
    clock.now += 30
    await vision.describe("何が見える？")
    assert camera.calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["前に進んでから何が見えるか教えて", "右に曲がって何が見える？"])
async def test_requests_to_move_first_are_not_prefetched(text):
    camera = Camera()
    vision = LocalVision(client=camera, llm=vlm())
    assert not vision.prefetch(text)
    await vision.describe(text)
    assert camera.calls == 1


@pytest.mark.asyncio
async def test_warm_up_loads_the_vlm_and_keeps_it():
    http = AsyncMock()
    http.post = AsyncMock(return_value=httpx.Response(200, request=httpx.Request("POST", "x")))
    assert await LocalVision(http=http).warm_up()
    body = http.post.await_args.kwargs["json"]
    assert (body["model"], body["keep_alive"]) == ("qwen3.5:4b", "30m")
    # 見る時と同じ文脈の長さで読み込まないと、見る時に読み込み直しになる
    assert body["options"]["num_ctx"] == OLLAMA_NUM_CTX


@pytest.mark.asyncio
async def test_warm_up_failure_does_not_raise():
    http = AsyncMock()
    http.post = AsyncMock(side_effect=httpx.ConnectError("ollama down"))
    assert not await LocalVision(http=http).warm_up()
