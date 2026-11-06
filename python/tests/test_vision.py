from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.local_agent import LocalAgent
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
