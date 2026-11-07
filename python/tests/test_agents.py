from unittest.mock import MagicMock, patch

from src.agents.graph import create_agent_graph
from src.config import ModelPreset
from src.tools import ALL_TOOLS


def test_agent_graph_is_a_single_agent_with_tools():
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    with patch("src.llm_factory.LLMFactory.create", return_value=mock_llm):
        app = create_agent_graph(preset=ModelPreset.GEMINI_FLASH)

    assert set(app.nodes.keys()) == {"__start__", "agent", "tools"}


def test_agent_cannot_speak_or_take_pictures_by_itself():
    # 発話は Speaker に一本化し、画像を見られないうちは撮影ツールを渡さない
    names = {tool.name for tool in ALL_TOOLS}
    assert "speak_text" not in names
    assert "take_snapshot" not in names
