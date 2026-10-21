from unittest.mock import MagicMock, patch

from src.agents.graph import create_agent_graph
from src.config import ModelPreset


def test_agent_graph_compile():
    # LLMの生成をモックしてStateGraphのコンパイルを確認
    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm

    with patch("src.llm_factory.LLMFactory.create", return_value=mock_llm):
        app = create_agent_graph(preset=ModelPreset.GEMINI_FLASH)
        assert app is not None
        # グラフに必要なノードが存在するか確認
        node_names = set(app.nodes.keys())
        assert "router" in node_names
        assert "drive_agent" in node_names
        assert "vision_agent" in node_names
        assert "voice_agent" in node_names
        assert "tools" in node_names
