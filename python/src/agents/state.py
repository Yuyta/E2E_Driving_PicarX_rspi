import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """LangGraphマルチエージェント全体で共有する状態"""

    messages: Annotated[List[BaseMessage], operator.add]
    distance_cm: Optional[float]
    grayscale: Optional[List[int]]
    snapshot_base64: Optional[str]
    current_agent: str
    is_finished: bool
