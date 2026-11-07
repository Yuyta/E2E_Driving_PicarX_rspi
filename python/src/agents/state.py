import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """会話 Agent の状態。車体の状態と記憶は毎ターン取得するので持たない"""

    messages: Annotated[List[BaseMessage], operator.add]
