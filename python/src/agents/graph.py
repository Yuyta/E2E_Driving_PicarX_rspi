import sqlite3
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.body_context import STATUS_TIMEOUT_SEC, fetch_body_context
from src.agents.state import AgentState
from src.config import ModelPreset, settings
from src.llm_factory import LLMFactory
from src.memory.store import get_memory_store
from src.picarx_client.client import PicarxClient
from src.token_optimizer import TokenOptimizer
from src.tools import ALL_TOOLS

optimizer = TokenOptimizer()
CONTEXT_HEADER = "参考情報。発言への回答に必要な時だけ使う。そのまま読み上げない。"
ROUTER_PROMPT = (
    "あなたはPiCar-Xのメインコントローラです。ユーザーの指示を達成するために適切なツールを呼び出すか、"
    "簡潔に回答してください。\n行動や記憶の指示には、説明せず必ずツールを呼び出してください。\n"
    "あなたの返事はそのまま読み上げられます。今見えている物を聞かれたら look を使います。"
    "look は細い物を見落とすので、走行してよいかの根拠にしないでください。"
)


def with_context(messages: List[BaseMessage], context: str) -> List[BaseMessage]:
    """毎ターン変わる情報を、最新の発言の直前に別のメッセージとして置く。

    発言に混ぜるより、実際の会話の比較でツール呼び出しが正確だった（qwen3.5:4b で 6/7 から 7/7）。
    Gemini では、途中の SystemMessage もシステム指示にまとめて送られる。
    """
    result = list(messages)
    for index in range(len(result) - 1, -1, -1):
        if isinstance(result[index], HumanMessage):
            result.insert(index, SystemMessage(content=f"{CONTEXT_HEADER}\n{context}"))
            break
    return result


def memory_context(memory, messages: List[BaseMessage]) -> str:
    """直近の発言に関連する記憶を返す。記憶が使えなくても会話は止めない。"""
    query = next(
        (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)),
        "",
    )
    try:
        return memory.build_context(query)
    except sqlite3.Error:
        return ""


def create_agent_graph(preset: Optional[ModelPreset] = None, status_client=None, memory=None):
    """会話 Agent を作る。Gemini は LLM とツール実行を往復する1つの Agent にする。

    ローカル LLM は呼び出しを減らすため、同じ `ainvoke` を持つ専用の Agent を使う。
    """
    # 応答を待たせないよう、状態の取得は短く打ち切る
    status_client = status_client or PicarxClient(timeout=STATUS_TIMEOUT_SEC)
    if ModelPreset(preset or settings.default_model) == ModelPreset.OLLAMA_LOCAL:
        from src.agents.local_agent import LocalAgent

        return LocalAgent(status_client=status_client, memory=memory)

    llm = LLMFactory.create(preset=preset).bind_tools(ALL_TOOLS)

    async def agent_node(state: AgentState) -> dict:
        # トークン節約: 会話履歴の圧縮適用
        compressed_msgs = optimizer.compress_history(state["messages"])
        body_context = await fetch_body_context(status_client)
        recalled = memory_context(memory or get_memory_store(), state["messages"])
        sections = "\n\n".join(section for section in (body_context, recalled) if section)
        system_prompt = SystemMessage(content=ROUTER_PROMPT)
        response = await llm.ainvoke([system_prompt] + with_context(compressed_msgs, sections))
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
    # ツール実行後は Agent に戻って結果を確認する
    workflow.add_edge("tools", "agent")
    return workflow.compile()
