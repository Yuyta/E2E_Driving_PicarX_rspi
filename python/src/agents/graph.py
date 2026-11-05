import sqlite3
from typing import List, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.body_context import fetch_body_context
from src.agents.state import AgentState
from src.config import ModelPreset
from src.llm_factory import LLMFactory
from src.memory.store import get_memory_store
from src.picarx_client.client import PicarxClient
from src.token_optimizer import TokenOptimizer
from src.tools import (
    ALL_TOOLS,
    drive_backward,
    drive_forward,
    read_distance,
    speak_text,
    steer_wheels,
    stop_moving,
    take_snapshot,
)

optimizer = TokenOptimizer()
STATUS_TIMEOUT_SEC = 1.0
CONTEXT_HEADER = "参考情報。発言への回答に必要な時だけ使う。そのまま読み上げない。"
ROUTER_PROMPT = (
    "あなたはPiCar-Xのメインコントローラです。ユーザーの指示を達成するために適切なツールを呼び出すか、"
    "簡潔に回答してください。\n行動や記憶の指示には、説明せず必ずツールを呼び出してください。"
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
    """マルチエージェント StateGraph を構築してコンパイル"""
    llm = LLMFactory.create(preset=preset)
    # 応答を待たせないよう、状態の取得は短く打ち切る
    status_client = status_client or PicarxClient(timeout=STATUS_TIMEOUT_SEC)

    # ツールバインドLLM
    drive_tools = [drive_forward, drive_backward, steer_wheels, stop_moving, read_distance]
    vision_tools = [take_snapshot]
    voice_tools = [speak_text]

    llm_with_all_tools = llm.bind_tools(ALL_TOOLS)
    llm_drive = llm.bind_tools(drive_tools)
    llm_vision = llm.bind_tools(vision_tools)
    llm_voice = llm.bind_tools(voice_tools)

    # 1. Main Router Node (タスクの分解と振り分け)
    async def router_node(state: AgentState) -> dict:
        # トークン節約: 会話履歴の圧縮適用
        compressed_msgs = optimizer.compress_history(state["messages"])
        body_context = await fetch_body_context(status_client)
        recalled = memory_context(memory or get_memory_store(), state["messages"])
        sections = "\n\n".join(section for section in (body_context, recalled) if section)
        system_prompt = SystemMessage(content=ROUTER_PROMPT)
        response = await llm_with_all_tools.ainvoke(
            [system_prompt] + with_context(compressed_msgs, sections)
        )
        return {
            "messages": [response],
            "current_agent": "main",
            "is_finished": len(response.tool_calls) == 0,
        }

    # 2. Drive Sub-Agent Node
    async def drive_node(state: AgentState) -> dict:
        system_prompt = SystemMessage(
            content="あなたはPiCar-Xの走行担当エージェントです。安全第一で移動・操舵ツールを実行してください。"
        )
        response = await llm_drive.ainvoke([system_prompt] + state["messages"][-3:])
        return {"messages": [response], "current_agent": "drive"}

    # 3. Vision Sub-Agent Node
    async def vision_node(state: AgentState) -> dict:
        system_prompt = SystemMessage(
            content="あなたはPiCar-Xの視覚担当エージェントです。カメラ撮影ツールを実行して周囲の状況を報告してください。"
        )
        response = await llm_vision.ainvoke([system_prompt] + state["messages"][-3:])
        return {"messages": [response], "current_agent": "vision"}

    # 4. Voice Sub-Agent Node
    async def voice_node(state: AgentState) -> dict:
        system_prompt = SystemMessage(
            content="あなたはPiCar-Xの発話担当エージェントです。発話ツール(speak_text)を使って音声合成させてください。"
        )
        response = await llm_voice.ainvoke([system_prompt] + state["messages"][-3:])
        return {"messages": [response], "current_agent": "voice"}

    # ルーティング判定関数
    def route_after_main(state: AgentState) -> Literal["tools", "__end__"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return END

    # グラフの構築
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("drive_agent", drive_node)
    workflow.add_node("vision_agent", vision_node)
    workflow.add_node("voice_agent", voice_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_after_main,
        {
            "tools": "tools",
            END: END,
        },
    )
    # ツール実行後はrouterに戻って結果を確認
    workflow.add_edge("tools", "router")

    return workflow.compile()
