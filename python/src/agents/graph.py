from typing import Literal, Optional

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from src.agents.state import AgentState
from src.config import ModelPreset
from src.llm_factory import LLMFactory
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


def create_agent_graph(preset: Optional[ModelPreset] = None):
    """マルチエージェント StateGraph を構築してコンパイル"""
    llm = LLMFactory.create(preset=preset)

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
        system_prompt = SystemMessage(
            content=(
                "あなたはPiCar-Xのメインコントローラです。ユーザーの指示を達成するために適切なツールを呼び出すか、"
                "簡潔に回答してください。"
            )
        )
        response = await llm_with_all_tools.ainvoke([system_prompt] + compressed_msgs)
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
