from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.token_optimizer import TokenOptimizer


def test_sensor_diff_filter():
    optimizer = TokenOptimizer()

    # 初回は必ずレポートされる
    assert optimizer.should_report_distance(30.0) is True

    # 差分が2.0cm未満（例: 0.5cm）の場合はスキップされる（トークン節約）
    assert optimizer.should_report_distance(30.5) is False
    assert optimizer.should_report_distance(29.0) is False

    # 差分が2.0cm以上の場合はレポートされる
    assert optimizer.should_report_distance(33.0) is True
    assert optimizer.should_report_distance(25.0) is True


def test_compress_history_short():
    optimizer = TokenOptimizer()
    messages = [
        HumanMessage(content="こんにちは"),
        AIMessage(content="こんにちは！何か用ですか？"),
    ]
    # max_history_turns (10) 未満ならそのまま
    compressed = optimizer.compress_history(messages)
    assert len(compressed) == 2


def test_compress_history_long():
    optimizer = TokenOptimizer()
    messages = []
    for i in range(15):
        messages.append(HumanMessage(content=f"質問 {i}"))
        messages.append(AIMessage(content=f"回答 {i}"))

    # 30件あるので圧縮されて先頭に要約SystemMessageが入り、直近のみ残る
    compressed = optimizer.compress_history(messages)
    assert len(compressed) < len(messages)
    assert "【過去の対話要約】" in compressed[0].content


def test_compress_history_keeps_tool_calls_with_their_results():
    optimizer = TokenOptimizer()
    call = AIMessage(content="", tool_calls=[{"name": "stop_moving", "args": {}, "id": "x"}])
    messages = [HumanMessage(content="質問"), AIMessage(content="回答")] * 3 + [
        HumanMessage(content="止まって"),
        call,
        ToolMessage(content="停止完了", tool_call_id="x"),
        AIMessage(content="停止しました"),
        HumanMessage(content="次"),
        AIMessage(content="はい"),
        HumanMessage(content="最新"),
    ]
    compressed = optimizer.compress_history(messages)
    assert "【過去の対話要約】" in compressed[0].content
    assert isinstance(compressed[1], HumanMessage)
    assert not any(isinstance(message, ToolMessage) for message in compressed)
