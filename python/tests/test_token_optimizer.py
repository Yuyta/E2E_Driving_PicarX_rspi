from langchain_core.messages import AIMessage, HumanMessage

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
