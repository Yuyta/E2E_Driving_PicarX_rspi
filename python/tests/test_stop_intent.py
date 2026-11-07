import pytest

from src.agents.stop_intent import StopIntent, parse_stop_intent


@pytest.mark.parametrize(
    "text",
    [
        "止まって",
        "止まって！",
        "止まってください",
        "とまれ",
        "ストップ",
        "ストップ！！",
        " stop ",
        "STOP",
        "停止",
    ],
)
def test_stop_only_commands(text):
    assert parse_stop_intent(text) == StopIntent.ONLY_STOP


@pytest.mark.parametrize(
    "text",
    [
        "そこで止まって、右を見て",
        "止まってからキッチンへ行って",
        "いったんストップして電池を教えて",
    ],
)
def test_stop_with_more_instructions(text):
    assert parse_stop_intent(text) == StopIntent.STOP_AND_MORE


@pytest.mark.parametrize(
    "text", ["こんにちは", "電池どう？", "少し前に進んで", "危ないからやめて", ""]
)
def test_other_messages_are_left_to_the_llm(text):
    assert parse_stop_intent(text) == StopIntent.NONE
