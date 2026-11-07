from unittest.mock import patch

import pytest

from src.voice import breath
from src.voice.breath import add_breath_pauses

pytest.importorskip("pyopenjtalk")


# 実際の返事で「したこと」「います」が遅く読まれた文。下の結果は Docker の Open JTalk で
# 伸びないことを確かめた（2026-11-07）
@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "11月6日と11月7日に床を校正したこと、",
            "11月6日と11月7日に、床を校正したこと、",
        ),
        (
            "11月7日に前方の障害物に近づいて自動で停止したことを覚えています。",
            "11月7日に、前方の障害物に、近づいて自動で停止したことを覚えています。",
        ),
        (
            "11月7日に前方の障害物（13.0cm）に近づいて自動で停止したことを覚えています。",
            "11月7日に、前方の障害物（13.0cm）に、近づいて自動で停止したことを覚えています。",
        ),
    ],
)
def test_long_breaths_get_a_comma_after_a_particle(text, expected):
    assert add_breath_pauses(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "床を校正しました。私の名前はがくです。",
        "こんにちは、ゆうたさん！今日もよろしくお願いします。",
        "黒い椅子と、その横にあるペットボトルが見えます。",
    ],
)
def test_short_breaths_are_left_alone(text):
    assert add_breath_pauses(text) == text


def test_commas_are_not_put_next_to_symbols_or_inside_words():
    text = (
        "私の学習状況や覚えていることとしては、ユーザーの名前が「ゆうた」さんであること、"
        "そして私の名前が「がく」であることを記憶しています。"
        "前方の障害物（13.0cm）に近づいて自動で停止したことを覚えています。"
    )
    result = add_breath_pauses(text)
    assert result.replace("、", "") == text.replace("、", "")
    for bad in ("、）", "（、", "、」", "「、", "覚えて、いる"):
        assert bad not in result


def test_text_is_spoken_as_is_when_the_analyzer_is_unavailable():
    text = "11月6日と11月7日に床を校正したこと、"
    with patch.object(breath, "_run_frontend", return_value=None):
        assert add_breath_pauses(text) == text
