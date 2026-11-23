import re

import pytest

from src.neuracpu.sidecar import best_passages, line_range, search_terms


def tokens(text):
    return re.findall(r"\w", text)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("富士山の高さは？", "富士山の高さ"),
        ("ネットで llama.cpp について調べて", "llama.cpp"),
        ("日本の首都はどこ？", "日本の首都"),
        ("Qwen3.5 はいつ発表された？", "Qwen3.5 はいつ発表された"),
        ("？", "？"),
    ],
)
def test_search_terms_drops_question_wording(query, expected):
    assert search_terms(query) == expected


def test_best_passages_keeps_short_text():
    assert best_passages(tokens, "犬", "犬は動物です。猫も動物です。", 100) == (
        "犬は動物です。\n猫も動物です。"
    )


def test_best_passages_picks_overlapping_paragraphs_within_budget():
    text = "冒頭の定義です。\n" + "\n".join(f"関係のない段落{i}です。" * 3 for i in range(20))
    text += "\n富士山の標高は3776メートルです。"
    picked = best_passages(tokens, "富士山の標高", text, 60)
    assert "3776" in picked
    assert picked.startswith("冒頭")  # 冒頭は定義が多いので残す
    assert len(picked) <= 60 * 1.3 + 20


def test_line_range_drops_the_file_name():
    assert line_range({"path": "C:/x/manual.html", "locator": "manual.html:181-204"}) == "181-204行"
    assert line_range({"path": "a.pdf", "locator": "a.pdf:p3"}) == "p3"
