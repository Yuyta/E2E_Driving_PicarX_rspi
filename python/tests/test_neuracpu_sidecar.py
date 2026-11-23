import re

import pytest

from src.neuracpu.sidecar import (
    Wikipedia,
    best_passages,
    is_relevant,
    keywords,
    line_range,
    search_terms,
    title_candidates,
)


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


@pytest.mark.parametrize(
    "dense, sparse, expected",
    [
        (0.66, 11.7, True),
        (0.53, 4.2, False),
        (0.56, 7.2, False),
        (0.63, 0.0, True),
        (0.56, 9.7, False),
        (0.44, 1.4, False),
    ],
)
def test_is_relevant(dense, sparse, expected):
    assert is_relevant({"dense_score": dense, "sparse_score": sparse}) is expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("将棋の駒は全部で何枚？", "将棋 駒"),
        ("東京タワーの高さは？", "東京タワー"),  # 送り仮名で切れた1文字は外す
        ("ダイナマイトを発明したのは誰？", "ダイナマイト 発明"),
        ("ネットで llama.cpp について調べて", "llama.cpp"),
        ("Raspberry Pi 5 の最新の OS のバージョンは？", "Raspberry Pi 5 最新 OS バージョン"),
    ],
)
def test_keywords_for_wikipedia(query, expected):
    assert keywords(query) == expected


def test_title_candidates_try_the_noun_phrase_then_the_longest_word():
    assert title_candidates("将棋の駒は全部で何枚？") == ["将棋の駒", "将棋"]
    assert title_candidates("東京タワーの高さは？") == ["東京タワー"]


def test_wikipedia_waits_between_requests_and_caches_answers():
    now = [100.0]
    slept = []
    wiki = Wikipedia(clock=lambda: now[0], sleep=slept.append)
    fetched = []

    def fake_fetch(request):
        fetched.append(request.full_url)
        if "list=search" in request.full_url:
            return {"query": {"search": [{"title": "富士山"}]}}
        page = {"title": "富士山", "extract": "標高3776m", "fullurl": "u"}
        return {"query": {"pages": {"1": page}}}

    wiki._fetch = fake_fetch
    assert wiki.search("富士山の高さは？")[0]["title"] == "富士山"
    assert wiki.search("富士山の高さは？")[0]["title"] == "富士山"
    assert len(fetched) == 2  # 検索と本文の2回。2回目は覚えた答えを使う
    wiki.search("エベレストの標高は？")
    assert slept and all(s == pytest.approx(1.0) for s in slept)  # 続けて問い合わせる時は1秒あける


def test_wikipedia_backs_off_after_429():
    import urllib.error

    now = [0.0]
    wiki = Wikipedia(clock=lambda: now[0], sleep=lambda s: None)
    calls = []

    def limited(request):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    wiki._fetch = limited
    with pytest.raises(urllib.error.HTTPError):
        wiki.search("富士山の高さは？")
    with pytest.raises(RuntimeError):
        wiki.search("エベレストの標高は？")  # 待っている間は問い合わせない
    assert len(calls) == 1
    now[0] += Wikipedia.BACKOFF_SEC + 1
    with pytest.raises(urllib.error.HTTPError):
        wiki.search("エベレストの標高は？")
    assert len(calls) == 2
