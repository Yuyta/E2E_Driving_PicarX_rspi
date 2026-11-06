import asyncio
import threading

import pytest

from src.voice.speaker import MAX_SENTENCE_CHARS, Speaker, split_sentences


class FakeClient:
    def __init__(self, on_speak=None, stop_delay=0.0, fail_stop=False):
        self.spoken = []
        self.stops = 0
        self._on_speak = on_speak
        self._stop_delay = stop_delay
        self._fail_stop = fail_stop
        self.stop_reached = threading.Event()

    async def speak(self, text):
        self.spoken.append(text)
        if self._on_speak:
            self._on_speak(text)
        return True

    def stop_speech_sync(self):
        threading.Event().wait(self._stop_delay)
        self.stops += 1
        self.stop_reached.set()
        if self._fail_stop:
            raise ConnectionError("Pi に届きません")
        return True


def test_split_sentences_keeps_short_pieces_with_the_previous_sentence():
    assert split_sentences("こんにちは。はい。電池は83%です！") == [
        "こんにちは。はい。",
        "電池は83%です！",
    ]


def test_split_sentences_cuts_long_sentences_at_commas():
    text = "、".join(["とても長い説明が続きます"] * 10) + "。"
    parts = split_sentences(text)
    assert len(parts) > 1
    assert all(len(part) <= MAX_SENTENCE_CHARS for part in parts)
    assert "".join(parts) == text


def test_split_sentences_ignores_empty_text():
    assert split_sentences("  \n ") == []


@pytest.mark.asyncio
async def test_say_sends_each_sentence_in_order():
    client = FakeClient()
    assert await Speaker(client).say("前に進みました。次は何をしますか？") is True
    assert client.spoken == ["前に進みました。", "次は何をしますか？"]


@pytest.mark.asyncio
async def test_interrupt_drops_the_remaining_sentences_and_stops_the_pi():
    speaker = None
    client = FakeClient(on_speak=lambda _: speaker.interrupt())
    speaker = Speaker(client)

    assert await speaker.say("一文目です。二文目です。三文目です。") is False
    assert client.spoken == ["一文目です。"]
    assert client.stop_reached.wait(1.0)


@pytest.mark.asyncio
async def test_new_speech_waits_until_the_stop_reaches_the_pi():
    # 停止より先に新しい文が届くと、その文まで止まってしまう
    client = FakeClient(stop_delay=0.2)
    speaker = Speaker(client)
    speaker.interrupt()

    assert await speaker.say("新しい返事です。") is True
    assert client.stops == 1
    assert client.spoken == ["新しい返事です。"]


@pytest.mark.asyncio
async def test_failed_stop_is_reported_and_does_not_block_new_speech():
    errors = []
    client = FakeClient(fail_stop=True)
    speaker = Speaker(client, on_error=errors.append)
    speaker.interrupt()

    assert await speaker.say("続けます。") is True
    assert len(errors) == 1


@pytest.mark.asyncio
async def test_replies_do_not_interleave():
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowClient(FakeClient):
        async def speak(self, text):
            self.spoken.append(text)
            if text == "最初の一文。":
                started.set()
                await release.wait()
            return True

    client = SlowClient()
    speaker = Speaker(client)
    first = asyncio.create_task(speaker.say("最初の一文。最初の二文。"))
    await started.wait()
    second = asyncio.create_task(speaker.say("次の返事。"))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert client.spoken == ["最初の一文。", "最初の二文。", "次の返事。"]
