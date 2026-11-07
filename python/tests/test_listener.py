"""音声入力のテスト。マイクと Whisper は偽物に置き換え、CI では音声ライブラリを使わない"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.voice.listener import (
    INITIAL_PROMPT,
    MAX_GAIN,
    MAX_SECONDS,
    NOT_HEARD,
    SAMPLE_RATE,
    TOO_QUIET,
    TOO_SHORT,
    Recorder,
    Transcriber,
    normalize,
)


def speech(seconds=1.0, peak=0.03):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (peak * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeStream:
    def __init__(self, callback):
        self.callback = callback
        self.started = self.stopped = self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def feed(self, audio):
        self.callback(audio.reshape(-1, 1), len(audio), None, None)


class FakeWhisper:
    def __init__(self, text="こんにちは"):
        self.text = text
        self.received = None

    def transcribe(self, audio, **kwargs):
        self.received = (audio, kwargs)
        return [SimpleNamespace(text=self.text)], None


def test_quiet_voice_is_raised_but_gain_is_limited():
    quiet = speech(peak=0.026)  # 実測の内蔵マイクのピーク
    assert np.max(np.abs(normalize(quiet))) == pytest.approx(0.5, rel=0.01)

    noise = speech(peak=0.001)
    assert np.max(np.abs(normalize(noise))) == pytest.approx(0.001 * MAX_GAIN, rel=0.01)


def test_recorder_returns_what_was_heard_between_press_and_release():
    streams = []
    recorder = Recorder(stream_factory=lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    recorder.start()
    assert recorder.recording
    streams[0].feed(speech(0.5))
    streams[0].feed(speech(0.5))

    audio = recorder.stop()
    assert not recorder.recording
    assert len(audio) == SAMPLE_RATE
    assert streams[0].stopped and streams[0].closed


def test_recorder_keeps_only_the_newest_recording():
    streams = []
    recorder = Recorder(stream_factory=lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    recorder.start()
    streams[-1].feed(speech(0.5))
    recorder.stop()
    recorder.start()
    streams[-1].feed(speech(0.25))
    assert len(recorder.stop()) == SAMPLE_RATE // 4


def test_very_long_recordings_are_cut():
    streams = []
    recorder = Recorder(stream_factory=lambda cb: streams.append(FakeStream(cb)) or streams[-1])
    recorder.start()
    streams[0].feed(speech(MAX_SECONDS + 2))
    assert len(recorder.stop()) == int(MAX_SECONDS * SAMPLE_RATE)


def test_stop_without_recording_returns_nothing():
    assert len(Recorder(stream_factory=FakeStream).stop()) == 0


def test_transcriber_uses_japanese_and_normalized_audio():
    whisper = FakeWhisper("  電池どう？ ")
    transcriber = Transcriber("small", model_factory=lambda name: whisper)
    result = transcriber.transcribe(speech(peak=0.026))

    assert result.text == "電池どう？"
    assert result.problem is None
    audio, options = whisper.received
    assert options["language"] == "ja"
    assert options["vad_filter"] is True
    assert np.max(np.abs(audio)) == pytest.approx(0.5, rel=0.01)


@pytest.mark.parametrize(
    "audio, problem",
    [(speech(0.2), TOO_SHORT), (speech(peak=0.001), TOO_QUIET)],
)
def test_short_or_silent_audio_is_not_sent_to_whisper(audio, problem):
    # 無音から「ご視聴ありがとうございました」のような文を作らせない
    loaded = []
    transcriber = Transcriber("small", model_factory=lambda name: loaded.append(name))
    assert transcriber.transcribe(audio).problem == problem
    assert loaded == []


def test_nothing_recognized_is_reported():
    transcriber = Transcriber("small", model_factory=lambda name: FakeWhisper(""))
    assert transcriber.transcribe(speech()).problem == NOT_HEARD


def test_model_is_loaded_once_and_can_be_chosen_by_environment(monkeypatch):
    monkeypatch.setenv("PICARX_WHISPER_MODEL", "large-v3-turbo")
    loaded = []
    transcriber = Transcriber(model_factory=lambda name: loaded.append(name) or FakeWhisper())
    transcriber.load()
    transcriber.transcribe(speech())
    assert loaded == ["large-v3-turbo"]


def test_transcriber_gives_whisper_common_phrases_but_no_motion_commands():
    from src.vision.local_vlm import MOTION_WORDS

    whisper = FakeWhisper("止まって")
    Transcriber("small", model_factory=lambda name: whisper).transcribe(speech(peak=0.026))
    prompt = whisper.received[1]["initial_prompt"]
    assert prompt == INITIAL_PROMPT
    assert "止まって" in prompt
    # 聞き取れない時に手がかりの文がそのまま出ても、車が動き出さないようにする
    assert not MOTION_WORDS.search(prompt)
