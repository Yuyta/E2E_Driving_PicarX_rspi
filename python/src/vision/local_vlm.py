"""カメラの画像を PC のローカル VLM に見せて、見える物を文章にする。

- 画像は Pi から PC へ1枚だけ取り、PC の外へは送らない。Gemini の Agent には説明の文章だけを返す
- CPU で1枚あたり約8秒かかる。聞かれた時だけ使う
- モデルの読み込みに約5.6秒かかるので、GUI の起動時に読み込み、30分は残す（`warm_up`）
- Gemini が look を使うと決める前に、見始めておく（`prefetch`）。Gemini の1往復分が縮む
- 生成した画像の試験では、床に置いた細い棒を見落とした。走行の安全の判断には使わない
"""

import asyncio
import re
import time
from typing import NamedTuple, Optional

import httpx
from langchain_core.messages import HumanMessage

from src.config import settings
from src.llm_factory import OLLAMA_NUM_CTX, OLLAMA_NUM_THREAD, LLMFactory
from src.picarx_client.client import PicarxClient

# 画像処理部を持つ元のモデル。会話用のテキスト専用モデルとは別に読み込まれる
VLM_MODEL = "qwen3.5:4b"
# 読み込みに約5.6秒かかる（2026-11-07 計測）。メモリを約3.2GB 使うが、会話の間は残す
VLM_KEEP_ALIVE = "30m"
MAX_DESCRIPTION_TOKENS = 120
# 初回はモデルの読み込みを含む
VLM_TIMEOUT_SEC = 60.0
CAMERA_TIMEOUT_SEC = 6.0
# 先に見た結果は、この秒数までに使われなければ捨てる。古い画像で答えないため
PREFETCH_MAX_AGE_SEC = 20.0

LOOK_PROMPT = (
    "これは PiCar-X（床を走る小型ロボットカー）の前方カメラの画像です。"
    "質問に、日本語の1〜2文で答えてください。"
    "画像に写っている物だけを答え、はっきりしない物は「よく分からない」と言ってください。\n"
    "質問: {question}"
)
# 今見えている物を聞く発言。過去の話は記憶で答えるので含めない
LOOK_QUESTION = re.compile(
    r"(何|なに)(が|か)(見え|写|映|ある|置)|見えるもの|見えてる|見えますか|"
    r"周り.{0,6}(様子|どう)|カメラ.{0,6}(見|映|写)|"
    r"(映|写)っ(た|てる|ている)(もの|物)"
)
PAST_WORDS = re.compile(r"さっき|昨日|きのう|前回|この前|先週|覚え")
# 動いてから見る頼みごと。先に見ると動く前の画像になるので、先には見ない
MOTION_WORDS = re.compile(
    r"進ん|進め|前進|下が|さが|後退|バック|戻っ|戻れ|曲が|回っ|回れ|向い|向け|動い|動か|走っ|走ら|行っ|来て"
)
ACKNOWLEDGEMENT = "見てみます。"
CAMERA_FAILED = "カメラの画像を取得できませんでした。"
VLM_FAILED = "画像を認識できませんでした。"


def is_look_question(text: str) -> bool:
    return bool(LOOK_QUESTION.search(text)) and not PAST_WORDS.search(text)


class Look(NamedTuple):
    """見た結果。失敗した時は `ok` が False で、`text` に利用者へ伝える文がある"""

    text: str
    ok: bool
    seconds: float = 0.0
    detail: Optional[str] = None


class LocalVision:
    def __init__(self, client=None, llm=None, clock=time.monotonic, http=None):
        self._client = client
        self._llm = llm
        self._clock = clock
        self._http = http
        self._prefetched = None  # (見始めた時刻, 結果を待つ task)

    def _camera(self):
        if self._client is None:
            self._client = PicarxClient(timeout=CAMERA_TIMEOUT_SEC)
        return self._client

    def _model(self):
        if self._llm is None:
            self._llm = LLMFactory.create_local(
                model=VLM_MODEL,
                temperature=0.0,
                num_predict=MAX_DESCRIPTION_TOKENS,
                keep_alive=VLM_KEEP_ALIVE,
                client_kwargs={"timeout": VLM_TIMEOUT_SEC},
            )
        return self._llm

    async def warm_up(self) -> bool:
        """モデルを読み込んでおく。失敗しても、見る時に読み込み直すだけなので会話は止めない"""
        # 見る時と同じ設定で読み込む。num_ctx が違うと、見る時に読み込み直しになる
        body = {
            "model": VLM_MODEL,
            "keep_alive": VLM_KEEP_ALIVE,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "num_thread": OLLAMA_NUM_THREAD},
        }
        try:
            if self._http is not None:
                response = await self._http.post("/api/generate", json=body)
            else:
                async with httpx.AsyncClient(
                    base_url=settings.ollama_base_url, timeout=VLM_TIMEOUT_SEC
                ) as http:
                    response = await http.post("/api/generate", json=body)
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def prefetch(self, question: str) -> bool:
        """見る質問を受けた時点で見始める。動いてから見る頼みごとでは見ない。

        実行中のイベントループの中から呼ぶ。次の `describe` がこの結果を使う。
        """
        self._discard_prefetch()
        if MOTION_WORDS.search(question):
            return False
        task = asyncio.ensure_future(self._look_now(question))
        self._prefetched = (self._clock(), task)
        return True

    async def describe(self, question: str) -> Look:
        if self._prefetched is not None:
            started, task = self._prefetched
            self._prefetched = None
            if self._clock() - started <= PREFETCH_MAX_AGE_SEC:
                return await task
            task.cancel()
        return await self._look_now(question)

    def _discard_prefetch(self) -> None:
        if self._prefetched is not None:
            self._prefetched[1].cancel()
            self._prefetched = None

    async def _look_now(self, question: str) -> Look:
        started = self._clock()
        try:
            snapshot = await self._camera().get_camera_snapshot()
        except (httpx.HTTPError, ValueError) as error:
            return Look(CAMERA_FAILED, False, detail=str(error))

        message = HumanMessage(
            content=[
                {"type": "text", "text": LOOK_PROMPT.format(question=question.strip())},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{snapshot.image_base64}",
                },
            ]
        )
        try:
            response = await self._model().ainvoke([message])
        except (httpx.HTTPError, ConnectionError, TimeoutError, ValueError) as error:
            return Look(VLM_FAILED, False, detail=str(error))
        text = str(response.content).strip()
        if not text:
            return Look(VLM_FAILED, False, detail="空の応答")
        return Look(text, True, self._clock() - started)


_shared: Optional[LocalVision] = None


def shared_vision() -> LocalVision:
    """Gemini の look と、ローカル LLM の Agent と、GUI の先読みで同じものを使う"""
    global _shared
    if _shared is None:
        _shared = LocalVision()
    return _shared
