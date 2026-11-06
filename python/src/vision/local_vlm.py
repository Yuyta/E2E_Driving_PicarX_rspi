"""カメラの画像を PC のローカル VLM に見せて、見える物を文章にする。

- 画像は Pi から PC へ1枚だけ取り、PC の外へは送らない。Gemini の Agent には説明の文章だけを返す
- CPU で1枚あたり約10秒かかる。聞かれた時だけ使う
- 生成した画像の試験では、床に置いた細い棒を見落とした。走行の安全の判断には使わない
"""

import re
import time
from typing import NamedTuple, Optional

import httpx
from langchain_core.messages import HumanMessage

from src.llm_factory import LLMFactory
from src.picarx_client.client import PicarxClient

# 画像処理部を持つ元のモデル。会話用のテキスト専用モデルとは別に読み込まれる
VLM_MODEL = "qwen3.5:4b"
# 両方を読み込むとメモリが増えるので、使い終わったら早めに解放する
VLM_KEEP_ALIVE = "2m"
MAX_DESCRIPTION_TOKENS = 120
# 初回はモデルの読み込みを含む
VLM_TIMEOUT_SEC = 60.0
CAMERA_TIMEOUT_SEC = 6.0

LOOK_PROMPT = (
    "これは PiCar-X（床を走る小型ロボットカー）の前方カメラの画像です。"
    "質問に、日本語の1〜2文で答えてください。"
    "画像に写っている物だけを答え、はっきりしない物は「よく分からない」と言ってください。\n"
    "質問: {question}"
)
# 今見えている物を聞く発言。過去の話は記憶で答えるので含めない
LOOK_QUESTION = re.compile(
    r"(何|なに)(が|か)(見え|写|映|ある|置)|見えるもの|見えてる|見えますか|"
    r"周り.{0,6}(様子|どう)|カメラ.{0,6}(見|映|写)"
)
PAST_WORDS = re.compile(r"さっき|昨日|きのう|前回|この前|先週|覚え")
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
    def __init__(self, client=None, llm=None, clock=time.monotonic):
        self._client = client
        self._llm = llm
        self._clock = clock

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

    async def describe(self, question: str) -> Look:
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
