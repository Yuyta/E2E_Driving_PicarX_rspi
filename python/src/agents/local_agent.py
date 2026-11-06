"""ローカル LLM 用の Agent。CPU で動かすので、LLM に読ませる量と呼ぶ回数を減らす。

実測では応答時間の 93〜98% がプロンプトの処理だった。qwen3.5 は途中まで巻き戻せないので、
プロンプトの途中が変わると 500 トークン以上を読み直す。そこで次のようにする。

- 確実に答えられる車体の質問は、LLM を呼ばない
- 固定の指示を先頭に置き、履歴は追記だけにして、読み直しを最新の発言に限る
- ツール定義の代わりに、1つの JSON で行動を受け取る
- 実行後の返事は定型文にして、LLM を2回呼ばない
- 実行の可否はコードで検証する
"""

import asyncio
import json
import re
import sqlite3
from typing import Dict, List, NamedTuple, Optional

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from src.agents.body_context import REASON_TEXT, answer_state_question
from src.llm_factory import LLMFactory
from src.memory.store import get_memory_store
from src.tools.drive_tool import drive_backward, drive_forward, steer_wheels, stop_moving
from src.vision.local_vlm import LocalVision, is_look_question

MOTION_SPEED = 50  # 実機では速度50前後でないと動きが弱い
SHORT_MS = 500
NORMAL_MS = 1000
STEER_ANGLE = 30
MAX_REPLY_TOKENS = 128
# 履歴を毎ターン削ると先頭がずれて読み直しになるので、溜まってからまとめて削る
HISTORY_MAX_MESSAGES = 24
HISTORY_KEEP_MESSAGES = 8
MEMORY_PREVIOUS_LIMIT = 4
MEMORY_RELATED_LIMIT = 2
MEMORY_EVENT_LIMIT = 3

ACTIONS = (
    "reply",
    "forward",
    "backward",
    "left",
    "right",
    "center",
    "stop",
    "remember",
    "forget",
    "look",
)
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": list(ACTIONS)},
        "text": {"type": "string"},
        "key": {"type": "string"},
        "value": {"type": "string"},
    },
    "required": ["action"],
}
LOCAL_PROMPT = """あなたは小型ロボットカー PiCar-X です。発言を読み、JSON を1つだけ出力します。
action の意味:
- forward: 前進する
- backward: 後退する
- left, right, center: 舵を左、右、中央にする
- stop: 止まる。「待って」「やめて」「動かないで」も含む
- remember: 事実を覚える。key に短い見出し、value に内容を入れる
- forget: 事実を忘れる。key に見出しを入れる
- look: カメラで前を見て、見える物を答える
- reply: 上のどれでもない会話。text に短い日本語で答える
動かすのは、今動くように頼まれた時だけです。
電池やセンサーの値は分かりません。聞かれたら分からないと答えます。
参考情報の記憶があれば、過去のことはそれだけを使って答えます。

例:
少し前に進んで → {"action":"forward"}
バックして → {"action":"backward"}
右に曲がって → {"action":"right"}
危ないからやめて → {"action":"stop"}
前に何がある？ → {"action":"look"}
私の名前はたろうです。覚えておいて → {"action":"remember","key":"ユーザーの名前","value":"たろう"}
前回の旅行は楽しかったね → {"action":"reply","text":"楽しかったですね。"}
「前に進んで」は英語で何て言う？ → {"action":"reply","text":"Go forward です。"}"""

# 否定、過去の話、引用は、今動いてほしいという指示ではない
NOT_A_REQUEST = re.compile(
    r"ないで|なくて|するな|進むな|動くな|禁止|やめ|昨日|きのう|前回|先週|「|英語"
)
MOTION_WORDS = {
    "forward": re.compile(
        r"前に|前へ|前進|進ん|進め|進む|すすん|すすめ|まっすぐ|真っすぐ|直進|行って|走"
    ),
    "backward": re.compile(r"後ろ|うしろ|後退|バック|下が|さが|戻|もど"),
    "left": re.compile(r"左|ひだり"),
    "right": re.compile(r"右|みぎ"),
    "center": re.compile(r"まっすぐ|真っすぐ|直進|中央|正面|センター|戻|もど"),
}
SHORT_MOTION = re.compile(r"少し|すこし|ちょっと|ちょい")
MEMORY_QUESTION = re.compile(
    r"前回|この前|このまえ|さっき|昨日|きのう|おととい|先週|何の話|なんの話|話した|っけ|"
    r"覚えてる|覚えている|どうだった|名前は(何|なに|[?？])|名前を?(知|教|分か|わか)"
)
# 車体の出来事を聞く発言。読む量を減らすため、この時だけ出来事を渡す
EVENT_QUESTION = re.compile(r"止まった|とまった|止まっちゃ|ぶつか|何があった|何かあった|緊急停止")
STEER_ONLY = re.compile(r"舵|ハンドル|タイヤ")
TURN_WORDS = re.compile(r"曲が|まが|カーブ")
ASK_TURN_SIDE = "右と左のどちらに曲がりますか？"
REFUSED_MOTION = "動かす指示かどうか分からなかったので、動かしていません。"
NOT_UNDERSTOOD = "うまく理解できませんでした。もう一度お願いします。"
DRIVE_REPLIES = {"forward": "前に進みました。", "backward": "後ろに下がりました。"}
DRIVE_LABELS = {"forward": "前進", "backward": "後退"}
STEER_REPLIES = {
    "left": "舵を左に切りました。",
    "right": "舵を右に切りました。",
    "center": "舵を中央に戻しました。",
}
SIDE_TEXT = {"left": "左", "right": "右"}
SUCCESS_WORDS = ("成功", "完了")
DEFAULT_TOOLS = {
    "drive_forward": drive_forward,
    "drive_backward": drive_backward,
    "steer_wheels": steer_wheels,
    "stop_moving": stop_moving,
}


class Reply(NamedTuple):
    """画面に出す文章と、読み上げて記憶する文章。エラーの詳細は読み上げない。"""

    text: str
    speech: Optional[str] = None


class Motion(NamedTuple):
    drive: Optional[str] = None  # forward または backward
    steer: Optional[str] = None  # left、right または center


class LocalAction(NamedTuple):
    action: str
    text: str = ""
    key: str = ""
    value: str = ""


def parse_action(content: object) -> Optional[LocalAction]:
    """LLM の出力を行動に変換する。解釈できなければ None を返す。"""
    try:
        data = json.loads(str(content))
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("action") not in ACTIONS:
        return None
    return LocalAction(
        action=data["action"],
        text=str(data.get("text") or "").strip(),
        key=str(data.get("key") or "").strip(),
        value=str(data.get("value") or "").strip(),
    )


def steer_side(text: str) -> Optional[str]:
    """発言が示す左右を返す。両方ある時や、どちらもない時は None を返す。"""
    left = bool(MOTION_WORDS["left"].search(text))
    right = bool(MOTION_WORDS["right"].search(text))
    if left == right:
        return None
    return "left" if left else "right"


def plan_motion(action: str, text: str) -> Optional[Motion]:
    """LLM の行動と発言から動きを決める。

    「前に進みながら右に曲がって」を LLM が forward と right のどちらで返しても、
    同じ動きになるようにする。発言に裏付けがなければ None を返す。
    """
    if NOT_A_REQUEST.search(text):
        return None
    if action in ("forward", "backward"):
        if not MOTION_WORDS[action].search(text):
            return None
        return Motion(drive=action, steer=steer_side(text))
    if not MOTION_WORDS[action].search(text):
        return None
    if action == "center" or STEER_ONLY.search(text):
        return Motion(steer=action)
    for drive in ("forward", "backward"):
        if MOTION_WORDS[drive].search(text):
            return Motion(drive=drive, steer=action)
    return Motion(steer=action)


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    parts = (part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return "".join(parts)


def conversation_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """発言と返事だけを残す。Gemini から切り替えた時のツール履歴は読ませない。"""
    result: List[BaseMessage] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            result.append(message)
        elif isinstance(message, AIMessage) and message_text(message) and not message.tool_calls:
            # エラーの詳細は次の判断に使わない
            speech = message.additional_kwargs.get("speech")
            result.append(AIMessage(content=speech or message_text(message)))
    return result


class LocalAgent:
    """`create_agent_graph` の戻り値と同じく `ainvoke` で呼べるローカル用の Agent"""

    def __init__(
        self,
        llm=None,
        status_client=None,
        memory=None,
        tools: Optional[Dict] = None,
        sleep=asyncio.sleep,
        vision=None,
    ):
        self._llm = llm or LLMFactory.create_local(
            temperature=0.0, format=ACTION_SCHEMA, num_predict=MAX_REPLY_TOKENS
        )
        self._status_client = status_client
        self._memory = memory
        self._tools = tools or DEFAULT_TOOLS
        self._sleep = sleep
        self._vision = vision
        self._history_start = 0
        # 記憶を渡したターンは、次のターンも同じ位置に同じ文章を置いて読み直しを防ぐ
        self._contexts: Dict[int, str] = {}

    async def ainvoke(self, state: dict, config: Optional[dict] = None) -> dict:
        messages = list(state["messages"])
        latest = next(m for m in reversed(messages) if isinstance(m, HumanMessage))
        reply = await self._respond(messages, latest)
        if isinstance(reply, str):
            reply = Reply(reply)
        extra = {"speech": reply.speech} if reply.speech else {}
        return {"messages": messages + [AIMessage(content=reply.text, additional_kwargs=extra)]}

    async def _respond(self, messages: List[BaseMessage], latest: HumanMessage) -> Reply | str:
        text = message_text(latest)
        # 「周りの様子どう？」は車体の状態ではなく、見える物を聞いている
        if is_look_question(text):
            return await self._look(text)
        answer = await answer_state_question(text, self._status_client)
        if answer is not None:
            return answer

        self._contexts.pop(id(latest), None)
        context = self._memory_context(text)
        if context:
            self._contexts[id(latest)] = context
        response = await self._llm.ainvoke(self._build_prompt(messages))
        action = parse_action(message_text(response))
        if action is None:
            return NOT_UNDERSTOOD
        return await self._execute(action, text)

    def _build_prompt(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        history = conversation_messages(messages)
        if self._history_start >= len(history):
            self._history_start = 0
        if len(history) - self._history_start > HISTORY_MAX_MESSAGES:
            start = len(history) - HISTORY_KEEP_MESSAGES
            while not isinstance(history[start], HumanMessage):
                start += 1
            self._history_start = start

        prompt: List[BaseMessage] = [SystemMessage(content=LOCAL_PROMPT)]
        for message in history[self._history_start :]:
            context = self._contexts.get(id(message))
            if context:
                prompt.append(SystemMessage(content=f"参考情報\n{context}"))
            prompt.append(message)
        return prompt

    def _memory_store(self):
        return self._memory or get_memory_store()

    def _memory_context(self, text: str) -> str:
        """過去や事実に関わる発言の時だけ記憶を渡す。記憶が使えなくても会話は止めない。"""
        try:
            memory = self._memory_store()
            mentions_fact = any(
                word and word in text for pair in memory.facts().items() for word in pair
            )
            asks_event = bool(EVENT_QUESTION.search(text))
            if not (MEMORY_QUESTION.search(text) or mentions_fact or asks_event):
                return ""
            return memory.build_context(
                text,
                previous_limit=MEMORY_PREVIOUS_LIMIT,
                related_limit=MEMORY_RELATED_LIMIT,
                event_limit=MEMORY_EVENT_LIMIT if asks_event else 0,
            )
        except (OSError, sqlite3.Error):
            return ""

    async def _execute(self, action: LocalAction, text: str) -> Reply | str:
        name = action.action
        if name == "reply":
            return action.text or NOT_UNDERSTOOD
        if name == "remember":
            return self._remember(action, text)
        if name == "forget":
            return self._forget(action)
        if name == "stop":
            return await self._stop()
        if name == "look":
            return await self._look(text)
        motion = plan_motion(name, text)
        if motion is None:
            return REFUSED_MOTION
        if motion.drive and not motion.steer and TURN_WORDS.search(text):
            return ASK_TURN_SIDE
        return await self._run_motion(motion, text)

    async def _look(self, text: str) -> Reply | str:
        if self._vision is None:
            self._vision = LocalVision()
        result = await self._vision.describe(text)
        if result.ok:
            return result.text
        return Reply(f"{result.text}（詳細: {result.detail}）", result.text)

    async def _stop(self) -> Reply | str:
        result = str(await self._tools["stop_moving"].ainvoke({}))
        if self._succeeded(result):
            return "停止しました。"
        return await self._failure("停止", result)

    async def _run_motion(self, motion: Motion, text: str) -> Reply | str:
        if motion.steer:
            result = await self._steer(motion.steer)
            if not self._succeeded(result):
                return await self._failure("舵の操作", result)
        if not motion.drive:
            return STEER_REPLIES[motion.steer]

        duration_ms = SHORT_MS if SHORT_MOTION.search(text) else NORMAL_MS
        tool = "drive_forward" if motion.drive == "forward" else "drive_backward"
        result = str(
            await self._tools[tool].ainvoke({"speed": MOTION_SPEED, "duration_ms": duration_ms})
        )
        turning = motion.steer in ("left", "right")
        if not self._succeeded(result):
            if turning:
                await self._steer("center")
            return await self._failure(DRIVE_LABELS[motion.drive], result, motion.drive)
        if not turning:
            return DRIVE_REPLIES[motion.drive]
        # 次の走行が曲がらないよう、走り終わったら舵を戻す
        await self._sleep(duration_ms / 1000)
        await self._steer("center")
        return f"{SIDE_TEXT[motion.steer]}に曲がりながら{DRIVE_REPLIES[motion.drive]}"

    async def _steer(self, side: str) -> str:
        angle = {"left": -STEER_ANGLE, "right": STEER_ANGLE, "center": 0}[side]
        return str(await self._tools["steer_wheels"].ainvoke({"angle": angle}))

    @staticmethod
    def _succeeded(result: str) -> bool:
        return any(word in result for word in SUCCESS_WORDS)

    async def _failure(self, label: str, detail: str, drive: Optional[str] = None) -> Reply:
        speech = f"{label}できませんでした。"
        reason = await self._refusal_reason(drive) if drive else None
        if reason:
            speech += f"{reason}。"
        return Reply(f"{speech}（詳細: {detail}）", speech)

    async def _refusal_reason(self, drive: str) -> Optional[str]:
        """Pi が走行を拒否した理由を返す。分からなければ None を返す。"""
        if self._status_client is None:
            return None
        try:
            status = await self._status_client.get_status()
        except (httpx.HTTPError, ValueError):
            return None
        reason = getattr(status.blocked, drive)
        return REASON_TEXT.get(reason.value, reason.value) if reason else None

    def _remember(self, action: LocalAction, text: str) -> Reply | str:
        # 「たろう = 覚えておいて」のような誤りを保存しない
        if not action.key or not action.value or action.value not in text:
            return NOT_UNDERSTOOD
        if "覚え" in action.value or "記憶" in action.value:
            return NOT_UNDERSTOOD
        try:
            self._memory_store().remember_fact(action.key, action.value)
        except (OSError, sqlite3.Error) as error:
            return Reply(f"記憶できませんでした（詳細: {error}）", "記憶できませんでした。")
        return f"{action.key}は{action.value}、と覚えました。"

    def _forget(self, action: LocalAction) -> Reply | str:
        try:
            forgotten = self._memory_store().forget_fact(action.key)
        except (OSError, sqlite3.Error) as error:
            return Reply(
                f"忘れることができませんでした（詳細: {error}）", "忘れることができませんでした。"
            )
        return f"{action.key}を忘れました。" if forgotten else f"{action.key}の記憶はありません。"
