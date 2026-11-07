import asyncio
import base64
import importlib.util
import sqlite3
import sys
import threading
import time

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.agents.stop_intent import StopIntent, parse_stop_intent
from src.config import ModelPreset, settings
from src.gui.async_runner import AsyncRunner
from src.memory.events import EventRecorder
from src.memory.store import get_memory_store
from src.metrics import turn_timing
from src.metrics.turn_timing import ACK_AUDIO, REPLY_AUDIO, REPLY_READY, STOPPED, TurnTiming
from src.models import DriveDirection
from src.picarx_client.client import PicarxClient
from src.vision.local_vlm import ACKNOWLEDGEMENT, is_look_question, shared_vision
from src.voice.breath import preload as preload_breath
from src.voice.listener import VOICE_EXTRA_HINT, Recorder, Transcriber
from src.voice.speaker import Speaker

AGENT_TIMEOUT_SEC = 60.0
# ローカル LLM は最初の読み込みに時間がかかる
LOCAL_AGENT_TIMEOUT_SEC = 120.0


class PicarxMainWindow(QMainWindow):
    """PiCar-X ローカルPC管理GUI（デスクトップアプリ）"""

    # 背景スレッドからUIスレッドへ処理を渡すためのシグナル
    _ui_call = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PiCar-X Agentic AI Controller")
        self.resize(950, 700)

        self.client = PicarxClient()
        self._speaker = Speaker(self.client, on_error=self._on_speech_stop_failed)
        # 読点を足す処理の読み込みに約0.5秒かかるので、最初の返事の前に済ませておく
        threading.Thread(target=preload_breath, daemon=True).start()
        self._events = EventRecorder(self.client, lambda: get_memory_store())
        self._event_error = None
        self.current_model = settings.default_model
        self._agent_graph = None
        self._agent_model = None
        self._chat_messages = []
        self._camera_request_active = False
        self._emergency_active = False

        self._runner = AsyncRunner()
        self._ui_call.connect(self._run_on_ui, Qt.QueuedConnection)

        self._recorder = None
        self._transcriber = None
        # 次に始まる会話の応答時間。入力を受けた時に作り、Agent の処理へ渡す
        self._turn_timing = None

        self._init_ui()
        self._init_timer()
        self._init_voice()
        self._warm_up_vision()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左カラム: コントロール & センサー & カメラ
        left_layout = QVBoxLayout()

        # 1. モデル選択セクション (4種モデル切り替え)
        model_group = QGroupBox("LLM Model Selection (4 Presets)")
        model_layout = QHBoxLayout(model_group)
        model_layout.addWidget(QLabel("Active Model:"))

        self.model_combo = QComboBox()
        for preset in ModelPreset:
            is_default = " (Default)" if preset == settings.default_model else ""
            self.model_combo.addItem(f"{preset.value}{is_default}", preset)
        # 一覧の先頭ではなく、実際に使う既定のモデルを表示する
        self.model_combo.setCurrentIndex(self.model_combo.findData(self.current_model))

        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        left_layout.addWidget(model_group)

        # 2. カメラプレビューエリア
        camera_group = QGroupBox("Camera Stream / Snapshot")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Camera Connecting...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(True)
        self.camera_label.setStyleSheet("background-color: #222; color: #aaa; min-height: 240px;")
        camera_layout.addWidget(self.camera_label)

        snap_btn = QPushButton("Capture On-Demand Snapshot (Save Tokens)")
        snap_btn.clicked.connect(self._on_snapshot_clicked)
        camera_layout.addWidget(snap_btn)
        left_layout.addWidget(camera_group)

        # 3. センサー状態モニタ
        sensor_group = QGroupBox("Telemetry & Sensor Monitor")
        sensor_layout = QGridLayout(sensor_group)
        sensor_layout.addWidget(QLabel("Obstacle Distance:"), 0, 0)
        self.lbl_distance = QLabel("-- cm")
        sensor_layout.addWidget(self.lbl_distance, 0, 1)

        sensor_layout.addWidget(QLabel("Grayscale (L/M/R):"), 1, 0)
        self.lbl_grayscale = QLabel("[--, --, --]")
        sensor_layout.addWidget(self.lbl_grayscale, 1, 1)

        sensor_layout.addWidget(QLabel("Safety Status:"), 2, 0)
        self.lbl_safety = QLabel("SAFE (Limiter Active)")
        self.lbl_safety.setStyleSheet("color: green; font-weight: bold;")
        sensor_layout.addWidget(self.lbl_safety, 2, 1)
        left_layout.addWidget(sensor_group)

        # 4. 手動制御パネル (緊急停止含む)
        control_group = QGroupBox("Manual Vehicle Controls")
        grid = QGridLayout(control_group)

        btn_forward = QPushButton("▲ Forward")
        btn_forward.clicked.connect(
            lambda: self._async_call(self.client.move(DriveDirection.FORWARD, 30, 400))
        )
        btn_backward = QPushButton("▼ Backward")
        btn_backward.clicked.connect(
            lambda: self._async_call(self.client.move(DriveDirection.BACKWARD, 30, 400))
        )
        btn_left = QPushButton("◄ Left (Steer)")
        btn_left.clicked.connect(lambda: self._async_call(self.client.steer(-30)))
        btn_right = QPushButton("Right (Steer) ►")
        btn_right.clicked.connect(lambda: self._async_call(self.client.steer(30)))
        btn_center = QPushButton("● Center")
        btn_center.clicked.connect(lambda: self._async_call(self.client.steer(0)))

        btn_stop = QPushButton("🛑 EMERGENCY STOP")
        btn_stop.setStyleSheet(
            "background-color: #d9534f; color: white; font-weight: bold; padding: 10px;"
        )
        btn_stop.clicked.connect(self._on_emergency_stop)

        btn_reset = QPushButton("Reset Emergency Stop")
        btn_reset.clicked.connect(self._on_emergency_reset)

        btn_calibrate = QPushButton("Calibrate Floor")
        btn_calibrate.setToolTip("車体を床に置いた状態で、段差検知の基準を測ります")
        btn_calibrate.clicked.connect(self._on_calibrate_floor)

        grid.addWidget(btn_forward, 0, 1)
        grid.addWidget(btn_left, 1, 0)
        grid.addWidget(btn_center, 1, 1)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_backward, 2, 1)
        grid.addWidget(btn_stop, 3, 0, 1, 3)
        grid.addWidget(btn_reset, 4, 0, 1, 3)
        grid.addWidget(btn_calibrate, 5, 0, 1, 3)
        left_layout.addWidget(control_group)

        main_layout.addLayout(left_layout, 1)

        # 右カラム: チャット対話 & エージェントログ
        right_layout = QVBoxLayout()
        chat_group = QGroupBox("Agentic AI Chat & Autonomous Reasoning")
        chat_vbox = QVBoxLayout(chat_group)

        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setPlaceholderText(
            "Agentic AIの推論過程や対話ログがここに表示されます..."
        )
        chat_vbox.addWidget(self.chat_history)

        input_layout = QHBoxLayout()
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("指示を入力 (例: 前の障害物までの距離を測って)")
        self.chat_input.returnPressed.connect(self._on_send_chat)
        input_layout.addWidget(self.chat_input)

        btn_send = QPushButton("Send")
        btn_send.clicked.connect(self._on_send_chat)
        input_layout.addWidget(btn_send)

        # 車体にはマイクがないので、PC のマイクで話す
        self.btn_talk = QPushButton("🎤 押して話す")
        self.btn_talk.setToolTip("押している間だけ PC のマイクで録音します")
        self.btn_talk.pressed.connect(self._on_talk_pressed)
        self.btn_talk.released.connect(self._on_talk_released)
        input_layout.addWidget(self.btn_talk)
        chat_vbox.addLayout(input_layout)

        right_layout.addWidget(chat_group)
        main_layout.addLayout(right_layout, 1)

    def _init_timer(self):
        # センサー値の定期取得（2秒間隔でトークン・帯域節約）
        self.poll_timer = QTimer(self)
        self.poll_timer.timeout.connect(self._poll_telemetry)
        self.poll_timer.start(2000)

        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self._on_snapshot_clicked)
        self.camera_timer.start(1000)

    def _on_model_changed(self, index: int):
        preset = self._coerce_model_preset(self.model_combo.itemData(index))
        if preset:
            self.current_model = preset
            self._agent_graph = None
            self._agent_model = None
            self.chat_history.append(
                f"<b>[System]</b> モデルを <code>{preset.value}</code> に切り替えました。"
            )

    @staticmethod
    def _coerce_model_preset(value):
        if isinstance(value, ModelPreset):
            return value
        try:
            return ModelPreset(value)
        except (TypeError, ValueError):
            return None

    def _poll_telemetry(self):
        async def task():
            try:
                dist = await self.client.get_ultrasonic()
                self._ui(lambda: self._show_distance(dist))
            except Exception:
                self._ui(lambda: self.lbl_distance.setText("Disconnected"))
                return
            await self._record_events()

        self._async_call(task())

    async def _record_events(self):
        """安全イベントを記憶へ保存する。失敗は内容が変わった時だけ表示する。"""
        try:
            await self._events.poll()
        except (OSError, sqlite3.Error, httpx.HTTPError, ValueError) as error:
            message = str(error)
            if message != self._event_error:
                self._event_error = message
                self._append_chat(f"<b>[Memory Error]</b> 出来事を保存できません: {message}")
        else:
            self._event_error = None

    def _show_distance(self, dist: float) -> None:
        self.lbl_distance.setText(f"{dist:.1f} cm")
        if self._emergency_active:
            return
        if dist < 15.0:
            self.lbl_safety.setText("ALERT: Obstacle Close!")
            self.lbl_safety.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.lbl_safety.setText("SAFE (Limiter Active)")
            self.lbl_safety.setStyleSheet("color: green; font-weight: bold;")

    def _on_emergency_stop(self):
        # LLMの処理や共有イベントループの状態に左右されないよう、専用スレッドで送信する
        threading.Thread(target=self._send_emergency_stop, daemon=True).start()
        self._speaker.interrupt()

    def _on_speech_stop_failed(self, error):
        self._append_chat(f"<b>[TTS Error]</b> 発話を止められませんでした: {error}")

    def _send_emergency_stop(self):
        try:
            self.client.emergency_stop_sync()
        except Exception as exc:
            error = exc
            self._ui(lambda: self._append_chat(f"<b>[Error]</b> 緊急停止に失敗しました: {error}"))
        else:
            self._ui(self._show_emergency_active)

    def _show_emergency_active(self):
        self._emergency_active = True
        self.lbl_safety.setText("EMERGENCY STOP (Reset Required)")
        self.lbl_safety.setStyleSheet("color: red; font-weight: bold;")

    def _on_emergency_reset(self):
        async def task():
            try:
                await self.client.reset_emergency_stop()
            except Exception as exc:
                error = exc
                self._ui(lambda: self._append_chat(f"<b>[Error]</b> 解除に失敗しました: {error}"))
            else:
                self._ui(self._show_emergency_cleared)

        self._async_call(task())

    def _on_calibrate_floor(self):
        async def task():
            try:
                await self.client.calibrate_floor()
            except Exception as exc:
                error = exc
                self._ui(lambda: self._append_chat(f"<b>[Error]</b> 床の校正に失敗: {error}"))
            else:
                self._ui(lambda: self._append_chat("<b>[System]</b> 床を校正しました"))

        self._async_call(task())

    def _show_emergency_cleared(self):
        self._emergency_active = False
        self.lbl_safety.setText("SAFE (Limiter Active)")
        self.lbl_safety.setStyleSheet("color: green; font-weight: bold;")

    def _on_snapshot_clicked(self):
        if self._camera_request_active:
            return
        self._camera_request_active = True

        async def task():
            try:
                data = await self.client.get_camera_snapshot()
                image = QImage.fromData(base64.b64decode(data.image_base64), "JPEG")
                if image.isNull():
                    raise ValueError("Pi側から有効なJPEG画像が返されませんでした")
                self._ui(lambda: self.camera_label.setPixmap(QPixmap.fromImage(image)))
            except Exception as e:
                message = f"Camera Error: {e}"
                self._ui(lambda: self.camera_label.setText(message))
            finally:
                self._camera_request_active = False

        self._async_call(task())

    def _init_voice(self):
        """音声入力を準備する。必要なライブラリがなければボタンを無効にする"""
        if not all(importlib.util.find_spec(name) for name in ("sounddevice", "faster_whisper")):
            self.btn_talk.setEnabled(False)
            self.btn_talk.setToolTip(VOICE_EXTRA_HINT)
            return
        self._recorder = Recorder()
        self._transcriber = Transcriber()
        # モデルの読み込みに数秒かかるので、最初に話す前に済ませておく
        threading.Thread(target=self._preload_transcriber, daemon=True).start()

    def _preload_transcriber(self):
        try:
            self._transcriber.load()
        except Exception as error:
            message = str(error)
            self._append_chat(f"<b>[Voice Error]</b> 音声認識を準備できません: {message}")

    def _on_talk_pressed(self):
        if self._recorder is None:
            self._append_chat(f"<b>[Voice Error]</b> {VOICE_EXTRA_HINT}")
            return
        # 話し始めたら、読み上げ中の返事を止める
        self._speaker.interrupt()
        try:
            self._recorder.start()
        except Exception as error:
            self._append_chat(f"<b>[Voice Error]</b> マイクを使えません: {error}")
            return
        self.btn_talk.setText("🔴 録音中（離すと送信）")

    def _on_talk_released(self):
        if self._recorder is None or not self._recorder.recording:
            return
        self.btn_talk.setText("🎤 押して話す")
        released = time.monotonic()
        audio = self._recorder.stop()
        transcriber = self._transcriber

        async def task():
            try:
                result = await asyncio.to_thread(transcriber.transcribe, audio)
                stt_sec = round(time.monotonic() - released, 3)
            except Exception as error:
                message = str(error)
                self._append_chat(f"<b>[Voice Error]</b> 文字起こしに失敗しました: {message}")
                return
            if result.problem:
                self._append_chat(f"<i>[System] {result.problem}</i>")
                return
            timing = TurnTiming("voice", started=released, stt_sec=stt_sec)
            self._ui(lambda: self._submit_text(result.text, voice=True, timing=timing))

        self._async_call(task())

    def _on_send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self._submit_text(text)

    def _submit_text(self, text, voice=False, timing=None):
        """文字入力と音声入力の共通の入口"""
        self._turn_timing = timing or TurnTiming("voice" if voice else "text")
        label = "User (🎤)" if voice else "User"
        self.chat_history.append(f"<b>{label}:</b> {text}")
        # 話しかけられたら、前の返事の読み上げを止める
        self._speaker.interrupt()

        intent = parse_stop_intent(text)
        if intent != StopIntent.NONE:
            # 共有のイベントループに左右されないよう、専用スレッドで送る
            threading.Thread(target=self._send_fast_stop, args=(text, intent), daemon=True).start()
            return
        self._start_agent_turn(text)

    def _take_turn_timing(self):
        timing, self._turn_timing = self._turn_timing, None
        return timing or TurnTiming("text")

    def _finish_turn_timing(self, timing, ok=True):
        """応答時間を保存して表示する。保存に失敗しても会話は止めない"""
        timing.ok = ok
        try:
            turn_timing.save(timing)
        except OSError as error:
            self._append_chat(f"<b>[Timing Error]</b> 応答時間を保存できません: {error}")
        if ok and timing.summary():
            self._append_chat(f"<i>[Time] {timing.summary()}</i>")

    def _send_fast_stop(self, text, intent):
        """LLM を通さず停止する。失敗や続きの指示があれば LLM へ渡す。"""
        timing = self._turn_timing
        try:
            stopped = self.client.stop_sync()
            if stopped and timing:
                timing.mark(STOPPED)
            failure = None if stopped else "Pi が停止を受け付けませんでした"
        except Exception as exc:
            stopped = False
            failure = str(exc)

        if stopped and intent == StopIntent.ONLY_STOP:
            self._ui(lambda: self._append_chat("<b>Assistant:</b> 停止しました。"))
            self._remember_turn(text, "停止しました。")
            timing = self._take_turn_timing()
            timing.kind = "stop"

            async def speak_and_record():
                await self._say("停止しました。", timing, REPLY_AUDIO)
                self._finish_turn_timing(timing)

            self._async_call(speak_and_record())
            return
        if stopped:
            self._ui(lambda: self._append_chat("<i>[System] 停止しました。続きを処理します...</i>"))
        else:
            self._ui(
                lambda: self._append_chat(
                    f"<b>[Error]</b> 停止に失敗しました: {failure}。LLM からも試します"
                )
            )
        self._ui(lambda: self._start_agent_turn(text))

    def _start_agent_turn(self, text):
        timing = self._take_turn_timing()
        self.chat_history.append(f"<i>Thinking with {self.current_model.value}...</i>")
        if is_look_question(text):
            timing.kind = "look"
            # 画像の認識に約10秒かかるので、先に返事をしておく
            self.chat_history.append(f"<b>Assistant:</b> {ACKNOWLEDGEMENT}")
            self._async_call(self._say(ACKNOWLEDGEMENT, timing, ACK_AUDIO))
            # Gemini が look を使うと決める前に見始める
            self._async_call(self._prefetch_look(text))

        async def task():
            try:
                speech_text = self._speech_text_from_command(text)
                if speech_text is not None:
                    timing.kind = "speech_command"
                    self._remove_pending_user_message(text)
                    self._append_chat("<b>Assistant:</b> 発話します。")
                    await self._say(speech_text, timing, REPLY_AUDIO)
                    self._finish_turn_timing(timing)
                    return

                from src.agents.graph import create_agent_graph

                if self._agent_graph is None or self._agent_model != self.current_model:
                    self._agent_graph = create_agent_graph(preset=self.current_model)
                    self._agent_model = self.current_model

                self._chat_messages.append(HumanMessage(content=text))
                result = await self._invoke_agent_with_recovery()
                timing.mark(REPLY_READY)
                # 切り替えた時は、実際に答えたモデルを記録する
                timing.model = self._agent_model.value if self._agent_model else None
                messages = result.get("messages", [])
                response, speech = self._reply_from_messages(messages)
                self._chat_messages.extend(messages[len(self._chat_messages) :])
                self._remember_turn(text, speech)
                self._append_chat(f"<b>Assistant:</b> {response}")
                await self._say(speech, timing, REPLY_AUDIO)
                self._finish_turn_timing(timing)
            except asyncio.TimeoutError:
                self._remove_pending_user_message(text)
                self._append_chat("<b>[Error]</b> AI応答がタイムアウトしました。")
                self._finish_turn_timing(timing, ok=False)
            except Exception as exc:
                self._remove_pending_user_message(text)
                self._append_chat(f"<b>[Error]</b> {exc}")
                self._finish_turn_timing(timing, ok=False)

        self._async_call(task())

    def _warm_up_vision(self):
        """見る質問の VLM は読み込みに約5.6秒かかるので、起動時に読み込んでおく"""
        self._async_call(shared_vision().warm_up())

    @staticmethod
    async def _prefetch_look(text):
        shared_vision().prefetch(text)

    async def _say(self, text, timing=None, mark=None):
        """発話はすべて Speaker を通す。読み上げに失敗しても会話は続ける。"""
        on_first = (lambda: timing.mark(mark)) if timing and mark else None
        try:
            await self._speaker.say(text, on_first_sentence=on_first)
        except Exception as error:
            self._append_chat(f"<b>[TTS Error]</b> {error}")

    @classmethod
    def _reply_from_messages(cls, messages):
        """画面に出す文章と、読み上げて記憶する文章を返す"""
        message = next(
            (m for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
            None,
        )
        if message is None:
            fallback = "応答を取得できませんでした。"
            return fallback, fallback
        response = cls._extract_response_text(message.content)
        return response, message.additional_kwargs.get("speech") or response

    def _remember_turn(self, user_text, response):
        """会話を保存する。保存に失敗しても応答は止めない。"""
        try:
            memory = get_memory_store()
            memory.add_message("user", user_text)
            memory.add_message("assistant", response)
        except (OSError, sqlite3.Error) as error:
            self._append_chat(f"<b>[Memory Error]</b> {error}")

    async def _invoke_agent_with_recovery(self):
        state = {"messages": self._chat_messages}
        config = {"recursion_limit": 20}

        async def invoke(timeout=AGENT_TIMEOUT_SEC):
            return await asyncio.wait_for(
                self._agent_graph.ainvoke(state, config=config), timeout=timeout
            )

        try:
            return await invoke()
        except Exception as error:
            if self.current_model == ModelPreset.OLLAMA_LOCAL or not self._is_model_unavailable(
                error
            ):
                raise
            first_error = error

        # 混雑、応答なし、上限超過は、Flash-Lite なら答えられることがある。通信断は直らない
        if self._should_try_flash_lite(first_error):
            self._append_chat(
                "<i>[System] Geminiが混雑中か上限に達しています。Flash-Liteで応答します...</i>"
            )
            self._agent_graph = self._create_agent_graph(ModelPreset.GEMINI_FLASH_LITE)
            try:
                return await invoke()
            except Exception as error:
                if not self._is_model_unavailable(error):
                    raise

        self._append_chat("<i>[System] Geminiが使えないため、ローカルLLMで応答します...</i>")
        self._agent_graph = self._create_agent_graph(ModelPreset.OLLAMA_LOCAL)
        return await invoke(LOCAL_AGENT_TIMEOUT_SEC)

    @staticmethod
    def _is_model_busy(error):
        message = str(error).upper()
        return "503" in message or "UNAVAILABLE" in message or "高負荷" in message

    def _should_try_flash_lite(self, error):
        """Flash が使えない時も Flash-Lite は答えられた（2026-11-06）。

        混雑と応答なしのほか、無料枠の上限も対象にする。
        上限はモデルごとで、Flash は1日20回しかない。
        """
        if self.current_model == ModelPreset.GEMINI_FLASH_LITE:
            return False
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
            return True
        message = f"{type(error).__name__} {error}".upper()
        return self._is_model_busy(error) or any(
            word in message for word in ("TIMEOUT", "TIMED OUT", "429", "RESOURCE_EXHAUSTED")
        )

    @classmethod
    def _is_model_unavailable(cls, error):
        """別のモデルへ切り替えれば解決しそうなエラーかどうか"""
        if isinstance(error, (asyncio.TimeoutError, TimeoutError, ConnectionError, OSError)):
            return True
        message = f"{type(error).__name__} {error}".upper()
        return cls._is_model_busy(error) or any(
            word in message
            for word in (
                "429",
                "RESOURCE_EXHAUSTED",
                "QUOTA",
                "CONNECT",
                "TIMEOUT",
                "NAME RESOLUTION",
                "GETADDRINFO",
                "API_KEY",
                "API KEY",
            )
        )

    def _create_agent_graph(self, preset):
        from src.agents.graph import create_agent_graph

        self._agent_model = preset
        return create_agent_graph(preset=preset)

    def _remove_pending_user_message(self, text):
        if self._chat_messages and isinstance(self._chat_messages[-1], HumanMessage):
            if self._chat_messages[-1].content == text:
                self._chat_messages.pop()

    @staticmethod
    def _speech_text_from_command(command):
        normalized = command.strip()
        speech_prefixes = ("喋って", "しゃべって", "話して", "発話して")
        if not normalized.startswith(speech_prefixes):
            return None

        speech_text = normalized
        for prefix in speech_prefixes:
            if speech_text.startswith(prefix):
                speech_text = speech_text[len(prefix) :].strip()
                break

        if speech_text.startswith(("「", "『")) and speech_text.endswith(("」", "』")):
            speech_text = speech_text[1:-1].strip()
        return speech_text or "こんにちは。PiCar-Xです。"

    @staticmethod
    def _extract_response_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(part for part in text_parts if part) or "応答を取得できませんでした。"
        return str(content)

    def _async_call(self, coro):
        """コルーチンを背景のイベントループで実行し、すぐに戻る"""
        return self._runner.submit(coro)

    def _ui(self, callback) -> None:
        """背景スレッドからUIスレッドで処理を実行する"""
        self._ui_call.emit(callback)

    @staticmethod
    def _run_on_ui(callback) -> None:
        callback()

    def _append_chat(self, html: str) -> None:
        self._ui(lambda: self.chat_history.append(html))

    def closeEvent(self, event):
        if self._recorder is not None and self._recorder.recording:
            self._recorder.stop()
        self.poll_timer.stop()
        self.camera_timer.stop()
        self._runner.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = PicarxMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
