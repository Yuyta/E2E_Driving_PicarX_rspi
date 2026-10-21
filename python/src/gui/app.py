import asyncio
import base64
import sys

from langchain_core.messages import AIMessage, HumanMessage
from PySide6.QtCore import Qt, QTimer
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

from src.config import ModelPreset, settings
from src.models import DriveDirection
from src.picarx_client.client import PicarxClient


class PicarxMainWindow(QMainWindow):
    """PiCar-X ローカルPC管理GUI（デスクトップアプリ）"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PiCar-X Agentic AI Controller")
        self.resize(950, 700)

        self.client = PicarxClient()
        self.current_model = settings.default_model
        self._agent_graph = None
        self._agent_model = None
        self._chat_messages = []
        self._camera_request_active = False

        self._init_ui()
        self._init_timer()

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
        btn_stop.clicked.connect(lambda: self._async_call(self.client.stop()))

        grid.addWidget(btn_forward, 0, 1)
        grid.addWidget(btn_left, 1, 0)
        grid.addWidget(btn_center, 1, 1)
        grid.addWidget(btn_right, 1, 2)
        grid.addWidget(btn_backward, 2, 1)
        grid.addWidget(btn_stop, 3, 0, 1, 3)
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
                self.lbl_distance.setText(f"{dist:.1f} cm")
                if dist < 15.0:
                    self.lbl_safety.setText("ALERT: Obstacle Close!")
                    self.lbl_safety.setStyleSheet("color: red; font-weight: bold;")
                else:
                    self.lbl_safety.setText("SAFE (Limiter Active)")
                    self.lbl_safety.setStyleSheet("color: green; font-weight: bold;")
            except Exception:
                self.lbl_distance.setText("Disconnected")

        self._async_call(task())

    def _on_snapshot_clicked(self):
        if self._camera_request_active:
            return
        self._camera_request_active = True

        async def task():
            try:
                data = await self.client.get_camera_snapshot(width=320, quality=70)
                image = QImage.fromData(base64.b64decode(data.image_base64), "JPEG")
                if image.isNull():
                    raise ValueError("Pi側から有効なJPEG画像が返されませんでした")
                self.camera_label.setPixmap(QPixmap.fromImage(image))
            except Exception as e:
                self.camera_label.setText(f"Camera Error: {e}")
            finally:
                self._camera_request_active = False

        self._async_call(task())

    def _on_send_chat(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self.chat_history.append(f"<b>User:</b> {text}")
        self.chat_history.append(f"<i>Thinking with {self.current_model.value}...</i>")

        async def task():
            try:
                speech_text = self._speech_text_from_command(text)
                if speech_text is not None:
                    await self.client.speak(speech_text)
                    self._remove_pending_user_message(text)
                    self.chat_history.append("<b>Assistant:</b> 発話しました。")
                    return

                from src.agents.graph import create_agent_graph

                if self._agent_graph is None or self._agent_model != self.current_model:
                    self._agent_graph = create_agent_graph(preset=self.current_model)
                    self._agent_model = self.current_model

                self._chat_messages.append(HumanMessage(content=text))
                result = await self._invoke_agent_with_recovery()
                messages = result.get("messages", [])
                response = next(
                    (
                        message.content
                        for message in reversed(messages)
                        if isinstance(message, AIMessage) and message.content
                    ),
                    "応答を取得できませんでした。",
                )
                response = self._extract_response_text(response)
                self._chat_messages.extend(messages[len(self._chat_messages) :])
                self.chat_history.append(f"<b>Assistant:</b> {response}")
                try:
                    await self.client.speak(response)
                except Exception as tts_error:
                    self.chat_history.append(f"<b>[TTS Error]</b> {tts_error}")
            except asyncio.TimeoutError:
                self._remove_pending_user_message(text)
                self.chat_history.append("<b>[Error]</b> AI応答がタイムアウトしました。")
            except Exception as exc:
                self._remove_pending_user_message(text)
                self.chat_history.append(f"<b>[Error]</b> {exc}")

        self._async_call(task())

    async def _invoke_agent_with_recovery(self):
        state = {
            "messages": self._chat_messages,
            "distance_cm": None,
            "grayscale": None,
            "snapshot_base64": None,
            "current_agent": "main",
            "is_finished": False,
        }
        config = {"recursion_limit": 20}

        try:
            return await asyncio.wait_for(
                self._agent_graph.ainvoke(state, config=config), timeout=60.0
            )
        except Exception as error:
            if not self._is_transient_model_error(error):
                raise

            self.chat_history.append("<i>[System] Geminiが混雑しています。再試行しています...</i>")
            await asyncio.sleep(3.0)
            try:
                return await asyncio.wait_for(
                    self._agent_graph.ainvoke(state, config=config), timeout=60.0
                )
            except Exception:
                fallback = ModelPreset.GEMINI_FLASH_LITE
                if self.current_model == fallback:
                    raise
                self.chat_history.append(
                    "<i>[System] Flash-Liteへ切り替えて再試行しています...</i>"
                )
                self._agent_graph = self._create_agent_graph(fallback)
                return await asyncio.wait_for(
                    self._agent_graph.ainvoke(state, config=config), timeout=60.0
                )

    @staticmethod
    def _is_transient_model_error(error):
        message = str(error).upper()
        return "503" in message or "UNAVAILABLE" in message or "高負荷" in message

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
        """非同期コルーチンをイベントループで実行"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
        else:
            loop.create_task(coro)


def main():
    app = QApplication(sys.argv)
    window = PicarxMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
