"""
E2E Self-Driving Data Collector GUI
Agentic AIのPicarxClientおよびGUIコンポーネントを共通基盤として再利用し、
カメラ映像を見ながらキーボード操縦・データセット記録を行う。
"""

import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

# OpenCV (保存時のフォーマット変換・リサイズ用)
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class CameraWorker(QThread):
    """PicarxClientを利用してカメラ画像とセンサー値をバックグラウンドで定期取得するワーカースレッド"""

    frame_received = Signal(bytes, float)

    def __init__(self, client: PicarxClient):
        super().__init__()
        self.client = client
        self.running = True

    def run(self):
        while self.running:
            img_bytes = None
            distance = 999.0

            # 1. センサー値取得 (PicarxClient再利用)
            try:
                distance = self.client.get_ultrasonic_sync()
            except Exception:
                pass

            # 2. カメラ画像取得 (PicarxClient再利用)
            try:
                snapshot = self.client.get_camera_snapshot_sync(width=320, quality=70)
                if snapshot and snapshot.image_base64:
                    img_bytes = base64.b64decode(snapshot.image_base64)
            except Exception:
                pass

            if img_bytes is not None and self.running:
                self.frame_received.emit(img_bytes, distance)

            time.sleep(0.08)  # 約12Hz

    def stop(self):
        self.running = False
        self.wait()


class E2ECollectorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PiCar-X E2E Data Collector")
        self.resize(850, 680)

        # 既存クライアントの再利用
        self.client = PicarxClient()

        self.is_recording = False
        self.frame_count = 0
        self.current_steering = 0  # -35 ~ +35 度
        self.current_throttle = 0.0  # 0.0 ~ 1.0
        self.max_steering_angle = 35.0
        self.drive_speed = 40

        # 保存先設定
        self.output_base = Path("data/raw")
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_base / self.session_id
        self.images_dir = self.session_dir / "images"
        self.labels_file = self.session_dir / "labels.jsonl"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._init_ui()

        # カメラワーカースレッド起動
        self.camera_worker = CameraWorker(self.client)
        self.camera_worker.frame_received.connect(self._on_frame_received)
        self.camera_worker.start()

    def _init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === 左側: カメラ映像 & 録画ステータス ===
        left_layout = QVBoxLayout()

        camera_group = QGroupBox("Camera Live View")
        cam_box = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Connecting to Camera...")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(True)
        self.camera_label.setMinimumSize(480, 360)
        self.camera_label.setStyleSheet(
            "background-color: #111; color: #888; border: 2px solid #444;"
        )
        cam_box.addWidget(self.camera_label)
        left_layout.addWidget(camera_group)

        # 録画コントロール
        rec_group = QGroupBox("Recording Status")
        rec_layout = QHBoxLayout(rec_group)
        self.rec_btn = QPushButton("● Start Recording (R)")
        style_rec = "background-color: #d9534f; color: white; font-weight: bold; padding: 10px;"
        self.rec_btn.setStyleSheet(style_rec)
        self.rec_btn.clicked.connect(self._toggle_recording)
        rec_layout.addWidget(self.rec_btn)

        self.lbl_rec_count = QLabel("Saved: 0 frames")
        self.lbl_rec_count.setStyleSheet("font-size: 14px; font-weight: bold;")
        rec_layout.addWidget(self.lbl_rec_count)
        left_layout.addWidget(rec_group)

        main_layout.addLayout(left_layout, stretch=3)

        # === 右側: テレメトリ & コントロール ===
        right_layout = QVBoxLayout()

        # テレメトリ
        telem_group = QGroupBox("Telemetry")
        telem_layout = QGridLayout(telem_group)

        telem_layout.addWidget(QLabel("Steering Angle:"), 0, 0)
        self.lbl_steer = QLabel("0° (0.00)")
        self.lbl_steer.setStyleSheet("font-weight: bold;")
        telem_layout.addWidget(self.lbl_steer, 0, 1)

        telem_layout.addWidget(QLabel("Speed (Throttle):"), 1, 0)
        self.lbl_speed = QLabel("0")
        self.lbl_speed.setStyleSheet("font-weight: bold;")
        telem_layout.addWidget(self.lbl_speed, 1, 1)

        telem_layout.addWidget(QLabel("Obstacle Dist:"), 2, 0)
        self.lbl_dist = QLabel("-- cm")
        telem_layout.addWidget(self.lbl_dist, 2, 1)
        right_layout.addWidget(telem_group)

        # 速度設定
        speed_group = QGroupBox("Driving Speed")
        speed_layout = QHBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Speed (0-100):"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(20, 100)
        self.spin_speed.setValue(40)
        self.spin_speed.valueChanged.connect(self._on_speed_changed)
        speed_layout.addWidget(self.spin_speed)
        right_layout.addWidget(speed_group)

        # 操作ガイド
        guide_group = QGroupBox("Keyboard Guide")
        guide_layout = QVBoxLayout(guide_group)
        guide_text = QLabel(
            "<b>W</b> : 前進 (Forward)<br>"
            "<b>S</b> : 停止 (Stop)<br>"
            "<b>A</b> : 左折 (Steer Left)<br>"
            "<b>D</b> : 右折 (Steer Right)<br>"
            "<b>X</b> : 直進に戻す (Center)<br>"
            "<b>R</b> : 録画トグル (REC Toggle)<br><br>"
            "<small style='color:#555;'>※ ウィンドウをフォーカスして操作してください。</small>"
        )
        guide_layout.addWidget(guide_text)
        right_layout.addWidget(guide_group)

        # 緊急停止ボタン
        stop_btn = QPushButton("🛑 EMERGENCY STOP")
        stop_btn.setStyleSheet(
            "background-color: #222; color: #ff5555; font-weight: bold; padding: 12px;"
        )
        stop_btn.clicked.connect(self._emergency_stop)
        right_layout.addWidget(stop_btn)

        main_layout.addLayout(right_layout, stretch=2)

    def _on_speed_changed(self, val):
        self.drive_speed = val

    def _toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.rec_btn.setText("⏸ Pause Recording (R)")
            self.rec_btn.setStyleSheet(
                "background-color: #f0ad4e; color: black; font-weight: bold; padding: 10px;"
            )
        else:
            self.rec_btn.setText("● Start Recording (R)")
            self.rec_btn.setStyleSheet(
                "background-color: #d9534f; color: white; font-weight: bold; padding: 10px;"
            )

    def _emergency_stop(self):
        self.current_throttle = 0.0
        self.current_steering = 0
        self.lbl_speed.setText("0")
        self.lbl_steer.setText("0° (0.00)")
        self.client.stop_sync()
        self.client.steer_sync(0)

    @Slot(bytes, float)
    def _on_frame_received(self, img_bytes: bytes, distance: float):
        if distance < 900.0:
            self.lbl_dist.setText(f"{distance:.1f} cm")

        qimg = QImage()
        if qimg.loadFromData(img_bytes, "JPEG"):
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

        if self.is_recording:
            self._save_sample(img_bytes)

    def _save_sample(self, img_bytes: bytes):
        filename = f"frame_{self.frame_count:06d}.jpg"
        filepath = self.images_dir / filename

        if cv2 is not None and np is not None:
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                resized = cv2.resize(img, (160, 120))
                cv2.imwrite(str(filepath), resized)
            else:
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
        else:
            with open(filepath, "wb") as f:
                f.write(img_bytes)

        steering_norm = self.current_steering / self.max_steering_angle
        record = {
            "index": self.frame_count,
            "timestamp": time.time(),
            "frame": filename,
            "steering": round(steering_norm, 4),
            "throttle": round(self.current_throttle, 4),
        }

        with open(self.labels_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.frame_count += 1
        self.lbl_rec_count.setText(f"Saved: {self.frame_count} frames")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        if key == Qt.Key_W:
            self.current_throttle = 0.5
            self.lbl_speed.setText(str(self.drive_speed))
            self.client.move_sync(DriveDirection.FORWARD, self.drive_speed, 600)

        elif key == Qt.Key_S:
            self.current_throttle = 0.0
            self.lbl_speed.setText("0")
            self.client.stop_sync()

        elif key == Qt.Key_A:
            self.current_steering = max(-35, self.current_steering - 10)
            norm = self.current_steering / self.max_steering_angle
            self.lbl_steer.setText(f"{self.current_steering}° ({norm:+.2f})")
            self.client.steer_sync(self.current_steering)

        elif key == Qt.Key_D:
            self.current_steering = min(35, self.current_steering + 10)
            norm = self.current_steering / self.max_steering_angle
            self.lbl_steer.setText(f"{self.current_steering}° ({norm:+.2f})")
            self.client.steer_sync(self.current_steering)

        elif key == Qt.Key_X:
            self.current_steering = 0
            self.lbl_steer.setText("0° (0.00)")
            self.client.steer_sync(0)

        elif key == Qt.Key_R:
            self._toggle_recording()

        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.camera_worker.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = E2ECollectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
