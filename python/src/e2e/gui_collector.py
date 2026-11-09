"""
E2E Self-Driving Data Collector GUI
Agentic AIのPicarxClientおよびGUIコンポーネントを共通基盤として再利用し、
カメラ映像を見ながらキーボード操縦・データセット記録を行う。
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QEvent, Qt, QThread, QTimer, Signal, Slot
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

from src.e2e.motion.monitor import (
    CommandState,
    MonitorUpdate,
    MotionMonitor,
    describe,
    record_motion_event,
)
from src.models import DRIVE_SPEED, MIN_MOVING_SPEED, DriveDirection, RobotStatus
from src.picarx_client.client import PicarxClient

# W キーで送る走行指令の期限。キーを押し続けると繰り返し送られる
MOVE_TTL_MS = 600
# W を押している間、この間隔で走行指令を送り直す。キーの自動の連打（オートリピート）は、
# 走りながら A や D を押すと OS が止めてしまうので使わない
# （2026-11-09 の収集では、そのせいで走行中のフレームが16%だった）
DRIVE_RESEND_MS = 100
# A/D を押している間は舵を一定の速さで切り、離すと同じ速さで中央へ戻す。
# 押し続けると一瞬で最大角になる操作では、舵角が最大か0に偏った（2026-11-09）
MAX_STEER_DEG = 35
STEER_RATE_DPS = 60.0
STEER_TICK_MS = 50
# 配信のフレームを待つ時間。届かなければ待ち直す
FRAME_TIMEOUT_SEC = 1.0

# OpenCV (保存時のフォーマット変換・リサイズ用)
try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


def sensor_fields(status: Optional[RobotStatus]) -> dict:
    """1フレーム分のセンサー値。状態が読めなかった時は値を None にする"""
    if status is None:
        return {"distance_cm": None, "distance_far": None, "floor": None, "battery_v": None}
    sensors = status.sensors
    distance = sensors.distance
    cm = None if distance.far or distance.cm is None else round(distance.cm, 1)
    return {
        # 何も捉えていない時（反射が戻らず遠い時を含む）は None
        "distance_cm": cm,
        "distance_far": distance.far,
        "floor": sensors.floor.values,
        "battery_v": sensors.battery.voltage,
    }


class CameraWorker(QThread):
    """カメラの常時配信（MJPEG、約15fps）を受け、フレームごとに Pi の状態を1回読む。

    以前は静止画と超音波を毎回別々に取りに行っていて、10fps に届かなかった（M3）。
    """

    frame_received = Signal(bytes, int, object)  # JPEG、配信の通し番号、RobotStatus か None

    def __init__(self, client: PicarxClient):
        super().__init__()
        self.client = client
        self.running = True

    def run(self):
        stream = self.client.open_camera_stream().start()
        last_seq = 0
        try:
            while self.running:
                frame = stream.wait_for_frame(last_seq, timeout=FRAME_TIMEOUT_SEC)
                if frame is None:
                    continue
                last_seq = frame.seq
                try:
                    status = self.client.get_status_sync()
                except Exception:
                    status = None
                if self.running:
                    self.frame_received.emit(frame.jpeg, frame.seq, status)
        finally:
            stream.stop()

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
        self._frame_times: list = []
        self.current_steering = 0  # -35 ~ +35 度
        self.current_throttle = 0.0  # 0.0 ~ 1.0
        self.max_steering_angle = 35.0
        self.drive_speed = DRIVE_SPEED

        # 指令と実際の動きの食い違いの検出（ROADMAP の M3）
        self.command_state = CommandState(time.time())
        self.motion_monitor = MotionMonitor.from_config(on_sustained=self._on_motion_mismatch)

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

        self._drive_held = False
        self.drive_timer = QTimer(self)
        self.drive_timer.setInterval(DRIVE_RESEND_MS)
        self.drive_timer.timeout.connect(self._send_drive)

        self._steer_left = False
        self._steer_right = False
        self._steer_value = 0.0
        self.steer_timer = QTimer(self)
        self.steer_timer.setInterval(STEER_TICK_MS)
        self.steer_timer.timeout.connect(self._update_steer)
        self.steer_timer.start()

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

        telem_layout.addWidget(QLabel("Motion Check:"), 3, 0)
        self.lbl_motion = QLabel("--")
        self.lbl_motion.setWordWrap(True)
        telem_layout.addWidget(self.lbl_motion, 3, 1)
        right_layout.addWidget(telem_group)

        # 速度設定
        speed_group = QGroupBox("Driving Speed")
        speed_layout = QHBoxLayout(speed_group)
        speed_layout.addWidget(QLabel("Speed (0-100):"))
        self.spin_speed = QSpinBox()
        self.spin_speed.setRange(MIN_MOVING_SPEED, 100)
        self.spin_speed.setValue(DRIVE_SPEED)
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

        reset_btn = QPushButton("Reset Emergency Stop")
        reset_btn.clicked.connect(self._reset_emergency_stop)
        right_layout.addWidget(reset_btn)

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

    def _send_drive(self) -> None:
        """W を押している間、前進の指令を送り続ける。Pi が拒否したら送るのをやめる"""
        if not self._drive_held:
            return
        self.current_throttle = 0.5
        self.lbl_speed.setText(str(self.drive_speed))
        try:
            accepted = self.client.move_sync(DriveDirection.FORWARD, self.drive_speed, MOVE_TTL_MS)
        except Exception as exc:
            self.lbl_dist.setText(f"MOVE FAILED: {exc}")
            accepted = False
        if accepted:
            self.command_state.move(self.drive_speed, MOVE_TTL_MS / 1000.0, time.time())
        else:
            self._release_drive(send_stop=False)

    def _release_drive(self, send_stop: bool = True) -> None:
        """前進をやめる。W を離した時、S、食い違い、ウィンドウから操作が移った時に呼ぶ"""
        self._drive_held = False
        self.drive_timer.stop()
        self.current_throttle = 0.0
        self.lbl_speed.setText("0")
        self.command_state.stop(time.time())
        if send_stop:
            try:
                self.client.stop_sync()
            except Exception as exc:
                self.lbl_dist.setText(f"STOP FAILED: {exc}")

    def _update_steer(self) -> None:
        """押しているキーの向きへ舵を近づける。どちらも押していなければ中央へ戻す"""
        if self._steer_left and not self._steer_right:
            target = -MAX_STEER_DEG
        elif self._steer_right and not self._steer_left:
            target = MAX_STEER_DEG
        else:
            target = 0
        step = STEER_RATE_DPS * STEER_TICK_MS / 1000.0
        if self._steer_value < target:
            self._steer_value = min(target, self._steer_value + step)
        else:
            self._steer_value = max(target, self._steer_value - step)
        angle = round(self._steer_value)
        if angle != self.current_steering:
            self._apply_steer(angle)

    def _apply_steer(self, angle: int) -> None:
        self.current_steering = angle
        norm = angle / self.max_steering_angle
        self.lbl_steer.setText(f"{angle}° ({norm:+.2f})")
        try:
            self.client.steer_sync(angle)
        except Exception as exc:
            self.lbl_dist.setText(f"STEER FAILED: {exc}")
        self.command_state.steer(angle, time.time())

    def _release_keys(self) -> None:
        """押しているキーをすべて離したことにする（操作が別のウィンドウへ移った時など）"""
        self._steer_left = False
        self._steer_right = False
        if self._drive_held:
            self._release_drive()

    def _emergency_stop(self):
        self._steer_left = False
        self._steer_right = False
        self._steer_value = 0.0
        self._drive_held = False
        self.drive_timer.stop()
        self.current_throttle = 0.0
        self.command_state.stop(time.time())
        self.command_state.steer(0, time.time())
        self.current_steering = 0
        self.lbl_speed.setText("0")
        self.lbl_steer.setText("0° (0.00)")
        try:
            self.client.emergency_stop_sync()
            self.client.steer_sync(0)
        except Exception as exc:
            self.lbl_dist.setText(f"E-STOP FAILED: {exc}")

    def _reset_emergency_stop(self):
        try:
            self.client.reset_emergency_stop_sync()
        except Exception as exc:
            self.lbl_dist.setText(f"E-STOP RESET FAILED: {exc}")

    @Slot(bytes, int, object)
    def _on_frame_received(self, img_bytes: bytes, seq: int, status: Optional[RobotStatus]):
        now = time.time()
        sensors = sensor_fields(status)
        distance_cm = sensors["distance_cm"]
        if distance_cm is not None:
            self.lbl_dist.setText(f"{distance_cm:.1f} cm")
        elif sensors["distance_far"]:
            self.lbl_dist.setText("far")
        self._update_rate(now)

        qimg = QImage()
        if qimg.loadFromData(img_bytes, "JPEG"):
            self.camera_label.setPixmap(QPixmap.fromImage(qimg))

        img = None
        if cv2 is not None and np is not None:
            img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        update = None
        if img is not None:
            update = self.motion_monitor.update(img, now, distance_cm, self.command_state.at(now))
            if update is not None:
                self._show_motion(update)

        if self.is_recording:
            self._save_sample(img_bytes, img, now, seq, sensors, update)

    def _update_rate(self, now: float) -> None:
        """直近1秒に届いたフレームの数を表示する（10fps 以上で集める。M3）"""
        self._frame_times = [t for t in self._frame_times if now - t < 1.0]
        self._frame_times.append(now)
        self.setWindowTitle(f"PiCar-X E2E Data Collector - {len(self._frame_times)} fps")

    def _show_motion(self, update: MonitorUpdate) -> None:
        check = update.check
        if check.flags:
            text = " / ".join(describe(flag) for flag in check.flags)
            self.lbl_motion.setStyleSheet("color: #d9534f; font-weight: bold;")
        else:
            text = f"OK (unknown: {check.unknown})" if check.unknown else "OK"
            self.lbl_motion.setStyleSheet("")
        sample = update.sample
        self.lbl_motion.setText(
            f"{text}<br><small>yaw {sample.yaw_rate_dps:+.0f}°/s, "
            f"flow {sample.edge_flow_px_s:.0f}px/s</small>"
        )

    def _on_motion_mismatch(self, flag: str, update: MonitorUpdate) -> None:
        """食い違いが続いた。走行を止め、記憶に残す（ROADMAP の M3）"""
        # W を押したままでも走り出さない。もう一度押し直すまで止まったまま
        self._drive_held = False
        self.drive_timer.stop()
        self.current_throttle = 0.0
        self.command_state.stop(time.time())
        self.lbl_speed.setText("0")
        try:
            self.client.stop_sync()
        except Exception as exc:
            self.lbl_dist.setText(f"STOP FAILED: {exc}")
        _, summary = record_motion_event(flag, update)
        print(f"[MOTION MISMATCH] {summary} {update.check.details}")

    def _save_sample(self, img_bytes, img, now, seq, sensors, update):
        filename = f"frame_{self.frame_count:06d}.jpg"
        filepath = self.images_dir / filename

        if img is not None:
            cv2.imwrite(str(filepath), cv2.resize(img, (160, 120)))
        else:
            with open(filepath, "wb") as f:
                f.write(img_bytes)

        command = self.command_state.at(now)
        steering_norm = self.current_steering / self.max_steering_angle
        record = {
            "index": self.frame_count,
            "timestamp": now,
            "frame": filename,
            "steering": round(steering_norm, 4),
            # 指令の期限が切れて止まった後は 0 にする（LRN-6）
            "throttle": 0.5 if command.speed > 0 else 0.0,
            "stream_seq": seq,
            **sensors,
        }
        if update is not None:
            record.update(update.to_record())
        else:
            record["command"] = command.to_record()

        with open(self.labels_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self.frame_count += 1
        self.lbl_rec_count.setText(f"Saved: {self.frame_count} frames")

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()

        if key == Qt.Key_W:
            if not event.isAutoRepeat() and not self._drive_held:
                self._drive_held = True
                self._send_drive()
                if self._drive_held:
                    self.drive_timer.start()

        elif key == Qt.Key_S:
            self._release_drive()

        elif key == Qt.Key_A:
            if not event.isAutoRepeat():
                self._steer_left = True

        elif key == Qt.Key_D:
            if not event.isAutoRepeat():
                self._steer_right = True

        elif key == Qt.Key_X:
            self._steer_value = 0.0
            self._apply_steer(0)

        elif key == Qt.Key_R:
            self._toggle_recording()

        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        key = event.key()
        if event.isAutoRepeat():
            return
        if key == Qt.Key_W:
            if self._drive_held:
                self._release_drive()
        elif key == Qt.Key_A:
            self._steer_left = False
        elif key == Qt.Key_D:
            self._steer_right = False
        else:
            super().keyReleaseEvent(event)

    def changeEvent(self, event):
        # 別のウィンドウへ操作が移るとキーを離したことが届かないので、その時点で離したことにする
        if event.type() == QEvent.ActivationChange and not self.isActiveWindow():
            self._release_keys()
        super().changeEvent(event)

    def closeEvent(self, event):
        self.steer_timer.stop()
        self._release_keys()
        self.camera_worker.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = E2ECollectorWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
