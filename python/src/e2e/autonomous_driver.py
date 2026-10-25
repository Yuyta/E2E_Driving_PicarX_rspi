"""
Autonomous Driving Script using ONNX Runtime for PiCar-X (Plan A: Remote Mode)
PC上でONNXモデルを実行し、Raspberry Pi (Rust APIサーバー) を介して
家庭内フリースペース自律走行を行うスクリプト。
Phase 2.1: Disengagement Logger (直前3秒間の自動タグ付け & 介入記録)
Phase 3: ハイブリッド安全系 (Safety Layer)
  - 3.1: Cliff Detection (底面グレースケールによる段差・階段落下防止 & 自動後退退避)
  - 3.2: Dynamic Speed Limiter (急旋回・障害物接近時の自動車速制御)
  - 3.3: Communication Watchdog (Wi-Fi通信断・ハートビート監視・タイムアウト停止)
"""

import argparse
import base64
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

try:
    from .dataset.disengagement_manager import DisengagementBuffer
    from .safety.cliff_detector import CliffDetector
    from .safety.speed_limiter import SpeedLimiter
    from .safety.watchdog import CommunicationWatchdog
except ImportError:
    from src.e2e.dataset.disengagement_manager import DisengagementBuffer
    from src.e2e.safety.cliff_detector import CliffDetector
    from src.e2e.safety.speed_limiter import SpeedLimiter
    from src.e2e.safety.watchdog import CommunicationWatchdog


class RemoteAutonomousDriver:
    def __init__(
        self,
        model_path="models/v001/model.onnx",
        speed=35,
        min_speed=20,
        curve_factor=1.0,
        safe_distance=15.0,
        slowdown_distance=40.0,
        cliff_threshold=150,
        enable_cliff=True,
        watchdog_timeout=0.6,
        max_consecutive_errors=3,
        log_dir="data/disengagements",
        buffer_frames=30,
    ):
        self.model_path = model_path
        self.speed = speed
        self.safe_distance = safe_distance
        self.max_steering_angle = 35.0
        self.log_dir = Path(log_dir)
        self.enable_cliff = enable_cliff

        print(f"Loading ONNX Model from: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # 既存クライアント再利用 (Rust APIサーバー経由)
        self.client = PicarxClient()

        # Phase 2.1: 直前走行データを常時保持するリングバッファ
        self.buffer = DisengagementBuffer(max_frames=buffer_frames)
        self.is_obstacle_stopped = False
        self.is_cliff_stopped = False
        self.is_watchdog_stopped = False

        # Phase 3.1: 段差・階段落下検知モジュール
        self.cliff_detector = CliffDetector(threshold=cliff_threshold)

        # Phase 3.2: 旋回・障害物動的減速リミッター
        self.speed_limiter = SpeedLimiter(
            base_speed=speed,
            min_speed=min_speed,
            curve_factor=curve_factor,
            safe_distance=safe_distance,
            slowdown_distance=slowdown_distance,
        )

        # Phase 3.3: 通信タイムアウト・監視ウォッチドッグ
        self.watchdog = CommunicationWatchdog(
            timeout_sec=watchdog_timeout,
            max_consecutive_errors=max_consecutive_errors,
        )

    def preprocess(self, bgr_img):
        """BGR画像 -> (1, 3, 120, 160) の正規化Tensor"""
        resized = cv2.resize(bgr_img, (160, 120))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = rgb_img.astype(np.float32) / 255.0

        # ImageNet正規化
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std

        # HWC -> CHW -> NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def run(self):
        print("\n=== PiCar-X Remote Autonomous Driving Started (Indoor Home) ===")
        print(f"Target Robot: {self.client.base_url}")
        print(
            f"Speed: {self.speed} (Min: {self.speed_limiter.min_speed}) | "
            f"Safe Dist: {self.safe_distance}cm "
            f"(Slowdown: {self.speed_limiter.slowdown_distance}cm)"
        )
        cliff_status = (
            f"ACTIVE (Threshold: {self.cliff_detector.threshold})"
            if self.enable_cliff
            else "DISABLED"
        )
        print(f"Cliff Safety Guard: {cliff_status}")
        print(
            f"Watchdog Guard: ACTIVE (Timeout: {self.watchdog.timeout_sec}s, "
            f"Max Errors: {self.watchdog.max_consecutive_errors})"
        )
        print("Disengagement Logging: ACTIVE (auto-records pre-event 3s)")
        print("Press Ctrl+C to emergency stop.\n")

        # 前進開始
        try:
            self.client.move_sync(
                DriveDirection.FORWARD,
                speed=self.speed_limiter.min_speed,
                duration_ms=1000,
            )
            self.watchdog.heartbeat()
        except Exception:
            self.watchdog.record_error()

        last_frame_bgr = None
        last_steer_norm = 0.0
        last_steer_angle = 0

        try:
            while True:
                # ----------------------------------------------------
                # 1. 多重安全チェック (Phase 3 Safety Layer)
                # ----------------------------------------------------

                # 1-1. 通信ウォッチドッグ確認 (Phase 3.3 Watchdog)
                if self.watchdog.should_stop:
                    msg = (
                        f"\r[WATCHDOG STOP] Comm timeout or errors "
                        f"({self.watchdog.consecutive_errors})! STOPPING."
                    )
                    print(msg, end="", flush=True)
                    try:
                        self.client.stop_sync()
                    except Exception:
                        pass

                    if not self.is_watchdog_stopped:
                        saved_path = self.buffer.save_event(
                            trigger="communication_timeout",
                            output_base_dir=self.log_dir,
                            model_name=self.model_path,
                            note=(
                                f"Triggered by watchdog timeout "
                                f"(errors={self.watchdog.consecutive_errors})"
                            ),
                        )
                        if saved_path:
                            print(f"\n[DISENGAGEMENT LOGGED] Saved to: {saved_path}")
                        self.is_watchdog_stopped = True

                    time.sleep(0.2)
                    # 再接続試行
                    try:
                        self.client.get_ultrasonic_sync()
                        self.watchdog.heartbeat()
                        self.is_watchdog_stopped = False
                        print("\n[WATCHDOG RECOVERED] Communication restored.")
                    except Exception:
                        self.watchdog.record_error()
                    continue

                # 1-2. 段差・階段落下防止ガード (Phase 3.1 Cliff Detection)
                if self.enable_cliff:
                    try:
                        gs = self.client.get_grayscale_sync()
                        self.watchdog.heartbeat()
                    except Exception:
                        self.watchdog.record_error()
                        gs = [999, 999, 999]

                    if self.cliff_detector.is_cliff(gs):
                        steer, escape_spd, dur_ms = self.cliff_detector.get_escape_action(gs)
                        msg = f"\r[CLIFF GUARD] Drop-off Detected! {gs} -> STOPPING & ESCAPING"
                        print(msg, end="", flush=True)
                        try:
                            self.client.stop_sync()
                        except Exception:
                            pass

                        # 落下危機の初回のみ直前3秒間を自動タグ付け
                        if not self.is_cliff_stopped:
                            saved_path = self.buffer.save_event(
                                trigger="cliff_detected",
                                output_base_dir=self.log_dir,
                                model_name=self.model_path,
                                note=f"Triggered by floor drop-off sensor values: {gs}",
                            )
                            if saved_path:
                                print(f"\n[DISENGAGEMENT LOGGED] Saved to: {saved_path}")
                            self.is_cliff_stopped = True

                        # 安全エリアへ自動後退（エスケープ）
                        try:
                            self.client.steer_sync(steer)
                            self.client.move_sync(
                                DriveDirection.BACKWARD,
                                speed=abs(escape_spd),
                                duration_ms=dur_ms,
                            )
                            time.sleep(dur_ms / 1000.0)
                            self.client.stop_sync()
                            self.client.steer_sync(0)
                        except Exception:
                            self.watchdog.record_error()
                        continue

                self.is_cliff_stopped = False

                # 1-3. 前方障害物チェック (超音波センサー)
                try:
                    distance = self.client.get_ultrasonic_sync()
                    self.watchdog.heartbeat()
                except Exception:
                    self.watchdog.record_error()
                    distance = 999.0

                if 0 < distance < self.safe_distance:
                    msg = f"\r[SAFETY GUARD] Obstacle Close: {distance:.1f}cm! STOPPING."
                    print(msg, end="", flush=True)
                    try:
                        self.client.stop_sync()
                    except Exception:
                        pass

                    # 障害物急ブレーキの初回のみ直前3秒間を自動タグ付け
                    if not self.is_obstacle_stopped:
                        saved_path = self.buffer.save_event(
                            trigger="obstacle_stop",
                            output_base_dir=self.log_dir,
                            model_name=self.model_path,
                            note=f"Triggered by ultrasonic obstacle distance {distance:.1f}cm",
                        )
                        if saved_path:
                            print(f"\n[DISENGAGEMENT LOGGED] Saved to: {saved_path}")
                        self.is_obstacle_stopped = True

                    time.sleep(0.1)
                    continue

                # 障害物を脱したらフラグリセット
                self.is_obstacle_stopped = False

                # ----------------------------------------------------
                # 2. カメラ画像取得
                # ----------------------------------------------------
                try:
                    snapshot = self.client.get_camera_snapshot_sync(width=320, quality=70)
                    if not snapshot or not snapshot.image_base64:
                        self.watchdog.record_error()
                        continue
                    img_bytes = base64.b64decode(snapshot.image_base64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        self.watchdog.record_error()
                        continue
                    last_frame_bgr = frame
                    self.watchdog.heartbeat()
                except Exception:
                    self.watchdog.record_error()
                    continue

                # ----------------------------------------------------
                # 3. 前処理 & ONNX推論
                # ----------------------------------------------------
                input_tensor = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                steering_norm = float(outputs[0][0][0])  # -1.0 ~ 1.0

                # ----------------------------------------------------
                # 4. Phase 3.2: 動的車速制御 (Speed Limiter)
                # ----------------------------------------------------
                dynamic_speed = self.speed_limiter.calculate(
                    steering_norm=steering_norm,
                    obstacle_distance_cm=distance if distance > 0 else None,
                )

                # ----------------------------------------------------
                # 5. 実機制御 (Rust APIへ送信)
                # ----------------------------------------------------
                steer_angle = int(steering_norm * self.max_steering_angle)
                last_steer_norm = steering_norm
                last_steer_angle = steer_angle

                # リングバッファに常時蓄積 (直前3秒分)
                self.buffer.append(
                    frame_bgr=last_frame_bgr,
                    steering_norm=last_steer_norm,
                    steer_angle=last_steer_angle,
                    ultrasonic_dist=distance,
                    speed=dynamic_speed,
                )

                try:
                    self.client.steer_sync(steer_angle)
                    self.client.move_sync(
                        DriveDirection.FORWARD, speed=dynamic_speed, duration_ms=400
                    )
                    self.watchdog.heartbeat()
                except Exception:
                    self.watchdog.record_error()

                status = (
                    f"\rSteer: {steering_norm:+.2f} | Angle: {steer_angle:+02d}° | "
                    f"Spd: {dynamic_speed:02d} | Dist: {distance:.1f}cm"
                )
                print(status, end="", flush=True)
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[MANUAL INTERVENTION] Emergency Stop requested by user.")
            saved_path = self.buffer.save_event(
                trigger="manual_intervention",
                output_base_dir=self.log_dir,
                model_name=self.model_path,
                note="Interrupted by Ctrl+C (human intervention to prevent collision/stuck)",
            )
            if saved_path:
                print(f"[DISENGAGEMENT LOGGED] Pre-event frames saved to: {saved_path}")
        finally:
            try:
                self.client.steer_sync(0)
                self.client.stop_sync()
            except Exception:
                pass
            print("Vehicle safely stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Driving with ONNX via Rust API (Indoor Home Free-Space)"
    )
    parser.add_argument(
        "--model", type=str, default="models/v001/model.onnx", help="Path to ONNX model"
    )
    parser.add_argument("--speed", type=int, default=35, help="Base driving speed (0-100)")
    parser.add_argument(
        "--min_speed", type=int, default=20, help="Minimum curve/slowdown speed (0-100)"
    )
    parser.add_argument(
        "--curve_factor",
        type=float,
        default=1.0,
        help="Steering curve slowdown sensitivity factor",
    )
    parser.add_argument(
        "--safe_dist", type=float, default=15.0, help="Safety emergency stop distance in cm"
    )
    parser.add_argument(
        "--slowdown_dist",
        type=float,
        default=40.0,
        help="Obstacle distance to begin dynamic deceleration in cm",
    )
    parser.add_argument(
        "--cliff_threshold",
        type=int,
        default=150,
        help="Cliff/drop-off threshold ADC value (values below this trigger cliff guard)",
    )
    parser.add_argument(
        "--disable_cliff",
        action="store_true",
        help="Disable cliff/drop-off safety guard",
    )
    parser.add_argument(
        "--watchdog_timeout",
        type=float,
        default=0.6,
        help="Communication watchdog timeout in seconds",
    )
    parser.add_argument(
        "--max_errors",
        type=int,
        default=3,
        help="Maximum consecutive communication errors before emergency stop",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="data/disengagements",
        help="Directory to save disengagement events",
    )
    parser.add_argument(
        "--buffer_frames",
        type=int,
        default=30,
        help="Number of pre-event frames to retain in buffer",
    )

    args = parser.parse_args()
    driver = RemoteAutonomousDriver(
        model_path=args.model,
        speed=args.speed,
        min_speed=args.min_speed,
        curve_factor=args.curve_factor,
        safe_distance=args.safe_dist,
        slowdown_distance=args.slowdown_dist,
        cliff_threshold=args.cliff_threshold,
        enable_cliff=not args.disable_cliff,
        watchdog_timeout=args.watchdog_timeout,
        max_consecutive_errors=args.max_errors,
        log_dir=args.log_dir,
        buffer_frames=args.buffer_frames,
    )
    driver.run()


if __name__ == "__main__":
    main()
