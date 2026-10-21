"""
Autonomous Driving Script using ONNX Runtime for PiCar-X (Plan A: Remote Mode)
PC上でONNXモデルを実行し、Raspberry Pi (Rust APIサーバー) を介して
リアルタイム推論・自動走行を行うスクリプト。
"""

import argparse
import base64
import time

import cv2
import numpy as np
import onnxruntime as ort

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient


class RemoteAutonomousDriver:
    def __init__(self, model_path="models/v001/model.onnx", speed=35, safe_distance=15.0):
        self.model_path = model_path
        self.speed = speed
        self.safe_distance = safe_distance
        self.max_steering_angle = 35.0

        print(f"Loading ONNX Model from: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # 既存クライアント再利用 (Rust APIサーバー経由)
        self.client = PicarxClient()

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
        print("\n=== PiCar-X Remote Autonomous Driving Started ===")
        print(f"Target Robot: {self.client.base_url}")
        print(f"Speed: {self.speed} | Safe Dist: {self.safe_distance}cm")
        print("Press Ctrl+C to emergency stop.\n")

        # 前進開始
        self.client.move_sync(DriveDirection.FORWARD, speed=self.speed, duration_ms=1000)

        try:
            while True:
                # 1. 安全チェック (Safety Layer)
                try:
                    distance = self.client.get_ultrasonic_sync()
                except Exception:
                    distance = 999.0

                if 0 < distance < self.safe_distance:
                    msg = f"\r[SAFETY GUARD] Obstacle: {distance:.1f}cm! STOPPING."
                    print(msg, end="", flush=True)
                    self.client.stop_sync()
                    time.sleep(0.1)
                    continue

                # 2. カメラ画像取得
                try:
                    snapshot = self.client.get_camera_snapshot_sync(width=320, quality=70)
                    if not snapshot or not snapshot.image_base64:
                        continue
                    img_bytes = base64.b64decode(snapshot.image_base64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                except Exception:
                    continue

                # 3. 前処理 & ONNX推論
                input_tensor = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                steering_norm = float(outputs[0][0][0])  # -1.0 ~ 1.0

                # 4. 実機制御 (Rust APIへ送信)
                steer_angle = int(steering_norm * self.max_steering_angle)
                self.client.steer_sync(steer_angle)
                self.client.move_sync(DriveDirection.FORWARD, speed=self.speed, duration_ms=400)

                status = (
                    f"\rSteer: {steering_norm:+.2f} | Angle: {steer_angle:+02d}° | "
                    f"Dist: {distance:.1f}cm"
                )
                print(status, end="", flush=True)
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.client.steer_sync(0)
            self.client.stop_sync()
            print("Vehicle safely stopped.")


def main():
    parser = argparse.ArgumentParser(description="Autonomous Driving with ONNX via Rust API")
    parser.add_argument(
        "--model", type=str, default="models/v001/model.onnx", help="Path to ONNX model"
    )
    parser.add_argument("--speed", type=int, default=35, help="Driving speed (0-100)")
    parser.add_argument(
        "--safe_dist", type=float, default=15.0, help="Safety stop distance in cm"
    )

    args = parser.parse_args()
    driver = RemoteAutonomousDriver(
        model_path=args.model, speed=args.speed, safe_distance=args.safe_dist
    )
    driver.run()


if __name__ == "__main__":
    main()
