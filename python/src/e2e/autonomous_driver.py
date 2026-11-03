"""
Autonomous Driving Script using ONNX Runtime for PiCar-X (Plan A: Remote Mode)
PC上でONNXモデルを実行し、Raspberry Pi (Rust APIサーバー) を介して
家庭内フリースペース自律走行を行うスクリプト。
Phase 2.1: Disengagement Logger (直前3秒間の自動タグ付け & 介入記録)
Phase 3: ハイブリッド安全系 & 空間マッピング基盤 (Safety Layer & Spatial Intelligence)
  - 3.1: Cliff Detection (底面グレースケールによる段差・階段落下防止)
  - 3.2: Dynamic Speed Limiter (急旋回・障害物接近時の自動車速制御)
  - 3.3: Communication Watchdog (Wi-Fi通信断・ハートビート監視・タイムアウト停止)
  - 3.4: Breadcrumb Safety (直前走行軌跡の逆再生による十分な距離の安全後退・回避ターン)
  - 3.5: Dynamic Occupancy Grid (2D確率格子地図構築 & PC画面リアルタイム可視化UI)
  - 3.6: Stuck Detector (カメラ視覚変化量による引っかかり・空転検知 & 複合脱出ビヘイビア)
"""

import argparse
import base64
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

try:
    from .dataset.disengagement_manager import DisengagementBuffer
    from .mapping.occupancy_grid import OccupancyGridMap
    from .mapping.visualizer import MapVisualizer
    from .safety.cliff_detector import CliffDetector
    from .safety.speed_limiter import SpeedLimiter
    from .safety.stuck_detector import StuckDetector
    from .safety.trajectory_tracker import TrajectoryTracker
    from .safety.watchdog import CommunicationWatchdog
except ImportError:
    from src.e2e.dataset.disengagement_manager import DisengagementBuffer
    from src.e2e.mapping.occupancy_grid import OccupancyGridMap
    from src.e2e.mapping.visualizer import MapVisualizer
    from src.e2e.safety.cliff_detector import CliffDetector
    from src.e2e.safety.speed_limiter import SpeedLimiter
    from src.e2e.safety.stuck_detector import StuckDetector
    from src.e2e.safety.trajectory_tracker import TrajectoryTracker
    from src.e2e.safety.watchdog import CommunicationWatchdog


class RemoteAutonomousDriver:
    def __init__(
        self,
        model_path,
        speed=35,
        min_speed=20,
        curve_factor=1.0,
        safe_distance=15.0,
        slowdown_distance=40.0,
        cliff_threshold=150,
        enable_cliff=True,
        watchdog_timeout=0.6,
        max_consecutive_errors=3,
        reverse_duration_ms=1200,
        stuck_motion_threshold=4.0,
        stuck_duration_frames=15,
        enable_stuck_detection=True,
        enable_mapping=True,
        grid_size=200,
        cell_size_m=0.05,
        decay_rate=0.001,
        log_dir="data/disengagements",
        buffer_frames=30,
    ):
        self.model_path = model_path
        self.speed = speed
        self.safe_distance = safe_distance
        self.max_steering_angle = 35.0
        self.log_dir = Path(log_dir)
        self.enable_cliff = enable_cliff
        self.reverse_duration_ms = reverse_duration_ms
        self.enable_stuck_detection = enable_stuck_detection
        self.enable_mapping = enable_mapping

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

        # Phase 3.4: 直前軌跡メモリ & 安全後退シーケンサ
        self.trajectory_tracker = TrajectoryTracker(
            max_history_points=100,
            default_reverse_duration_ms=reverse_duration_ms,
            reverse_speed=min_speed,
        )

        # Phase 3.6: スタック（引っかかり・車輪空転）検知モジュール
        self.stuck_detector = StuckDetector(
            history_frames=8,
            min_motion_threshold=stuck_motion_threshold,
            stuck_duration_frames=stuck_duration_frames,
            min_moving_speed=min_speed,
        )

        # Phase 3.5: 動的オキュパンシーグリッド地図 & 可視化ダッシュボード
        self.grid_map: Optional[OccupancyGridMap] = None
        self.visualizer: Optional[MapVisualizer] = None
        if self.enable_mapping:
            self.grid_map = OccupancyGridMap(
                grid_width=grid_size,
                grid_height=grid_size,
                cell_size_m=cell_size_m,
                decay_rate=decay_rate,
            )
            self.visualizer = MapVisualizer(self.grid_map, display_scale=3)

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

    def execute_breadcrumb_reverse(
        self,
        target_duration_ms: Optional[int] = None,
        last_cliff_sensors: Optional[Tuple[bool, bool, bool]] = None,
    ) -> None:
        """
        直前に通ってきた安全な軌跡を逆再生して十分な距離を後退し、
        さらに車首を安全な方向へ振って脱出する。
        """
        dur_ms = target_duration_ms or self.reverse_duration_ms
        try:
            self.client.stop_sync()
        except Exception:
            pass
        time.sleep(0.05)

        sequence = self.trajectory_tracker.get_reverse_sequence(dur_ms)
        print(
            f"\n[BREADCRUMB REVERSE] Executing safe reverse along history "
            f"({len(sequence)} steps, total {dur_ms}ms)..."
        )

        for steer, spd, step_dur in sequence:
            try:
                self.client.steer_sync(steer)
                self.client.move_sync(
                    DriveDirection.BACKWARD, speed=spd, duration_ms=step_dur
                )
                if self.grid_map:
                    # 後退時は負の速度でオドメトリ更新
                    self.grid_map.update_odometry(
                        speed=-spd, steer_angle_deg=steer, duration_ms=step_dur
                    )
                if self.visualizer:
                    canvas = self.visualizer.render(
                        camera_frame_bgr=None,
                        speed=-spd,
                        steer_angle=steer,
                        status_text="BREADCRUMB REVERSE",
                        status_color=(0, 140, 255),
                    )
                    self.visualizer.show(canvas, wait_ms=1)
                time.sleep(step_dur / 1000.0)
            except Exception:
                self.watchdog.record_error()
                break

        try:
            self.client.stop_sync()
        except Exception:
            pass
        time.sleep(0.05)

        # 後退後の方向転換 (Turn Away)
        turn_steer, turn_spd, turn_dur = self.trajectory_tracker.get_escape_turn(
            last_cliff_sensors
        )
        print(
            f"[ESCAPE TURN] Turning away from hazard (Steer: {turn_steer:+02d}°, {turn_dur}ms)..."
        )
        try:
            self.client.steer_sync(turn_steer)
            self.client.move_sync(
                DriveDirection.FORWARD, speed=turn_spd, duration_ms=turn_dur
            )
            if self.grid_map:
                self.grid_map.update_odometry(
                    speed=turn_spd, steer_angle_deg=turn_steer, duration_ms=turn_dur
                )
            if self.visualizer:
                canvas = self.visualizer.render(
                    camera_frame_bgr=None,
                    speed=turn_spd,
                    steer_angle=turn_steer,
                    status_text="HAZARD ESCAPE TURN",
                    status_color=(0, 215, 255),
                )
                self.visualizer.show(canvas, wait_ms=1)
            time.sleep(turn_dur / 1000.0)
            self.client.stop_sync()
            self.client.steer_sync(0)
            self.watchdog.heartbeat()
        except Exception:
            self.watchdog.record_error()

        # 脱出完了後に軌跡履歴とスタック検知をリセット
        self.trajectory_tracker.clear()
        self.stuck_detector.reset()
        print("[ESCAPE COMPLETED] Resuming autonomous driving.")

    def execute_unstuck(self) -> None:
        """
        引っかかり・車輪空転状態からの自動脱出（バック ＋ 左右車体振り ＋ 迂回前進）
        """
        print(
            f"\n[STUCK DETECTED] Vehicle trapped/spinning wheels "
            f"(motion: {self.stuck_detector.last_motion_value:.2f}) -> Executing Unstuck Action!"
        )
        try:
            self.client.stop_sync()
        except Exception:
            pass

        # Disengagement ロギング
        saved_path = self.buffer.save_event(
            trigger="stuck_detected",
            output_base_dir=self.log_dir,
            model_name=self.model_path,
            note=(
                f"Vehicle stuck/wheels spinning: visual motion dropped "
                f"to {self.stuck_detector.last_motion_value:.2f}"
            ),
        )
        if saved_path:
            print(f"[DISENGAGEMENT LOGGED] Saved to: {saved_path}")

        # Step 1: まず直前軌跡を逆再生してしっかり後退 (1000ms)
        print("[UNSTUCK: STEP 1] Reversing along trajectory away from trap...")
        self.execute_breadcrumb_reverse(target_duration_ms=1000)

        # Step 2: 左右の車体振り (Wiggle) でタイヤの引っかかり・乗り上げを解消
        print("[UNSTUCK: STEP 2] Wiggling chassis left and right...")
        wiggle_actions = [
            (-35, self.speed_limiter.min_speed, 350),  # 左切り前進
            (35, self.speed_limiter.min_speed, 350),   # 右切り前進
            (0, self.speed_limiter.min_speed, 400),    # まっすぐ少し後退
            (35, self.speed_limiter.min_speed, 450),   # 迂回前進
        ]
        for steer, spd, dur in wiggle_actions:
            try:
                self.client.steer_sync(steer)
                if steer == 0:
                    self.client.move_sync(
                        DriveDirection.BACKWARD, speed=spd, duration_ms=dur
                    )
                    if self.grid_map:
                        self.grid_map.update_odometry(-spd, steer, dur)
                else:
                    self.client.move_sync(
                        DriveDirection.FORWARD, speed=spd, duration_ms=dur
                    )
                    if self.grid_map:
                        self.grid_map.update_odometry(spd, steer, dur)
                if self.visualizer:
                    canvas = self.visualizer.render(
                        camera_frame_bgr=None,
                        speed=spd,
                        steer_angle=steer,
                        status_text="UNSTUCK WIGGLE ACTION",
                        status_color=(0, 215, 255),
                    )
                    self.visualizer.show(canvas, wait_ms=1)
                time.sleep(dur / 1000.0)
            except Exception:
                break

        try:
            self.client.stop_sync()
            self.client.steer_sync(0)
        except Exception:
            pass

        self.stuck_detector.reset()
        self.trajectory_tracker.clear()
        print("[UNSTUCK COMPLETED] Free from trap! Resuming autonomous driving.\n")

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
        print(
            f"Breadcrumb Reverse: ACTIVE (Target Distance/Duration: {self.reverse_duration_ms}ms)"
        )
        stuck_status = (
            f"ACTIVE (Threshold: {self.stuck_detector.min_motion_threshold})"
            if self.enable_stuck_detection
            else "DISABLED"
        )
        print(f"Stuck Auto-Recovery Guard: {stuck_status}")
        map_status = (
            f"ACTIVE (Grid: {self.grid_map.width}x{self.grid_map.height}, "
            f"Resolution: {self.grid_map.cell_size * 100:.0f}cm)"
            if self.enable_mapping and self.grid_map
            else "DISABLED"
        )
        print(f"Dynamic Occupancy Mapping UI: {map_status}")
        print("Disengagement Logging: ACTIVE (auto-records pre-event 3s)")
        print("Press Ctrl+C or 'Q' in Map Window to emergency stop.\n")

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
        obstacle_encounter_count = 0
        step_counter = 0
        last_loop_time = time.time()

        try:
            while True:
                now = time.time()
                # 実際のループ経過時間 (10ms〜200msの範囲に制限)
                loop_dt_ms = max(10, min(200, int((now - last_loop_time) * 1000)))
                last_loop_time = now

                step_counter += 1
                status_badge = "NORMAL DRIVING"
                badge_color = (0, 220, 0)

                # ----------------------------------------------------
                # 1. 多重安全チェック (Phase 3 Safety Layer)
                # ----------------------------------------------------

                # 1-1. 通信ウォッチドッグ確認 (Phase 3.3 Watchdog)
                if self.watchdog.should_stop:
                    status_badge = "WATCHDOG TIMEOUT"
                    badge_color = (0, 0, 220)
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

                # 1-2. 段差・階段落下防止ガード (Phase 3.1 & 3.4 Breadcrumb Reverse)
                if self.enable_cliff:
                    try:
                        gs = self.client.get_grayscale_sync()
                        self.watchdog.heartbeat()
                    except Exception:
                        self.watchdog.record_error()
                        gs = [999, 999, 999]

                    if self.cliff_detector.is_cliff(gs):
                        status_badge = "CLIFF DROP-OFF DETECTED"
                        badge_color = (0, 0, 255)
                        msg = f"\r[CLIFF GUARD] Drop-off Detected! {gs} -> STOPPING & RETREATING"
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

                        # 直前軌跡（パンくずリスト）を逆再生して安全に後退 ＋ 回避旋回
                        cliff_status = self.cliff_detector.get_cliff_status(gs)
                        self.execute_breadcrumb_reverse(
                            target_duration_ms=self.reverse_duration_ms,
                            last_cliff_sensors=cliff_status,
                        )
                        continue

                self.is_cliff_stopped = False

                # 1-3. 前方障害物チェック (超音波センサー & 行き止まり脱出)
                try:
                    distance = self.client.get_ultrasonic_sync()
                    self.watchdog.heartbeat()
                except Exception:
                    self.watchdog.record_error()
                    distance = 999.0

                # 障害物距離をマップに反映
                if self.grid_map:
                    self.grid_map.update_sensor_ultrasonic(distance)
                    if step_counter % 5 == 0:
                        self.grid_map.decay()

                if 0 < distance < self.safe_distance:
                    status_badge = "OBSTACLE STOP"
                    badge_color = (0, 140, 255)
                    obstacle_encounter_count += 1
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

                    # 壁や家具の前で行き止まりになった場合は、直前軌跡を逆走して脱出
                    if obstacle_encounter_count >= 2:
                        print(
                            "\n[OBSTACLE DEAD-END] In front of obstacle "
                            "-> Executing reverse escape."
                        )
                        self.execute_breadcrumb_reverse(
                            target_duration_ms=self.reverse_duration_ms
                        )
                        obstacle_encounter_count = 0
                    else:
                        time.sleep(0.1)
                    continue

                # 障害物を脱したらフラグリセット
                self.is_obstacle_stopped = False
                obstacle_encounter_count = 0

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
                # 3. Phase 3.6: スタック（引っかかり・空転）検知 & 自動脱出
                # ----------------------------------------------------
                if self.enable_stuck_detection:
                    if self.stuck_detector.update(
                        frame, motor_speed=self.speed_limiter.min_speed
                    ):
                        status_badge = "UNSTUCK ACTION"
                        badge_color = (0, 215, 255)
                        self.execute_unstuck()
                        continue

                # ----------------------------------------------------
                # 4. 前処理 & ONNX推論
                # ----------------------------------------------------
                input_tensor = self.preprocess(frame)
                outputs = self.session.run(None, {self.input_name: input_tensor})
                steering_norm = float(outputs[0][0][0])  # -1.0 ~ 1.0

                # ----------------------------------------------------
                # 5. Phase 3.2: 動的車速制御 (Speed Limiter)
                # ----------------------------------------------------
                dynamic_speed = self.speed_limiter.calculate(
                    steering_norm=steering_norm,
                    obstacle_distance_cm=distance if distance > 0 else None,
                )

                # ----------------------------------------------------
                # 6. 実機制御 & 軌跡記録 (Phase 3.4 Breadcrumb Tracker)
                # ----------------------------------------------------
                steer_angle = int(steering_norm * self.max_steering_angle)
                last_steer_norm = steering_norm
                last_steer_angle = steer_angle

                # 直前3秒リングバッファに保存
                self.buffer.append(
                    frame_bgr=last_frame_bgr,
                    steering_norm=last_steer_norm,
                    steer_angle=last_steer_angle,
                    ultrasonic_dist=distance,
                    speed=dynamic_speed,
                )

                # Phase 3.4: 安全走行軌跡に記録 (実経過時間を記録)
                self.trajectory_tracker.record(
                    speed=dynamic_speed,
                    steer_angle=steer_angle,
                    duration_ms=loop_dt_ms,
                    timestamp=now,
                )

                # Phase 3.5: オドメトリ更新 (実経過時間の移動量を積算)
                if self.grid_map:
                    self.grid_map.update_odometry(
                        speed=dynamic_speed,
                        steer_angle_deg=steer_angle,
                        duration_ms=loop_dt_ms,
                    )

                try:
                    self.client.steer_sync(steer_angle)
                    self.client.move_sync(
                        DriveDirection.FORWARD, speed=dynamic_speed, duration_ms=400
                    )
                    self.watchdog.heartbeat()
                except Exception:
                    self.watchdog.record_error()

                # ----------------------------------------------------
                # 7. Phase 3.5: マッピングUI リアルタイム描画
                # ----------------------------------------------------
                if self.visualizer:
                    canvas = self.visualizer.render(
                        camera_frame_bgr=last_frame_bgr,
                        speed=dynamic_speed,
                        steer_angle=steer_angle,
                        ultrasonic_dist=distance,
                        motion_value=self.stuck_detector.last_motion_value,
                        status_text=status_badge,
                        status_color=badge_color,
                    )
                    key = self.visualizer.show(canvas, wait_ms=1)
                    if key == ord("q") or key == ord("Q"):
                        print("\n[STOP] User pressed 'Q' in Map Window.")
                        break

                motion_val = self.stuck_detector.last_motion_value
                status = (
                    f"\rSteer: {steering_norm:+.2f} | Angle: {steer_angle:+02d}° | "
                    f"Spd: {dynamic_speed:02d} | Flow: {motion_val:04.1f} | Dist: {distance:.1f}cm"
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
            if self.visualizer:
                self.visualizer.close()
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
        "--model", type=str, required=True, help="Path to ONNX model"
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
        "--reverse_duration_ms",
        type=int,
        default=1200,
        help="Target duration in ms to reverse along safe trajectory during escape",
    )
    parser.add_argument(
        "--stuck_threshold",
        type=float,
        default=4.0,
        help="Visual motion threshold below which vehicle is considered visually static",
    )
    parser.add_argument(
        "--stuck_frames",
        type=int,
        default=15,
        help="Consecutive static frames before triggering auto-unstuck behavior",
    )
    parser.add_argument(
        "--disable_stuck",
        action="store_true",
        help="Disable visual stuck detection and auto-recovery",
    )
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Disable real-time 2D map visualizer window (headless mode)",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=200,
        help="Map grid dimensions in cells (default: 200 = 10m x 10m)",
    )
    parser.add_argument(
        "--cell_size",
        type=float,
        default=0.05,
        help="Map cell resolution in meters (default: 0.05 = 5cm)",
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
        reverse_duration_ms=args.reverse_duration_ms,
        stuck_motion_threshold=args.stuck_threshold,
        stuck_duration_frames=args.stuck_frames,
        enable_stuck_detection=not args.disable_stuck,
        enable_mapping=not args.no_gui,
        grid_size=args.grid_size,
        cell_size_m=args.cell_size,
        log_dir=args.log_dir,
        buffer_frames=args.buffer_frames,
    )
    driver.run()


if __name__ == "__main__":
    main()
