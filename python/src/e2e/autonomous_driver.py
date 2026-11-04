"""
Autonomous Driving Script using ONNX Runtime for PiCar-X (Plan A: Remote Mode)
PC上でONNXモデルを実行し、Raspberry Pi (Rust APIサーバー) を介して
家庭内フリースペース自律走行を行うスクリプト。

Pi の反射ループが、障害物、段差、不明なセンサー値、指令の期限切れで車両を止める。
このスクリプトが止まっても、最後の指令の TTL が切れた時点で車両は止まる。
ここで担当するのは推論と、Pi の判定を受けた後の行動（脱出、待機、介入の記録）である。

  - 配信の最新フレームごとに推論する。周期の内訳は CycleProfiler で計測する
  - Phase 2.1: Disengagement Logger (直前の自動タグ付け & 介入記録)
  - 3.2: Dynamic Speed Limiter (急旋回・障害物接近時の自動車速制御)
  - 3.3: Communication Watchdog (通信状態の表示と記録。停止は Pi が行う)
  - 3.4: Breadcrumb Safety (記録した軌跡の範囲に限った安全後退・回避ターン)
  - 3.5: Dynamic Occupancy Grid (2D確率格子地図構築 & PC画面リアルタイム可視化UI)
  - 3.6: Stuck Detector (カメラ視覚変化量による引っかかり・空転検知 & 複合脱出ビヘイビア)
"""

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import httpx
import numpy as np
import onnxruntime as ort

from src.models import DriveDirection
from src.picarx_client.client import PicarxClient

try:
    from .cycle_profiler import CycleProfiler
    from .dataset.disengagement_manager import DisengagementBuffer
    from .mapping.occupancy_grid import OccupancyGridMap
    from .mapping.visualizer import MapVisualizer
    from .safety.body_status import BodyAction, cliff_sensors, decide_action
    from .safety.speed_limiter import SpeedLimiter
    from .safety.stuck_detector import StuckDetector
    from .safety.trajectory_tracker import TrajectoryTracker
    from .safety.watchdog import CommunicationWatchdog
except ImportError:
    from src.e2e.cycle_profiler import CycleProfiler
    from src.e2e.dataset.disengagement_manager import DisengagementBuffer
    from src.e2e.mapping.occupancy_grid import OccupancyGridMap
    from src.e2e.mapping.visualizer import MapVisualizer
    from src.e2e.safety.body_status import BodyAction, cliff_sensors, decide_action
    from src.e2e.safety.speed_limiter import SpeedLimiter
    from src.e2e.safety.stuck_detector import StuckDetector
    from src.e2e.safety.trajectory_tracker import TrajectoryTracker
    from src.e2e.safety.watchdog import CommunicationWatchdog


# 走行指令の TTL。指令が届かなくなると、この時間で Pi が車両を止める
MOVE_TTL_MS = 400
# 制御の通信は短く打ち切る。止まっていても車両は TTL で止まる
CONTROL_TIMEOUT_SEC = 1.0
FRAME_TIMEOUT_SEC = 0.5
WAIT_SLEEP_SEC = 0.1
# 障害物の前でこの周回数（約1秒）待っても塞がったままなら、行き止まりとして脱出する
OBSTACLE_ESCAPE_CYCLES = 10


class RemoteAutonomousDriver:
    def __init__(
        self,
        model_path,
        speed=35,
        min_speed=20,
        curve_factor=1.0,
        safe_distance=15.0,
        slowdown_distance=40.0,
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
        self.reverse_duration_ms = reverse_duration_ms
        self.enable_stuck_detection = enable_stuck_detection
        self.enable_mapping = enable_mapping

        print(f"Loading ONNX Model from: {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

        # 既存クライアント再利用 (Rust APIサーバー経由)
        self.client = PicarxClient(timeout=CONTROL_TIMEOUT_SEC)
        self.profiler = CycleProfiler()

        # Phase 2.1: 直前走行データを常時保持するリングバッファ
        self.buffer = DisengagementBuffer(max_frames=buffer_frames)
        self.is_obstacle_stopped = False
        self.is_cliff_stopped = False
        self.is_watchdog_stopped = False
        self._reverse_refused = False

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

    def _send_move(self, direction, speed: int, steer: int, duration_ms: int) -> bool:
        """舵角と走行を指令する。Pi に拒否された時や通信失敗時は `False` を返す。"""
        try:
            self.client.steer_sync(steer)
            moved = self.client.move_sync(direction, speed=speed, ttl_ms=duration_ms)
            self.watchdog.heartbeat()
        except httpx.HTTPError:
            self.watchdog.record_error()
            return False
        if moved and self.grid_map:
            signed = -speed if direction == DriveDirection.BACKWARD else speed
            self.grid_map.update_odometry(
                speed=signed, steer_angle_deg=steer, duration_ms=duration_ms
            )
        return moved

    def _stop_quietly(self) -> None:
        try:
            self.client.stop_sync()
        except httpx.HTTPError:
            self.watchdog.record_error()

    def _show_status(self, speed: int, steer: int, text: str, color) -> None:
        if self.visualizer:
            canvas = self.visualizer.render(
                camera_frame_bgr=None,
                speed=speed,
                steer_angle=steer,
                status_text=text,
                status_color=color,
            )
            self.visualizer.show(canvas, wait_ms=1)

    def execute_breadcrumb_reverse(
        self,
        target_duration_ms: Optional[int] = None,
        last_cliff_sensors: Optional[Tuple[bool, bool, bool]] = None,
    ) -> bool:
        """
        記録した軌跡の範囲で後退し、車首を安全な方向へ振って脱出する。

        後方にセンサーがないので、記録のない範囲へは後退しない。
        1歩も後退できなかった時は何もせず `False` を返す。
        """
        dur_ms = target_duration_ms or self.reverse_duration_ms
        self._stop_quietly()

        sequence = self.trajectory_tracker.get_reverse_sequence(dur_ms)
        if not sequence:
            if not self._reverse_refused:
                print("\n[BREADCRUMB REVERSE] No recorded path behind the car. Waiting for help.")
            self._reverse_refused = True
            return False
        self._reverse_refused = False

        print(
            f"\n[BREADCRUMB REVERSE] Reversing along history "
            f"({len(sequence)} steps, target {dur_ms}ms)..."
        )
        reversed_ms = 0
        for steer, spd, step_dur in sequence:
            if not self._send_move(DriveDirection.BACKWARD, spd, steer, step_dur):
                break
            self._show_status(-spd, steer, "BREADCRUMB REVERSE", (0, 140, 255))
            time.sleep(step_dur / 1000.0)
            reversed_ms += step_dur
        self.trajectory_tracker.consume_reverse(reversed_ms)
        self._stop_quietly()
        if reversed_ms == 0:
            return False

        # 後退後の方向転換 (Turn Away)。危険があれば Pi が止める
        turn_steer, turn_spd, turn_dur = self.trajectory_tracker.get_escape_turn(
            last_cliff_sensors
        )
        print(
            f"[ESCAPE TURN] Turning away from hazard (Steer: {turn_steer:+02d}°, {turn_dur}ms)..."
        )
        if self._send_move(DriveDirection.FORWARD, turn_spd, turn_steer, turn_dur):
            self.trajectory_tracker.record(turn_spd, turn_steer, turn_dur, time.time())
            self._show_status(turn_spd, turn_steer, "HAZARD ESCAPE TURN", (0, 215, 255))
            time.sleep(turn_dur / 1000.0)
        self._stop_quietly()

        self.stuck_detector.reset()
        print("[ESCAPE COMPLETED] Resuming autonomous driving.")
        return True

    def execute_unstuck(self) -> None:
        """
        引っかかり・車輪空転状態からの自動脱出（軌跡に沿った後退 ＋ 左右車体振り）
        """
        print(
            f"\n[STUCK DETECTED] Vehicle trapped/spinning wheels "
            f"(motion: {self.stuck_detector.last_motion_value:.2f}) -> Executing Unstuck Action!"
        )
        self._stop_quietly()

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

        # Step 1: まず直前軌跡を逆再生して後退 (1000ms)
        print("[UNSTUCK: STEP 1] Reversing along trajectory away from trap...")
        self.execute_breadcrumb_reverse(target_duration_ms=1000)

        # Step 2: 左右の車体振り (Wiggle) でタイヤの引っかかり・乗り上げを解消。
        # 前進のみとし、危険があれば Pi が止める。
        print("[UNSTUCK: STEP 2] Wiggling chassis left and right...")
        spd = self.speed_limiter.min_speed
        for steer, dur in [(-35, 350), (35, 350), (35, 450)]:
            if not self._send_move(DriveDirection.FORWARD, spd, steer, dur):
                break
            self.trajectory_tracker.record(spd, steer, dur, time.time())
            self._show_status(spd, steer, "UNSTUCK WIGGLE ACTION", (0, 215, 255))
            time.sleep(dur / 1000.0)

        self._stop_quietly()
        self.stuck_detector.reset()
        print("[UNSTUCK COMPLETED] Resuming autonomous driving.\n")

    def _log_event_once(self, flag: str, trigger: str, note: str) -> None:
        """危険が続く間に同じ介入を何度も記録しない"""
        if getattr(self, flag):
            return
        saved_path = self.buffer.save_event(
            trigger=trigger,
            output_base_dir=self.log_dir,
            model_name=self.model_path,
            note=note,
        )
        if saved_path:
            print(f"\n[DISENGAGEMENT LOGGED] Saved to: {saved_path}")
        setattr(self, flag, True)

    def _prepare_body(self) -> bool:
        """走行前に Pi の状態を確認し、今いる床で段差を校正する"""
        try:
            status = self.client.get_status_sync()
            if not status.cliff.guard_enabled:
                print("Cliff Guard on Pi: DISABLED (PICARX_DISABLE_CLIFF_GUARD=1)")
                return True
            self.client.calibrate_floor_sync()
            calibration = self.client.get_status_sync().cliff.calibration
            print(f"Cliff Guard on Pi: ACTIVE ({calibration})")
            return True
        except httpx.HTTPStatusError as exc:
            print(f"[START REFUSED] Floor calibration failed: {exc.response.text}")
        except httpx.HTTPError as exc:
            print(f"[START REFUSED] Cannot reach the Pi: {exc}")
        return False

    def run(self):
        print("\n=== PiCar-X Remote Autonomous Driving Started (Indoor Home) ===")
        print(f"Target Robot: {self.client.base_url}")
        print(
            f"Speed: {self.speed} (Min: {self.speed_limiter.min_speed}) | "
            f"Safe Dist: {self.safe_distance}cm "
            f"(Slowdown: {self.speed_limiter.slowdown_distance}cm)"
        )
        print(f"Reflex on Pi: obstacle, cliff, unknown sensors, TTL {MOVE_TTL_MS}ms")
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
        print("Disengagement Logging: ACTIVE")
        print("Press Ctrl+C or 'Q' in Map Window to stop.\n")

        if not self._prepare_body():
            return

        stream = self.client.open_camera_stream().start()
        last_seq = 0
        obstacle_cycles = 0
        step_counter = 0
        last_loop_time = time.time()

        try:
            while True:
                self.profiler.mark_cycle()
                now = time.time()
                # 実際のループ経過時間。TTL を超えて走り続けることはない
                loop_dt_ms = max(10, min(MOVE_TTL_MS, int((now - last_loop_time) * 1000)))
                last_loop_time = now
                step_counter += 1

                # ----------------------------------------------------
                # 1. カメラ画像取得（新しいフレームが届くまで待つ）
                # ----------------------------------------------------
                with self.profiler.stage("capture"):
                    stream_frame = stream.wait_for_frame(last_seq, timeout=FRAME_TIMEOUT_SEC)
                    frame = None
                    if stream_frame is not None:
                        last_seq = stream_frame.seq
                        frame = cv2.imdecode(
                            np.frombuffer(stream_frame.jpeg, np.uint8), cv2.IMREAD_COLOR
                        )
                if frame is None:
                    # 指令を送らなければ、車両は TTL で止まる
                    self.watchdog.record_error()
                    print(f"\r[NO FRAME] {stream.last_error}", end="", flush=True)
                    continue

                # ----------------------------------------------------
                # 2. Pi の状態確認
                # ----------------------------------------------------
                with self.profiler.stage("status"):
                    try:
                        status = self.client.get_status_sync()
                        self.watchdog.heartbeat()
                        self.is_watchdog_stopped = False
                    except httpx.HTTPError:
                        status = None
                        self.watchdog.record_error()
                if status is None:
                    if self.watchdog.should_stop:
                        self._log_event_once(
                            "is_watchdog_stopped",
                            "communication_timeout",
                            f"Status unavailable (errors={self.watchdog.consecutive_errors})",
                        )
                    continue

                distance = status.sensors.distance.cm
                if self.grid_map:
                    # 反射が戻らない時は、何も見えていないので地図を書き換えない
                    if distance is not None and not status.sensors.distance.far:
                        self.grid_map.update_sensor_ultrasonic(distance)
                    if step_counter % 5 == 0:
                        self.grid_map.decay()

                # ----------------------------------------------------
                # 3. Pi の判定に応じた行動
                # ----------------------------------------------------
                action, reason = decide_action(status)
                if action != BodyAction.ESCAPE_CLIFF:
                    self.is_cliff_stopped = False
                if action != BodyAction.AVOID_OBSTACLE:
                    self.is_obstacle_stopped = False
                    obstacle_cycles = 0

                if action == BodyAction.WAIT:
                    print(f"\r[WAITING] Pi refuses to move: {reason.value}", end="", flush=True)
                    time.sleep(WAIT_SLEEP_SEC)
                    continue

                if action == BodyAction.ESCAPE_CLIFF:
                    floor = status.sensors.floor.values
                    print(f"\r[CLIFF] Floor sensors {floor}", end="", flush=True)
                    self._log_event_once(
                        "is_cliff_stopped",
                        "cliff_detected",
                        f"Pi reported a drop-off, floor sensor values: {floor}",
                    )
                    if not self.execute_breadcrumb_reverse(
                        target_duration_ms=self.reverse_duration_ms,
                        last_cliff_sensors=cliff_sensors(status),
                    ):
                        time.sleep(WAIT_SLEEP_SEC)
                    continue

                if action == BodyAction.AVOID_OBSTACLE:
                    obstacle_cycles += 1
                    print(f"\r[OBSTACLE] {distance}cm", end="", flush=True)
                    self._log_event_once(
                        "is_obstacle_stopped",
                        "obstacle_stop",
                        f"Pi reported an obstacle at {distance}cm",
                    )
                    if obstacle_cycles >= OBSTACLE_ESCAPE_CYCLES:
                        if not self._reverse_refused:
                            print("\n[OBSTACLE DEAD-END] Executing reverse escape.")
                        self.execute_breadcrumb_reverse(
                            target_duration_ms=self.reverse_duration_ms
                        )
                        obstacle_cycles = 0
                    continue

                # ----------------------------------------------------
                # 4. Phase 3.6: スタック（引っかかり・空転）検知 & 自動脱出
                # ----------------------------------------------------
                if self.enable_stuck_detection:
                    if self.stuck_detector.update(
                        frame, motor_speed=self.speed_limiter.min_speed
                    ):
                        self.execute_unstuck()
                        continue

                # ----------------------------------------------------
                # 5. 前処理 & ONNX推論
                # ----------------------------------------------------
                with self.profiler.stage("inference"):
                    input_tensor = self.preprocess(frame)
                    outputs = self.session.run(None, {self.input_name: input_tensor})
                steering_norm = float(outputs[0][0][0])  # -1.0 ~ 1.0

                # ----------------------------------------------------
                # 6. Phase 3.2: 動的車速制御 (Speed Limiter)
                # ----------------------------------------------------
                dynamic_speed = self.speed_limiter.calculate(
                    steering_norm=steering_norm,
                    obstacle_distance_cm=distance,
                )
                steer_angle = int(steering_norm * self.max_steering_angle)

                # 直前リングバッファに保存
                self.buffer.append(
                    frame_bgr=frame,
                    steering_norm=steering_norm,
                    steer_angle=steer_angle,
                    ultrasonic_dist=distance if distance is not None else -1.0,
                    speed=dynamic_speed,
                )

                # ----------------------------------------------------
                # 7. 実機制御 & 軌跡記録 (Phase 3.4 Breadcrumb Tracker)
                # ----------------------------------------------------
                with self.profiler.stage("command"):
                    try:
                        self.client.steer_sync(steer_angle)
                        moved = self.client.move_sync(
                            DriveDirection.FORWARD, speed=dynamic_speed, ttl_ms=MOVE_TTL_MS
                        )
                        self.watchdog.heartbeat()
                    except httpx.HTTPError:
                        moved = False
                        self.watchdog.record_error()

                # Pi が受け付けた移動だけを軌跡と地図に反映する
                if moved:
                    self.trajectory_tracker.record(
                        speed=dynamic_speed,
                        steer_angle=steer_angle,
                        duration_ms=loop_dt_ms,
                        timestamp=now,
                    )
                    if self.grid_map:
                        self.grid_map.update_odometry(
                            speed=dynamic_speed,
                            steer_angle_deg=steer_angle,
                            duration_ms=loop_dt_ms,
                        )

                # ----------------------------------------------------
                # 8. Phase 3.5: マッピングUI リアルタイム描画
                # ----------------------------------------------------
                if self.visualizer:
                    with self.profiler.stage("display"):
                        canvas = self.visualizer.render(
                            camera_frame_bgr=frame,
                            speed=dynamic_speed,
                            steer_angle=steer_angle,
                            ultrasonic_dist=distance if distance is not None else -1.0,
                            motion_value=self.stuck_detector.last_motion_value,
                            status_text="NORMAL DRIVING",
                            status_color=(0, 220, 0),
                        )
                        key = self.visualizer.show(canvas, wait_ms=1)
                    if key == ord("q") or key == ord("Q"):
                        print("\n[STOP] User pressed 'Q' in Map Window.")
                        break

                motion_val = self.stuck_detector.last_motion_value
                print(
                    f"\rSteer: {steering_norm:+.2f} | Angle: {steer_angle:+02d}° | "
                    f"Spd: {dynamic_speed:02d} | Flow: {motion_val:04.1f} | Dist: {distance}cm",
                    end="",
                    flush=True,
                )

        except KeyboardInterrupt:
            print("\n[MANUAL INTERVENTION] Stop requested by user.")
            saved_path = self.buffer.save_event(
                trigger="manual_intervention",
                output_base_dir=self.log_dir,
                model_name=self.model_path,
                note="Interrupted by Ctrl+C (human intervention to prevent collision/stuck)",
            )
            if saved_path:
                print(f"[DISENGAGEMENT LOGGED] Pre-event frames saved to: {saved_path}")
        finally:
            stream.stop()
            if self.visualizer:
                self.visualizer.close()
            try:
                self.client.stop_sync()
                self.client.steer_sync(0)
            except httpx.HTTPError:
                print("Could not reach the Pi. The car stops on its own when the TTL expires.")
            print("Vehicle stopped.")
            print(self.profiler.format_table())
            print(f"Cycle profile: {self.profiler.save(self.log_dir.parent / 'cycle_profiles')}")


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
