"""M3: 指令と実際の動きの食い違いの検出"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.e2e.motion.analyze import LEGACY_SPEED, analyze_session, command_from_record
from src.e2e.motion.checker import (
    NO_MOTION,
    TURN_MISMATCH,
    UNCOMMANDED_MOTION,
    CheckerConfig,
    Command,
    MotionChecker,
)
from src.e2e.motion.estimator import (
    DEFAULT_HFOV_DEG,
    WORK_WIDTH,
    MotionSample,
    MotionTracker,
    estimate_flow,
)
from src.e2e.motion.monitor import CommandState, MotionMonitor
from src.e2e.training.dataset import DrivingDataset

# --- 合成画像: 模様のある大きな画像から、ずらしたり拡大したりして切り出す ---

_rng = np.random.default_rng(0)
_BASE = cv2.normalize(
    cv2.GaussianBlur((_rng.random((480, 640)) * 255).astype(np.uint8), (0, 0), 3),
    None,
    0,
    255,
    cv2.NORM_MINMAX,
)


def view(shift_px: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """作業解像度で `shift_px` だけ景色を横にずらし、`scale` 倍に拡大した BGR 画像"""
    matrix = cv2.getRotationMatrix2D((320, 240), 0, scale)
    matrix[0, 2] += shift_px * 2  # 切り出し（320 幅）を 160 幅へ縮小するので 2 倍
    warped = cv2.warpAffine(_BASE, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
    small = cv2.resize(warped[120:360, 160:480], (160, 120), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)


BLANK = np.full((120, 160, 3), 128, np.uint8)


# --- 画像の動きの推定 ---


def test_flow_is_still_for_the_same_view():
    flow = estimate_flow(view(), view())
    assert flow.valid
    assert abs(flow.shift_x_px) < 0.1
    assert flow.scale == pytest.approx(1.0, abs=0.005)


def test_flow_measures_a_sideways_shift():
    flow = estimate_flow(view(), view(shift_px=-4))
    assert flow.shift_x_px == pytest.approx(-4.0, abs=0.2)


def test_flow_measures_expansion():
    flow = estimate_flow(view(), view(scale=1.05))
    assert flow.scale == pytest.approx(1.05, abs=0.005)
    assert abs(flow.shift_x_px) < 0.3


def test_flow_is_unknown_without_texture():
    flow = estimate_flow(BLANK, BLANK)
    assert not flow.valid
    assert flow.reason == "few_features"


# --- 時間あたりの推定 ---


def test_tracker_converts_a_shift_into_a_right_turn():
    tracker = MotionTracker(interval_s=0.3)
    assert tracker.update(view(), 0.0) is None  # 比べる過去のフレームがまだない
    sample = tracker.update(view(shift_px=-4), 0.3)

    # 景色が左へずれた = 右へ曲がった
    expected = 4 * DEFAULT_HFOV_DEG / WORK_WIDTH / 0.3
    assert sample.yaw_rate_dps == pytest.approx(expected, rel=0.1)
    assert sample.image_moving


def test_tracker_reports_forward_expansion_and_approach():
    tracker = MotionTracker(interval_s=0.3)
    tracker.update(view(), 0.0, distance_cm=60.0)
    sample = tracker.update(view(scale=1.03), 0.3, distance_cm=57.0)

    assert sample.expansion_rate == pytest.approx(0.1, abs=0.02)
    assert sample.approach_cm_s == pytest.approx(10.0)


def test_tracker_ignores_far_or_jumping_distances():
    tracker = MotionTracker(interval_s=0.3, max_distance_cm=150)
    tracker.update(view(), 0.0, distance_cm=300.0)
    assert tracker.update(view(), 0.3, distance_cm=120.0).approach_cm_s is None

    tracker = MotionTracker(interval_s=0.3)
    tracker.update(view(), 0.0, distance_cm=100.0)
    # 0.3 秒で 60cm は速すぎる。別の物を捉えたとみなす
    assert tracker.update(view(), 0.3, distance_cm=40.0).approach_cm_s is None


def test_tracker_compares_with_the_frame_closest_to_the_interval():
    tracker = MotionTracker(interval_s=0.3)
    for i in range(10):
        sample = tracker.update(view(), i * 0.066)
    assert sample.interval_s == pytest.approx(0.33, abs=0.04)


# --- 食い違いの判定 ---


def sample(**overrides) -> MotionSample:
    values = dict(
        timestamp=10.0,
        interval_s=0.3,
        image_valid=True,
        image_reason=None,
        points=100,
        inlier_ratio=0.9,
        yaw_rate_dps=0.0,
        expansion_rate=0.1,
        edge_flow_px_s=20.0,
        image_moving=True,
        distance_cm=None,
        approach_cm_s=None,
    )
    values.update(overrides)
    return MotionSample(**values)


STILL = dict(edge_flow_px_s=0.5, image_moving=False, expansion_rate=0.0)


def test_moving_as_commanded_is_ok():
    result = MotionChecker().check(Command(60, 0, since_s=2.0), sample())
    assert result.flags == []
    assert result.unknown is None


def test_wheels_spinning_without_motion_is_flagged():
    result = MotionChecker().check(Command(60, 0, since_s=2.0), sample(**STILL))
    assert result.flags == [NO_MOTION]


def test_approaching_wall_without_texture_is_not_flagged():
    # 画像はほとんど動かないが、超音波で近づいているのが分かる
    result = MotionChecker().check(Command(60, 0, since_s=2.0), sample(**STILL, approach_cm_s=12.0))
    assert result.flags == []


def test_no_texture_and_still_distance_is_flagged():
    result = MotionChecker().check(
        Command(60, 0, since_s=2.0),
        sample(image_valid=False, image_reason="few_features", approach_cm_s=0.3),
    )
    assert result.flags == [NO_MOTION]


def test_no_texture_and_no_distance_is_unknown():
    result = MotionChecker().check(
        Command(60, 0, since_s=2.0), sample(image_valid=False, image_reason="few_features")
    )
    assert result.flags == []
    assert result.unknown == "few_features"


def test_right_steering_but_turning_left_is_flagged():
    result = MotionChecker().check(Command(60, 30, since_s=2.0), sample(yaw_rate_dps=-20.0))
    assert result.flags == [TURN_MISMATCH]


def test_reversing_flips_the_expected_turn():
    # 後退で右へ舵を切ると、車の向きは左へ回る
    result = MotionChecker().check(Command(-40, 30, since_s=2.0), sample(yaw_rate_dps=-20.0))
    assert result.flags == []


def test_turning_while_steering_straight_is_flagged():
    result = MotionChecker().check(Command(60, 0, since_s=2.0), sample(yaw_rate_dps=25.0))
    assert result.flags == [TURN_MISMATCH]


def test_calibrated_turn_compares_the_amount():
    config = CheckerConfig(speed_cm_s_per_unit=0.5, wheelbase_cm=10.0)
    checker = MotionChecker(config)
    expected = checker.expected_yaw_rate(Command(60, 20, since_s=2.0))
    assert expected > 0

    ok = checker.check(Command(60, 20, since_s=2.0), sample(yaw_rate_dps=expected * 1.2))
    assert ok.flags == []
    slipping = checker.check(Command(60, 20, since_s=2.0), sample(yaw_rate_dps=expected * 0.2))
    assert slipping.flags == [TURN_MISMATCH]


def test_pushed_while_stopped_is_flagged():
    result = MotionChecker().check(Command(0, 0, since_s=2.0), sample())
    assert result.flags == [UNCOMMANDED_MOTION]


def test_a_person_walking_by_is_not_flagged():
    # 景色の一部だけが動いている（特徴点の多くは止まっている）
    result = MotionChecker().check(Command(0, 0, since_s=2.0), sample(inlier_ratio=0.3))
    assert result.flags == []


def test_camera_pan_is_not_judged():
    result = MotionChecker().check(Command(0, 0, since_s=2.0, camera_moving=True), sample())
    assert result.flags == []
    assert result.unknown == "camera_moving"


def test_just_after_a_command_change_is_not_judged():
    result = MotionChecker().check(Command(60, 0, since_s=0.2), sample(**STILL))
    assert result.flags == []
    assert result.unknown == "settling"


def test_mismatch_is_sustained_after_the_configured_time():
    checker = MotionChecker(CheckerConfig(sustain_s=1.0))
    command = Command(60, 0, since_s=2.0)
    assert checker.check(command, sample(**STILL, timestamp=10.0)).sustained is None
    assert checker.check(command, sample(**STILL, timestamp=10.6)).sustained is None
    assert checker.check(command, sample(**STILL, timestamp=11.1)).sustained == NO_MOTION
    # 解消すると数え直す
    assert checker.check(command, sample(timestamp=11.2)).sustained is None
    assert checker.check(command, sample(**STILL, timestamp=11.3)).sustained is None


# --- 指令の状態 ---


def test_command_state_expires_with_the_ttl():
    state = CommandState(now=0.0)
    state.move(40, ttl_s=0.6, now=1.0)
    assert state.at(1.3).speed == 40
    assert state.at(1.3).since_s == pytest.approx(0.3)

    stopped = state.at(2.0)
    assert stopped.speed == 0
    assert stopped.since_s == pytest.approx(0.4)  # 期限切れ（1.6 秒）から数える


def test_repeated_moves_do_not_reset_the_settle_time():
    state = CommandState(now=0.0)
    state.move(40, 0.6, now=1.0)
    state.move(40, 0.6, now=1.4)
    assert state.at(1.5).since_s == pytest.approx(0.5)


def test_steering_change_resets_the_settle_time():
    state = CommandState(now=0.0)
    state.move(40, 0.6, now=1.0)
    state.steer(20, now=1.4)
    assert state.at(1.5).since_s == pytest.approx(0.1)


def test_camera_move_marks_the_command():
    state = CommandState(now=0.0)
    state.camera_moved(now=1.0, settle_s=0.5)
    assert state.at(1.2).camera_moving
    assert not state.at(1.6).camera_moving


# --- 監視（収集 GUI と自律走行から使う） ---


def test_monitor_reports_a_sustained_mismatch_once():
    reported = []
    monitor = MotionMonitor(
        tracker=MotionTracker(interval_s=0.3),
        config=CheckerConfig(sustain_s=0.6, settle_s=0.0),
        on_sustained=lambda flag, update: reported.append(flag),
    )
    command = Command(60, 0, since_s=5.0)
    frame = view()
    updates = [monitor.update(frame, t * 0.1, None, command) for t in range(20)]

    assert reported == [NO_MOTION]
    record = [u for u in updates if u is not None][-1].to_record()
    assert record["motion_check"]["flags"] == [NO_MOTION]
    assert record["command"]["speed"] == 60
    assert json.dumps(record)  # 記録に書ける


# --- 学習データから外す ---


def write_session(root: Path, records) -> Path:
    session = root / "s1"
    (session / "images").mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i, extra in enumerate(records):
            frame = f"frame_{i:04d}.jpg"
            cv2.imwrite(str(session / "images" / frame), view(shift_px=-i))
            record = {"frame": frame, "timestamp": i * 0.1, "steering": 0.0, "throttle": 0.5}
            record.update(extra)
            f.write(json.dumps(record) + "\n")
    return session


def test_dataset_skips_frames_with_a_mismatch(tmp_path: Path):
    session = write_session(
        tmp_path,
        [{}, {"motion_check": {"flags": [NO_MOTION]}}, {"motion_check": {"flags": []}}],
    )
    dataset = DrivingDataset(session)
    assert len(dataset) == 2
    assert dataset.skipped_motion_mismatch == 1
    assert len(DrivingDataset(session, include_motion_mismatch=True)) == 3


# --- 記録したセッションの解析 ---


def test_command_from_record_handles_new_and_legacy_records():
    assert command_from_record({"command": {"speed": 30, "steering_deg": -10}}) == (
        30,
        -10.0,
        False,
    )
    assert command_from_record({"steering": 0.5, "throttle": 0.5}) == (LEGACY_SPEED, 17.5, True)
    assert command_from_record({"steering": 0.0, "throttle": 0.0})[0] == 0


def test_analyze_session_counts_mismatches(tmp_path: Path):
    stopped = {"command": {"speed": 60, "steering_deg": 0, "since_s": 2.0}}
    session = tmp_path / "s1"
    (session / "images").mkdir(parents=True)
    with open(session / "labels.jsonl", "w", encoding="utf-8") as f:
        for i in range(8):
            frame = f"frame_{i:04d}.jpg"
            cv2.imwrite(str(session / "images" / frame), view())  # 景色が変わらない
            f.write(json.dumps({"frame": frame, "timestamp": i * 0.1, **stopped}) + "\n")

    results, counts = analyze_session(session, MotionTracker(interval_s=0.3))

    assert counts[NO_MOTION] == counts["frames"] > 0
    assert counts["legacy_commands"] == 0
    assert all(r["check"]["flags"] == [NO_MOTION] for r in results)


def test_turn_is_not_judged_from_an_unsure_estimate():
    # 白い壁の前など、模様の点が少ない時の向きの推定は信じない
    config = CheckerConfig(turn_min_points=60, turn_min_inliers=0.6)
    unsure = sample(points=30, inlier_ratio=0.4, yaw_rate_dps=-35.0)
    result = MotionChecker(config).check(Command(50, 0, since_s=2.0), unsure)
    assert result.flags == [] and result.unknown == "turn_unsure"
    sure = sample(points=100, inlier_ratio=0.8, yaw_rate_dps=-35.0)
    assert MotionChecker(config).check(Command(50, 0, since_s=2.0), sure).flags == ["turn_mismatch"]
