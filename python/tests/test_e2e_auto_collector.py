"""決まったルールで走る収集（`src.e2e.auto_collector`）の走り方"""

import random
from collections import deque

from src.e2e.auto_collector import (
    LEFT,
    RIGHT,
    AttemptBuffer,
    Observation,
    PolicyConfig,
    RulePolicy,
    ramp,
    smoothed_distance,
)


def policy(**overrides):
    config = PolicyConfig(**{"intent_at_obstacle": 0.0, "open_turn_per_s": 0.0, **overrides})
    return RulePolicy(config=config, rng=random.Random(0))


def run(p, distances, start=0.0, dt=0.1, blocked_at=()):
    actions = []
    for i, d in enumerate(distances):
        actions.append(p.step(Observation(start + i * dt, d, blocked=i in blocked_at)))
    return actions


def test_open_floor_goes_straight_with_any():
    actions = run(policy(), [None] * 30)
    assert all(a.drive == 1 and a.target_deg == 0 and a.intent == "any" for a in actions)


def test_without_intent_it_tries_right_first_and_finishes_when_clear():
    p = policy()
    actions = run(p, [200, 120, 80, 45, 45, 45] + [None] * 12)
    begin = next(a for a in actions if a.attempt == "begin")
    assert begin.target_deg == 30 and begin.intent == "any"
    assert actions[-1].state == "cruise" or any(a.attempt == "success" for a in actions)
    assert any(a.attempt == "success" for a in actions)


def test_failed_turn_backs_up_and_tries_the_other_side():
    p = policy(avoid_max_s=0.5, backup_s=0.3, pivot_s=0.3)
    actions = run(p, [40] * 20)
    kinds = [a.attempt for a in actions if a.attempt]
    assert kinds[:3] == ["begin", "fail", "begin"]
    fail = next(a for a in actions if a.attempt == "fail")
    assert fail.drive == -1
    second = [a for a in actions if a.attempt == "begin"][1]
    assert second.target_deg == -30  # 右が駄目だったので左
    # 下がった後、止まったまま舵を左へ切り替えてから進む
    pivot = [a for a in actions if a.state == "pivot"]
    assert pivot and all(a.drive == 0 and a.target_deg == -30 for a in pivot)


def test_blocked_while_turning_is_a_failure():
    p = policy()
    actions = run(p, [40, 40, 40], blocked_at={2})
    assert actions[2].attempt == "fail"


def test_intent_is_set_before_turning_near_an_obstacle():
    p = policy(intent_at_obstacle=1.0)
    actions = run(p, [200, 95, 90, 80, 70, 60, 45, 45])
    approach = [a for a in actions if a.state == "approach"]
    assert approach and all(a.target_deg == 0 for a in approach)  # 意図を付けてもまだまっすぐ
    intent = approach[0].intent
    assert intent in ("left", "right")
    begin = next(a for a in actions if a.attempt == "begin")
    assert begin.intent == intent
    assert begin.target_deg == (30 if intent == "right" else -30)


def test_open_space_turn_has_a_lead_then_turns_then_releases_the_intent():
    p = policy(open_turn_per_s=100.0, cruise_min_s=0.0, lead_s=0.5, open_turn_s=0.5)
    actions = run(p, [None] * 20)
    states = [a.state for a in actions]
    lead = states.index("lead")
    turn = states.index("open_turn")
    assert turn - lead >= 4  # 意図を付けてから曲がるまで 0.5 秒まっすぐ
    assert actions[lead].intent == actions[turn].intent != "any"
    after = turn + states[turn:].index("cruise")
    assert actions[after].intent == "any"


def test_intent_sides_are_balanced():
    p = policy()
    p.intent_counts = {"left": 10, "right": 3}
    assert p._balanced_side() == RIGHT
    p.intent_counts = {"left": 3, "right": 10}
    assert p._balanced_side() == LEFT


def test_repeated_failures_stop_the_session():
    p = policy(avoid_max_s=0.2, backup_s=0.1, pivot_s=0.1, max_failures=2)
    actions = run(p, [40] * 40)
    assert any(a.done for a in actions)


def test_attempt_buffer_marks_failed_turns_and_keeps_order():
    written = []
    buffer = AttemptBuffer(written.append)
    buffer.add({"i": 0})
    buffer.begin()
    buffer.add({"i": 1})
    buffer.resolve(success=False)
    buffer.begin()
    buffer.add({"i": 2})
    buffer.resolve(success=True)
    buffer.begin()
    buffer.add({"i": 3})
    buffer.flush()  # 結果が分からないまま終わった
    assert [r["i"] for r in written] == [0, 1, 2, 3]
    assert [r.get("exclude", False) for r in written] == [False, True, False, True]


def test_ramp_and_smoothed_distance():
    assert ramp(0, 30, 60, 0.1) == 6
    assert ramp(10, 0, 60, 1.0) == 0
    assert smoothed_distance(deque([40.0, 300.0, 42.0])) == 42.0
    assert smoothed_distance(deque([None, None, 42.0])) is None


def test_a_turn_is_not_finished_by_a_lost_echo_right_after_it_starts():
    p = policy()
    actions = run(p, [45, None, None, None, None, None, None, None])
    assert not any(a.attempt == "success" for a in actions)  # 0.7 秒では終えない
    actions = run(p, [None] * 6, start=0.8)
    assert any(a.attempt == "success" for a in actions)


def test_one_far_reading_does_not_drop_the_intent_while_approaching():
    p = policy(intent_at_obstacle=1.0)
    actions = run(p, [95, 90, None, 85, 80])
    assert all(a.state == "approach" for a in actions[1:])


def test_stall_detector_needs_a_still_image_for_a_while_after_the_start():
    from src.e2e.auto_collector import StallDetector

    s = StallDetector(stalled_s=1.0, grace_s=0.5)
    assert not s.update(0.0, 0.2, "still")  # 動き出しの途中
    assert not s.update(0.5, 0.7, "still")
    assert not s.update(1.0, 1.2, "dark")  # ふさがれた暗い画像も止まっているに数える
    assert s.update(1.6, 1.8, "dark")
    assert not s.update(1.7, 1.9, "moving")
    assert not s.update(2.0, None, "still")  # 前進していなければ数えない
    # 白い壁のように模様が少ないだけの時は数えない
    s = StallDetector(stalled_s=1.0, grace_s=0.0)
    assert not any(s.update(t * 0.5, 1.0 + t, "unknown") for t in range(10))


def test_view_and_straight_drift():
    from types import SimpleNamespace

    from src.e2e.auto_collector import straight_drift, view_of

    assert view_of(None, 10.0) == "dark"
    assert view_of(None, 120.0) == "unknown"
    assert view_of(SimpleNamespace(image_valid=True, image_moving=False), 120.0) == "still"
    good = {
        "command": {"steering_deg": 0.0, "speed": 50, "since_s": 2.0},
        "motion": {"image_valid": True, "points": 120, "inlier_ratio": 0.8, "yaw_rate_dps": -3.0},
    }
    unsure = {**good, "motion": {**good["motion"], "points": 20, "yaw_rate_dps": -30.0}}
    turning = {**good, "command": {**good["command"], "steering_deg": 20.0}}
    assert straight_drift([good, good, unsure, turning]) == {"samples": 2, "yaw_dps": -3.0}
