"""障害物を避ける判断: 曲がり始め、向き、曲がり終え、切り替え、減速"""

import pytest

from src.e2e.avoidance import AVOID, CRUISE, AvoidanceConfig, AvoidancePolicy


def run(policy, frames, start=0.0, dt=0.1, upper="any"):
    """frames: (distance_cm, blocked_probability) の並び。決定の並びを返す"""
    decisions = []
    for i, (distance, probability) in enumerate(frames):
        decisions.append(policy.step(start + i * dt, distance, probability, upper))
    return decisions


def test_goes_straight_with_the_upper_intent_while_clear():
    policy = AvoidancePolicy()
    decisions = run(policy, [(None, 0.1), (150.0, 0.2)], upper="left")
    assert [d.intent for d in decisions] == ["left", "left"]
    assert all(d.state == CRUISE and not d.slow for d in decisions)


def test_turns_right_first_when_the_ultrasonic_is_near():
    policy = AvoidancePolicy()
    decision = policy.step(0.0, 45.0, None)
    assert (decision.intent, decision.state, decision.event) == ("right", AVOID, "begin")
    assert decision.reason == "ultrasonic"


def test_upper_intent_chooses_the_side():
    policy = AvoidancePolicy()
    assert policy.step(0.0, 30.0, None, "left").intent == "left"


def test_camera_needs_consecutive_frames_to_start_a_turn():
    policy = AvoidancePolicy()
    first = policy.step(0.0, None, 0.9)
    assert first.state == CRUISE and first.slow  # まだ曲がらないが、速度は落とす
    second = policy.step(0.1, None, 0.95)
    assert (second.state, second.reason, second.intent) == (AVOID, "camera", "right")


def test_one_noisy_camera_frame_does_not_start_a_turn():
    policy = AvoidancePolicy()
    decisions = run(policy, [(None, 0.9), (None, 0.1), (None, 0.9), (None, 0.1)])
    assert all(d.state == CRUISE for d in decisions)


def test_finishes_the_turn_after_clear_frames_and_the_minimum_time():
    policy = AvoidancePolicy()
    policy.step(0.0, 40.0, 0.8)
    # 空いても 1.0 秒たつまでは曲がり続ける
    early = run(policy, [(None, 0.1)] * 5, start=0.1)
    assert all(d.state == AVOID and d.intent == "right" for d in early)
    done = policy.step(1.0, None, 0.1)
    assert (done.state, done.event, done.intent) == (CRUISE, "success", "any")
    assert policy.failures == 0


def test_camera_must_also_be_clear_to_finish():
    policy = AvoidancePolicy()
    policy.step(0.0, 40.0, None)
    # 超音波は遠いが、カメラはまだ塞がっていそう（細い脚など）
    decisions = run(policy, [(None, 0.5)] * 20, start=0.1)
    assert all(d.state == AVOID for d in decisions[:-1])


def test_switches_side_when_the_turn_takes_too_long():
    policy = AvoidancePolicy()
    policy.step(0.0, 40.0, None)
    decision = policy.step(4.1, 40.0, None)
    assert (decision.intent, decision.event, decision.reason) == ("left", "switch", "timeout")
    # 曲がれたら、次の曲がりはまた右から
    for i in range(4):
        decision = policy.step(5.2 + i * 0.1, None, None)
    assert decision.event == "success"
    assert policy.step(10.0, 30.0, None).intent == "right"


def test_stopped_by_an_obstacle_switches_side():
    policy = AvoidancePolicy()
    policy.step(0.0, 40.0, None)
    decision = policy.on_stopped_by_obstacle(0.5)
    assert (decision.intent, decision.event, decision.slow) == ("left", "switch", True)
    assert policy.failures == 1


def test_stopped_while_cruising_starts_a_turn():
    policy = AvoidancePolicy()
    decision = policy.on_stopped_by_obstacle(0.0)
    assert (decision.state, decision.intent, decision.reason) == (AVOID, "right", "stopped")


def test_without_the_camera_output_only_the_ultrasonic_counts():
    policy = AvoidancePolicy()
    decisions = run(policy, [(None, None)] * 5)
    assert all(d.state == CRUISE and not d.slow for d in decisions)


def test_config_is_checked():
    with pytest.raises(ValueError):
        AvoidanceConfig(first_side="any")
    with pytest.raises(ValueError):
        AvoidanceConfig(blocked_enter=0.3, blocked_exit=0.5)
    with pytest.raises(ValueError):
        AvoidancePolicy().step(0.0, None, None, "straight")
