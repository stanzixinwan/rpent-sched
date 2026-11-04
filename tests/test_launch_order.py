"""Launch-order rules shared by the sweep and the simulator."""

from __future__ import annotations

import json

import pytest

from rsched.intent import Remaining, plan_work
from rsched.launch_order import jit_pick, load_program, plan_estimate, plan_name, static_order


def test_static_orders():
    est = [5, 1, 9, 3, 7, 2]
    assert static_order(est, "given", 2) == [0, 1, 2, 3, 4, 5]
    assert static_order(est, "lpt", 2) == [2, 4, 0, 3, 5, 1]
    assert static_order(est, "spt", 2) == [1, 5, 3, 0, 4, 2]
    assert static_order(est, "front", 2) == [2, 4, 0, 1, 3, 5]
    assert static_order(est, "front_back", 2) == [2, 4, 0, 3, 5, 1]
    with pytest.raises(ValueError):
        static_order(est, "jit_lpt", 2)


def test_jit_pick_waits_until_the_long_one_threatens_the_end():
    # Lots of short work pending: starting the long one now is not needed yet.
    pending = [10.0] * 20 + [60.0]
    assert jit_pick(0.0, pending, [], [], n=2, slack=1.0, prior_scale=1.0) == 0
    # Little left: the long one would be the tail, so it goes now.
    assert jit_pick(0.0, [10.0, 60.0], [], [], n=2, slack=1.0, prior_scale=1.0) == 1
    # Work still owed by a running episode pushes the projected end out: wait.
    assert jit_pick(0.0, [10.0, 10.0, 60.0], [], [], n=2, prior_scale=1.0) == 2
    assert jit_pick(0.0, [10.0, 10.0, 60.0], [(200.0, 0.0)], [], n=2, prior_scale=1.0) == 0
    # The wall/plan ratio comes from finished episodes and scales that owed work.
    assert jit_pick(100.0, [10.0, 10.0, 60.0], [(100.0, 0.0)], [(10.0, 10.0)], n=2) == 2
    assert jit_pick(100.0, [10.0, 10.0, 60.0], [(100.0, 0.0)], [(10.0, 30.0)], n=2) == 0
    with pytest.raises(ValueError):
        jit_pick(0.0, [], [], [], n=2)


def test_plan_estimate_reads_the_plan_files(tmp_path):
    plan = {"plan": [{"action": "move_to", "arguments": {"xyz": [0, 0, 1]}},
                     {"action": "pi0_pick", "arguments": {"prompt": "pick up the bowl", "max_chunks": 10}},
                     {"action": "set_gripper", "arguments": {"gripper": 1}}]}
    anchors = {"anchors": [{"phrase": "the bowl", "locator": "molmo", "median_xy": [0.1, 0.2]}]}
    assert plan_name("libero_10_swap", 3) == "10_swap_t3"
    (tmp_path / "10_swap_t3_plan.json").write_text(json.dumps(plan))
    (tmp_path / "10_swap_t3_anchors.json").write_text(json.dumps(anchors))
    expected = Remaining(plan_work(load_program(tmp_path, "10_swap_t3"))).snapshot()["intent_s"]
    assert plan_estimate(tmp_path, "libero_10_swap", 3) == pytest.approx(expected)
    # 10 chunks, 3 Molmo calls (survey, refine, held), move_to x2 (plan + refine), set_gripper
    assert expected == pytest.approx(10 * 0.64 + 3 * 1.1 + 2 * 4.4 + 1.0)
