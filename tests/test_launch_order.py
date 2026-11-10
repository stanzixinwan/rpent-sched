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


def test_launch_estimates_history_hybrid_and_cold():
    from rsched.launch_order import cold_tasks, launch_estimates

    plan = {("libero_a", 0): 10.0, ("libero_a", 1): 20.0, ("libero_b", 0): 30.0, ("libero_b", 1): 40.0}
    history = {"libero_a_t0": 100.0, "libero_a_t1": 400.0, "libero_b_t0": 300.0}  # b_t1 never run
    assert launch_estimates(plan, history, "plan") == plan

    h = launch_estimates(plan, history, "history")
    assert sum(h.values()) == pytest.approx(sum(plan.values()))  # rescaled to the plan's total
    # history order wins where known (a_t1 > b_t0 > a_t0), the unknown task gets the known mean
    assert h[("libero_a", 1)] > h[("libero_b", 0)] > h[("libero_a", 0)]
    assert h[("libero_b", 1)] == pytest.approx(h[("libero_a", 0)] * (800 / 3) / 100)

    y = launch_estimates(plan, history, "hybrid")
    # unknown task: its plan in history's units (ratio 800 / 60), then the common rescale
    raw = {**{t: history[f"{t[0]}_t{t[1]}"] for t in plan if t != ("libero_b", 1)},
           ("libero_b", 1): 40.0 * 800 / 60}
    scale = sum(plan.values()) / sum(raw.values())
    assert y[("libero_b", 1)] == pytest.approx(raw[("libero_b", 1)] * scale)

    cold = cold_tasks(["libero_a_t0", "libero_a_t1", "libero_b_t0", "libero_b_t1"], 0.5, seed=3)
    assert len(cold) == 2 and cold == cold_tasks(["libero_b_t1", "libero_b_t0", "libero_a_t1", "libero_a_t0"], 0.5, 3)
    all_cold = launch_estimates(plan, history, "history", cold=set(f"{s}_t{t}" for s, t in plan))
    assert len(set(round(v, 9) for v in all_cold.values())) == 1  # nothing known: no information
    with pytest.raises(ValueError):
        launch_estimates(plan, None, "history")


def test_history_table_from_a_level(tmp_path):
    from rsched.history_table import build
    from rsched.launch_order import load_history

    m = {"specs": [{"episode_id": "e0", "suite": "libero_a", "task": 0, "seed": 0},
                   {"episode_id": "e1", "suite": "libero_a", "task": 0, "seed": 1},
                   {"episode_id": "e2", "suite": "libero_b", "task": 2, "seed": 0},
                   {"episode_id": "e3", "suite": "libero_b", "task": 3, "seed": 0}],
         "episodes": [{"episode": "e0", "returncode": 0, "wall_s": 10.0, "tool_calls": 3},
                      {"episode": "e1", "returncode": 1, "wall_s": 30.0, "tool_calls": 5},
                      {"episode": "e2", "returncode": "timeout", "wall_s": 3600.0, "tool_calls": 2},
                      {"episode": "e3", "returncode": 1, "wall_s": 5.0, "tool_calls": 0}]}
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    table = build([tmp_path])
    assert table["tasks"] == {"libero_a_t0": 20.0} and table["skipped"] == 2
    (tmp_path / "h.json").write_text(json.dumps(table))
    assert load_history(tmp_path / "h.json") == {"libero_a_t0": 20.0}
