"""Remaining-work counting from a Flash program, history estimates, and priorities."""

from __future__ import annotations

import pytest
import yaml
from conftest import ROOT

from rsched.intent import (
    DEFAULT_COSTS,
    HistoryEstimate,
    Remaining,
    plan_work,
    priority,
)
from rsched.sweep import scheduling

# Two Molmo anchors and one segment anchor; one planned segment step (skipped by
# replay), two picks (one with the default budget), one set_gripper.
PROGRAM = {
    "plan": [
        {"action": "segment", "arguments": {"prompt": "bowl"}},
        {"action": "move_to", "arguments": {"xyz": [0, 0, 1]}, "anchor": "bowl"},
        {"action": "pi0_pick", "arguments": {"prompt": "pick up the bowl", "max_chunks": 12}},
        {"action": "set_gripper", "arguments": {"gripper": 1}},
        {"action": "move_to", "arguments": {"xyz": [0, 0, 1]}},
        {"action": "release", "arguments": {}},
        {"action": "pi0_pick", "arguments": {"prompt": "pick up the plate"}},
    ],
    "reference": {"bowl": None, "plate": None, "drawer": None},
    "locator_of": {"bowl": "molmo", "plate": "molmo", "drawer": "segment"},
}
COSTS = {"chunk_s": 1.0, "molmo_s": 10.0, "move_to": 100.0, "release": 1000.0,
         "set_gripper": 0.0, "segment": 0.5}


def test_plan_work_counts_what_replay_runs():
    w = plan_work(PROGRAM)
    assert w.pi0 == [("pi0_pick", "pick up the bowl", 12), ("pi0_pick", "pick up the plate", 24)]
    assert w.chunks == 36
    assert w.molmo_calls == 2 * 2 + 1  # survey + refine per Molmo anchor, one held grounding
    # survey of the segment anchor, a refine hover per Molmo anchor, plan steps minus segment
    assert dict(w.steps) == {"segment": 1, "move_to": 2 + 2, "set_gripper": 1, "release": 1}


def test_remaining_counts_down_and_handles_retries():
    r = Remaining(plan_work(PROGRAM), COSTS)
    s0 = r.snapshot()
    assert s0["chunks_left"] == 36 and s0["molmo_left"] == 5 and s0["steps_left"] == 7
    assert s0["intent_s"] == 36 * 1 + 5 * 10 + 4 * 100 + 1000 + 0.5

    r.rpc_done("molmo.ground")
    r.tool_start("move_to", {})
    assert r.snapshot()["molmo_left"] == 4 and r.snapshot()["steps_left"] == 6

    r.tool_start("pi0_pick", {"prompt": "pick up the bowl", "max_chunks": 12})
    for _ in range(5):
        r.rpc_done("vla.predict")
    assert r.snapshot()["chunks_left"] == 24 + (12 - 5)
    r.tool_end("pi0_pick")
    assert r.snapshot()["chunks_left"] == 24  # the unused budget is gone once the pick returns

    # Replay retries a failed pick with the same prompt: extra work, plan untouched.
    r.tool_start("pi0_pick", {"prompt": "pick up the bowl", "max_chunks": 12})
    assert r.snapshot()["chunks_left"] == 24 + 12
    r.tool_end("pi0_pick")
    r.tool_start("pi0_pick", {"prompt": "pick up the plate", "max_chunks": 24})
    assert r.snapshot()["chunks_left"] == 24
    r.tool_end("pi0_pick")
    assert r.snapshot()["chunks_left"] == 0

    # Steps beyond the plan (replayed approach moves) never go negative.
    for _ in range(10):
        r.tool_start("move_to", {})
    assert r.snapshot()["steps_left"] >= 0
    assert r.done["chunks"] == 5 and r.done["molmo"] == 1


def test_history_estimate_and_priority_signs(tmp_path):
    r = Remaining(plan_work(PROGRAM), COSTS)
    hist = HistoryEstimate(tmp_path / "history.jsonl")
    assert hist.remaining_s(0.0) is None
    assert priority("history_srpt", r, hist) is None  # nothing finished yet: FIFO
    assert priority("none", r, hist) is None

    hist.record("a", 300.0)
    hist.record("b", 500.0)
    r.rpc_done("molmo.ground")  # 10 s done
    assert hist.remaining_s(r.done_s()) == pytest.approx(400.0 - 10.0)
    assert priority("history_srpt", r, hist) == pytest.approx(390.0)
    assert priority("history_lrpt", r, hist) == pytest.approx(-390.0)
    assert priority("intent_srpt", r, hist) == pytest.approx(r.intent_s())
    assert priority("intent_lrpt", r, hist) == pytest.approx(-r.intent_s())
    with pytest.raises(ValueError):
        priority("bogus", r, hist)


def test_default_costs_cover_every_replayed_primitive():
    for name in ("move_to", "move_pose", "release", "set_gripper", "segment", "rotate_wrist", "rotate_pitch"):
        assert name in DEFAULT_COSTS


def test_g3_config_and_scheduling_validation():
    cfg = yaml.safe_load((ROOT / "configs" / "g3.yaml").read_text())
    assert scheduling(cfg)["mode"] == "none"
    eps = cfg["episodes"]["list"]
    assert len(eps) == cfg["episodes_per_level"] == 32
    assert len({(e["suite"], e["task"], e["seed"]) for e in eps}) == 32
    assert len({(e["suite"], e["task"]) for e in eps}) == 16
    assert cfg["concurrency"] == [8, 16]
    for mode in ("intent_srpt", "intent_lrpt", "history_srpt", "history_lrpt"):
        assert scheduling({**cfg, "scheduling": {"mode": mode}})["mode"] == mode
    fifo_proxies = {**cfg, "services": {**cfg["services"], "vla": {"policy": "fifo"}}}
    with pytest.raises(ValueError, match="priority"):
        scheduling({**fifo_proxies, "scheduling": {"mode": "intent_srpt"}})
    with pytest.raises(ValueError):
        scheduling({**cfg, "scheduling": {"mode": "sjf"}})
