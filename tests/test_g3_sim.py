"""Trace-driven G3 simulator on hand-made episodes."""

from __future__ import annotations

import sys

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT / "analysis"))
from g3_sim import Episode, load_library, save_library, simulate  # noqa: E402


def ep(key, local, server, service, intent=None, done=None, work_s=1.0):
    n = len(server)
    return Episode(key=key, task=key, seed=0, level=1, mode="none", local=local, server=server,
                   service=service, intent=intent or [0.0] * n, done=done or [0.0] * n, work_s=work_s)


def test_one_at_a_time_has_no_waiting():
    eps = [ep("a", [1, 2, 3], [0, 1], [0.5, 0.5]), ep("b", [0.1, 0], [0], [4.0])]
    r = simulate(eps, 1, "fifo")
    assert r["wait_mean_s"] == 0
    assert r["makespan_s"] == pytest.approx(sum(e.work_time for e in eps))


def test_orders_one_queue_by_policy():
    # "hold" occupies the VLA from t=0 to 2; "long" then "short" queue behind it.
    hold = ep("hold", [0, 0], [0], [2.0], intent=[1.0])
    long = ep("long", [0.1, 0], [0], [10.0], intent=[50.0])
    short = ep("short", [0.2, 0], [0], [1.0], intent=[5.0])
    eps = [hold, long, short]
    fifo = simulate(eps, 3, "fifo")
    assert fifo["wall_mean_s"] == pytest.approx((2 + 12 + 13) / 3)  # long then short
    for policy in ("intent_srpt", "oracle_srpt"):
        assert simulate(eps, 3, policy)["wall_mean_s"] == pytest.approx((2 + 13 + 3) / 3)
    assert simulate(eps, 3, "oracle_lrpt")["wall_mean_s"] == pytest.approx(fifo["wall_mean_s"])
    assert fifo["makespan_s"] == pytest.approx(13.0)
    assert fifo["vla_util"] == pytest.approx(1.0) and fifo["molmo_util"] == 0


def test_history_is_fifo_until_an_episode_finishes():
    first = ep("first", [0, 0], [0], [3.0], done=[0.0], work_s=10.0)
    second = ep("second", [0, 0], [0], [1.0], done=[0.0])
    third = ep("third", [0.5, 0], [0], [1.0], done=[9.0])  # arrives second, nearly done by history
    # No episode has finished when the second and third queue: FIFO between them.
    r = simulate([first, second, third], 3, "history_srpt")
    assert r["makespan_s"] == pytest.approx(5.0)
    assert r["wait_mean_s"] == pytest.approx((0 + 3 + 3.5) / 3)


def test_true_remaining_time_and_round_trip(tmp_path):
    e = ep("a", [1, 2, 3], [0, 1], [0.5, 0.25])
    assert e.rest == [0.5 + 2 + 0.25 + 3, 0.25 + 3]
    save_library([e], tmp_path / "lib.json.gz")
    (back,) = load_library(tmp_path / "lib.json.gz")
    assert back.rest == e.rest and back.local == e.local


def test_launch_orders():
    from g3_sim import launch_order

    eps = [ep(f"e{i}", [w, 0], [0], [0.1]) for i, w in enumerate([5, 1, 9, 3, 7, 2])]
    for e, plan in zip(eps, [5, 1, 9, 3, 7, 2]):
        e.plan_s = float(plan)
    names = lambda xs: [e.key for e in xs]  # noqa: E731
    assert names(launch_order(eps, "lpt", 2)) == ["e2", "e4", "e0", "e3", "e5", "e1"]
    assert names(launch_order(eps, "spt", 2)) == ["e1", "e5", "e3", "e0", "e4", "e2"]
    # two longest first, two shortest last (shortest at the very end), the rest as listed
    assert names(launch_order(eps, "front_back", 2)) == ["e2", "e4", "e0", "e3", "e5", "e1"][:2] + ["e0", "e3"] + ["e5", "e1"]
    assert names(launch_order(eps, "given", 2)) == names(eps)


def test_jit_lpt_keeps_the_long_episode_off_the_tail():
    from g3_sim import jit_lpt

    # Twelve short episodes listed before one long one, two slots: listed order
    # starts the long one last and it becomes the tail.
    eps = [ep(f"s{i}", [10, 0], [0], [0.1]) for i in range(12)] + [ep("long", [60, 0], [0], [0.1])]
    for e in eps:
        e.plan_s = e.work_time
    given = simulate(eps, 2, "fifo")["makespan_s"]
    jit = simulate(eps, 2, "fifo", pick=jit_lpt())["makespan_s"]
    assert jit < given
    assert jit == pytest.approx((12 * 10.1 + 60.1) / 2, rel=0.15)  # close to perfectly balanced
