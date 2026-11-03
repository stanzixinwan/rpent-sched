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
