"""Unit tests for the proxy's dispatch gate (no network)."""

from __future__ import annotations

import threading
import time

import pytest

from rsched.proxy import Gate


def run_clients(gate: Gate, priorities: list[float], hold_s: float = 0.02) -> list[int]:
    """Start one blocked holder, queue the others, return dispatch order of the queued ones."""
    order: list[int] = []
    lock = threading.Lock()

    holder_ready = threading.Event()

    def holder():
        gate.acquire()
        holder_ready.set()
        time.sleep(0.2)  # let every client queue up behind it
        gate.release()

    def client(i: int, prio: float):
        gate.acquire(priority=prio)
        with lock:
            order.append(i)
        time.sleep(hold_s)
        gate.release()

    h = threading.Thread(target=holder)
    h.start()
    holder_ready.wait()
    threads = []
    for i, prio in enumerate(priorities):
        t = threading.Thread(target=client, args=(i, prio))
        t.start()
        threads.append(t)
        time.sleep(0.005)  # deterministic arrival order
    for t in threads + [h]:
        t.join()
    return order


def test_fifo_serves_in_arrival_order():
    assert run_clients(Gate("fifo", 1), [5, 1, 4, 2, 3]) == [0, 1, 2, 3, 4]


def test_priority_serves_smallest_first():
    assert run_clients(Gate("priority", 1), [5, 1, 4, 2, 3]) == [1, 3, 4, 2, 0]


def test_priority_aging_promotes_old_requests():
    # Huge aging: waiting time dominates priority, so behaviour approaches FIFO.
    assert run_clients(Gate("priority", 1, aging=1e6), [5, 1, 4, 2, 3]) == [0, 1, 2, 3, 4]


def test_max_inflight_is_respected():
    gate = Gate("fifo", 2)
    peak = 0
    current = 0
    lock = threading.Lock()

    def client():
        nonlocal peak, current
        gate.acquire()
        with lock:
            current += 1
            peak = max(peak, current)
        time.sleep(0.03)
        with lock:
            current -= 1
        gate.release()

    threads = [threading.Thread(target=client) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak == 2
    assert gate.stats()["dispatched"] == 10


def test_passthrough_never_blocks():
    gate = Gate("passthrough", 1)
    for _ in range(50):
        gate.acquire()
    assert gate.stats()["inflight"] == 50


def test_unknown_policy():
    with pytest.raises(ValueError):
        Gate("lottery", 1)
