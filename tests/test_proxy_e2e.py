"""Proxy in front of a fake VLA built on RPent's real RpcFacade (same lock)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np
from conftest import ROOT, free_port, healthy
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.rpc.http_rpc import HttpRpcClient

from rsched.traceio import read_jsonl


def start_proxy(name, backend_port, policy, log, max_inflight=1):
    port = free_port()
    p = subprocess.Popen(
        [sys.executable, "-m", "rsched.proxy", "--name", name, "--listen", str(port),
         "--backend", f"http://127.0.0.1:{backend_port}", "--policy", policy,
         "--max-inflight", str(max_inflight), "--log", str(log)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 15
    while not healthy(port):
        assert time.time() < deadline and p.poll() is None, "proxy did not start"
        time.sleep(0.05)
    return p, port


def obs():
    return {"main_images": np.zeros((16, 16, 3), np.uint8), "wrist_images": None,
            "states": np.zeros(8, np.float32), "task_descriptions": "pick"}


def test_fifo_proxy_measures_wait_and_service(fake_services, tmp_path):
    log = tmp_path / "proxy_vla.jsonl"
    proxy, port = start_proxy("vla", fake_services["vla"], "fifo", log)
    try:
        client = Pi05VLAClient(HttpRpcClient(f"http://127.0.0.1:{port}"), embodiment="libero")
        results, errors = [], []

        def worker():
            try:
                results.append(client.predict(obs()))
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert all(r.shape == (5, 7) for r in results)

        # stats endpoint
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/rs/stats") as r:
            assert b'"dispatched": 8' in r.read()
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)

    recs = [r for r in read_jsonl(log) if r["method"] == "vla.predict"]
    assert len(recs) == 8
    assert all(r["ok"] and r["gated"] for r in recs)
    # Service time ~ the fake's 40 ms; served one at a time, never overlapping.
    services = sorted(r["service_s"] for r in recs)
    assert 0.035 < services[len(services) // 2] < 0.2
    spans = sorted((r["t_dispatch"], r["t_done"]) for r in recs)
    assert all(a[1] <= b[0] + 1e-3 for a, b in zip(spans, spans[1:]))
    # Under contention most requests waited.
    assert sum(r["wait_s"] > 0.02 for r in recs) >= 5


def test_passthrough_proxy_forwards_errors(fake_services, tmp_path):
    log = tmp_path / "proxy.jsonl"
    proxy, port = start_proxy("vla", fake_services["vla"], "passthrough", log)
    try:
        client = HttpRpcClient(f"http://127.0.0.1:{port}")
        try:
            client.call("no.such.method")
            raised = False
        except Exception:
            raised = True
        assert raised, "server error must reach the client unchanged"
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)
    rec = next(r for r in read_jsonl(log) if r["method"] == "no.such.method")
    assert rec["ok"] is False and rec["gated"] is False and rec["error"] is None
