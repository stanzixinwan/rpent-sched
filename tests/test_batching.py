"""Cross-episode batching: wire format, merge/split, dispatch modes, and the proxy end to end."""

from __future__ import annotations

import json
import threading
import time
import urllib.request

import numpy as np
from conftest import free_port, start_fake

from rsched.batching import Batcher, decode, encode, merge, signature


def _obs(i: int, view: bool = True) -> dict:
    return {"main_images": np.full((1, 4, 4, 3), i, np.uint8),
            "wrist_images": np.full((1, 4, 4, 3), i, np.uint8) if view else None,
            "states": np.full((1, 8), i, np.float32), "task_descriptions": [f"task {i}"]}


def test_wire_roundtrip_and_merge():
    o = decode(json.loads(encode({"a": _obs(3), "x": np.float32(1.5)})))
    assert o["a"]["states"].shape == (1, 8) and o["a"]["states"][0, 0] == 3 and o["x"] == np.float32(1.5)
    m = merge([_obs(1), _obs(2), _obs(3)])
    assert m["main_images"].shape == (3, 4, 4, 3) and m["task_descriptions"] == ["task 1", "task 2", "task 3"]
    assert signature(_obs(1), None) == signature(_obs(2), None)
    assert signature(_obs(1), None) != signature(_obs(1, view=False), None)
    assert signature(_obs(1), None) != signature(_obs(1), {"mode": "train"})
    assert signature(merge([_obs(1), _obs(2)]), None) is None  # already batched: not ours to merge


class FakeBackend:
    """Echoes each row's state[0] as its action; ``ms`` per call regardless of batch size."""

    def __init__(self, ms: float = 50) -> None:
        self.ms, self.sizes, self.lock = ms, [], threading.Lock()

    def __call__(self, body: bytes) -> bytes:
        req = json.loads(body)
        obs = decode(req["args"][0])
        with self.lock:
            self.sizes.append(len(obs["task_descriptions"]))
        time.sleep(self.ms / 1e3)
        acts = np.repeat(obs["states"][:, None, :1], 5, axis=1)  # [B, 5, 1]
        return encode({"ok": True, "result": acts})


def _fire(b: Batcher, idx: list[int], **kw) -> dict[int, np.ndarray]:
    out = {}

    def one(i):
        it = b.submit(_obs(i), None, None, kw.get("episode", f"e{i}"), kw.get("burst_left"))
        out[i] = decode(json.loads(it.response))["result"]

    ts = [threading.Thread(target=one, args=(i,)) for i in idx]
    for t in ts:
        t.start()
        time.sleep(0.002)
    for t in ts:
        t.join()
    return out


def test_greedy_batches_what_piles_up_and_routes_rows_back():
    be = FakeBackend(ms=60)
    b = Batcher(be, max_batch=16, mode="greedy")
    out = _fire(b, list(range(6)))
    assert all(out[i].shape == (1, 5, 1) and out[i][0, 0, 0] == i for i in range(6))  # each got its own row
    assert be.sizes[0] == 1 and sum(be.sizes) == 6 and max(be.sizes) == 5  # the rest piled up behind the first


def test_max_batch_and_window():
    be = FakeBackend(ms=10)
    b = Batcher(be, max_batch=3, mode="window", window_s=0.2)
    t0 = time.time()
    _fire(b, list(range(5)))
    assert max(be.sizes) == 3 and sum(be.sizes) == 5
    assert time.time() - t0 < 1.0


def test_predict_waits_only_for_an_expected_episode():
    be = FakeBackend(ms=5)
    b = Batcher(be, max_batch=8, mode="predict", window_s=0.15, use_intent=True, default_gap_s=0.05)
    # episode A finishes a request; its next one is expected ~50 ms later
    b.submit(_obs(1), None, None, "A", 5)
    time.sleep(0.01)
    res = {}

    def b_req():
        res["B"] = b.submit(_obs(2), None, None, "B", 3)

    tb = threading.Thread(target=b_req)
    tb.start()
    time.sleep(0.03)
    b.submit(_obs(3), None, None, "A", 4)  # arrives while B is held for it
    tb.join()
    assert res["B"].batch_size == 2
    # A's burst is over (burst_left 1 -> 0 after it finishes): nobody is waited for
    b.submit(_obs(4), None, None, "A", 1)
    t0 = time.time()
    it = b.submit(_obs(5), None, None, "C", 2)
    assert it.batch_size == 1 and time.time() - t0 < 0.1


def test_backend_error_reaches_every_caller():
    def broken(body):
        return encode({"ok": False, "error": "boom"})

    b = Batcher(broken, mode="greedy")
    out = []
    ts = [threading.Thread(target=lambda i=i: out.append(json.loads(b.submit(_obs(i), None, None, None,
                                                                             None).response)))
          for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(out) == 3 and all(r["ok"] is False for r in out)


def test_proxy_batch_policy_with_rpent_client(tmp_path):
    """Real RPent Pi0.5 client -> rsched proxy (batch) -> fake VLA facade (batch-aware)."""
    import subprocess
    import sys

    from conftest import ROOT

    from rsched.bench_interference import VlaWorker

    vla, vport = start_fake("vla", 80)
    pport = free_port()
    log = tmp_path / "proxy.jsonl"
    proxy = subprocess.Popen([sys.executable, "-m", "rsched.proxy", "--name", "vla", "--listen", str(pport),
                              "--backend", f"http://127.0.0.1:{vport}", "--policy", "batch",
                              "--batch-mode", "greedy", "--log", str(log)], cwd=ROOT)
    try:
        deadline = time.time() + 20
        while True:
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                opener.open(f"http://127.0.0.1:{pport}/rs/stats", timeout=1).read()
                break
            except Exception:
                assert time.time() < deadline
                time.sleep(0.1)
        workers = [VlaWorker(f"http://127.0.0.1:{pport}", 0.0, seed=i) for i in range(6)]
        errs = []

        def run(w, i):
            w.set_headers({"X-RS-Episode": f"e{i}", "X-RS-Burst-Left": "3"})
            for _ in range(3):
                try:
                    a = w.client.predict(w.obs)
                    assert a.shape == (5, 7)
                except Exception as exc:  # noqa: BLE001
                    errs.append(exc)

        ts = [threading.Thread(target=run, args=(w, i)) for i, w in enumerate(workers)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errs
        stats = json.loads(opener.open(f"http://127.0.0.1:{pport}/rs/stats").read())["batching"]
        assert stats["requests"] == 18 and stats["mean_batch"] > 1.5
    finally:
        proxy.terminate()
        proxy.wait(timeout=10)
        vla.terminate()
        vla.wait(timeout=10)
    recs = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(recs) == 18 and all(r["ok"] for r in recs) and max(r["batch_size"] for r in recs) > 1
    assert {r["burst_left"] for r in recs} == {3}
