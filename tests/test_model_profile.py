"""Proxy request dump -> profiled server -> bench_models -> analysis/model_profile.py, on CPU."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "fakes"
sys.path.insert(0, str(ROOT / "third_party" / "RPent"))
torch = pytest.importorskip("torch")
pytest.importorskip("rpent.utils.rpc.rpc_facade")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def call(port: int, body: bytes, headers: dict | None = None) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}/call", data=body,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=30) as r:
        return json.loads(r.read())


def wait_up(port: int, proc: subprocess.Popen) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited")
        try:
            if call(port, json.dumps({"method": "healthz", "args": [], "kwargs": {}}).encode()).get("ok"):
                return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError("server not up")


def obs_body(rng: np.random.Generator) -> bytes:
    from rsched.batching import encode

    obs = {"main_images": rng.integers(0, 255, (1, 8, 8, 3), dtype=np.uint8),
           "wrist_images": rng.integers(0, 255, (1, 8, 8, 3), dtype=np.uint8),
           "extra_view_images": None, "states": rng.normal(size=(1, 8)).astype(np.float32),
           "task_descriptions": ["pick up the bowl"]}
    return encode({"method": "vla.predict", "args": [obs], "kwargs": {}, "session_id": None})


def test_profile_pipeline(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "third_party" / "RPent"),
                                                         os.environ.get("PYTHONPATH", "")])}
    port, pport = free_port(), free_port()
    log = tmp_path / "run" / "vla" / "server.jsonl"
    server = subprocess.Popen([sys.executable, "-m", "rsched.profiled_server", "--service", "vla", "--log", str(log),
                               "--", "tests.fakes.fake_model_server", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    proxy = subprocess.Popen([sys.executable, "-m", "rsched.proxy", "--name", "vla", "--listen", str(pport),
                              "--backend", f"http://127.0.0.1:{port}", "--dump-dir", str(tmp_path / "req"),
                              "--dump-per-episode", "3"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    try:
        wait_up(port, server)
        wait_up(pport, proxy)
        rng = np.random.default_rng(0)
        for ep in ("e1", "e2"):
            for _ in range(5):
                assert call(pport, obs_body(rng), {"X-RS-Episode": ep})["ok"]
        dumped = sorted((tmp_path / "req" / "vla.predict").glob("*.json"))
        assert len(dumped) == 6, "first 3 per episode"

        out = tmp_path / "run" / "vla"
        subprocess.run([sys.executable, "-m", "rsched.bench_models", "--service", "vla",
                        "--url", f"http://127.0.0.1:{port}", "--requests", str(tmp_path / "req"),
                        "--out", str(out), "--idle-s", "0.2", "--warmup", "2", "--calls", "6",
                        "--stage-calls", "3", "--torch-calls", "2", "--batches", "1", "4", "--batch-calls", "3"],
                       cwd=ROOT, env=env, check=True, timeout=120)
    finally:
        for p in (server, proxy):
            p.terminate()
            p.wait(timeout=10)

    recs = [json.loads(line) for line in log.read_text().splitlines()]
    calls = [r for r in recs if r["type"] == "call"]
    phases = {r["phase"] for r in calls}
    assert {"timing", "stages", "torch", "batch_1", "batch_4"} <= phases
    assert all(r["decode_s"] >= 0 and r["encode_s"] >= 0 and r["method_s"] > 0 for r in calls)
    staged = [r for r in calls if r["phase"] == "stages"]
    assert all("vision" in r["stages"] and "head.0" in r["stages"] for r in staged)
    assert any(r["type"] == "modules" for r in recs)
    ops = pd.read_csv(out / "torch" / "vla.ops.csv")
    assert (ops["name"] == "aten::addmm").any() and ops["flops"].sum() > 0

    subprocess.run([sys.executable, "analysis/model_profile.py", str(tmp_path / "run")], cwd=ROOT, env=env,
                   check=True, timeout=60)
    an = tmp_path / "run" / "analysis"
    hd = pd.read_csv(an / "host_device.csv")
    assert hd.loc[0, "calls"] == 6 and hd.loc[0, "method_ms"] > 0
    st = pd.read_csv(an / "stages.csv")
    assert {"vision", "head"} <= set(st["module"])
    batch = pd.read_csv(an / "batch.csv")
    assert list(batch["batch"]) == [1, 4] and batch["throughput_x_vs_b1"].iloc[0] == pytest.approx(1.0)
    roof = pd.read_csv(an / "roofline.csv")
    assert (roof["flops_per_op"] > 0).all() and (roof["intensity"] > 0).all()
