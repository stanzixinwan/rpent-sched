"""Batch-scaling probe for the VLA server, on the fake VLA and fake planner."""

from __future__ import annotations

import subprocess
import sys
import time

import pandas as pd
from conftest import FAKES, ROOT, free_port, start_fake
from test_bench_interference import _llm_up

from rsched.bench_batch import batch_obs


def test_batch_obs_shapes():
    import numpy as np

    o = batch_obs(3, np.random.default_rng(0))
    assert o["main_images"].shape == (3, 256, 256, 3) and o["states"].shape == (3, 8)
    assert len(o["task_descriptions"]) == 3


def test_bench_batch_runs(tmp_path):
    vla, vport = start_fake("vla", 20)
    lport = free_port()
    llm = subprocess.Popen([sys.executable, str(FAKES / "fake_llm.py"), "--port", str(lport),
                            "--ttft-ms", "20", "--tpot-ms", "2"])
    try:
        deadline = time.time() + 20
        while not _llm_up(lport):
            assert time.time() < deadline and llm.poll() is None
            time.sleep(0.1)
        out = tmp_path / "bb"
        common = ["--out", str(out), "--batch-sizes", "1", "2", "8", "--iters", "5", "--warmup", "1",
                  "--vla-endpoint", f"http://127.0.0.1:{vport}", "--llm-url", f"http://127.0.0.1:{lport}",
                  "--llm-prompt-tokens", "200", "--llm-output-tokens", "20"]
        subprocess.run([sys.executable, "-m", "rsched.bench_batch", *common, "--llm-streams", "0", "2"],
                       cwd=ROOT, check=True, timeout=120)
        subprocess.run([sys.executable, "-m", "rsched.bench_batch", *common, "--llm-streams", "0",
                        "--tag", "mps"], cwd=ROOT, check=True, timeout=120)
        t = pd.read_csv(out / "batch.csv")
        assert len(t) == 9 and (t["errors"] == 0).all() and (t["n"] == 5).all()
        assert set(t["setup"]) == {"timeslice", "mps"}
        # the fake sleeps a fixed time per call, so throughput grows ~linearly with B
        r8 = t[(t.setup == "timeslice") & (t.llm_streams == 0) & (t.batch == 8)].iloc[0]
        assert r8["throughput_x_vs_b1"] > 3
    finally:
        for p in (vla, llm):
            p.terminate()
            p.wait(timeout=10)
