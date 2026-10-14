"""End-to-end CPU smoke test: sweep -> launch hooks -> proxies -> analysis."""

from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import yaml
from conftest import FAKES, ROOT

from rsched.traceio import read_jsonl


def test_sweep_and_breakdown(fake_services, tmp_path):
    cfg = {
        "name": "smoke_cpu",
        "rpent": {"command": [sys.executable, str(FAKES / "fake_rpent_episode.py")],
                  "planner": "fake", "robot": "libero"},
        "services": {
            "vla": {"backend": f"http://127.0.0.1:{fake_services['vla']}", "policy": "fifo"},
            "sam3": {"backend": f"http://127.0.0.1:{fake_services['sam3']}", "policy": "passthrough"},
        },
        "episodes": {"suites": ["libero_object_swap", "libero_goal_swap"], "tasks": [0, 1], "seeds": [0]},
        "concurrency": [1, 4],
        "episodes_per_level": 4,
        "episode_timeout_s": 120,
        "gpu_sampler": False,
    }
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    env = {**os.environ, "RS_FAKE_ENV_ENDPOINT": f"http://127.0.0.1:{fake_services['env']}",
           "RS_FAKE_THINK_S": "0.05"}
    subprocess.run([sys.executable, "-m", "rsched.sweep", str(cfg_path), "--out", str(out)],
                   cwd=ROOT, env=env, check=True, timeout=300)

    # Every episode produced a complete trace with tool calls attributed to proxy requests.
    traces = list((out / "N4_r0" / "traces").glob("episode_*.jsonl"))
    assert len(traces) == 4
    recs = read_jsonl(traces[0])
    types = {r["type"] for r in recs}
    assert {"episode_start", "tool", "client_rpc", "outcome", "episode_end"} <= types
    picks = [r for r in recs if r["type"] == "tool" and r["tool"] == "pi0_pick"]
    assert picks and all("chunks_used" in r["result"] for r in picks), "raw primitive result captured"

    proxy = read_jsonl(out / "N4_r0" / "proxy_vla.jsonl")
    predicts = [r for r in proxy if r["method"] == "vla.predict"]
    assert predicts and all(r["episode"] and r["call_id"] and r["tool"] == "pi0_pick" for r in predicts)
    # VLA calls per episode in the proxy == chunks_used reported by the tools.
    ep = recs[0]["episode"]
    assert sum(r["result"]["chunks_used"] for r in picks) == sum(1 for r in predicts if r["episode"] == ep)

    subprocess.run([sys.executable, "analysis/breakdown.py", str(out)], cwd=ROOT, check=True, timeout=120)
    eps = pd.read_csv(out / "analysis" / "episodes.csv")
    levels = pd.read_csv(out / "analysis" / "levels.csv")
    assert len(eps) == 8 and set(levels["level"]) == {1, 4}

    comps = ["startup", "planner", "vla_wait", "vla_service", "sam3_wait", "sam3_service",
             "sim", "rpc_overhead", "tool_other"]
    gap = (eps[comps].sum(axis=1) - eps["wall_s"]).abs()
    assert (gap < 0.25 * eps["wall_s"] + 0.3).all(), "components should account for wall time"
    lv = levels.set_index("level")
    assert lv.loc[4, "vla_wait_mean_s"] > lv.loc[1, "vla_wait_mean_s"], "contention adds queueing"
    assert (out / "analysis" / "breakdown.png").exists()
