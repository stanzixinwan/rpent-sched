"""G3 end to end on CPU: driver -> sweep --sched -> launch intent -> priority proxy -> g3_compare."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml
from conftest import FAKES, ROOT

from rsched.traceio import read_jsonl


def _config(fake_services, path):
    cfg = {
        "name": "g3_fake",
        "rpent": {"command": [sys.executable, str(FAKES / "fake_rpent_episode.py")],
                  "planner": "fake", "robot": "libero"},
        "services": {
            "vla": {"backend": f"http://127.0.0.1:{fake_services['vla']}", "policy": "priority"},
            "sam3": {"backend": f"http://127.0.0.1:{fake_services['sam3']}", "policy": "passthrough"},
        },
        "scheduling": {"mode": "none"},
        "episodes": {"suites": ["libero_object_swap", "libero_goal_swap"], "tasks": [0, 1], "seeds": [0]},
        "concurrency": [4],
        "episodes_per_level": 4,
        "episode_timeout_s": 120,
        "gpu_sampler": False,
    }
    path.write_text(yaml.safe_dump(cfg))


def test_g3_driver_modes_and_compare(fake_services, tmp_path):
    cfg_path = tmp_path / "g3_fake.yaml"
    _config(fake_services, cfg_path)
    stamp = f"pytest-{os.getpid()}-{int(time.time())}"
    root = ROOT / "results" / "g3" / stamp
    env = {**os.environ, "RS_FAKE_ENV_ENDPOINT": f"http://127.0.0.1:{fake_services['env']}",
           "RS_FAKE_THINK_S": "0.02", "G3_CONFIG": str(cfg_path),
           "G3_MODES": "none intent_srpt history_lrpt", "G3_LEVELS": "4", "G3_REPEATS": "2"}
    try:
        subprocess.run(["bash", "scripts/run_g3.sh", stamp], cwd=ROOT, env=env, check=True, timeout=600)
        for mode in ("none", "intent_srpt", "history_lrpt"):
            for rep in (0, 1):
                assert (root / mode / f"N4_r{rep}" / "manifest.json").exists(), (mode, rep)

        # FIFO sends no priorities; intent sends one on every VLA request, and an
        # episode's remaining work only shrinks as it runs.
        fifo = read_jsonl(root / "none" / "N4_r0" / "proxy_vla.jsonl")
        assert fifo and all(r["priority"] is None for r in fifo)
        intent = [r for r in read_jsonl(root / "intent_srpt" / "N4_r0" / "proxy_vla.jsonl")
                  if r["method"] == "vla.predict"]
        assert intent and all(r["priority"] is not None and r["priority"] > 0 for r in intent)
        for ep, part in pd.DataFrame(intent).sort_values("t_arrive").groupby("episode"):
            assert part["priority"].is_monotonic_decreasing, ep

        # LRPT sends negated history estimates once an episode has finished.
        hist = [r["priority"] for r in read_jsonl(root / "history_lrpt" / "N4_r1" / "proxy_vla.jsonl")
                if r["method"] == "vla.predict" and r["priority"] is not None]
        assert hist and all(p <= 0 for p in hist)
        assert len((root / "history_lrpt" / "N4_r1" / "history.jsonl").read_text().splitlines()) == 4

        trace = read_jsonl(next((root / "intent_srpt" / "N4_r0" / "traces").glob("episode_*.jsonl")))
        types = [r["type"] for r in trace]
        assert types.count("plan_work") == 1 and types.count("remaining") == 5  # one per tool call
        manifest = yaml.safe_load((root / "intent_srpt" / "N4_r0" / "manifest.json").read_text())
        assert manifest["scheduling"]["mode"] == "intent_srpt"

        # Resuming skips finished cells.
        out = subprocess.run(["bash", "scripts/run_g3.sh", stamp], cwd=ROOT, env=env, check=True,
                             timeout=120, capture_output=True, text=True).stdout
        assert out.count("(done)") == 6

        subprocess.run([sys.executable, "analysis/g3_compare.py", str(root)], cwd=ROOT, check=True, timeout=300)
        runs = pd.read_csv(root / "analysis" / "runs.csv")
        assert len(runs) == 6 and set(runs["mode"]) == {"none", "intent_srpt", "history_lrpt"}
        summary = pd.read_csv(root / "analysis" / "summary.csv")
        row = summary[(summary["mode"] == "intent_srpt") & (summary["level"] == 4)].iloc[0]
        assert row["makespan_s_count"] == 2
        assert row["makespan_s_vs_fifo_pct_mean"] == row["makespan_s_vs_fifo_pct_mean"]  # paired change computed
        pred = pd.read_csv(root / "analysis" / "predict.csv")
        assert pred["spearman_intent_s"].notna().all()
        assert (pred["vla_plan_minus_actual_mean"] >= 0).all()  # the plan is an upper bound
        assert (root / "analysis" / "g3_compare.png").exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)
