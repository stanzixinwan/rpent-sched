"""G4 end to end on CPU: plan files -> sweep --launch -> run_launch.sh -> launch_compare."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import yaml
from conftest import FAKES, ROOT

CHUNKS = {("object_swap", 0): 4, ("object_swap", 1): 20, ("goal_swap", 0): 12, ("goal_swap", 1): 8}


def _plans(d):
    d.mkdir()
    for (suite, task), chunks in CHUNKS.items():
        plan = {"plan": [{"action": "pi0_pick", "arguments": {"prompt": "p", "max_chunks": chunks}}]}
        (d / f"{suite}_t{task}_plan.json").write_text(json.dumps(plan))
        (d / f"{suite}_t{task}_anchors.json").write_text(json.dumps({"anchors": []}))


def _config(fake_services, path, plans):
    cfg = {
        "name": "g4_fake",
        "rpent": {"command": [sys.executable, str(FAKES / "fake_rpent_episode.py")],
                  "planner": "fake", "robot": "libero"},
        "services": {
            "vla": {"backend": f"http://127.0.0.1:{fake_services['vla']}", "policy": "priority"},
            "sam3": {"backend": f"http://127.0.0.1:{fake_services['sam3']}", "policy": "passthrough"},
        },
        "launch": {"rule": "given", "shuffle": True, "seed": 0, "plans_dir": str(plans)},
        "episodes": {"suites": ["libero_object_swap", "libero_goal_swap"], "tasks": [0, 1], "seeds": [0]},
        "concurrency": [2],
        "episodes_per_level": 4,
        "episode_timeout_s": 120,
        "gpu_sampler": False,
    }
    path.write_text(yaml.safe_dump(cfg))


def test_launch_rules_in_the_sweep_and_driver(fake_services, tmp_path):
    plans = tmp_path / "plans"
    _plans(plans)
    cfg_path = tmp_path / "g4_fake.yaml"
    _config(fake_services, cfg_path, plans)
    env = {**os.environ, "RS_FAKE_ENV_ENDPOINT": f"http://127.0.0.1:{fake_services['env']}",
           "RS_FAKE_THINK_S": "0.02"}

    # LPT launches by planned work, longest first.
    out = tmp_path / "lpt"
    subprocess.run([sys.executable, "-m", "rsched.sweep", str(cfg_path), "--launch", "lpt", "--out", str(out)],
                   cwd=ROOT, env=env, check=True, timeout=300)
    m = json.loads((out / "N2_r0" / "manifest.json").read_text())
    est = [x["estimate_s"] for x in m["launched"]]
    assert est == sorted(est, reverse=True) and est[0] == 20 * 0.64
    assert m["launch"]["rule"] == "lpt"

    stamp = f"pytest-{os.getpid()}-{int(time.time())}"
    root = ROOT / "results" / "launch" / stamp
    drv = {**env, "LAUNCH_CONFIG": str(cfg_path), "LAUNCH_LEVEL": "2", "LAUNCH_REPEATS": "2",
           "LAUNCH_WARMUP": "2", "LAUNCH_RULES": "given jit_lpt"}
    try:
        subprocess.run(["bash", "scripts/run_launch.sh", stamp], cwd=ROOT, env=drv, check=True, timeout=600)
        assert (root / "warmup" / "N2_r0" / "manifest.json").exists()
        for rule in ("given", "jit_lpt"):
            for rep in (0, 1):
                assert (root / rule / f"N2_r{rep}" / "manifest.json").exists(), (rule, rep)
        # Both rules at a repeat start from the same shuffled batch; repeats differ.
        def batch(rule, rep):
            m = json.loads((root / rule / f"N2_r{rep}" / "manifest.json").read_text())
            return [(s["suite"], s["task"]) for s in m["specs"]]
        assert sorted(batch("given", 0)) == sorted(batch("jit_lpt", 0))

        out2 = subprocess.run(["bash", "scripts/run_launch.sh", stamp], cwd=ROOT, env=drv, check=True,
                              timeout=120, capture_output=True, text=True).stdout
        assert out2.count("(done)") == 4

        subprocess.run([sys.executable, "analysis/launch_compare.py", str(root)], cwd=ROOT, check=True, timeout=120)
        summary = pd.read_csv(root / "analysis" / "summary.csv")
        row = summary[summary["rule"] == "jit_lpt"].iloc[0]
        assert row["repeats"] == 2 and row["makespan_pct_mean"] == row["makespan_pct_mean"]
        cells = pd.read_csv(root / "analysis" / "cells.csv")
        assert len(cells) == 4 and "warmup" not in set(cells["rule"])
    finally:
        shutil.rmtree(root, ignore_errors=True)
