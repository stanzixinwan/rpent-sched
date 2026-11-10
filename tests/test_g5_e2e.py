"""G5 end to end on CPU: record -> history table -> held-out arms (plan, history, cold) -> compare."""

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


def _cfg(fake_services, plans, seeds, launch):
    cfg = {
        "name": "g5_fake",
        "rpent": {"command": [sys.executable, str(FAKES / "fake_rpent_episode.py")],
                  "planner": "fake", "robot": "libero"},
        "services": {
            "vla": {"backend": f"http://127.0.0.1:{fake_services['vla']}", "policy": "priority"},
            "sam3": {"backend": f"http://127.0.0.1:{fake_services['sam3']}", "policy": "passthrough"},
        },
        "episodes": {"suites": ["libero_object_swap", "libero_goal_swap"], "tasks": [0, 1], "seeds": seeds},
        "concurrency": [2],
        "episodes_per_level": 4 * len(seeds),
        "episode_timeout_s": 120,
        "gpu_sampler": False,
    }
    if launch:
        cfg["launch"] = {"rule": "given", "shuffle": True, "seed": 0, "plans_dir": str(plans)}
    return cfg


def test_g5_driver(fake_services, tmp_path):
    plans = tmp_path / "plans"
    _plans(plans)
    rec, held = tmp_path / "rec.yaml", tmp_path / "held.yaml"
    rec.write_text(yaml.safe_dump(_cfg(fake_services, plans, [0], launch=False)))
    held.write_text(yaml.safe_dump(_cfg(fake_services, plans, [1], launch=True)))
    stamp = f"pytest-{os.getpid()}-{int(time.time())}"
    root = ROOT / "results" / "g5" / stamp
    env = {**os.environ, "RS_FAKE_ENV_ENDPOINT": f"http://127.0.0.1:{fake_services['env']}",
           "RS_FAKE_THINK_S": "0.02", "G5_RECORD_CONFIG": str(rec), "G5_CONFIG": str(held),
           "G5_LEVEL": "2", "G5_REPEATS": "2"}
    try:
        subprocess.run(["bash", "scripts/run_g5.sh", stamp], cwd=ROOT, env=env, check=True, timeout=900)
        history = json.loads((root / "history.json").read_text())
        assert set(history["tasks"]) == {"libero_object_swap_t0", "libero_object_swap_t1",
                                         "libero_goal_swap_t0", "libero_goal_swap_t1"}
        for label in ("given", "plan", "history", "history_cold50"):
            for rep in (0, 1):
                assert (root / label / f"N2_r{rep}" / "manifest.json").exists(), (label, rep)
        m = json.loads((root / "history_cold50" / "N2_r0" / "manifest.json").read_text())
        info = m["launch_estimates"]
        assert info["estimate"] == "history" and len(info["cold_tasks"]) == 2
        assert len(info["history_tasks"]) == 4
        assert sum(info["estimate_s"].values()) == __import__("pytest").approx(sum(info["plan_s"].values()))
        # every arm at a repeat starts from the same shuffled batch
        batches = {lab: sorted((s["suite"], s["task"]) for s in json.loads(
            (root / lab / "N2_r1" / "manifest.json").read_text())["specs"]) for lab in ("given", "plan", "history")}
        assert len({tuple(b) for b in batches.values()}) == 1

        out = subprocess.run(["bash", "scripts/run_g5.sh", stamp], cwd=ROOT, env=env, check=True,
                             timeout=300, capture_output=True, text=True).stdout
        assert out.count("(done)") == 8

        subprocess.run([sys.executable, "analysis/launch_compare.py", str(root), "--baseline", "given"],
                       cwd=ROOT, check=True, timeout=120)
        summary = pd.read_csv(root / "analysis" / "summary.csv")
        assert set(summary["rule"]) == {"plan", "history", "history_cold50"}
        assert (summary["repeats"] == 2).all()
    finally:
        shutil.rmtree(root, ignore_errors=True)
