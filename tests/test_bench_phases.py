"""B stage 2a: phase-structured episodes, coordinator policies, and freezing the planner server."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pandas as pd
import pytest
from conftest import FAKES, ROOT, free_port, start_fake
from test_bench_interference import _llm_up

from rsched.bench_phases import (Coordinator, Pauser, Step, make_workload, mean_burst, parse_policy,
                                 process_tree)
from rsched.traceio import read_jsonl


class _Null:
    def write(self, rec):
        pass


def test_workload_is_deterministic_and_shaped():
    a, b = make_workload(6, 3), make_workload(6, 3)
    assert a == b and a != make_workload(6, 4)
    for ep in a:
        assert ep[0].kind == "plan" and ep[-1].kind == "plan"
        for s in ep:
            if s.kind == "vla":
                assert 1 <= s.chunks <= s.announced
    assert mean_burst(a) > 0


def test_parse_policy():
    assert parse_policy("none") == ("none", {"budget": 2.0, "max_pause": 3.0, "min_run": 0.5})
    assert parse_policy("intent:budget=1.5,min_run=1")[1]["budget"] == 1.5
    with pytest.raises(ValueError):
        parse_policy("lpt")
    with pytest.raises(ValueError):
        parse_policy("intent:slack=1")


def _coord(name, budget=2.0):
    return Coordinator(name, {"budget": budget, "max_pause": 3.0, "min_run": 0.5}, Pauser(None),
                       hist_chunks=6.0, chunk_period_s=0.25, events=_Null(), run_id="t")


def test_want_pause_by_policy():
    long_burst = Step("vla", announced=16, chunks=4)   # announced 4 s, really 1 s
    for name, expect in (("none", False), ("vla_first", True), ("intent", False), ("oracle", True),
                         ("history", True)):  # history: 6 chunks = 1.5 s
        c = _coord(name)
        c.planner_begin()
        c.burst_begin(0, long_burst)
        assert c.want_pause() is expect, name
    c = _coord("vla_first")
    c.burst_begin(0, long_burst)
    assert not c.want_pause()  # no planner call in flight: nothing to pause
    c = _coord("intent")
    c.planner_begin()
    c.burst_begin(0, long_burst)
    for _ in range(9):
        c.burst_chunk(0)
    assert c.want_pause()  # 7 announced chunks left = 1.75 s


def test_process_tree_and_pauser():
    p = subprocess.Popen([sys.executable, "-c",
                          "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
                          "'import time; time.sleep(60)']); time.sleep(60)"])
    try:
        deadline = time.time() + 10
        while len(process_tree(p.pid)) < 2:
            assert time.time() < deadline
            time.sleep(0.1)
        pz = Pauser(p.pid)
        pz.pause()
        assert all(_state(pid) == "T" for pid in pz.pids)
        pz.resume()
        assert all(_state(pid) != "T" for pid in pz.pids)
    finally:
        for pid in reversed(process_tree(p.pid)):
            os.kill(pid, signal.SIGKILL)
        p.wait()


def _state(pid: int) -> str:
    return subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()[:1]


def test_bench_phases_end_to_end(tmp_path):
    vla, vport = start_fake("vla", 30)
    molmo, mport = start_fake("molmo", 40)
    lport = free_port()
    llm = subprocess.Popen([sys.executable, str(FAKES / "fake_llm.py"), "--port", str(lport),
                            "--ttft-ms", "30", "--tpot-ms", "2"])
    pid_file = tmp_path / "llm.pid"
    pid_file.write_text(str(llm.pid))
    try:
        deadline = time.time() + 20
        while not _llm_up(lport):
            assert time.time() < deadline and llm.poll() is None, "fake llm did not start"
            time.sleep(0.1)
        out = tmp_path / "b2a"
        cmd = [sys.executable, "-m", "rsched.bench_phases", "--out", str(out),
               "--levels", "2", "--repeats", "2", "--episodes-per-slot", "2", "--warmup-episodes", "1",
               "--steps", "1", "2", "--tokens", "10", "30", "--chunk-choices", "2", "4", "--vla-gap", "0.02",
               "--vla-alone-s", "0.03", "--llm-prompt-tokens", "200", "--no-gpu-sampler",
               "--llm-pid-file", str(pid_file),
               "--vla-endpoint", f"http://127.0.0.1:{vport}", "--molmo-endpoint", f"http://127.0.0.1:{mport}",
               "--llm-url", f"http://127.0.0.1:{lport}"]
        subprocess.run([*cmd, "--policies", "none", "vla_first", "intent:budget=0.1"],
                       cwd=ROOT, check=True, timeout=180)
        assert _state(llm.pid) != "T"  # left running
        runs = pd.DataFrame(read_jsonl(out / "runs.jsonl"))
        assert len(runs) == 6 and (runs["n_errors"] == 0).all()
        assert set(runs["label"]) == {"none", "vla_first", "intent:budget=0.1"}
        assert (runs[runs["label"] == "none"]["n_pauses"] == 0).all()
        assert runs[runs["label"] == "vla_first"]["n_pauses"].sum() > 0
        ev = read_jsonl(out / "events.jsonl")
        assert sum(e["event"] == "pause" for e in ev) == sum(e["event"] == "resume" for e in ev)
        s = pd.read_csv(out / "summary.csv")
        assert set(s["label"]) == set(runs["label"]) and (s["errors"] == 0).all()
        assert s[s["label"] == "none"]["vs_none_pct"].abs().max() < 1e-9
        # an MPS-tagged rerun pairs with the same batches; finished runs are skipped
        subprocess.run([*cmd, "--policies", "none", "--tag", "mps"], cwd=ROOT, check=True, timeout=180)
        runs = pd.DataFrame(read_jsonl(out / "runs.jsonl"))
        assert len(runs) == 8 and "none@mps" in set(runs["label"])
        assert set(runs.groupby("label")["seed"].apply(frozenset)) == {frozenset({2, 1002})}
    finally:
        os.kill(llm.pid, signal.SIGCONT)
        for p in (vla, molmo, llm):
            p.terminate()
            p.wait(timeout=10)
