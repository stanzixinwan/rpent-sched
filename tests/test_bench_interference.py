"""B feasibility load generator on fake VLA, Molmo, and LLM servers."""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request

import pandas as pd
import pytest
from conftest import FAKES, ROOT, free_port, start_fake

from rsched.bench_interference import parse_scenario


def _llm_up(port: int) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"http://127.0.0.1:{port}/v1/models", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def test_parse_scenario():
    assert parse_scenario("vla=1,llm=4") == {"vla": 1, "llm": 4}
    with pytest.raises(ValueError):
        parse_scenario("gpu=1")


def test_bench_runs_and_reports(tmp_path):
    vla, vport = start_fake("vla", 30)
    molmo, mport = start_fake("molmo", 40)
    lport = free_port()
    llm = subprocess.Popen([sys.executable, str(FAKES / "fake_llm.py"), "--port", str(lport),
                            "--ttft-ms", "40", "--tpot-ms", "2"])
    try:
        deadline = time.time() + 20
        while not _llm_up(lport):
            assert time.time() < deadline and llm.poll() is None, "fake llm did not start"
            time.sleep(0.1)
        out = tmp_path / "b"
        subprocess.run([sys.executable, "-m", "rsched.bench_interference", "--out", str(out),
                        "--scenarios", "vla=1", "llm=2", "molmo=1", "vla=1,llm=2",
                        "--duration", "2", "--warmup", "0.5", "--no-gpu-sampler",
                        "--vla-endpoint", f"http://127.0.0.1:{vport}",
                        "--molmo-endpoint", f"http://127.0.0.1:{mport}",
                        "--llm-url", f"http://127.0.0.1:{lport}",
                        "--llm-output-tokens", "20", "--llm-prompt-tokens", "200"],
                       cwd=ROOT, check=True, timeout=120)
        s = pd.read_csv(out / "summary.csv")
        assert set(zip(s["scenario"], s["kind"])) == {("vla=1", "vla"), ("llm=2", "llm"), ("molmo=1", "molmo"),
                                                      ("vla=1,llm=2", "vla"), ("vla=1,llm=2", "llm")}
        assert (s["errors"] == 0).all()
        v = s[s["scenario"] == "vla=1"].iloc[0]
        assert 0.03 <= v["lat_p50_s"] < 0.2 and v["per_s"] > 1  # 30 ms predict + 170 ms gap
        lrow = s[(s["scenario"] == "llm=2")].iloc[0]
        assert 0.03 <= lrow["ttft_p50_s"] < 0.3 and 1.5 <= lrow["tpot_p50_ms"] < 20
        inf = pd.read_csv(out / "inflation.csv")
        assert set(inf["kind"]) == {"vla", "llm"} and (inf["with"].str.len() > 0).all()
    finally:
        for p in (vla, molmo, llm):
            p.terminate()
            p.wait(timeout=10)
