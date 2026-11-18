"""cpu.episode_cpus: parsing, validation, and that episodes really run pinned."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import yaml
from conftest import FAKES, ROOT

from rsched.sweep import episode_cpus, parse_cpus

linux_only = pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="needs sched_setaffinity")


def test_parse_cpus():
    assert parse_cpus("0-3,8,10-11") == [0, 1, 2, 3, 8, 10, 11]
    assert parse_cpus([3, 1, 1]) == [1, 3]
    assert parse_cpus(5) == [5]
    assert parse_cpus(None) is None and parse_cpus("") is None


@linux_only
def test_episode_cpus_validation():
    assert episode_cpus({}) is None
    mine = sorted(os.sched_getaffinity(0))
    assert episode_cpus({"cpu": {"episode_cpus": [mine[0]]}}) == [mine[0]]
    with pytest.raises(ValueError):
        episode_cpus({"cpu": {"episode_cpus": [max(mine) + 4096]}})


@linux_only
def test_episodes_run_pinned(tmp_path):
    cpu = sorted(os.sched_getaffinity(0))[0]
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "json.dump(sorted(os.sched_getaffinity(0)), open(os.path.join(out, 'affinity.json'), 'w'))\n")
    cfg = {"name": "pin", "rpent": {"command": [sys.executable, str(probe)], "planner": "fake"},
           "services": {}, "episodes": {"suites": ["s"], "tasks": [0], "seeds": [0]},
           "concurrency": [2], "episodes_per_level": 2, "gpu_sampler": False, "fail_fast": 0,
           "cpu": {"episode_cpus": str(cpu)}}
    cfg_path = tmp_path / "pin.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    subprocess.run([sys.executable, "-m", "rsched.sweep", str(cfg_path), "--out", str(out)],
                   cwd=ROOT, check=True, timeout=120)
    seen = [json.load(open(p)) for p in (out / "N2_r0" / "episodes").glob("*/affinity.json")]
    assert len(seen) == 2 and all(a == [cpu] for a in seen)
    manifest = json.load(open(out / "N2_r0" / "manifest.json"))
    assert manifest["episode_cpus"] == [cpu]


def _healthz_server():
    """A minimal stand-in for an RPent service: answers healthz on /call."""
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"ok": True, "result": None}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_service_without_proxy_hands_out_the_backend(tmp_path):
    srv = _healthz_server()
    backend = f"http://127.0.0.1:{srv.server_address[1]}"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "json.dump(sys.argv, open(os.path.join(out, 'argv.json'), 'w'))\n")
    cfg = {"name": "direct", "rpent": {"command": [sys.executable, str(probe)], "planner": "fake"},
           "services": {"vla": {"backend": backend, "proxy": False}},
           "episodes": {"suites": ["s"], "tasks": [0], "seeds": [0]},
           "concurrency": [1], "episodes_per_level": 1, "gpu_sampler": False, "fail_fast": 0}
    cfg_path = tmp_path / "direct.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    try:
        subprocess.run([sys.executable, "-m", "rsched.sweep", str(cfg_path), "--out", str(out)],
                       cwd=ROOT, check=True, timeout=120)
    finally:
        srv.shutdown()
    argv = json.load(open(next((out / "N1_r0" / "episodes").glob("*/argv.json"))))
    assert argv[argv.index("--vla-endpoint") + 1] == backend
    assert not (out / "N1_r0" / "proxy_vla.jsonl").exists()


def test_layout_topology_order(tmp_path):
    """SMT siblings 0/4, 1/5, 2/6, 3/7 on one socket: budgets take whole cores first."""
    from rsched.cpu_layout import fmt, layout

    for cpu in range(8):
        topo = tmp_path / f"cpu{cpu}" / "topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text("0")
        (topo / "core_id").write_text(str(cpu % 4))
    svc, eps = layout(2, 4, cpus=list(range(8)), sysfs=tmp_path)
    assert svc == [0, 4] and eps == [1, 5, 2, 6]
    assert fmt(eps) == "1,5,2,6" and fmt([3, 4, 5, 9]) == "3-5,9"
    small, _ = layout(2, 2, cpus=list(range(8)), sysfs=tmp_path)
    assert small == svc, "the service set does not depend on the episode budget"
    with pytest.raises(ValueError):
        layout(4, 8, cpus=list(range(8)), sysfs=tmp_path)


@linux_only
def test_rs_pin_env_pins_episodes(tmp_path):
    cpu = sorted(os.sched_getaffinity(0))[0]
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('--output-dir') + 1]\n"
        "os.makedirs(out, exist_ok=True)\n"
        "json.dump(sorted(os.sched_getaffinity(0)), open(os.path.join(out, 'affinity.json'), 'w'))\n")
    cfg = {"name": "pinenv", "rpent": {"command": [sys.executable, str(probe)], "planner": "fake"},
           "services": {}, "episodes": {"suites": ["s"], "tasks": [0], "seeds": [0]},
           "concurrency": [1], "episodes_per_level": 1, "gpu_sampler": False, "fail_fast": 0}
    cfg_path = tmp_path / "pinenv.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "run"
    env = {**os.environ, "RS_PIN": "1", "RS_EPISODE_CPUS": str(cpu), "RS_INFRA_CPUS": str(cpu)}
    subprocess.run([sys.executable, "-m", "rsched.sweep", str(cfg_path), "--out", str(out)],
                   cwd=ROOT, check=True, timeout=120, env=env)
    seen = json.load(open(next((out / "N1_r0" / "episodes").glob("*/affinity.json"))))
    assert seen == [cpu]
    manifest = json.load(open(out / "N1_r0" / "manifest.json"))
    assert manifest["episode_cpus"] == [cpu] and manifest["infra_cpus"] == str(cpu)
