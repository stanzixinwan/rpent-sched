from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "fakes"
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def healthy(port: int) -> bool:
    body = json.dumps({"method": "healthz", "args": [], "kwargs": {}, "session_id": None}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/call", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=1) as r:
            return json.loads(r.read()).get("ok", False)
    except Exception:
        return False


def start_fake(kind: str, ms: float) -> tuple[subprocess.Popen, int]:
    port = free_port()
    p = subprocess.Popen(
        [sys.executable, str(FAKES / "fake_services.py"), "--kind", kind, "--port", str(port), "--ms", str(ms)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while not healthy(port):
        if time.time() > deadline or p.poll() is not None:
            p.kill()
            raise RuntimeError(f"fake {kind} did not start")
        time.sleep(0.1)
    return p, port


@pytest.fixture
def fake_services():
    procs = {}
    ports = {}
    try:
        for kind, ms in (("vla", 40), ("sam3", 25), ("env", 5)):
            procs[kind], ports[kind] = start_fake(kind, ms)
        yield ports
    finally:
        for p in procs.values():
            p.terminate()
            p.wait(timeout=10)
