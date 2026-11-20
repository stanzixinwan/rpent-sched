"""A proxy with two comma-separated backends spreads gated calls over both replicas."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rsched.proxy import build_server
from rsched.traceio import read_jsonl


def _port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Slow(BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        srv = self.server
        with srv.lock:
            srv.active += 1
            srv.peak = max(srv.peak, srv.active)
        time.sleep(0.2)
        with srv.lock:
            srv.active -= 1
            srv.calls += 1
        body = json.dumps({"ok": True, "result": srv.server_address[1]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _backend():
    srv = ThreadingHTTPServer(("127.0.0.1", _port()), _Slow)
    srv.lock, srv.active, srv.peak, srv.calls = threading.Lock(), 0, 0, 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _call(port):
    body = json.dumps({"method": "molmo.ground", "args": [], "kwargs": {}, "session_id": None}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/call", data=body,
                                 headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as r:
        return json.loads(r.read())


def test_two_replicas(tmp_path):
    a, b = _backend(), _backend()
    log = tmp_path / "proxy.jsonl"
    args = argparse.Namespace(
        host="127.0.0.1", listen=_port(), name="molmo",
        backend=f"http://127.0.0.1:{a.server_address[1]},http://127.0.0.1:{b.server_address[1]}",
        policy="fifo", max_inflight=2, aging=0.0, gate_methods=["molmo.ground"],
        backend_timeout_s=10, log=str(log), max_batch=16, batch_mode="greedy",
        batch_window_ms=0.0, batch_intent=False, dump_dir=None, dump_per_episode=0)
    proxy = build_server(args)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        t0 = time.time()
        threads = [threading.Thread(target=_call, args=(args.listen,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy.log.close()
    assert a.calls + b.calls == 8 and a.calls >= 3 and b.calls >= 3
    assert a.peak == 1 and b.peak == 1          # one request per replica at a time
    assert elapsed < 8 * 0.2 * 0.75             # ran two at a time
    rows = [r for r in read_jsonl(log) if r.get("method") == "molmo.ground"]
    assert sorted({r["replica"] for r in rows}) == [0, 1]
