"""Tracing and scheduling proxy for RPent model services.

Put one proxy in front of each shared service and point every ``rpent``
process at the proxy instead of the service::

    python -m rsched.proxy --name vla  --listen 9100 --backend http://127.0.0.1:9000 \
        --policy fifo --max-inflight 1 --log results/run1/proxy_vla.jsonl

Why a proxy: RPent's Pi0.5 server runs every ``vla.predict`` under one
exclusive lock (``RpcFacade`` + ``RWLock.write``), and waiting threads are
released by ``notify_all``, so the service order is whatever thread wins the
race. Holding requests here lets us (a) measure queueing delay separately from
service time and (b) choose the order, which is where scheduling policies plug in.

Policies
--------
passthrough   forward immediately; the backend's own lock decides the order
              (RPent's default behaviour; only end-to-end latency is observable).
fifo          at most ``--max-inflight`` requests at the backend, served in
              arrival order.
priority      like fifo, but serve the waiting request with the smallest
              ``X-RS-Priority`` header (e.g. predicted remaining work), minus
              ``--aging`` seconds of credit per second waited.

Only methods listed in ``--gate-methods`` are queued; health checks and
status calls always pass straight through.
"""

from __future__ import annotations

import argparse
import http.client
import itertools
import json
import math
import signal
import sys
import threading
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from rsched import HDR_CALL, HDR_EPISODE, HDR_PRIORITY, HDR_TOOL
from rsched.traceio import JsonlWriter, now

DEFAULT_GATE_METHODS = ("vla.predict", "sam3.segment", "molmo.ground")
POLICIES = ("passthrough", "fifo", "priority")


# ---------------------------------------------------------------------------
# Admission gate (the scheduling decision point)
# ---------------------------------------------------------------------------


@dataclass
class Ticket:
    seq: int
    t_arrive: float
    priority: float = math.inf
    meta: dict[str, Any] = field(default_factory=dict)
    event: threading.Event = field(default_factory=threading.Event)
    t_dispatch: float | None = None


class Gate:
    """Limits requests at the backend and picks which waiting request goes next.

    ``acquire`` blocks the caller's thread until its ticket is dispatched;
    ``release`` must be called once the backend has answered.
    """

    def __init__(self, policy: str, max_inflight: int, aging: float = 0.0) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; choose from {POLICIES}")
        self.policy = policy
        self.max_inflight = math.inf if policy == "passthrough" else max(1, max_inflight)
        self.aging = float(aging)
        self._lock = threading.Lock()
        self._waiting: list[Ticket] = []
        self._inflight = 0
        self._seq = itertools.count()
        self.dispatched = 0

    def _key(self, t: Ticket, t_now: float) -> tuple[float, int]:
        if self.policy == "priority":
            return (t.priority - self.aging * (t_now - t.t_arrive), t.seq)
        return (0.0, t.seq)  # fifo: arrival order

    def _dispatch_ready(self) -> None:
        # Caller holds self._lock.
        while self._waiting and self._inflight < self.max_inflight:
            t_now = now()
            best = min(self._waiting, key=lambda t: self._key(t, t_now))
            self._waiting.remove(best)
            self._inflight += 1
            self.dispatched += 1
            best.t_dispatch = t_now
            best.event.set()

    def acquire(self, priority: float = math.inf, **meta: Any) -> tuple[Ticket, dict]:
        """Block until dispatched; return the ticket and the state seen on arrival."""
        with self._lock:
            ticket = Ticket(seq=next(self._seq), t_arrive=now(), priority=priority, meta=meta)
            seen = {"queue_len": len(self._waiting), "inflight": self._inflight}
            self._waiting.append(ticket)
            self._dispatch_ready()
        ticket.event.wait()
        return ticket, seen

    def release(self) -> None:
        with self._lock:
            self._inflight -= 1
            self._dispatch_ready()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy": self.policy,
                "max_inflight": None if math.isinf(self.max_inflight) else self.max_inflight,
                "waiting": len(self._waiting),
                "inflight": self._inflight,
                "dispatched": self.dispatched,
            }


# ---------------------------------------------------------------------------
# HTTP proxy
# ---------------------------------------------------------------------------


def _peek_ok(body: bytes) -> bool | None:
    """Read RPent's ``{"ok": ...}`` envelope flag without decoding large payloads."""
    head = body[:32].replace(b" ", b"")
    if head.startswith(b'{"ok":true'):
        return True
    if head.startswith(b'{"ok":false'):
        return False
    return None


def _peek_method(body: bytes) -> str | None:
    """Extract the RPC method name; RPent puts ``"method"`` first in the payload."""
    head = body[:256]
    try:
        start = head.index(b'"method"')
        rest = head[start + len(b'"method"') :].lstrip(b" :")
        if rest.startswith(b'"'):
            return rest[1 : rest.index(b'"', 1)].decode()
    except ValueError:
        pass
    try:
        return json.loads(body).get("method")
    except Exception:
        return None


class ProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 256

    def __init__(
        self,
        listen: tuple[str, int],
        *,
        name: str,
        backend: str,
        gate: Gate,
        gate_methods: set[str],
        log: JsonlWriter | None,
        backend_timeout_s: float = 600.0,
    ) -> None:
        super().__init__(listen, _ProxyHandler)
        parsed = urllib.parse.urlsplit(backend if "://" in backend else f"http://{backend}")
        if parsed.scheme != "http":
            raise ValueError("only http:// backends are supported (start services with --transport http)")
        self.name = name
        self.backend_host = parsed.hostname or "127.0.0.1"
        self.backend_port = parsed.port or 80
        self.gate = gate
        self.gate_methods = gate_methods
        self.log = log
        self.backend_timeout_s = backend_timeout_s

    def forward(self, path: str, body: bytes, content_type: str) -> tuple[int, bytes, str]:
        conn = http.client.HTTPConnection(
            self.backend_host, self.backend_port, timeout=self.backend_timeout_s
        )
        try:
            conn.request("POST", path, body=body, headers={"Content-Type": content_type})
            resp = conn.getresponse()
            return resp.status, resp.read(), resp.getheader("Content-Type", "application/json")
        finally:
            conn.close()


class _ProxyHandler(BaseHTTPRequestHandler):
    server: ProxyServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _reply(self, status: int, body: bytes, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/rs/stats":
            payload = {"name": self.server.name, **self.server.gate.stats()}
            self._reply(200, json.dumps(payload).encode())
        else:
            self._reply(404, b"{}")

    def do_POST(self) -> None:
        srv = self.server
        t_arrive = now()
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b""
        content_type = self.headers.get("Content-Type", "application/json")
        method = _peek_method(body)

        gated = method in srv.gate_methods
        try:
            priority = float(self.headers.get(HDR_PRIORITY, "inf"))
        except ValueError:
            priority = math.inf

        seen: dict[str, Any] = {}
        t_dispatch = t_arrive
        if gated:
            ticket, seen = srv.gate.acquire(priority=priority)
            t_dispatch = ticket.t_dispatch or now()

        status, resp_body, resp_type, error = 502, b"", "application/json", None
        try:
            status, resp_body, resp_type = srv.forward(self.path, body, content_type)
        except Exception as exc:  # backend down or timed out
            error = f"{type(exc).__name__}: {exc}"
            resp_body = json.dumps({"ok": False, "error": f"rsched proxy: {error}"}).encode()
            status = 200  # RPent clients read failures from the body, like the real server
        finally:
            if gated:
                srv.gate.release()
        t_done = now()

        self._reply(status, resp_body, resp_type)

        if srv.log is not None:
            srv.log.write(
                {
                    "type": "rpc",
                    "service": srv.name,
                    "method": method,
                    "gated": gated,
                    "policy": srv.gate.policy,
                    "episode": self.headers.get(HDR_EPISODE),
                    "call_id": self.headers.get(HDR_CALL),
                    "tool": self.headers.get(HDR_TOOL),
                    "priority": None if math.isinf(priority) else priority,
                    "t_arrive": t_arrive,
                    "t_dispatch": t_dispatch,
                    "t_done": t_done,
                    "wait_s": t_dispatch - t_arrive,
                    "service_s": t_done - t_dispatch,
                    "queue_len_at_arrival": seen.get("queue_len"),
                    "inflight_at_arrival": seen.get("inflight"),
                    "req_bytes": len(body),
                    "resp_bytes": len(resp_body),
                    "http_status": status,
                    "ok": _peek_ok(resp_body) if error is None else False,
                    "error": error,
                }
            )


def build_server(args: argparse.Namespace) -> ProxyServer:
    gate = Gate(args.policy, args.max_inflight, args.aging)
    log = JsonlWriter(args.log) if args.log else None
    return ProxyServer(
        (args.host, args.listen),
        name=args.name,
        backend=args.backend,
        gate=gate,
        gate_methods=set(args.gate_methods),
        log=log,
        backend_timeout_s=args.backend_timeout_s,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--name", required=True, help="service label written to the log, e.g. vla")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--listen", type=int, required=True, help="port clients connect to")
    p.add_argument("--backend", required=True, help="real service, e.g. http://127.0.0.1:9000")
    p.add_argument("--policy", choices=POLICIES, default="fifo")
    p.add_argument("--max-inflight", type=int, default=1)
    p.add_argument("--aging", type=float, default=0.0, help="priority credit per second waited")
    p.add_argument("--gate-methods", nargs="*", default=list(DEFAULT_GATE_METHODS))
    p.add_argument("--backend-timeout-s", type=float, default=600.0)
    p.add_argument("--log", default=None, help="JSONL trace path")
    args = p.parse_args(argv)

    server = build_server(args)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"rsched proxy [{args.name}] listening on http://{args.host}:{server.server_address[1]} "
        f"-> {args.backend} policy={args.policy} max_inflight={args.max_inflight}",
        flush=True,
    )
    stop.wait()
    server.shutdown()
    server.server_close()
    if server.log is not None:
        server.log.close()


if __name__ == "__main__":
    sys.exit(main())
