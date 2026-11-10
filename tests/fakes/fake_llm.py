"""A CPU stand-in for an OpenAI-compatible streaming chat server (vLLM's API).

    python tests/fakes/fake_llm.py --port 9003 --ttft-ms 50 --tpot-ms 5

Streams ``max_tokens`` one-token chunks after ``--ttft-ms``, ``--tpot-ms`` apart,
then a usage chunk and ``[DONE]``; ``GET /v1/models`` answers health checks.
"""

from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ttft-ms", type=float, default=50.0)
    p.add_argument("--tpot-ms", type=float, default=5.0)
    args = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # noqa: D401
            return

        def do_GET(self):
            body = json.dumps({"object": "list", "data": [{"id": "planner"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            n = int(req.get("max_tokens", 16))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            time.sleep(args.ttft_ms / 1e3)
            for i in range(n):
                chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                if i < n - 1:
                    time.sleep(args.tpot_ms / 1e3)
            usage = {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": n}}
            self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
            self.wfile.flush()
            self.close_connection = True

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
