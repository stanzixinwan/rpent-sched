"""Does Pi0.5 get cheaper per request when requests are batched? (go/no-go for cross-episode batching)

    python -m rsched.bench_batch --out results/b_batch/<stamp> \
        --batch-sizes 1 2 4 8 16 --llm-streams 0 4

RPent's Pi0.5 server already takes batched openpi observations
(``predict_action_batch``; ``main_images [B,H,W,3]`` ...), but every agent sends
B=1, so concurrent episodes queue one request at a time (stage 2a: VLA p50
0.85 s at N=8 under MPS against 0.20 s alone). This sends batches of B distinct
LIBERO-shaped observations straight to the server, back to back, and times them,
optionally while ``--llm-streams`` closed-loop planner streams (vLLM) decode on
the same GPU. Latency is client-side (encoding, HTTP, inference, decoding), as a
batching proxy would see it.

Outputs in ``--out``: requests.jsonl (one per batch call) and batch.csv:
per (llm_streams, batch): latency p50/p95, ms per item, items per second, and
the speedup of items per second over B=1 at the same planner load.

Rule of thumb printed at the end: if a batch of 8 costs at most ~2x a batch of 1,
batching can absorb the VLA queue that concurrent episodes build up.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

from rsched.bench_interference import LlmWorker, filler
from rsched.traceio import JsonlWriter, now, read_jsonl

TASKS = ["pick up the black bowl and place it on the plate", "open the top drawer of the cabinet",
         "put the cream cheese in the bowl", "turn on the stove", "push the plate to the front of the stove",
         "put the bowl on top of the cabinet", "pick up the ketchup and place it in the basket",
         "put the wine bottle on the rack"]


def batch_obs(b: int, rng: np.random.Generator) -> dict:
    """``b`` distinct observations in the openpi wire format the Pi0.5 client sends (here with B=b)."""
    return {
        "main_images": rng.integers(0, 255, (b, 256, 256, 3), dtype=np.uint8),
        "wrist_images": rng.integers(0, 255, (b, 256, 256, 3), dtype=np.uint8),
        "extra_view_images": None,
        "states": rng.normal(size=(b, 8)).astype(np.float32),
        "task_descriptions": [TASKS[i % len(TASKS)] for i in range(b)],
    }


class Background:
    """Closed-loop planner streams that run until stopped."""

    def __init__(self, n: int, make) -> None:
        self.stop = threading.Event()
        self.done = 0
        self.failed = 0
        self.last_error = None
        self.threads = [threading.Thread(target=self._loop, args=(make(i),), daemon=True) for i in range(n)]

    def _loop(self, worker) -> None:
        while not self.stop.is_set():
            try:
                worker.once()
                self.done += 1
            except Exception as exc:
                self.failed += 1
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                time.sleep(0.5)

    def wait_until_decoding(self, timeout_s: float = 120.0) -> None:
        """Fail loudly unless the planner streams actually complete requests."""
        deadline = time.time() + timeout_s
        while self.done == 0:
            if time.time() > deadline:
                raise RuntimeError(f"planner streams completed nothing in {timeout_s:.0f} s "
                                   f"({self.failed} failures; last: {self.last_error}); is the LLM up?")
            time.sleep(0.5)

    def __enter__(self):
        for t in self.threads:
            t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=650)


def summarize(out: Path) -> None:
    import pandas as pd

    req = pd.DataFrame(read_jsonl(out / "requests.jsonl"))
    rows = []
    for (setup, llm, b), g in req.groupby(["setup", "llm_streams", "batch"], sort=False):
        g_ok = g[g["error"].isna()]
        p50 = g_ok["lat_s"].median()
        rows.append({"setup": setup, "llm_streams": llm, "batch": b, "n": len(g_ok), "lat_p50_s": p50,
                     "lat_p95_s": g_ok["lat_s"].quantile(0.95), "ms_per_item": p50 / b * 1e3,
                     "items_per_s": b / p50, "errors": int(g["error"].notna().sum())})
    t = pd.DataFrame(rows)
    one = t[t["batch"] == 1].set_index(["setup", "llm_streams"])
    key = list(zip(t["setup"], t["llm_streams"]))
    t["lat_x_vs_b1"] = [r / one["lat_p50_s"].get(k, np.nan) for r, k in zip(t["lat_p50_s"], key)]
    t["throughput_x_vs_b1"] = [r / one["items_per_s"].get(k, np.nan) for r, k in zip(t["items_per_s"], key)]
    t.to_csv(out / "batch.csv", index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(t.to_string(index=False))
    for (setup, llm), g in t.groupby(["setup", "llm_streams"], sort=False):
        r8 = g[g["batch"] == 8]
        if len(r8):
            x = float(r8["lat_x_vs_b1"].iloc[0])
            verdict = "GO: batching should absorb the VLA queue" if x <= 2.0 else (
                "MAYBE: some headroom" if x <= 4.0 else "NO-GO: the GPU is near saturated at B=1")
            print(f"{setup}, {llm} planner streams: a batch of 8 costs {x:.2f}x a batch of 1 -> {verdict}")
    print(f"wrote {out}/batch.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--llm-streams", type=int, nargs="*", default=[0, 4])
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--tag", default="", help="server setup label, e.g. mps (default: timeslice)")
    ap.add_argument("--vla-endpoint", default=f"http://127.0.0.1:{os.environ.get('VLA_PORT', '9000')}")
    ap.add_argument("--llm-url", default=f"http://127.0.0.1:{os.environ.get('LLM_PORT', '9003')}")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_SERVED_NAME", "planner"))
    ap.add_argument("--llm-prompt-tokens", type=int, default=4000)
    ap.add_argument("--llm-output-tokens", type=int, default=256)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out)
        return
    from rpent.robots.components.vla_client_base import BaseVLAClient
    from rpent.utils.rpc.http_rpc import HttpRpcClient

    client = BaseVLAClient(HttpRpcClient(args.vla_endpoint))
    rng = np.random.default_rng(0)
    shared = filler(int(args.llm_prompt_tokens * 0.6 * 0.75), random.Random(0))

    def make_llm(i: int):
        return LlmWorker(args.llm_url, args.llm_model, args.llm_prompt_tokens, args.llm_output_tokens,
                         True, seed=2000 + i, shared_prefix=shared)

    (out / f"args{'_' + args.tag if args.tag else ''}.json").write_text(json.dumps(vars(args), indent=2))
    log = JsonlWriter(out / "requests.jsonl")
    setup = args.tag or "timeslice"
    try:
        for n_llm in args.llm_streams:
            with Background(n_llm, make_llm) as bg:
                if n_llm:
                    bg.wait_until_decoding()
                for b in args.batch_sizes:
                    obs = [batch_obs(b, rng) for _ in range(4)]
                    for k in range(args.warmup + args.iters):
                        t0 = now()
                        try:
                            actions, err = np.asarray(client.predict(obs[k % 4], None)), None
                            if actions.shape[0] != b:
                                err = f"expected {b} action chunks, got shape {actions.shape}"
                        except Exception as exc:
                            actions, err = None, f"{type(exc).__name__}: {exc}"[:300]
                        lat = now() - t0
                        if k >= args.warmup:
                            log.write({"setup": setup, "llm_streams": n_llm, "batch": b, "iter": k - args.warmup,
                                       "t_start": t0, "lat_s": lat, "error": err,
                                       "shape": list(actions.shape) if actions is not None else None})
                    print(f"{setup}, {n_llm} planner streams, B={b}: done "
                          f"(planner requests so far: {bg.done} ok, {bg.failed} failed)", flush=True)
                    if n_llm and bg.failed > bg.done:
                        raise RuntimeError(f"planner streams mostly failing: {bg.last_error}")
    finally:
        log.close()
    summarize(out)


if __name__ == "__main__":
    main()
