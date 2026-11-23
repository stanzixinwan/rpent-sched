"""Replay recorded model requests against one profiled server: latency, energy, stages, profiler, batch.

    python -m rsched.bench_models --service vla --url http://127.0.0.1:9100 \
        --requests results/model_profile/<stamp>/record/N1_r0/requests/vla --out results/model_profile/<stamp>/vla

The server must run under ``rsched.profiled_server`` (it answers ``prof.*``).
Requests are the bodies the proxy saved during real episodes
(``services.<name>.dump_per_episode``), so observations, prompts and image
sizes are the workload's own. Calls go one at a time, straight to the server
(no proxy), in phases the server log is tagged with (``prof.mark``):

    idle        no calls; GPU power at rest (baseline for energy)
    warmup      --warmup calls, discarded
    timing      --calls calls: latency, throughput, energy per call
    stages      --stage-calls calls with the module hooks on (per-stage GPU time)
    torch       --torch-calls calls under torch.profiler (kernels, launches, FLOPs)
    batch_<b>   vla only: <b> recorded observations merged into one call, for each --batches

Energy comes from NVML's total-energy counter when the GPU has one, else from
integrating sampled power; ``J_per_item_net`` subtracts idle power over the
same time. Outputs in --out: client.csv (one row per phase), calls.jsonl.
"""

from __future__ import annotations

import argparse
import os
import http.client
import itertools
import json
import statistics
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from rsched.traceio import JsonlWriter


class TegraEnergy:
    """Jetson (RS_GPU_SAMPLER=tegra): board input power and the GPU rail from tegrastats.

    power_mean_W is the board's input power (VIN), which includes CPU, memory and
    idle draw; the GPU rail alone is reported as power_gpu_mean_W.
    """

    counter = False
    nvml = None
    source = "tegrastats (board input power)"

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_ms = max(int(interval_s * 1000), 10)

    def power_w(self) -> float | None:
        return None

    def start(self) -> None:
        import subprocess

        from rsched.tegra import parse

        self.t0 = time.time()
        self.board: list[float] = []
        self.gpu: list[float] = []
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(self.interval_ms)],
                                     stdout=subprocess.PIPE, text=True, bufsize=1)

        def loop():
            for line in self.proc.stdout:
                r = parse(line)
                if r and r["power_w"] is not None:
                    self.board.append(r["power_w"])
                    if "VDD_GPU" in r["rails"]:
                        self.gpu.append(r["rails"]["VDD_GPU"])

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.proc.terminate()
        self.thread.join(timeout=2)
        dt = time.time() - self.t0
        mean_w = statistics.fmean(self.board) if self.board else None
        gpu_w = statistics.fmean(self.gpu) if self.gpu else None
        return {"seconds": dt, "energy_J": mean_w * dt if mean_w is not None else None,
                "power_mean_W": mean_w, "power_gpu_mean_W": gpu_w,
                "energy_source": "tegrastats_board" if mean_w is not None else "none"}


class Energy:
    """GPU energy over an interval: NVML counter if present, else sampled power."""

    def __new__(cls, gpu: int = 0, interval_s: float = 0.05):
        if os.environ.get("RS_GPU_SAMPLER") == "tegra":
            return TegraEnergy()
        return super().__new__(cls)

    def __init__(self, gpu: int, interval_s: float = 0.05) -> None:
        self.nvml = None
        self.counter = False
        self.interval_s = interval_s
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                self.counter = True
            except pynvml.NVMLError:
                self.counter = False
        except Exception:
            self.nvml = None

    def power_w(self) -> float | None:
        if self.nvml is None:
            return None
        try:
            return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return None

    def start(self) -> None:
        self.t0 = time.time()
        self.samples: list[float] = []
        self.stop_ev = threading.Event()
        self.mj0 = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) if self.counter else None

        def loop():
            while not self.stop_ev.is_set():
                p = self.power_w()
                if p is not None:
                    self.samples.append(p)
                self.stop_ev.wait(self.interval_s)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self) -> dict:
        self.stop_ev.set()
        self.thread.join()
        dt = time.time() - self.t0
        mean_w = statistics.fmean(self.samples) if self.samples else None
        if self.counter:
            joules = (self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) - self.mj0) / 1000.0
        else:
            joules = mean_w * dt if mean_w is not None else None
        return {"seconds": dt, "energy_J": joules, "power_mean_W": mean_w,
                "energy_source": "counter" if self.counter else ("sampled" if mean_w is not None else "none")}


class Client:
    def __init__(self, url: str, timeout_s: float = 600.0) -> None:
        u = urllib.parse.urlsplit(url)
        self.host, self.port, self.timeout_s = u.hostname or "127.0.0.1", u.port or 80, timeout_s

    def post(self, body: bytes) -> tuple[float, dict]:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/call", body=body, headers={"Content-Type": "application/json"})
            raw = conn.getresponse().read()
        finally:
            conn.close()
        dt = time.perf_counter() - t0
        return dt, json.loads(raw)

    def rpc(self, method: str, *args, **kwargs):
        _, resp = self.post(json.dumps({"method": method, "args": list(args), "kwargs": kwargs,
                                        "session_id": None}).encode())
        if not resp.get("ok"):
            raise RuntimeError(f"{method}: {resp.get('error')}")
        return resp.get("result")


def load_bodies(dirs: list[str], method: str, limit: int) -> list[bytes]:
    files: list[Path] = []
    for d in dirs:
        p = Path(d)
        files += sorted((p / method).glob("*.json")) if (p / method).is_dir() else sorted(p.glob("*.json"))
    bodies = [f.read_bytes() for f in files[:limit] if f.stat().st_size > 0]
    if not bodies:
        sys.exit(f"no recorded {method} requests under {dirs}")
    return bodies


def batched_bodies(bodies: list[bytes], b: int, n: int) -> list[bytes]:
    """n request bodies, each merging b recorded observations of the same structure."""
    from rsched.batching import decode, encode, merge, signature

    groups: dict = {}
    for body in bodies:
        req = json.loads(body)
        args = req.get("args") or []
        obs = decode(args[0])
        opts = decode(args[1]) if len(args) > 1 else None
        sig = signature(obs, opts)
        if sig is not None:
            groups.setdefault(sig, []).append((obs, opts))
    if not groups:
        raise ValueError("no batchable vla.predict requests recorded")
    pool = max(groups.values(), key=len)
    cyc = itertools.cycle(pool)
    out = []
    for _ in range(n):
        items = [next(cyc) for _ in range(b)]
        payload = {"method": "vla.predict", "args": [merge([o for o, _ in items])] +
                   ([items[0][1]] if items[0][1] is not None else []), "kwargs": {}, "session_id": None}
        out.append(encode(payload))
    return out


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def run_phase(client: Client, energy: Energy, label: str, bodies: list[bytes], n: int, items_per_call: int,
              calls_log: JsonlWriter, idle_w: float | None) -> dict:
    client.rpc("prof.mark", label)
    lat = []
    energy.start()
    t0 = time.time()
    for i, body in zip(range(n), itertools.cycle(bodies)):
        dt, resp = client.post(body)
        if not resp.get("ok"):
            raise RuntimeError(f"{label} call {i}: {resp.get('error')}")
        lat.append(dt)
        calls_log.write({"phase": label, "i": i, "latency_s": dt, "items": items_per_call})
    wall = time.time() - t0
    e = energy.stop()
    items = n * items_per_call
    row = {"phase": label, "batch": items_per_call, "calls": n, "items": items, "wall_s": wall,
           "lat_mean_ms": 1e3 * statistics.fmean(lat), "lat_p50_ms": 1e3 * pct(lat, 0.5),
           "lat_p95_ms": 1e3 * pct(lat, 0.95), "items_per_s": items / wall, **e, "idle_power_W": idle_w}
    if e["energy_J"] is not None:
        row["J_per_item_gross"] = e["energy_J"] / items
        if idle_w is not None:
            row["J_per_item_net"] = (e["energy_J"] - idle_w * e["seconds"]) / items
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--service", required=True, choices=["vla", "molmo", "sam3"])
    ap.add_argument("--url", required=True, help="the profiled server itself, not a proxy")
    ap.add_argument("--requests", nargs="+", required=True, help="dump dir(s) from the proxy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-requests", type=int, default=200, help="distinct recorded requests to cycle through")
    ap.add_argument("--gpu", type=int, default=0, help="NVML index of the GPU the server uses")
    ap.add_argument("--idle-s", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--calls", type=int, default=100)
    ap.add_argument("--stage-calls", type=int, default=30)
    ap.add_argument("--torch-calls", type=int, default=5)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8, 16], help="vla only")
    ap.add_argument("--batch-calls", type=int, default=30)
    args = ap.parse_args(argv)

    method = {"vla": "vla.predict", "molmo": "molmo.ground", "sam3": "sam3.segment"}[args.service]
    bodies = load_bodies(args.requests, method, args.max_requests)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client, energy = Client(args.url), Energy(args.gpu)
    calls_log = JsonlWriter(out / "calls.jsonl")
    client.rpc("healthz")
    print(f"[{args.service}] {len(bodies)} recorded requests; energy from "
          f"{getattr(energy, 'source', None) or ('NVML counter' if energy.counter else 'sampled power' if energy.nvml else 'nowhere (no NVML)')}",
          flush=True)

    rows = []
    client.rpc("prof.mark", "idle")
    energy.start()
    time.sleep(args.idle_s)
    idle = energy.stop()
    idle_w = idle["energy_J"] / idle["seconds"] if idle["energy_J"] is not None else None
    rows.append({"phase": "idle", "batch": 0, "calls": 0, "items": 0, **idle, "idle_power_W": idle_w})

    run_phase(client, energy, "warmup", bodies, args.warmup, 1, calls_log, idle_w)
    rows.append(run_phase(client, energy, "timing", bodies, args.calls, 1, calls_log, idle_w))
    if args.stage_calls:
        client.rpc("prof.set", stages=True)
        rows.append(run_phase(client, energy, "stages", bodies, args.stage_calls, 1, calls_log, idle_w))
        client.rpc("prof.set", stages=False)
    if args.torch_calls:
        client.rpc("prof.torch_start", str(out / "torch"), args.service)
        try:
            rows.append(run_phase(client, energy, "torch", bodies, args.torch_calls, 1, calls_log, idle_w))
        finally:
            info = client.rpc("prof.torch_stop")
        print(f"[{args.service}] torch profile: {info}", flush=True)
    if args.service == "vla":
        for b in args.batches:
            merged = batched_bodies(bodies, b, min(args.batch_calls, 64))
            run_phase(client, energy, f"warmup_b{b}", merged, 3, b, calls_log, idle_w)
            rows.append(run_phase(client, energy, f"batch_{b}", merged, args.batch_calls, b, calls_log, idle_w))
    client.rpc("prof.mark", "done")
    calls_log.close()

    import csv

    cols = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("phase", "batch", "calls", "items"), k))
    with open(out / "client.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        if r["phase"] != "idle":
            print(f"[{args.service}] {r['phase']:>10s}: p50 {r['lat_p50_ms']:.1f} ms, {r['items_per_s']:.2f} items/s"
                  + (f", {r['J_per_item_net']:.2f} J/item net" if r.get("J_per_item_net") is not None else ""),
                  flush=True)
    print(f"-> {out}/client.csv")


if __name__ == "__main__":
    main()
