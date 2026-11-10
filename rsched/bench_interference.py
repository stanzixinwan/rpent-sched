"""B feasibility: how much do a local LLM planner and the robot models slow each other on one GPU?

    python -m rsched.bench_interference --out results/b_feasibility/<stamp> \
        --scenarios vla=1 llm=4 vla=1,llm=4 ...

Plan B (project doc 投稿评估.md): put the planner on the same GPU as Pi0.5 and
Molmo, and use the planner's tool calls (e.g. ``pi0_pick(max_chunks=12)``) as
advance notice to move GPU time between them. That is only worth building if
the models interfere. This measures it with closed-loop synthetic load, straight
to the backends (no rsched proxy, no simulator):

    vla     Pi0.5 ``vla.predict`` on a 256x256 LIBERO-shaped observation, then
            ``--vla-gap`` seconds (the env.chunk_step a real pick does)
    molmo   ``molmo.ground`` on a 1024x1024 synthetic scene, then ``--molmo-gap``
    llm     a streamed chat completion from the OpenAI-compatible planner server
            (vLLM): one image plus a ~``--llm-prompt-tokens`` prompt (a shared
            prefix and a unique suffix), exactly ``--llm-output-tokens`` tokens out

A scenario such as ``vla=1,llm=4`` runs that many concurrent workers of each kind
for ``--warmup`` + ``--duration`` seconds; requests that start during warm-up are
dropped. Outputs in ``--out``:

    requests.jsonl   one record per request (scenario, kind, latency; LLM: TTFT, TPOT)
    scenarios.jsonl  scenario start and end times (to join with gpu.jsonl)
    gpu.jsonl        GPU utilization and memory samples (rsched.nvml)
    summary.csv      per scenario and kind: count, throughput, latency percentiles
    inflation.csv    each kind against the scenario where it ran alone at the same count
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from rsched.traceio import JsonlWriter, now

KINDS = ("vla", "molmo", "llm")
DEFAULT_SCENARIOS = [
    "vla=1", "vla=4", "molmo=1", "llm=1", "llm=4", "llm=16",
    "vla=1,llm=1", "vla=1,llm=4", "vla=1,llm=16", "vla=4,llm=4",
    "molmo=1,llm=4", "molmo=1,llm=16", "vla=1,molmo=1,llm=4",
]


def parse_scenario(text: str) -> dict[str, int]:
    out = {}
    for part in text.split(","):
        kind, _, count = part.partition("=")
        if kind not in KINDS or not count.isdigit():
            raise ValueError(f"bad scenario part {part!r}; use e.g. vla=1,llm=4")
        out[kind] = int(count)
    return out


def _opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------


class VlaWorker:
    def __init__(self, endpoint: str, gap_s: float, seed: int) -> None:
        from rpent.robots.components.pi05_vla_client import Pi05VLAClient
        from rpent.utils.rpc.http_rpc import HttpRpcClient

        self.client = Pi05VLAClient(HttpRpcClient(endpoint), embodiment="libero")
        self.gap_s = gap_s
        rng = np.random.default_rng(seed)
        self.obs = {
            "main_images": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "wrist_images": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
            "states": rng.normal(size=8).astype(np.float32),
            "task_descriptions": "pick up the black bowl and place it on the plate",
        }

    def once(self) -> dict:
        t0 = now()
        self.client.predict(self.obs)
        t1 = now()
        if self.gap_s > 0:
            time.sleep(self.gap_s)
        return {"t_start": t0, "lat_s": t1 - t0}


def synthetic_scene(size: int = 1024) -> np.ndarray:
    img = np.full((size, size, 3), 180, dtype=np.uint8)
    img[size // 3: size // 2, size // 3: size // 2] = (200, 30, 30)       # red block
    img[size // 2: 2 * size // 3, size // 2: 3 * size // 4] = (30, 30, 200)  # blue plate
    return img


class MolmoWorker:
    def __init__(self, endpoint: str, gap_s: float) -> None:
        from rpent.robots.components.molmo_client import MolmoClient
        from rpent.utils.rpc.http_rpc import HttpRpcClient

        self.client = MolmoClient(HttpRpcClient(endpoint))
        self.gap_s = gap_s
        self.image = synthetic_scene()

    def once(self) -> dict:
        t0 = now()
        self.client.ground(self.image, "the red block")
        t1 = now()
        if self.gap_s > 0:
            time.sleep(self.gap_s)
        return {"t_start": t0, "lat_s": t1 - t0}


def _png_b64(img: np.ndarray) -> str:
    import imageio.v2 as imageio

    buf = io.BytesIO()
    imageio.imwrite(buf, img, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


WORDS = ("robot gripper bowl plate drawer cabinet stove pick place move rotate open close left right "
         "above below near far table object camera wrist agent view position target anchor").split()


def filler(n_words: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


class LlmWorker:
    """Streamed chat completions; TTFT and inter-token gaps from the SSE stream."""

    def __init__(self, base_url: str, model: str, prompt_tokens: int, output_tokens: int,
                 image: bool, seed: int, shared_prefix: str) -> None:
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.output_tokens = output_tokens
        self.rng = random.Random(seed)
        self.unique_words = int(prompt_tokens * 0.4 * 0.75)
        self.prefix = shared_prefix
        self.image_b64 = _png_b64(synthetic_scene(512)) if image else None

    def once(self) -> dict:
        text = (f"{self.prefix}\n\nObservation {self.rng.random():.9f}: "
                f"{filler(self.unique_words, self.rng)}\n\nNext tool call?")
        content = ([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self.image_b64}"}},
                    {"type": "text", "text": text}] if self.image_b64 else text)
        body = {"model": self.model, "stream": True, "max_tokens": self.output_tokens,
                "temperature": 0.0, "ignore_eos": True, "min_tokens": self.output_tokens,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": content}]}
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = now()
        token_times: list[float] = []
        usage = {}
        with _opener().open(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                msg = json.loads(data)
                if msg.get("usage"):
                    usage = msg["usage"]
                for ch in msg.get("choices") or []:
                    if (ch.get("delta") or {}).get("content"):
                        token_times.append(now())
        t1 = now()
        gaps = np.diff(token_times) if len(token_times) > 1 else np.array([])
        return {"t_start": t0, "lat_s": t1 - t0,
                "ttft_s": (token_times[0] - t0) if token_times else None,
                "tpot_s": float(gaps.mean()) if len(gaps) else None,
                "tpot_p95_s": float(np.quantile(gaps, 0.95)) if len(gaps) else None,
                "chunks": len(token_times),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens")}


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------


def run_scenario(name: str, counts: dict[str, int], make, writer: JsonlWriter, warmup_s: float,
                 duration_s: float) -> tuple[float, float]:
    """Record every request that starts inside the window, including ones that end after it.

    When the window closes, each worker finishes the request it has in flight
    (still under load from the others) and then stops, so slow requests near the
    end are not dropped.
    """
    closed = threading.Event()
    t_begin = now()
    t_measure = t_begin + warmup_s
    t_close = t_measure + duration_s

    def loop(kind: str, idx: int) -> None:
        worker = make(kind, idx)
        while not closed.is_set():
            try:
                rec, err = worker.once(), None
            except Exception as exc:  # keep going; record the failure
                rec, err = {"t_start": now(), "lat_s": None}, f"{type(exc).__name__}: {exc}"[:300]
                time.sleep(0.5)
            if t_measure <= rec["t_start"] < t_close:
                writer.write({"type": "request", "scenario": name, "kind": kind, "worker": idx,
                              "error": err, **rec})

    threads = [threading.Thread(target=loop, args=(k, i), daemon=True)
               for k, c in counts.items() for i in range(c)]
    for th in threads:
        th.start()
    time.sleep(max(t_close - now(), 0.0))
    closed.set()
    for th in threads:
        th.join(timeout=650)
    return t_measure, t_close


def summarize(out: Path) -> None:
    import pandas as pd

    req = pd.read_json(out / "requests.jsonl", lines=True)
    sc = pd.read_json(out / "scenarios.jsonl", lines=True).set_index("scenario")
    ok = req[req["error"].isna()].copy()
    rows = []
    for (scen, kind), g in ok.groupby(["scenario", "kind"], sort=False):
        span = sc.loc[scen, "t_end"] - sc.loc[scen, "t_measure"]
        counts = sc.loc[scen, "counts"]
        row = {"scenario": scen, "kind": kind, "workers": counts.get(kind, 0),
               "others": ",".join(f"{k}={v}" for k, v in counts.items() if k != kind) or "-",
               "n": len(g), "per_s": len(g) / span,
               "lat_p50_s": g["lat_s"].median(), "lat_p95_s": g["lat_s"].quantile(0.95)}
        if kind == "llm":
            row.update(ttft_p50_s=g["ttft_s"].median(), ttft_p95_s=g["ttft_s"].quantile(0.95),
                       tpot_p50_ms=g["tpot_s"].median() * 1e3, tpot_p95_ms=g["tpot_p95_s"].median() * 1e3,
                       out_tok_per_s=g["completion_tokens"].fillna(g["chunks"]).sum() / span)
        rows.append(row)
    s = pd.DataFrame(rows)
    errors = req["error"].notna().groupby([req["scenario"], req["kind"]]).sum()
    s["errors"] = [int(errors.get((r.scenario, r.kind), 0)) for r in s.itertuples()]
    s.to_csv(out / "summary.csv", index=False)

    alone = s[s["others"] == "-"].set_index(["kind", "workers"])
    inf = []
    for r in s[s["others"] != "-"].itertuples(index=False):
        if (r.kind, r.workers) not in alone.index:
            continue
        b = alone.loc[(r.kind, r.workers)]
        rec = {"kind": r.kind, "workers": r.workers, "with": r.others,
               "lat_p50_x": r.lat_p50_s / b["lat_p50_s"], "lat_p95_x": r.lat_p95_s / b["lat_p95_s"],
               "throughput_x": r.per_s / b["per_s"]}
        if r.kind == "llm":
            rec.update(ttft_p50_x=r.ttft_p50_s / b["ttft_p50_s"], tpot_p50_x=r.tpot_p50_ms / b["tpot_p50_ms"])
        inf.append(rec)
    inflation = pd.DataFrame(inf)
    inflation.to_csv(out / "inflation.csv", index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(s.to_string(index=False))
        if not inflation.empty:
            print("\nslowdown against the same kind and count running alone (x = ratio; throughput < 1 is worse):")
            print(inflation.to_string(index=False))
    print(f"\nwrote {out}/summary.csv, inflation.csv")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--vla-endpoint", default=f"http://127.0.0.1:{os.environ.get('VLA_PORT', '9000')}")
    ap.add_argument("--molmo-endpoint", default=f"http://127.0.0.1:{os.environ.get('MOLMO_PORT', '9002')}")
    ap.add_argument("--llm-url", default=f"http://127.0.0.1:{os.environ.get('LLM_PORT', '9003')}")
    ap.add_argument("--llm-model", default=os.environ.get("LLM_SERVED_NAME", "planner"))
    ap.add_argument("--llm-prompt-tokens", type=int, default=4000)
    ap.add_argument("--llm-output-tokens", type=int, default=256)
    ap.add_argument("--llm-no-image", action="store_true")
    ap.add_argument("--vla-gap", type=float, default=0.17, help="s after each predict (G2: chunk_step ~0.17 s)")
    ap.add_argument("--molmo-gap", type=float, default=0.0)
    ap.add_argument("--no-gpu-sampler", action="store_true")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(out)
        return
    scenarios = [(s, parse_scenario(s)) for s in args.scenarios]
    shared = filler(int(args.llm_prompt_tokens * 0.6 * 0.75), random.Random(0))
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    def make(kind: str, idx: int):
        if kind == "vla":
            return VlaWorker(args.vla_endpoint, args.vla_gap, seed=idx)
        if kind == "molmo":
            return MolmoWorker(args.molmo_endpoint, args.molmo_gap)
        return LlmWorker(args.llm_url, args.llm_model, args.llm_prompt_tokens, args.llm_output_tokens,
                         not args.llm_no_image, seed=1000 + idx, shared_prefix=shared)

    sampler = None
    if not args.no_gpu_sampler:
        sampler = subprocess.Popen([sys.executable, "-m", "rsched.nvml", "--interval", "0.25",
                                    "--log", str(out / "gpu.jsonl")])
    writer = JsonlWriter(out / "requests.jsonl")
    scen_log = JsonlWriter(out / "scenarios.jsonl")
    try:
        for k, (name, counts) in enumerate(scenarios, 1):
            print(f"[{k}/{len(scenarios)}] {name}: {args.warmup:.0f} s warm-up + {args.duration:.0f} s",
                  flush=True)
            t_measure, t_end = run_scenario(name, counts, make, writer, args.warmup, args.duration)
            scen_log.write({"scenario": name, "counts": counts, "t_measure": t_measure, "t_end": t_end})
    finally:
        writer.close()
        scen_log.close()
        if sampler is not None:
            sampler.terminate()
            sampler.wait(timeout=10)
    summarize(out)


if __name__ == "__main__":
    main()
