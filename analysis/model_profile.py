"""Single-model characterization from rsched.profiled_server + rsched.bench_models output.

    python analysis/model_profile.py results/model_profile/<stamp> [--peak-tflops 312 --peak-gbps 2039]

Expects <root>/<service>/{server.jsonl, client.csv, torch/<service>.ops.csv}
for any of vla, molmo, sam3 (scripts/run_model_profile.sh lays it out so).
Writes <root>/analysis/:

    host_device.csv   per service, one call: RPC decode, model call, response encode
                      (timing phase), GPU span, GPU kernel busy time, kernel launches
                      and copies per call (torch phase). host_other_ms = model call
                      minus kernel busy: Python, launch overhead, waiting on copies.
    stages.csv        per hooked module: GPU ms and invocations per call, share of the
                      call's GPU span (stages phase). Modules nest: read one level.
    forwards.csv      top model forward calls per request: prefill (sequence > 1) vs
                      decode steps (sequence 1), count and ms (generate() workloads).
    ops.csv           GPU time per call by operator category (self device time of aten ops)
    roofline.csv      ops with a FLOP count (GEMMs, convolutions): FLOPs, estimated
                      bytes, arithmetic intensity, achieved TFLOP/s, bound by the
                      roofline of --peak-tflops / --peak-gbps (analytic roofline: bytes
                      are tensor sizes, not measured DRAM traffic)
    energy.csv        latency, throughput, J per call (gross and net of idle power)
    batch.csv         vla only: latency and J per item vs batch size

Peaks default to an A100-SXM4-80GB with bf16 tensor cores (312 TFLOP/s, 2039
GB/s); pass the right ones for another GPU or dtype (fp32 without TF32: 19.5).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

SERVICES = ("vla", "molmo", "sam3")
LAUNCH = re.compile(r"^(cudaLaunchKernel|cuLaunchKernel|cudaLaunchKernelExC|cudaLaunchCooperativeKernel)")
COPY = re.compile(r"memcpy|memset", re.I)
CATEGORIES = [
    ("GEMM", r"aten::(mm|addmm|bmm|baddbmm|matmul|linear|_scaled_mm|addbmm)$"),
    ("attention", r"(scaled_dot_product|flash|efficient_attention|_cudnn_attention|aten::softmax|_softmax)"),
    ("convolution", r"(convolution|cudnn_conv|conv2d|conv1d|conv3d)"),
    ("normalization", r"(layer_norm|rms_norm|group_norm|batch_norm|native_norm)"),
    ("activation", r"aten::(gelu|silu|relu|sigmoid|tanh|erf|exp|log|pow|rsqrt|sqrt|sin|cos)"),
    ("elementwise", r"aten::(add|sub|mul|div|neg|where|clamp|lerp|addcmul|fill_|zero_|masked_fill|abs|maximum|minimum)"),
    ("data movement", r"aten::(copy_|_to_copy|cat|stack|index|index_select|gather|scatter|embedding|contiguous|clone|"
                      r"repeat|expand|flip|roll|pad|constant_pad_nd|upsample|interpolate|_unsafe_index)"),
    ("reduction", r"aten::(sum|mean|max|min|argmax|topk|sort|cumsum|norm|var|std|any|all)"),
]
BYTES = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4, "torch.float64": 8, "torch.int8": 1}


def read_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.open():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def category(name: str) -> str:
    for cat, pat in CATEGORIES:
        if re.search(pat, name):
            return cat
    return "other"


def prod(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def mm_bytes(name: str, shapes: list, bpe: int) -> float:
    """Tensor bytes touched by one call: inputs plus the output of a GEMM-like op."""
    shapes = [s for s in shapes if isinstance(s, list) and s and all(isinstance(d, int) for d in s)]
    total = sum(prod(s) for s in shapes)
    out = 0
    if re.search(r"aten::(mm|matmul|linear)$", name) and len(shapes) >= 2:
        a, b = shapes[0], shapes[1]
        out = prod(a[:-1]) * (b[0] if name.endswith("linear") else b[-1])
    elif re.search(r"aten::(addmm)$", name) and len(shapes) >= 3:
        out = shapes[1][0] * shapes[2][-1]
    elif re.search(r"aten::(bmm|baddbmm)$", name) and len(shapes) >= 2:
        a, b = shapes[-2], shapes[-1]
        out = a[0] * a[1] * b[-1]
    return (total + out) * bpe


def service_tables(root: Path, svc: str, peak_tflops: float, peak_gbps: float) -> dict[str, pd.DataFrame]:
    d = root / svc
    log = read_jsonl(d / "server.jsonl") if (d / "server.jsonl").exists() else []
    calls = pd.DataFrame([r for r in log if r.get("type") == "call"])
    mods = next((r["modules"] for r in log if r.get("type") == "modules"), [])
    root_dtype = next((m.get("dtype") for m in mods if m["name"] == ""), None)
    bpe = BYTES.get(root_dtype or "", 2)
    out: dict[str, pd.DataFrame] = {}

    ops_path = d / "torch" / f"{svc}.ops.csv"
    ops = pd.read_csv(ops_path) if ops_path.exists() else pd.DataFrame()
    n_prof = int(ops["calls_profiled"].iloc[0]) if len(ops) else 0

    row = {"service": svc, "dtype": root_dtype}
    timing = calls[calls["phase"] == "timing"] if len(calls) else calls
    if len(timing):
        for col in ("decode_s", "method_s", "encode_s", "handler_s"):
            row[col.replace("_s", "_ms")] = 1e3 * timing[col].mean()
        row["gpu_span_ms"] = timing["gpu_span_ms"].mean()
        row["calls"] = len(timing)
    if n_prof:
        cuda = ops[ops["device_type"].astype(str).str.upper().str.contains("CUDA")]
        row["kernel_busy_ms"] = cuda["self_device_us"].sum() / n_prof / 1e3
        row["kernels_per_call"] = cuda["count"].sum() / n_prof
        row["launches_per_call"] = ops[ops["name"].str.match(LAUNCH)]["count"].sum() / n_prof
        row["launch_cpu_ms"] = ops[ops["name"].str.match(LAUNCH)]["self_cpu_us"].sum() / n_prof / 1e3
        row["copy_ms"] = cuda[cuda["name"].str.contains(COPY)]["self_device_us"].sum() / n_prof / 1e3
        if "method_ms" in row:
            row["host_other_ms"] = row["method_ms"] - row["kernel_busy_ms"]
    out["host_device"] = pd.DataFrame([row])

    stage = calls[calls["phase"] == "stages"] if len(calls) else calls
    if len(stage) and "stages" in stage:
        acc: dict[str, list[float]] = {}
        for st, span in zip(stage["stages"], stage["gpu_span_ms"]):
            if not isinstance(st, dict):
                continue
            for name, v in st.items():
                a = acc.setdefault(name, [0.0, 0.0, 0.0])
                a[0] += v["ms"]
                a[1] += v["n"]
                a[2] += span
        cls = {m["name"]: m["class"] for m in mods}
        params = {m["name"]: m["params"] for m in mods}
        n = len(stage)
        rows = [{"service": svc, "module": k or "(root)", "depth": 0 if k == "" else k.count(".") + 1,
                 "class": cls.get(k), "params": params.get(k), "ms_per_call": v[0] / n,
                 "invocations_per_call": v[1] / n, "share_of_span": v[0] / v[2] if v[2] else math.nan}
                for k, v in acc.items()]
        out["stages"] = pd.DataFrame(rows).sort_values(["depth", "ms_per_call"], ascending=[True, False])
        fw = []
        for fwd in stage.get("forwards", []):
            if not isinstance(fwd, list) or not fwd:
                continue
            pre = [f for f in fwd if (f.get("seq") or 0) > 1]
            dec = [f for f in fwd if f.get("seq") == 1]
            fw.append({"prefill_n": len(pre), "prefill_seq": sum(f["seq"] for f in pre) / len(pre) if pre else math.nan,
                       "prefill_ms": sum(f["ms"] for f in pre), "decode_steps": len(dec),
                       "decode_ms": sum(f["ms"] for f in dec)})
        if fw:
            f = pd.DataFrame(fw)
            out["forwards"] = pd.DataFrame([{
                "service": svc, "requests": len(f), "prefill_per_request": f["prefill_n"].mean(),
                "prefill_seq_mean": f["prefill_seq"].mean(), "prefill_ms_mean": f["prefill_ms"].mean(),
                "decode_steps_mean": f["decode_steps"].mean(), "decode_ms_mean": f["decode_ms"].mean(),
                "decode_ms_per_step": f["decode_ms"].sum() / max(f["decode_steps"].sum(), 1)}])

    if n_prof:
        aten = ops[ops["name"].str.startswith("aten::")].copy()
        # GPU runs time ops by their kernels; a CPU-only run (tests) has none, so use CPU time.
        on_gpu = aten["device_total_us"].sum() > 0
        self_col, total_col = ("self_device_us", "device_total_us") if on_gpu else ("self_cpu_us", "cpu_total_us")
        aten["category"] = aten["name"].map(category)
        cat = aten.groupby("category")[self_col].sum() / n_prof / 1e3
        total = cat.sum()
        out["ops"] = pd.DataFrame({"service": svc, "category": cat.index, "ms_per_call": cat.values,
                                   "share": cat.values / total if total else math.nan,
                                   "time_source": "device" if on_gpu else "cpu"}).sort_values(
            "ms_per_call", ascending=False)
        fl = aten[(aten["flops"] > 0) & (aten[total_col] > 0)].copy()
        if len(fl):
            fl["shapes"] = fl["input_shapes"].map(json.loads)
            fl["flops_per_op"] = fl["flops"] / fl["count"]
            fl["bytes_per_op"] = [mm_bytes(nm, sh, bpe) for nm, sh in zip(fl["name"], fl["shapes"])]
            fl["intensity"] = fl["flops_per_op"] / fl["bytes_per_op"].where(fl["bytes_per_op"] > 0)
            fl["us_per_op"] = fl[total_col] / fl["count"]
            fl["achieved_tflops"] = fl["flops_per_op"] / (fl["us_per_op"] * 1e6)
            ridge = peak_tflops * 1e12 / (peak_gbps * 1e9)
            fl["roof_tflops"] = [min(peak_tflops, i * peak_gbps / 1e3) if i == i else math.nan
                                 for i in fl["intensity"]]
            fl["bound"] = ["memory" if i < ridge else "compute" for i in fl["intensity"].fillna(0)]
            fl["ms_per_call"] = fl[total_col] / n_prof / 1e3
            fl["service"] = svc
            fl["time_source"] = "device" if on_gpu else "cpu"
            out["roofline"] = fl[["service", "name", "input_shapes", "count", "flops_per_op", "bytes_per_op",
                                  "intensity", "us_per_op", "achieved_tflops", "roof_tflops", "bound",
                                  "ms_per_call", "time_source"]].sort_values("ms_per_call", ascending=False)

    if (d / "client.csv").exists():
        cl = pd.read_csv(d / "client.csv").assign(service=svc)
        keep = [c for c in ("service", "phase", "batch", "calls", "items", "lat_p50_ms", "lat_p95_ms",
                            "items_per_s", "power_mean_W", "idle_power_W", "J_per_item_gross",
                            "J_per_item_net", "energy_source") if c in cl]
        out["energy"] = cl[cl["phase"].isin(["idle", "timing"])][keep]
        b = cl[cl["phase"].astype(str).str.startswith("batch_")][keep]
        if len(b):
            base = b[b["batch"] == 1]
            if len(base):
                b = b.assign(lat_x_vs_b1=b["lat_p50_ms"] / float(base["lat_p50_ms"].iloc[0]),
                             throughput_x_vs_b1=b["items_per_s"] / float(base["items_per_s"].iloc[0]))
            out["batch"] = b
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--peak-tflops", type=float, default=312.0)
    ap.add_argument("--peak-gbps", type=float, default=2039.0)
    args = ap.parse_args(argv)
    root = Path(args.root)
    tables: dict[str, list[pd.DataFrame]] = {}
    found = [s for s in SERVICES if (root / s).is_dir()]
    if not found:
        sys.exit(f"no service directories ({', '.join(SERVICES)}) under {root}")
    for svc in found:
        for name, df in service_tables(root, svc, args.peak_tflops, args.peak_gbps).items():
            tables.setdefault(name, []).append(df)
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for name, dfs in tables.items():
        df = pd.concat(dfs, ignore_index=True)
        df.to_csv(out / f"{name}.csv", index=False)
        if name in ("host_device", "forwards", "energy", "batch"):
            print(f"== {name}\n{df.round(3).to_string(index=False)}")
        elif name == "ops":
            print(f"== ops (ms per call)\n{df.pivot_table(index='category', columns='service', values='ms_per_call').round(2).to_string()}")
        elif name == "stages":
            print(f"== stages, depth 1\n{df[df['depth'] <= 1].round(3).to_string(index=False)}")
        elif name == "roofline":
            g = df.groupby(["service", "bound"])["ms_per_call"].sum().unstack(fill_value=0)
            print(f"== analytic roofline: GPU ms per call in FLOP-counted ops by bound\n{g.round(2).to_string()}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
