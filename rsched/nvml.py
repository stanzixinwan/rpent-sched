"""Sample GPU memory and utilization to JSONL at a fixed interval.

    python -m rsched.nvml --interval 0.5 --log results/run1/gpu.jsonl

Uses NVML (``pip install nvidia-ml-py``); falls back to parsing ``nvidia-smi``.
Per-process memory lets the analysis separate model services from the
per-episode simulators.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import threading

from rsched.traceio import JsonlWriter, now


def _sample_nvml(pynvml) -> list[dict]:
    out = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        procs = []
        try:
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                procs.append({"pid": p.pid, "used_mb": (p.usedGpuMemory or 0) / 2**20})
        except pynvml.NVMLError:
            pass
        out.append(
            {
                "gpu": i,
                "used_mb": mem.used / 2**20,
                "total_mb": mem.total / 2**20,
                "util_gpu": util.gpu,
                "util_mem": util.memory,
                "procs": procs,
            }
        )
    return out


def _sample_smi() -> list[dict]:
    q = "index,memory.used,memory.total,utilization.gpu,utilization.memory"
    text = subprocess.run(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout
    out = []
    for line in text.strip().splitlines():
        i, used, total, ug, um = (x.strip() for x in line.split(","))
        out.append(
            {"gpu": int(i), "used_mb": float(used), "total_mb": float(total),
             "util_gpu": float(ug), "util_mem": float(um), "procs": []}
        )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--log", required=True)
    args = p.parse_args(argv)

    try:
        import pynvml

        pynvml.nvmlInit()
        sample = lambda: _sample_nvml(pynvml)  # noqa: E731
        backend = "nvml"
    except Exception:
        sample = _sample_smi
        backend = "nvidia-smi"

    log = JsonlWriter(args.log)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    print(f"rsched gpu sampler: backend={backend} interval={args.interval}s", flush=True)
    while not stop.is_set():
        t = now()
        try:
            for rec in sample():
                log.write({"type": "gpu", "t": t, **rec})
        except Exception as exc:
            log.write({"type": "gpu_error", "t": t, "error": str(exc)})
        stop.wait(args.interval)
    log.close()


if __name__ == "__main__":
    main()
