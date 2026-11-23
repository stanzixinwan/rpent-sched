"""Jetson sampler: tegrastats (power rails, RAM, CPU) plus the GPU load from sysfs.

    python -m rsched.tegra --interval 0.5 --log results/run1/gpu.jsonl

NVML reports "Not Supported" on Jetson, so rsched.sweep runs this instead of
rsched.nvml when RS_GPU_SAMPLER=tegra (env/.env on Thor/Orin). Records use the
same fields as rsched.nvml (``util_gpu``, ``used_mb``, ``total_mb``, ``power_w``)
so analysis/gpu_util.py reads them unchanged:

  power_w   board input power (tegrastats ``VIN``, or the sum of the rails when
            the board does not report VIN), in watts
  rails     every power rail tegrastats prints, in watts (VDD_GPU, VDD_CPU_SOC_MSS, ...)
  used_mb   RAM in use (CPU and GPU share it on Jetson)
  util_gpu  GPU load in percent: tegrastats GR3D_FREQ when printed, otherwise
            the devfreq/GPU ``load`` file (permille) when one is readable, else None
  cpu_util  per-core CPU load in percent
"""

from __future__ import annotations

import argparse
import glob
import re
import signal
import subprocess
import threading

from rsched.traceio import JsonlWriter, now

RAIL = re.compile(r"\b([A-Z][A-Z0-9_]+) (\d+)mW/(\d+)mW")
RAM = re.compile(r"RAM (\d+)/(\d+)MB")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
CPU = re.compile(r"CPU \[([^\]]*)\]")
LOAD_FILES = ["/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
              "/sys/devices/platform/17000000.gpu/load", "/sys/devices/platform/bus@0/17000000.gpu/load"]


def find_load_file() -> str | None:
    for path in LOAD_FILES + sorted(glob.glob("/sys/class/devfreq/*gpu*/device/load")) \
            + sorted(glob.glob("/sys/devices/platform/**/gpu*/load", recursive=True)):
        try:
            int(open(path).read().strip())
            return path
        except (OSError, ValueError):
            continue
    return None


def parse(line: str, load_file: str | None = None) -> dict | None:
    m = RAM.search(line)
    if not m:
        return None
    rails = {k: int(cur) / 1000.0 for k, cur, _ in RAIL.findall(line)}
    power = rails.get("VIN")
    if power is None and rails:
        power = sum(v for k, v in rails.items() if k.startswith("VDD"))
    util = None
    g = GR3D.search(line)
    if g:
        util = float(g.group(1))
    elif load_file:
        try:
            util = int(open(load_file).read().strip()) / 10.0
        except (OSError, ValueError):
            util = None
    cpu = []
    c = CPU.search(line)
    if c:
        for part in c.group(1).split(","):
            pct = part.split("%")[0].strip()
            cpu.append(float(pct) if pct.replace(".", "", 1).isdigit() else None)
    return {"type": "gpu", "gpu": 0, "used_mb": float(m.group(1)), "total_mb": float(m.group(2)),
            "util_gpu": util, "util_mem": None, "power_w": power, "rails": rails, "cpu_util": cpu,
            "procs": []}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--log", required=True)
    args = p.parse_args(argv)
    load_file = find_load_file()
    log = JsonlWriter(args.log)
    log.write({"type": "sampler", "t": now(), "kind": "tegrastats", "gpu_load_file": load_file})
    proc = subprocess.Popen(["tegrastats", "--interval", str(int(args.interval * 1000))],
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())

    def reader():
        for line in proc.stdout:
            rec = parse(line, load_file)
            if rec is not None:
                log.write({**rec, "t": now()})
            if stop.is_set():
                break

    threading.Thread(target=reader, daemon=True).start()
    stop.wait()
    proc.terminate()
    log.close()


if __name__ == "__main__":
    main()
