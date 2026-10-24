"""Run concurrent RPent episodes at several concurrency levels.

    python -m rsched.sweep configs/p1_fifo_sweep.yaml [--levels 1 2 4] [--dry-run]

The model services (Pi0.5, SAM3, and Molmo for Flash) must already be running
(see ``scripts/launch_services.sh``); loading them takes minutes, so they are
shared across levels. For each level this script starts fresh proxies and a GPU
sampler, keeps N episodes in flight (closed loop) or launches them as a Poisson
process, and writes everything under ``results/<name>/<stamp>/N<level>_r<rep>/``:

    manifest.json           config, git commits, host and GPU info, timings
    proxy_vla.jsonl         per-request trace from the VLA proxy
    proxy_sam3.jsonl        per-request trace from the SAM3 proxy
    gpu.jsonl               GPU memory / utilization samples
    traces/episode_*.jsonl  per-episode tool calls, client RPCs, outcome
    episodes/<id>/          RPent's own output directory per episode
    logs/                   stdout/stderr of every process
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load_env_file(path: Path) -> list[str]:
    """Load KEY=VALUE lines from env/.env without overriding the shell.

    ``scripts/*.sh`` source this file, but ``python -m rsched.sweep`` run directly
    did not, so episodes started without the planner API key.
    """
    if not path.exists():
        return []
    os.environ.setdefault("RS_ROOT", str(REPO_ROOT))
    loaded = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = os.path.expandvars(value)
            loaded.append(key)
    return loaded


PLANNER_KEY_ENV = {"claude_code": "ANTHROPIC_API_KEY", "codex": "OPENAI_API_KEY"}


def preflight_planner(cfg: dict) -> None:
    """One cheap request through RPent's own checker before launching any episode."""
    r = cfg["rpent"]
    if "command" in r or r.get("planner") not in ("claude_code", "codex", "api"):
        return  # custom command (tests) or a planner the checker does not cover
    cmd = [sys.executable, "-c", "import sys; from rpent.cli.check_llm import main; sys.exit(main())",
           "--planner", r["planner"]]
    if r.get("model"):
        cmd += ["--model", r["model"]]
    print("preflight: checking planner credentials with rpent-check-llm ...", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    print("  " + (proc.stdout + proc.stderr).strip().replace("\n", "\n  "), flush=True)
    if proc.returncode != 0:
        key = PLANNER_KEY_ENV.get(r["planner"], "the planner's API key")
        sys.exit(f"preflight FAILED: the planner cannot authenticate. Set {key} in env/.env "
                 f"(currently {'set' if os.environ.get(key) else 'NOT set'}) and retry. "
                 "Skip with --skip-preflight.")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def git_rev(path: Path) -> str | None:
    try:
        rev = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return None


def gpu_info() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout
        return [line.strip() for line in out.strip().splitlines()]
    except Exception:
        return []


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_info() -> dict:
    """CPU visibility vs. quota. In containers ``os.cpu_count()`` is the host's
    count, so OpenMP/torch size their thread pools for cores the cgroup does
    not grant; ``cpu.stat`` throttle counters show the cost."""
    stat = _read("/sys/fs/cgroup/cpu.stat") or ""
    return {
        "os_cpu_count": os.cpu_count(),
        "affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "cgroup_cpu_max": _read("/sys/fs/cgroup/cpu.max"),
        "cgroup_cpu_stat": dict(
            (k, int(v)) for k, v in (line.split() for line in stat.splitlines() if line.strip())
        ),
        "thread_env": {k: os.environ[k] for k in THREAD_VARS if k in os.environ},
    }


THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def wait_http(url: str, timeout_s: float = 30.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.2)
    raise TimeoutError(f"{url} not ready after {timeout_s}s")


def check_backend(endpoint: str) -> None:
    """Call RPent's built-in ``healthz`` on a service (works through any proxy)."""
    body = json.dumps({"method": "healthz", "args": [], "kwargs": {}, "session_id": None}).encode()
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/call", data=body, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=10) as r:
        payload = json.loads(r.read())
    if not payload.get("ok"):
        raise RuntimeError(f"{endpoint} healthz failed: {payload}")


@dataclass
class EpisodeSpec:
    episode_id: str
    suite: str
    task: int
    seed: int


def episode_pool(cfg: dict) -> list[tuple[str, int, int]]:
    ep = cfg["episodes"]
    if "list" in ep:
        return [(e["suite"], int(e["task"]), int(e.get("seed", 0))) for e in ep["list"]]
    pool = list(itertools.product(ep["suites"], ep["tasks"], ep.get("seeds", [0])))
    if ep.get("shuffle_seed") is not None:
        random.Random(ep["shuffle_seed"]).shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# one level
# ---------------------------------------------------------------------------


class SweepAborted(RuntimeError):
    pass


def count_tool_calls(trace: Path) -> int:
    if not trace.exists():
        return 0
    with trace.open() as fh:
        return sum(1 for line in fh if '"type":"tool"' in line)


# Planner failures that will hit every later episode too; matched in the episode log.
FATAL_PATTERNS = {
    "billing_error": "billing error",
    "Credit balance is too low": "billing error",
    "authentication_failed": "authentication failure",
    "Not logged in": "authentication failure",
}


def fatal_planner_error(log: Path) -> str | None:
    if not log.exists():
        return None
    tail = log.read_text(errors="replace")[-20000:]
    return next((kind for pat, kind in FATAL_PATTERNS.items() if pat in tail), None)


def print_log_tail(path: Path, n: int = 25) -> None:
    if not path.exists():
        return
    lines = path.read_text(errors="replace").splitlines()[-n:]
    print(f"  ----- tail {path} -----")
    for line in lines:
        print("  | " + line)
    print("  -----", flush=True)


class Level:
    def __init__(self, cfg: dict, level: int, rep: int, out: Path, dry_run: bool) -> None:
        self.cfg, self.level, self.rep, self.out, self.dry_run = cfg, level, rep, out, dry_run
        # Abort after this many consecutive episodes that exit without a single tool call.
        self.fail_fast = 0 if dry_run else int(cfg.get("fail_fast", 2))
        self.consecutive_failures = 0
        self.procs: list[subprocess.Popen] = []
        self.logs = out / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        (out / "traces").mkdir(exist_ok=True)
        (out / "episodes").mkdir(exist_ok=True)

    def _spawn(self, name: str, cmd: list[str], env: dict | None = None) -> subprocess.Popen:
        log = open(self.logs / f"{name}.log", "w")
        log.write("$ " + shlex.join(cmd) + "\n")
        log.flush()
        if self.dry_run:
            print("[dry-run]", shlex.join(cmd))
        p = subprocess.Popen(
            cmd if not self.dry_run else [sys.executable, "-c", "pass"],
            stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})},
            cwd=REPO_ROOT, start_new_session=True,
        )
        return p

    def start_infra(self) -> dict[str, str]:
        """Start one proxy per service plus the GPU sampler; return client endpoints."""
        endpoints = {}
        for svc, scfg in self.cfg["services"].items():
            port = free_port()
            cmd = [
                sys.executable, "-m", "rsched.proxy",
                "--name", svc, "--listen", str(port), "--backend", scfg["backend"],
                "--policy", scfg.get("policy", "fifo"),
                "--max-inflight", str(scfg.get("max_inflight", 1)),
                "--aging", str(scfg.get("aging", 0.0)),
                "--log", str(self.out / f"proxy_{svc}.jsonl"),
            ]
            self.procs.append(self._spawn(f"proxy_{svc}", cmd))
            endpoints[svc] = f"http://127.0.0.1:{port}"
        if self.cfg.get("gpu_sampler", True):
            cmd = [sys.executable, "-m", "rsched.nvml", "--interval",
                   str(self.cfg.get("gpu_interval_s", 0.5)), "--log", str(self.out / "gpu.jsonl")]
            self.procs.append(self._spawn("gpu_sampler", cmd))
        if not self.dry_run:
            for svc, url in endpoints.items():
                wait_http(url + "/rs/stats")
                check_backend(url)
        return endpoints

    def episode_cmd(self, spec: EpisodeSpec, endpoints: dict[str, str]) -> list[str]:
        r = self.cfg["rpent"]
        cmd = list(r.get("command", [sys.executable, "-m", "rsched.launch"]))
        cmd += ["--robot", r.get("robot", "libero"),
                "--suite", spec.suite, "--task", str(spec.task), "--seed", str(spec.seed),
                "--planner", r["planner"],
                "--output-dir", str(self.out / "episodes" / spec.episode_id)]
        if r.get("model"):
            cmd += ["--model", r["model"]]
        if r.get("libero_type"):
            cmd += ["--libero-type", r["libero_type"]]
        if r.get("cuda_device") is not None:
            cmd += ["--cuda-device", str(r["cuda_device"])]
        if "vla" in endpoints:
            cmd += ["--vla-endpoint", endpoints["vla"]]
        if "sam3" in endpoints:
            cmd += ["--sam3-endpoint", endpoints["sam3"]]
        if "molmo" in endpoints:
            cmd += ["--molmo-endpoint", endpoints["molmo"]]
        cmd += [str(x) for x in r.get("extra_args", [])]
        return cmd

    def run(self) -> dict[str, Any]:
        endpoints = self.start_infra()
        pool = episode_pool(self.cfg)
        n_total = int(self.cfg["episodes_per_level"])
        timeout_s = float(self.cfg.get("episode_timeout_s", 3600))
        arrival = self.cfg.get("arrival", {"mode": "closed"})
        rng = random.Random(arrival.get("seed", 0))

        specs = [
            EpisodeSpec(f"N{self.level}r{self.rep}e{i:03d}", *pool[i % len(pool)])
            for i in range(n_total)
        ]
        pending = list(specs)
        running: dict[str, tuple[subprocess.Popen, float]] = {}
        finished: list[dict] = []
        next_arrival = time.time()
        t0 = time.time()

        def launch(spec: EpisodeSpec) -> None:
            # episode_env (e.g. thread caps) reaches rpent and the env_server it spawns.
            env = {k: str(v) for k, v in (self.cfg.get("episode_env") or {}).items()}
            env.update({"RS_EPISODE_ID": spec.episode_id, "RS_TRACE_DIR": str(self.out / "traces")})
            p = self._spawn(f"episode_{spec.episode_id}", self.episode_cmd(spec, endpoints), env)
            running[spec.episode_id] = (p, time.time())
            print(f"  start {spec.episode_id} {spec.suite} task={spec.task} seed={spec.seed} "
                  f"(running={len(running)})", flush=True)

        try:
            while pending or running:
                now_t = time.time()
                if arrival["mode"] == "closed":
                    while pending and len(running) < self.level:
                        launch(pending.pop(0))
                else:  # poisson: rate in episodes/minute, capped at `level` in flight
                    while pending and now_t >= next_arrival and len(running) < self.level:
                        launch(pending.pop(0))
                        next_arrival += rng.expovariate(arrival["rate_per_min"] / 60.0)
                for eid, (p, ts) in list(running.items()):
                    rc = p.poll()
                    if rc is None and now_t - ts > timeout_s:
                        os.killpg(p.pid, signal.SIGTERM)
                        rc = "timeout"
                    if rc is not None:
                        running.pop(eid)
                        n_tools = count_tool_calls(self.out / "traces" / f"episode_{eid}.jsonl")
                        finished.append({"episode": eid, "returncode": rc, "wall_s": now_t - ts,
                                         "tool_calls": n_tools})
                        log_path = self.logs / f"episode_{eid}.log"
                        fatal = fatal_planner_error(log_path)
                        if fatal:
                            status = f"  <-- PLANNER {fatal.upper()}"
                        elif n_tools == 0:
                            status = "  <-- CRASHED (no tool calls)"
                        elif rc != 0:
                            status = "  (exit != 0: task unsolved or planner stopped, e.g. max turns)"
                        else:
                            status = ""
                        print(f"  done  {eid} rc={rc} tools={n_tools} wall={now_t - ts:.1f}s{status}",
                              flush=True)
                        if fatal or n_tools == 0:
                            print_log_tail(log_path)
                        if fatal:
                            raise SweepAborted(
                                f"planner {fatal} (e.g. out of API credit or bad key); every further "
                                "episode would fail. Fix it, then rerun the remaining levels with --levels")
                        self.consecutive_failures = self.consecutive_failures + 1 if n_tools == 0 else 0
                        if self.consecutive_failures >= self.fail_fast > 0:
                            raise SweepAborted(
                                f"{self.consecutive_failures} episodes in a row exited without a single "
                                "tool call; aborting to save time and API cost (see log above)")
                time.sleep(0.5)
        except SweepAborted as exc:
            aborted = str(exc)
        else:
            aborted = None
        finally:
            for p, _ in running.values():
                os.killpg(p.pid, signal.SIGTERM)
            self.stop_infra()
        return {"t_start": t0, "t_end": time.time(), "episodes": finished,
                "specs": [s.__dict__ for s in specs], "endpoints": endpoints, "aborted": aborted}

    def stop_infra(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--levels", type=int, nargs="*", help="override concurrency levels")
    ap.add_argument("--repeats", type=int, help="override repeats")
    ap.add_argument("--out", default=None, help="output root (default results/<name>/<stamp>)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--skip-preflight", action="store_true", help="skip the planner credential check")
    args = ap.parse_args(argv)

    loaded = load_env_file(REPO_ROOT / "env" / ".env")
    if loaded:
        print(f"loaded from env/.env: {', '.join(sorted(loaded))}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not (args.dry_run or args.skip_preflight):
        preflight_planner(cfg)
    levels = args.levels or cfg["concurrency"]
    repeats = args.repeats or cfg.get("repeats", 1)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = Path(args.out) if args.out else REPO_ROOT / "results" / cfg["name"] / stamp
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.yaml").write_text(Path(args.config).read_text())

    for rep in range(repeats):
        for level in levels:
            out = root / f"N{level}_r{rep}"
            out.mkdir(parents=True, exist_ok=True)
            print(f"== level N={level} repeat={rep} -> {out}", flush=True)
            manifest = {
                "name": cfg["name"], "level": level, "repeat": rep, "config": cfg,
                "rsched_commit": git_rev(REPO_ROOT),
                "rpent_commit": git_rev(REPO_ROOT / "third_party" / "RPent"),
                "host": platform.node(), "platform": platform.platform(),
                "python": sys.version.split()[0], "gpus": gpu_info(),
                "episode_env": cfg.get("episode_env") or {}, "cpu_start": cpu_info(),
            }
            manifest.update(Level(cfg, level, rep, out, args.dry_run).run())
            manifest["cpu_end"] = cpu_info()
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
            if manifest["aborted"]:
                sys.exit(f"SWEEP ABORTED at N={level}: {manifest['aborted']}")
    print(f"all levels done: {root}")


if __name__ == "__main__":
    main()
