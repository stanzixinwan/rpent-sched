"""Run an RPent model server with per-call timing, stage hooks and an on-demand profiler.

    python -m rsched.profiled_server --service vla --log <dir>/server.jsonl -- \
        rpent.robots.components.pi05_vla_server --embodiment libero --transport http --port 9100
    python -m rsched.profiled_server --service molmo --log ... -- rsched.molmo_server --transport http --port 9102

Everything after ``--`` is the module (or a ``.py`` path) to run and its own arguments, exactly as
``scripts/launch_services.sh`` starts it. RPent is not modified: before the
module runs, its RPC classes are patched in this process only.

Per model call (``vla.predict``, ``molmo.ground``, ``sam3.segment``) the log gets
one ``call`` record:

    decode_s        HTTP handler start to model call (read body, JSON decode, ndarray decode, lock)
    method_s        the model call itself (Python, launches, GPU, copies back to the host)
    gpu_span_ms     CUDA events around the call: first to last GPU work, idle gaps included
    encode_s        result to response written (JSON encode, socket write)
    stages          (when stages are on) GPU ms and invocations per hooked module
    forwards        (when stages are on) one entry per call of the top model's forward:
                    sequence length and GPU ms (prefill vs decode for generate())

Control RPCs added to the server (sent by ``rsched.bench_models``):

    prof.mark(label)                   tag later call records with a phase label
    prof.set(stages=bool)              switch the module hooks on or off
    prof.torch_start(out_dir, label)   start torch.profiler (CPU + CUDA, shapes, FLOPs)
    prof.torch_stop()                  stop it; write <label>.trace.json.gz and <label>.ops.csv

Stage hooks cover the model root and every module up to ``--stage-depth``
levels below it, plus names matching ``--stage-match``. On a GPU they time with
CUDA events (no extra synchronization inside the call); without one, with the
host clock. A ``modules`` record at start lists the tree (class, parameters,
dtype) so the analysis can map names to stages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import runpy
import sys
import threading
import time
from pathlib import Path
from typing import Any

MODEL_METHODS = ("vla.predict", "molmo.ground", "sam3.segment")
_tls = threading.local()


class Executor:
    """One worker thread that runs every model call and the torch profiler.

    torch.profiler must start, record and stop on one thread, but RPent's HTTP
    server handles each request on a new one. Model calls therefore run here,
    one at a time (the benchmark sends them one at a time anyway); the handoff
    happens outside the timed region.
    """

    def __init__(self) -> None:
        import queue

        self._q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name="profiled-model").start()

    def _loop(self) -> None:
        while True:
            fn, box, done = self._q.get()
            try:
                box["value"] = fn()
            except BaseException as exc:  # handed back to the caller
                box["error"] = exc
            done.set()

    def call(self, fn):
        box: dict = {}
        done = threading.Event()
        self._q.put((fn, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["value"]


class _Log:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, default=str)
        with self._lock:
            self._fh.write(line + "\n")


class Profiler:
    """Shared state of the patched server."""

    def __init__(self, service: str, log: _Log, stage_depth: int, stage_match: list[str]) -> None:
        self.service, self.log = service, log
        self.stage_depth, self.stage_match = stage_depth, [re.compile(p) for p in stage_match]
        self.phase = "default"
        self.stages_on = False
        self.hooks_installed = False
        self.torch_prof = None
        self.torch_target: tuple[Path, str] | None = None
        self.torch_calls = 0
        self.n_calls = 0
        self.executor = Executor()

    # -- torch helpers -----------------------------------------------------
    @staticmethod
    def torch():
        import torch

        return torch

    def cuda(self) -> bool:
        try:
            return self.torch().cuda.is_available()
        except Exception:
            return False

    def marker(self):
        """A timestamp source: a recorded CUDA event on GPU, the host clock otherwise."""
        if self.cuda():
            ev = self.torch().cuda.Event(enable_timing=True)
            ev.record()
            return ev
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(a, b) -> float:
        if isinstance(a, float):
            return (b - a) * 1e3
        return float(a.elapsed_time(b))

    # -- module hooks --------------------------------------------------------
    def model_of(self, facade) -> Any:
        model = getattr(facade, "_model", None)
        torch = self.torch()
        return model if isinstance(model, torch.nn.Module) else None

    def describe(self, model) -> None:
        rows = []
        for name, mod in model.named_modules():
            if name.count(".") >= 3:
                continue
            params = list(mod.parameters(recurse=True))
            dtypes: dict[str, int] = {}
            for prm in params:
                dtypes[str(prm.dtype)] = dtypes.get(str(prm.dtype), 0) + prm.numel()
            rows.append({"name": name, "class": type(mod).__name__, "params": sum(p.numel() for p in params),
                         "dtype": max(dtypes, key=dtypes.get) if dtypes else None})
        self.log.write({"type": "modules", "service": self.service, "t": time.time(), "modules": rows})

    def selected(self, name: str) -> bool:
        if name == "":
            return True
        if name.count(".") < self.stage_depth:
            return True
        return any(p.search(name) for p in self.stage_match)

    def install_hooks(self, model) -> None:
        if self.hooks_installed:
            return
        self.hooks_installed = True
        for name, mod in model.named_modules():
            if not self.selected(name):
                continue
            mod.register_forward_pre_hook(self._pre(name), with_kwargs=True)
            mod.register_forward_hook(self._post(name), with_kwargs=True)

    def _pre(self, name: str):
        def hook(module, args, kwargs):
            st = getattr(_tls, "stage", None)
            if st is None:
                return None
            seq = None
            if name == "":
                for v in list(kwargs.values()) + list(args):
                    shape = getattr(v, "shape", None)
                    if shape is not None and len(shape) >= 2:
                        seq = int(shape[1])
                        break
            st["open"].setdefault(name, []).append((self.marker(), seq))
            return None
        return hook

    def _post(self, name: str):
        def hook(module, args, kwargs, output):
            st = getattr(_tls, "stage", None)
            if st is None or not st["open"].get(name):
                return None
            start, seq = st["open"][name].pop()
            st["done"].append((name, start, self.marker(), seq))
            return None
        return hook

    # -- wrapping the model method -------------------------------------------
    def wrap(self, method: str, handler):
        def wrapped(*args, **kwargs):
            result, pending = self.executor.call(lambda: self._timed(method, handler, args, kwargs))
            _tls.pending = pending
            return result
        return wrapped

    def _timed(self, method: str, handler, args, kwargs):
        """Runs on the executor thread. A call that raises is not logged; the error goes back."""
        t_enter = time.perf_counter()
        stage = {"open": {}, "done": []} if self.stages_on else None
        _tls.stage = stage
        begin = self.marker()
        try:
            result = handler(*args, **kwargs)
        finally:
            _tls.stage = None
        end = self.marker()
        t_exit = time.perf_counter()
        if not isinstance(end, float):
            end.synchronize()
        rec = {"type": "call", "service": self.service, "method": method, "phase": self.phase,
               "call": self.n_calls, "t": time.time(), "method_s": t_exit - t_enter,
               "gpu_span_ms": self.elapsed_ms(begin, end), "torch_profiled": self.torch_prof is not None}
        self.n_calls += 1
        if self.torch_prof is not None:
            self.torch_calls += 1
        if stage is not None:
            agg: dict[str, list[float]] = {}
            forwards = []
            for name, a, b, seq in stage["done"]:
                ms = self.elapsed_ms(a, b)
                cur = agg.setdefault(name, [0.0, 0])
                cur[0] += ms
                cur[1] += 1
                if name == "":
                    forwards.append({"seq": seq, "ms": ms})
            rec["stages"] = {k: {"ms": v[0], "n": v[1]} for k, v in agg.items()}
            rec["forwards"] = forwards
        return result, (rec, t_enter, t_exit)

    # -- control RPCs -----------------------------------------------------------
    def rpc_mark(self, label: str) -> dict:
        self.phase = str(label)
        self.log.write({"type": "mark", "t": time.time(), "label": self.phase})
        return {"ok": True}

    def rpc_set(self, stages: bool | None = None) -> dict:
        if stages is not None:
            self.stages_on = bool(stages)
        return {"stages": self.stages_on}

    def rpc_torch_start(self, out_dir: str, label: str) -> dict:
        return self.executor.call(lambda: self._torch_start(out_dir, label))

    def rpc_torch_stop(self) -> dict:
        return self.executor.call(self._torch_stop)

    def _torch_start(self, out_dir: str, label: str) -> dict:
        if self.torch_prof is not None:
            raise RuntimeError("torch profiler already running")
        torch = self.torch()
        acts = [torch.profiler.ProfilerActivity.CPU]
        if self.cuda():
            acts.append(torch.profiler.ProfilerActivity.CUDA)
        self.torch_target = (Path(out_dir), str(label))
        self.torch_calls = 0
        self.torch_prof = torch.profiler.profile(activities=acts, record_shapes=True, with_flops=True)
        self.torch_prof.__enter__()
        return {"ok": True}

    def _torch_stop(self) -> dict:
        if self.torch_prof is None:
            raise RuntimeError("torch profiler not running")
        prof, self.torch_prof = self.torch_prof, None
        prof.__exit__(None, None, None)
        out_dir, label = self.torch_target
        out_dir.mkdir(parents=True, exist_ok=True)
        trace = out_dir / f"{label}.trace.json"
        prof.export_chrome_trace(str(trace))
        with open(trace, "rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
            dst.write(src.read())
        trace.unlink()
        ops = out_dir / f"{label}.ops.csv"
        write_ops(prof, ops, self.torch_calls)
        self.log.write({"type": "torch", "t": time.time(), "label": label, "calls": self.torch_calls,
                        "ops": str(ops), "trace": str(trace) + ".gz"})
        return {"ops": str(ops), "trace": str(trace) + ".gz", "calls": self.torch_calls}


def _attr(evt, *names, default=0):
    for n in names:
        if hasattr(evt, n):
            return getattr(evt, n)
    return default


def write_ops(prof, path: Path, calls: int) -> None:
    """key_averages grouped by input shape; one row per (op or kernel, shapes)."""
    cols = ["name", "device_type", "input_shapes", "count", "calls_profiled", "self_cpu_us", "cpu_total_us",
            "self_device_us", "device_total_us", "flops"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in prof.key_averages(group_by_input_shape=True):
            w.writerow([e.key, str(_attr(e, "device_type", default="")).split(".")[-1],
                        json.dumps(_attr(e, "input_shapes", default=[]) or []), e.count, calls,
                        e.self_cpu_time_total, e.cpu_time_total,
                        _attr(e, "self_device_time_total", "self_cuda_time_total"),
                        _attr(e, "device_time_total", "cuda_time_total"), _attr(e, "flops") or 0])


def install(prof: Profiler) -> None:
    """Patch RPent's RPC facade and HTTP handler in this process."""
    from rpent.utils.rpc import http_rpc
    from rpent.utils.rpc.rpc_facade import RpcFacade

    orig_serve = RpcFacade.serve

    def serve(self, *args, **kwargs):
        model = prof.model_of(self)
        if model is not None:
            prof.describe(model)
            prof.install_hooks(model)
        for m in MODEL_METHODS:
            if m in self._rpc:
                self._rpc[m] = prof.wrap(m, self._rpc[m])
        self._rpc["prof.mark"] = prof.rpc_mark
        self._rpc["prof.set"] = prof.rpc_set
        self._rpc["prof.torch_start"] = prof.rpc_torch_start
        self._rpc["prof.torch_stop"] = prof.rpc_torch_stop
        prof.log.write({"type": "start", "service": prof.service, "t": time.time(), "pid": os.getpid(),
                        "cuda": prof.cuda(), "methods": sorted(self._rpc)})
        return orig_serve(self, *args, **kwargs)

    RpcFacade.serve = serve

    handler_cls = http_rpc._HttpRpcHandler
    orig_post = handler_cls.do_POST

    def do_POST(self):  # noqa: N802
        t0 = time.perf_counter()
        _tls.pending = None
        orig_post(self)
        pending, _tls.pending = getattr(_tls, "pending", None), None
        if pending is None:
            return
        rec, t_enter, t_exit = pending
        rec["decode_s"] = t_enter - t0
        rec["encode_s"] = time.perf_counter() - t_exit
        rec["handler_s"] = time.perf_counter() - t0
        prof.log.write(rec)

    handler_cls.do_POST = do_POST


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        sys.exit("usage: python -m rsched.profiled_server [options] -- <module> <module args>")
    cut = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--service", required=True, choices=["vla", "molmo", "sam3"])
    ap.add_argument("--log", required=True)
    ap.add_argument("--stage-depth", type=int, default=2, help="hook modules up to this depth below the root")
    ap.add_argument("--stage-match", nargs="*", default=[], help="also hook modules whose name matches")
    args = ap.parse_args(argv[:cut])
    module, mod_args = argv[cut + 1], argv[cut + 2:]
    prof = Profiler(args.service, _Log(args.log), args.stage_depth, args.stage_match)
    install(prof)
    sys.argv = [module, *mod_args]
    if module.endswith(".py"):  # a script path (tests); a module name otherwise
        runpy.run_path(module, run_name="__main__")
    else:
        runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
