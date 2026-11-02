"""Run the ``rpent`` CLI with tracing hooks, without modifying RPent's source.

Usage (same arguments as ``rpent``)::

    RS_EPISODE_ID=ep0007 RS_TRACE_DIR=results/run1/traces \
    python -m rsched.launch --robot libero --suite libero_object_swap --task 2 --seed 0 \
        --planner codex --model gpt-5.6-sol \
        --vla-endpoint http://127.0.0.1:9100 --sam3-endpoint http://127.0.0.1:9101

What gets patched (all inside this one ``rpent`` process):

``Toolkit.execute_tool``
    the single funnel every planner (Claude Code, Codex, API) uses to run a
    tool. We log tool name, arguments, start/end time, and outcome. Gaps between
    tool calls are planner (LLM) time.
``Tool.__call__``
    captures the primitive's raw result (e.g. ``chunks_used`` of ``pi0_pick``)
    before the toolkit replaces it with an environment-state observation.
``HttpRpcClient.call``
    adds episode / tool-call headers to every RPC so the proxy can attribute
    model requests, and logs client-side latency for every RPC, including the
    simulator's ``env_server`` calls (simulator time).
``robots.libero.flash.replay.load`` (Flash only)
    reads the program the episode is about to replay and counts its remaining
    model work down as it runs (``rsched.intent``). Every tool start logs a
    ``remaining`` record; with ``RS_SCHED`` set, every model request carries the
    estimate as ``X-RS-Priority`` for the proxy's ``priority`` policy.
``RunFinalizationContext``
    records the episode outcome, token usage, and elapsed time. LIBERO does not
    define ``finalize_run``, and the CLI builds this context only when that hook
    is set, so robots without one get a no-op. Their own finalizer is left as is.

Validated against RPent commit ef70215 (see ``third_party/RPent``). If RPent's
internals move, ``check_compat`` fails loudly instead of tracing silently wrong.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from rsched import HDR_CALL, HDR_EPISODE, HDR_PRIORITY, HDR_TOOL
from rsched import intent as intent_mod
from rsched.traceio import JsonlWriter, now

_tls = threading.local()


class _Intent:
    """Per-process scheduling state: the program's remaining work and the mode."""

    def __init__(self) -> None:
        self.mode = "none"
        self.costs: dict[str, float] = {}
        self.remaining: intent_mod.Remaining | None = None
        self.history: intent_mod.HistoryEstimate | None = None
        self.trace: JsonlWriter | None = None
        self.episode = ""

    def configure(self, trace: JsonlWriter, episode: str) -> None:
        self.trace, self.episode = trace, episode
        self.mode = os.environ.get("RS_SCHED", "none") or "none"
        if self.mode not in intent_mod.MODES:
            raise ValueError(f"RS_SCHED={self.mode!r}; choose from {intent_mod.MODES}")
        self.costs = json.loads(os.environ.get("RS_COSTS") or "{}")
        self.history = intent_mod.HistoryEstimate(os.environ.get("RS_HISTORY_FILE"))

    def set_program(self, program: dict, plan_name: str = "") -> None:
        work = intent_mod.plan_work(program)
        self.remaining = intent_mod.Remaining(work, self.costs)
        if self.trace is not None:
            self.trace.write({"type": "plan_work", "episode": self.episode, "t": now(),
                              "plan": plan_name, "mode": self.mode, **work.to_dict(),
                              **self.remaining.snapshot()})

    def log_remaining(self, call_id: str, tool: str) -> None:
        if self.remaining is None or self.trace is None:
            return
        snap = self.remaining.snapshot()
        hist = self.history.remaining_s(snap["done_s"]) if self.history else None
        self.trace.write({"type": "remaining", "episode": self.episode, "t": now(),
                          "call_id": call_id, "tool": tool, **snap, "history_s": hist})

    def priority(self) -> float | None:
        return intent_mod.priority(self.mode, self.remaining, self.history)

    def finish(self) -> None:
        if self.remaining is not None and self.history is not None:
            self.history.record(self.episode, self.remaining.done_s())


INTENT = _Intent()


def set_program(program: dict, plan_name: str = "") -> None:
    """Start counting a program's remaining work (also the hook for tests and fakes)."""
    INTENT.set_program(program, plan_name)


def _hook_flash_load() -> None:
    """Wrap ``replay.load`` so the program is seen before its first model call."""
    from robots.libero.flash import replay

    orig_load = replay.load

    def load(root, plan_name):
        program = orig_load(root, plan_name)
        set_program(program, str(plan_name))
        return program

    replay.load = load
_RESULT_KEYS = ("name", "success", "chunks_used", "max_chunks", "executed_steps", "task_success")


def _jsonable(value: Any, limit: int = 2000) -> Any:
    try:
        text = json.dumps(value, default=repr)
    except Exception:
        return repr(value)[:limit]
    return value if len(text) <= limit else text[:limit] + "...[truncated]"


def check_compat() -> None:
    """Fail fast if the RPent internals we hook have changed."""
    import rpent.cli.main as cli
    from rpent.robots.robot_spec import RobotSpec
    from rpent.tools.base import Tool
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.rpc import http_rpc

    spec_fields = (
        {f.name for f in dataclasses.fields(RobotSpec)} if dataclasses.is_dataclass(RobotSpec) else set()
    )
    missing = [
        n
        for n, ok in [
            ("Toolkit.execute_tool", hasattr(Toolkit, "execute_tool")),
            ("Tool.__call__", "__call__" in Tool.__dict__),
            ("HttpRpcClient.call", hasattr(http_rpc.HttpRpcClient, "call")),
            ("http_rpc._NumpyEncoder", hasattr(http_rpc, "_NumpyEncoder")),
            ("http_rpc._from_json", hasattr(http_rpc, "_from_json")),
            ("http_rpc.check_response", hasattr(http_rpc, "check_response")),
            ("cli.get_robot_spec", hasattr(cli, "get_robot_spec")),
            ("cli.RunFinalizationContext", hasattr(cli, "RunFinalizationContext")),
            ("RobotSpec.finalize_run", "finalize_run" in spec_fields),
        ]
        if not ok
    ]
    if missing:
        raise RuntimeError(
            "rsched.launch: RPent internals changed, cannot trace: " + ", ".join(missing)
        )


def install(trace: JsonlWriter, episode: str, flash: bool = False) -> None:
    """Install all hooks. Safe to call once per process."""
    check_compat()
    INTENT.configure(trace, episode)
    if flash:
        _hook_flash_load()
    import urllib.error
    import urllib.request

    from rpent.tools.base import Tool
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.rpc import http_rpc

    seq = itertools.count()

    # ---- raw primitive results ------------------------------------------
    orig_tool_call = Tool.__call__

    def traced_tool_call(self, *args, **kwargs):
        result = orig_tool_call(self, *args, **kwargs)
        try:
            _tls.raw_result = result.to_dict()
        except Exception:
            _tls.raw_result = None
        return result

    Tool.__call__ = traced_tool_call

    # ---- tool calls ------------------------------------------------------
    orig_execute = Toolkit.execute_tool

    def traced_execute(self, name, input_dict):
        call_id = f"{episode}-{next(seq):04d}"
        prev = getattr(_tls, "call", None)
        _tls.call = (call_id, name)
        _tls.raw_result = None
        if INTENT.remaining is not None:
            INTENT.remaining.tool_start(name, input_dict if isinstance(input_dict, dict) else None)
            INTENT.log_remaining(call_id, name)
        t_start = now()
        error = None
        result = None
        try:
            result = orig_execute(self, name, input_dict)
            return result
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t_end = now()
            if INTENT.remaining is not None:
                INTENT.remaining.tool_end(name)
            raw = getattr(_tls, "raw_result", None) or {}
            final = result.to_dict() if result is not None else {}
            if error is None:
                error = final.get("error")
            trace.write(
                {
                    "type": "tool",
                    "episode": episode,
                    "call_id": call_id,
                    "tool": name,
                    "args": _jsonable(input_dict),
                    "t_start": t_start,
                    "t_end": t_end,
                    "dur_s": t_end - t_start,
                    "ok": error is None,
                    "error": None if error is None else str(error)[:500],
                    "result": {k: raw[k] for k in _RESULT_KEYS if k in raw},
                    "thread": threading.get_ident(),
                }
            )
            _tls.call = prev

    Toolkit.execute_tool = traced_execute

    # ---- RPC client: headers + client-side latency ------------------------
    def traced_call(self, method, args=(), kwargs=None, *, timeout_s=None):
        payload = {
            "method": method,
            "args": list(args),
            "kwargs": kwargs or {},
            "session_id": self._session_id,
        }
        body = json.dumps(payload, cls=http_rpc._NumpyEncoder).encode("utf-8")
        url = f"{self._base_url}/call"
        timeout = timeout_s if timeout_s is not None else http_rpc.DEFAULT_TIMEOUT_S
        call = getattr(_tls, "call", None)
        headers = {"Content-Type": "application/json", HDR_EPISODE: episode}
        if call is not None:
            headers[HDR_CALL], headers[HDR_TOOL] = call
        priority = INTENT.priority() if method in intent_mod.MODEL_METHODS else None
        if priority is not None:
            headers[HDR_PRIORITY] = repr(float(priority))

        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        t0 = now()
        transport_error = None
        try:
            opener = self._opener.open if self._opener else urllib.request.urlopen
            with opener(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
        except OSError as exc:
            transport_error = str(exc)
            raise http_rpc.RpcError(method, f"HTTP request failed: {exc}") from exc
        finally:
            t1 = now()
            trace.write(
                {
                    "type": "client_rpc",
                    "episode": episode,
                    "endpoint": self._base_url,
                    "method": method,
                    "call_id": call[0] if call else None,
                    "tool": call[1] if call else None,
                    "t_start": t0,
                    "t_end": t1,
                    "dur_s": t1 - t0,
                    "req_bytes": len(body),
                    "priority": priority,
                    "error": transport_error,
                }
            )
            if transport_error is None and INTENT.remaining is not None:
                INTENT.remaining.rpc_done(method)
        try:
            response = http_rpc._from_json(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise http_rpc.RpcError(method, f"invalid JSON response: {exc}") from exc
        return http_rpc.check_response(response, method)

    http_rpc.HttpRpcClient.call = traced_call


def _noop_finalize(_context: Any) -> None:
    return None


def _with_outcome_finalizer(spec: Any) -> Any:
    """Return ``spec`` with a no-op ``finalize_run`` when the robot left it unset.

    ``rpent.cli.main`` reads ``toolkit.solved()`` and builds
    ``RunFinalizationContext`` only inside ``if robot_spec.finalize_run is not None``
    (``main.py`` around the session cleanup and again after the transcript).
    LIBERO's spec omits the hook, so without this the outcome record is never
    written and ``environment_success`` stays ``None``.
    """
    if getattr(spec, "finalize_run", None) is not None:
        return spec
    return dataclasses.replace(spec, finalize_run=_noop_finalize)


def _patch_finalization(trace: JsonlWriter, episode: str) -> None:
    import rpent.cli.main as cli

    orig_ctx = cli.RunFinalizationContext
    orig_get_robot_spec = cli.get_robot_spec

    def ctx_factory(**kwargs):
        trace.write(
            {
                "type": "outcome",
                "episode": episode,
                "t": now(),
                "environment_success": kwargs.get("environment_success"),
                "agent_error": kwargs.get("agent_error"),
                "elapsed_s": kwargs.get("elapsed_s"),
                "planner": kwargs.get("planner"),
                "model": kwargs.get("model"),
                "stats": _jsonable(dict(kwargs.get("stats") or {})),
                "task": _jsonable(dict(kwargs.get("task_desc") or {})),
            }
        )
        return orig_ctx(**kwargs)

    def get_robot_spec(name: str):
        return _with_outcome_finalizer(orig_get_robot_spec(name))

    cli.RunFinalizationContext = ctx_factory
    cli.get_robot_spec = get_robot_spec


def _planner(argv: list[str]) -> str | None:
    try:
        return argv[argv.index("--planner") + 1]
    except (ValueError, IndexError):
        return None


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    episode = os.environ.get("RS_EPISODE_ID") or f"ep-{uuid.uuid4().hex[:8]}"
    trace_dir = Path(os.environ.get("RS_TRACE_DIR", "results/traces"))
    trace = JsonlWriter(trace_dir / f"episode_{episode}.jsonl")
    t0 = now()
    trace.write({"type": "episode_start", "episode": episode, "t": t0, "pid": os.getpid(), "argv": argv})

    install(trace, episode, flash=_planner(argv) == "flash")
    _patch_finalization(trace, episode)

    import rpent.cli.main as cli

    sys.argv = ["rpent", *argv]
    code: int | None = None
    try:
        code = cli.main()
        return int(code or 0)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        raise
    finally:
        t1 = now()
        INTENT.finish()
        trace.write(
            {"type": "episode_end", "episode": episode, "t": t1, "elapsed_s": t1 - t0, "exit_code": code}
        )
        trace.close()


if __name__ == "__main__":
    sys.exit(main())
