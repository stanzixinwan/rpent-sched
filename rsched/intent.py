"""Planner intent -> remaining model work, the signal G3 schedules on.

A Flash plan fixes, before the episode starts, every primitive the episode will
run, each ``pi0_pick``'s chunk budget (``max_chunks``, one ``vla.predict`` per
chunk), and every anchor Molmo will ground. ``PlanWork`` reads that off the
program that ``robots.libero.flash.replay.load`` returns; ``Remaining`` counts
it down as the episode runs and turns it into seconds with a per-primitive cost
table. ``HistoryEstimate`` is the history-only counterpart: the mean total work
of episodes that already finished in this run, minus what this one has done.

Both estimates are upper-bound flavoured: a pick that succeeds early leaves
part of its budget unused, an episode that solves the task early skips the rest
of its plan, and a failed pick is retried (up to ``PICK_ATTEMPTS`` in replay).

How Flash replay spends model calls (``robots/libero/flash/replay.py``):

- survey: one ``molmo.ground`` per Molmo-located anchor; one ``segment`` tool
  call per segment-located anchor;
- refine: per Molmo anchor, one ``move_to`` then one ``molmo.ground``;
- plan: every step except ``segment``/``segment_point`` (skipped) runs as a tool;
  each ``set_gripper`` is followed by one ``molmo.ground`` (held-object offset).

Scheduling modes (``RS_SCHED``), sent as the ``X-RS-Priority`` header (the
proxy's ``priority`` policy serves the smallest value first):

    none            no header: the proxy serves in arrival order (FIFO)
    intent_srpt     least remaining intent work first
    intent_lrpt     most remaining intent work first
    history_srpt    least remaining history estimate first
    history_lrpt    most remaining history estimate first
"""

from __future__ import annotations

import json
import os
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = ("none", "intent_srpt", "intent_lrpt", "history_srpt", "history_lrpt")
PI0_TOOLS = {"pi0_pick": 24, "pi0_doubled": 20}  # tool -> default max_chunks
SKIPPED_IN_REPLAY = {"segment", "segment_point"}
MODEL_METHODS = ("vla.predict", "molmo.ground", "sam3.segment")

# Seconds per unit, from G2 at N=1 (no queueing; A100, 2026-10-24): per-chunk
# pi0_pick time (VLA + chunk_step), molmo.ground client latency, and mean tool
# wall time for scripted primitives. Configs may override them.
DEFAULT_COSTS: dict[str, float] = {
    "chunk_s": 0.64,
    "molmo_s": 1.1,
    "move_to": 4.4,
    "move_pose": 11.1,
    "release": 2.6,
    "set_gripper": 1.0,  # tool time without the held-object grounding (counted in molmo_s)
    "segment": 0.7,
    "rotate_wrist": 3.0,
    "rotate_pitch": 3.0,
    "step_default": 3.0,
}


@dataclass
class PlanWork:
    """Model and tool work one Flash program will issue, at most."""

    pi0: list[tuple[str, str, int]] = field(default_factory=list)  # (tool, prompt, max_chunks)
    molmo_calls: int = 0
    steps: Counter = field(default_factory=Counter)  # non-pi0 tool name -> planned calls

    @property
    def chunks(self) -> int:
        return sum(b for _, _, b in self.pi0)

    def to_dict(self) -> dict[str, Any]:
        return {"pi0": [list(p) for p in self.pi0], "chunks": self.chunks,
                "molmo_calls": self.molmo_calls, "steps": dict(self.steps)}


def plan_work(program: dict) -> PlanWork:
    """Count a program's work as ``robots.libero.flash.replay.replay`` would run it."""
    plan = program["plan"]
    locator_of = program.get("locator_of") or {}
    phrases = list(program.get("reference") or locator_of)
    segment_anchors = [p for p in phrases if locator_of.get(p) == "segment"]
    molmo_anchors = [p for p in phrases if locator_of.get(p) != "segment"]

    work = PlanWork()
    work.steps["segment"] += len(segment_anchors)  # survey of segment-located anchors
    work.steps["move_to"] += len(molmo_anchors)  # refine: hover over each Molmo anchor
    work.molmo_calls = 2 * len(molmo_anchors)  # survey + refine
    for entry in plan:
        name = entry["action"]
        args = entry.get("arguments") or {}
        if name in SKIPPED_IN_REPLAY:
            continue
        if name in PI0_TOOLS:
            work.pi0.append((name, str(args.get("prompt", "")),
                             int(args.get("max_chunks", PI0_TOOLS[name]))))
            continue
        work.steps[name] += 1
        if name == "set_gripper":
            work.molmo_calls += 1  # held-object grounding
    return work


def step_cost(costs: dict[str, float], name: str) -> float:
    return float(costs.get(name, costs.get("step_default", 3.0)))


class Remaining:
    """Counts a program's work down as tools and model calls complete. Thread-safe."""

    def __init__(self, work: PlanWork, costs: dict[str, float] | None = None) -> None:
        self.costs = {**DEFAULT_COSTS, **(costs or {})}
        self.work = work
        self._lock = threading.Lock()
        self._pi0 = deque(work.pi0)
        self._cur_budget = 0  # chunk budget of the pi0 call in progress
        self._cur_used = 0
        self._last_pi0: tuple[str, str] | None = None
        self._molmo_left = work.molmo_calls
        self._steps_left = Counter(work.steps)
        # Work actually done, priced with the same table (the history estimate's unit).
        self.done = Counter()

    # -- events ---------------------------------------------------------------
    def tool_start(self, name: str, args: dict | None) -> None:
        args = args or {}
        with self._lock:
            if name in PI0_TOOLS:
                prompt = str(args.get("prompt", ""))
                budget = int(args.get("max_chunks", PI0_TOOLS[name]))
                if self._pi0 and self._pi0[0][:2] == (name, prompt):
                    self._pi0.popleft()  # the planned pick
                elif self._last_pi0 != (name, prompt) and self._pi0:
                    self._pi0.popleft()  # prompt rewritten: still the next planned pick
                # else: a retry of the previous pick, on top of the plan
                self._last_pi0 = (name, prompt)
                self._cur_budget, self._cur_used = budget, 0
            else:
                if self._steps_left[name] > 0:
                    self._steps_left[name] -= 1
                self.done["step_s"] += step_cost(self.costs, name)

    def tool_end(self, name: str) -> None:
        if name in PI0_TOOLS:
            with self._lock:
                self._cur_budget = self._cur_used = 0

    def rpc_done(self, method: str) -> None:
        with self._lock:
            if method == "vla.predict":
                self._cur_used += 1
                self.done["chunks"] += 1
            elif method == "molmo.ground":
                self._molmo_left = max(self._molmo_left - 1, 0)
                self.done["molmo"] += 1

    # -- estimates ------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            chunks_left = sum(b for _, _, b in self._pi0) + max(self._cur_budget - self._cur_used, 0)
            steps_s = sum(step_cost(self.costs, n) * k for n, k in self._steps_left.items())
            return {
                "chunks_left": chunks_left,
                "molmo_left": self._molmo_left,
                "steps_left": sum(self._steps_left.values()),
                "intent_s": (self.costs["chunk_s"] * chunks_left
                             + self.costs["molmo_s"] * self._molmo_left + steps_s),
                "done_s": self.done_s_locked(),
            }

    def done_s_locked(self) -> float:
        return (self.costs["chunk_s"] * self.done["chunks"]
                + self.costs["molmo_s"] * self.done["molmo"] + self.done["step_s"])

    def intent_s(self) -> float:
        return float(self.snapshot()["intent_s"])

    def done_s(self) -> float:
        with self._lock:
            return self.done_s_locked()


class HistoryEstimate:
    """Mean total work of episodes already finished in this run, from a shared JSONL file.

    Every episode appends ``{"work_s": ...}`` when it ends (``record``). Re-read
    only when the file grows. Returns ``None`` before any episode has finished,
    which leaves the request without a priority (served in arrival order).
    """

    def __init__(self, path: str | os.PathLike | None) -> None:
        self.path = Path(path) if path else None
        self._size = -1
        self._mean: float | None = None
        self._lock = threading.Lock()

    def mean_total_s(self) -> float | None:
        if self.path is None:
            return None
        with self._lock:
            try:
                size = self.path.stat().st_size
            except OSError:
                return self._mean
            if size != self._size:
                values = []
                for line in self.path.read_text().splitlines():
                    try:
                        values.append(float(json.loads(line)["work_s"]))
                    except (ValueError, KeyError, TypeError):
                        continue  # a line still being written
                self._mean = sum(values) / len(values) if values else None
                self._size = size
            return self._mean

    def remaining_s(self, done_s: float) -> float | None:
        mean = self.mean_total_s()
        return None if mean is None else max(mean - done_s, 0.0)

    def record(self, episode: str, work_s: float) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"episode": episode, "work_s": work_s}) + "\n"
        with self.path.open("a") as fh:  # one short O_APPEND write per episode
            fh.write(line)


def priority(mode: str, remaining: Remaining | None, history: HistoryEstimate | None) -> float | None:
    """Header value for one model request; ``None`` means no header (FIFO)."""
    if mode == "none" or remaining is None:
        return None
    if mode.startswith("intent_"):
        value = remaining.intent_s()
    elif mode.startswith("history_"):
        value = history.remaining_s(remaining.done_s()) if history is not None else None
        if value is None:
            return None
    else:
        raise ValueError(f"unknown scheduling mode {mode!r}; choose from {MODES}")
    return -value if mode.endswith("_lrpt") else value
