"""Which episode of a batch enters a free slot: launch-order rules.

A Flash plan states each episode's work before it starts. ``plan_estimate``
prices it with ``rsched.intent``'s cost table; the rules below order a batch by
it. The same functions drive the real sweep (``rsched.sweep``, ``launch:`` in a
config) and the trace-driven simulator (``analysis/g3_sim.py``), so a GPU run
tests exactly the rule that was simulated.

    given       the batch as listed (shuffled per repeat in the experiments)
    lpt         longest planned work first
    spt         shortest planned work first
    front       the n longest first (longest leading), the rest as listed
    front_back  like front, and the n shortest last (shortest at the very end)
    jit_lpt     decided at each free slot: start the longest pending episode
                only once it would otherwise finish after the projected batch
                end; otherwise the next as listed (see ``jit_pick``)

What the rules order by (``launch_estimates``):

    plan        the Flash plan's estimate (known before any episode has run)
    history     a table "task -> measured wall time" from an earlier run;
                tasks missing from it (or marked cold) get the mean of the known
    hybrid      history where it exists, the plan (in history's units) elsewhere
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from rsched.intent import Remaining, plan_work

STATIC_RULES = ("given", "lpt", "spt", "front", "front_back")
RULES = STATIC_RULES + ("jit_lpt",)
ESTIMATES = ("plan", "history", "hybrid")


def plan_name(suite: str, task: int) -> str:
    """``libero_10_swap``, 3 -> ``10_swap_t3``, the Flash plan file stem."""
    return f"{suite.removeprefix('libero_')}_t{task}"


def load_program(plans_dir: Path, name: str) -> dict:
    """The program as ``robots.libero.flash.replay.load`` builds it (without numpy)."""
    plan = json.loads((plans_dir / f"{name}_plan.json").read_text())["plan"]
    anchors = json.loads((plans_dir / f"{name}_anchors.json").read_text())["anchors"]
    return {"plan": plan, "reference": {a["phrase"]: a.get("median_xy") for a in anchors},
            "locator_of": {a["phrase"]: a["locator"] for a in anchors}}


def plan_estimate(plans_dir: Path, suite: str, task: int, costs: dict | None = None) -> float:
    """Planned work of one episode in seconds, before it starts."""
    program = load_program(plans_dir, plan_name(suite, task))
    return float(Remaining(plan_work(program), costs).snapshot()["intent_s"])


def static_order(est: list[float], rule: str, n: int) -> list[int]:
    """Positions of a batch in launch order for a static rule."""
    if rule not in STATIC_RULES:
        raise ValueError(f"unknown static rule {rule!r}; choose from {STATIC_RULES}")
    idx = list(range(len(est)))
    if rule == "given":
        return idx
    by_len = sorted(idx, key=lambda i: est[i], reverse=True)
    if rule == "lpt":
        return by_len
    if rule == "spt":
        return by_len[::-1]
    k = min(n, len(est))
    front = by_len[:k]
    taken = set(front)
    back = [i for i in reversed(by_len) if i not in taken][:k] if rule == "front_back" else []
    taken |= set(back)
    return front + [i for i in idx if i not in taken] + back[::-1]


def jit_pick(now: float, pending_est: list[float], running: list[tuple[float, float]],
             finished: list[tuple[float, float]], n: int, slack: float = 1.0,
             prior_scale: float = 1.5) -> int:
    """Just-in-time LPT: position in ``pending_est`` to launch now.

    ``running`` holds (estimate, start time) of episodes in flight; ``finished``
    holds (estimate, wall seconds). Planned work is turned into wall time with
    the ratio seen on finished episodes (``prior_scale`` before any finish). The
    batch end is projected as now + (pending work + what running episodes still
    need) / n. If the longest pending episode, started now, would end later than
    that minus ``slack`` typical episodes, launch it; otherwise launch position 0.
    """
    if not pending_est:
        raise ValueError("nothing pending")
    planned = sum(e for e, _ in finished)
    scale = sum(w for _, w in finished) / planned if finished and planned > 0 else prior_scale
    typical = scale * statistics.median(pending_est)
    need = sum(e * scale for e in pending_est)
    need += sum(max(e * scale - (now - t0), 0.0) for e, t0 in running)
    end = now + need / n
    longest = max(range(len(pending_est)), key=lambda k: pending_est[k])
    if now + pending_est[longest] * scale >= end - slack * typical:
        return longest
    return 0


def task_key(suite: str, task: int) -> str:
    """``libero_10_swap``, 3 -> ``libero_10_swap_t3`` (history tables, simulator libraries)."""
    return f"{suite}_t{task}"


def load_history(path: Path) -> dict[str, float]:
    """``{"libero_10_swap_t3": 123.4, ...}``: mean measured seconds per task."""
    table = json.loads(Path(path).read_text())
    return {str(k): float(v) for k, v in (table.get("tasks", table)).items()}


def cold_tasks(keys: list[str], fraction: float, seed: int) -> set[str]:
    """A reproducible ``fraction`` of the distinct task keys, treated as never run."""
    import random

    distinct = sorted(set(keys))
    k = round(fraction * len(distinct))
    return set(random.Random(seed).sample(distinct, k)) if k else set()


def launch_estimates(plan: dict, history: dict[str, float] | None, mode: str,
                     cold: set[str] | frozenset = frozenset(), weights: dict | None = None) -> dict:
    """Per-task launch estimates for one mode, keyed like ``plan`` ((suite, task) tuples).

    History and hybrid are rescaled so their weighted sum equals the plan's
    (rank-preserving), so ``jit_pick``'s prior wall/plan ratio fits every mode
    the same way; ``weights`` counts how often each task occurs in the batch.
    This is the estimate the trace-driven simulator compared (history_baseline.py).
    """
    if mode not in ESTIMATES:
        raise ValueError(f"launch.estimate {mode!r}; choose from {ESTIMATES}")
    if mode == "plan":
        return dict(plan)
    if history is None:
        raise ValueError(f"launch.estimate={mode} needs launch.history_file")
    keys = {t: task_key(*t) for t in plan}
    known = [t for t in plan if keys[t] in history and keys[t] not in cold]
    w = weights or {t: 1 for t in plan}
    if known:
        fill = sum(history[keys[t]] for t in known) / len(known)
        ratio = sum(history[keys[t]] for t in known) / max(sum(plan[t] for t in known), 1e-9)
    else:
        fill, ratio = 1.0, 1.0
    if mode == "history":
        est = {t: (history[keys[t]] if t in known else fill) for t in plan}
    else:
        est = {t: (history[keys[t]] if t in known else plan[t] * ratio) for t in plan}
    total = sum(est[t] * w.get(t, 1) for t in plan)
    target = sum(plan[t] * w.get(t, 1) for t in plan)
    return {t: est[t] * target / total for t in plan} if total > 0 else dict(plan)

