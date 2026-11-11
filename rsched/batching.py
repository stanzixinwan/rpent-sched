"""Cross-episode batching of ``vla.predict`` calls (used by ``rsched.proxy --policy batch``).

Every RPent agent sends Pi0.5 one observation at a time (openpi wire format
with batch dimension 1), and the server runs them one by one. The server itself
takes any batch size, and a batch of 8 costs about 2x a batch of 1 on an A100
(``rsched.bench_batch``). The batcher sits in the proxy: requests that wait for
the backend are merged into one call, the actions are split back by row.

When to dispatch (``mode``):

    greedy    as soon as the backend is free, send everything waiting (up to
              ``max_batch``). Batches form only from requests that piled up
              during the previous call.
    window    hold the oldest waiting request up to ``window_s`` (or until
              ``max_batch`` are waiting), then send.
    predict   hold it only while some episode that is not waiting is expected to
              send its next request before ``oldest + window_s``. An episode is
              expected when its previous request ended less than ~2 of its own
              request gaps ago (it is inside a burst of chunks) and its gap
              puts the next arrival in time. With ``use_intent`` the episode's
              ``X-RS-Burst-Left`` header (chunks the current ``pi0_*`` call may
              still use, from its ``max_chunks``) also rules out episodes whose
              burst is over.

Only requests with the same structure (keys, which views are present, image
sizes, options) are merged; others go in a later batch.
"""

from __future__ import annotations

import base64
import itertools
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from rsched.traceio import now

MODES = ("greedy", "window", "predict")


# ---------------------------------------------------------------------------
# wire format (RPent http_rpc: ndarrays as {"__ndarray__": b64, "dtype", "shape"})
# ---------------------------------------------------------------------------


def decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj and set(obj) <= {"__ndarray__", "dtype", "shape"}:
            arr = np.frombuffer(base64.b64decode(obj["__ndarray__"]), dtype=obj.get("dtype"))
            return arr.reshape(obj.get("shape", (-1,)))
        if "__npscalar__" in obj and set(obj) <= {"__npscalar__", "dtype"}:
            return np.dtype(obj["dtype"]).type(obj["__npscalar__"])
        return {k: decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode(v) for v in obj]
    return obj


class _Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            arr = np.ascontiguousarray(obj)
            return {"__ndarray__": base64.b64encode(arr.tobytes()).decode("ascii"),
                    "dtype": str(arr.dtype), "shape": list(arr.shape)}
        if isinstance(obj, np.generic):
            return {"__npscalar__": obj.item(), "dtype": str(obj.dtype)}
        return super().default(obj)


def encode(obj: Any) -> bytes:
    return json.dumps(obj, cls=_Encoder).encode("utf-8")


# ---------------------------------------------------------------------------
# merge / split
# ---------------------------------------------------------------------------


def signature(obs: Any, options: Any) -> tuple | None:
    """Requests with equal signatures can share a batch; None = not batchable."""
    if not isinstance(obs, dict):
        return None
    parts = []
    for k in sorted(obs):
        v = obs[k]
        if v is None:
            parts.append((k, None))
        elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == 1:
            parts.append((k, "arr", str(v.dtype), tuple(v.shape[1:])))
        elif isinstance(v, list) and len(v) == 1:
            parts.append((k, "list"))
        else:
            return None
    return tuple(parts), json.dumps(options, sort_keys=True, default=str)


def merge(obs_list: list[dict]) -> dict:
    out = {}
    for k, v in obs_list[0].items():
        if v is None:
            out[k] = None
        elif isinstance(v, np.ndarray):
            out[k] = np.concatenate([o[k] for o in obs_list], axis=0)
        else:
            out[k] = list(itertools.chain.from_iterable(o[k] for o in obs_list))
    return out


# ---------------------------------------------------------------------------
# batcher
# ---------------------------------------------------------------------------


@dataclass
class Item:
    seq: int
    t_arrive: float
    episode: str | None
    burst_left: int | None
    obs: dict
    options: Any
    session_id: Any
    sig: tuple
    event: threading.Event = field(default_factory=threading.Event)
    response: bytes = b""
    t_dispatch: float | None = None
    t_done: float | None = None
    batch_id: int | None = None
    batch_size: int = 0


@dataclass
class _Episode:
    last_done: float
    gap: float
    burst_left: int | None  # chunks left after the last finished request; None = unknown


class Batcher:
    """Merges concurrent ``vla.predict`` calls; one batch at the backend at a time."""

    def __init__(self, forward: Callable[[bytes], bytes], max_batch: int = 16, mode: str = "greedy",
                 window_s: float = 0.0, use_intent: bool = False, default_gap_s: float = 0.2,
                 method: str = "vla.predict") -> None:
        if mode not in MODES:
            raise ValueError(f"unknown batch mode {mode!r}; one of {MODES}")
        self.forward = forward
        self.max_batch = max(1, int(max_batch))
        self.mode, self.window_s, self.use_intent = mode, float(window_s), bool(use_intent)
        self.default_gap_s = default_gap_s
        self.method = method
        self._cond = threading.Condition()
        self._queue: list[Item] = []
        self._episodes: dict[str, _Episode] = {}
        self._seq = itertools.count()
        self._batch_ids = itertools.count()
        self.batches = 0
        self.requests = 0
        self.waited_for_prediction = 0
        threading.Thread(target=self._run, daemon=True, name="batcher").start()

    # -- caller side -------------------------------------------------------
    def parse(self, body: bytes) -> tuple[dict, Any, Any] | None:
        """(obs, options, session_id) if this request can be batched, else None."""
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            return None
        if req.get("method") != self.method or req.get("kwargs"):
            return None
        args = req.get("args") or []
        if not 1 <= len(args) <= 2:
            return None
        obs = decode(args[0])
        options = decode(args[1]) if len(args) == 2 else None
        if signature(obs, options) is None:
            return None
        return obs, options, req.get("session_id")

    def submit(self, obs: dict, options: Any, session_id: Any, episode: str | None,
               burst_left: int | None) -> Item:
        item = Item(next(self._seq), now(), episode, burst_left, obs, options, session_id,
                    signature(obs, options))
        with self._cond:
            self._queue.append(item)
            self._cond.notify_all()
        item.event.wait()
        return item

    # -- dispatcher ----------------------------------------------------------
    def _expected_soon(self, t: float, deadline: float) -> bool:
        waiting = {it.episode for it in self._queue}
        for ep, st in self._episodes.items():
            if ep in waiting:
                continue
            if self.use_intent and st.burst_left is not None and st.burst_left <= 0:
                continue
            arrival = st.last_done + st.gap
            if t - st.last_done < 2 * st.gap + 0.05 and arrival <= deadline:
                return True
        return False

    def _pick(self) -> list[Item]:
        """Called with the lock held and a non-empty queue; waits per mode, then takes a batch."""
        oldest = self._queue[0]
        deadline = oldest.t_arrive + (self.window_s if self.mode != "greedy" else 0.0)
        counted = False
        while len(self._queue) < self.max_batch:
            t = now()
            if t >= deadline:
                break
            if self.mode == "predict" and not self._expected_soon(t, deadline):
                break
            if self.mode == "predict" and not counted:
                self.waited_for_prediction += 1
                counted = True
            self._cond.wait(timeout=min(deadline - t, 0.002))
        sig = self._queue[0].sig
        batch = [it for it in self._queue if it.sig == sig][: self.max_batch]
        ids = {id(it) for it in batch}
        self._queue = [it for it in self._queue if id(it) not in ids]
        return batch

    def _run(self) -> None:
        while True:
            with self._cond:
                while not self._queue:
                    self._cond.wait()
                batch = self._pick()
            self._dispatch(batch)

    def _dispatch(self, batch: list[Item]) -> None:
        bid = next(self._batch_ids)
        t0 = now()
        sessions = {it.session_id for it in batch}
        payload = {"method": self.method,
                   "args": [merge([it.obs for it in batch])] + ([batch[0].options] if batch[0].options is not None
                                                                 else []),
                   "kwargs": {}, "session_id": sessions.pop() if len(sessions) == 1 else None}
        try:
            raw = self.forward(encode(payload))
            resp = json.loads(raw)
            if resp.get("ok"):
                actions = np.asarray(decode(resp["result"]))
                if actions.shape[0] != len(batch):
                    raise ValueError(f"backend returned {actions.shape[0]} rows for a batch of {len(batch)}")
                bodies = [encode({"ok": True, "result": actions[i:i + 1]}) for i in range(len(batch))]
            else:
                bodies = [raw] * len(batch)
        except Exception as exc:  # backend down, timeout, or a malformed answer
            err = encode({"ok": False, "error": f"rsched batcher: {type(exc).__name__}: {exc}"[:500]})
            bodies = [err] * len(batch)
        t1 = now()
        with self._cond:
            for it, body in zip(batch, bodies):
                it.response, it.t_dispatch, it.t_done = body, t0, t1
                it.batch_id, it.batch_size = bid, len(batch)
                if it.episode is not None:
                    st = self._episodes.get(it.episode)
                    gap = self.default_gap_s
                    if st is not None and 0 < it.t_arrive - st.last_done < 2.0:
                        gap = 0.7 * st.gap + 0.3 * (it.t_arrive - st.last_done)
                    elif st is not None:
                        gap = st.gap
                    left = None if it.burst_left is None else it.burst_left - 1
                    self._episodes[it.episode] = _Episode(t1, gap, left)
            self.batches += 1
            self.requests += len(batch)
            # forget episodes idle for a while (finished, or between tools)
            self._episodes = {e: s for e, s in self._episodes.items() if t1 - s.last_done < 30.0}
        for it in batch:
            it.event.set()

    def stats(self) -> dict[str, Any]:
        with self._cond:
            return {"mode": self.mode, "max_batch": self.max_batch, "window_s": self.window_s,
                    "use_intent": self.use_intent, "batches": self.batches, "requests": self.requests,
                    "mean_batch": self.requests / self.batches if self.batches else None,
                    "waited_for_prediction": self.waited_for_prediction, "queue_len": len(self._queue)}
