"""RS_ARTIFACTS=lazy: keep per-step observation artifacts in memory, write on demand.

RPent's Libero toolkit saves the environment state after every tool call
(``robots/libero/tools.py:dump_state``): PNG images and compressed ``.npz``
depth / world maps for two cameras. py-spy (E1) puts 21-27% of the episode
main thread in that compression and encoding. Tools read a few of them back
(``locate`` and ``segment`` load world maps, depth and images), so they cannot
simply be dropped.

With ``lazy`` the images and ``.npz`` arrays are kept in memory instead of
being encoded. Any access through the state's path resolution
(``load``, ``load_bytes``, ``artifact_path``, ``exists``; all go through
``EnvState._artifact_file``) writes the pending artifact first with RPent's own
``save``, so the bytes a reader sees are identical. ``prune_artifacts`` drops
pending entries without writing them. What is never read is never written,
so a lazy episode leaves no per-step images or depth maps on disk; step
records and JSON files are unchanged.

    RS_ARTIFACTS=store   (default) RPent unchanged
    RS_ARTIFACTS=lazy    as above

Installed by ``rsched.launch`` before RPent runs.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

LAZY_SUFFIXES = {".png", ".jpg", ".jpeg", ".npz"}
STATS = {"deferred": 0, "written": 0, "dropped": 0}


def mode() -> str:
    return (os.environ.get("RS_ARTIFACTS") or "store").strip().lower()


def install() -> bool:
    if mode() != "lazy":
        return False
    import numpy as np
    from rpent.session.base import EnvState

    orig_save = EnvState.save
    orig_file = EnvState._artifact_file
    orig_prune = EnvState.prune_artifacts
    local = threading.local()

    def pending(self) -> dict:
        p = self.__dict__.get("_rs_pending")
        if p is None:
            p = self.__dict__["_rs_pending"] = {}
        return p

    def flush(self, path: Path) -> None:
        item = pending(self).pop(str(path), None)
        if item is None:
            return
        name, value, step = item
        local.writing = True
        try:
            orig_save(self, name, value, step=step)
        finally:
            local.writing = False
        STATS["written"] += 1

    def artifact_file(self, name, step):
        path = orig_file(self, name, step)
        if not getattr(local, "writing", False) and str(path) in pending(self):
            # lock is not held here by save(); load paths take no lock
            flush(self, path)
        return path

    def save(self, name, value, *, step=-1, **options):
        if getattr(local, "writing", False) or Path(name).suffix.lower() not in LAZY_SUFFIXES or options:
            return orig_save(self, name, value, step=step, **options)
        with self._artifact_lock:
            step = self._resolve_read_step(step)
            if step is None:
                return orig_save(self, name, value, step=step)
            local.writing = True  # resolve the path without triggering a flush
            try:
                path = orig_file(self, name, step)
            finally:
                local.writing = False
            pending(self)[str(path)] = (name, np.array(value, copy=True), step)
            self._record_for(step).artifacts.add(name)
            if not self._step_open:
                self._write_manifest()
        STATS["deferred"] += 1
        return name

    def prune(self, names, *, step, keep_last):
        old = step - keep_last
        if old >= 0:
            for name in names:
                local.writing = True
                try:
                    path = orig_file(self, name, old)
                finally:
                    local.writing = False
                if pending(self).pop(str(path), None) is not None:
                    STATS["dropped"] += 1
        return orig_prune(self, names, step=step, keep_last=keep_last)

    EnvState.save = save
    EnvState._artifact_file = artifact_file
    EnvState.prune_artifacts = prune
    return True
