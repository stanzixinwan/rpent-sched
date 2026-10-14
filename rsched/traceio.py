"""Thread-safe, append-only JSONL trace writer shared by all components."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


def now() -> float:
    """Wall-clock seconds; comparable across processes on one machine."""
    return time.time()


def _default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return f"<ndarray shape={list(obj.shape)} dtype={obj.dtype}>"
    except ImportError:
        pass
    return repr(obj)


class JsonlWriter:
    """Append one JSON object per line; flushes every record so crashes lose little."""

    def __init__(self, path: str | os.PathLike) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, default=_default, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping a possibly truncated final line."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
