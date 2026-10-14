"""CPU stand-ins for RPent's model and simulator services.

They are built on RPent's own ``RpcFacade``/``BaseVLAFacade``, so locking and the
wire protocol match the real servers: ``vla.predict`` runs under the exclusive
write lock (serialized), ``sam3.segment`` is read-only (concurrent).

    python tests/fakes/fake_services.py --kind vla  --port 9000 --ms 40
    python tests/fakes/fake_services.py --kind sam3 --port 9001 --ms 25
    python tests/fakes/fake_services.py --kind env  --port 9002 --ms 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from rpent.robots.components.vla_facade_base import BaseVLAFacade
from rpent.utils.rpc import RpcFacade


class FakeVLA(BaseVLAFacade):
    def __init__(self, ms: float) -> None:
        self.ms = ms
        super().__init__()

    def predict(self, obs, options=None):
        time.sleep(self.ms / 1e3)
        batch = len(obs.get("task_descriptions", [""])) if isinstance(obs, dict) else 1
        return np.zeros((batch, 5, 7), dtype=np.float32)


class FakeSAM3(RpcFacade):
    def __init__(self, ms: float) -> None:
        super().__init__()
        self.ms = ms
        self._rpc["sam3.segment"] = self.segment
        self._readonly_methods.add("sam3.segment")

    def segment(self, image=None, text_prompt=None, **_):
        time.sleep(self.ms / 1e3)
        return {"score": 0.9, "box": [0, 0, 10, 10], "prompt": text_prompt}


class FakeEnv(RpcFacade):
    def __init__(self, ms: float) -> None:
        super().__init__()
        self.ms = ms
        self._rpc["env.chunk_step"] = self.chunk_step

    def chunk_step(self, actions=None):
        time.sleep(self.ms / 1e3)
        return {"terminated": False}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["vla", "sam3", "env"], required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ms", type=float, default=30.0)
    args = p.parse_args()
    facade = {"vla": FakeVLA, "sam3": FakeSAM3, "env": FakeEnv}[args.kind](args.ms)
    facade.serve(transport="http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
