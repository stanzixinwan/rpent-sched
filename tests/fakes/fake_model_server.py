"""A tiny torch model behind RPent's real RpcFacade, answering vla.predict (tests only).

    python tests/fakes/fake_model_server.py --port 9100

Actions are a linear map of the state plus image means, one row per batch row,
so batched calls work like Pi0.5's predict_action_batch.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from rpent.utils.rpc.rpc_facade import RpcFacade


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.GELU(), torch.nn.Linear(64, 32))
        self.head = torch.nn.Sequential(torch.nn.Linear(32 + 8, 64), torch.nn.GELU(), torch.nn.Linear(64, 35))

    def forward(self, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.vision(img), state], dim=-1))


class FakeVLA(RpcFacade):
    def __init__(self) -> None:
        super().__init__()
        self._model = Tiny().eval()
        self._rpc["vla.predict"] = self.predict

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        img = torch.from_numpy(np.asarray(obs["main_images"], dtype=np.float32)).mean(dim=(1, 2))
        state = torch.from_numpy(np.asarray(obs["states"], dtype=np.float32))
        with torch.no_grad():
            out = self._model(img, state)
        return out.reshape(-1, 5, 7).numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    FakeVLA().serve(transport="http", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
