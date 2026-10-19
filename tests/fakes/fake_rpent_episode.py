"""A fake RPent episode that exercises rsched.launch's real hooks on CPU.

It replaces only ``rpent.cli.main.main`` (the LLM planner and the simulator)
and keeps RPent's real ``Toolkit.execute_tool``, ``Tool``, ``HttpRpcClient``,
and ``RunFinalizationContext``, so the same patches run as in a real episode.
Used as ``rpent.command`` in tests/configs/smoke_cpu.yaml.

Environment:
    RS_FAKE_ENV_ENDPOINT   http endpoint of the fake simulator service
    RS_FAKE_THINK_S        planner think time between tool calls (default 0.15)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import rpent.cli.main as cli  # noqa: E402
from rpent.robots.components.pi05_vla_client import Pi05VLAClient  # noqa: E402
from rpent.tools.base import ToolResult, iter_tools, tool  # noqa: E402
from rpent.tools.toolkit import Toolkit, _Scheduler  # noqa: E402
from rpent.utils.rpc.http_rpc import HttpRpcClient  # noqa: E402

from rsched import launch  # noqa: E402


class FakeLiberoTools:
    def __init__(self, vla: Pi05VLAClient, sam3: HttpRpcClient, env: HttpRpcClient, rng: random.Random):
        self.vla, self.sam3, self.env, self.rng = vla, sam3, env, rng
        self.obs = {
            "main_images": np.zeros((32, 32, 3), np.uint8),
            "wrist_images": np.zeros((32, 32, 3), np.uint8),
            "states": np.zeros(8, np.float32),
            "task_descriptions": "",
        }

    @tool
    def pi0_pick(self, prompt: str, *, max_chunks: int = 24) -> ToolResult:
        """Closed-loop pick; the number of chunks depends on the target.

        Args:
            prompt: instruction
            max_chunks: chunk budget
        """
        need = 3 + (sum(map(ord, prompt)) % 9) + self.rng.randint(0, 2)
        used = 0
        for _ in range(min(need, max_chunks)):
            self.obs["task_descriptions"] = prompt
            actions = self.vla.predict(self.obs)
            self.env.call("env.chunk_step", args=(actions,))
            used += 1
        return ToolResult(data={"name": "pick", "success": need <= max_chunks,
                                "chunks_used": used, "max_chunks": max_chunks})

    @tool
    def segment(self, text_prompt: str) -> ToolResult:
        """Segment an object.

        Args:
            text_prompt: object name
        """
        out = self.sam3.call("sam3.segment", kwargs={"text_prompt": text_prompt})
        return ToolResult(data={"name": "segment", "score": out["score"]})

    @tool
    def move_to(self, x: float, y: float, z: float) -> ToolResult:
        """Scripted move.

        Args:
            x: x
            y: y
            z: z
        """
        time.sleep(0.02)
        self.env.call("env.chunk_step", args=(np.zeros((5, 7), np.float32),))
        return ToolResult(data={"name": "move_to", "success": True})


class FakeToolkit(Toolkit):
    """Real Toolkit dispatch path without memory/dashboard plumbing."""

    def __init__(self, owner: FakeLiberoTools) -> None:  # noqa: D107 - intentionally skip super
        self._tools = {}
        self._state = None
        self._scheduler = _Scheduler()
        self.add_tools(iter_tools(owner))

    def get_env_state(self, *, command, result, elapsed_s):
        # Like the real LIBERO toolkit: replace the primitive's result with an observation.
        return ToolResult(data={"observation": "state", "command": command["action"]})

    def latest_record(self):
        return None


def fake_main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite")
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vla-endpoint", required=True)
    ap.add_argument("--sam3-endpoint", required=True)
    ap.add_argument("--output-dir")
    args, _ = ap.parse_known_args(sys.argv[1:])

    if os.environ.get("RS_FAKE_CRASH"):
        time.sleep(0.2)
        raise RuntimeError("[env] wait / client connect failed: fake env_server crash")
    rng = random.Random(f"{args.suite}-{args.task}-{args.seed}")
    think = float(os.environ.get("RS_FAKE_THINK_S", "0.15"))
    owner = FakeLiberoTools(
        Pi05VLAClient(HttpRpcClient(args.vla_endpoint), embodiment="libero"),
        HttpRpcClient(args.sam3_endpoint),
        HttpRpcClient(os.environ["RS_FAKE_ENV_ENDPOINT"]),
        rng,
    )
    toolkit = FakeToolkit(owner)
    toolkit._state = type("S", (), {"latest_record": lambda self: None})()

    t0 = time.time()
    time.sleep(0.1)  # "startup"
    plan = [
        ("segment", {"text_prompt": "bowl"}),
        ("pi0_pick", {"prompt": f"pick up the {args.suite} object {args.task}", "max_chunks": 12}),
        ("move_to", {"x": 0.1, "y": 0.2, "z": 0.3}),
        ("segment", {"text_prompt": "plate"}),
        ("pi0_pick", {"prompt": f"place it {args.task}", "max_chunks": 8}),
    ]
    for name, params in plan:
        time.sleep(think)  # planner (LLM) time
        toolkit.execute_tool(name, params)
    time.sleep(think)

    cli.RunFinalizationContext(
        output_dir=Path(args.output_dir or "."), robot_name="libero",
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
        environment_success=rng.random() < 0.7, agent_error=os.environ.get("RS_FAKE_AGENT_ERROR"),
        elapsed_s=time.time() - t0, planner="fake", model="fake",
        reasoning_effort="none", max_turns=10, planner_timeout_s=None,
        finish_result=None, stats={"total_input_tokens": 1000, "total_output_tokens": 100},
    )
    if os.environ.get("RS_FAKE_PRINT"):
        print(os.environ["RS_FAKE_PRINT"], flush=True)
    return int(os.environ.get("RS_FAKE_EXIT", "0"))


cli.main = fake_main

if __name__ == "__main__":
    sys.exit(launch.main())
