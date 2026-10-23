"""Outcome recording when the robot does not define finalize_run.

LIBERO leaves ``RobotSpec.finalize_run`` unset. The CLI then skips both
``toolkit.solved()`` and ``RunFinalizationContext``. ``rsched.launch`` fills in
a no-op so that path still runs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import rpent.cli.main as cli

from rsched.launch import _patch_finalization, check_compat
from rsched.traceio import JsonlWriter, read_jsonl


@dataclasses.dataclass(frozen=True)
class _Spec:
    name: str = "libero"
    finalize_run: object = None


def _context(tmp_path: Path, *, success: bool) -> cli.RunFinalizationContext:
    return cli.RunFinalizationContext(
        output_dir=tmp_path,
        robot_name="libero",
        task_desc={"suite": "libero_object_swap", "task": 2, "seed": 0},
        environment_success=success,
        agent_error=None,
        elapsed_s=1.0,
        planner="claude_code",
        model="claude",
        reasoning_effort="none",
        max_turns=10,
        planner_timeout_s=None,
        finish_result=None,
        stats={
            "total_input_tokens": 1000,
            "total_output_tokens": 100,
            "total_cost_usd": 1.25,
        },
    )


def _patched(tmp_path: Path, spec_fn):
    orig_get = cli.get_robot_spec
    orig_ctx = cli.RunFinalizationContext
    cli.get_robot_spec = spec_fn
    trace = JsonlWriter(tmp_path / "episode_ep.jsonl")
    _patch_finalization(trace, "ep")
    return orig_get, orig_ctx, trace


def _restore(orig_get, orig_ctx, trace: JsonlWriter) -> None:
    cli.get_robot_spec = orig_get
    cli.RunFinalizationContext = orig_ctx
    trace.close()


def test_check_compat_sees_outcome_hooks():
    check_compat()


def test_missing_finalizer_records_success_cost_and_tokens(tmp_path):
    class Toolkit:
        def solved(self) -> bool:
            return True

    orig_get, orig_ctx, trace = _patched(tmp_path, lambda name: _Spec(name=name))
    try:
        spec = cli.get_robot_spec("libero")
        toolkit = Toolkit()
        assert spec.finalize_run is not None
        environment_success = bool(toolkit.solved())
        assert spec.finalize_run(
            _context(tmp_path, success=environment_success)
        ) is None
    finally:
        _restore(orig_get, orig_ctx, trace)

    outcome = next(r for r in read_jsonl(trace.path) if r["type"] == "outcome")
    assert outcome["environment_success"] is True
    assert outcome["stats"]["total_input_tokens"] == 1000
    assert outcome["stats"]["total_output_tokens"] == 100
    assert outcome["stats"]["total_cost_usd"] == 1.25
    assert outcome["task"]["suite"] == "libero_object_swap"


def test_existing_finalizer_is_left_in_place(tmp_path):
    seen = {}

    def real(ctx):
        seen["success"] = ctx.environment_success
        return tmp_path / "result.json"

    orig_get, orig_ctx, trace = _patched(tmp_path, lambda name: _Spec(name=name, finalize_run=real))
    try:
        spec = cli.get_robot_spec("yam")
        assert spec.finalize_run is real
        assert spec.finalize_run(_context(tmp_path, success=False)) == tmp_path / "result.json"
    finally:
        _restore(orig_get, orig_ctx, trace)

    assert seen["success"] is False
    outcome = next(r for r in read_jsonl(trace.path) if r["type"] == "outcome")
    assert outcome["environment_success"] is False
    assert outcome["stats"]["total_cost_usd"] == 1.25
