"""Build a per-task history table ("task -> measured seconds") from earlier sweep levels.

    python -m rsched.history_table results/g5/<stamp>/record/N8_r0 [more level dirs] \
        --out results/g5/<stamp>/history.json

The value per task is the mean wall time of its finished episodes (at least one
tool call, not timed out), as a batch-evaluation operator would have it from a
previous run: no proxy logs needed. ``launch.estimate: history`` (or
``hybrid``) with ``launch.history_file`` reads it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsched.launch_order import task_key  # noqa: E402


def build(level_dirs: list[Path]) -> dict:
    walls: dict[str, list[float]] = {}
    skipped = 0
    for d in level_dirs:
        m = json.loads((Path(d) / "manifest.json").read_text())
        spec = {s["episode_id"]: s for s in m.get("specs") or []}
        for e in m.get("episodes") or []:
            s = spec.get(e["episode"])
            if s is None or e.get("returncode") == "timeout" or not e.get("tool_calls"):
                skipped += 1
                continue
            walls.setdefault(task_key(s["suite"], int(s["task"])), []).append(float(e["wall_s"]))
    return {"tasks": {k: sum(v) / len(v) for k, v in sorted(walls.items())},
            "episodes": {k: len(v) for k, v in sorted(walls.items())},
            "skipped": skipped, "sources": [str(d) for d in level_dirs]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("levels", nargs="+", help="sweep level directories (with manifest.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    table = build([Path(p) for p in args.levels])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, indent=2))
    print(f"{len(table['tasks'])} tasks from {sum(table['episodes'].values())} episodes "
          f"({table['skipped']} skipped) -> {args.out}")


if __name__ == "__main__":
    main()
