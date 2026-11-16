"""X3: where inside the episode and the simulator does the time go? From py-spy collapsed stacks.

    python analysis/profile_summary.py <traces dir> [<traces dir> ...] [--top 25] [--out csv]

Reads ``pyspy_*.txt`` (``py-spy record --format raw --subprocesses --idle``):
one line per distinct stack, frames separated by ``;`` from the outermost in,
followed by the sample count. With --subprocesses the first frame names the
process. Prints, per process kind (episode / env_server / other):

* samples by category of the innermost frame (waiting on a socket, sleeping,
  MuJoCo / rendering, numpy, image encoding, JSON/HTTP, other Python),
* the functions with the most samples in their own frame.

Samples are wall-clock (idle threads included), so shares can be read against
the "client" and "simulator" parts of analysis/anatomy.py. Threads add
up: a process with a mostly idle helper thread shows that thread as waiting.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import pandas as pd

CATEGORIES = [  # first match on the innermost frames wins
    ("sleep", r"\bsleep\b"),
    ("wait: socket / HTTP", r"(socket\.py|http/client\.py|selectors\.py|ssl\.py|urllib|\brecv|readinto|\bselect\b|\bpoll\b)"),
    ("wait: lock / queue / thread", r"(threading\.py|queue\.py|\bwait\b|\bacquire\b|concurrent/futures)"),
    ("simulator: rendering", r"(render|egl|opengl)"),
    ("simulator: MuJoCo physics", r"(mj_step|mj_forward|mujoco|robosuite|\bsim\.step)"),
    ("image encode / decode", r"(PIL|cv2|imageio|base64|jpeg|png|encode|decode)"),
    ("serialization", r"(json|pickle|msgpack|struct)"),
    ("numpy / linear algebra", r"(numpy|scipy|linalg|\bnp\.)"),
]


def kind_of(stack: list[str]) -> str:
    head = " ".join(stack[:3]).lower()
    if "env_server" in head:
        return "env_server"
    if "rsched.launch" in head or "rsched/launch" in head or "rpent" in head:
        return "episode"
    return "other"


def category(stack: list[str]) -> str:
    inner = " ".join(stack[-3:])
    for name, pat in CATEGORIES:
        if re.search(pat, inner, flags=re.IGNORECASE):
            return name
    return "other Python"


def load(files: list[Path]) -> pd.DataFrame:
    rows = []
    for f in files:
        for line in f.open(errors="replace"):
            line = line.rstrip("\n")
            stack_text, _, count = line.rpartition(" ")
            if not stack_text or not count.isdigit():
                continue
            stack = stack_text.split(";")
            rows.append({"file": f.name, "kind": kind_of(stack), "category": category(stack),
                         "leaf": stack[-1], "samples": int(count)})
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None, help="write the category table as CSV")
    args = ap.parse_args(argv)
    files = [f for d in args.dirs for f in sorted(Path(d).glob("pyspy_*.txt"))]
    if not files:
        sys.exit("no pyspy_*.txt found")
    df = load(files)
    if df.empty:
        sys.exit("profiles are empty (did py-spy have ptrace rights?)")
    cat = df.groupby(["kind", "category"])["samples"].sum().reset_index()
    cat["share_of_kind"] = cat["samples"] / cat.groupby("kind")["samples"].transform("sum")
    cat = cat.sort_values(["kind", "samples"], ascending=[True, False])
    print(f"{len(files)} profiles, {df['samples'].sum()} samples")
    print(cat.assign(share_of_kind=(cat["share_of_kind"] * 100).round(1)).to_string(index=False))
    for kind, g in df.groupby("kind"):
        top = g.groupby("leaf")["samples"].sum().sort_values(ascending=False).head(args.top)
        print(f"\n== {kind}: top functions by own samples")
        print((top / g["samples"].sum() * 100).round(1).to_string())
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cat.to_csv(args.out, index=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
