"""RPent's Molmo server with a point parser that accepts both Molmo2 formats.

    PYTHONPATH=<repo>:<repo>/third_party/RPent python -m rsched.molmo_server <molmo_server args>

RPent's ``_parse_point`` reads ``coords="<id> <x> <y>"``. The Molmo2-8B weights
we run emit a leading image index, ``coords="<image> <id> <x> <y>"``, so RPent
took ``(<id>, <x>)`` as the point: every grounding landed on the left image
edge and Flash replay stopped or grasped at the wrong place (G2 N=1, 0/16).
RPent stays unmodified; this patches the module before the server starts.
"""

from __future__ import annotations

import re

_GROUPS = re.compile(r"[;\t]")
_INT = re.compile(r"\d+")


def parse_point(answer: str) -> tuple[float, float] | None:
    """First point of Molmo2 markup, with or without a leading image index."""
    from rpent.robots.components import molmo_server as ms

    coords = ms._COORDS.search(answer)
    if coords is None:
        return None
    for group in _GROUPS.split(coords.group(1)):
        nums = [int(n) for n in _INT.findall(group)]
        if len(nums) >= 4 and len(nums) % 3 == 1:
            nums = nums[1:]  # drop the image index
        if len(nums) < 3:
            continue
        x, y = float(nums[1]), float(nums[2])
        if 0 <= x <= 1000 and 0 <= y <= 1000:
            return x, y
        return None
    return None


def main() -> None:
    from rpent.robots.components import molmo_server as ms

    ms._parse_point = parse_point
    ms.main()


if __name__ == "__main__":
    main()
