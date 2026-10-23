"""Flash replay calls Molmo and the simulator between tools; those are not planner time."""

from __future__ import annotations

import json

from conftest import ROOT  # noqa: F401  (puts the repo on sys.path)

from analysis.breakdown import COMPONENTS, load_level


def _write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_rpcs_outside_tools_are_charged_to_their_service(tmp_path):
    (tmp_path / "traces").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"level": 1, "repeat": 0}))

    def rpc(method, s, e, call=None):
        return {"type": "client_rpc", "episode": "e", "method": method,
                "call_id": call, "tool": None, "t_start": s, "t_end": e, "dur_s": e - s}

    _write(tmp_path / "traces" / "episode_e.jsonl", [
        {"type": "episode_start", "episode": "e", "t": 0.0},
        rpc("env.reset", 1.0, 2.0),                 # startup: before any work
        rpc("molmo.ground", 10.0, 13.0),            # anchor survey before the first tool
        {"type": "tool", "call_id": "c1", "name": "pi0_pick", "t_start": 14.0, "t_end": 24.0},
        rpc("vla.predict", 15.0, 19.0, "c1"),
        rpc("env.chunk_step", 19.0, 23.0, "c1"),
        rpc("env.get_state", 25.0, 26.0),           # observation read between tools
        rpc("molmo.ground", 26.0, 29.0),            # refine between tools
        {"type": "tool", "call_id": "c2", "name": "move_to", "t_start": 30.0, "t_end": 35.0},
        {"type": "outcome", "episode": "e", "environment_success": True, "task": {}, "stats": {}},
        {"type": "episode_end", "episode": "e", "t": 40.0, "exit_code": 0},
    ])
    _write(tmp_path / "proxy_molmo.jsonl", [
        {"episode": "e", "service": "molmo", "gated": True, "wait_s": 1.0, "service_s": 1.5},
        {"episode": "e", "service": "molmo", "gated": True, "wait_s": 0.5, "service_s": 2.0},
    ])
    _write(tmp_path / "proxy_vla.jsonl", [
        {"episode": "e", "service": "vla", "gated": True, "wait_s": 1.0, "service_s": 2.5},
    ])

    eps, _, _, _ = load_level(tmp_path)
    ep = eps.iloc[0]
    assert ep["startup"] == 10.0
    assert ep["molmo_wait"] == 1.5 and ep["molmo_service"] == 3.5
    assert ep["sim"] == 5.0                       # 4 s in tool + 1 s between tools
    assert ep["planner"] == 30.0 - 15.0 - 7.0     # 30 s after start, 15 s tools, 7 s RPCs between
    assert ep["n_molmo_calls"] == 2
    assert abs(sum(ep[c] for c in COMPONENTS) - ep["wall_s"]) < 1e-9
