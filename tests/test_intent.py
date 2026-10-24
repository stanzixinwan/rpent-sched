"""G1 predictor definitions on a synthetic sweep.

Two episodes disagree on how many VLA calls a (tool, max_chunks) pair uses.
Leave-one-episode-out intent must predict each episode from the other one.
At a tool change, that prediction beats copying the previous primitive.
"""

from __future__ import annotations

import importlib.util
import json

import pandas as pd
from conftest import ROOT


def _intent():
    path = ROOT / "analysis" / "intent.py"
    spec = importlib.util.spec_from_file_location("intent_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tool(episode, call_id, t, name, n_vla, dur, max_chunks=None):
    args = {"prompt": name}
    if max_chunks is not None:
        args["max_chunks"] = max_chunks
    records = [
        {
            "type": "tool", "episode": episode, "call_id": call_id, "tool": name,
            "args": args, "t_start": t, "t_end": t + dur, "dur_s": dur, "ok": True,
        }
    ]
    for i in range(n_vla):
        records.append(
            {
                "type": "client_rpc", "episode": episode, "call_id": call_id,
                "method": "vla.predict", "t_start": t, "t_end": t + 0.1, "dur_s": 0.1,
            }
        )
    if n_vla == 0:
        records.append(
            {
                "type": "client_rpc", "episode": episode, "call_id": call_id,
                "method": "env.chunk_step", "t_start": t, "t_end": t + 0.1, "dur_s": 0.1,
            }
        )
    return records


def _write_episode(directory, episode, steps):
    recs = [{"type": "episode_start", "episode": episode, "t": 0}]
    for i, (name, n_vla, dur, max_chunks) in enumerate(steps):
        recs.extend(_tool(episode, f"{episode}-{i:04d}", i + 1, name, n_vla, dur, max_chunks))
    recs.append({"type": "episode_end", "episode": episode, "t": 100, "exit_code": 0})
    path = directory / f"episode_{episode}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))


def test_intent_beats_history_at_tool_changes(tmp_path, capsys):
    level = tmp_path / "N1_r0"
    (level / "traces").mkdir(parents=True)
    (level / "manifest.json").write_text(json.dumps({"level": 1, "repeat": 0, "t_start": 0, "t_end": 1}))
    # Same durations, different VLA counts, so the (tool, max_chunks) mean is the other episode.
    _write_episode(level / "traces", "A", [
        ("segment", 0, 3, None),
        ("pi0_pick", 10, 14, 12),
        ("pi0_pick", 4, 5, 8),
        ("move_to", 0, 6, None),
        ("move_to", 0, 6, None),
    ])
    _write_episode(level / "traces", "B", [
        ("segment", 0, 3, None),
        ("pi0_pick", 20, 14, 12),
        ("pi0_pick", 8, 5, 8),
        ("move_to", 0, 6, None),
        ("move_to", 0, 6, None),
    ])
    (level / "traces" / "episode_C.jsonl").write_text(
        '{"type":"episode_start","episode":"C","t":0}\n{"type":"episode_end","episode":"C","t":1}\n'
    )

    _intent().main([str(tmp_path)])
    text = capsys.readouterr().out
    assert "skipped_no_tools: 1" in text

    prims = pd.read_csv(tmp_path / "analysis" / "intent_primitives.csv")
    summary = pd.read_csv(tmp_path / "analysis" / "intent_summary.csv")
    assert (tmp_path / "analysis" / "intent_mae.png").exists()

    a = prims[(prims["episode"] == "A") & (prims["tool"] == "pi0_pick") & (prims["max_chunks"] == 12)].iloc[0]
    b = prims[(prims["episode"] == "B") & (prims["tool"] == "pi0_pick") & (prims["max_chunks"] == 12)].iloc[0]
    assert a["pred_intent_n_vla"] == 20
    assert b["pred_intent_n_vla"] == 10
    # The second pick keeps the tool name but changes max_chunks. Intent follows the argument.
    a8 = prims[(prims["episode"] == "A") & (prims["max_chunks"] == 8)].iloc[0]
    assert a8["boundary"] == False  # noqa: E712
    assert a8["pred_intent_n_vla"] == 8

    def mae(target, slice_name, predictor):
        hit = summary[
            (summary["target"] == target)
            & (summary["slice"] == slice_name)
            & (summary["predictor"] == predictor)
        ]
        return float(hit["mae"].iloc[0])

    for target in ("n_vla", "dur_s"):
        assert mae(target, "boundary", "intent") < mae(target, "boundary", "last_round")
        assert mae(target, "boundary", "intent") < mae(target, "boundary", "fixed_ratio")
        assert mae(target, "vla", "intent") < mae(target, "vla", "last_round")
        assert mae(target, "vla", "intent") < mae(target, "vla", "fixed_ratio")
    vla_n = summary[(summary["slice"] == "vla") & (summary["target"] == "n_vla")]["n"]
    assert set(vla_n) == {4}  # two pi0_pick calls in each episode; segment and move_to stay out


def _level(root, level, episodes):
    d = root / f"N{level}_r0"
    (d / "traces").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"level": level, "repeat": 0, "t_start": 0, "t_end": 1}))
    for name, steps in episodes.items():
        _write_episode(d / "traces", name, steps)


def test_duration_is_fit_per_level_and_scaled_by_observed_rate(tmp_path):
    # Same plan at two loads: a pick makes 10 VLA calls; under N=16 each call queues.
    calm = [("pi0_pick", 10, 5.0, 12), ("move_to", 0, 2.0, None), ("pi0_pick", 10, 5.0, 12)]
    busy = [("pi0_pick", 10, 50.0, 12), ("move_to", 0, 2.0, None), ("pi0_pick", 10, 40.0, 12)]
    _level(tmp_path, 1, {"A": calm, "B": calm})
    _level(tmp_path, 16, {"C": busy, "D": busy})

    mod = _intent()
    mod.main([str(tmp_path)])
    prims = pd.read_csv(tmp_path / "analysis" / "intent_primitives.csv")
    c = prims[prims["episode"] == "C"].sort_values("t_start")
    # Same-level fit: C is predicted from D only, not from the calm N=1 runs.
    assert c.iloc[0]["pred_intent_dur_s"] == 45.0  # mean of D's picks (50, 40)
    # Second pick: 10 predicted calls x 5 s/call observed on C's first pick.
    assert c.iloc[2]["pred_intent_rate_dur_s"] == 50.0
    # No VLA call observed yet: falls back to intent.
    assert c.iloc[0]["pred_intent_rate_dur_s"] == c.iloc[0]["pred_intent_dur_s"]

    summary = pd.read_csv(tmp_path / "analysis" / "intent_summary.csv", dtype={"level": str})
    assert {"all", "1", "16"} <= set(summary["level"])
    assert not ((summary["target"] == "n_vla") & (summary["predictor"] == "intent_rate")).any()

    mod.main([str(tmp_path), "--pool-levels", "--out", str(tmp_path / "pooled")])
    pooled = pd.read_csv(tmp_path / "pooled" / "intent_primitives.csv")
    c0 = pooled[pooled["episode"] == "C"].sort_values("t_start").iloc[0]
    assert round(c0["pred_intent_dur_s"], 3) == round((50 + 40 + 4 * 5) / 6, 3)  # calm level mixed in
