"""G2 Flash config: no LLM planner, Molmo endpoint, and the tasks that have plans."""

from __future__ import annotations

import yaml
from conftest import ROOT

from rsched.sweep import EpisodeSpec, Level, episode_pool, preflight_planner


def test_flash_sweep_is_ready_for_a_cloud_gpu(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "g2_flash.yaml").read_text())
    assert cfg["rpent"]["planner"] == "flash"
    assert "model" not in cfg["rpent"]
    assert cfg["concurrency"] == [1, 4, 16]
    assert cfg["episodes_per_level"] >= 16
    assert preflight_planner(cfg) is None

    pool = episode_pool(cfg)
    assert ("libero_goal_swap", 0, 0) not in pool
    assert ("libero_10_swap", 9, 0) not in pool
    assert ("libero_object_swap", 2, 0) in pool

    level = Level(cfg, 1, 0, tmp_path, dry_run=True)
    cmd = level.episode_cmd(
        EpisodeSpec("e", "libero_object_swap", 2, 0),
        {"vla": "http://vla", "sam3": "http://sam3", "molmo": "http://molmo"},
    )
    assert cmd[cmd.index("--planner") + 1] == "flash"
    assert cmd[cmd.index("--molmo-endpoint") + 1] == "http://molmo"
    assert "--model" not in cmd
    assert cfg["services"]["vla"]["policy"] == "fifo"
    assert cfg["services"]["molmo"]["policy"] == "passthrough"
