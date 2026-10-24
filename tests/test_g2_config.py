"""G2 Flash config: no LLM planner, Molmo endpoint, and the tasks that have plans."""

from __future__ import annotations

import yaml
from conftest import ROOT

from rsched.sweep import EpisodeSpec, Level, episode_pool, preflight_planner


def test_flash_sweep_is_ready_for_a_cloud_gpu(tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "g2_flash.yaml").read_text())
    assert cfg["rpent"]["planner"] == "flash"
    assert "model" not in cfg["rpent"]
    assert cfg["concurrency"] == [1, 4, 8, 16]
    assert cfg["episode_env"]["OMP_NUM_THREADS"] == 1
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
    assert cfg["services"]["molmo"]["policy"] == "fifo"
    assert cfg["services"]["molmo"].get("max_inflight", 1) == 1


def test_proxy_gates_molmo():
    from rsched.proxy import DEFAULT_GATE_METHODS

    assert "molmo.ground" in DEFAULT_GATE_METHODS


def test_uncapped_control_differs_only_in_threads_and_levels():
    capped = yaml.safe_load((ROOT / "configs" / "g2_flash.yaml").read_text())
    ctl = yaml.safe_load((ROOT / "configs" / "g2_flash_uncapped.yaml").read_text())
    assert "episode_env" not in ctl and ctl["concurrency"] == [4]
    drop = {"name", "episode_env", "concurrency"}
    assert {k: v for k, v in capped.items() if k not in drop} == \
           {k: v for k, v in ctl.items() if k not in drop}


def test_cpu_info_is_jsonable():
    import json

    from rsched.sweep import cpu_info

    info = cpu_info()
    json.dumps(info)
    assert info["os_cpu_count"] >= 1
