"""RS_ARTIFACTS=lazy keeps what readers see identical and writes only what is read."""

import importlib

import numpy as np
import pytest

base = pytest.importorskip("rpent.session.base")


def _fresh(monkeypatch, mode):
    monkeypatch.setenv("RS_ARTIFACTS", mode)
    importlib.reload(base)  # undo earlier patching
    import rsched.artifacts as art
    importlib.reload(art)
    assert art.install() is (mode == "lazy")
    return base.EnvState, art


def _run(EnvState, root):
    st = EnvState(root)
    img = (np.arange(64 * 64 * 3) % 251).astype(np.uint8).reshape(64, 64, 3)
    dep = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    for k in range(3):
        with st.record_step(state={"k": k}, terminated=False, truncated=False) as i:
            st.save("agentview.png", img + k, step=i)
            st.save("agentview_depth.npz", dep * k, step=i)
            st.save("agentview_metadata.json", {"k": k}, step=i)
    return st, img, dep


def test_lazy_reads_match_store(tmp_path, monkeypatch):
    E, _ = _fresh(monkeypatch, "store")
    a, img, dep = _run(E, tmp_path / "store")
    want = [(a.load("agentview.png", step=k), a.load("agentview_depth.npz", step=k)) for k in range(3)]

    E, art = _fresh(monkeypatch, "lazy")
    b, _, _ = _run(E, tmp_path / "lazy")
    # nothing encoded yet, json written as usual
    assert not (tmp_path / "lazy" / "agentview.png").exists()
    assert (tmp_path / "lazy" / "agentview_metadata.json" / "02.json").exists()
    assert art.STATS["deferred"] == 6
    np.testing.assert_array_equal(b.load("agentview.png", step=1), want[1][0])
    np.testing.assert_array_equal(b.load("agentview_depth.npz", step=2), want[2][1])
    assert b.exists("agentview_depth.npz", step=0)
    assert art.STATS["written"] == 3
    assert "agentview.png" in b.get(0).artifacts
    assert not (tmp_path / "lazy" / "agentview.png" / "00.png").exists()
    b.prune_artifacts(("agentview.png",), step=2, keep_last=2)  # drops step 0 unwritten
    assert art.STATS["dropped"] == 1
    importlib.reload(base)
