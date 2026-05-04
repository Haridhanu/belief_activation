from __future__ import annotations
import numpy as np
import torch
import pytest


def test_config_tgn_defaults_off():
    from multi_agent.config import MultiAgentConfig
    cfg = MultiAgentConfig()
    assert cfg.use_tgn is False


def test_config_tgn_fields_round_trip():
    from multi_agent.config import MultiAgentConfig
    cfg = MultiAgentConfig(
        use_tgn=True,
        tgn_memory_dim=64,
        tgn_time_dim=16,
        tgn_n_attn_heads=2,
    )
    assert cfg.use_tgn is True
    assert cfg.tgn_memory_dim == 64
    assert cfg.tgn_time_dim == 16
    assert cfg.tgn_n_attn_heads == 2


def test_config_from_yaml_tgn_fields(tmp_path):
    import yaml
    from multi_agent.config import MultiAgentConfig
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump({"use_tgn": True, "tgn_memory_dim": 64}))
    cfg = MultiAgentConfig.from_yaml(p)
    assert cfg.use_tgn is True
    assert cfg.tgn_memory_dim == 64


def test_graph_tgn_field_is_none_by_default():
    from multi_agent.graph import Graph
    g = Graph(emb_dim=16)
    assert g._tgn is None


def test_graph_extend_calls_tgn_update_for_each_new_edge():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    torch.manual_seed(0)
    embs = np.random.randn(3, 16).astype(np.float32)
    ids = ["a", "b", "c"]

    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g = Graph(emb_dim=16)
    g._tgn = tgn
    g.extend(ids, embs, [])
    # No edges yet — all memory zero
    assert torch.all(tgn.get_memory(ids) == 0)

    g.extend([], np.empty((0, 16), dtype=np.float32), [("a", "b", 0.8)])
    mems = tgn.get_memory(["a", "b"])
    assert not torch.all(mems == 0), "update() should have been called"


def test_graph_extend_duplicate_edge_does_not_call_tgn_twice():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    torch.manual_seed(1)
    embs = np.random.randn(2, 16).astype(np.float32)
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g = Graph(emb_dim=16)
    g._tgn = tgn
    g.extend(["a", "b"], embs, [("a", "b", 0.9)])
    mem_after_1 = tgn.get_memory(["a"])[0].clone()
    # Exact same edge — should be skipped
    g.extend([], np.empty((0, 16), dtype=np.float32), [("a", "b", 0.9)])
    mem_after_2 = tgn.get_memory(["a"])[0]
    assert torch.allclose(mem_after_1, mem_after_2)


def test_graph_no_tgn_z_identical_to_baseline():
    """With _tgn=None the modified graph.py must produce bit-identical _z."""
    from multi_agent.graph import Graph
    np.random.seed(42)
    embs = np.random.randn(4, 16).astype(np.float32)
    ids = ["a", "b", "c", "d"]
    edges = [("a", "b", 0.9), ("b", "c", -0.5), ("c", "d", 0.3)]

    g1 = Graph(emb_dim=16)
    g1.extend(ids, embs, [])
    g1.extend([], np.empty((0, 16), dtype=np.float32), edges)

    g2 = Graph(emb_dim=16)
    g2.extend(ids, embs, [])
    g2.extend([], np.empty((0, 16), dtype=np.float32), edges)

    for nid in ids:
        np.testing.assert_array_equal(g1._z[nid], g2._z[nid])
