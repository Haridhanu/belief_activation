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


def test_graph_tgn_blend_changes_z_vs_baseline():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    torch.manual_seed(0)
    np.random.seed(0)
    embs = np.random.randn(3, 16).astype(np.float32)
    ids = ["a", "b", "c"]
    edges = [("a", "b", 0.8)]

    # Baseline without TGN
    g_base = Graph(emb_dim=16)
    g_base.extend(ids, embs, [])
    g_base.extend([], np.empty((0, 16), dtype=np.float32), edges)
    z_base_a = g_base._z["a"].copy()

    # Graph with TGN attached
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g_tgn = Graph(emb_dim=16)
    g_tgn._tgn = tgn
    g_tgn.extend(ids, embs, [])
    g_tgn.extend([], np.empty((0, 16), dtype=np.float32), edges)
    z_tgn_a = g_tgn._z["a"]

    assert not np.allclose(z_base_a, z_tgn_a), \
        "_z should differ when TGN memory is blended in"


def test_graph_tgn_z_is_unit_normalized():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    torch.manual_seed(2)
    np.random.seed(2)
    embs = np.random.randn(4, 16).astype(np.float32)
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g = Graph(emb_dim=16)
    g._tgn = tgn
    g.extend(["a", "b", "c", "d"], embs, [])
    g.extend([], np.empty((0, 16), dtype=np.float32),
             [("a", "b", 0.9), ("b", "c", -0.4)])
    for nid in ["a", "b"]:
        norm = np.linalg.norm(g._z[nid])
        assert abs(norm - 1.0) < 1e-5, f"_z[{nid}] not unit: norm={norm}"


def test_graph_tgn_z_tensor_cache_invalidated_after_extend():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    embs = np.random.randn(3, 16).astype(np.float32)
    ids = ["a", "b", "c"]
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g = Graph(emb_dim=16)
    g._tgn = tgn
    g.extend(ids, embs, [])
    # Prime the cache
    _ = g.get_representations_fast(ids)
    assert g._z_tensor is not None
    # New edge must invalidate it
    g.extend([], np.empty((0, 16), dtype=np.float32), [("a", "b", 0.8)])
    assert g._z_tensor is None


def test_psro_step_completes_with_tgn_attached():
    """Full PSROLoop.step() runs without error when TGN is attached to Graph."""
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule
    from multi_agent.config import MultiAgentConfig
    from multi_agent.agent import AgentPopulation
    from multi_agent.psro import PSROLoop
    from multi_agent.judge import StaticJudge

    torch.manual_seed(0)
    np.random.seed(0)

    emb_dim = 16
    n_nodes = 6
    embs = np.random.randn(n_nodes, emb_dim).astype(np.float32)
    node_ids = [f"n{i}" for i in range(n_nodes)]

    tgn = TGNModule(emb_dim=emb_dim, memory_dim=16, time_dim=8, n_heads=2)
    graph = Graph(emb_dim=emb_dim)
    graph._tgn = tgn
    graph.extend(node_ids, embs, [])

    config = MultiAgentConfig(
        emb_dim=emb_dim, num_agents=2, k=2,
        use_tgn=True, tgn_memory_dim=16, tgn_time_dim=8, tgn_n_attn_heads=2,
    )
    population = AgentPopulation(config)
    loop = PSROLoop(config, judge=StaticJudge(0.5), graph=graph)

    q_ids = node_ids[:2]
    q_embs = population.embeddings_to_device(embs[:2])
    pool_embs = population.embeddings_to_device(embs)

    results = loop.step(population, q_embs, q_ids, pool_embs, node_ids)

    # Basic shape checks
    assert len(results) == 2
    assert all("proposals" in r for r in results)
    assert all("rewards" in r for r in results)

    # TGN memories updated by judged edges wired via graph.extend inside PSROLoop
    mems = tgn.get_memory(node_ids)
    assert mems.shape == (n_nodes, 16)
    # No NaN in any memory
    assert not torch.any(torch.isnan(mems))


def test_graph_edge_count_starts_at_zero():
    from multi_agent.graph import Graph

    g = Graph(emb_dim=16)
    assert g._edge_count == 0


def test_graph_edge_count_increments_per_new_edge_without_tgn():
    """_edge_count must increment for every new unique edge, regardless of TGN."""
    from multi_agent.graph import Graph

    embs = np.random.randn(3, 16).astype(np.float32)
    g = Graph(emb_dim=16)
    g.extend(["a", "b", "c"], embs, [("a", "b", 0.9)])
    assert g._edge_count == 1
    g.extend([], np.empty((0, 16), dtype=np.float32), [("b", "c", -0.7)])
    assert g._edge_count == 2


def test_graph_edge_count_increments_with_tgn():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(3, 16).astype(np.float32)
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)
    g = Graph(emb_dim=16)
    g._tgn = tgn
    g.extend(["a", "b", "c"], embs, [("a", "b", 0.9)])
    assert g._edge_count == 1
    g.extend([], np.empty((0, 16), dtype=np.float32), [("b", "c", -0.7)])
    assert g._edge_count == 2


def test_graph_duplicate_edge_does_not_increment_count():
    from multi_agent.graph import Graph

    embs = np.random.randn(2, 16).astype(np.float32)
    g = Graph(emb_dim=16)
    g.extend(["a", "b"], embs, [("a", "b", 0.9)])
    g.extend([], np.empty((0, 16), dtype=np.float32), [("a", "b", 0.9)])
    assert g._edge_count == 1


def test_graph_edge_timestamps_recorded():
    """Every committed edge gets a stable timestamp = its insertion order."""
    from multi_agent.graph import Graph

    embs = np.random.randn(3, 16).astype(np.float32)
    g = Graph(emb_dim=16)
    g.extend(["a", "b", "c"], embs, [("a", "b", 0.9), ("b", "c", -0.5)])
    assert g._edge_timestamps[("a", "b")] == 1
    assert g._edge_timestamps[("b", "c")] == 2


def test_graph_time_decay_field_default():
    from multi_agent.graph import Graph

    g = Graph(emb_dim=16)
    assert g.time_decay == 0.1
    assert g.baseline_norm == 1.0


def test_tgn_use_tgn_false_config_does_not_affect_graph_without_tgn():
    """use_tgn=True in config but no tgn attached to graph → no crash, same behaviour."""
    from multi_agent.graph import Graph
    from multi_agent.config import MultiAgentConfig
    from multi_agent.agent import AgentPopulation
    from multi_agent.psro import PSROLoop
    from multi_agent.judge import StaticJudge

    torch.manual_seed(3)
    np.random.seed(3)
    emb_dim = 16
    embs = np.random.randn(4, emb_dim).astype(np.float32)
    ids = [f"x{i}" for i in range(4)]

    graph = Graph(emb_dim=emb_dim)
    graph.extend(ids, embs, [])

    config = MultiAgentConfig(emb_dim=emb_dim, num_agents=2, k=2, use_tgn=True)
    population = AgentPopulation(config)
    loop = PSROLoop(config, judge=StaticJudge(0.0), graph=graph)

    results = loop.step(
        population,
        population.embeddings_to_device(embs[:2]),
        ids[:2],
        population.embeddings_to_device(embs),
        ids,
    )
    assert len(results) == 2
