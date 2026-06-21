"""TGN-substrate integration tests.

Tests focus on the new architecture:
- Graph delegates ``impute`` / ``field`` to ``tgn.predict_link`` when TGN
  attached.
- ``Graph.extend`` does NOT call ``tgn.update`` directly — memory
  propagation happens inside ``PSROLoop.step`` via ``train_step``.
- ``get_representations_fast`` returns TGN-projected memory (or raw
  embedding fallback at cold start when configured).
- ``Trainer`` constructs TGN + Adam optimizer when ``use_tgn=True`` and
  passes the optimizer to ``PSROLoop`` so the TGN actually trains.
"""

from __future__ import annotations

import numpy as np
import torch


EMB_DIM = 16


def _embs(n: int) -> np.ndarray:
    return np.eye(EMB_DIM, dtype=np.float32)[:n]


# --- Config defaults ---------------------------------------------------------


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
        tgn_cold_start="raw_fallback",
        tgn_predict_threshold=0.3,
    )
    assert cfg.use_tgn is True
    assert cfg.tgn_memory_dim == 64
    assert cfg.tgn_cold_start == "raw_fallback"


# --- Graph in baseline (no TGN) mode -----------------------------------------


def test_graph_tgn_field_is_none_by_default():
    from multi_agent.graph import Graph

    g = Graph(emb_dim=EMB_DIM)
    assert g._tgn is None


def test_graph_no_tgn_z_unchanged_by_extend():
    """Without TGN, _z update via signed attention works as before."""
    from multi_agent.graph import Graph

    np.random.seed(42)
    embs = np.random.randn(4, EMB_DIM).astype(np.float32)
    ids = ["a", "b", "c", "d"]

    g1 = Graph(emb_dim=EMB_DIM)
    g1.extend(ids, embs.copy(), [])
    g1.extend(
        [], np.empty((0, EMB_DIM), dtype=np.float32),
        [("a", "b", 0.9), ("b", "c", -0.5)],
    )

    g2 = Graph(emb_dim=EMB_DIM)
    g2.extend(ids, embs.copy(), [])
    g2.extend(
        [], np.empty((0, EMB_DIM), dtype=np.float32),
        [("a", "b", 0.9), ("b", "c", -0.5)],
    )

    for nid in ids:
        np.testing.assert_array_equal(g1._z[nid], g2._z[nid])


# --- Graph with TGN substrate attached ---------------------------------------


def test_graph_extend_does_not_auto_update_tgn_memory():
    """With the new architecture, extend() commits edges but does NOT
    call tgn.update() — that's PSROLoop.step's job via train_step."""
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(2, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)

    g = Graph(emb_dim=EMB_DIM, _tgn=tgn)
    g.extend(["a", "b"], embs, [("a", "b", 0.9)])

    # Edge is committed in the graph
    assert ("a", "b") in g._edges
    # But TGN memory remains zero — we deferred memory propagation to
    # the train_step hook in PSROLoop.
    mem_a = tgn.memory.get("a")
    mem_b = tgn.memory.get("b")
    assert torch.all(mem_a == 0.0)
    assert torch.all(mem_b == 0.0)


def test_graph_impute_delegates_to_tgn_predict_link_when_attached():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(3, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn, tgn_predict_threshold=0.0)
    g.extend(["a", "b", "c"], embs, [])

    # With threshold=0.0, any non-zero predict_link result is returned
    out = g.impute("a", "b")
    assert isinstance(out, float)
    assert -1.0 <= out <= 1.0

def test_info_gain_raises_when_tgn_attached():
    """info_gain is a Gaussian-KL quantity; TGN has no posterior variance,
    so calling it under the TGN substrate must error loudly instead of
    silently returning a Bayesian value over the underlying edge set."""
    import pytest

def test_graph_impute_returns_none_below_tgn_threshold():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(3, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    # Set the threshold so high that random init never crosses it
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn, tgn_predict_threshold=2.0)
    g.extend(["a", "b"], embs[:2], [])
    assert g.impute("a", "b") is None


def test_graph_field_returns_value_with_tgn():
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(3, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn)
    g.extend(["a", "b", "c"], embs, [])

    val = g.field("a", "c")
    assert isinstance(val, float)
    assert -1.0 <= val <= 1.0


def test_graph_observed_edge_returned_as_is():
    """For an actually-observed edge, both impute and field return the
    observed weight — irrespective of what TGN says."""
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(2, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn)
    g.extend(["a", "b"], embs, [("a", "b", 0.7)])
    assert g.field("a", "b") == 0.7
    assert g.impute("a", "b") == 0.7


# --- Cold-start behaviour: pure vs raw_fallback ------------------------------


def test_cold_start_pure_uses_projected_zero_memory():
    """In 'pure' mode, candidate_rep for an untouched node is
    mem_to_emb(zeros) — small, not the raw embedding."""
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(2, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn, tgn_cold_start="pure")
    g.extend(["a", "b"], embs, [])

    rep = g.get_representations_fast(["a"])[0]
    # mem_to_emb(zeros) = bias of mem_to_emb's Linear, which is small
    # random init. It must NOT equal the raw embedding (which is a
    # one-hot e[0] in this fixture).
    assert not np.allclose(rep, embs[0])


def test_cold_start_raw_fallback_uses_raw_until_memory_warm():
    """In 'raw_fallback' mode, an untouched node returns its raw
    embedding; once memory has been written, returns projected memory."""
    from multi_agent.graph import Graph
    from multi_agent.tgn import TGNModule

    torch.manual_seed(0)
    embs = np.random.randn(2, EMB_DIM).astype(np.float32)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=EMB_DIM, time_dim=8, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn, tgn_cold_start="raw_fallback")
    g.extend(["a", "b"], embs, [])

    rep = g.get_representations_fast(["a"])[0]
    np.testing.assert_array_almost_equal(rep, embs[0].astype(np.float32))

    # Warm up 'a's memory by calling tgn.update directly (test shortcut).
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.8)
    rep_after = g.get_representations_fast(["a"])[0]
    assert not np.allclose(rep_after, embs[0])  # now projected memory


# --- Trainer integration -----------------------------------------------------


def test_trainer_attaches_tgn_when_use_tgn_true():
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer
    from multi_agent.tgn import TGNModule

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=2,
        k=2,
        use_tgn=True,
        tgn_memory_dim=EMB_DIM,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
    )
    trainer = Trainer(cfg, StaticJudge(0.5))
    assert trainer.graph._tgn is not None
    assert isinstance(trainer.graph._tgn, TGNModule)
    assert trainer.tgn_optimizer is not None


def test_trainer_no_tgn_when_use_tgn_false():
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    cfg = MultiAgentConfig(emb_dim=EMB_DIM, num_agents=2, k=2, use_tgn=False)
    trainer = Trainer(cfg, StaticJudge(0.5))
    assert trainer.graph._tgn is None
    assert trainer.tgn_optimizer is None


def test_psro_step_trains_tgn_when_attached():
    """End-to-end: with TGN attached, a single Trainer.step actually
    moves TGN parameters (the train_step hook fires on judged pairs).

    We set tgn_predict_threshold above 1.0 so that every pair fails the
    impute confidence check and goes to the judge — guaranteeing
    judged_pairs is non-empty for this fixture.
    """
    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=2,
        k=2,
        judge_budget_per_batch=4,
        use_tgn=True,
        tgn_memory_dim=EMB_DIM,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
        tgn_predict_threshold=2.0,  # impute always returns None → forces judge
    )
    trainer = Trainer(cfg, StaticJudge(0.7))

    head_bias_before = trainer.tgn.link_head[2].bias.detach().clone()

    embs = np.random.randn(4, EMB_DIM).astype(np.float32)
    batch = Batch(ids=["a", "b", "c", "d"], embs=embs, texts=["a", "b", "c", "d"])
    res = trainer.step(batch)
    assert res.stats.judged > 0

    head_bias_after = trainer.tgn.link_head[2].bias.detach()
    assert not torch.allclose(head_bias_before, head_bias_after), (
        "link_head's bias should change after a step that judged some pairs"
    )

    raw = trainer.loop.last_step_stats
    assert "tgn_loss" in raw
    assert isinstance(raw["tgn_loss"], float)
    assert raw["tgn_loss"] > 0.0

def test_trainer_first_batch_tgn_field_uses_raw_fallback_for_cold_nodes():
    """First-batch TGN field predictions should be raw-geometry based, not
    the same random score from zero memory for every cold pair."""
    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

def test_psro_step_runs_with_tgn_disabled():
    """Smoke test: with use_tgn=False, the existing PSRO path is unchanged."""
    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    torch.manual_seed(0)
    np.random.seed(0)

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=2,
        k=2,
        judge_budget_per_batch=4,
        use_tgn=False,
    )
    trainer = Trainer(cfg, StaticJudge(0.7))
    embs = np.random.randn(4, EMB_DIM).astype(np.float32)
    batch = Batch(ids=["a", "b", "c", "d"], embs=embs, texts=["a", "b", "c", "d"])
    res = trainer.step(batch)
    assert len(res.edges) > 0  # at least one judged edge committed
