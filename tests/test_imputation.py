"""Tests for the BlendedImputer."""

from __future__ import annotations

import numpy as np
import torch

from multi_agent.graph import Graph
from multi_agent.imputation import BlendedImputer, ImputationScorer
from multi_agent.tgn import TGNModule


EMB_DIM = 16
MEM_DIM = 16
TIME_DIM = 8


def _embs(n: int) -> np.ndarray:
    return np.eye(EMB_DIM, dtype=np.float32)[:n]


def _make_graph_with_imputer(seed: int = 0) -> tuple[Graph, BlendedImputer]:
    torch.manual_seed(seed)
    tgn = TGNModule(emb_dim=EMB_DIM, memory_dim=MEM_DIM, time_dim=TIME_DIM, n_heads=2)
    g = Graph(emb_dim=EMB_DIM, _tgn=tgn)
    imputer = BlendedImputer(graph=g, tgn=tgn, lr=1e-2)
    g._imputer = imputer
    return g, imputer


def test_scorer_outputs_softmax_three_weights():
    s = ImputationScorer()
    out = s(torch.tensor([0.5, -0.2, 0.1], dtype=torch.float32), density=0.3)
    assert out.shape == (3,)
    assert abs(float(out.sum().item()) - 1.0) < 1e-5
    assert torch.all(out >= 0.0)


def test_imputer_returns_observed_edge_directly():
    g, imp = _make_graph_with_imputer()
    g.extend(["a", "b"], _embs(2), [("a", "b", 0.7)])
    assert imp.impute("a", "b") == 0.7
    assert imp.field("a", "b") == 0.7


def test_imputer_returns_none_below_confidence_floor():
    """With brand-new gate weights and no edges, confidence should be too low."""
    g, imp = _make_graph_with_imputer()
    g.extend(["a", "b"], _embs(2), [])  # no edges → tgn=0, bayes=0, cosine=0 (orthogonal)
    out = imp.impute("a", "b")
    # All three components are zero/near-zero → tgn/bayes agreement is 1, but
    # max gate weight needs to clear the floor. With softmax-of-untrained-MLP
    # the max weight is typically ~0.4 — set a high floor and confirm None.
    imp.confidence_floor = 0.99
    assert imp.impute("a", "b") is None


def test_imputer_field_always_returns_a_value():
    g, imp = _make_graph_with_imputer()
    g.extend(["a", "b", "c"], _embs(3), [])
    f = imp.field("a", "c")
    assert isinstance(f, float)
    assert -1.0 <= f <= 1.0


def test_train_on_judged_decreases_loss():
    """Repeated training on the same supervised pairs must drive loss down."""
    g, imp = _make_graph_with_imputer()
    g.extend(["a", "b", "c"], _embs(3), [("a", "b", 0.9)])

    judged = [(("a", "c"), 0.8), (("b", "c"), -0.4)]
    losses = [imp.train_on_judged(judged) for _ in range(40)]
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} → {losses[-1]}"


def test_imputer_records_weight_history():
    g, imp = _make_graph_with_imputer()
    g.extend(["a", "b", "c"], _embs(3), [("a", "b", 0.9), ("b", "c", -0.3)])
    imp.confidence_floor = -1.0  # force non-None to populate history
    _ = imp.impute("a", "c")
    assert len(imp.weight_history) == 1
    weights = imp.weight_history[0]
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-5


def test_graph_without_imputer_uses_bayesian_path():
    """Regression: with no imputer attached, graph.impute is unchanged."""
    g_ref = Graph(emb_dim=EMB_DIM)
    g_ref.extend(
        ["a", "b", "c"], _embs(3), [("a", "b", 0.9), ("b", "c", 0.8)]
    )
    g_new = Graph(emb_dim=EMB_DIM, _imputer=None)
    g_new.extend(
        ["a", "b", "c"], _embs(3), [("a", "b", 0.9), ("b", "c", 0.8)]
    )
    assert g_ref.impute("a", "c") == g_new.impute("a", "c")


def test_trainer_attaches_imputer_when_use_tgn_true():
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=2,
        k=2,
        use_tgn=True,
        tgn_memory_dim=MEM_DIM,
        tgn_time_dim=TIME_DIM,
        tgn_n_attn_heads=2,
    )
    trainer = Trainer(cfg, StaticJudge(0.5))
    assert trainer.graph._imputer is not None
    assert isinstance(trainer.graph._imputer, BlendedImputer)


def test_trainer_no_imputer_when_use_tgn_false():
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    cfg = MultiAgentConfig(emb_dim=EMB_DIM, num_agents=2, k=2, use_tgn=False)
    trainer = Trainer(cfg, StaticJudge(0.5))
    assert trainer.graph._imputer is None


def test_psro_step_records_imputer_loss_when_active():
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
        use_tgn=True,
        tgn_memory_dim=MEM_DIM,
        tgn_time_dim=TIME_DIM,
        tgn_n_attn_heads=2,
    )
    trainer = Trainer(cfg, StaticJudge(0.7))

    embs = np.random.randn(4, EMB_DIM).astype(np.float32)
    batch = Batch(ids=["a", "b", "c", "d"], embs=embs, texts=["a", "b", "c", "d"])
    trainer.step(batch)

    raw = trainer.loop.last_step_stats
    assert "imputer_loss" in raw
    # Loss is float — could be 0.0 if the budget was 0 and nothing was judged,
    # but the field must be present.
    assert isinstance(raw["imputer_loss"], float)
