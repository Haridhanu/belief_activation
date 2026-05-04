"""Unit tests for Graph's Bayesian belief methods (impute / info_gain)."""

from __future__ import annotations

import numpy as np

from multi_agent.graph import Graph


EMB_DIM = 16


def _graph(ids: list[str]) -> Graph:
    g = Graph(emb_dim=EMB_DIM)
    embs = np.eye(EMB_DIM, dtype=np.float32)[: len(ids)].copy()
    g.extend(ids, embs, edges=[])
    return g


def test_belief_graph_empty_defers_every_pair():
    """No edges → impute() returns None for every pair."""
    g = _graph(["a", "b", "c"])
    for q in ["a", "b", "c"]:
        for c in ["a", "b", "c"]:
            if q == c:
                continue
            assert g.impute(q, c) is None


def test_single_observation_propagates_through_attention_edge():
    """Edge on (k, c) + edge (q, k) = 1 → impute(q, c) ≈ score on (k, c)."""
    g = _graph(["q", "k", "c"])
    g.extend([], np.empty((0, EMB_DIM), dtype=np.float32), [("q", "k", 1.0)])
    g.confidence_floor = 0.1

    assert g.impute("q", "c") is None  # no support for q→c yet

    g.extend([], np.empty((0, EMB_DIM), dtype=np.float32), [("k", "c", 0.8)])
    imputed = g.impute("q", "c")
    assert imputed is not None
    assert imputed > 0.5, imputed
    assert imputed <= 0.8 + 1e-6, imputed


def test_info_gain_zero_when_observation_is_unrelated():
    """Unrelated observation (y ≈ 0) → info_gain ≈ 0 regardless of prior."""
    g = _graph(["q", "c"])
    assert g.info_gain("q", "c", y=0.0) == 0.0


def test_info_gain_large_when_resolving_low_confidence_dissonance():
    """Broad prior + strong |y| → large info gain. Tight prior → small gain."""
    g = _graph(["q", "k", "c"])
    g.extend([], np.empty((0, EMB_DIM), dtype=np.float32), [("q", "k", 1.0)])

    # No neighboring observations at (k, c) → broad prior at (q, c).
    broad_gain = g.info_gain("q", "c", y=-0.9)

    # Tighten the prior by adding (k, c) = -0.9.
    g.extend([], np.empty((0, EMB_DIM), dtype=np.float32), [("k", "c", -0.9)])
    tight_gain = g.info_gain("q", "c", y=-0.9)

    assert broad_gain > tight_gain + 0.1, (broad_gain, tight_gain)


def test_info_gain_positive_on_first_obs_zero_on_repeat():
    """Pair with prior signal gives IG > 0 before it's wired; 0 after."""
    g = _graph(["q", "k", "c"])
    g.extend(
        [],
        np.empty((0, EMB_DIM), dtype=np.float32),
        [("q", "k", 1.0), ("k", "c", 0.8)],
    )

    first_gain = g.info_gain("q", "c", y=0.8)
    assert first_gain > 0.0, first_gain

    g.extend([], np.empty((0, EMB_DIM), dtype=np.float32), [("q", "c", 0.8)])
    assert g.info_gain("q", "c", y=0.8) == 0.0
    assert g.info_gain("c", "q", y=0.8) == 0.0
