"""Tests for the Router class — fusion modes and degenerate-pool behaviour."""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from multi_agent.agent import AgentPopulation
from multi_agent.config import MultiAgentConfig
from multi_agent.router import (
    Router,
    RouterMissing,
    RouterSchemaMismatch,
    RouterVerifyFailed,
)
from multi_agent.snapshot import RouterSnapshot, SCHEMA_VERSION

EMB_DIM = 8


def _config() -> MultiAgentConfig:
    return MultiAgentConfig(
        emb_dim=EMB_DIM,
        device="cpu",
        num_agents=2,
        k=3,
        temperature=0.3,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
    )


def _population_and_weights() -> tuple[AgentPopulation, bytes]:
    pop = AgentPopulation(_config())
    buf = io.BytesIO()
    torch.save({"state_dict": pop.state_dict(), "optimizers": {}}, buf)
    return pop, buf.getvalue()


def _bid_z(n: int) -> dict[str, np.ndarray]:
    return {
        f"b{i}": np.random.RandomState(i).randn(EMB_DIM).astype(np.float32)
        for i in range(n)
    }


def test_router_rank_returns_topk_in_score_order():
    pop, _ = _population_and_weights()
    bids = _bid_z(10)
    router = Router(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: f"text-{bid}" for bid in bids},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = np.random.RandomState(99).randn(EMB_DIM).astype(np.float32)
    ranked = router.rank(q, k=3)
    assert len(ranked) == 3
    bids_out = [bid for bid, _ in ranked]
    scores_out = [s for _, s in ranked]
    assert len(set(bids_out)) == 3  # distinct
    assert scores_out == sorted(scores_out, reverse=True)


def test_router_rank_empty_pool_returns_empty():
    pop, _ = _population_and_weights()
    router = Router(
        population=pop,
        bid_to_z={},
        bid_to_text={},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = np.zeros(EMB_DIM, dtype=np.float32)
    assert router.rank(q, k=5) == []


def test_router_rank_singleton_returns_single_entry_with_zero_score():
    pop, _ = _population_and_weights()
    bids = _bid_z(1)
    router = Router(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: bid for bid in bids},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = np.zeros(EMB_DIM, dtype=np.float32)
    ranked = router.rank(q, k=3)
    assert ranked == [("b0", 0.0)]


def test_router_rank_all_degenerate_returns_empty():
    """If all agents produce zero-variance logits, rank returns [] so the
    consumer falls through to its fallback rather than getting a tied list."""
    pop, _ = _population_and_weights()
    # 3 candidates, but query is identical to all candidates so cosine is
    # constant; we additionally force AttentionAgent logits to be uniform by
    # zeroing query proj weights.
    for agent in pop.agents:
        if hasattr(agent, "attn_query_proj"):
            with torch.no_grad():
                agent.attn_query_proj.weight.zero_()
                agent.attn_key_proj.weight.zero_()
                agent.residual_mlp_hidden.weight.zero_()
                agent.residual_mlp_out.weight.zero_()
    z = np.ones(EMB_DIM, dtype=np.float32)
    bids = {"b0": z, "b1": z, "b2": z}
    router = Router(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: bid for bid in bids},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = z.copy()
    assert router.rank(q, k=3) == []


def test_router_rank_no_nan_when_one_agent_degenerate():
    pop, _ = _population_and_weights()
    bids = _bid_z(5)
    router = Router(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: bid for bid in bids},
        sigma={"agent_0": 1.0, "agent_1": 0.0, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = np.random.RandomState(1).randn(EMB_DIM).astype(np.float32)
    for _, s in router.rank(q, k=5):
        assert np.isfinite(s)


def test_router_score_returns_finite_array_for_caller_supplied_pool():
    pop, _ = _population_and_weights()
    router = Router(
        population=pop,
        bid_to_z={},
        bid_to_text={},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    q = np.random.RandomState(2).randn(EMB_DIM).astype(np.float32)
    cands = np.random.RandomState(3).randn(7, EMB_DIM).astype(np.float32)
    out = router.score(q, cands)
    assert out.shape == (7,)
    assert np.all(np.isfinite(out))


def test_router_from_snapshot_rebuilds_population_and_emb_dim_check():
    pop, weights = _population_and_weights()
    snap = RouterSnapshot(
        schema_version=SCHEMA_VERSION,
        session_id="sess-1",
        step=1,
        emb_dim=EMB_DIM,
        multi_agent_config={
            "emb_dim": EMB_DIM,
            "num_agents": 2,
            "k": 3,
            "temperature": 0.3,
            "agent_roles": {
                "agent_0": "coherence",
                "agent_1": "contradiction",
                "cosine": "semantic",
            },
            "device": "cpu",
        },
        graph_hyperparams={
            "attention_step": 0.2,
            "prior_variance": 1.0,
            "obs_variance": 0.05,
            "confidence_floor": 0.25,
        },
        bid_to_text={"b0": "x"},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        history=[],
        meta_weights={"agent_0": 1.0, "agent_1": 1.0, "cosine": 1.0},
        score_cache={},
        graph_z={"b0": np.zeros(EMB_DIM, dtype=np.float32)},
        graph_raw={"b0": np.zeros(EMB_DIM, dtype=np.float32)},
        graph_adj={"b0": []},
        graph_edges={},
        agent_pop_stats={
            "agent_0": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
            "agent_1": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
            "cosine": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
        },
    )
    router = Router.from_snapshot(snap, weights, fusion="sigma")
    q = np.zeros(EMB_DIM, dtype=np.float32)
    assert router.rank(q, k=1) == [("b0", 0.0)]


def test_router_from_snapshot_rejects_emb_dim_mismatch():
    pop, weights = _population_and_weights()
    snap_bad = _sample_snap_for_dim_check(emb_dim=EMB_DIM + 1, weights=weights)
    with pytest.raises(RouterVerifyFailed, match="emb_dim"):
        Router.from_snapshot(snap_bad, weights)


def test_router_rank_single_agent_mode_selects_only_that_agent():
    pop, _ = _population_and_weights()
    bids = _bid_z(6)
    common_kwargs = dict(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: bid for bid in bids},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
    )
    r0 = Router(fusion="single:agent_0", **common_kwargs)
    r1 = Router(fusion="single:agent_1", **common_kwargs)
    q = np.random.RandomState(7).randn(EMB_DIM).astype(np.float32)
    rank0 = [bid for bid, _ in r0.rank(q, k=6)]
    rank1 = [bid for bid, _ in r1.rank(q, k=6)]
    # Two trained AttentionAgents should produce *different* orderings.
    assert rank0 != rank1


def test_router_single_mode_unknown_agent_raises():
    pop, _ = _population_and_weights()
    bids = _bid_z(3)
    router = Router(
        population=pop,
        bid_to_z=bids,
        bid_to_text={bid: bid for bid in bids},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        emb_dim=EMB_DIM,
        fusion="single:does_not_exist",
    )
    q = np.random.RandomState(8).randn(EMB_DIM).astype(np.float32)
    with pytest.raises(RouterVerifyFailed, match="does_not_exist"):
        router.rank(q, k=3)


def test_router_init_rejects_bid_to_z_shape_mismatch():
    pop, _ = _population_and_weights()
    bids = {
        "good": np.zeros(EMB_DIM, dtype=np.float32),
        "bad": np.zeros(EMB_DIM + 1, dtype=np.float32),
    }
    with pytest.raises(RouterVerifyFailed, match="shape mismatch"):
        Router(
            population=pop,
            bid_to_z=bids,
            bid_to_text={bid: bid for bid in bids},
            sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
            roles={
                "agent_0": "coherence",
                "agent_1": "contradiction",
                "cosine": "semantic",
            },
            emb_dim=EMB_DIM,
        )


def _sample_snap_for_dim_check(emb_dim: int, weights: bytes) -> RouterSnapshot:
    return RouterSnapshot(
        schema_version=SCHEMA_VERSION,
        session_id="s",
        step=1,
        emb_dim=emb_dim,
        multi_agent_config={
            "emb_dim": emb_dim,
            "num_agents": 2,
            "k": 3,
            "temperature": 0.3,
            "agent_roles": {
                "agent_0": "coherence",
                "agent_1": "contradiction",
                "cosine": "semantic",
            },
            "device": "cpu",
        },
        graph_hyperparams={
            "attention_step": 0.2,
            "prior_variance": 1.0,
            "obs_variance": 0.05,
            "confidence_floor": 0.25,
        },
        bid_to_text={},
        sigma={"agent_0": 0.5, "agent_1": 0.5, "cosine": 0.0},
        roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        history=[],
        meta_weights={"agent_0": 1.0, "agent_1": 1.0, "cosine": 1.0},
        score_cache={},
        graph_z={},
        graph_raw={},
        graph_adj={},
        graph_edges={},
        agent_pop_stats={
            "agent_0": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
            "agent_1": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
            "cosine": {"wins": 0.0, "rounds": 0.0, "cum_reward": 0.0},
        },
    )
