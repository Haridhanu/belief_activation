"""Integration tests for multi_agent.

Exercises end-to-end flows and behavioral properties, not class surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

import torch

from multi_agent.config import MultiAgentConfig
from multi_agent.graph import Graph
from multi_agent.utils.helpers import build_self_mask, score_and_sample_agent

EMB_DIM = 16


def _graph_with(embeddings: np.ndarray, node_ids: list[str]) -> Graph:
    g = Graph(emb_dim=EMB_DIM)
    g.extend(node_ids, embeddings, edges=[])
    return g


# ----------------------------------------------------------------------
# Online graph construction: runner.run grows an edge-wired graph
# ----------------------------------------------------------------------


def test_train_online_grows_graph_and_holds_out_tail():
    """``runner.run`` should build a graph from scratch: every trained batch
    adds its nodes, wires the winner's proposals as edges, and held-out
    batches (sliced by the caller) never enter the graph.
    """
    pytest.importorskip("torch")
    import torch

    from dataclasses import dataclass
    from typing import Iterator

    from multi_agent.benchmarks import Batch, Dataset, Query
    from multi_agent.runner import run
    from multi_agent.judge import StaticJudge

    torch_emb_dim = 64
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    cluster_centers = np.stack(
        [
            np.eye(torch_emb_dim, dtype=np.float32)[0],
            np.eye(torch_emb_dim, dtype=np.float32)[1],
        ]
    )
    cluster_size = 6
    n = 2 * cluster_size
    embs_rows: list[np.ndarray] = []
    texts: list[str] = []
    for c in range(2):
        for i in range(cluster_size):
            jitter = 0.05 * rng.standard_normal(torch_emb_dim).astype(np.float32)
            embs_rows.append(cluster_centers[c] + jitter)
            texts.append(f"cluster-{c} node-{i}")
    embs = np.stack(embs_rows)

    ds = Dataset(
        id="toy",
        label="toy 2-cluster",
        candidates=texts,
        queries=[Query(description="seed", correct_idx=0)],
        cand_embs=embs,
        query_embs=embs[:1],
    )

    @dataclass
    class ToyBenchmark:
        name: str = "toy"

        def get_batches(self, ds: Dataset, batch_size: int) -> Iterator[Batch]:
            for start in range(0, len(ds.candidates), batch_size):
                end = min(start + batch_size, len(ds.candidates))
                yield Batch(
                    ids=[str(i) for i in range(start, end)],
                    embs=ds.cand_embs[start:end],
                    texts=ds.candidates[start:end],
                )

    config = MultiAgentConfig(
        emb_dim=torch_emb_dim,
        num_agents=3,
        k=3,
        temperature=0.5,
        tournament_size=3,
        batch_size=3,
        learning_rate=0.05,
    )

    def cluster_judge(q: str, c: str) -> float:
        q_cluster = q.split()[0] if q else ""
        c_cluster = c.split()[0] if c else ""
        return 1.0 if q_cluster == c_cluster else 0.0

    all_beliefs = list(ToyBenchmark().get_batches(ds, batch_size=config.batch_size))
    train_beliefs = all_beliefs[:-1]
    held_out = all_beliefs[-1:]

    result = run(config, StaticJudge(cluster_judge), train_beliefs)

    # Held-out batch ids never enter the graph.
    held_ids = set(held_out[0].ids)
    graph = result["graph"]
    graph_ids = set(graph.get_nodes())
    assert held_ids.isdisjoint(graph_ids)
    assert len(graph_ids) == n - len(held_ids)

    # At least one winner wired at least one edge.
    total_edges = sum(len(graph.get_neighbors(nid)) for nid in graph.get_nodes())
    assert total_edges > 0

    # One winner recorded per training batch.
    assert len(result["winners"]) == len(train_beliefs)


# ----------------------------------------------------------------------
# PSROLoop no longer accepts a policy= kwarg
# ----------------------------------------------------------------------


def test_psro_loop_has_no_policy_knob():
    """PSROLoop no longer accepts a policy arg — amortization is internal."""
    from multi_agent.psro import PSROLoop
    from multi_agent.judge import StaticJudge

    cfg = MultiAgentConfig(num_agents=2, emb_dim=EMB_DIM, device="cpu")
    with pytest.raises(TypeError):
        PSROLoop(cfg, judge=StaticJudge(0.0), policy=None)


def test_psro_reward_is_zero_for_unrelated_observations():
    """When every judge score is 0 (unrelated), info-gain reward is 0."""
    pytest.importorskip("torch")
    import torch

    from multi_agent.agent import AgentPopulation
    from multi_agent.psro import PSROLoop
    from multi_agent.judge import StaticJudge

    torch_emb_dim = 32
    np.random.seed(0)
    torch.manual_seed(0)

    n_nodes = 8
    embs = np.random.randn(n_nodes, torch_emb_dim).astype(np.float32)
    node_ids = [f"n{i}" for i in range(n_nodes)]
    graph = Graph(emb_dim=torch_emb_dim)
    graph.extend(node_ids, embs, edges=[])

    config = MultiAgentConfig(
        emb_dim=torch_emb_dim,
        num_agents=2,
        k=2,
        temperature=0.5,
        tournament_size=2,
        batch_size=2,
        learning_rate=0.05,
    )
    population = AgentPopulation(config)
    cand_embs = population.embeddings_to_device(embs)

    loop = PSROLoop(config, judge=StaticJudge(0.0), graph=graph)

    q_ids = node_ids[:2]
    q_embs = cand_embs[:2]
    step = loop.step(population, q_embs, q_ids, cand_embs, node_ids)
    for r in step:
        assert max(r["rewards"]) < 1e-6, r["rewards"]


# ----------------------------------------------------------------------
# _backward gradient: empty row must not stall the batch
# ----------------------------------------------------------------------


def test_backward_does_not_stall_on_empty_row():
    """When one query in the batch has no valid proposals (empty row_indices),
    _backward must still update agent parameters for the rows that do have
    proposals.

    Regression test: the old indices = torch.stack([t[:k_eff] for t in row_indices])
    path produced k_eff = min(2, 0) = 0 → zero-tensor log-probs → zero loss →
    no gradient for any row, including row 0 which had valid proposals.
    """
    pytest.importorskip("torch")
    import torch

    from multi_agent.agent import AgentPopulation
    from multi_agent.psro import PSROLoop
    from multi_agent.utils.types import AgentProposal

    torch_emb_dim = 16
    torch.manual_seed(42)
    np.random.seed(42)

    cfg = MultiAgentConfig(emb_dim=torch_emb_dim, num_agents=2, k=3, learning_rate=0.05)
    loop = PSROLoop(cfg)
    population = AgentPopulation(cfg)

    B = 2
    N = 4
    pool_embs = population.embeddings_to_device(
        np.random.randn(N, torch_emb_dim).astype(np.float32)
    )
    query_embs = pool_embs[:B]

    # Build by_agent manually: agent 0 has row 0 with 2 valid proposals,
    # row 1 with 0 valid proposals (simulates singleton pool = query).
    # Scores must NOT be detached — they need to stay in the computation graph
    # so that gradients flow back to agent parameters via _backward.
    by_agent: dict = {}
    for agent in population.agents:
        scores = agent.score_candidates_batch(
            query_embs, pool_embs
        )  # live, not detached
        row0_idx = torch.tensor([0, 2], dtype=torch.long)
        row1_idx = torch.empty(0, dtype=torch.long)
        indices = torch.zeros((B, 0), dtype=torch.long)
        proposal = AgentProposal(
            scores=scores,
            indices=indices,
            proposals=[["a", "c"], []],
            row_indices=[row0_idx, row1_idx],
        )
        by_agent[agent.agent_id] = proposal

    # Snapshot parameters before backward.
    params_before = [p.detach().clone() for p in population.agents[0].parameters()]

    n_agents = len(population.agents)
    # Non-zero advantage for both rows — row 0 should produce gradient.
    rewards = np.full((B, n_agents), 0.5, dtype=np.float32)
    rewards[0, 0] = 0.9  # strong signal for agent 0, row 0

    loop._backward(rewards, population.agents, by_agent)

    params_after = [p.detach().clone() for p in population.agents[0].parameters()]
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(params_before, params_after)
    )
    assert changed, (
        "Agent 0 parameters did not change after _backward with non-zero advantage — "
        "gradient stall: empty row 1 zeroed out row 0's gradient"
    )


# ----------------------------------------------------------------------
# Self-pair masking: proposals must never contain the query itself
# ----------------------------------------------------------------------


def test_score_and_sample_agent_never_proposes_self():
    """score_and_sample_agent must not include (q, q) in proposals even when
    k >= pool_size - 1 (the edge case where torch.multinomial would otherwise
    fall back to sampling zero-probability entries).

    Regression test for the k-capping fix in helpers.py.
    """
    from multi_agent.agent import AgentPopulation

    torch.manual_seed(0)
    np.random.seed(0)

    pool_ids = ["a", "b", "c", "q"]
    query_ids = ["q"]
    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM, k=10
    )  # k >> pool size to trigger the edge case
    population = AgentPopulation(cfg)
    agent = population.agents[0]

    pool_embs = population.embeddings_to_device(
        np.random.randn(len(pool_ids), EMB_DIM).astype(np.float32)
    )
    query_embs = pool_embs[-1:].clone()  # q's embedding is last

    k = min(cfg.k, len(pool_ids))
    self_cols = build_self_mask(query_ids, pool_ids)
    proposal = score_and_sample_agent(
        agent, query_embs, pool_embs, pool_ids, self_cols, k
    )

    assert (
        "q" not in proposal.proposals[0]
    ), f"Self-pair proposed: got {proposal.proposals[0]}"


def test_score_and_sample_agent_singleton_pool_does_not_crash():
    """When pool == query (first Trainer.step with a single node), every
    candidate is the self-column.  The per-row sampler must return empty
    proposals rather than raising RuntimeError from multinomial(probs, 0).
    """
    from multi_agent.agent import AgentPopulation

    torch.manual_seed(1)
    np.random.seed(1)

    pool_ids = ["q"]
    query_ids = ["q"]
    cfg = MultiAgentConfig(emb_dim=EMB_DIM, k=8)
    population = AgentPopulation(cfg)
    agent = population.agents[0]

    embs = population.embeddings_to_device(
        np.random.randn(1, EMB_DIM).astype(np.float32)
    )
    k = min(cfg.k, len(pool_ids))
    self_cols = build_self_mask(query_ids, pool_ids)
    proposal = score_and_sample_agent(agent, embs, embs, pool_ids, self_cols, k)

    assert (
        proposal.proposals[0] == []
    ), f"Expected empty proposals for singleton pool, got {proposal.proposals[0]}"
    assert proposal.indices.shape == (1, 0)


def test_score_and_sample_agent_extreme_logits_no_self_pair():
    """When all valid candidates underflow to zero probability (e.g. one
    valid logit at 0, another at -1000 in float32), the uniform fallback must
    be used and the masked self-column must never appear in proposals.
    """
    from multi_agent.agent import AgentPopulation
    from unittest.mock import patch

    torch.manual_seed(2)
    np.random.seed(2)

    pool_ids = ["a", "b", "q"]
    query_ids = ["q"]
    cfg = MultiAgentConfig(emb_dim=EMB_DIM, k=2)
    population = AgentPopulation(cfg)
    agent = population.agents[0]

    embs = population.embeddings_to_device(
        np.random.randn(3, EMB_DIM).astype(np.float32)
    )
    query_embs = embs[-1:]

    # Force scores so that after masking q (col 2), the remaining valid
    # candidates a and b have logits [0, -1000] — b underflows to prob≈0.
    def fake_score(q, p):
        s = torch.zeros(1, 3)
        s[0, 0] = 0.0  # a: finite
        s[0, 1] = -1000.0  # b: will underflow to 0 after softmax
        s[0, 2] = 999.0  # q (self): will be masked to -inf
        return s

    self_cols = build_self_mask(query_ids, pool_ids)
    with patch.object(agent, "score_candidates_batch", side_effect=fake_score):
        proposal = score_and_sample_agent(
            agent, query_embs, embs, pool_ids, self_cols, k=2
        )

    assert (
        "q" not in proposal.proposals[0]
    ), f"Self-pair leaked through underflow path: {proposal.proposals[0]}"
    assert len(proposal.proposals[0]) == 2
    assert set(proposal.proposals[0]) == {"a", "b"}
