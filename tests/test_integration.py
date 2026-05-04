"""Integration tests for multi_agent.

Exercises end-to-end flows and behavioral properties, not class surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from multi_agent.config import MultiAgentConfig
from multi_agent.graph import Graph


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
