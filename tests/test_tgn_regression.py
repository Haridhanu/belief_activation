"""Regression tests: with TGN disabled, behaviour must match the pre-TGN baseline.

These tests pin the contract that ``use_tgn=False`` (and an unattached
``Graph._tgn``) reproduce the original outputs exactly. Any deviation here
is a bug, not a feature.
"""

from __future__ import annotations

import numpy as np
import torch

from multi_agent.graph import Graph

EMB_DIM = 32
SEED = 42


def _ids_embs(n: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(SEED)
    embs = rng.standard_normal((n, EMB_DIM)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return [f"n{i}" for i in range(n)], embs


def test_impute_identical_without_tgn():
    ids, embs = _ids_embs(6)
    edges = [(ids[0], ids[1], 0.9), (ids[1], ids[2], -0.5), (ids[2], ids[3], 0.7)]

    g_ref = Graph(emb_dim=EMB_DIM)
    g_ref.extend(ids, embs.copy(), edges)

    g_new = Graph(emb_dim=EMB_DIM, _tgn=None)
    g_new.extend(ids, embs.copy(), edges)

    for q in ids:
        for c in ids:
            if q == c:
                continue
            ref = g_ref.impute(q, c)
            new = g_new.impute(q, c)
            assert ref == new, f"impute({q},{c}): ref={ref} new={new}"


def test_z_representations_identical_without_tgn():
    ids, embs = _ids_embs(4)
    edges = [(ids[0], ids[1], 0.9), (ids[1], ids[2], -0.5)]

    g_ref = Graph(emb_dim=EMB_DIM)
    g_ref.extend(ids, embs.copy(), edges)

    g_new = Graph(emb_dim=EMB_DIM, _tgn=None)
    g_new.extend(ids, embs.copy(), edges)

    for nid in ids:
        np.testing.assert_array_equal(
            g_ref._z[nid], g_new._z[nid], err_msg=f"_z[{nid}] differs without TGN"
        )


def test_field_identical_without_tgn():
    ids, embs = _ids_embs(5)
    edges = [(ids[i], ids[i + 1], 0.8 if i % 2 == 0 else -0.6) for i in range(4)]

    g_ref = Graph(emb_dim=EMB_DIM)
    g_ref.extend(ids, embs.copy(), edges)

    g_new = Graph(emb_dim=EMB_DIM, _tgn=None)
    g_new.extend(ids, embs.copy(), edges)

    for q in ids:
        for c in ids:
            if q == c:
                continue
            assert g_ref.field(q, c) == g_new.field(
                q, c
            ), f"field({q},{c}) differs without TGN"


def test_info_gain_identical_without_tgn():
    ids, embs = _ids_embs(4)
    g_ref = Graph(emb_dim=EMB_DIM)
    g_ref.extend(ids, embs.copy(), [(ids[0], ids[1], 0.9)])

    g_new = Graph(emb_dim=EMB_DIM, _tgn=None)
    g_new.extend(ids, embs.copy(), [(ids[0], ids[1], 0.9)])

    for q in ids:
        for c in ids:
            if q == c:
                continue
            ref = g_ref.info_gain(q, c, y=0.8)
            new = g_new.info_gain(q, c, y=0.8)
            assert ref == new, f"info_gain({q},{c}): ref={ref} new={new}"


def test_prior_unchanged_when_edge_count_inflated_without_tgn():
    """_prior must ignore _edge_count/_edge_timestamps when TGN is absent."""
    ids, embs = _ids_embs(3)
    g = Graph(emb_dim=EMB_DIM)
    g.extend(ids, embs, [(ids[0], ids[1], 0.9), (ids[1], ids[2], 0.8)])

    fresh = g._prior(ids[0], ids[2])
    g._edge_count = 10_000  # would heavily decay any TGN-aware prior
    aged = g._prior(ids[0], ids[2])

    assert fresh == aged, "Without TGN, _prior must not depend on edge_count"


def test_full_trainer_step_identical_without_tgn():
    """End-to-end: graph nodes & edges identical for use_tgn=False, two configs."""
    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    ids, embs = _ids_embs(8)
    batch = Batch(ids=ids, embs=embs, texts=ids)

    cfg_ref = MultiAgentConfig(emb_dim=EMB_DIM, num_agents=2, k=3, seed=SEED)
    cfg_new = MultiAgentConfig(
        emb_dim=EMB_DIM, num_agents=2, k=3, seed=SEED, use_tgn=False
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_ref = Trainer(cfg_ref, StaticJudge(0.5))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_new = Trainer(cfg_new, StaticJudge(0.5))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_ref.step(batch)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_new.step(batch)

    assert set(t_ref.graph.get_nodes()) == set(t_new.graph.get_nodes())
    assert set(t_ref.graph._edges.keys()) == set(t_new.graph._edges.keys())
    for key, w_ref in t_ref.graph._edges.items():
        assert w_ref == t_new.graph._edges[key], f"edge weight differs at {key}"
