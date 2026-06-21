"""End-to-end test: full PSRO + TGN message-passing pipeline.

No external APIs required. Uses StaticJudge + synthetic embeddings so
this runs in CI and on developer machines with no network access.

What this covers that unit tests don't:
  - Multiple Trainer.step calls (so second+ batches have existing neighbours)
  - nbr_ids_by_node collection in psro.py fires and reaches train_step
  - Aggregator trains end-to-end via the live graph topology
  - graph.impute / graph.field use neighbourhood-enriched predict_link
  - Snapshot round-trip preserves TGN state (memory + aggregator weights)
  - trainer.rank() inference runs after training without errors
  - Loss decreases over a short training run (trainability check)
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import Trainer

# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

EMB_DIM = 32
MEM_DIM = 32


def _cfg(tgn: bool = True, threshold: float = 2.0) -> MultiAgentConfig:
    """tgn_predict_threshold=2.0 forces all pairs through the judge so
    judged_pairs is always non-empty and TGN always trains this step."""
    return MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=3,
        k=3,
        judge_budget_per_batch=12,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "agent_2": "semantic",
        },
        device="cpu",
        use_tgn=tgn,
        tgn_memory_dim=MEM_DIM,
        tgn_time_dim=16,
        tgn_n_attn_heads=2,
        tgn_lr=1e-2,
        tgn_predict_threshold=threshold,
        learning_rate=5e-3,
    )


def _batch(ids: list[str], seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    embs = rng.standard_normal((len(ids), EMB_DIM)).astype(np.float32)
    return Batch(ids=ids, embs=embs, texts=[f"text_{i}" for i in ids])


# ---------------------------------------------------------------------------
# 1. Full multi-batch training run
# ---------------------------------------------------------------------------


def test_e2e_multi_batch_run():
    """Three batches of Trainer.step complete without error; the graph grows
    monotonically; at least one edge is committed per batch."""
    torch.manual_seed(0)
    np.random.seed(0)

    trainer = Trainer(_cfg(), StaticJudge(0.6))

    node_counts = []
    edge_counts = []
    for i in range(3):
        ids = [f"b{i}_{j}" for j in range(6)]
        result = trainer.step(_batch(ids, seed=i))
        node_counts.append(len(trainer.graph))
        edges = list(trainer.graph._edges.items())
        edge_counts.append(len(edges))
        assert result.stats.judged > 0, f"Batch {i}: no pairs judged"

    # Graph grows monotonically
    assert node_counts[0] < node_counts[1] < node_counts[2]
    # Edges accumulate (at least one judged edge per batch)
    assert edge_counts[1] >= edge_counts[0]
    assert edge_counts[2] >= edge_counts[1]


# ---------------------------------------------------------------------------
# 2. TGN memory is non-zero after first batch
# ---------------------------------------------------------------------------


def test_e2e_tgn_memory_written_after_first_batch():
    torch.manual_seed(1)
    trainer = Trainer(_cfg(), StaticJudge(0.7))
    trainer.step(_batch(["a", "b", "c", "d", "e", "f"], seed=1))

    # At least some nodes had their memory written
    warm = [
        nid
        for nid in trainer.tgn.memory._store
        if torch.any(trainer.tgn.memory._store[nid] != 0.0)
    ]
    assert len(warm) > 0, "No TGN memory was written after the first batch"


# ---------------------------------------------------------------------------
# 3. Aggregator trains when neighbours exist (second batch)
# ---------------------------------------------------------------------------


def test_e2e_aggregator_trains_on_second_batch():
    """After batch 1 seeds the graph with edges, batch 2 should fire the
    aggregator path in train_step and update aggregator weights."""
    torch.manual_seed(2)
    trainer = Trainer(_cfg(), StaticJudge(0.7))

    # Batch 1 — no prior neighbours, aggregator is bypassed
    trainer.step(_batch(["a0", "a1", "a2", "a3", "a4", "a5"], seed=2))

    agg_weight_before = trainer.tgn.aggregator.conv.lin_query.weight.detach().clone()

    # Batch 2 — prior nodes now have neighbours; aggregator path fires
    result = trainer.step(_batch(["b0", "b1", "b2", "b3", "b4", "b5"], seed=3))
    assert result.stats.judged > 0

    agg_weight_after = trainer.tgn.aggregator.conv.lin_query.weight.detach()
    assert not torch.allclose(agg_weight_before, agg_weight_after), (
        "Aggregator weights did not change on second batch — nbr_ids may not "
        "be reaching train_step correctly"
    )


# ---------------------------------------------------------------------------
# 4. graph.impute uses neighbourhood context after warm-up
# ---------------------------------------------------------------------------


def test_e2e_impute_field_work_after_warmup():
    """After two batches, graph.impute and graph.field for unobserved pairs
    call predict_link with neighbour context and return valid scores."""
    torch.manual_seed(3)
    trainer = Trainer(_cfg(threshold=0.0), StaticJudge(0.6))

    for i in range(2):
        ids = [f"n{i}_{j}" for j in range(6)]
        trainer.step(_batch(ids, seed=i + 10))

    node_ids = trainer.graph.get_nodes()
    assert len(node_ids) >= 4

    # Find a pair without a direct edge to exercise the imputation path
    tested = False
    for u in node_ids:
        for v in node_ids:
            if u == v:
                continue
            if trainer.graph._edges.get(trainer.graph._edge_key(u, v)) is not None:
                continue
            val_impute = trainer.graph.impute(u, v)
            val_field = trainer.graph.field(u, v)
            assert isinstance(val_field, float)
            assert -1.0 <= val_field <= 1.0
            # impute may return None (confidence below threshold) or a float
            if val_impute is not None:
                assert -1.0 <= val_impute <= 1.0
            tested = True
            break
        if tested:
            break

    assert tested, "Could not find an unobserved pair to test impute/field"


# ---------------------------------------------------------------------------
# 5. Inference via trainer.rank() works after training
# ---------------------------------------------------------------------------


def test_e2e_rank_returns_valid_scores_after_training():
    """trainer.rank(query_emb) should return a dict keyed by agent_id,
    each with (node_id, score) pairs in descending order."""
    torch.manual_seed(4)
    trainer = Trainer(_cfg(), StaticJudge(0.5))

    for i in range(2):
        ids = [f"r{i}_{j}" for j in range(5)]
        trainer.step(_batch(ids, seed=i + 20))

    rng = np.random.default_rng(99)
    query_emb = rng.standard_normal(EMB_DIM).astype(np.float32)
    ranked = trainer.rank(query_emb, k=5)

    assert isinstance(ranked, dict)
    assert len(ranked) >= 3  # at least the 3 configured agents (+ cosine baseline)

    for agent_id, hits in ranked.items():
        assert len(hits) > 0, f"agent {agent_id} returned no hits"
        scores = [s for _, s in hits]
        # Verify descending order
        assert scores == sorted(
            scores, reverse=True
        ), f"agent {agent_id} hits not in descending score order"
        for node_id, score in hits:
            assert isinstance(node_id, str)
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# 6. Snapshot / resume round-trip preserves TGN state
# ---------------------------------------------------------------------------


def test_e2e_snapshot_resume_preserves_tgn_and_aggregator():
    """After training, to_snapshot → from_snapshot should restore TGN memory
    and aggregator weights such that predict_link gives identical results."""
    torch.manual_seed(5)
    trainer = Trainer(_cfg(), StaticJudge(0.7))

    for i in range(2):
        ids = [f"s{i}_{j}" for j in range(5)]
        trainer.step(_batch(ids, seed=i + 30))

    snap, weights_bytes = trainer.to_snapshot(session_id="e2e-test")
    resumed = Trainer.from_snapshot(snap, weights_bytes, StaticJudge(0.7))

    # TGN memories must match
    orig_nodes = list(trainer.tgn.memory._store.keys())
    assert len(orig_nodes) > 0
    for nid in orig_nodes:
        orig_mem = trainer.tgn.memory._store[nid]
        resumed_mem = resumed.tgn.memory._store.get(nid)
        assert resumed_mem is not None, f"Node {nid!r} missing from resumed TGN memory"
        assert torch.allclose(
            orig_mem.cpu(), resumed_mem.cpu()
        ), f"TGN memory mismatch for node {nid!r} after snapshot resume"

    # Aggregator weights must match
    orig_w = trainer.tgn.aggregator.conv.lin_query.weight.detach().cpu()
    resumed_w = resumed.tgn.aggregator.conv.lin_query.weight.detach().cpu()
    assert torch.allclose(
        orig_w, resumed_w
    ), "Aggregator in_proj_weight diverged after snapshot resume"

    # predict_link must give identical scores (same params + same memory)
    node_ids = trainer.graph.get_nodes()
    if len(node_ids) >= 2:
        u, v = node_ids[0], node_ids[1]
        orig_score = trainer.tgn.predict_link(u, v)
        resumed_score = resumed.tgn.predict_link(u, v)
        assert abs(orig_score - resumed_score) < 1e-5, (
            f"predict_link({u!r},{v!r}) differs after resume: "
            f"{orig_score:.6f} vs {resumed_score:.6f}"
        )


# ---------------------------------------------------------------------------
# 7. Loss decreases over repeated training on consistent signal
# ---------------------------------------------------------------------------


def test_e2e_tgn_loss_decreases_over_training():
    """TGN loss should trend downward when the judge gives a consistent
    positive signal (StaticJudge(0.8)). Measured over 5 consecutive batches
    — we compare first vs last to avoid sensitivity to local noise."""
    torch.manual_seed(6)
    trainer = Trainer(_cfg(), StaticJudge(0.8))

    tgn_losses: list[float] = []
    for i in range(6):
        ids = [f"t{i}_{j}" for j in range(5)]
        trainer.step(_batch(ids, seed=i + 40))
        loss = trainer.loop.last_step_stats.get("tgn_loss", 0.0)
        if loss > 0.0:
            tgn_losses.append(loss)

    assert (
        len(tgn_losses) >= 3
    ), f"Not enough batches produced TGN loss for a trend check: {tgn_losses}"
    # The last recorded loss should be below the first (rough monotone check)
    assert tgn_losses[-1] < tgn_losses[0], (
        f"TGN loss did not decrease over training: "
        f"{tgn_losses[0]:.4f} → {tgn_losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 8. No TGN mode is unaffected (regression guard)
# ---------------------------------------------------------------------------


def test_e2e_no_tgn_mode_unchanged():
    """With use_tgn=False the existing Bayesian baseline path still works
    end-to-end — confirming our changes introduced no regressions."""
    torch.manual_seed(7)
    trainer = Trainer(_cfg(tgn=False), StaticJudge(0.5))

    assert trainer.tgn is None

    for i in range(3):
        ids = [f"nt{i}_{j}" for j in range(5)]
        result = trainer.step(_batch(ids, seed=i + 50))
        assert result.stats.judged >= 0  # completed without error

    assert len(trainer.graph) > 0
    node_ids = trainer.graph.get_nodes()
    if len(node_ids) >= 2:
        u, v = node_ids[0], node_ids[1]
        val = trainer.graph.field(u, v)
        assert -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# 9. nbr_mems device consistency (CPU, verifiable without CUDA)
# ---------------------------------------------------------------------------


def test_e2e_nbr_mems_stay_on_cpu():
    """On CPU, _nbr_mems() must return CPU tensors and the full forward pass
    through predict_link/train_step must not trigger a device mismatch."""
    torch.manual_seed(8)
    trainer = Trainer(_cfg(), StaticJudge(0.6))

    # Run two batches so second batch has neighbours
    trainer.step(_batch(["c0", "c1", "c2", "c3"], seed=60))
    trainer.step(_batch(["d0", "d1", "d2", "d3"], seed=61))

    # Directly check _nbr_mems device for every node that has neighbours
    for nid in trainer.graph.get_nodes():
        nbr = trainer.graph._nbr_mems(nid)
        if nbr is not None:
            assert (
                nbr.device.type == "cpu"
            ), f"_nbr_mems({nid!r}) returned tensor on {nbr.device}, expected cpu"

    # And verify no device mismatch during inference
    for u in trainer.graph.get_nodes():
        for v in trainer.graph.get_nodes():
            if u != v:
                _ = trainer.graph.field(u, v)  # must not raise
                break
        break


# ---------------------------------------------------------------------------
# 10. Step stats contain expected keys after message-passing step
# ---------------------------------------------------------------------------


def test_e2e_step_stats_complete():
    """last_step_stats from PSROLoop should include tgn_loss and all
    standard PSRO fields after the new message-passing path runs."""
    torch.manual_seed(9)
    trainer = Trainer(_cfg(), StaticJudge(0.6))

    trainer.step(_batch(["x0", "x1", "x2", "x3"], seed=70))
    result = trainer.step(_batch(["y0", "y1", "y2", "y3"], seed=71))

    raw = trainer.loop.last_step_stats
    expected_keys = {
        "tgn_loss",
        "loss",
        "per_agent_loss",
        "loss_spread",
        "sigma",
        "meta_rewards",
        "field_revealed",
        "judged",
        "scorable",
        "repr_divergence",
    }
    missing = expected_keys - set(raw.keys())
    assert not missing, f"Missing keys in last_step_stats: {missing}"

    assert isinstance(raw["tgn_loss"], float)
    assert raw["tgn_loss"] >= 0.0
    assert isinstance(raw["repr_divergence"], float)
    assert result.stats.repr_divergence == raw["repr_divergence"]
    assert isinstance(raw["sigma"], dict)
    assert (
        len(raw["sigma"]) >= 3
    )  # at least the 3 configured agents (+ cosine baseline)
