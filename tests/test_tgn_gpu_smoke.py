"""End-to-end smoke: TGN training + snapshot resume with device hinting.

Runs with CUDA if available; falls back to ``device="meta"`` for
placement-only checks on CPU-only hosts (incl. local dev + CPU CI). The
point is to assert that the wiring from ``MultiAgentConfig.device`` →
``TGNModule.to(device)`` → ``Trainer.from_snapshot`` produces a Trainer
whose TGN parameters AND node memory live on the requested device.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import Trainer


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "meta"


def _cfg(device: str) -> MultiAgentConfig:
    return MultiAgentConfig(
        emb_dim=16,
        num_agents=2,
        k=2,
        judge_budget_per_batch=4,
        agent_roles={"agent_0": "coherence", "agent_1": "contradiction"},
        device=device,
        use_tgn=True,
        tgn_memory_dim=16,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
        tgn_lr=1e-3,
        tgn_predict_threshold=2.0,
    )


def test_tgn_end_to_end_placement_via_snapshot():
    device = _select_device()
    cfg = _cfg(device)

    trainer = Trainer(cfg, StaticJudge(lambda q, c: 0.5))
    assert trainer.tgn is not None
    for p in trainer.tgn.parameters():
        assert (
            p.device.type == device
        ), f"After Trainer init, TGN param on {p.device.type}, expected {device}"

    # Meta tensors have no storage so we can't run a forward; only the
    # placement assertions above apply. On cuda we run one real step plus
    # a snapshot round-trip to verify memory follows the device.
    if device != "cuda":
        return

    rng = np.random.default_rng(0)
    batch = Batch(
        ids=["a0", "a1", "a2", "a3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["b0", "b1", "b2", "b3"],
    )
    trainer.step(batch)
    for nid, mem in trainer.tgn.memory._store.items():
        assert (
            mem.device.type == device
        ), f"After step, memory for {nid!r} on {mem.device}, expected {device}"

    snap, weights = trainer.to_snapshot(session_id="tgn-gpu-smoke")
    snap.multi_agent_config = {**snap.multi_agent_config, "device": device}
    resumed = Trainer.from_snapshot(snap, weights, StaticJudge(lambda q, c: 0.5))
    assert resumed.tgn is not None
    for p in resumed.tgn.parameters():
        assert (
            p.device.type == device
        ), f"After resume, TGN param on {p.device.type}, expected {device}"
    for nid, mem in resumed.tgn.memory._store.items():
        assert (
            mem.device.type == device
        ), f"After resume, memory for {nid!r} on {mem.device}, expected {device}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_message_passing_aggregator_on_cuda():
    """With two batches on CUDA, the aggregator path executes:
    - second batch's nodes see the first batch's nodes as neighbours
    - nbr id lists are collected and resolved per-event to CUDA memory tensors
    - _enrich_with_neighbors runs on CUDA with no device mismatch
    - aggregator parameters stay on CUDA after the step
    """
    device = "cuda"
    cfg = _cfg(device)
    trainer = Trainer(cfg, StaticJudge(lambda q, c: 0.5))

    rng = np.random.default_rng(1)

    # First batch — seeds the graph; no neighbours yet, aggregator bypassed
    b1 = Batch(
        ids=["a0", "a1", "a2", "a3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["t0", "t1", "t2", "t3"],
    )
    trainer.step(b1)

    agg_weight_before = trainer.tgn.aggregator.conv.lin_query.weight.detach().clone()

    # Second batch — a0-a3 now have neighbours; aggregator should fire
    b2 = Batch(
        ids=["b0", "b1", "b2", "b3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["t4", "t5", "t6", "t7"],
    )
    res = trainer.step(b2)
    assert res.stats.judged > 0, "No pairs were judged — aggregator path not exercised"

    # All TGN parameters (including aggregator) must still live on CUDA
    for p in trainer.tgn.parameters():
        assert (
            p.device.type == device
        ), f"TGN parameter moved off {device} after message-passing step: {p.device}"

    # All node memories must live on CUDA
    for nid, mem in trainer.tgn.memory._store.items():
        assert (
            mem.device.type == device
        ), f"Memory for {nid!r} on {mem.device} after message-passing step"

    # Aggregator weights must have changed (it received gradient via nbr_mems)
    agg_weight_after = trainer.tgn.aggregator.conv.lin_query.weight.detach()
    assert not torch.allclose(agg_weight_before, agg_weight_after), (
        "Aggregator in_proj_weight unchanged on CUDA after second batch — "
        "nbr_ids may not be reaching train_step on the GPU path"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_nbr_mems_tensors_on_cuda():
    """_nbr_mems() must return a tensor on the TGN's device, not on CPU."""
    import torch

    from multi_agent.tgn import TGNModule
    from multi_agent.graph import Graph
    import numpy as np

    device = torch.device("cuda")
    tgn = TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2).to(device)
    g = Graph(emb_dim=16, _tgn=tgn)

    embs = np.eye(16, dtype=np.float32)[:3]
    g.extend(["a", "b", "c"], embs, [("a", "b", 0.8), ("a", "c", 0.6)])
    tgn.update("a", "b", sign=1.0, timestamp=1.0, edge_weight=0.8)
    tgn.update("a", "c", sign=1.0, timestamp=2.0, edge_weight=0.6)

    nbr = g._nbr_mems("a")
    assert nbr is not None
    assert (
        nbr.device.type == "cuda"
    ), f"_nbr_mems returned tensor on {nbr.device}, expected cuda"
    assert nbr.shape == (2, 16)
