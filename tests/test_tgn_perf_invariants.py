"""Perf-invariant tests: refactors that touch TGN compute must not change
its outputs on a fixed seed.

Pin values were captured against the pre-refactor (per-event GRUCell)
train_step at torch seed 0. Post-refactor (batched GRUCell), they must
still match within 1e-6 — GRUCell processes rows independently, so
stacking two rows into one call is numerically equivalent.
"""

from __future__ import annotations

import torch

from multi_agent.tgn import TGNModule

_SEED_EVENTS = [
    ("a", "b", +1.0, 1.0, 0.9, +0.8),
    ("b", "c", -1.0, 2.0, 0.7, -0.5),
    ("a", "c", +1.0, 3.0, 0.6, +0.4),
    ("c", "d", +1.0, 4.0, 0.8, +0.2),
]

# Pinned at torch seed 0 against the PyG-backed train_step. Bit-for-bit
# reproducible on the current build. (Re-pinned from the pre-PyG value
# 0.25965067744255066 — the TransformerConv aggregator draws different init
# RNG than the old MultiheadAttention, shifting link_head init and thus the
# loss. The pairwise memory update is unaffected, so the memory-coord pins
# below are unchanged.)
_PINNED_LOSS = 0.45305827260017395
# First memory_dim coordinate of each node after train_step — small,
# stable subset of state for cross-impl parity.
_PINNED_FIRST_COORDS = {
    "a": -0.07386770099401474,
    "b": -0.024530915543437004,
    "c": -0.2252953052520752,
    "d": -0.16158141195774078,
}


def _build_tgn() -> TGNModule:
    torch.manual_seed(0)
    return TGNModule(emb_dim=16, memory_dim=16, time_dim=8, n_heads=2)


def test_train_step_loss_value_pinned():
    tgn = _build_tgn()
    loss = tgn.train_step(_SEED_EVENTS)
    assert (
        abs(float(loss.item()) - _PINNED_LOSS) < 1e-6
    ), f"train_step loss drifted: {float(loss.item())} vs pinned {_PINNED_LOSS}"


def test_train_step_memory_state_pinned():
    tgn = _build_tgn()
    tgn.train_step(_SEED_EVENTS)
    for nid, expected in _PINNED_FIRST_COORDS.items():
        actual = float(tgn.memory._store[nid][0].item())
        assert (
            abs(actual - expected) < 1e-6
        ), f"Memory[{nid!r}][0] drifted: {actual} vs pinned {expected}"


def test_train_step_bptt_truncation_preserves_forward_values():
    full = _build_tgn()
    truncated = _build_tgn()

    full_loss = full.train_step(_SEED_EVENTS, max_bptt_events=0)
    trunc_loss = truncated.train_step(_SEED_EVENTS, max_bptt_events=1)

    torch.testing.assert_close(trunc_loss, full_loss)
    assert truncated.memory._store.keys() == full.memory._store.keys()
    for nid in full.memory._store:
        torch.testing.assert_close(
            truncated.memory._store[nid], full.memory._store[nid]
        )


def test_train_step_bptt_truncation_actually_truncates_gradient():
    """Forward values match between full and truncated BPTT (pinned by
    ``test_train_step_bptt_truncation_preserves_forward_values``), but the
    *gradient* on parameters that participate **only** through the
    cross-event memory chain (msg_encoder, updater.gru) MUST be cut by
    truncation.

    Mechanism: a single event's loss depends on pre-event memory (read at
    the top of the loop) but NOT on that same event's msg_encoder / GRU
    forward pass — those only feed the *stored* post-event memory. So
    msg_encoder and updater.gru receive gradient solely via the next
    event's loss reading the stored memory. With ``max_bptt_events=1`` the
    storage is detached before the next read, severing this path; with
    ``max_bptt_events=0`` it stays connected.

    Concrete expectation:
      * full BPTT  → ``msg_encoder.proj.weight.grad`` is non-None and non-zero.
      * truncated → ``msg_encoder.proj.weight.grad`` is None (path absent).

    If ``detach_all_memory()`` at the chunk boundary were removed, the
    truncated grad would be non-None and this test would fail.
    """
    full = _build_tgn()
    truncated = _build_tgn()

    full.train_step(_SEED_EVENTS, max_bptt_events=0).backward()
    truncated.train_step(_SEED_EVENTS, max_bptt_events=1).backward()

    g_full_msg = full.msg_encoder.proj.weight.grad
    g_trunc_msg = truncated.msg_encoder.proj.weight.grad
    assert g_full_msg is not None and g_full_msg.abs().sum() > 0, (
        "Full BPTT must accumulate gradient on msg_encoder via the "
        "cross-event memory chain."
    )
    assert g_trunc_msg is None or g_trunc_msg.abs().sum() == 0, (
        "max_bptt_events=1 should sever the only path to msg_encoder — "
        "got nonzero gradient, suggesting detach_all_memory() is not "
        "running at chunk boundaries."
    )

    g_full_gru = full.updater.gru.weight_ih.grad
    g_trunc_gru = truncated.updater.gru.weight_ih.grad
    assert g_full_gru is not None and g_full_gru.abs().sum() > 0
    assert (
        g_trunc_gru is None or g_trunc_gru.abs().sum() == 0
    ), "Truncation must also sever the path to updater.gru."

    # And link_head, which IS reached intra-event, must train under both.
    g_full_head = full.link_head[0].weight.grad
    g_trunc_head = truncated.link_head[0].weight.grad
    assert g_full_head is not None and g_full_head.abs().sum() > 0
    assert g_trunc_head is not None and g_trunc_head.abs().sum() > 0


def test_train_step_gru_update_ignores_neighbourhood():
    """Design invariant (T-13): the GRU memory update uses the **plain**
    pairwise message, NOT the neighbourhood-enriched representation. So
    stored memory after ``train_step`` must be identical whether
    ``nbr_ids_by_node`` is provided or not — neighbourhood context only
    affects the link-head prediction, never the stored state.

    If a future refactor accidentally feeds ``m_src_ctx`` / ``m_dst_ctx``
    (the enriched copies) into ``msg_encoder`` instead of ``m_src`` /
    ``m_dst``, neighbours would leak into stored memory and this test
    would fail. The bug class is silent — model trains fine, predictions
    drift forever — so this trip-wire is load-bearing.
    """
    plain = _build_tgn()
    with_nbrs = _build_tgn()

    # Seed both with identical non-trivial memories so the aggregator has
    # something to attend over on the with_nbrs run.
    for tgn in (plain, with_nbrs):
        tgn.update("nbr_a", "nbr_b", sign=1.0, timestamp=0.5, edge_weight=0.7)
        tgn.update("nbr_c", "nbr_d", sign=-1.0, timestamp=0.6, edge_weight=0.5)
        tgn.detach_all_memory()

    nbr_ids = {
        "a": ["nbr_a", "nbr_b"],
        "b": ["nbr_c"],
        "c": ["nbr_a", "nbr_d"],
        "d": ["nbr_b"],
    }

    plain.train_step(_SEED_EVENTS, nbr_ids_by_node=None)
    with_nbrs.train_step(_SEED_EVENTS, nbr_ids_by_node=nbr_ids)

    # The seeded neighbour nodes must be unchanged across both runs
    # (the batch doesn't touch them). The judged-event endpoints must
    # have identical post-batch memory because their GRU updates only
    # ever saw plain pairwise messages — the nbr_ids path influences
    # only the link-head prediction.
    assert (
        plain.memory._store.keys() == with_nbrs.memory._store.keys()
    ), "Stored memory key set diverged between runs"
    for nid in plain.memory._store:
        torch.testing.assert_close(
            with_nbrs.memory._store[nid],
            plain.memory._store[nid],
            msg=(
                f"Stored memory for {nid!r} differs between train_step "
                f"with vs without nbr_ids_by_node. The GRU update path "
                f"is being contaminated by the aggregator output — "
                f"check that msg_encoder receives m_src/m_dst (plain), "
                f"not m_src_ctx/m_dst_ctx (enriched)."
            ),
        )


def test_psro_drives_monotonic_tgn_timestamps():
    """Across Trainer.step calls, TGN's _ref_time must be strictly increasing.
    Without a monotonic edge clock, batch N+1's timestamps may be smaller
    than batch N's terminal _ref_time, producing negative delta_t at the
    time encoder."""
    import numpy as np

    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    cfg = MultiAgentConfig(
        emb_dim=16,
        num_agents=2,
        k=2,
        judge_budget_per_batch=4,
        agent_roles={"agent_0": "coherence", "agent_1": "contradiction"},
        device="cpu",
        use_tgn=True,
        tgn_memory_dim=16,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
        tgn_lr=1e-3,
        tgn_predict_threshold=2.0,
    )
    trainer = Trainer(cfg, StaticJudge(lambda q, c: 0.5))
    rng = np.random.default_rng(0)

    ref_times: list[float] = []
    for prefix in ("a", "b", "c"):
        batch = Batch(
            ids=[f"{prefix}{i}" for i in range(4)],
            embs=rng.standard_normal((4, 16)).astype(np.float32),
            texts=[f"{prefix}t{i}" for i in range(4)],
        )
        trainer.step(batch)
        ref_times.append(trainer.tgn._ref_time)

    assert ref_times == sorted(
        ref_times
    ), f"_ref_time regressed across batches: {ref_times}"
    assert (
        ref_times[-1] > ref_times[0]
    ), f"_ref_time did not advance across batches: {ref_times}"


def test_edge_clock_persists_across_snapshot():
    """_edge_clock must round-trip via to_snapshot/from_snapshot so resumed
    sessions keep increasing TGN timestamps instead of restarting near zero."""
    import numpy as np

    from multi_agent.benchmarks import Batch
    from multi_agent.config import MultiAgentConfig
    from multi_agent.judge import StaticJudge
    from multi_agent.runner import Trainer

    cfg = MultiAgentConfig(
        emb_dim=16,
        num_agents=2,
        k=2,
        judge_budget_per_batch=4,
        agent_roles={"agent_0": "coherence", "agent_1": "contradiction"},
        device="cpu",
        use_tgn=True,
        tgn_memory_dim=16,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
        tgn_lr=1e-3,
        tgn_predict_threshold=2.0,
    )
    trainer = Trainer(cfg, StaticJudge(lambda q, c: 0.5))
    rng = np.random.default_rng(1)
    for prefix in ("a", "b"):
        batch = Batch(
            ids=[f"{prefix}{i}" for i in range(4)],
            embs=rng.standard_normal((4, 16)).astype(np.float32),
            texts=[f"{prefix}t{i}" for i in range(4)],
        )
        trainer.step(batch)

    clock_before = trainer.loop._edge_clock
    assert clock_before > 0.0

    snap, weights = trainer.to_snapshot(session_id="edge-clock-rt")
    resumed = Trainer.from_snapshot(snap, weights, StaticJudge(lambda q, c: 0.5))
    assert (
        resumed.loop._edge_clock == clock_before
    ), f"_edge_clock did not round-trip: {resumed.loop._edge_clock} vs {clock_before}"
