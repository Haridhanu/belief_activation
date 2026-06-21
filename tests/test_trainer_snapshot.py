"""Trainer snapshot/restore round-trip and multi-step resume."""

from __future__ import annotations

import numpy as np
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import Trainer
from multi_agent.utils.notebook import make_synthetic_batches


def _config(emb_dim: int) -> MultiAgentConfig:
    return MultiAgentConfig(
        emb_dim=emb_dim,
        device="cpu",
        num_agents=2,
        k=3,
        temperature=0.3,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        n_epochs=1,
        log_every=0,
        meta_lr=0.5,
        meta_eps=0.05,
        judge_budget_per_batch=8,
    )


def _make_two_batches(seed: int = 0) -> list[Batch]:
    return make_synthetic_batches(
        n_nodes=20, n_batches=2, n_topic_pairs=2, emb_dim=16, noise=0.2, seed=seed
    )


def test_trainer_to_from_snapshot_preserves_population_state():
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.0))
    trainer.step(batches[0])

    snapshot, weights = trainer.to_snapshot(session_id="test-sess")
    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.0))

    # Population weights should match exactly.
    for a_orig, a_new in zip(trainer.population.agents, restored.population.agents):
        for (n1, p1), (n2, p2) in zip(
            a_orig.state_dict().items(), a_new.state_dict().items()
        ):
            assert n1 == n2
            assert torch.equal(p1, p2)
    assert restored._step == trainer._step
    assert restored.score_cache == trainer.score_cache
    assert restored.node_texts == trainer.node_texts


def test_trainer_resume_after_snapshot_continues_training():
    """Structural equivalence: post-resume step counter and node coverage
    match a continuous run. Note: random state is not captured in the
    snapshot, so individual proposals after resume diverge from a
    continuous run; only deterministic structural properties are asserted."""
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    # Continuous reference run.
    ref = Trainer(cfg, StaticJudge(0.0))
    ref.step(batches[0])
    ref.step(batches[1])

    # Snapshot-and-resume run.
    a = Trainer(cfg, StaticJudge(0.0))
    a.step(batches[0])
    snapshot, weights = a.to_snapshot(session_id="test-sess")
    b = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.0))
    b.step(batches[1])

    # Final step counter and node coverage should agree.
    assert b._step == ref._step
    assert set(b.graph.get_nodes()) == set(ref.graph.get_nodes())


def test_trainer_resume_carries_actor_weights_into_next_batch():
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    a = Trainer(cfg, StaticJudge(1.0))
    a.step(batches[0])
    snapshot, weights = a.to_snapshot(session_id="weights-sess")
    before = {
        agent.agent_id: {
            name: tensor.detach().clone()
            for name, tensor in agent.state_dict().items()
            if tensor.is_floating_point()
        }
        for agent in a.population.agents
        if agent.agent_id != "cosine"
    }

    b = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(1.0))
    b.step(batches[1])
    snapshot_2, weights_2 = b.to_snapshot(session_id="weights-sess")
    c = Trainer.from_snapshot(snapshot_2, weights_2, judge=StaticJudge(1.0))

    assert c._step == 2
    assert c.loop._meta_weights == b.loop._meta_weights
    changed = False
    for agent in c.population.agents:
        if agent.agent_id == "cosine":
            continue
        for name, tensor in agent.state_dict().items():
            if tensor.is_floating_point() and not torch.equal(
                tensor, before[agent.agent_id][name]
            ):
                changed = True
                break
    assert changed, "restored actor heads should keep training on the next batch"


def test_trainer_to_snapshot_rejects_pipe_in_bid():
    import pytest

    cfg = _config(emb_dim=4)
    trainer = Trainer(cfg, StaticJudge(0.0))
    bad_batch = Batch(
        ids=["bid|with|pipe", "ok-bid"],
        embs=np.zeros((2, 4), dtype=np.float32),
        texts=["a", "b"],
    )
    trainer.step(bad_batch)
    with pytest.raises(ValueError, match="reserved separator"):
        trainer.to_snapshot(session_id="test")


def test_trainer_history_preserved_across_snapshot():
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    a = Trainer(cfg, StaticJudge(0.0))
    a.step(batches[0])
    snapshot, weights = a.to_snapshot(session_id="hist-sess")
    b = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.0))

    assert len(b.history) == 1
    assert b.history[0]["step"] == a.history[0].step


def _tgn_config(emb_dim: int) -> MultiAgentConfig:
    # tgn_predict_threshold is set above the link_head's tanh range (>1.0)
    # so every pair fails the impute confidence check and goes to the
    # judge. This guarantees `tgn.train_step` fires on each batch, which
    # in turn populates NodeMemory for the touched nodes. Without this,
    # the default threshold (0.2) lets the freshly-initialised link_head
    # impute most pairs and TGN tests pass vacuously over empty memory.
    return MultiAgentConfig(
        emb_dim=emb_dim,
        device="cpu",
        num_agents=2,
        k=3,
        temperature=0.3,
        agent_roles={
            "agent_0": "coherence",
            "agent_1": "contradiction",
            "cosine": "semantic",
        },
        n_epochs=1,
        log_every=0,
        meta_lr=0.5,
        meta_eps=0.05,
        judge_budget_per_batch=8,
        use_tgn=True,
        tgn_memory_dim=emb_dim,
        tgn_time_dim=8,
        tgn_n_attn_heads=2,
        tgn_predict_threshold=2.0,
    )


def test_snapshot_round_trip_preserves_tgn_state_with_use_tgn_true():
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _tgn_config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.7))
    trainer.step(batches[0])
    trainer.step(batches[1])

    snapshot, weights = trainer.to_snapshot(session_id="tgn-sess")
    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.7))

    assert restored.tgn is not None
    assert restored.tgn_optimizer is not None

    # TGN parameters round-trip exactly. TGNModule.state_dict() is
    # mostly flat tensors with two special keys: ``_node_memory`` (a
    # nested dict of tensors from NodeMemory.state_dict) and
    # ``_ref_time`` (a 0-d tensor).
    orig_sd = trainer.tgn.state_dict()
    new_sd = restored.tgn.state_dict()
    assert orig_sd.keys() == new_sd.keys()
    for k, v_orig in orig_sd.items():
        v_new = new_sd[k]
        if isinstance(v_orig, torch.Tensor):
            assert torch.equal(v_orig, v_new), f"TGN param mismatch at {k}"
        elif isinstance(v_orig, dict):
            assert v_orig.keys() == v_new.keys(), f"TGN nested key mismatch at {k}"
            for sk, sv_orig in v_orig.items():
                assert torch.equal(
                    sv_orig, v_new[sk]
                ), f"TGN nested tensor mismatch at {k}/{sk}"
        else:
            assert v_orig == v_new, f"TGN value mismatch at {k}"

    # Optimizer momentum buffers round-trip too (otherwise resumed
    # training would diverge from a continuous run on the next step).
    orig_opt = trainer.tgn_optimizer.state_dict()
    new_opt = restored.tgn_optimizer.state_dict()
    assert orig_opt["param_groups"] == new_opt["param_groups"]
    assert set(orig_opt["state"].keys()) == set(new_opt["state"].keys())
    for pid, orig_state in orig_opt["state"].items():
        new_state = new_opt["state"][pid]
        for k, v_orig in orig_state.items():
            v_new = new_state[k]
            if isinstance(v_orig, torch.Tensor):
                assert torch.equal(
                    v_orig, v_new
                ), f"TGN optimizer state mismatch at param={pid} key={k}"
            else:
                assert v_orig == v_new


def test_snapshot_persists_node_memory_and_predict_link_unchanged():
    """NodeMemory is path-dependent state — the GRU consumed each event
    with whatever weights existed at that moment, so memory cannot be
    regenerated from edges alone. Dropping it on snapshot would silently
    break inference-only resumes, since ``predict_link`` reads memory
    through ``link_head``. This test locks in the persistence contract:
    touched-node memory round-trips byte-for-byte AND post-restore
    predictions match pre-snapshot predictions exactly.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _tgn_config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.7))
    trainer.step(batches[0])
    trainer.step(batches[1])

    touched = list(trainer.tgn.memory._store.keys())
    assert touched, "fixture failure: no nodes touched by TGN train_step"

    a, b = touched[0], touched[-1]
    mem_a_before = trainer.tgn.memory.get(a).clone()
    assert not torch.all(
        mem_a_before == 0.0
    ), "fixture failure: touched node memory is all-zero, persistence would be vacuous"
    pred_before = trainer.tgn.predict_link(a, b)
    ref_time_before = trainer.tgn._ref_time

    snapshot, weights = trainer.to_snapshot(session_id="memory-sess")
    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.7))

    assert set(restored.tgn.memory._store.keys()) == set(touched)
    torch.testing.assert_close(restored.tgn.memory.get(a), mem_a_before)
    assert restored.tgn._ref_time == ref_time_before
    assert restored.tgn.predict_link(a, b) == pred_before


def test_tgn_snapshot_predict_link_stable_under_normal_threshold():
    """Normal-threshold TGN inference should survive a snapshot round-trip.

    This exercises the production graph-aware path via Graph.field(), which
    supplies neighbour memories to TGNModule.predict_link.
    """
    torch.manual_seed(1)
    np.random.seed(1)
    batches = _make_two_batches(seed=1)
    cfg = _tgn_config(emb_dim=batches[0].embs.shape[1])
    cfg.tgn_predict_threshold = 0.2
    cfg.judge_budget_per_batch = 8

    trainer = Trainer(cfg, StaticJudge(0.7))
    trainer.step(batches[0])
    trainer.step(batches[1])

    node_ids = trainer.graph.get_nodes()
    assert len(node_ids) >= 4
    pairs = []
    for i, a in enumerate(node_ids):
        for b in node_ids[i + 1 :]:
            if trainer.graph._edge_key(a, b) in trainer.graph._edges:
                continue
            if (
                trainer.graph._nbr_mems(a) is None
                and trainer.graph._nbr_mems(b) is None
            ):
                continue
            pairs.append((a, b))
            if len(pairs) == 3:
                break
        if len(pairs) == 3:
            break
    assert len(pairs) == 3
    scores_before = [trainer.graph.field(a, b) for a, b in pairs]
    ref_time_before = trainer.tgn._ref_time

    snapshot, weights = trainer.to_snapshot(session_id="normal-threshold-tgn")
    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.7))

    assert restored.tgn._ref_time == ref_time_before
    scores_after = [restored.graph.field(a, b) for a, b in pairs]
    for before, after in zip(scores_before, scores_after):
        assert abs(before - after) < 1e-6


def test_snapshot_persists_graph_edge_clock_for_tgn_resume():
    """The graph edge clock is the TGN event-time base after resume.

    If it resets to zero while TGN _ref_time survives, the next resumed
    training step emits backwards timestamps.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _tgn_config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.7))
    trainer.step(batches[0])
    trainer.step(batches[1])

    assert trainer.graph._edge_count > 0
    edge_count_before = trainer.graph._edge_count
    edge_timestamps_before = dict(trainer.graph._edge_timestamps)

    snapshot, weights = trainer.to_snapshot(session_id="edge-clock-sess")
    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.7))

    assert restored.graph._edge_count == edge_count_before
    assert restored.graph._edge_timestamps == edge_timestamps_before


def test_tgn_snapshot_graph_z_uses_current_projection_not_stale_cache():
    """TGN snapshots feed Router via graph_z, so graph_z must reflect the
    current trainable mem_to_emb projection even if Graph._z is stale."""
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _tgn_config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.7))
    trainer.step(batches[0])

    node_id = trainer.graph.get_nodes()[0]
    trainer.graph._z[node_id] = np.zeros_like(trainer.graph._z[node_id])

    snapshot, _ = trainer.to_snapshot(session_id="fresh-z-sess")
    live = trainer.graph.get_representations_fast([node_id])[0]

    np.testing.assert_allclose(snapshot.graph_z[node_id], live)
    assert not np.allclose(snapshot.graph_z[node_id], np.zeros_like(live))


def test_snapshot_omits_tgn_keys_when_use_tgn_false():
    """Baseline trainer's snapshot has no tgn / tgn_optimizer keys, so
    existing snapshot consumers see exactly what they always did."""
    import io as _io

    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.0))
    trainer.step(batches[0])

    _, weights = trainer.to_snapshot(session_id="baseline-sess")
    blob = torch.load(_io.BytesIO(weights), weights_only=True)

    assert "tgn" not in blob
    assert "tgn_optimizer" not in blob
    assert "state_dict" in blob
    assert "optimizers" in blob


def test_old_snapshot_without_tgn_fields_loads_as_baseline():
    """A snapshot written before TGN fields existed (no use_tgn etc. in
    multi_agent_config, no tgn keys in weights_blob) restores as a
    use_tgn=False trainer without error."""
    torch.manual_seed(0)
    np.random.seed(0)
    batches = _make_two_batches()
    cfg = _config(emb_dim=batches[0].embs.shape[1])

    trainer = Trainer(cfg, StaticJudge(0.0))
    trainer.step(batches[0])
    snapshot, weights = trainer.to_snapshot(session_id="legacy-sess")

    # Strip the TGN fields to simulate an old-format snapshot.
    legacy_cfg = dict(snapshot.multi_agent_config)
    for tgn_key in (
        "use_tgn",
        "tgn_memory_dim",
        "tgn_time_dim",
        "tgn_n_attn_heads",
        "tgn_lr",
        "tgn_predict_threshold",
        "tgn_rep_align_weight",
        "tgn_cold_start",
    ):
        legacy_cfg.pop(tgn_key, None)
    snapshot.multi_agent_config = legacy_cfg

    restored = Trainer.from_snapshot(snapshot, weights, judge=StaticJudge(0.0))
    assert restored.config.use_tgn is False
    assert restored.tgn is None
    assert restored.tgn_optimizer is None
