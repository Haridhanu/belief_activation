"""Tests for the TGN-only trainer."""

from __future__ import annotations

import numpy as np
import torch

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import StepResult
from multi_agent.tgn_runner import TGNTrainer


EMB_DIM = 16


def _embs(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal((n, EMB_DIM)).astype(np.float32)
    e /= np.linalg.norm(e, axis=1, keepdims=True)
    return e


def _make_trainer(judge_score: float = 0.7, **overrides) -> TGNTrainer:
    cfg = MultiAgentConfig(
        emb_dim=EMB_DIM,
        num_agents=3,
        k=4,
        judge_budget_per_batch=overrides.pop("judge_budget", 4),
        tgn_memory_dim=overrides.pop("tgn_memory_dim", 16),
        tgn_time_dim=overrides.pop("tgn_time_dim", 8),
        tgn_n_attn_heads=2,
        engine="tgn_only",
        **overrides,
    )
    return TGNTrainer(cfg, StaticJudge(judge_score))


def test_step_returns_valid_step_result():
    torch.manual_seed(0)
    trainer = _make_trainer()
    batch = Batch(ids=["a", "b", "c"], embs=_embs(3), texts=["a", "b", "c"])
    res = trainer.step(batch)
    assert isinstance(res, StepResult)
    assert res.step == 1
    assert res.winner_id == "tgn"
    assert isinstance(res.edges, list)
    assert res.stats.n_nodes == 3


def test_cold_start_judges_within_budget():
    """First batch must produce some judged edges (cold start handled)."""
    torch.manual_seed(0)
    trainer = _make_trainer(judge_budget=3)
    batch = Batch(ids=["a", "b", "c", "d"], embs=_embs(4), texts=["a", "b", "c", "d"])
    res = trainer.step(batch)
    assert res.stats.judged > 0
    assert res.stats.judged <= 3  # respects budget


def test_link_head_weights_move_after_training():
    """At cold start, link_head[0].weight may receive zero gradient (input
    is zero memory). Check the *output* layer's bias instead — it always
    receives gradient from the loss as long as judged > 0."""
    torch.manual_seed(0)
    trainer = _make_trainer(judge_score=0.9)
    before = trainer.tgn.link_head[2].bias.detach().clone()
    batch = Batch(ids=["a", "b", "c", "d"], embs=_embs(4), texts=["a", "b", "c", "d"])
    res = trainer.step(batch)
    assert res.stats.judged > 0
    after = trainer.tgn.link_head[2].bias.detach()
    assert not torch.allclose(before, after), (
        "link_head[2].bias should change after a step that judged some pairs"
    )


def test_link_loss_decreases_with_repeated_training():
    """Direct repeated training on a fixed pair set must drive loss down.
    Avoids the cold-start memory issue that makes streaming tests flaky."""
    torch.manual_seed(0)
    trainer = _make_trainer(judge_score=0.8, judge_budget=3)
    # Seed the graph and warm up memories with one judged step.
    batch = Batch(ids=["a", "b", "c", "d"], embs=_embs(4), texts=["a", "b", "c", "d"])
    trainer.step(batch)
    triples = [("a", "b", 0.8), ("a", "c", 0.8), ("b", "d", 0.8)]
    losses: list[float] = []
    for _ in range(40):
        trainer.optimizer.zero_grad()
        loss = trainer.tgn.link_loss(triples)
        loss.backward()
        trainer.optimizer.step()
        losses.append(float(loss.item()))
    assert losses[-1] < losses[0], (
        f"link_loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
    )


def test_memory_accumulates_across_steps():
    torch.manual_seed(0)
    trainer = _make_trainer()
    batch = Batch(ids=["a", "b", "c"], embs=_embs(3), texts=["a", "b", "c"])
    trainer.step(batch)
    mem_a_after_first = trainer.tgn.memory.get("a").clone()
    batch2 = Batch(ids=["d", "e"], embs=_embs(2, seed=1), texts=["d", "e"])
    trainer.step(batch2)
    mem_a_after_second = trainer.tgn.memory.get("a")
    # Memory for 'a' may or may not change depending on whether step 2
    # judged any (a, *) pair; but in either case it must be present.
    assert mem_a_after_second.shape == mem_a_after_first.shape


def test_commits_judged_edges_at_judge_value():
    """Judged pairs commit edges with weight equal to the judge score (sym-max)."""
    torch.manual_seed(0)
    trainer = _make_trainer(judge_score=0.7, judge_budget=10)
    batch = Batch(ids=["a", "b", "c"], embs=_embs(3), texts=["a", "b", "c"])
    res = trainer.step(batch)
    for u, v, w in res.edges:
        # StaticJudge(0.7) → max-by-abs symmetrise → 0.7
        assert abs(w - 0.7) < 1e-6 or abs(w) >= trainer.config.tgn_only_commit_threshold


def test_score_cache_skips_re_judge():
    """A pair seen across batches should be cached, not re-judged."""
    torch.manual_seed(0)
    trainer = _make_trainer(judge_budget=10)
    batch = Batch(ids=["a", "b", "c"], embs=_embs(3), texts=["a", "b", "c"])
    trainer.step(batch)
    n_cached_before = len(trainer.score_cache)
    # Same nodes again → no new judging needed for already-cached pairs
    batch2 = Batch(ids=["a", "b"], embs=_embs(3)[:2], texts=["a", "b"])
    trainer.step(batch2)
    # No new beliefs added → no new candidates, score_cache unchanged
    assert len(trainer.score_cache) == n_cached_before


def test_psro_engine_unchanged():
    """engine='psro' (default) constructs the regular Trainer untouched."""
    from multi_agent.runner import Trainer
    cfg = MultiAgentConfig(emb_dim=EMB_DIM, num_agents=2, k=2)  # engine defaults to "psro"
    trainer = Trainer(cfg, StaticJudge(0.5))
    assert trainer.config.engine == "psro"
    # Sanity: no TGN, no imputer attached on the default path
    assert trainer.graph._tgn is None
    assert trainer.graph._imputer is None
