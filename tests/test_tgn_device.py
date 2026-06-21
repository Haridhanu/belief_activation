"""Device-placement and backward-failure tests for the TGN substrate."""

from __future__ import annotations

import numpy as np
import pytest

from multi_agent.benchmarks import Batch
from multi_agent.config import MultiAgentConfig
from multi_agent.judge import StaticJudge
from multi_agent.runner import Trainer


def _config(device: str = "cpu") -> MultiAgentConfig:
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


def test_psro_tgn_backward_failure_does_not_corrupt_memory():
    """If TGN backward raises mid-batch, subsequent batches must still
    succeed. detach_all_memory must run even on the failure path."""
    trainer = Trainer(_config(), StaticJudge(lambda q, c: 0.5))
    assert trainer.tgn is not None
    assert trainer.tgn_optimizer is not None

    rng = np.random.default_rng(0)
    b1 = Batch(
        ids=["a0", "a1", "a2", "a3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["t0", "t1", "t2", "t3"],
    )
    trainer.step(b1)

    original_step = trainer.tgn_optimizer.step

    class _BoomOnce:
        def __init__(self) -> None:
            self.fired = False

        def __call__(self, *a, **kw):
            if not self.fired:
                self.fired = True
                raise RuntimeError("simulated optimizer.step failure")
            return original_step(*a, **kw)

    trainer.tgn_optimizer.step = _BoomOnce()

    b2 = Batch(
        ids=["b0", "b1", "b2", "b3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["t4", "t5", "t6", "t7"],
    )
    with pytest.raises(RuntimeError, match="simulated optimizer.step failure"):
        trainer.step(b2)

    for nid, mem in trainer.tgn.memory._store.items():
        assert not mem.requires_grad, (
            f"Memory for {nid!r} still requires grad after backward failure. "
            "PSRO is not detaching in a finally block."
        )

    b3 = Batch(
        ids=["c0", "c1", "c2", "c3"],
        embs=rng.standard_normal((4, 16)).astype(np.float32),
        texts=["t8", "t9", "t10", "t11"],
    )
    trainer.step(b3)
