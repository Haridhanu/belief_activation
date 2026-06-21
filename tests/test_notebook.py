from __future__ import annotations

import pytest

from multi_agent.config import MultiAgentConfig
from multi_agent.utils.notebook import make_synthetic_batches
from notebooks.utils import run_belief_activation


def test_run_belief_activation_offline_notebook_helper():
    pytest.importorskip("torch")

    batches = make_synthetic_batches(
        n_nodes=12,
        n_batches=3,
        n_topic_pairs=2,
        emb_dim=16,
        seed=7,
    )

    result = run_belief_activation(
        batches,
        num_agents=2,
        k=2,
        learning_rate=0.01,
        log_every=None,
    )

    assert result["config"].emb_dim == 16
    assert len(result["step_history"]) == len(batches)
    assert len(result["graph"].get_nodes()) == 12
    assert result["final_sigma"]
    assert result["judge_calls"]["judged"] > 0


def test_run_belief_activation_rejects_config_emb_dim_mismatch():
    batches = make_synthetic_batches(
        n_nodes=4,
        n_batches=2,
        n_topic_pairs=1,
        emb_dim=8,
        seed=3,
    )

    with pytest.raises(ValueError, match="config.emb_dim"):
        run_belief_activation(
            batches,
            config=MultiAgentConfig(emb_dim=16),
            log_every=None,
        )
